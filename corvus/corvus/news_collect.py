"""新闻采集器（1.1 底座版，注册表驱动）。

核心改动：
- 每条原文完整落盘 data/raw/news/{source_id}/{YYYYMMDD}/{raw_id}.json（只追加，含 raw_id，不覆盖）；
  SQLite 只存元数据 + raw_text_hash，正文不进库。
- 每次运行记 collection_run 账本。
- 时间一律经 timeutil.parse_source_time() 解析，带 tz_inference / time_parse_status。
- 官方源详情抓取失败：标 detail_status=FAILED，绝不拿列表 HTML 冒充详情正文。
- backfill_legacy() 回填无 raw_path 的旧记录；verify_raw_news() 逐条校验找回完整性。
"""
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml

from .db import get_conn
from .timeutil import parse_source_time, now_iso

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "sources.yaml"
RAW_NEWS_DIR = BASE_DIR / "data" / "raw" / "news"
COLLECTOR_VERSION = "2"   # 1.1 底座版

HEADERS = {"User-Agent": "Mozilla/5.0 (CorvusInfoBot/0.2; +local-research)"}


def load_sources():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return {s["source_id"]: s for s in yaml.safe_load(f)["sources"]}


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------- 原文落盘 ----------------

def _raw_file_path(source_id, published_at, raw_id, suffix=".json"):
    """目录 data/raw/news/{source_id}/{YYYYMMDD}/，日期取 published_at 否则取采集日。"""
    day = "unknown_date"
    if published_at:
        day = str(published_at)[:10].replace("-", "")
    else:
        day = now_iso()[:10].replace("-", "")
    d = RAW_NEWS_DIR / source_id / day
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{raw_id}{suffix}"


def _write_raw_file(path, text):
    """只追加：已存在则不覆盖（幂等）。"""
    if path.exists():
        return
    path.write_text(text, encoding="utf-8")


def _detail_path(list_path):
    """详情文件与列表文件同目录，按 raw_id 派生（raw_path -> {raw_id}_detail.html）。"""
    return list_path.with_name(list_path.stem + "_detail.html")


def save_raw_news(conn, source_id, title, url, raw_text, source_time_raw=None,
                  mime="application/json", encoding="utf-8", content_license=None,
                  detail_status="TODO"):
    """写 raw_news：去重 -> 插行 -> 落盘原文 -> 回写 raw_path。返回 (raw_id, is_new)。

    原文文件名为 {raw_id}.json，重复采集同内容只命中去重、不覆盖文件。
    """
    raw_hash = _sha256_text(raw_text)
    dup = conn.execute(
        "SELECT raw_id FROM raw_news WHERE source_id=? AND raw_text_hash=?",
        (source_id, raw_hash)).fetchone()
    if dup:
        return dup["raw_id"], False

    published_at, tz_inf, tps, _ = parse_source_time(source_time_raw)
    data_status = "COMPLETE" if tps == "OK" else "PARTIAL"
    cur = conn.execute(
        """INSERT INTO raw_news
           (source_id, title, url, raw_text_hash, published_at, collected_at,
            raw_path, content_license, mime, encoding, source_time_raw,
            tz_inference, time_parse_status, detail_status, data_status)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (source_id, title, url, raw_hash, published_at, now_iso(),
         None, content_license, mime, encoding, source_time_raw,
         tz_inf, tps, detail_status, data_status))
    raw_id = cur.lastrowid
    path = _raw_file_path(source_id, published_at, raw_id)
    _write_raw_file(path, raw_text)
    conn.execute("UPDATE raw_news SET raw_path=? WHERE raw_id=?",
                 (str(path), raw_id))
    return raw_id, True


# ---------------- 采集运行账本 ----------------

def start_run(conn, source_id):
    cur = conn.execute(
        """INSERT INTO collection_run (source_id, started_at, collector_version)
           VALUES (?,?,?)""",
        (source_id, now_iso(), COLLECTOR_VERSION))
    conn.commit()
    return cur.lastrowid


def finish_run(conn, run_id, stats):
    """stats: fetched/inserted/duplicate/failed/latest_published_at/errors[]。"""
    errors = stats.get("errors") or []
    if stats.get("failed") and stats.get("inserted") == 0:
        status = "FAILED"
    elif errors:
        status = "PARTIAL"
    else:
        status = "SUCCESS"
    conn.execute(
        """UPDATE collection_run SET finished_at=?, status=?, fetched_count=?,
           inserted_count=?, duplicate_count=?, failed_count=?,
           latest_published_at=?, error_summary=? WHERE run_id=?""",
        (now_iso(), status, stats.get("fetched"), stats.get("inserted"),
         stats.get("duplicate"), stats.get("failed"),
         stats.get("latest_published_at"),
         "; ".join(str(e)[:200] for e in errors[:5]) or None, run_id))
    conn.commit()


# ---------------- akshare 聚合资讯适配器（东财/财联社） ----------------

def collect_eastmoney_cls(conn, limit=50):
    """财联社快讯，akshare 真实接口 stock_info_global_cls。正文在"内容"列，detail_status=OK。"""
    import akshare as ak
    df = ak.stock_info_global_cls(symbol="全部")
    stats = {"fetched": 0, "inserted": 0, "duplicate": 0, "failed": 0,
             "latest_published_at": None, "errors": []}
    latest = None
    for _, r in df.head(limit).iterrows():
        stats["fetched"] += 1
        title = str(r.get("标题") or "")
        content = str(r.get("内容") or "")
        if not title and not content:
            stats["failed"] += 1
            continue
        d, t = r.get("发布日期"), r.get("发布时间")
        source_time_raw = f"{d} {t}" if d is not None and t is not None else None
        payload = {"标题": title, "内容": content, "发布日期": str(d) if d is not None else None,
                   "发布时间": str(t) if t is not None else None}
        try:
            raw_id, is_new = save_raw_news(
                conn, "EASTMONEY_NEWS", title or None, None,
                json.dumps(payload, ensure_ascii=False), source_time_raw=source_time_raw,
                detail_status="OK")  # 列表项即含正文，无需详情页
            stats["inserted" if is_new else "duplicate"] += 1
            pub, _, tps, _ = parse_source_time(source_time_raw)
            if pub and (latest is None or pub > latest):
                latest = pub
        except Exception as e:
            stats["failed"] += 1
            stats["errors"].append(f"row: {type(e).__name__}: {e}")
    stats["latest_published_at"] = latest
    return stats


# ---------------- 通用 HTTP 抓取骨架（官方网页源） ----------------

def http_fetch(url, timeout=10):
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text, resp.url


def collect_generic_official(conn, source_id, list_url, parse_list, parse_detail):
    """官方网页源通用采集器。

    - 列表项元数据照常存 raw_news，raw_path 指列表项 JSON（绝不冒充详情）；
    - 详情抓取成功：同目录写 {raw_id}_detail.html，detail_status=OK；
    - 详情抓取失败：detail_status=FAILED，正文列保持 NULL，可用 retry_detail(raw_id) 补拉。
    """
    stats = {"fetched": 0, "inserted": 0, "duplicate": 0, "failed": 0,
             "latest_published_at": None, "errors": []}
    try:
        html, _ = http_fetch(list_url)
    except Exception as e:
        stats["errors"].append(f"list_page: {type(e).__name__}: {e}")
        stats["failed"] += 1
        return stats

    try:
        items = parse_list(html)
    except Exception as e:
        stats["errors"].append(f"parse_list: {type(e).__name__}: {e}")
        return stats

    _t0 = time.time()
    for it in items:
        if time.time() - _t0 > 180:  # 单源详情预算：防止网络挂死拖垮整轮
            stats["errors"].append("detail time budget 180s exhausted, rest deferred to retry_detail")
            break
        stats["fetched"] += 1
        # 1) 列表元数据落库（列表片段即原文，正文还没抓到）
        list_payload = json.dumps({"list_item": it, "list_page_url": list_url},
                                  ensure_ascii=False)
        try:
            raw_id, is_new = save_raw_news(
                conn, source_id, it.get("title"), it.get("url"), list_payload,
                source_time_raw=it.get("published_at"), detail_status="PENDING")
            stats["inserted" if is_new else "duplicate"] += 1
        except Exception as e:
            stats["failed"] += 1
            stats["errors"].append(f"save: {type(e).__name__}: {e}")
            continue
        pub, _, tps, _ = parse_source_time(it.get("published_at"))
        if pub and (stats["latest_published_at"] is None or pub > stats["latest_published_at"]):
            stats["latest_published_at"] = pub
        if not it.get("url"):
            conn.execute("UPDATE raw_news SET detail_status='TODO' WHERE raw_id=?", (raw_id,))
            continue
        # 2) 详情页
        if not is_new:
            continue  # 旧记录不重抓详情，交由 retry_detail 按需补
        try:
            detail_html, final_url = http_fetch(it["url"])
            text = parse_detail(detail_html)
            row = conn.execute("SELECT raw_path FROM raw_news WHERE raw_id=?",
                               (raw_id,)).fetchone()
            dpath = _detail_path(Path(row["raw_path"]))
            _write_raw_file(dpath, text or detail_html)
            conn.execute(
                """UPDATE raw_news SET detail_status='OK', url=?,
                   mime='application/json' WHERE raw_id=?""",
                (final_url, raw_id))
        except Exception as e:
            # 详情失败：只标状态，列表 HTML 不冒充详情正文
            conn.execute("UPDATE raw_news SET detail_status='FAILED' WHERE raw_id=?",
                         (raw_id,))
            stats["errors"].append(f"detail raw_id={raw_id}: {type(e).__name__}: {e}")
        time.sleep(1)  # 礼貌抓取
    return stats


def retry_detail(conn, raw_id):
    """单独重试某条记录的详情页。返回 (success, message)。"""
    row = conn.execute(
        "SELECT raw_id, source_id, url, raw_path, detail_status FROM raw_news WHERE raw_id=?",
        (raw_id,)).fetchone()
    if row is None:
        return False, "raw_id 不存在"
    if not row["url"]:
        return False, "无详情 URL"
    parser = DETAIL_PARSERS.get(row["source_id"])
    if parser is None:
        if row["source_id"] in ("CNINFO", "SSE", "SZSE"):
            from .official import retry_pdf_detail  # 延迟导入避免环
            return retry_pdf_detail(conn, raw_id)
        return False, f"{row['source_id']} 详情解析器 TODO"
    try:
        detail_html, final_url = http_fetch(row["url"])
        text = parser(detail_html)
        dpath = _detail_path(Path(row["raw_path"]))
        _write_raw_file(dpath, text or detail_html)
        conn.execute(
            "UPDATE raw_news SET detail_status='OK', url=? WHERE raw_id=?",
            (final_url, raw_id))
        conn.commit()
        return True, f"OK -> {dpath}"
    except Exception as e:
        conn.execute("UPDATE raw_news SET detail_status='FAILED' WHERE raw_id=?", (raw_id,))
        conn.commit()
        return False, f"{type(e).__name__}: {e}"


# ---------------- 官方源解析器占位（TODO） ----------------

def _todo_parser(source_id):
    def _fn(conn):
        raise NotImplementedError(
            f"{source_id} 解析器待实现（通用骨架已就绪，见 collect_generic_official）")
    return _fn


# 详情正文解析器（per-source，TODO）
DETAIL_PARSERS = {}


COLLECTORS = {
    "EASTMONEY_NEWS": collect_eastmoney_cls,
    "CSRC": _todo_parser("CSRC"),
    "PBOC": _todo_parser("PBOC"),
    "CNINFO": "official.cninfo_collect",
    "SSE": "official.sse_collect",
    "SZSE": "official.szse_collect",
    # TOOLS_NEWS：本地聚合器，对接方式 TODO
}


# ---------------- 一次性迁移：旧记录回填 ----------------

def backfill_legacy(conn, source_id="EASTMONEY_NEWS"):
    """对 raw_path 为 NULL 的旧记录重新拉源补齐原文；拉不到的打 LEGACY_MISSING，不许删。

    只对聚合源可行（官方源历史列表已不可重建）。返回 {raw_id: 结果}。
    """
    rows = conn.execute(
        "SELECT raw_id, source_id, raw_text_hash FROM raw_news WHERE raw_path IS NULL"
    ).fetchall()
    results = {}
    if not rows:
        return results
    if source_id == "EASTMONEY_NEWS":
        import akshare as ak
        df = ak.stock_info_global_cls(symbol="全部")
        by_hash = {}
        for _, r in df.head(500).iterrows():
            d, t = r.get("发布日期"), r.get("发布时间")
            payload = {"标题": str(r.get("标题") or ""), "内容": str(r.get("内容") or ""),
                       "发布日期": str(d) if d is not None else None,
                       "发布时间": str(t) if t is not None else None}
            by_hash[_sha256_text(json.dumps(payload, ensure_ascii=False))] = \
                (payload, f"{d} {t}" if d is not None and t is not None else None)
    else:
        by_hash = {}
    for row in rows:
        hit = by_hash.get(row["raw_text_hash"])
        if hit:
            payload, source_time_raw = hit
            published_at, tz_inf, tps, _ = parse_source_time(source_time_raw)
            path = _raw_file_path(row["source_id"], published_at, row["raw_id"])
            _write_raw_file(path, json.dumps(payload, ensure_ascii=False))
            conn.execute(
                """UPDATE raw_news SET raw_path=?, source_time_raw=?, published_at=?,
                   tz_inference=?, time_parse_status=?, detail_status='OK',
                   mime='application/json', encoding='utf-8',
                   data_status='COMPLETE' WHERE raw_id=?""",
                (str(path), source_time_raw, published_at, tz_inf, tps, row["raw_id"]))
            results[row["raw_id"]] = "RECOVERED"
        else:
            conn.execute(
                "UPDATE raw_news SET time_parse_status='LEGACY_MISSING' WHERE raw_id=?",
                (row["raw_id"],))
            results[row["raw_id"]] = "LEGACY_MISSING（源已不可找回，仅保留元数据）"
    conn.commit()
    return results


# ---------------- 原文找回校验 ----------------

def verify_raw_news(db_path=None):
    """逐条校验：raw_path 存在、非空、重算 sha256 与库中 raw_text_hash 一致。

    LEGACY_MISSING（旧采集器丢正文且源已不可找回）是显式登记的审计缺口：
    要求已被标记且没有 raw_path，否则不算绿。
    返回 (ok, details)；ok=True = 所有记录要么通过哈希校验，要么已登记 LEGACY_MISSING。
    """
    conn = get_conn(db_path)
    details = []
    ok = True
    try:
        rows = conn.execute(
            "SELECT raw_id, source_id, raw_path, raw_text_hash, time_parse_status "
            "FROM raw_news ORDER BY raw_id").fetchall()
        for r in rows:
            rid, rhash, tps = r["raw_id"], r["raw_text_hash"], r["time_parse_status"]
            p = r["raw_path"]
            if not p:
                if tps == "LEGACY_MISSING":
                    # 已登记的审计缺口：旧版采集器丢正文，源已不可找回
                    details.append({"raw_id": rid, "ok": True, "known_missing": True,
                                    "reason": "LEGACY_MISSING（已登记）"})
                else:
                    details.append({"raw_id": rid, "ok": False,
                                    "reason": f"无 raw_path（{tps or '未标记'}）"})
                    ok = False
                continue
            path = Path(p)
            if not path.exists():
                details.append({"raw_id": rid, "ok": False, "reason": "文件不存在"})
                ok = False
                continue
            blob = path.read_bytes()
            if len(blob) == 0:
                details.append({"raw_id": rid, "ok": False, "reason": "文件为空"})
                ok = False
                continue
            actual = hashlib.sha256(blob).hexdigest()
            if actual != rhash:
                details.append({"raw_id": rid, "ok": False, "reason": "sha256 不一致"})
                ok = False
                continue
            details.append({"raw_id": rid, "ok": True, "path": p, "bytes": len(blob)})
    finally:
        conn.close()
    return ok, details


def run(source_id=None, db_path=None):
    """运行采集：source_id 为空则跑全部已注册的源，每源记 collection_run 账本。"""
    conn = get_conn(db_path)
    results = {}
    try:
        ids = [source_id] if source_id else list(COLLECTORS.keys())
        for sid in ids:
            fn = COLLECTORS.get(sid)
            if fn is None:
                results[sid] = "未注册采集器"
                continue
            if isinstance(fn, str):  # "official.cninfo_collect" 形式，延迟导入避免环
                import importlib
                mod_name, fn_name = fn.split(".")
                fn = getattr(importlib.import_module("." + mod_name, __package__), fn_name)
            run_id = start_run(conn, sid)
            try:
                stats = fn(conn)
                finish_run(conn, run_id, stats)
                results[sid] = (f"{stats['inserted']} 新增 / {stats['duplicate']} 重复 / "
                                f"{stats['failed']} 失败")
            except NotImplementedError as e:
                conn.execute(
                    "UPDATE collection_run SET finished_at=?, status='FAILED', error_summary=? WHERE run_id=?",
                    (now_iso(), str(e)[:200], run_id))
                conn.commit()
                results[sid] = f"TODO: {e}"
            except Exception as e:
                conn.execute(
                    "UPDATE collection_run SET finished_at=?, status='FAILED', error_summary=? WHERE run_id=?",
                    (now_iso(), f"{type(e).__name__}: {e}"[:200], run_id))
                conn.commit()
                results[sid] = f"失败: {type(e).__name__}: {e}"
    finally:
        conn.close()
    return results


if __name__ == "__main__":
    import sys
    print(json.dumps(run(sys.argv[1] if len(sys.argv) > 1 else None),
                     ensure_ascii=False, indent=2))
