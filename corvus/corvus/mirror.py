"""SSE → 巨潮正文镜像（mirror.py，规格 v1.2 任务 A）。

背景：本网络访问 www.sse.com.cn 被 SSL 重置，上交所公告 PDF 下载失败。
用户决策（2026-09-13）："官方元数据 + 巨潮正文镜像"降级方案，代理不作为生产依赖。

流程：
    SSE 官方页面 → 保存权威元数据（采集链已做）
    → 官方正文失败 → 巨潮多键匹配（股票代码 + 公告日期 + 规范化标题相似度 + PDF 哈希）
    → 匹配成功：复制巨潮 .detail.pdf 为 SSE 详情，body_status=MIRROR_MATCHED
    → 匹配失败：body_status=METADATA_ONLY 且 data_status=PARTIAL（PARTIAL 不进自动风险判断）
    → 官方正文本就可得：body_status=FULL_TEXT

镜像匹配保留双来源证据链：body_source=CNINFO_MIRROR + body_hash 可回验。
幂等可重跑。SSE 独有内容（监管函/纪律处分/业务通知，巨潮无镜像）保持 METADATA_ONLY。
"""
import difflib
import hashlib
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

_CODE_RE = re.compile(r"\b(\d{6})\b")
_TITLE_NORM_RE = re.compile(r"[\s\W_]+", re.UNICODE)
SIM_THRESHOLD = 0.6          # 规范化标题相似度阈值
DATE_TOLERANCE_DAYS = 1      # 公告日期容差（±1 天）


def _norm_title(t):
    return _TITLE_NORM_RE.sub("", str(t or ""))


def _load_meta(raw_path, mime):
    """从列表页 JSON 提取 (股票代码, 公告日期, 标题)。字段因源而异：
    SSE=SECURITY_CODE/SSEDATE/TITLE；CNINFO=secCode/announcementTime/announcementTitle。"""
    import json as _json
    try:
        payload = _json.load(open(str(raw_path), encoding="utf-8"))
    except Exception:
        return None, None, None
    if not isinstance(payload, dict):
        return None, None, None
    code = payload.get("SECURITY_CODE") or payload.get("secCode")
    raw_day = payload.get("SSEDATE") or payload.get("announcementTime")
    day = str(raw_day or "")[:10]
    if isinstance(raw_day, (int, float)) or (isinstance(raw_day, str) and raw_day.isdigit()):
        # 巨潮 announcementTime 为毫秒时间戳
        day = datetime.fromtimestamp(int(raw_day) / 1000).strftime("%Y-%m-%d")
    title = payload.get("TITLE") or payload.get("announcementTitle")
    return (str(code).strip() if code else None,
            day if re.match(r"\d{4}-\d{2}-\d{2}", day) else None,
            title)


def _pdf_path(raw_path):
    return Path(str(raw_path)).with_suffix(".detail.pdf")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def match_cninfo_mirror(conn, verbose=True):
    """对 SSE 且官方详情失败的行做巨潮镜像匹配。返回统计 dict。"""
    stats = {"full_text": 0, "mirror_matched": 0, "metadata_only": 0,
             "no_code": 0, "skipped": 0}
    rows = conn.execute(
        """SELECT raw_id, title, url, published_at, raw_path, detail_status, detail_mime
           FROM raw_news WHERE source_id = 'SSE'""").fetchall()

    # 预载巨潮池：从列表页 JSON 读 secCode + 公告日期（标题常不含代码，不能从标题抓）
    pool = {}
    for r in conn.execute(
            """SELECT raw_id, title, raw_path, mime, published_at FROM raw_news
               WHERE source_id = 'CNINFO' AND raw_path IS NOT NULL""").fetchall():
        code, day, _t = _load_meta(r["raw_path"], r["mime"])
        if not code or not day:
            continue
        pool.setdefault((code, day), []).append(r)

    for r in rows:
        if not r["raw_path"]:
            stats["skipped"] += 1
            continue
        pdf = _pdf_path(r["raw_path"])
        if pdf.exists():
            conn.execute(
                """UPDATE raw_news SET body_source='SSE', body_status='FULL_TEXT',
                   body_hash=? WHERE raw_id=?""",
                (_sha256(pdf), r["raw_id"]))
            stats["full_text"] += 1
            continue
        if r["detail_status"] != "FAILED":
            stats["skipped"] += 1   # 非失败态（如 PENDING）留待官方重试，不由镜像接管
            continue
        code, _d, _t = _load_meta(r["raw_path"], None)
        if not code:
            stats["no_code"] += 1   # 监管函/业务通知等无证券代码 → 巨潮无从匹配
            conn.execute(
                """UPDATE raw_news SET body_status='METADATA_ONLY',
                   data_status='PARTIAL' WHERE raw_id=?""", (r["raw_id"],))
            stats["metadata_only"] += 1
            continue
        pub_day = str(r["published_at"] or "")[:10]
        cands = []
        if re.match(r"\d{4}-\d{2}-\d{2}", pub_day):
            base = datetime.strptime(pub_day, "%Y-%m-%d")
            for d in range(-DATE_TOLERANCE_DAYS, DATE_TOLERANCE_DAYS + 1):
                cands += pool.get((code, (base + timedelta(days=d)).strftime("%Y-%m-%d")), [])
        best, best_sim = None, 0.0
        tgt = _norm_title(r["title"])
        for c in cands:
            sim = difflib.SequenceMatcher(None, tgt, _norm_title(c["title"])).ratio()
            if sim > best_sim:
                best, best_sim = c, sim
        if best and best_sim >= SIM_THRESHOLD:
            src_pdf = _pdf_path(best["raw_path"])
            if src_pdf.exists():
                shutil.copy2(src_pdf, pdf)
                conn.execute(
                    """UPDATE raw_news SET body_source='CNINFO_MIRROR',
                       body_status='MIRROR_MATCHED', body_hash=?,
                       mirror_match_method='code+date+title_sim',
                       mirror_match_confidence=?, detail_mime='application/pdf'
                       WHERE raw_id=?""",
                    (_sha256(pdf), round(best_sim, 4), r["raw_id"]))
                stats["mirror_matched"] += 1
                if verbose:
                    print(f"  MIRROR raw_id={r['raw_id']} <- cninfo raw_id={best['raw_id']} "
                          f"sim={best_sim:.2f} {str(r['title'])[:36]}", flush=True)
                continue
        conn.execute(
            """UPDATE raw_news SET body_status='METADATA_ONLY',
               data_status='PARTIAL' WHERE raw_id=?""", (r["raw_id"],))
        stats["metadata_only"] += 1
    conn.commit()
    return stats


def main(argv=None):
    from .db import get_conn, init_db
    init_db()
    conn = get_conn()
    try:
        stats = match_cninfo_mirror(conn)
        print(f"SSE 镜像匹配：{stats}", flush=True)
        day = datetime.now().strftime("%Y%m%d")
        out = Path(__file__).resolve().parent.parent / "data" / "reports" / f"mirror_report_{day}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(str(stats), encoding="utf-8")
        print(f"报告 → {out}", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
