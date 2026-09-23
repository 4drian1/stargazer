"""extract_candidates.py：raw_news -> Qwen 事实抽取 -> 规则校验 -> 候选表 批量运行器。

- 只处理未抽取过的 raw_news（LEFT JOIN 候选表，幂等）；
- PDF 公告（detail_mime=application/pdf）且无正文的跳过，统计单列"跳过 PDF"；
- 单条失败不中断整轮，失败原因记候选表；任何状态都不自动进 information_event。
"""
import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

from .db import get_conn
from .event_taxonomy import is_p2_by_title, map_canonical, tier_of
from .llm_extract import extract_facts
from .timeutil import now_iso
from .validate import validate_candidate

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "sources.yaml"

_TEXT_KEYS = ["内容", "content", "正文", "text", "DESCRIPTION", "摘要", "remark"]
_MAX_INPUT_CHARS = 2000
_TAG_RE = re.compile(r"<[^>]+>")


def load_source_levels():
    import yaml
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return {s["source_id"]: s["level"] for s in yaml.safe_load(f)["sources"]}


def read_pdf_text(raw_path, max_pages=20):
    """读详情 PDF 正文（pypdf）。PDF 落盘约定：与原文同目录、同 stem 的 .detail.pdf。
    公告正文通常在前几页，限制页数防爆。抽取失败/无文本返回 None（绝不编造）。"""
    p = Path(raw_path).with_suffix(".detail.pdf")
    if not p.exists():
        return None
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        parts = [(pg.extract_text() or "") for pg in reader.pages[:max_pages]]
        text = "\n".join(parts).strip()
        return text or None
    except Exception:
        return None


def read_raw_text(raw_path, mime):
    """读原文文件，返回 (title, text)。"""
    path = Path(raw_path)
    blob = path.read_bytes()
    title, text = None, None
    if mime == "application/json" or path.suffix == ".json":
        try:
            payload = json.loads(blob.decode("utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            title = payload.get("标题") or payload.get("title") or payload.get("TITLE")
            if isinstance(title, list):
                title = title[0] if title else None
            for k in _TEXT_KEYS:
                v = payload.get(k)
                if isinstance(v, str) and v.strip():
                    text = v.strip()
                    break
    elif mime == "text/html":
        text = _TAG_RE.sub(" ", blob.decode("utf-8", errors="ignore"))
    return title, text


def build_input(title, text, fallback_title):
    t = title or fallback_title or ""
    body = text or ""
    return (t + "\n" + body)[:_MAX_INPUT_CHARS]


def run_round(conn, limit=30, source_id=None, since=None, verbose=False,
              p2_pre_route=True, p2_sample_rate=0.08, seed=42):
    """跑一轮抽取。返回统计 dict。

    p2_pre_route=True 时，标题命中 P2 例行中性的新闻不调 LLM，直记 P2_ROUTED
    （2026-09-12 规格）；p2_sample_rate=8% 的 P2 仍走完整抽取作中性对照。"""
    import random as _random
    rng = _random.Random(seed)
    levels = load_source_levels()
    sql = """SELECT r.raw_id, r.source_id, r.title, r.raw_path, r.mime, r.detail_mime,
                    r.published_at, r.collected_at
             FROM raw_news r
             LEFT JOIN information_event_candidate c ON c.raw_id = r.raw_id
             WHERE c.raw_id IS NULL"""
    params = []
    if source_id:
        sql += " AND r.source_id = ?"
        params.append(source_id)
    if since:
        sql += " AND r.published_at >= ?"
        params.append(since)
    sql += " ORDER BY r.published_at DESC LIMIT ?"
    params.append(limit * 3)  # 多取一些，扣除 PDF 跳过后尽量凑够 limit 条文本

    rows = conn.execute(sql, params).fetchall()
    stats = {"seen": 0, "processed": 0, "extract_ok": 0, "schema_validated": 0,
             "needs_review": 0, "failed": 0, "skipped_pdf": 0, "p2_routed": 0}

    for r in rows:
        if stats["processed"] >= limit:
            break
        stats["seen"] += 1
        if not r["raw_path"]:
            stats["skipped_no_path"] = stats.get("skipped_no_path", 0) + 1
            continue
        title, text = read_raw_text(r["raw_path"], r["mime"])
        if not text and r["detail_mime"] == "application/pdf":
            text = read_pdf_text(r["raw_path"])   # 详情 PDF 正文抽取（pypdf）
        if not text and not title:
            stats["skipped_pdf"] += 1  # 无文本可抽（PDF 缺失/抽取失败），如实计数跳过
            continue
        # P2 预路由（2026-09-12 规格）：例行中性标题不调 LLM，只记轻量分类。
        # 抽样 5%–10% 作中性对照仍走完整抽取（由 daily_run 用 --p2-sample-rate 控制）。
        if p2_pre_route and is_p2_by_title(r["title"]) and rng.random() >= p2_sample_rate:
            canon, note = map_canonical(None, r["title"])
            rc = {"checks": [{"rule": "p2_pre_route", "pass": True, "level": "info",
                              "detail": f"标题预路由 P2：{note or '例行中性'}"}],
                  "status": "P2_ROUTED", "warnings": []}
            stats["p2_routed"] += 1
            conn.execute(
                """INSERT INTO information_event_candidate
                   (raw_id, source_id, canonical_event_type, materiality_tier,
                    rule_check, candidate_status, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (r["raw_id"], r["source_id"], canon, tier_of(canon) if canon else "P2",
                 json.dumps(rc, ensure_ascii=False), "P2_ROUTED", now_iso()))
            conn.commit()
            continue
        stats["processed"] += 1
        prompt_input = build_input(title, text, r["title"])
        try:
            res = extract_facts(prompt_input)
        except Exception as e:
            res = {"parsed": None, "raw_text": "", "parse_ok": False,
                   "latency_s": None, "error": f"{type(e).__name__}: {e}"}
        p = res.get("parsed")
        if res.get("parse_ok") and p is not None:
            stats["extract_ok"] += 1
            rc = validate_candidate(levels.get(r["source_id"], "AGGREGATOR"), p,
                                    r["title"], text, r["published_at"])
            status = rc["status"]
        else:
            stats["failed"] += 1
            rc = {"checks": [{"rule": "parse", "pass": False, "level": "error",
                              "detail": res.get("error") or "Qwen 输出解析失败（已重试）"}],
                  "status": "NEEDS_REVIEW", "warnings": [],
                  "canonical_event_type": None, "materiality_tier": None,
                  "is_container_document": 0, "underlying_event_types": []}
            status = "NEEDS_REVIEW"
            p = {}
        stats["schema_validated" if status == "SCHEMA_VALIDATED" else "needs_review"] += 1
        conn.execute(
            """INSERT INTO information_event_candidate
               (raw_id, source_id, event_type, canonical_event_type, materiality_tier,
                is_container_document, underlying_event_types,
                entities, action, amount, official_scope, effective_date,
                extract_raw, parse_ok, latency_s, model_version, rule_check,
                candidate_status, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (r["raw_id"], r["source_id"], p.get("event_type"), rc.get("canonical_event_type"),
             rc.get("materiality_tier"), rc.get("is_container_document", 0),
             json.dumps(rc.get("underlying_event_types") or [], ensure_ascii=False),
             json.dumps(p.get("entities"), ensure_ascii=False) if p.get("entities") is not None else None,
             p.get("action"), p.get("amount"), p.get("official_scope"),
             p.get("effective_date"), res.get("raw_text"),
             1 if res.get("parse_ok") else 0, res.get("latency_s"),
             "qwen3-4b-q4_k_m", json.dumps(rc, ensure_ascii=False), status, now_iso()))
        conn.commit()
        if verbose:
            print(f"  raw_id={r['raw_id']} [{r['source_id']}] "
                  f"parse_ok={res.get('parse_ok')} status={status} "
                  f"type={rc.get('canonical_event_type') or p.get('event_type')} "
                  f"tier={rc.get('materiality_tier')} latency={res.get('latency_s')}s",
                  flush=True)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description="批量抽取 raw_news -> 候选表")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--source-id", default=None)
    ap.add_argument("--since", default=None, help="如 2026-09-10")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    conn = get_conn()
    try:
        t0 = time.time()
        stats = run_round(conn, args.limit, args.source_id, args.since, args.verbose)
        print(f"本轮统计（耗时 {time.time()-t0:.0f}s）：{json.dumps(stats, ensure_ascii=False)}")
        if stats.get("skipped_pdf"):
            print(f"跳过无文本 {stats['skipped_pdf']} 条（PDF 缺失/抽取失败，已如实计数）")
        if stats.get("skipped_no_path"):
            print(f"跳过无原文文件 {stats['skipped_no_path']} 条（LEGACY_MISSING）")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
