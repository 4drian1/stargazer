"""200 条分层人工质检：抽样 + 定向抽取（新闻线 F-v2.2 并行任务）。

抽样口径（规格 11.3，2026-09-12 冻结）：
- CNINFO 80 / SZSE 30 / SSE 30 / EASTMONEY_NEWS 40（财联社未入库，缺口记录于报告）
- 每层先按事件类型关键词桶覆盖（业绩/增减持/回购/ST/退市/立案处罚/问询/停复牌/政策），
  不足配额用随机补足；绝不按 2024 后表现挑选——质检是评估抽取质量，不是选样本外证据。
- 只抽尚未进候选表（未被 Qwen 处理过）的 raw_news。

用法（corvus/ 目录）：
    python -m corvus.qc_extract --sample-only          # 只生成抽样清单
    python -m corvus.qc_extract                        # 抽样 + 逐条 Qwen 抽取入候选表
抽完后用 python -m corvus.review_tool 人工评审。
"""
import argparse
import json
import random
import re
from pathlib import Path

from .db import get_conn
from .extract_candidates import (build_input, load_source_levels, read_pdf_text,
                                 read_raw_text, validate_candidate)
from .llm_extract import extract_facts
from .timeutil import now_iso

QUOTA = {"CNINFO": 80, "SZSE": 30, "SSE": 30, "EASTMONEY_NEWS": 40}
# 事件类型关键词桶（覆盖规格要求的类型面）
KEYWORD_BUCKETS = [
    "业绩预告|业绩快报|扭亏|预亏|净利润",
    "减持|增持",
    "回购",
    "ST|退市|终止上市",
    "立案|处罚|警示函|违规",
    "问询|关注函|监管函",
    "停牌|复牌",
    "政策|规划|意见|通知|决定",
]
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "qc_sample_200.json"


def _pick_sample(conn):
    """分层抽样：每桶按关键词取最早未处理的若干条，不足配额随机补足。"""
    rng = random.Random(42)   # 可复现
    sample = []
    for source, quota in QUOTA.items():
        rows = conn.execute(
            """SELECT r.raw_id, r.title FROM raw_news r
               LEFT JOIN information_event_candidate c ON c.raw_id = r.raw_id
               WHERE c.raw_id IS NULL AND r.source_id = ? AND r.raw_path IS NOT NULL
               ORDER BY r.published_at""", (source,)).fetchall()
        titles = {r["raw_id"]: (r["title"] or "") for r in rows}
        picked, used = [], set()

        def take(pred, n):
            got = 0
            for r in rows:
                if got >= n:
                    break
                rid = r["raw_id"]
                if rid in used:
                    continue
                if pred(titles[rid]):
                    used.add(rid)
                    picked.append(rid)
                    got += 1
            return got

        # 每桶先取 3 条（配额小的源减到 2），覆盖事件类型面
        per_bucket = 3 if quota >= 60 else 2
        for kw in KEYWORD_BUCKETS:
            take(lambda t, kw=kw: re.search(kw, t) is not None, per_bucket)
        # 不足配额随机补足
        rest = [r["raw_id"] for r in rows if r["raw_id"] not in used]
        rng.shuffle(rest)
        for rid in rest[:max(quota - len(picked), 0)]:
            used.add(rid)
            picked.append(rid)
        sample.append({"source_id": source, "quota": quota, "picked": len(picked),
                       "raw_ids": picked[:quota]})
    return sample


def run_extract(conn, sample, verbose=True):
    """对抽样清单逐条 Qwen 抽取，写入候选表（与 extract_candidates.run_round 同写入口径）。"""
    levels = load_source_levels()
    total = {"processed": 0, "parse_ok": 0, "auto_validated": 0, "needs_review": 0,
             "failed": 0, "skipped_pdf": 0}
    for grp in sample:
        for rid in grp["raw_ids"]:
            r = conn.execute(
                """SELECT raw_id, source_id, title, raw_path, mime, detail_mime, published_at
                   FROM raw_news WHERE raw_id=?""", (rid,)).fetchone()
            if not r or not r["raw_path"]:
                continue
            already = conn.execute(
                "SELECT 1 FROM information_event_candidate WHERE raw_id=?", (rid,)).fetchone()
            if already:
                continue
            title, text = read_raw_text(r["raw_path"], r["mime"])
            if not text and r["detail_mime"] == "application/pdf":
                text = read_pdf_text(r["raw_path"])   # 详情 PDF 正文抽取（pypdf）
            if not text and not title:
                total["skipped_pdf"] += 1
                continue
            total["processed"] += 1
            try:
                res = extract_facts(build_input(title, text, r["title"]))
            except Exception as e:
                res = {"parsed": None, "raw_text": "", "parse_ok": False,
                       "latency_s": None, "error": f"{type(e).__name__}: {e}"}
            p = res.get("parsed")
            if res.get("parse_ok") and p is not None:
                total["parse_ok"] += 1
                rc = validate_candidate(levels.get(r["source_id"], "AGGREGATOR"), p,
                                        r["title"], text, r["published_at"])
                status = rc["status"]
            else:
                total["failed"] += 1
                rc = {"checks": [{"rule": "parse", "pass": False, "level": "error",
                                  "detail": res.get("error") or "Qwen 输出解析失败（已重试）"}],
                      "status": "NEEDS_REVIEW", "warnings": [],
                      "canonical_event_type": None, "materiality_tier": None,
                      "is_container_document": 0, "underlying_event_types": []}
                status = "NEEDS_REVIEW"
                p = {}
            total["schema_validated" if status == "SCHEMA_VALIDATED" else "needs_review"] += 1
            conn.execute(
                """INSERT INTO information_event_candidate
                   (raw_id, source_id, event_type, canonical_event_type, materiality_tier,
                    is_container_document, underlying_event_types,
                    entities, action, amount, official_scope, effective_date,
                    extract_raw, parse_ok, latency_s, model_version, rule_check,
                    candidate_status, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rid, r["source_id"], p.get("event_type"), rc.get("canonical_event_type"),
                 rc.get("materiality_tier"), rc.get("is_container_document", 0),
                 json.dumps(rc.get("underlying_event_types") or [], ensure_ascii=False),
                 json.dumps(p.get("entities"), ensure_ascii=False)
                 if p.get("entities") is not None else None,
                 p.get("action"), p.get("amount"), p.get("official_scope"),
                 p.get("effective_date"), res.get("raw_text"),
                 1 if res.get("parse_ok") else 0, res.get("latency_s"),
                 "qwen3-4b-q4_k_m", json.dumps(rc, ensure_ascii=False), status, now_iso()))
            conn.commit()
            if verbose:
                print(f"  raw_id={rid} [{r['source_id']}] parse_ok={res.get('parse_ok')} "
                      f"status={status} type={rc.get('canonical_event_type') or p.get('event_type')}",
                      flush=True)
    total.setdefault("schema_validated", 0)
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description="200 条分层质检抽样与抽取")
    ap.add_argument("--sample-only", action="store_true")
    ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)
    conn = get_conn(args.db)
    try:
        sample = _pick_sample(conn)
        n = sum(len(g["raw_ids"]) for g in sample)
        SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SAMPLE_PATH.write_text(json.dumps(sample, ensure_ascii=False, indent=1),
                               encoding="utf-8")
        for g in sample:
            print(f"  {g['source_id']}: 配额 {g['quota']} → 实抽 {g['picked']}", flush=True)
        print(f"抽样清单 → {SAMPLE_PATH}（共 {n} 条）", flush=True)
        if args.sample_only:
            return
        print("开始逐条 Qwen 抽取...", flush=True)
        total = run_extract(conn, sample)
        print(f"抽取完成: {total}", flush=True)
        print("下一步：python -m corvus.review_tool 人工评审（a接受/r拒绝/e编辑/s跳过）", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
