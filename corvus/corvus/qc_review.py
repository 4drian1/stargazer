"""约 60 条分层人工评审（QC 验收）：按用户规格选样 + 导出评审包。

选样口径（2026-09-12）：
- RETROACTIVE（effective 早于发布 >5 天）最多 12 条
- 日期不可解析 3 条
- SCHEMA_VALIDATED 8 条（格式通过的最需验证"事实是否真的对"）
- 每源随机 8 条（共 32）
- 重大事件 P0 8 条
合计约 63 条。导出 data/qc_review_pack.txt 供逐条评审，评审结论写回候选表。
"""
import json
import random
import sqlite3
from pathlib import Path

from .db import get_conn, init_db
from .extract_candidates import read_pdf_text, read_raw_text

OUT = Path(__file__).resolve().parent.parent / "data" / "qc_review_pack.txt"


def _first_fail(rule_check):
    try:
        rc = json.loads(rule_check or "{}")
        for c in rc.get("checks", []):
            if not c["pass"]:
                return f"{c['rule']}: {c.get('detail', '')[:80]}"
    except Exception:
        pass
    return ""


def pick(conn):
    rng = random.Random(7)
    rows = conn.execute(
        """SELECT c.candidate_id, c.source_id, c.event_type, c.canonical_event_type,
                  c.materiality_tier, c.entities, c.action, c.amount,
                  c.official_scope, c.effective_date, c.candidate_status, c.rule_check,
                  r.title, r.raw_path, r.mime, r.detail_mime, r.published_at
           FROM information_event_candidate c JOIN raw_news r ON r.raw_id=c.raw_id
           WHERE c.candidate_status IN ('NEEDS_REVIEW','SCHEMA_VALIDATED')""").fetchall()
    by = {"retro": [], "date": [], "schema": [], "p0": [], "src": {}}
    for r in rows:
        ff = _first_fail(r["rule_check"])
        if "RETROACTIVE" in ff:
            by["retro"].append(r)
        elif "effective_date_parseable" in ff:
            by["date"].append(r)
        elif r["candidate_status"] == "SCHEMA_VALIDATED":
            by["schema"].append(r)
        elif r["materiality_tier"] == "P0":
            by["p0"].append(r)
        by["src"].setdefault(r["source_id"], []).append(r)
    picked = []
    picked += rng.sample(by["retro"], min(12, len(by["retro"])))
    picked += rng.sample(by["date"], min(3, len(by["date"])))
    picked += rng.sample(by["schema"], min(8, len(by["schema"])))
    picked += rng.sample(by["p0"], min(8, len(by["p0"])))
    for s, lst in by["src"].items():
        rest = [r for r in lst if r not in picked]
        picked += rng.sample(rest, min(8, len(rest)))
    # 去重（一行可能命中多个桶）
    seen, out = set(), []
    for r in picked:
        if r["candidate_id"] not in seen:
            seen.add(r["candidate_id"])
            out.append(r)
    return out


def export(conn, picked):
    L = []
    for r in picked:
        title, text = (None, None)
        if r["raw_path"]:
            title, text = read_raw_text(r["raw_path"], r["mime"])
            if not text and r["detail_mime"] == "application/pdf":
                text = read_pdf_text(r["raw_path"])
        L.append("=" * 78)
        L.append(f"candidate_id={r['candidate_id']} [{r['source_id']}] "
                 f"status={r['candidate_status']} tier={r['materiality_tier']} "
                 f"canonical={r['canonical_event_type']}")
        L.append(f"标题: {r['title']}")
        L.append(f"正文(截断400字): {(text or '')[:400]}")
        L.append(f"published_at={r['published_at']}")
        L.append(f"Qwen六字段: type={r['event_type']} | entities={r['entities']}")
        L.append(f"  action={r['action']} | amount={r['amount']} | scope={r['official_scope']}")
        L.append(f"  effective_date={r['effective_date']}")
        L.append(f"首因失败: {_first_fail(r['rule_check']) or '(无，SCHEMA 全过)'}")
        L.append("")
    OUT.write_text("\n".join(L), encoding="utf-8")
    return OUT


def main(argv=None):
    init_db()
    conn = get_conn()
    try:
        picked = pick(conn)
        path = export(conn, picked)
        print(f"评审包 {len(picked)} 条 → {path}")
        for r in picked:
            print(f"  {r['candidate_id']} [{r['source_id']}] {r['candidate_status']} "
                  f"{r['materiality_tier']} {str(r['title'])[:40]}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
