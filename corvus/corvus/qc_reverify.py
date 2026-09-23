"""50 条针对性复验导出（规格 v1.2 任务 E）：
- 年份编造嫌疑：effective CONFLICT 且 Qwen 年份 < 发布年份（全部，目标 18）
- 否定语境：assertion_status='NEGATED'/'HISTORICAL_REFERENCE'（目标 10）
- 金额：amount_status in (CALCULATED, AMBIGUOUS) 抽样（目标 10）
- SSE 镜像：body_status='MIRROR_MATCHED' 抽样（目标 10）
- UNMAPPED：tier='UNKNOWN' 抽样（目标 10）
"""
import json
import random
import sqlite3
from pathlib import Path

from .db import get_conn, init_db
from .extract_candidates import read_pdf_text, read_raw_text

OUT = Path(__file__).resolve().parent.parent / "data" / "qc_reverify_pack.txt"


def build(conn):
    rng = random.Random(11)
    picks, seen = [], set()

    def add(rows, n):
        rows = [r for r in rows if r["candidate_id"] not in seen]
        take = rows if len(rows) <= n else rng.sample(rows, n)
        for r in take:
            seen.add(r["candidate_id"])
            picks.append(r)

    y1 = """SELECT c.candidate_id, c.source_id, c.event_type, c.materiality_tier,
                   c.assertion_status, c.effective_date_status, c.amount_status,
                   c.amount, c.effective_date, c.candidate_status,
                   r.title, r.raw_path, r.mime, r.detail_mime, r.published_at,
                   r.body_status, r.body_source, r.mirror_match_confidence
            FROM information_event_candidate c JOIN raw_news r ON r.raw_id=c.raw_id"""
    add(conn.execute(y1 + """
        WHERE c.effective_date_status='CONFLICT'
          AND substr(c.effective_date_normalized,1,4) < substr(r.published_at,1,4)""").fetchall(), 18)
    add(conn.execute(y1 + """
        WHERE c.assertion_status IN ('NEGATED','HISTORICAL_REFERENCE')""").fetchall(), 10)
    add(conn.execute(y1 + """
        WHERE c.amount_status IN ('CALCULATED','AMBIGUOUS')""").fetchall(), 10)
    add(conn.execute(y1 + """
        WHERE r.body_status='MIRROR_MATCHED'""").fetchall(), 10)
    add(conn.execute(y1 + """
        WHERE c.materiality_tier='UNKNOWN'""").fetchall(), 10)
    return picks


def export(conn, picks):
    L = [f"# 50 条针对性复验包（{len(picks)} 条，种子 11）"]
    for r in picks:
        title, text = (None, None)
        if r["raw_path"]:
            title, text = read_raw_text(r["raw_path"], r["mime"])
            if not text and r["detail_mime"] == "application/pdf":
                text = read_pdf_text(r["raw_path"])
        L.append("=" * 78)
        L.append(f"candidate_id={r['candidate_id']} [{r['source_id']}] tier={r['materiality_tier']} "
                 f"assertion={r['assertion_status']} eff={r['effective_date_status']}({r['effective_date']}) "
                 f"amount={r['amount_status']}({r['amount']})")
        L.append(f"标题: {r['title']}")
        L.append(f"正文(截500): {(text or '')[:500]}")
        L.append(f"published_at={r['published_at']} body={r['body_status']}/{r['body_source']} "
                 f"conf={r['mirror_match_confidence']}")
        L.append("")
    OUT.write_text("\n".join(L), encoding="utf-8")
    return OUT


def main(argv=None):
    init_db()
    conn = get_conn()
    try:
        picks = build(conn)
        path = export(conn, picks)
        print(f"复验包 {len(picks)} 条 → {path}")
        from collections import Counter
        print(Counter((r["source_id"], r["materiality_tier"]) for r in picks))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
