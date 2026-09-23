"""review_tool.py：候选表人工质检（命令行交互）。

- 列出 PENDING/NEEDS_REVIEW（P2_ROUTED 为预路由中性，不需评审） 候选（默认 10 条一批）；
- a 接受 / r 拒绝（记原因）/ e 编辑（逐字段改）/ s 跳过 / q 退出；
- 接受才复制晋升到 information_event；--stats 非交互输出质检统计。
SCHEMA_VALIDATED=通过格式与字典检查，不代表事实已验证（2026-09-12 由 AUTO_VALIDATED 改名）。
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

from .db import get_conn
from .extract_candidates import read_raw_text
from .timeutil import now_iso

REJECT_REASONS = ["误抽取", "旧闻", "无关", "类型错", "其他"]
EDIT_FIELDS = ["event_type", "entities", "action", "amount", "official_scope", "effective_date"]


def _content_snippet(conn, cand, n=300):
    raw = conn.execute("SELECT raw_path, mime, detail_mime, title FROM raw_news WHERE raw_id=?",
                       (cand["raw_id"],)).fetchone()
    if not raw or not raw["raw_path"]:
        return "(无原文)"
    title, text = read_raw_text(raw["raw_path"], raw["mime"])
    if not text and raw["detail_mime"] == "application/pdf":
        text = read_pdf_text(raw["raw_path"])   # 评审时也要能看到 PDF 正文
    body = (text or "")[:n]
    src_title = title or raw["title"] or "(无)"
    return f"标题: {cand['source_id']} | {src_title}\n正文: {body or '(PDF/无文本)'}"


def _show(conn, cand):
    print("=" * 70)
    print(f"candidate_id={cand['candidate_id']} raw_id={cand['raw_id']} "
          f"[{cand['source_id']}] status={cand['candidate_status']}")
    print(_content_snippet(conn, cand))
    print(f"-- Qwen 六字段: event_type={cand['event_type']} entities={cand['entities']}")
    print(f"   action={cand['action']} amount={cand['amount']}")
    print(f"   official_scope={cand['official_scope']} effective_date={cand['effective_date']}")
    if cand["rule_check"]:
        rc = json.loads(cand["rule_check"])
        for c in rc.get("checks", []):
            mark = "PASS" if c["pass"] else "FAIL"
            print(f"   [{mark}] {c['rule']}: {c.get('detail', '')[:60]}")


def accept(conn, cand, reviewer="cli"):
    """晋升到 information_event（只有人工接受才走这条路）。"""
    raw = conn.execute(
        "SELECT * FROM raw_news WHERE raw_id=?", (cand["raw_id"],)).fetchone()
    evidence = json.dumps([cand["raw_id"]], ensure_ascii=False)
    cur = conn.execute(
        """INSERT INTO information_event
           (canonical_event_id, event_version, first_seen_at, source_id, source_level,
            source_url, published_at, collected_at, available_at, language,
            mapping_version, raw_id, raw_text_hash, evidence_ids, data_status,
            event_type, entities, action, amount, official_scope, effective_date)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (f"CAND-{cand['candidate_id']:08d}", 1, now_iso(), cand["source_id"],
         _source_level(cand["source_id"]), raw["url"] if raw else None,
         raw["published_at"] if raw else None,
         raw["collected_at"] if raw else now_iso(),
         raw["published_at"] if raw else None, "zh", "v1",
         cand["raw_id"], raw["raw_text_hash"] if raw else None,
         evidence, "COMPLETE", cand["event_type"], cand["entities"],
         cand["action"], cand["amount"], cand["official_scope"], cand["effective_date"]))
    conn.execute(
        """UPDATE information_event_candidate SET candidate_status='HUMAN_ACCEPTED',
           reviewed_by=?, reviewed_at=? WHERE candidate_id=?""",
        (reviewer, now_iso(), cand["candidate_id"]))
    conn.commit()
    return cur.lastrowid


def _source_level(source_id):
    from .extract_candidates import load_source_levels
    return load_source_levels().get(source_id, "AGGREGATOR")


def reject(conn, cand, note, reviewer="cli"):
    conn.execute(
        """UPDATE information_event_candidate SET candidate_status='HUMAN_REJECTED',
           reviewed_by=?, reviewed_at=?, review_note=? WHERE candidate_id=?""",
        (reviewer, now_iso(), note, cand["candidate_id"]))
    conn.commit()


def edit(conn, cand):
    print("编辑模式（回车保持不变）：")
    updates = {}
    for f in EDIT_FIELDS:
        cur = cand[f]
        if f == "entities" and cur:
            cur = json.loads(cur)
        new = input(f"  {f} [{cur}]: ").strip()
        if new:
            if f == "entities":
                updates[f] = json.dumps([e.strip() for e in new.split(",") if e.strip()],
                                        ensure_ascii=False)
            elif f == "amount":
                updates[f] = float(new)
            else:
                updates[f] = new
    if updates:
        sets = ", ".join(f"{k}=?" for k in updates)
        conn.execute(f"UPDATE information_event_candidate SET {sets} WHERE candidate_id=?",
                     (*updates.values(), cand["candidate_id"]))
        conn.commit()
        print("已保存修改（建议随后重新校验规则，本期由人工把关）")
    return updates


def interactive(conn, batch=10, reviewer="cli"):
    while True:
        rows = conn.execute(
            """SELECT * FROM information_event_candidate
               WHERE candidate_status IN ('PENDING','NEEDS_REVIEW')
               ORDER BY candidate_id LIMIT ?""", (batch,)).fetchall()
        if not rows:
            print("没有待审候选。")
            return
        for cand in rows:
            _show(conn, cand)
            while True:
                cmd = input("命令 [a接受/r拒绝/e编辑/s跳过/q退出]: ").strip().lower()
                if cmd == "a":
                    eid = accept(conn, cand, reviewer)
                    print(f"  -> 已晋升 information_event event_id={eid}")
                    break
                elif cmd == "r":
                    print("拒绝原因:", "/".join(REJECT_REASONS))
                    reason = input("  原因（枚举+自由文本）: ").strip() or "其他"
                    reject(conn, cand, reason, reviewer)
                    print("  -> 已拒绝")
                    break
                elif cmd == "e":
                    edit(conn, cand)
                    cand = conn.execute(
                        "SELECT * FROM information_event_candidate WHERE candidate_id=?",
                        (cand["candidate_id"],)).fetchone()
                    continue
                elif cmd == "s":
                    break
                elif cmd == "q":
                    return
                else:
                    print("  无效命令")


def stats(db_path=None):
    conn = get_conn(db_path)
    try:
        print("--- 候选状态计数 ---")
        for r in conn.execute(
                "SELECT candidate_status, COUNT(*) FROM information_event_candidate "
                "GROUP BY candidate_status ORDER BY 2 DESC"):
            print(f"  {r[0]}: {r[1]}")
        print("--- 拒绝原因分布 ---")
        for r in conn.execute(
                "SELECT review_note, COUNT(*) FROM information_event_candidate "
                "WHERE candidate_status='HUMAN_REJECTED' GROUP BY review_note"):
            print(f"  {r[0]}: {r[1]}")
        print("--- 按 source_id 解析成功率 ---")
        for r in conn.execute(
                "SELECT source_id, COUNT(*), SUM(parse_ok), "
                "ROUND(100.0*SUM(parse_ok)/COUNT(*),1) "
                "FROM information_event_candidate GROUP BY source_id"):
            print(f"  {r[0]}: {r[1]} 条, parse_ok {r[2]}, 成功率 {r[3]}%")
    finally:
        conn.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="候选表人工质检")
    ap.add_argument("--stats", action="store_true", help="非交互统计")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--reviewer", default="cli")
    args = ap.parse_args(argv)
    if args.stats:
        stats()
        return
    conn = get_conn()
    try:
        interactive(conn, args.batch, args.reviewer)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
