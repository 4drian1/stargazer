"""daily_run.py：Corvus 前瞻采集每日运行器（新闻无法回补，稳定挂机优先）。

模式：
- news   增量采集全部已注册新闻源（TODO 源自动跳过）
- market 增量采集全部可用外围序列（从库内最大 event_time 前 10 天续传）
- extract 对未抽取 raw_news 跑 Qwen 候选抽取（限量）
- health 生成采集健康报告（含"程序成功但连续无新增"报警）
- all    news + market + extract + health（每日收盘后跑一次）

健康报告位置：data/health/health_YYYYMMDD_HHMMSS.txt
用法：python -m corvus.daily_run --mode all
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from .db import get_conn, init_db
from .market_collect import backfill, load_registry

BASE_DIR = Path(__file__).resolve().parent.parent
HEALTH_DIR = BASE_DIR / "data" / "health"

CORE_TIER = "core_v0"
ZERO_NEW_ALERT_RUNS = 3  # 连续 N 轮零新增即报警

# 前瞻采集新闻源（CSRC/PBOC 解析器 TODO 完成前不跑，避免账本噪音）
NEWS_SOURCES = ["EASTMONEY_NEWS", "CNINFO", "SZSE", "SSE"]
NEWS_TIMEOUT_S = 300     # 单源硬超时：挂死只杀它自己，不拖垮整轮


def _stamp_timeout_failed(source_id):
    """子进程被强杀后，把它未完成的账本行记为 FAILED。"""
    from .timeutil import now_iso
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT run_id FROM collection_run WHERE source_id=? AND finished_at IS NULL "
            "ORDER BY run_id DESC LIMIT 1", (source_id,)).fetchone()
        if row:
            conn.execute(
                "UPDATE collection_run SET finished_at=?, status='FAILED', error_summary=? "
                "WHERE run_id=?", (now_iso(), f"父进程超时 {NEWS_TIMEOUT_S}s 强杀", row[0]))
            conn.commit()
    finally:
        conn.close()


def run_news(source_id=None):
    init_db()   # 全新机器上库是空的：先建表再采集（news-one 子进程路径此前漏了这步）
    if source_id:
        from .news_collect import run as news_run
        return news_run(source_id)
    # 父进程：每源一个子进程硬隔离（本网络下 SSE 详情会挂死，不能让它拖垮整轮）
    import subprocess
    results = {}
    for sid in NEWS_SOURCES:
        try:
            cp = subprocess.run(
                [sys.executable, "-m", "corvus.daily_run", "--mode", "news-one",
                 "--source", sid],
                cwd=str(BASE_DIR), timeout=NEWS_TIMEOUT_S, capture_output=True, text=True,
                encoding="utf-8", errors="replace")
            # 合并 stderr：子进程崩溃原因必须可见（静默失败无法归因的教训）
            tail = (((cp.stdout or "") + " | " + (cp.stderr or ""))
                    .strip())[-150:]
            results[sid] = f"exit={cp.returncode} {tail}"
        except subprocess.TimeoutExpired:
            _stamp_timeout_failed(sid)
            results[sid] = f"TIMEOUT>{NEWS_TIMEOUT_S}s 已强杀（账本记 FAILED，可 retry_detail 补）"
    return results


def run_market():
    init_db()
    reg = load_registry()
    conn = get_conn()
    today = datetime.now().strftime("%Y-%m-%d")
    results = {}
    try:
        for iid, row in reg.items():
            if row.get("tier") != CORE_TIER:
                continue
            if row.get("provider") in ("MISSING", None) or not row.get("provider_symbol") \
                    or row.get("provider_symbol") in ("TODO", "null"):
                results[iid] = "SKIP(缺失/TODO，如实不报)"
                continue
            last = conn.execute(
                "SELECT MAX(event_time) FROM market_observation WHERE instrument_id=?",
                (iid,)).fetchone()[0]
            start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d") \
                if not last else str(last)[:10]
            try:
                n = backfill(iid, start, today)
                results[iid] = f"新增 {n}"
            except Exception as e:
                results[iid] = f"FAIL {type(e).__name__}: {str(e)[:80]}"
    finally:
        conn.close()
    return results


def run_extract(limit=30):
    from .extract_candidates import run_round
    from .mirror import match_cninfo_mirror
    conn = get_conn()
    try:
        mirror_stats = match_cninfo_mirror(conn, verbose=False)
        stats = run_round(conn, limit=limit)
        stats["mirror"] = mirror_stats
    finally:
        conn.close()
    return stats


def run_health():
    init_db()
    conn = get_conn()
    lines, alerts = [], []
    try:
        lines.append(f"# Corvus 采集健康报告 {datetime.now().isoformat(timespec='seconds')}")
        # 1. 每源最近运行 + 连续零新增报警
        lines.append("\n## 采集账本（每源最近 %d 轮）" % ZERO_NEW_ALERT_RUNS)
        for (sid,) in conn.execute("SELECT DISTINCT source_id FROM collection_run"):
            runs = conn.execute(
                "SELECT started_at, status, fetched_count, inserted_count, duplicate_count, "
                "failed_count, latest_published_at, error_summary FROM collection_run "
                "WHERE source_id=? ORDER BY started_at DESC LIMIT ?", (sid, ZERO_NEW_ALERT_RUNS)
            ).fetchall()
            lines.append(f"\n[{sid}]")
            for r in runs:
                lines.append(f"  {r['started_at'][:19]} {r['status'] or 'RUNNING'} "
                             f"fetched={r['fetched_count']} new={r['inserted_count']} "
                             f"dup={r['duplicate_count']} fail={r['failed_count']} "
                             f"latest_pub={str(r['latest_published_at'])[:19]}")
                if r["error_summary"]:
                    lines.append(f"    err: {r['error_summary'][:100]}")
            done = [r for r in runs if r["status"]]
            if len(done) >= ZERO_NEW_ALERT_RUNS and all(r["inserted_count"] == 0 for r in done):
                msg = f"[ALERT] {sid} 连续 {len(done)} 轮零新增（程序可能正常但源无产出/解析失效）"
                alerts.append(msg)
                lines.append("  " + msg)
        # 2. 原文找回率
        from .news_collect import verify_raw_news
        ok, details = verify_raw_news()
        n_ok = sum(1 for d in details if d["ok"])
        n_fail = sum(1 for d in details if not d["ok"])
        lines.append(f"\n## 原文找回率: {n_ok}/{len(details)} "
                     f"({100.0*n_ok/max(1,len(details)):.1f}%)，校验失败 {n_fail}")
        if n_fail:
            alerts.append(f"[ALERT] {n_fail} 条原文找回失败")
        # 3. 候选抽取统计
        lines.append("\n## Qwen 候选抽取")
        for r in conn.execute(
                "SELECT candidate_status, COUNT(*) FROM information_event_candidate GROUP BY 1"):
            lines.append(f"  {r[0]}: {r[1]}")
        for r in conn.execute(
                "SELECT source_id, COUNT(*), SUM(parse_ok), "
                "ROUND(100.0*SUM(parse_ok)/COUNT(*),1) FROM information_event_candidate "
                "GROUP BY source_id"):
            lines.append(f"  {r[0]}: {r[1]} 条, 解析成功率 {r[3]}%")
        # 4. 外围序列最新数据日期（发现停更）
        lines.append("\n## 外围序列最新 event_time")
        for r in conn.execute(
                "SELECT instrument_id, MAX(event_time) FROM market_observation GROUP BY 1"):
            stale = (datetime.now().astimezone() - datetime.fromisoformat(r[1])).days
            mark = "  [ALERT] 停超3天" if stale > 3 else ""
            lines.append(f"  {r[0]}: {str(r[1])[:10]}{mark}")
            if stale > 3:
                alerts.append(f"[ALERT] {r[0]} 数据停更 {stale} 天")
        lines.append("\n## 报警汇总" if alerts else "\n## 无报警")
        lines.extend(alerts)
    finally:
        conn.close()
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    path = HEALTH_DIR / f"health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path), alerts


def main(argv=None):
    ap = argparse.ArgumentParser(description="Corvus 每日运行器")
    ap.add_argument("--mode", choices=["news", "news-one", "market", "extract", "health", "all"],
                    default="all")
    ap.add_argument("--source", default=None, help="news-one 模式用：单源采集")
    ap.add_argument("--extract-limit", type=int, default=30)
    args = ap.parse_args(argv)
    out = {}
    if args.mode == "news-one":
        out["news-one"] = {args.source: run_news(args.source)}
    if args.mode in ("news", "all"):
        out["news"] = run_news()
    if args.mode in ("market", "all"):
        out["market"] = run_market()
    if args.mode in ("extract", "all"):
        out["extract"] = run_extract(args.extract_limit)
    if args.mode in ("health", "all"):
        path, alerts = run_health()
        out["health_report"] = path
        out["alerts"] = alerts
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
