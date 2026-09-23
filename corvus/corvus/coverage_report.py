"""coverage_report.py：外围序列覆盖与质量报告。

每个序列输出：起止日期、总条数、缺失率（按工作日历粗算）、重复率、异常跳变率
（SUSPECT 占比）、时区与交易日、盘前可用率（available_at 不晚于次一交易日 8:30 的比例）、
quality_status 分布。

用法：python -m corvus.coverage_report [--instrument SPX] [--db PATH]
"""
import argparse
import sqlite3
from datetime import datetime, timedelta

from .db import get_conn
from .market_collect import load_registry

PREMARKET_CUTOFF_H = 30  # 盘前快照 8:30：available_at <= event_time + 30 小时视为盘前可用


def _bizdays(start_d, end_d):
    """粗算预期交易日：区间内工作日数（不区分各市场节假日，缺失率因此是上限估计）。"""
    days = 0
    d = start_d
    while d <= end_d:
        if d.weekday() < 5:
            days += 1
        d += timedelta(days=1)
    return days


def report_one(conn, instrument_id, reg_row):
    rows = conn.execute(
        "SELECT event_time, available_at, quality_status FROM market_observation "
        "WHERE instrument_id=? ORDER BY event_time", (instrument_id,)).fetchall()
    n = len(rows)
    if n == 0:
        return {"instrument_id": instrument_id, "rows": 0,
                "note": reg_row.get("notes") or "无数据"}

    times = [datetime.fromisoformat(r["event_time"]) for r in rows]
    start_d, end_d = times[0].date(), times[-1].date()
    expected = _bizdays(start_d, end_d)
    dup = conn.execute(
        "SELECT COUNT(*) FROM (SELECT event_time FROM market_observation WHERE instrument_id=? "
        "GROUP BY event_time HAVING COUNT(*)>1)", (instrument_id,)).fetchone()[0]
    suspect = sum(1 for r in rows if r["quality_status"] == "SUSPECT")
    premkt = 0
    for r in rows:
        try:
            if datetime.fromisoformat(r["available_at"]) <= \
                    datetime.fromisoformat(r["event_time"]) + timedelta(hours=PREMARKET_CUTOFF_H):
                premkt += 1
        except (TypeError, ValueError):
            pass
    qdist = {}
    for r in rows:
        qdist[r["quality_status"]] = qdist.get(r["quality_status"], 0) + 1

    return {
        "instrument_id": instrument_id,
        "name_cn": reg_row.get("name_cn"),
        "tier": reg_row.get("tier"),
        "provider": reg_row.get("provider"),
        "rows": n,
        "start": str(start_d), "end": str(end_d),
        "missing_pct": round(100.0 * max(0, expected - n) / expected, 1) if expected else None,
        "dup_days": dup,
        "suspect_pct": round(100.0 * suspect / n, 2),
        "premkt_avail_pct": round(100.0 * premkt / n, 1),
        "quality_status": qdist,
        "calendar": reg_row.get("calendar"),
        "tz": reg_row.get("exchange_timezone"),
        "note": reg_row.get("notes"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="外围序列覆盖与质量报告")
    ap.add_argument("--instrument", default=None, help="只看某序列")
    ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)
    conn = get_conn(args.db)
    try:
        reg = load_registry()
        ids = [args.instrument] if args.instrument else list(reg.keys())
        print(f"{'ID':<12}{'rows':>7}{'start':>12}{'end':>12}{'缺%':>6}{'dup':>5}"
              f"{'Sus%':>7}{'盘前%':>7}  provider")
        for iid in ids:
            try:
                r = report_one(conn, iid, reg.get(iid, {}))
            except Exception as e:
                print(f"{iid:<12} 报告失败: {type(e).__name__}: {e}")
                continue
            if r.get("rows", 0) == 0:
                print(f"{iid:<12}{0:>7}  {'--':>12}{'--':>12}{'--':>6}{'--':>5}"
                      f"{'--':>7}{'--':>7}  {r.get('note') or '无数据'}")
                continue
            print(f"{iid:<12}{r['rows']:>7}{r['start']:>12}{r['end']:>12}"
                  f"{r['missing_pct']:>6}{r['dup_days']:>5}{r['suspect_pct']:>7}"
                  f"{r['premkt_avail_pct']:>7}  {r['provider']}")
            if r.get("note"):
                print(f"{'':<12}note: {str(r['note'])[:100]}")
        # 全库重复键检查（应为 0）
        dups = conn.execute(
            "SELECT COUNT(*) FROM (SELECT instrument_id, event_time FROM market_observation "
            "GROUP BY instrument_id, event_time HAVING COUNT(*)>1)").fetchone()[0]
        print(f"\n全库重复 (instrument_id, event_time) 组合数: {dups}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
