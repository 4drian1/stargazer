"""snapshot_coverage.py：快照级可用率报告（训练闸门）。

区分两件事：
- 数据库里有没有数据（calendar_coverage）
- 在某个决策快照时这条数据是否已经可见（visibility_coverage）

只有 visibility_coverage 合格的序列才允许进入对应快照的模型。

语义：
- OVERNIGHT / PREMARKET_*：消费"前一交易日"的外围 bar（event_date 的下一工作日快照）
- INTRADAY_*：消费"当日"的亚洲 bar（event_date 同日快照）
判定一律走 visibility.is_available（available_at + session 推导 + 特殊规则）。

用法：python -m corvus.snapshot_coverage [--instrument SPX] [--db PATH]
"""
import argparse
from datetime import datetime, timedelta

from .db import get_conn
from .market_collect import load_registry
from .visibility import SNAPSHOTS, BEIJING, is_available

# 快照消费窗口：盘前/隔夜类消费 event_date 的下一工作日；盘中类消费当日
PREMARKET_TYPES = {"OVERNIGHT", "PREMARKET_830", "PREMARKET_915", "AUCTION_925"}
INTRADAY_TYPES = {"INTRADAY_1000", "INTRADAY_1030", "INTRADAY_1330"}

USABLE_CALENDAR = 85.0   # calendar_coverage 下限（%）
USABLE_VISIBILITY = 90.0  # visibility_coverage 下限（%）

# 代理/替代口径警告（不能解释成名义标的）
CALIVER_WARNINGS = {
    "GOLD": "国内商品代理（上金所现货 CNY/克），不能解释成纽约金价",
    "N225": "期货口径（NID），不能解释成日经现货",
    "USDCNY_BOC": "在岸 CNY 牌价，不能解释成离岸 CNH",
    "WTI": "EIA 现货口径，非期货连续合约",
}


def _next_bizday(d):
    d = d + timedelta(days=1)
    while d.weekday() >= 5:
        d = d + timedelta(days=1)
    return d


def _bizdays(start_d, end_d):
    n, d = 0, start_d
    while d <= end_d:
        if d.weekday() < 5:
            n += 1
        d += timedelta(days=1)
    return n


def _parse(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def report_instrument(conn, instrument_id, reg_row):
    rows = conn.execute(
        "SELECT event_time, available_at, quality_status FROM market_observation "
        "WHERE instrument_id=? ORDER BY event_time", (instrument_id,)).fetchall()
    if not rows:
        return []
    bars = [{"event_time": r["event_time"], "available_at": r["available_at"],
             "q": r["quality_status"]} for r in rows]
    start_d, end_d = _parse(bars[0]["event_time"]).date(), _parse(bars[-1]["event_time"]).date()
    calendar_cov = 100.0 * len(bars) / max(1, _bizdays(start_d, end_d))
    suspect_pct = 100.0 * sum(1 for b in bars if b["q"] == "SUSPECT") / len(bars)

    out = []
    for snap_type, hhmm in SNAPSHOTS.items():
        if snap_type in ("POST_MARKET",):
            continue  # 训练标签用盘前/盘中快照，盘后快照不在训练口径内
        vis = late = 0
        for b in bars:
            d = _parse(b["event_time"]).astimezone(BEIJING).date()
            if snap_type in PREMARKET_TYPES:
                dec = datetime.combine(_next_bizday(d), datetime.strptime(hhmm, "%H:%M").time(), BEIJING)
            else:
                dec = datetime.combine(d, datetime.strptime(hhmm, "%H:%M").time(), BEIJING)
            if is_available(b, dec.isoformat(), reg_row):
                vis += 1
            else:
                late += 1
        vis_cov = 100.0 * vis / len(bars)
        usable = calendar_cov >= USABLE_CALENDAR and vis_cov >= USABLE_VISIBILITY \
            and suspect_pct < 5.0
        out.append({
            "instrument_id": instrument_id,
            "snapshot_type": snap_type,
            "calendar_coverage": round(calendar_cov, 1),
            "visibility_coverage": round(vis_cov, 1),
            "stale_rate": round(100.0 - vis_cov, 1),
            "suspect_rate": round(suspect_pct, 2),
            "usable_for_training": "YES" if usable else "NO",
        })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="快照级可用率报告")
    ap.add_argument("--instrument", default=None)
    ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)
    conn = get_conn(args.db)
    try:
        reg = load_registry()
        ids = [args.instrument] if args.instrument else \
            [i for i, r in reg.items() if r.get("tier") == "core_v0"]
        print(f"{'instrument':<12}{'snapshot':<16}{'日历%':>7}{'可见%':>7}"
              f"{'滞后%':>7}{'Sus%':>7}  usable")
        for iid in ids:
            row = dict(reg[iid], instrument_id=iid)
            try:
                lines = report_instrument(conn, iid, row)
            except Exception as e:
                print(f"{iid:<12} 计算失败: {type(e).__name__}: {e}")
                continue
            if not lines:
                print(f"{iid:<12}{'(无数据)':<16}  provider={row.get('provider')}")
                continue
            for L in lines:
                print(f"{L['instrument_id']:<12}{L['snapshot_type']:<16}"
                      f"{L['calendar_coverage']:>7}{L['visibility_coverage']:>7}"
                      f"{L['stale_rate']:>7}{L['suspect_rate']:>7}  {L['usable_for_training']}")
            warn = CALIVER_WARNINGS.get(iid)
            if warn:
                print(f"{'':<12}口径警告: {warn}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
