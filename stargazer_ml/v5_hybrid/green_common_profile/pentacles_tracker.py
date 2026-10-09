"""pentacles_tracker.py — Pentacles 名单 7 交易日路径跟踪（学习记录配套工具）。

口径（冻结）：
- 信号 = 观察日首次 low <= 区域上沿（无区股票不入场，仅记收盘路径）
- 入场 = 信号后下一合法连续竞价分钟 open（冻结执行合同语义；涨停/停牌无 bar 记 NO_FILL）
- 窗口 = 入场日起 7 个交易日（含当日），逐日记录 high/low/close（原始价）+ MFE/MAE（相对入场价，%）
- MFE/MAE 为路径质量指标，非收益、非卖出建议（PENTACLES_LEARNING.md 纪律）

用法：
  python -X utf8 pentacles_tracker.py --snapshot <snapshot.json> --out <tracker.json>
  python -X utf8 pentacles_tracker.py --update <tracker.json>   # 每日收盘后增量更新

零网络（只读本机 stockdb）；回填=确定性重放，输出标 backfilled=true 非前瞻证据。
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:7899"
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
LEGAL = [(931, 1130), (1301, 1500)]  # 合法连续竞价分钟（HHMM）


def sdb(t, timeout=60):
    url = BASE + '/?cmd=get&t=' + urllib.parse.quote(t)
    req = urllib.request.Request(url, headers={'Accept': 'application/x-msgpack'})
    import msgpack
    d = msgpack.unpackb(_OPENER.open(req, timeout=timeout).read(), raw=False,
                        strict_map_key=False)
    return d.get('data') if isinstance(d, dict) else d


def legal_minute(hhmm: int) -> bool:
    return any(a <= hhmm <= b for a, b in LEGAL)


def minute_bars(code: str, day: int):
    rows = sdb(f'分钟k:{code}:{day}*') or []
    out = []
    for row in rows:
        d = row[1] if isinstance(row, list) else row
        ts = int(d['date'])
        hhmm = ts % 1000000 // 100
        if legal_minute(hhmm) or hhmm == 1500:
            out.append((ts, float(d['open']), float(d['high']),
                        float(d['low']), float(d['close'])))
    return sorted(out)


def daily_bars(code: str, days: list):
    """逐日 OHLC（用分钟聚合；无分钟日回退日线）。"""
    out = {}
    for day in days:
        bars = minute_bars(code, day)
        if bars:
            out[day] = dict(high=max(b[2] for b in bars), low=min(b[3] for b in bars),
                            close=bars[-1][4], source='MINUTE')
            continue
        rows = sdb(f'日k:{code}:{day}*') or []
        for row in rows:
            d = row[1] if isinstance(row, list) else row
            if int(d.get('date', 0)) == day:
                out[day] = dict(high=float(d['high']), low=float(d['low']),
                                close=float(d['close']), source='DAILY')
                break
    return out


def trade_days(upto: int) -> list:
    rows = sdb(f'日k:600000:*') or []
    days = sorted({int((r[1] if isinstance(r, list) else r)['date']) for r in rows})
    return [d for d in days if d <= upto]


def find_entry(code: str, observe_day: int, zone_upper):
    """观察日分钟：首次 low<=zone_upper 的 bar → 下一合法分钟 open。"""
    bars = minute_bars(code, observe_day)
    for i, (ts, o, h, l, c) in enumerate(bars):
        if zone_upper is not None and l <= zone_upper:
            for ts2, o2, h2, l2, c2 in bars[i + 1:]:
                hhmm = ts2 % 1000000 // 100
                if legal_minute(hhmm):
                    return dict(signal_ts=ts, entry_ts=ts2, entry_price=o2,
                                signal_price=c)
            return dict(signal_ts=ts, entry_ts=None, entry_price=None,
                        signal_price=c, reason='NO_LEGAL_MINUTE_AFTER_SIGNAL')
    return None


def track(snapshot_path: str, out_path: str):
    snap = json.loads(Path(snapshot_path).read_text(encoding='utf-8'))
    observe = int(snap['monitor_date'])
    days_all = trade_days(20991231)
    try:
        oi = days_all.index(observe)
    except ValueError:
        raise SystemExit(f'观察日 {observe} 不在交易日历')
    window = days_all[oi: oi + 7]
    rows_out = []
    for r in snap['rows']:
        code = r['code']
        zu = float(r['zone']['zone_upper']) if r.get('zone') else None
        entry = find_entry(code, observe, zu) if zu else None
        ep = entry.get('entry_price') if entry else None
        rec = dict(code=code, name=r.get('name', ''), setup=r.get('setup'),
                   zone=f"{r['zone']['zone_lower']}-{r['zone']['zone_upper']}" if r.get('zone') else None,
                   signal_ts=entry.get('signal_ts') if entry else None,
                   entry_ts=entry.get('entry_ts') if entry else None,
                   entry_price=ep,
                   entry_note=(entry.get('reason') if entry and entry.get('reason')
                               else ('FILLED' if ep else ('NO_ZONE' if not zu else 'NOT_TOUCHED_D1'))),
                   days={})
        bars = daily_bars(code, window)
        for d in window:
            if d not in bars:
                rec['days'][d] = None
                continue
            b = bars[d]
            day = dict(high=b['high'], low=b['low'], close=b['close'], source=b['source'])
            if ep:
                day.update(mfe_pct=round((b['high'] / ep - 1) * 100, 2),
                           mae_pct=round((b['low'] / ep - 1) * 100, 2),
                           close_pct=round((b['close'] / ep - 1) * 100, 2))
            rec['days'][d] = day
        rows_out.append(rec)
    out = dict(tracker_version='pentacles_tracker_v1', snapshot=str(snapshot_path),
               observe_date=observe, window=window, backfilled=True,
               disclaimer='MFE/MAE=路径质量指标，非收益非卖出建议；backfilled=回溯描述，非前瞻证据',
               rows=rows_out)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                              encoding='utf-8')
    n_fill = sum(1 for r in rows_out if r['entry_price'])
    print(f'tracker → {out_path} | 入场 {n_fill}/{len(rows_out)} | 窗口 {window[0]}..{window[-1]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot')
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.snapshot and a.out:
        track(a.snapshot, a.out)
    else:
        ap.error('需要 --snapshot 和 --out')


if __name__ == '__main__':
    main()
