"""Recompute frozen common rules on latest local daily data; no network/training."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'stargazer_desktop'))

import pandas as pd
from gcp_round_b_engine import build_atr_ref, build_zones, target_zone
from gcp_demo_pools import process_latest_code


def calendar_file():
    return Path(importlib.util.find_spec('akshare').origin).parent / 'file_fold/calendar.json'


def plan_dates(days, now):
    today = int(now.strftime('%Y%m%d'))
    completed = [day for day in days if day < today or (day == today and now.hour >= 15)]
    if today in days and now.hour < 15:
        raise ValueError('请收盘后再生成次日名单，盘中日线尚未完成')
    if not completed:
        raise ValueError('交易日历未覆盖当前日期')
    previous = max(completed)
    following = next((day for day in days if day > previous), None)
    if following is None or today > max(days):
        raise ValueError('交易日历已到期，请更新日历')
    return previous, following


def ensure_daily_current(daily_dir, now=None):
    """Prepare the local TDX source, never start a stockdb refresh."""
    now = now or datetime.now()
    days = sorted(map(int,json.loads(calendar_file().read_bytes())))
    expected, _ = plan_dates(days, now)
    from tdx_daily_source_v2 import daily_directory
    return daily_directory(Path(__file__).resolve().parents[3],expected)


def build(daily_dir, now=None):
    now = now or datetime.now()
    calendar = calendar_file()
    days = sorted(int(day) for day in json.loads(calendar.read_text(encoding='utf-8')))
    expected_date, next_day = plan_dates(days, now)
    from tdx_daily_source_v2 import daily_directory
    daily_dir=daily_directory(Path(__file__).resolve().parents[3],expected_date)
    parts = sorted(daily_dir.glob('part_*.parquet'))
    if not parts:
        raise ValueError('本地日线分片不存在')
    frames = []
    for part in parts:
        # Keep the full causal history for RSI/state-machine warmup; do not truncate to 60 rows.
        frame = pd.read_parquet(part, columns=['date', 'code', 'name', 'open', 'high', 'low',
                               'close', 'pre_close', 'volume', 'amount', 'is_st', 'float_mv'],
                               filters=[('date', '<=', expected_date)])
        frame['code'] = frame.code.astype(str).str.zfill(6)
        frame = frame[frame.code.str.match(r'^(60|68|00|30)\d{4}$')]
        if not frame.empty:
            frames.append(frame)
    daily = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not daily.empty:
        daily = daily.drop_duplicates(['code', 'date'], keep='last').sort_values(['code', 'date'])
    else:
        raise ValueError('本地日线为空')
    date = int(daily.date.max())
    if date < expected_date:
        raise ValueError(
            f'日线分片数据止于 {date}，未到预期数据日 {expected_date}'
            f'（stockdb 已更新但 data_daily_parts 未同步）。'
            f'请先跑 green_v2/refresh_pool.py 或数据同步工具后再生成名单。')
    rows = []
    for code, g in daily.groupby('code', sort=False):
        if int(g.date.iloc[-1]) != date:
            continue
        g = g.reset_index(drop=True)
        a, b = process_latest_code(g)
        members = pd.concat([p[p.date == date] for p in (a, b) if not p.empty], ignore_index=True) if not a.empty or not b.empty else pd.DataFrame()
        if members.empty:
            continue
        r = members.iloc[0]
        out = {'code': code, 'name': '', 'setup': r.setup_type, 'date': date,
               'reference_price': None, 'zone': None, 'atr_ref': None, 'distance_atr': None,
               'rsi14': float(r.rsi14), 'status': 'DAILY_MISSING', 'signal': '实时信号未验收'}
        if not g.empty and int(g.date.iloc[-1]) == date:
            out['name'] = str(g.iloc[-1].get('name', ''))
            price = float(g.close.iloc[-1])
            atr = build_atr_ref(g)
            zone = target_zone(build_zones(g, atr), price)
            out.update(reference_price=price, atr_ref=atr if pd.notna(atr) else None,
                       zone=zone, status='REFERENCE_ONLY' if zone else 'NO_ZONE')
            if zone:
                out['distance_atr'] = (price - zone['zone_upper']) / atr
        rows.append(out)
    rows.sort(key=lambda r: (r['distance_atr'] is None, r['distance_atr'] or 0, r['code']))
    for i, row in enumerate(rows, 1):
        row['rank'] = i
    return {'schema': 'PAGE_OF_PENTACLES_DEMO_V1', 'strategy': 'green_common_profile_demo',
            'created_at': now.astimezone().isoformat(), 'data_date': date,
            'expected_data_date': expected_date, 'monitor_date': next_day,
            'plan_ready': date == expected_date,
            'source': '最新本地日线重算冻结共性规则；盘后数据', 'rows': rows,
            'ranking': '关注区域距离（ATR），不是盈利评分',
            'model_status': '共性规则＋区域研究参考；逐股个性预测未验收',
            'inputs': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in [*parts, calendar, Path(__file__),
                                 Path(__file__).with_name('gcp_demo_pools.py'),
                                 Path(__file__).with_name('gcp_v0_1_spec.py')]]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--daily-dir', type=Path, required=True)
    args = parser.parse_args()
    ensure_daily_current(args.daily_dir)
    print(json.dumps(build(args.daily_dir), ensure_ascii=False, allow_nan=False))
