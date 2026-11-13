"""Prefix-only market repair observations. No labels, fits, orders or API calls."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GCP = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(GCP))
from golden_ladder_v3 import STOPS, _rsi, index_map

FEATURES = [
    'map_known', 'golden_distance_atr', 'two_close_reclaim', 'reclaim_tests',
    'stop_above', 'rsi6', 'rsi15', 'rsi_gap', 'rsi_above', 'cross_up_count',
    'crosses_since_low', 'bars_since_cross', 'bars_since_low',
    'last_two_lows_rising', 'last_two_closes_rising', 'volume_after_low_ratio',
    'volume_last3_previous3_ratio', 'low_rebound_return', 'low_rebound_atr',
    'return_3bars', 'repair_joint', 'repair_two_cross_joint',
]


def day_observations(bars, rsi6, rsi15, levels, atr, prior_rsi=None):
    """Map is frozen before this session; missing source bars invalidate its prefix."""
    if len(bars) != len(rsi6) or len(bars) != len(rsi15):
        raise ValueError('RSI_LENGTH_MISMATCH')
    out = []
    levels = sorted(set(float(x) for x in levels if x > 0))
    mapped = bool(levels) and np.isfinite(atr) and atr > 0
    invalid = False
    for i in range(len(bars)):
        row = dict.fromkeys(FEATURES, np.nan)
        prefix = bars.iloc[:i + 1]
        prices = prefix[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        invalid = invalid or not np.isfinite(prices[-1]).all()
        if invalid:
            out.append(row)
            continue
        low_i = int(np.argmin(prefix.low.to_numpy()))
        low = float(prefix.low.iloc[low_i])
        close = float(prefix.close.iloc[-1])
        a, b = np.asarray(rsi6[:i + 1]), np.asarray(rsi15[:i + 1])
        crosses = np.flatnonzero((a[:-1] <= b[:-1]) & (a[1:] > b[1:])) + 1
        if prior_rsi is not None and prior_rsi[0] <= prior_rsi[1] and a[0] > b[0]:
            crosses = np.r_[0, crosses]
        if np.isfinite(a[-1]) and np.isfinite(b[-1]):
            row.update(rsi6=float(a[-1]), rsi15=float(b[-1]),
                       rsi_gap=float(a[-1] - b[-1]), rsi_above=float(a[-1] > b[-1]))
        # Count is unknown until all cross comparisons in this session are warmed up.
        if np.isfinite(a).all() and np.isfinite(b).all():
            row['cross_up_count'] = float(len(crosses))
            row['crosses_since_low'] = float((crosses >= low_i).sum())
            if len(crosses):
                row['bars_since_cross'] = float(i - crosses[-1])
        row.update(map_known=float(mapped), bars_since_low=float(i - low_i),
                   low_rebound_return=close / low - 1)
        if i >= 1:
            row['last_two_lows_rising'] = float(prefix.low.iloc[-1] >= prefix.low.iloc[-2])
            row['last_two_closes_rising'] = float(close > prefix.close.iloc[-2])
        if i >= 3:
            row['return_3bars'] = close / prefix.close.iloc[-4] - 1
        volumes = prefix.volume.to_numpy(dtype=float)
        if i >= low_i + 2 and np.isfinite(volumes[[low_i, i - 1, i]]).all() and volumes[low_i] > 0:
            row['volume_after_low_ratio'] = float(volumes[-2:].mean() / volumes[low_i])
        if i >= 5 and np.isfinite(volumes[-6:]).all() and volumes[-6:-3].sum() > 0:
            row['volume_last3_previous3_ratio'] = float(volumes[-3:].sum() / volumes[-6:-3].sum())
        if mapped:
            row['low_rebound_atr'] = (close - low) / atr
            stops = [n for n in levels if n <= low and low - n <= .2 * atr]
            stop_node = max(stops) if stops else None
            stable = (i >= low_i + 2 and prefix.low.iloc[-1] >= prefix.low.iloc[-2] >= low
                      and close > prefix.close.iloc[-2] >= prefix.close.iloc[low_i])
            contracting = row['volume_after_low_ratio'] <= 1
            row['stop_above'] = (0. if stop_node is None or not stable else
                                 float(contracting) if np.isfinite(row['volume_after_low_ratio']) else np.nan)
            # A later new low removes an earlier reclaim until two closes confirm again.
            reclaimed = [n for n in levels if i >= max(1, low_i + 1)
                         and (prefix.low.iloc[:-1] < n).any()
                         and prefix.close.iloc[-2] >= n and close >= n]
            reclaim_node = min(reclaimed, key=lambda n: abs(close - n)) if reclaimed else None
            node = reclaim_node or stop_node or min(levels, key=lambda n: abs(close - n))
            row['golden_distance_atr'] = (close - node) / atr
            row['two_close_reclaim'] = float(reclaim_node is not None)
            if reclaim_node is not None:
                touched = prefix.low.to_numpy() < reclaim_node
                recovered = prefix.close.to_numpy() >= reclaim_node
                prior_below = np.r_[False, prefix.close.to_numpy()[:-1] < reclaim_node]
                prior_untouched = np.r_[True, ~touched[:-1]]
                row['reclaim_tests'] = float((recovered & touched & (prior_below | prior_untouched)).sum())
            else:
                row['reclaim_tests'] = 0.
            known = all(np.isfinite(row[k]) for k in ['rsi_above', 'volume_after_low_ratio'])
            if row['stop_above'] == 0 and row['two_close_reclaim'] == 0:
                row['repair_joint'] = row['repair_two_cross_joint'] = 0.
            elif known:
                joint = bool((row['stop_above'] == 1 or row['two_close_reclaim'] == 1)
                             and row['rsi_above'] == 1 and contracting)
                row['repair_joint'] = float(joint)
                if np.isfinite(row['cross_up_count']):
                    row['repair_two_cross_joint'] = float(joint and row['cross_up_count'] >= 2)
        out.append(row)
    return pd.DataFrame(out, index=bars.index)


def market_panel(native, name, wanted_days):
    """Use observed session dates and exact 48-slot grid. Never fill missing OHLCV."""
    f = native.rename(columns={'ts': 'date', 'raw_volume': 'volume'}).copy()
    if f.date.duplicated().any():
        raise ValueError('DUPLICATE_INDEX_BAR')
    f = f.sort_values('date')
    p = f[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
    if (not np.isfinite(p).all() or (p <= 0).any()
        or (f.low > f[['open', 'close']].min(axis=1)).any()
        or (f.high < f[['open', 'close']].max(axis=1)).any()):
        raise ValueError('INVALID_INDEX_OHLC')
    days = sorted(set((f.date // 1000000).astype(int)))
    clock = np.array([d * 1000000 + t for d in days for t in STOPS], dtype=np.int64)
    if not f.date.isin(clock).all():
        raise ValueError('OFF_SESSION_INDEX_BAR')
    grid = f.set_index('date').reindex(clock)
    grid.index.name = 'date'
    grid['day'] = grid.index.to_numpy() // 1000000
    daily = grid.groupby('day').agg(high=('high', 'max'), low=('low', 'min'),
                                   close=('close', 'last'), count=('close', 'count'))
    daily.loc[daily['count'] != 48, ['high', 'low', 'close']] = np.nan
    prev = daily.close.shift(1)
    tr = pd.concat([daily.high - daily.low, (daily.high - prev).abs(),
                    (daily.low - prev).abs()], axis=1).max(axis=1, skipna=False)
    atrs = tr.rolling(14, min_periods=14).mean().shift(1)
    a, b = _rsi(grid.close.to_numpy(), 6), _rsi(grid.close.to_numpy(), 15)
    output, receipts = [], []
    for day in sorted(set(wanted_days)):
        positions = np.flatnonzero(grid.day.to_numpy() == day)
        if not len(positions):
            continue
        prior = [d for d in days if d < day]
        mapping = index_map(f, prior[-1] * 1000000 + 150000) if prior else {}
        known = mapping.get('status') == 'AVAILABLE_EXECUTION_REFERENCE_NOT_PYXIS_FEATURE'
        nodes = [item['price'] for item in mapping['levels']] if known else []
        bars = grid.iloc[positions]
        prior_rsi = (a[positions[0] - 1], b[positions[0] - 1]) if positions[0] else (np.nan, np.nan)
        result = day_observations(bars, a[positions], b[positions], nodes, atrs.loc[day], prior_rsi)
        result = result.rename(columns={n: name + '_repair_' + n for n in FEATURES})
        result.insert(0, 'decision_ts_assumed_end', bars.index.to_numpy())
        output.append(result)
        receipts.append(dict(day=day, mapping=mapping, atr_prior=None if pd.isna(atrs.loc[day]) else float(atrs.loc[day])))
    if not output:
        raise ValueError('NO_INDEX_DATES_FOR_COHORT')
    return pd.concat(output, ignore_index=True), receipts
