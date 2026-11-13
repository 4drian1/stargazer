"""Market-only break/repair context using previously selected confirmed waves."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY))
from repair_features import day_observations, index_map, STOPS, _rsi
from golden_ladder_v3 import wave_levels

SELECT = ['map_known', 'two_close_reclaim', 'stop_above', 'repair_joint',
          'repair_two_cross_joint', 'reclaim_tests']
EXTRA = ['support_bands', 'broken_bands', 'broken_nearest_wave_count',
         'broken_nearest_peak_groups', 'broken_close_run', 'broken_close_depth_atr',
         'broken_no_repair', 'single_broken_no_repair', 'shared_peak_broken_no_repair',
         'distinct_peak_broken_no_repair', 'full_band_two_close_reclaim']


def cluster_sources(sources, previous_close):
    nodes = {}
    for source in sources:
        for level in wave_levels(source['low'], source['high']):
            if level['kind'] == 'REBOUND' or level['price'] > previous_close:
                continue
            nodes.setdefault(level['price'], []).append(dict(source, kind=level['kind'], ratio=level['ratio']))
    groups = []
    for price in sorted(nodes):
        if not groups or price / groups[-1][0] - 1 >= .008:
            groups.append([price])
        else:
            groups[-1].append(price)
    bands = []
    for prices in groups:
        node = min(prices, key=lambda p: abs(p - np.median(prices)))
        origins = [dict(s, level=p) for p in prices for s in nodes[p]]
        pairs = {(s['low_day'], s['high_day']) for s in origins}
        bands.append(dict(node=float(node), lower=float(min(prices)), upper=float(max(prices)),
                          source_waves=len(pairs), peak_groups=len({p[1] for p in pairs}),
                          sources=origins))
    return bands


def observe(bars, a, b, prior_rsi, bands, atr):
    basic = day_observations(bars, a, b, [s['node'] for s in bands], atr, prior_rsi)
    result = basic[SELECT].copy()
    for name in EXTRA:
        result[name] = np.nan
    for i in range(len(bars)):
        if basic.map_known.iloc[i] != 1:
            continue
        prefix = bars.iloc[:i + 1]
        closes, lows = prefix.close.to_numpy(), prefix.low.to_numpy()
        close, low = closes[-1], lows.min()
        broken = [s for s in bands if low < s['lower'] and close < s['lower']]
        nearest = min(broken, key=lambda s: s['lower'] - close) if broken else None
        result.loc[result.index[i], 'support_bands'] = len(bands)
        result.loc[result.index[i], 'broken_bands'] = len(broken)
        waves, peaks, run, depth = 0, 0, 0, 0.
        if nearest:
            waves, peaks = nearest['source_waves'], nearest['peak_groups']
            above = np.flatnonzero(closes >= nearest['lower'])
            run = i - int(above[-1]) if len(above) else i + 1
            depth = (nearest['lower'] - close) / atr
        for name, value in zip(['broken_nearest_wave_count', 'broken_nearest_peak_groups',
                                'broken_close_run', 'broken_close_depth_atr'], [waves, peaks, run, depth]):
            result.loc[result.index[i], name] = value
        repaired = basic.repair_joint.iloc[i]
        if not broken or np.isfinite(repaired):
            no_repair = bool(broken) and repaired == 0
            for name, value in dict(broken_no_repair=no_repair,
                    single_broken_no_repair=no_repair and waves == 1,
                    shared_peak_broken_no_repair=no_repair and waves > 1 and peaks == 1,
                    distinct_peak_broken_no_repair=no_repair and peaks > 1).items():
                result.loc[result.index[i], name] = float(value)
        last_low = int(np.argmin(lows))
        full = [s for s in bands if i >= max(1, last_low + 1) and (lows[:-1] < s['lower']).any()
                and closes[-2] >= s['upper'] and closes[-1] >= s['upper']]
        result.loc[result.index[i], 'full_band_two_close_reclaim'] = float(bool(full))
    return result.astype(float)


def build(native, name, wanted_days):
    f = native.rename(columns={'ts': 'date', 'raw_volume': 'volume'}).sort_values('date').copy()
    if f.date.duplicated().any():
        raise ValueError('DUPLICATE_INDEX_BAR')
    prices = f[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
    if (not np.isfinite(prices).all() or (prices <= 0).any()
            or (f.low > f[['open', 'close']].min(axis=1)).any()
            or (f.high < f[['open', 'close']].max(axis=1)).any()):
        raise ValueError('INVALID_INDEX_OHLC')
    days = sorted(set((f.date // 1000000).astype(int)))
    clock = np.array([d * 1000000 + t for d in days for t in STOPS])
    if not f.date.isin(clock).all():
        raise ValueError('OFF_SESSION_INDEX_BAR')
    grid = f.set_index('date').reindex(clock)
    grid['day'] = grid.index.to_numpy() // 1000000
    daily = grid.groupby('day').agg(high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), count=('close', 'count'))
    daily.loc[daily['count'] != 48, ['high', 'low', 'close']] = np.nan
    last = daily.close.shift(1)
    tr = pd.concat([daily.high - daily.low, (daily.high - last).abs(), (daily.low - last).abs()], axis=1).max(axis=1, skipna=False)
    atr = tr.rolling(14, min_periods=14).mean().shift(1)
    a, b = _rsi(grid.close.to_numpy(), 6), _rsi(grid.close.to_numpy(), 15)
    catalog, output, receipts = {}, [], []
    wanted = set(wanted_days)
    for n, day in enumerate(days):
        if n:
            pday = days[n - 1]
            primary = index_map(f, pday * 1000000 + 150000)
            if primary.get('status') == 'AVAILABLE_EXECUTION_REFERENCE_NOT_PYXIS_FEATURE':
                identity = (primary['low_day'], primary['high_day'])
                catalog.setdefault(identity, {k: primary[k] for k in ['low', 'high', 'low_day', 'high_day', 'anchors_known_at']})
                catalog[identity].setdefault('first_selected_day', pday)
        if day not in wanted:
            continue
        at = np.flatnonzero(grid.day.to_numpy() == day)
        known = [s for s in catalog.values() if s['first_selected_day'] < day and s['anchors_known_at'] < day]
        previous_close = float(daily.close.iloc[n - 1]) if n else np.nan
        bands = cluster_sources(known, previous_close) if np.isfinite(previous_close) else []
        prior = (a[at[0] - 1], b[at[0] - 1]) if at[0] else (np.nan, np.nan)
        result = observe(grid.iloc[at], a[at], b[at], prior, bands, atr.loc[day])
        result.columns = [name + '_break_' + c for c in result]
        result.insert(0, 'decision_ts_assumed_end', clock[at])
        output.append(result)
        receipts.append(dict(day=int(day), previous_day=int(days[n - 1]) if n else None,
                             catalog_sources=len(known), bands=bands,
                             atr_prior=float(atr.loc[day]) if np.isfinite(atr.loc[day]) else None))
    if not output:
        raise ValueError('NO_MARKET_DATES')
    return pd.concat(output, ignore_index=True), receipts
