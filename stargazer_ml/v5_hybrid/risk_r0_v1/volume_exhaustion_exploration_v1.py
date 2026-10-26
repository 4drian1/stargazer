"""Offline exploratory screen and price-path comparison; no fetch, fit or trade labels."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from historical_risk_core_v1 import in_scope
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = Path(__file__).resolve().parent
OUT = HERE / 'volume_exhaustion_exploration_20261016_v1'
DAILY = Path('D:/a股数据/data_daily_parts')
CORPUS = HERE / 'historical_archived_corpus_v4_20261015'
FIELDS = ['open', 'high', 'low', 'close', 'volume']
LEVELS = (.236, .382, .5, .618, .786)
CAP = 4 * 1024**2


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid(rows):
    x = np.asarray(rows, dtype=float)
    return (np.isfinite(x).all(axis=1) & (x > 0).all(axis=1)
            & (x[:, 2] <= np.minimum(x[:, 0], x[:, 3]))
            & (np.maximum(x[:, 0], x[:, 3]) <= x[:, 1]))


def signals(x):
    """All indices depend only on completed bars through S. Mean excludes shock day."""
    good = valid(x)
    volume = x[:, 4]
    mean8 = pd.Series(volume).rolling(8).mean().shift(1).to_numpy()
    shock = good & (x[:, 3] < x[:, 0]) & (volume > 1.25 * mean8)
    shock[1:] &= x[1:, 3] < x[:-1, 3]
    shock[0] = False
    shock &= pd.Series(good).rolling(9).sum().eq(9).to_numpy()
    result = []
    for s in range(9, len(x)):
        candidates = [d for d in range(max(8, s - 3), s) if shock[d]]
        if not candidates or not good[s]:
            continue
        d = candidates[-1]
        # First observed shrink after latest shock, not every day of the same decline.
        first = next((j for j in range(d + 1, s + 1)
                      if good[d:j + 1].all() and volume[j] < .7 * volume[d]
                      and volume[j] < volume[j - 1]), None)
        if first == s:
            result.append((d, s))
    return result


def anchor_context(x, d, s):
    if d < 60 or not valid(x[d - 60:s + 1]).all():
        return dict(anchor_status='INSUFFICIENT_VALID_HISTORY')
    a = d - 60 + int(np.argmin(x[d - 60:d, 2]))
    if a + 1 >= d:
        return dict(anchor_status='NO_LATER_HIGH')
    b = a + 1 + int(np.argmax(x[a + 1:d, 1]))
    low, high, close = x[a, 2], x[b, 1], x[s, 3]
    if high <= low:
        return dict(anchor_status='NO_UP_LEG')
    prices = [high - r * (high - low) for r in LEVELS]
    k = min(range(len(prices)), key=lambda j: abs(close / prices[j] - 1))
    return dict(anchor_status='MECHANICAL_TRAILING60_NOT_SWING_CONFIRMATION',
                a_index=int(a), b_index=int(b), a_low=float(low), b_high=float(high),
                preceding_rise=float(high / low - 1), retracement=float((high - close) / (high - low)),
                nearest_retracement_ratio=LEVELS[k], nearest_level=float(prices[k]),
                nearest_distance_pct=float(close / prices[k] - 1),
                rise_over_15pct=bool(high / low - 1 > .15),
                in_broad_236_786_band=bool(.236 <= (high - close) / (high - low) <= .786),
                within_half_percent_of_level=bool(abs(close / prices[k] - 1) <= .005))


def aggregate_prices(frames):
    """Exact duplicates are benign; contradictory keys never silently choose a winner."""
    frame = pd.concat(frames, ignore_index=True)
    dupe = frame.duplicated(['code', 'date'], keep=False)
    candidates = frame.loc[dupe]
    conflicts = []
    for (code, date), group in candidates.groupby(['code', 'date'], sort=True):
        if len(group[FIELDS].drop_duplicates()) > 1:
            conflicts.append(dict(code=str(code), date=int(date), sources=group['source'].tolist()))
    blocked = {(r['code'], r['date']) for r in conflicts}
    frame = frame.drop_duplicates(['code', 'date'], keep='first')
    if blocked:
        frame = frame[[tuple(k) not in blocked for k in frame[['code', 'date']].itertuples(index=False, name=None)]]
    return frame, conflicts


def reference_path(x, entry):
    """Hypothetical next-open reference, NOT certified fill or realized PnL."""
    return dict(entry_index=int(entry), reference_open=float(x[entry, 0]),
                close_return=float(x[entry + 4, 3] / x[entry, 0] - 1),
                mae_including_entry_day=float(x[entry:entry + 5, 2].min() / x[entry, 0] - 1))


def compare_arms(x, s):
    if s + 8 >= len(x) or not valid(x[max(0, s - 4):s + 9]).all():
        return None
    arms = {f'wait_{k}': reference_path(x, s + 1 + k) for k in range(4)}
    confirm = next((j for j in range(s + 1, s + 4)
                    if x[j, 3] > x[j, 0] and x[j, 4] > x[j - 5:j, 4].mean()), None)
    arms['reversal_within_3'] = reference_path(x, confirm + 1) if confirm is not None else None
    return arms


def summarize(events):
    result = {}
    baseline = np.array([r['arms']['wait_0']['close_return'] for r in events])
    for arm in ('wait_0', 'wait_1', 'wait_2', 'wait_3', 'reversal_within_3'):
        active = [r['arms'][arm] for r in events if r['arms'][arm] is not None]
        opportunity = np.array([r['arms'][arm]['close_return'] if r['arms'][arm] else 0. for r in events])
        result[arm] = dict(signals=len(events), reference_entries=len(active),
            no_confirmation=len(events) - len(active),
            opportunity_price_return_mean=float(opportunity.mean()) if len(events) else None,
            delta_vs_immediate=float((opportunity - baseline).mean()) if len(events) else None,
            own_reference_return_mean=float(np.mean([r['close_return'] for r in active])) if active else None,
            own_reference_return_median=float(np.median([r['close_return'] for r in active])) if active else None,
            reference_loss_5pct_rate=float(np.mean([r['close_return'] <= -.05 for r in active])) if active else None,
            path_tail_8pct_rate=float(np.mean([r['mae_including_entry_day'] <= -.08 for r in active])) if active else None)
    return result


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    if bytes_used() + CAP > TOTAL_CAP:
        raise ValueError('REGISTERED_BUDGET')
    inputs = []

    def track(path):
        path = Path(path)
        inputs.append(dict(path=str(path), sha256=digest(path)))
        return path

    frames = []
    for path in sorted(DAILY.glob('part_*.parquet')):
        track(path)
        frame = pd.read_parquet(path, columns=['code', 'date', 'name'] + FIELDS,
                                filters=[('date', '>=', 20260401)])
        frame['code'] = frame['code'].astype(str).str.zfill(6)
        frame['source'] = path.name
        frames.append(frame)
    frame, conflicts = aggregate_prices(frames)
    latest = int(frame['date'].max())
    calendar = sorted(frame['date'].unique().tolist())
    current, scope_count, latest_rows, incomplete, shape_count = [], 0, 0, 0, 0
    for code, group in frame.groupby('code', sort=True):
        if not in_scope(code):
            continue
        scope_count += 1
        group = group.sort_values('date')
        if int(group.iloc[-1]['date']) != latest:
            continue
        latest_rows += 1
        # Do not compress missing exchange sessions into adjacent bars.
        rows = group.set_index('date').reindex(calendar)
        x = rows[FIELDS].to_numpy(dtype=float)
        pairs = [(d, s) for d, s in signals(x) if s == len(x) - 1]
        if len(x) < 69 or not valid(x[-69:]).all():
            incomplete += 1
        for d, s in pairs:
            shape_count += 1
            context = anchor_context(x, d, s)
            current.append(dict(code=code, name=str(group.iloc[-1]['name']),
                signal_date=latest, shock_date=int(calendar[d]),
                shock_volume_ratio_prev8=float(x[d, 4] / x[d - 8:d, 4].mean()),
                signal_volume_to_shock=float(x[s, 4] / x[d, 4]),
                signal_close=float(x[s, 3]), context=context,
                scope='UNVERIFIED_SCALE_DAILY_SHAPE_ONLY_NOT_BUY_SIGNAL'))
    print(f'Latest local daily={latest}; scoped codes={scope_count}; shapes={shape_count}', flush=True)

    # Only existing archived-source research. No local HTTP or new external data.
    stop = json.loads(track(CORPUS / 'stop_record.json').read_bytes())
    old = HERE / 'historical_archived_corpus_v3_20261015'
    old_stop = json.loads(track(old / 'stop_record.json').read_bytes())
    evidence = json.loads(track(CORPUS / 'evidence_and_gate.json').read_bytes())['audit']
    cal = json.loads(track(HERE / 'historical_stock_day_preparation_v1/manifest.json').read_bytes())['calendar_dates']
    events, excluded = [], Counter()
    for code, ev in sorted(evidence.items()):
        daily_path = CORPUS / f'{code}_daily.json.gz'
        ledger = stop['request_ledger']
        if not daily_path.exists():
            daily_path, ledger = old / f'{code}_daily.json.gz', old_stop['request_ledger']
        raw = gzip.decompress(track(daily_path).read_bytes())
        ref = next(r for r in ledger if r['code'] == code and r['table'] == '日k')
        if len(raw) != ref['bytes'] or hashlib.sha256(raw).hexdigest() != ref['sha256']:
            raise ValueError('ARCHIVED_SOURCE_DRIFT')
        records = {}
        for key, row in json.loads(raw):
            date = int(key.rsplit(':', 1)[1])
            if row is None:
                continue
            if row.get('code') != code or row.get('date') != date:
                raise ValueError('ARCHIVED_IDENTITY')
            value = [row.get(k, np.nan) for k in FIELDS]
            if date in records and records[date] != value:
                raise ValueError('ARCHIVED_DUPLICATE_CONFLICT')
            records[date] = value
        x = np.array([records.get(d, [np.nan] * 5) for d in cal], dtype=float)
        good = valid(x)
        pairs = [(d, s) for d, s in signals(x) if 20190101 <= cal[s] <= 20251231]
        excluded['all_shape_signals'] += len(pairs)
        previous_end = -1
        for d, s in pairs:
            if s + 8 >= len(cal):
                excluded['right_censored'] += 1
                continue
            start, end = max(0, d - 60), s + 8
            if d < 60 or not good[start:end + 1].all():
                excluded['price_volume_history_or_future_missing'] += 1
                continue
            if any(i['date'] is None or cal[start] <= i['date'] <= cal[end] for i in ev['issues']):
                excluded['unresolved_source_event'] += 1
                continue
            # No raw volume/price rescaling assumption around known ordinary actions.
            if any(cal[start] <= e['date'] <= cal[end] for e in ev['events']):
                excluded['known_action_in_dependency'] += 1
                continue
            if s <= previous_end:
                excluded['overlapping_same_stock_episode'] += 1
                continue
            arms = compare_arms(x, s)
            if arms is None:
                raise ValueError('INTERNAL_COMMON_POPULATION_ERROR')
            previous_end = end
            events.append(dict(code=code, shock_date=cal[d], signal_date=cal[s],
                dependency_start=cal[start], dependency_end=cal[end], arms=arms,
                context=anchor_context(x, d, s),
                training_eligible=False, label_kind='EXPLORATORY_PRICE_PATH_NOT_EXECUTION_PNL'))
    # Keep all events irrespective of subsequent return. Full-context subset is descriptive only.
    spatial = [r for r in events if r['context'].get('rise_over_15pct')
               and r['context'].get('in_broad_236_786_band')]
    result = dict(status='EXPLORATORY_NOT_TRAINING_OR_TRADE_READY', local_data_date=latest,
        full_market_screen=dict(visible_scope_codes=scope_count, latest_rows=latest_rows,
            last69_sessions_incomplete=incomplete, price_key_conflicts=conflicts,
            signals=current, shape_signals=len(current),
            source_scale='UNVERIFIED', five_layer_resonance_verified=False),
        historical=dict(source_stock_count=len(evidence), start=20190101, end=20251231,
            events=len(events), unique_codes=len({r['code'] for r in events}),
            distinct_signal_dates=len({r['signal_date'] for r in events}),
            counts=dict(excluded), arms=summarize(events),
            by_year={str(y): summarize([r for r in events if r['signal_date'] // 10000 == y])
                     for y in range(2019, 2026)},
            preceding_rise_and_broad_band_subset=dict(events=len(spatial), arms=summarize(spatial)),
            coverage_verified=False, signal_rows=events),
        semantics=dict(entry='completed signal plus wait, NEXT session open reference; NOT fill',
            wait='unconditional 0/1/2/3 sessions, not hindsight survivor filtering',
            confirmation='first bullish close with V>previous5 mean within next3 sessions; NEXT open',
            observation='5 sessions including reference entry; terminal CLOSE not future best price',
            no_confirmation='retained as zero opportunity price return, own return absent',
            price_paths='no fee/liquidity/limit/ST/T+1/stop/portfolio execution certification',
            selection='same source-qualified signals for every arm; 43-stock archive selection bias',
            data_available_at='historical availability assumed, not reconstructed arrival logs',
            claims='no significance, no optimal waiting period, no full five-layer profitability claim'),
        requests=0, fits=0, formal_labels_generated=0, inputs=inputs)
    for name in ('volume_exhaustion_exploration_v1.py', 'test_volume_exhaustion_exploration_v1.py'):
        track(HERE / name)
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=1).encode('utf8')
    if len(payload) + 32768 > CAP or bytes_used() + len(payload) + 32768 > TOTAL_CAP:
        raise ValueError('EXPLORATION_OUTPUT_BUDGET')
    if any(digest(r['path']) != r['sha256'] for r in inputs):
        raise ValueError('INPUT_DRIFT')
    OUT.mkdir(exist_ok=False)
    with (OUT / 'result.json').open('xb') as f:
        f.write(payload)
    print(json.dumps(dict(output=str(OUT), bytes=len(payload), historical_events=len(events),
                         counts=dict(excluded), arms=result['historical']['arms']), ensure_ascii=False))


if __name__ == '__main__':
    run()
