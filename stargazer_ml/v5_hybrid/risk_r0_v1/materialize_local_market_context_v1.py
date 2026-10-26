"""Materialize existing native index prices; no requests, labels, or model fitting.

Daily anchor states are joined STRICTLY before each intraday day. No 15m anchors.
Unknown volume/arrival semantics stay null; thresholds and trade actions not invented.
"""
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from golden_anchor_rules_v1 import between_levels
from tdx_index_acceptance_v1 import read_binary, grid
from minute_risk_features_v1 import parse_ts

HERE = Path(__file__).resolve().parent
OUT = HERE/'local_market_context_20261017_v1'
SOURCES = {'000001.SH': Path('D:/a股数据/沪指/sh999999.lc5'),
           '399006.SZ': Path('D:/a股数据/创业板指/sz399006.lc5')}
REPLAY = HERE/'golden_anchor_replay_20261017_v2/replay.json'
RATIOS = ['0.236', '0.382', '0.5', '0.618', '0.786']
CAP = 32*1024**2


def digest(body):
    return hashlib.sha256(body).hexdigest()


def rsi_series(closes):
    previous, changes, gain, loss = None, [], None, None
    result = []
    for close in closes:
        value = None
        if close is None:
            previous, changes, gain, loss = None, [], None, None
        elif previous is not None:
            diff = close-previous
            if gain is None:
                changes.append(diff)
                if len(changes) == 14:
                    gain = sum(max(c, 0) for c in changes)/14
                    loss = sum(max(-c, 0) for c in changes)/14
            else:
                gain, loss = (13*gain+max(diff, 0))/14, (13*loss+max(-diff, 0))/14
            if gain is not None:
                value = 50. if gain == loss == 0 else 100. if loss == 0 else 100.-100./(1.+gain/loss)
        previous = close
        result.append(value)
    return result


def bars_and_days(rows, step):
    if step not in (5, 15):
        raise ValueError('NATIVE5_OR_AGGREGATED15_ONLY')
    grouped = defaultdict(dict)
    for row in rows:
        ts = row['date']
        parse_ts(ts)
        day = ts//1000000
        if ts in grouped[day]:
            raise ValueError('DUPLICATE_SOURCE_TIMESTAMP')
        values = [row[k] for k in ['open', 'high', 'low', 'close']]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('INVALID_SOURCE_PRICE')
        o, h, l, c = values
        if not l <= min(o, c) <= max(o, c) <= h:
            raise ValueError('INVALID_SOURCE_OHLC')
        grouped[day][ts] = row
    bars, daily = [], {}
    for day, records in sorted(grouped.items()):
        allowed = grid(day, 5)
        if set(records)-set(allowed):
            raise ValueError('SOURCE_NOT_NATIVE5_END_GRID')
        if len(records) == 48:
            values = [records[t] for t in allowed]
            daily[day] = dict(close=values[-1]['close'], high=max(r['high'] for r in values),
                              low=min(r['low'] for r in values))
        for slot, end in enumerate(grid(day, step)):
            dt = parse_ts(end)
            times = [int((dt-timedelta(minutes=i)).strftime('%Y%m%d%H%M%S'))
                     for i in reversed(range(0, step, 5))]
            selected = [records[t] for t in times if t in records]
            complete = len(selected) == step//5
            bars.append(dict(day=day, bar_end=end, slot=slot,
                             complete=complete, open=selected[0]['open'] if complete else None,
                             high=max(r['high'] for r in selected) if complete else None,
                             low=min(r['low'] for r in selected) if complete else None,
                             close=selected[-1]['close'] if complete else None))
    return bars, daily


def build(rows, symbol, states, step):
    bars, daily = bars_and_days(rows, step)
    day_keys = sorted(daily)
    daily_rsi = dict(zip(day_keys, rsi_series([daily[d]['close'] for d in day_keys])))
    anchor_days = [s['asof_day'] for s in states]
    if anchor_days != sorted(set(anchor_days)):
        raise ValueError('ANCHOR_DAYS_NOT_UNIQUE_SORTED')
    rsis = rsi_series([b['close'] for b in bars])
    result, current, high, low = [], None, None, None
    for i, (bar, rsi) in enumerate(zip(bars, rsis)):
        day, ts, close = bar['day'], bar['bar_end'], bar['close']
        if day != current:
            current, high, low = day, None, None
        dt = parse_ts(ts)
        morning = dt.hour < 13
        session_start = dt.replace(hour=9 if morning else 13, minute=30 if morning else 0, second=0)
        elapsed = (dt-session_start).total_seconds()/60
        j = bisect_left(day_keys, day)
        prior = day_keys[j-1] if j else None
        reference = daily[prior]['close'] if prior else None
        anchor_pos = bisect_left(anchor_days, day)-1
        anchor = states[anchor_pos] if symbol == '000001.SH' and anchor_pos >= 0 else None
        adjacent = (i > 0 and bar['complete'] and bars[i-1]['complete']
                    and parse_ts(bars[i-1]['bar_end'])+timedelta(minutes=step) == dt)
        change = close/bars[i-1]['close']-1 if adjacent else None
        prior_change = result[-1]['return_adjacent_bar'] if adjacent else None
        ret = close/reference-1 if close is not None and reference else None
        prev_ret = result[-1]['day_return_prev_close'] if adjacent else None
        if close is not None:
            high = max(high, bar['high']) if high is not None else bar['high']
            low = min(low, bar['low']) if low is not None else bar['low']
        point = dict(symbol=symbol, frequency_minutes=step, day=day, decision_ts_assumed_end=ts,
                     source_complete=bar['complete'], close=close,
                     return_adjacent_bar=change,
                     momentum_change=change-prior_change if change is not None and prior_change is not None else None,
                     rsi14=rsi, rsi14_delta1=rsi-rsis[i-1] if adjacent and rsi is not None and rsis[i-1] is not None else None,
                     reference_close_day=prior, reference_close=reference, day_return_prev_close=ret,
                     day_return_change_adjacent=ret-prev_ret if ret is not None and prev_ret is not None else None,
                     drawdown_from_observed_day_high=close/high-1 if close is not None else None,
                     recovery_from_observed_day_low=close/low-1 if close is not None else None,
                     daily_rsi14_asof_prior_day=daily_rsi.get(prior),
                     daily_return5_asof_prior_day=reference/daily[day_keys[j-6]]['close']-1 if j >= 6 else None,
                     daily_return20_asof_prior_day=reference/daily[day_keys[j-21]]['close']-1 if j >= 21 else None,
                     session='AM' if morning else 'PM', session_elapsed_minutes=elapsed,
                     session_fraction=elapsed/120,
                     anchor_asof_day=anchor['asof_day'] if anchor else None,
                     anchor_version=anchor['version'] if anchor else None,
                     anchor_low=anchor['low'] if anchor else None,
                     anchor_high=anchor['high'] if anchor else None,
                     anchor_position=(close-anchor['low'])/(anchor['high']-anchor['low']) if anchor and close is not None else None,
                     volume_feature=None, volume_verified=False, divergence_confirmed=None,
                     actual_available_at=None, live_admitted=False)
        levels = between_levels(anchor['low'], anchor['high'], RATIOS) if anchor else {}
        for ratio in RATIOS:
            suffix = ratio.replace('.', '_')
            point['golden_level_'+suffix] = levels.get(ratio)
            point['golden_distance_'+suffix] = close/levels[ratio]-1 if levels and close is not None else None
            point['time_fraction_distance_'+suffix] = elapsed/120-float(ratio)
        result.append(point)
    return result


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    replay_body = REPLAY.read_bytes()
    replay = json.loads(replay_body)
    manifest_path = REPLAY.parent/'manifest.json'
    old_manifest = json.loads(manifest_path.read_bytes())
    if digest(replay_body) != old_manifest['result_sha256']:
        raise ValueError('ACCEPTED_ANCHOR_OUTPUT_DRIFT')
    refs = {str(REPLAY): digest(replay_body), str(manifest_path): digest(manifest_path.read_bytes())}
    for path in [Path(__file__), HERE/'test_materialize_local_market_context_v1.py',
                 HERE/'tdx_index_acceptance_v1.py', HERE/'golden_anchor_rules_v1.py']:
        refs[str(path)] = digest(path.read_bytes())
    frames, summary = [], []
    for symbol, path in SOURCES.items():
        body = path.read_bytes()
        refs[str(path)] = digest(body)
        if symbol == '000001.SH' and old_manifest['inputs'].get(str(path)) != digest(body):
            raise ValueError('ACCEPTED_SSE_SOURCE_DRIFT')
        rows = read_binary(body)
        dates = sorted({r['date']//1000000 for r in rows})
        for step in [5, 15]:
            derived = build(rows, symbol, replay['states'], step)
            if any(not r['source_complete'] for r in derived):
                raise ValueError('SOURCE_GAP_DURING_MATERIALIZATION')
            # Build-time causality checks, not another data-availability probe.
            indexes = sorted({0, 2, 23 if step == 5 else 7, len(derived)//2, len(derived)-2})
            for index in indexes:
                cutoff = derived[index]['decision_ts_assumed_end']
                prefix = [r for r in rows if r['date'] <= cutoff]
                expected = derived[:index+1]
                if build(prefix, symbol, replay['states'], step)[:index+1] != expected:
                    raise AssertionError('PREFIX_MISMATCH')
                changed = [dict(r, **{k:r[k]*1.5 for k in ['open','high','low','close']})
                           if r['date'] > cutoff else r for r in rows]
                if build(changed, symbol, replay['states'], step)[:index+1] != expected:
                    raise AssertionError('FUTURE_PERTURBATION_MISMATCH')
            frames.extend(derived)
            summary.append(dict(symbol=symbol, frequency_minutes=step, rows=len(derived),
                                days=len(dates), start=dates[0], end=dates[-1],
                                rsi_nonnull=sum(r['rsi14'] is not None for r in derived),
                                anchor_nonnull=sum(r['anchor_asof_day'] is not None for r in derived),
                                prefix_checks=len(indexes), future_checks=len(indexes)))
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(frames), sink, compression='zstd')
    payload = sink.getvalue().to_pybytes()
    report = dict(status='MATERIALIZED_MARKET_COMPONENT_NOT_COMPLETE_RISK', http_requests=0,
                  inputs=refs, rows=len(frames), summary=summary,
                  time_semantics='ASSUMED_END_HISTORICAL_RECONSTRUCTION', actual_available_at=None,
                  golden_policy='SSE_DAILY_ANCHOR_ASOF_STRICTLY_PRIOR_DAY_NOT_15MIN_ANCHORS',
                  ratios=RATIOS, ratios_are_continuous_features_not_entry_thresholds=True,
                  excluded=['Unverified volume/amount rules', 'Unfrozen divergence confirmation',
                            'Trade actions and probability scores'],
                  remaining_components=['THS basket/leader and stock feature join', 'Decision-after labels'],
                  training_ready=False, training_run=False,
                  output=dict(path='market_price_features.parquet', bytes=len(payload), sha256=digest(payload)))
    metadata = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    from research_budget_v2 import TOTAL_CAP, bytes_used
    if len(payload)+len(metadata) > CAP or bytes_used()+len(payload)+len(metadata) > TOTAL_CAP:
        raise ValueError('OUTPUT_OR_REGISTERED_BUDGET_STOP')
    if any(digest(Path(p).read_bytes()) != expected for p, expected in refs.items()):
        raise ValueError('INPUT_CHANGED_DURING_BUILD')
    OUT.mkdir(exist_ok=False)
    for name, content in [('market_price_features.parquet', payload), ('manifest.json', metadata)]:
        with (OUT/name).open('xb') as stream:
            stream.write(content)
    print(json.dumps(dict(rows=len(frames), summary=summary, output_bytes=len(payload)+len(metadata)), ensure_ascii=False))


if __name__ == '__main__':
    run()
