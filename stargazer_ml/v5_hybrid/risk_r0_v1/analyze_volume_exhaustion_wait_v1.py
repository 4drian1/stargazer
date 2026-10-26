"""Append-only offline conditional-wait supplement; preserves first exploratory result."""
import gzip
import json
from pathlib import Path

import numpy as np

from volume_exhaustion_exploration_v1 import HERE, OUT, CAP, digest, reference_path
from research_budget_v2 import bytes_used, TOTAL_CAP


def conditional_arms(x, d, s):
    return {f'shrink_wait_{k}': reference_path(x, s + k + 1)
            if all(x[j, 4] < x[j - 1, 4] and x[j, 4] < .7 * x[d, 4]
                   for j in range(s + 1, s + k + 1)) else None
            for k in (1, 2, 3)}


def paired_block_interval(events, values, calendar, replicates=1000):
    """10 actual-session moving blocks, whole-date cross section retained; exploratory."""
    positions = {date: i for i, date in enumerate(calendar)}
    counts = np.zeros(len(calendar))
    totals = np.zeros(len(calendar))
    for row, value in zip(events, values):
        i = positions[row['signal_date']]
        counts[i] += 1
        totals[i] += value
    rng = np.random.default_rng(20261016)
    offsets = np.arange(10)
    samples = []
    for _ in range(replicates):
        start = rng.integers(0, len(calendar) - 9, size=(len(calendar) + 9) // 10)
        idx = (start[:, None] + offsets).ravel()[:len(calendar)]
        n = counts[idx].sum()
        if n:
            samples.append(totals[idx].sum() / n)
    return [float(v) for v in np.quantile(samples, [.025, .975])] if samples else None


def run():
    target = OUT / 'conditional_wait_supplement.json'
    if target.exists():
        raise ValueError('SUPPLEMENT_EXISTS_NO_OVERWRITE')
    base_path = OUT / 'result.json'
    base = json.loads(base_path.read_bytes())
    for ref in base['inputs']:
        if digest(ref['path']) != ref['sha256']:
            raise ValueError('FROZEN_INPUT_DRIFT')
    events = base['historical']['signal_rows']
    calendar = json.loads((HERE / 'historical_stock_day_preparation_v1/manifest.json').read_bytes())['calendar_dates']
    positions = {d: i for i, d in enumerate(calendar)}
    dense_calendar = [d for d in calendar if 20190101 <= d <= 20251231]
    raw_paths = {Path(r['path']).name[:6]: Path(r['path']) for r in base['inputs']
                 if r['path'].endswith('_daily.json.gz')}
    arrays = {}
    for code in sorted({r['code'] for r in events}):
        records = {int(k.rsplit(':', 1)[1]): row for k, row in
                   json.loads(gzip.decompress(raw_paths[code].read_bytes())) if row is not None}
        arrays[code] = np.array([[records.get(d, {}).get(k, np.nan)
                                 for k in ('open', 'high', 'low', 'close', 'volume')]
                                for d in calendar], dtype=float)
    rows = []
    for event in events:
        x = arrays[event['code']]
        d, s = positions[event['shock_date']], positions[event['signal_date']]
        arms = conditional_arms(x, d, s)
        for k in range(4):
            if reference_path(x, s + k + 1) != event['arms'][f'wait_{k}']:
                raise ValueError('INDEPENDENT_PRICE_PATH_REPLAY_DRIFT')
        rows.append(dict(code=event['code'], signal_date=event['signal_date'], arms=arms))
    summaries = {}
    all_arms = list(events[0]['arms']) + ['shrink_wait_1', 'shrink_wait_2', 'shrink_wait_3']
    baseline = np.array([e['arms']['wait_0']['close_return'] for e in events])
    for arm in all_arms:
        values = [e['arms'].get(arm) if arm in e['arms'] else r['arms'][arm]
                  for e, r in zip(events, rows)]
        active = [a for a in values if a is not None]
        opportunity = np.array([a['close_return'] if a else 0. for a in values])
        delta = opportunity - baseline
        summaries[arm] = dict(signals=len(events), reference_entries=len(active),
            declined_or_unconfirmed=len(events) - len(active),
            opportunity_price_return_mean=float(opportunity.mean()),
            own_reference_return_mean=float(np.mean([a['close_return'] for a in active])) if active else None,
            path_tail_8pct_rate=float(np.mean([a['mae_including_entry_day'] <= -.08 for a in active])) if active else None,
            delta_vs_immediate=float(delta.mean()),
            delta_10session_block_interval=paired_block_interval(events, delta, dense_calendar))
    result = dict(status='EXPLORATORY_CONDITIONAL_WAIT_SUPPLEMENT_NOT_OPTIMIZATION',
        base_sha256=digest(base_path), code_sha256=digest(__file__),
        test_sha256=digest(HERE / 'test_volume_exhaustion_wait_v1.py'),
        input_hashes_verified=len(base['inputs']),
        fixed_wait_reference_paths_replayed=len(events) * 4,
        continuous_shrink='Each additional completed waiting day: V<previous day AND V<0.7*shock V; else no reference entry',
        population='Same base signals; never preselect only future shrink survivors',
        intervals='Exploratory 1000 paired moving blocks of 10 exchange sessions; multiple arms not adjusted; no optimum claim',
        formal_labels_generated=0, fits=0, requests=0, summaries=summaries, rows=rows)
    data = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=1).encode('utf8')
    actual = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
    if actual + len(data) + 32768 > CAP or bytes_used() + len(data) + 32768 > TOTAL_CAP:
        raise ValueError('SUPPLEMENT_BUDGET')
    with target.open('xb') as f:
        f.write(data)
    print(json.dumps(dict(summaries=summaries, bytes=len(data)), ensure_ascii=False))


if __name__ == '__main__':
    run()
