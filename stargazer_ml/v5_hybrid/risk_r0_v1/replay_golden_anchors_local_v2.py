"""Read existing index bytes; asymmetric anchors only, no requests or training."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from golden_anchor_rules_v1 import between_levels
from golden_anchor_rules_v2 import replay
from research_budget_v2 import bytes_used, TOTAL_CAP
from tdx_index_acceptance_v1 import read_binary, grid

HERE = Path(__file__).resolve().parent
SOURCE = Path(r'D:\a股数据\沪指\sh999999.lc5')
OUT = HERE / 'golden_anchor_replay_20261017_v2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    if OUT.exists():
        raise RuntimeError('OUTPUT_EXISTS_NO_OVERWRITE')
    old = HERE / 'golden_anchor_replay_20261017_v1'
    protected = [old / 'replay.json', old / 'manifest.json', HERE / 'golden_anchor_rules_v1.py',
                 HERE / 'sector_concept_prep/energy_storage_fetch_20261017_v1/stop_record.json']
    old_manifest = json.loads((old / 'manifest.json').read_bytes())
    if digest(old / 'replay.json') != old_manifest['result_sha256'] or any(
            digest(Path(p)) != h for p, h in old_manifest['inputs'].items()):
        raise ValueError('OLD_EVIDENCE_DRIFT')
    inputs = [SOURCE, HERE / 'golden_anchor_rules_v1.py', HERE / 'golden_anchor_rules_v2.py',
              HERE / 'test_golden_anchor_rules_v2.py', HERE / 'tdx_index_acceptance_v1.py', Path(__file__)]
    hashes = {str(p): digest(p) for p in inputs}
    protected_hashes = {str(p): digest(p) for p in protected}
    grouped = defaultdict(list)
    for row in read_binary(SOURCE.read_bytes()):
        grouped[row['date'] // 1000000].append(row)
    days = []
    for day, rows in sorted(grouped.items()):
        if sorted(r['date'] for r in rows) != grid(day, 5):
            raise ValueError('INCOMPLETE_DAY_OR_DUPLICATE')
        for r in rows:
            if any(not math.isfinite(r[k]) or r[k] <= 0 for k in ('open', 'high', 'low', 'close')):
                raise ValueError('INVALID_PRICE')
            if not r['low'] <= min(r['open'], r['close']) <= max(r['open'], r['close']) <= r['high']:
                raise ValueError('INVALID_OHLC')
        rows.sort(key=lambda r: r['date'])
        days.append(dict(day=day, open=rows[0]['open'], close=rows[-1]['close'],
                         high=max(r['high'] for r in rows), low=min(r['low'] for r in rows), completed=True))
    states = replay(days)
    for end in range(7, len(days) + 1):
        if replay(days[:end]) != states[:end - 6]:
            raise AssertionError('PREFIX_FAIL')
    for index in (6, len(days) // 2, len(days) - 2):
        cutoff = days[index]['day']
        changed = [dict(b, **{k: b[k] * 1.5 for k in ('open', 'high', 'low', 'close')})
                   if b['day'] > cutoff else b for b in days]
        if [s for s in replay(changed) if s['asof_day'] <= cutoff] != [s for s in states if s['asof_day'] <= cutoff]:
            raise AssertionError('FUTURE_PERTURB_FAIL')
    result = dict(status='ENGINEERING_REPLAY_NOT_FINANCIAL_VALIDATION', training_ready=False,
                  day_count=len(days), start=days[0]['day'], end=days[-1]['day'],
                  source_clock='ASSUMED_END_NOT_CERTIFIED', actual_known_at=None,
                  calendar='OBSERVED_SOURCE_DAYS_NOT_INDEPENDENT_EXCHANGE_CALENDAR',
                  equality_policy='STRICT_ENGINEERING_GT_LT_EQUALS_RESET_RUN',
                  event_counts=dict(Counter(event for s in states for event in s['events'])),
                  watch_up_days=sum(s['watch_up'] for s in states),
                  prefix_checks=len(states), future_perturbations=3, latest=states[-1], states=states,
                  user_example_period=[s for s in states if 20260921 <= s['asof_day'] <= 20261016],
                  latest_levels=between_levels(states[-1]['low'], states[-1]['high'], ['0.236', '0.382', '0.5', '0.618', '0.786']),
                  original_manual_example=dict(low=3674.82, high=3822.11),
                  policy_notes=['one contraction per direction run', 'equal/reversal restarts the run',
                                'down candidate starts at a bullish-body day close',
                                'minimum three decline days; wait longer if magnitude not yet met',
                                'expand first, then legal opposite-end contraction; one version per close'])
    body = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    manifest = json.dumps(dict(inputs=hashes, protected=protected_hashes, result_sha256=hashlib.sha256(body).hexdigest()),
                          ensure_ascii=False, indent=2).encode('utf-8')
    if len(body) + len(manifest) > 2 * 1024**2 or bytes_used() + len(body) + len(manifest) > TOTAL_CAP:
        raise RuntimeError('BUDGET_STOP')
    if any(digest(Path(p)) != h for p, h in {**hashes, **protected_hashes}.items()):
        raise ValueError('INPUT_OR_PROTECTED_DRIFT')
    OUT.mkdir(exist_ok=False)
    for name, payload in [('replay.json', body), ('manifest.json', manifest)]:
        with (OUT / name).open('xb') as f:
            f.write(payload)
    print(json.dumps({k: result[k] for k in ('day_count', 'prefix_checks', 'future_perturbations',
                                            'event_counts', 'watch_up_days', 'latest', 'latest_levels')},
                     ensure_ascii=False, indent=2))
    print('written_bytes', len(body) + len(manifest))


if __name__ == '__main__':
    run()
