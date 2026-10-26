"""Existing local index bytes -> logical completed-day replay, never live timestamps."""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from golden_anchor_rules_v1 import initialize, close_update, between_levels, time_nodes
from research_budget_v2 import bytes_used, TOTAL_CAP
from tdx_index_acceptance_v1 import read_binary, grid

HERE = Path(__file__).resolve().parent
SOURCE = Path(r'D:\a股数据\沪指\sh999999.lc5')
OUT = HERE / 'golden_anchor_replay_20261017_v1'


def replay(days):
    if len(days) < 7:
        return []
    state = initialize(days[:7])
    states = [state]
    for day in days[7:]:
        state = close_update(state, day)
        states.append(state)
    return states


def run():
    if OUT.exists():
        raise RuntimeError('OUTPUT_EXISTS_NO_OVERWRITE')
    inputs = [SOURCE, HERE / 'golden_anchor_rules_v1.py', HERE / 'test_golden_anchor_rules_v1.py',
              Path(__file__), HERE / 'tdx_index_acceptance_v1.py']
    bindings = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    groups = defaultdict(list)
    for row in read_binary(SOURCE.read_bytes()):
        groups[row['date'] // 1000000].append(row)
    days = []
    for day, rows in sorted(groups.items()):
        if sorted(r['date'] for r in rows) != grid(day, 5):
            raise ValueError(f'INCOMPLETE_OR_DUPLICATE_DAY:{day}')
        for r in rows:
            if (any(not math.isfinite(r[k]) or r[k] <= 0 for k in ('open', 'high', 'low', 'close'))
                    or not r['low'] <= min(r['open'], r['close']) <= max(r['open'], r['close']) <= r['high']):
                raise ValueError('INVALID_OHLC')
        last = max(rows, key=lambda r: r['date'])
        days.append(dict(day=day, high=max(r['high'] for r in rows), low=min(r['low'] for r in rows),
                         close=last['close'], completed=True))
    full = replay(days)
    for index in range(7, len(days) + 1):
        if replay(days[:index]) != full[:index - 6]:
            raise AssertionError('PREFIX_FAIL')
    for index in (6, len(days) // 2, len(days) - 2):
        cutoff = days[index]['day']
        changed = [dict(b, **{k: b[k] * 1.5 for k in ('high', 'low', 'close')})
                   if b['day'] > cutoff else b for b in days]
        if [s for s in replay(changed) if s['asof_day'] <= cutoff] != [s for s in full if s['asof_day'] <= cutoff]:
            raise AssertionError('FUTURE_PERTURB_FAIL')
    seven = [b for b in days if b['day'] <= 20261015][-7:]
    fresh = initialize(seven)
    ratios = ['0.236', '0.382', '0.5', '0.618', '0.786']
    result = dict(status='ENGINEERING_REPLAY_NOT_FROZEN_STRATEGY', training_ready=False,
                  source_clock='ASSUMED_END_NOT_CERTIFIED', actual_known_at=None,
                  day_calendar='OBSERVED_COMPLETE_SOURCE_DAYS_NOT_INDEPENDENT_CALENDAR',
                  day_count=len(days), start=days[0]['day'], end=days[-1]['day'],
                  first_initialization=full[0], latest=full[-1], prefix_checks=len(full), future_perturbations=3,
                  close_updates=sum(s['event'].startswith('UPDATE') for s in full), states=full,
                  original_example=dict(low=3674.82, high=3822.11,
                                        levels=between_levels(3674.82, 3822.11, ratios)),
                  seven_observed_days_ending_20261015=dict(days=[b['day'] for b in seven], anchors=fresh,
                                                          levels=between_levels(fresh['low'], fresh['high'], ratios)),
                  example_mismatch_requires_review=True,
                  unresolved=['initialization date and any local-wave reset policy', 'equality comparison approval',
                              'final ratio set, extension direction, proximity tolerance and scoring weights'],
                  time_nodes_example=time_nodes(20261016, ratios))
    body = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    manifest = json.dumps(dict(inputs=bindings, result_sha256=hashlib.sha256(body).hexdigest()),
                          ensure_ascii=False, indent=2).encode('utf-8')
    if len(body) + len(manifest) > 2 * 1024**2 or bytes_used() + len(body) + len(manifest) > TOTAL_CAP:
        raise RuntimeError('BUDGET_STOP')
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in bindings.items()):
        raise RuntimeError('INPUT_DRIFT')
    OUT.mkdir(exist_ok=False)
    for name, value in [('replay.json', body), ('manifest.json', manifest)]:
        with (OUT / name).open('xb') as f:
            f.write(value)
    print(json.dumps({k: result[k] for k in ('day_count', 'start', 'end', 'prefix_checks', 'future_perturbations',
                                            'close_updates', 'first_initialization', 'latest',
                                            'seven_observed_days_ending_20261015')}, ensure_ascii=False, indent=2))
    print('written_bytes', len(body) + len(manifest))


if __name__ == '__main__':
    run()
