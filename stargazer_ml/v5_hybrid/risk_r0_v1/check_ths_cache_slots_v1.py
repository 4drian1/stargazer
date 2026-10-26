"""Stream existing cache windows. No HTTP, source merging, price fill, or fit."""
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from inventory_ths_stockdb_cache_v1 import HERE, ROOTS, digest

INVENTORY = HERE/'ths_stockdb_cache_inventory_20261017_v1.json'
AUTH = HERE/'contracts/ths_snapshot_reconstruction_chat_20261017.json'
OUT = HERE/'ths_cache_slots_20261017_v1.json'
DAYS = [20250102, 20250103, 20250106]
FIELDS = ['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']


def grid(day, semantics):
    if semantics not in ['START', 'END']:
        raise ValueError('EXPLICIT_CLOCK_REQUIRED')
    date = datetime.strptime(str(day), '%Y%m%d')
    offset = 1 if semantics == 'END' else 0
    return {int((date.replace(hour=hour, minute=minute)+timedelta(minutes=i+offset)).strftime('%Y%m%d%H%M%S'))
            for hour, minute in [(9, 30), (13, 0)] for i in range(120)}


def check_day(rows, code, day):
    values, invalid, conflicts = {}, set(), set()
    duplicates, wrong_identity, malformed = 0, 0, 0
    for row in rows:
        ts = row.get('date')
        if type(ts) is not int or len(str(ts)) != 14:
            malformed += 1
            continue
        try:
            stamp = datetime.strptime(str(ts), '%Y%m%d%H%M%S')
        except ValueError:
            malformed += 1
            continue
        if ts//1000000 != day:
            continue
        if row.get('code') != code:
            wrong_identity += 1
            continue
        data = tuple(row.get(f) for f in FIELDS[2:])
        finite = all(type(v) in (int, float) and math.isfinite(v) for v in data)
        if stamp.second or not finite:
            invalid.add(ts)
        else:
            o, h, l, c, v, amount = data
            if not (min(o, h, l, c) > 0 and l <= min(o, c) <= max(o, c) <= h and v >= 0 and amount >= 0):
                invalid.add(ts)
        if ts in values:
            duplicates += 1
            if values[ts] != data:
                conflicts.add(ts)
        else:
            values[ts] = data
    scenarios = {}
    for semantics in ['START', 'END']:
        required = grid(day, semantics)
        missing = required-values.keys()
        blocked = required & (invalid | conflicts)
        scenarios[semantics] = dict(required=240, missing=len(missing), blocked=len(blocked),
                                     extraneous=len(values.keys()-required),
                                     structural_complete=not missing and not blocked and not wrong_identity and not malformed)
    return dict(rows=len(rows), unique_minutes=len(values), conflicting_minutes=len(conflicts),
                invalid_minutes=len(invalid), duplicate_rows=duplicates,
                wrong_identity=wrong_identity, malformed_clock=malformed, scenarios=scenarios,
                clock_verified=False, volume_unit_verified=False, quarantine_applied=False,
                price_scale_verified=False, admitted=False)


def window_digest(rows):
    """Hash exact selected fields; explicitly tag nonfinite values, never silently zero/null."""
    records = []
    for row in rows:
        records.append([{'nonfinite': repr(v)} if type(v) is float and not math.isfinite(v) else v
                        for v in (row.get(f) for f in FIELDS)])
    return digest(json.dumps(records, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode())


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    inv_body, auth_body = INVENTORY.read_bytes(), AUTH.read_bytes()
    inv, auth = json.loads(inv_body), json.loads(auth_body)
    if auth.get('status') != 'USER_CHAT_APPROVED' or not auth.get('allow_archived_member_price_reconstruction'):
        raise ValueError('RECONSTRUCTION_SCOPE_NOT_APPROVED')
    if inv['source_sha256'] != auth['catalog_source_sha256']:
        raise ValueError('APPROVED_CATALOG_MISMATCH')
    source = Path(inv['source'])
    if digest(source.read_bytes()) != inv['source_sha256']:
        raise ValueError('CATALOG_SOURCE_DRIFT')
    boards = [b for b in pq.read_table(source).to_pylist() if b['source'] == 'ths' and b['type'] == 'concept']
    results, errors, candidates = [], [], {s: {d: set() for d in DAYS} for s in ['START', 'END']}
    totals = Counter()
    for number, info in enumerate(inv['files'], 1):
        path = ROOTS[info['root']]/f"{info['code']}.parquet"
        before = path.stat()
        try:
            if before.st_size != info['bytes'] or not set(FIELDS) <= set(info['fields']):
                raise ValueError('SIZE_OR_SCHEMA_MISMATCH')
            # No full source copy; Arrow selects the engineering interval per file.
            rows = pq.read_table(path, columns=FIELDS,
                                 filters=[('date', '>=', DAYS[0]*1000000),
                                          ('date', '<', (DAYS[-1]+1)*1000000)]).to_pylist()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('FILE_CHANGED_DURING_WINDOW_READ')
            per_day = []
            for day in DAYS:
                daily = [r for r in rows if type(r.get('date')) is int and r['date']//1000000 == day]
                checked = dict(day=day, **check_day(daily, info['code'], day))
                per_day.append(checked)
                for semantics in ['START', 'END']:
                    if checked['scenarios'][semantics]['structural_complete']:
                        candidates[semantics][day].add(info['code'])
                        totals[semantics+'_source_days_complete'] += 1
            results.append(dict(root=info['root'], code=info['code'], window_rows=len(rows),
                                window_sha256=window_digest(rows), hash_scope='SELECTED_FIELDS_AND_INTERVAL_ONLY',
                                days=per_day))
        except Exception as exc:
            errors.append(dict(root=info['root'], code=info['code'], error=f'{type(exc).__name__}:{exc}'))
        if number % 250 == 0 or number == len(inv['files']):
            print(json.dumps(dict(scanned=number, total=len(inv['files']), errors=len(errors))), flush=True)
    board_days = []
    for board in boards:
        members = set(board['symbols'])
        for day in DAYS:
            board_days.append(dict(code=board['code'], name=board['name'], day=day, members=len(members),
                                   structural_source_candidates={s: len(members & candidates[s][day]) for s in candidates},
                                   missing_candidates={s: len(members-candidates[s][day]) for s in candidates},
                                   source_selection='UNRESOLVED_NO_MERGING', ranking_admitted=False))
    report = dict(status='STRUCTURAL_ENGINEERING_UPPER_BOUND_NOT_TRAINING_READY', http_requests=0,
                  days=DAYS, dates_are_training_population=False,
                  reconstruction_scope_approved=True, historical_true_membership_verified=False,
                  member_bias_disclosed=True, results=results, errors=errors, totals=dict(totals),
                  candidate_codes_by_day={s: {str(d): sorted(codes) for d, codes in dates.items()}
                                          for s, dates in candidates.items()},
                  boards=board_days, training_ready=False, new_fetch_authorized=False,
                  inputs={str(INVENTORY): digest(inv_body), str(AUTH): digest(auth_body),
                          str(Path(__file__)): digest(Path(__file__).read_bytes()),
                          str(HERE/'test_check_ths_cache_slots_v1.py'):
                          digest((HERE/'test_check_ths_cache_slots_v1.py').read_bytes())})
    body = json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
    from research_budget_v2 import bytes_used, TOTAL_CAP
    if len(body) > 16*1024**2 or bytes_used()+len(body) > TOTAL_CAP:
        raise ValueError('OUTPUT_OR_REGISTERED_BUDGET_STOP')
    with OUT.open('xb') as stream:
        stream.write(body)
    print(json.dumps(dict(output_bytes=len(body), totals=dict(totals), errors=len(errors),
                          candidate_codes={s: {str(d): len(v) for d,v in dates.items()} for s,dates in candidates.items()})))


if __name__ == '__main__':
    run()
