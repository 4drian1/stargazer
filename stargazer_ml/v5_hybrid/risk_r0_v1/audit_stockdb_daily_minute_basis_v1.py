"""Reuse archived 128-query responses only; same-close diagnosis is not scale proof."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import msgpack

HERE = Path(__file__).resolve().parent
BATCH = HERE/'sector_concept_prep/energy_storage_fetch_20261017_v1'
OUT = HERE/'stockdb_daily_minute_basis_20261017_v1.json'
DATES = [20250102, 20250103, 20250106]


def digest(body):
    return hashlib.sha256(body).hexdigest()


def compare_close(daily, minute, code, day):
    selected = []
    for rows, target in [(daily, day), (minute, day*1000000+150000)]:
        matches = [r for r in rows if r.get('date') == target]
        if not matches:
            selected.append(None)
            continue
        if any(r.get('code') != code for r in matches):
            return dict(status='CODE_IDENTITY_CONFLICT')
        prices = [r.get('close') for r in matches]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in prices):
            return dict(status='INVALID_CLOSE')
        if len(set(prices)) != 1:
            return dict(status='DUPLICATE_CLOSE_CONFLICT')
        selected.append(prices[0])
    d, m = selected
    if d is None or m is None:
        return dict(status='MISSING_DAILY_OR_EXACT_1500', daily_close=d, minute_1500_close=m)
    return dict(status='SAME_CLOSE_WITHIN_001' if abs(d-m) <= 0.01+1e-9 else 'CLOSE_MISMATCH',
                daily_close=d, minute_1500_close=m, difference=m-d)


def run():
    if OUT.exists():
        raise RuntimeError('OUTPUT_EXISTS_NO_OVERWRITE')
    terminal = BATCH/'stop_record.json'
    terminal_body = terminal.read_bytes()
    stop = json.loads(terminal_body)
    if stop['status'] != 'COMPLETE' or stop['attempts'] != 128:
        raise ValueError('ORIGINAL_BATCH_NOT_COMPLETE')
    daily, minute, refs = {}, {}, {}
    for entry in stop['entries']:
        path = BATCH/entry['file']
        body = path.read_bytes()
        if len(body) != entry['bytes'] or digest(body) != entry['sha256']:
            raise ValueError('RAW_EVIDENCE_HASH_MISMATCH')
        refs[entry['file']] = digest(body)
        pairs = msgpack.unpackb(body, raw=False, strict_map_key=False)
        if not isinstance(pairs, list):
            raise ValueError('RAW_RESPONSE_NOT_LIST')
        rows = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not isinstance(pair[1], dict):
                raise ValueError('RAW_PAIR_SHAPE')
            if pair[1].get('code') != entry['code']:
                raise ValueError('RAW_CODE_MISMATCH')
            rows.append(pair[1])
        if entry['table'] == '日k':
            if entry['code'] in daily:
                raise ValueError('DUPLICATE_DAILY_QUERY')
            daily[entry['code']] = rows
        elif entry['table'] == '分钟k':
            key = entry['code'], int(entry['date'].rstrip('*'))
            if key in minute:
                raise ValueError('DUPLICATE_MINUTE_QUERY')
            minute[key] = rows
        else:
            raise ValueError('UNEXPECTED_ORIGINAL_QUERY')
    results = [dict(code=code, date=day, **compare_close(daily[code], minute[code, day], code, day))
               for code in sorted(daily) for day in DATES]
    report = dict(status='SAME_CLOSE_DIAGNOSTIC_ONLY', http_requests=0, tolerance_yuan=0.01,
                  code_count=len(daily), stock_days=len(results), dates=DATES,
                  counts=dict(Counter(r['status'] for r in results)), results=results,
                  price_scale_verified=False, corporate_actions_verified=False,
                  previous_close_baseline_verified=False, training_admitted=False,
                  warning='Same-day closing-price agreement does not prove cross-day scale or prior-session reference',
                  batch_terminal_sha256=digest(terminal_body), raw_sha256=refs,
                  code_sha256=digest(Path(__file__).read_bytes()),
                  test_sha256=digest((HERE/'test_audit_stockdb_daily_minute_basis_v1.py').read_bytes()))
    body = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    from research_budget_v2 import TOTAL_CAP, bytes_used
    if len(body) > 256*1024 or bytes_used()+len(body) > TOTAL_CAP:
        raise RuntimeError('OUTPUT_OR_REGISTERED_BUDGET_STOP')
    with OUT.open('xb') as stream:
        stream.write(body)
    print(json.dumps(dict(stock_days=len(results), counts=report['counts'], output_bytes=len(body))))


if __name__ == '__main__':
    run()
