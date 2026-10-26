"""Offline repair/replay only. Never reopens the terminal v3 fetch batch."""
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path

from build_archived_corpus_v3 import frozen_pages, normalize_external as normalize_v3
from build_restricted_history_v3 import reconcile_code
from sina_div_client import parse_bonus_page
from stockdb_adjustment_v2 import valid_date
from research_budget_v2 import bytes_used, TOTAL_CAP
from price_verification_guarded_run import encoded, sha

HERE = Path(__file__).resolve().parent
BATCH = HERE / 'historical_archived_corpus_v3_20261015'
OUTPUT = HERE / 'archived_corpus_offline_replay_v4_20261015.json'


def normalize_external(code, parsed):
    records = normalize_v3(code, parsed)
    bonus = (r for r in parsed['bonus'] if r.get('进度') == '实施')
    for record, original in zip((r for r in records if r['table'] == 'bonus'), bonus):
        value = original.get('派息(税前)(元)')
        if record['per_share_cash'] is not None:
            record['per_share_cash'] = float(Decimal(str(value).replace(',', '')) / Decimal(10))
    return records


def parse_daily(code, records):
    bars, missing = {}, []
    seen = {}
    for key, row in records:
        if not isinstance(key, str) or not key.startswith(f'日k:{code}:'):
            raise ValueError('DAILY_CODE_MISMATCH')
        date = valid_date(int(key.rsplit(':', 1)[1]))
        if row is None:
            record = None
        else:
            if not isinstance(row, dict):
                raise ValueError('INVALID_DAILY_RECORD')
            if row.get('code') != code or valid_date(row.get('date')) != date:
                raise ValueError('DAILY_IDENTITY_MISMATCH')
            record = {k: row.get(k) for k in ('open', 'high', 'low', 'close')}
        if date in seen and seen[date] != record:
            raise ValueError('DUPLICATE_PRICE_CONFLICT')
        seen[date] = record
    for date, record in sorted(seen.items()):
        if record is None:
            missing.append(dict(date=date, reason='SOURCE_NULL_RECORD'))
        else:
            bars[date] = record
    return bars, missing


def run():
    if OUTPUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    stop_path = BATCH / 'stop_record.json'
    stop = json.loads(stop_path.read_bytes())
    if stop['status'] != 'failed':
        raise ValueError('EXPECTED_TERMINATED_FAILED_BATCH')
    pages = frozen_pages()
    ledger = {(r['code'], r['table']): r for r in stop['request_ledger']}
    codes = sorted(c for c in pages if (c, '日k') in ledger and (c, '复权') in ledger)
    refs = [dict(path=str(stop_path), sha256=sha(stop_path)),
            dict(path=str(Path(__file__)), sha256=sha(Path(__file__))),
            dict(path=str(HERE / 'build_archived_corpus_v3.py'),
                 sha256=sha(HERE / 'build_archived_corpus_v3.py')),
            dict(path=str(HERE / 'build_restricted_history_v3.py'),
                 sha256=sha(HERE / 'build_restricted_history_v3.py')),
            dict(path=str(HERE / 'ordinary_action_reconciliation_v3.py'),
                 sha256=sha(HERE / 'ordinary_action_reconciliation_v3.py'))]
    results = {}
    for code in codes:
        raw = {}
        for table, tag in (('日k', 'daily'), ('复权', 'actions')):
            path = BATCH / f'{code}_{tag}.json.gz'
            body = gzip.decompress(path.read_bytes())
            entry = ledger[code, table]
            if len(body) != entry['bytes'] or hashlib.sha256(body).hexdigest() != entry['sha256']:
                raise ValueError('ARCHIVED_RESPONSE_DRIFT')
            refs.append(dict(path=str(path), sha256=sha(path)))
            raw[tag] = json.loads(body)
        blob, sidecar = pages[code]
        refs.extend(dict(path=str(p), sha256=sha(p)) for p in (blob, sidecar))
        bars, missing = parse_daily(code, raw['daily'])
        external = normalize_external(code, parse_bonus_page(blob.read_bytes()))
        events, audit, issues = reconcile_code(code, bars, external, raw['actions'])
        results[code] = dict(bars=len(bars), missing_records=missing,
                             corroborated_events=len(events), audit=audit, issues=issues)
    report = dict(status='OFFLINE_REPLAY_COMPLETED', local_requests=0, external_requests=0,
                  fit_performed=False, training_ready=False, coverage_verified=False,
                  source_batch_status=stop['status'], source_batch_requests=stop['local_requests'],
                  inputs=refs, results=results,
                  remaining_codes=[c for c in pages if c not in codes],
                  note='Offline parsing/reconciliation evidence, not a training matrix or completeness proof.')
    data = encoded(report)
    if len(data) > 1024**2 or bytes_used() + len(data) > TOTAL_CAP:
        raise ValueError('REPLAY_REPORT_BUDGET')
    if any(sha(r['path']) != r['sha256'] for r in refs):
        raise ValueError('INPUT_DRIFT')
    with OUTPUT.open('xb') as f:
        f.write(data)
    print(json.dumps({c: dict(events=r['corroborated_events'], issues=r['issues'],
                             missing=r['missing_records']) for c, r in results.items()}, ensure_ascii=False))


if __name__ == '__main__':
    run()
