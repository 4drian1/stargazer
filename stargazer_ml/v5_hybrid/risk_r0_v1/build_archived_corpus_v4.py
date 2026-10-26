"""Approved correction batch: 40 local codes, reuse 3, no external HTTP or fitting."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import urllib.parse
import urllib.request

from archived_corpus_replay_v4 import normalize_external, parse_daily
from build_archived_corpus_v3 import frozen_pages
from build_restricted_history_v3 import reconcile_code, SCOPE
from historical_risk_core_v1 import label_at, stock_features, market_features, segment, sample_gate
from historical_stream_v1 import write_stream
from independent_fetch_client import _NoRedirect
from market_index_source_v1 import bind_from_experiment_config
from price_verification_guarded_run import encoded, sha
from research_budget_v2 import bytes_used, TOTAL_CAP
from sina_div_client import parse_bonus_page
from stockdb_adjustment_v2 import rebuild_mult_factors

HERE = Path(__file__).resolve().parent
OLD = HERE / 'historical_archived_corpus_v3_20261015'
OUT = HERE / 'historical_archived_corpus_v4_20261015'
REUSED = ('000001', '000002', '000008')
REQUEST_CAP = 80
RESPONSE_CAP = 128 * 1024**2
DISK_CAP = 64 * 1024**2
RESERVE = 2 * 1024**2


def fetch_local(code, table, opener, ledger, save):
    if table not in ('日k', '复权') or len(code) != 6 or not code.isdigit():
        raise ValueError('LOCAL_QUERY_SCOPE')
    if len(ledger) >= REQUEST_CAP:
        raise ValueError('LOCAL_REQUEST_CAP')
    remaining = RESPONSE_CAP - sum(r.get('bytes', 0) for r in ledger)
    if remaining <= 0:
        raise ValueError('LOCAL_RESPONSE_CAP')
    url = 'http://127.0.0.1:7899/?' + urllib.parse.urlencode(
        dict(cmd='get', t=f'{table}:{code}:*', json=1))
    entry = dict(code=code, table=table, url=url,
                 fetched_at=datetime.now(timezone.utc).isoformat())
    ledger.append(entry)
    cap = min(8 * 1024**2, remaining)
    try:
        with opener.open(url, timeout=10) as response:
            body = response.read(cap)
    except Exception as exc:
        entry['error'] = f'{type(exc).__name__}:{exc}'
        raise
    entry.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    tag = 'daily' if table == '日k' else 'actions'
    save(f'{code}_{tag}.json.gz', gzip.compress(body, mtime=0))
    # Exactly cap bytes cannot distinguish EOF from truncation: conservatively stop.
    if len(body) >= cap:
        raise ValueError('LOCAL_RESPONSE_LIMIT_UNCERTIFIED')
    parsed = json.loads(body)
    if not isinstance(parsed, list):
        raise ValueError('LOCAL_RESPONSE_NOT_RECORD_LIST')
    return parsed


def read_reused(code, table, ledger):
    tag = 'daily' if table == '日k' else 'actions'
    path = OLD / f'{code}_{tag}.json.gz'
    body = gzip.decompress(path.read_bytes())
    entry = next(r for r in ledger if r['code'] == code and r['table'] == table)
    if len(body) != entry['bytes'] or hashlib.sha256(body).hexdigest() != entry['sha256']:
        raise ValueError('REUSED_RESPONSE_HASH')
    return json.loads(body)


def rows_for_code(code, bars, factors, issues, calendar, market_cache, evidence_hash):
    for i, p in enumerate(calendar):
        if not 20190101 <= p <= 20260930:
            continue
        end = calendar[i + 5] if i + 5 < len(calendar) else None
        start = calendar[max(0, i - 60)]
        blocked = (end is None or end > 20261013 or
                   any(x['date'] is None or start <= x['date'] <= end for x in issues))
        # Numeric primitive permission is conditional; persisted completeness stays False.
        label = label_at(bars, calendar, p, factors,
                         scale_verified=not blocked, coverage_verified=not blocked)
        features, fs = stock_features(bars, calendar, p, factors)
        market, ms = market_cache[p]
        features.update(market)
        seg, within = segment(p, label['label_end_date'])
        finite = all(type(v) in (int, float) and math.isfinite(v) for v in features.values())
        eligible = bool(not blocked and within and label['status'] == 'LABEL_COMPLETE' and finite)
        yield dict(code=code, P=p, segment=seg, label_end_date=label['label_end_date'],
                   mae_5d=label['mae_5d'], y_tail_5d=label['y_tail_5d'], features=features,
                   label_status=label['status'], feature_status=fs, market_status=ms,
                   training_eligible=eligible, source_evidence_sha256=evidence_hash,
                   admission_scope=SCOPE, coverage_verified=False)


def run():
    if OUT.exists():
        raise ValueError('BATCH_ALREADY_CLAIMED_NO_RESTART')
    if bytes_used() + DISK_CAP > TOTAL_CAP:
        raise ValueError('REGISTERED_BUDGET')
    pages = frozen_pages()
    remaining = [c for c in pages if c not in REUSED]
    if len(remaining) != 40 or not set(REUSED) <= pages.keys():
        raise ValueError('FROZEN_POPULATION_MISMATCH')
    old_stop = json.loads((OLD / 'stop_record.json').read_bytes())
    if old_stop['status'] != 'failed' or old_stop['local_requests'] != 6:
        raise ValueError('OLD_TERMINAL_MISMATCH')
    # Parse ALL archived external pages and verify reused responses before any new request.
    external = {c: normalize_external(c, parse_bonus_page(pair[0].read_bytes()))
                for c, pair in pages.items()}
    reused = {c: {t: read_reused(c, t, old_stop['request_ledger']) for t in ('日k', '复权')}
              for c in REUSED}
    calendar_path = HERE / 'historical_stock_day_preparation_v1/manifest.json'
    calendar = json.loads(calendar_path.read_bytes())['calendar_dates']
    closes = bind_from_experiment_config()
    market_cache = {p: market_features(closes, calendar, p)
                    for p in calendar if 20190101 <= p <= 20260930}
    overlay_path = HERE / 'contracts/historical_input_overlay_20261015.json'
    index_path = Path(json.loads(overlay_path.read_bytes())['market_source']['bound_input'])
    names = ('build_archived_corpus_v4.py', 'archived_corpus_replay_v4.py',
             'build_archived_corpus_v3.py', 'build_restricted_history_v3.py',
             'ordinary_action_reconciliation_v3.py', 'historical_risk_core_v1.py',
             'historical_stream_v1.py', 'stockdb_adjustment_v2.py',
             'sina_div_client.py', 'independent_fetch_client.py', 'market_index_source_v1.py',
             'research_budget_v2.py', 'price_verification_guarded_run.py',
             'contracts/RESTRICTED_HISTORICAL_RESEARCH_V3.md',
             'contracts/research_storage_4gib_20261015.json',
             'test_archived_corpus_batch_v4.py', 'test_archived_corpus_replay_v4.py')
    inputs = [HERE / n for n in names] + [calendar_path, overlay_path, index_path,
              index_path.with_name(index_path.name + '.binding.json'), OLD / 'stop_record.json']
    inputs.extend(p for pair in pages.values() for p in pair)
    inputs.extend(OLD / f'{c}_{tag}.json.gz' for c in REUSED for tag in ('daily', 'actions'))
    refs = [dict(path=str(p), sha256=sha(p)) for p in inputs]
    plan = dict(schema='ARCHIVED_CORPUS_CORRECTION_V4', codes=list(pages),
                reused_codes=list(REUSED), new_codes=remaining, inputs=refs,
                request_cap=REQUEST_CAP, response_cap=RESPONSE_CAP, disk_cap=DISK_CAP,
                cumulative_cap=TOTAL_CAP, old_failed_requests=6, external_request_cap=0,
                authorization=dict(user_reply='可以，请继续进行把', date='2026-10-15',
                                   scope='40_CODES_LOCAL_READ_AND_MATRIX_PREPARATION_ONLY'),
                scope=SCOPE, fit_authorized=False, coverage_verified=False,
                limitations=['archive availability selection bias', 'survivorship coverage unverified',
                             'vendor event omissions possible', 'historical availability assumed',
                             'not full-market and not a new blind test'])
    used = 0
    OUT.mkdir(exist_ok=False)

    def save(name, data):
        nonlocal used
        if used + len(data) + RESERVE > DISK_CAP or bytes_used() + len(data) + RESERVE > TOTAL_CAP:
            raise ValueError('CORRECTION_DISK_CAP')
        with (OUT / name).open('xb') as f:
            f.write(data)
        used += len(data)

    save('plan.json', encoded(plan))
    evidence_hash = sha(OUT / 'plan.json')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    ledger, audit, cases = [], {}, []
    error, gate, written = None, None, None
    try:
        for code in pages:
            raw = reused[code] if code in reused else {
                t: fetch_local(code, t, opener, ledger, save) for t in ('日k', '复权')}
            bars, missing = parse_daily(code, raw['日k'])
            try:
                events, event_audit, issues = reconcile_code(code, bars, external[code], raw['复权'])
            except ValueError as exc:
                # Conflicting event rows cannot be guessed or silently dropped.
                events, event_audit = [], []
                issues = [dict(date=None, reason='EVENT_RECONCILIATION_REJECTED:' + str(exc))]
            issues = [dict(x, date=None, recorded_date=x['date'])
                      if x['reason'] == 'SINGLE_SOURCE_EVENT' else x for x in issues]
            factors = rebuild_mult_factors(events, calendar)
            cases.append((code, bars, factors, issues))
            audit[code] = dict(events=event_audit, issues=issues, missing_records=missing)
            print(f'{code}: local={len(ledger)}/80, events={len(events)}, issues={len(issues)}', flush=True)
        if any(sha(r['path']) != r['sha256'] for r in refs):
            raise ValueError('FROZEN_INPUT_DRIFT')
        rows = [row for code, bars, factors, issues in cases
                for row in rows_for_code(code, bars, factors, issues, calendar, market_cache, evidence_hash)]
        gate = sample_gate(rows)
        gate.update(scope=SCOPE, fit_authorized=False, full_market_representative=False)
        years, boards, states = {}, {}, Counter()
        for row in rows:
            states['eligible' if row['training_eligible'] else 'isolated_or_incomplete'] += 1
            if not row['training_eligible']:
                continue
            board = ('STAR' if row['code'].startswith('688') else 'GEM' if row['code'].startswith(('300', '301'))
                     else 'SH_MAIN' if row['code'].startswith('6') else 'SZ_MAIN')
            for groups, key in ((years, str(row['P'] // 10000)), (boards, board)):
                count = groups.setdefault(key, Counter())
                count['events'] += 1
                count['positive'] += row['y_tail_5d']
                count['nonpositive'] += 1 - row['y_tail_5d']
        metadata = encoded(dict(audit=audit, gate=gate, states=dict(states), row_count=len(rows),
                                year_class_counts={k: dict(v) for k, v in years.items()},
                                board_class_counts={k: dict(v) for k, v in boards.items()},
                                coverage_verified=False, scope=SCOPE))
        written = write_stream(iter(rows), OUT / 'matrix', bytes_used, total_cap=TOTAL_CAP,
                               new_cap=DISK_CAP - used - len(metadata) - RESERVE, chunk_rows=4096)
        used += sum(p.stat().st_size for p in (OUT / 'matrix').rglob('*') if p.is_file())
        if any(sha(r['path']) != r['sha256'] for r in refs):
            raise ValueError('FROZEN_INPUT_DRIFT')
        save('evidence_and_gate.json', metadata)
    except (Exception, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}:{exc}'
    terminal = dict(status='failed' if error else 'completed', error=error,
                    local_requests=len(ledger), response_bytes=sum(r.get('bytes', 0) for r in ledger),
                    request_ledger=ledger, reused_local_requests=6, external_requests=0,
                    fit_performed=False, scope=SCOPE, coverage_verified=False,
                    sample_gate_passed=bool(not error and gate and gate['training_ready']),
                    training_ready=False, note='P3 sample gate is not FIT authorization',
                    matrix_counts=written['counts'] if written else None)
    data = encoded(terminal)
    actual = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    if len(data) > RESERVE or actual + len(data) > DISK_CAP or bytes_used() + len(data) > TOTAL_CAP:
        raise ValueError('TERMINAL_BUDGET_BREACH')
    with (OUT / 'stop_record.json').open('xb') as f:
        f.write(data)
    print(json.dumps({k: v for k, v in terminal.items() if k != 'request_ledger'}, ensure_ascii=False))
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(run())
