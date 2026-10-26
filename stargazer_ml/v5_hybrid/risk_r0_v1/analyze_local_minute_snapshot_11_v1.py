"""Offline terminal-batch ledger/readback and new four-code warmup diagnostics."""
from datetime import datetime
import json
from pathlib import Path

import pyarrow.parquet as pq

from expansion_first30_v1 import MANIFEST, verify_inputs
from local_minute_snapshot_11_v1 import AUTH, OUT as BATCH, Batch, check, decode_rows, encoded
from minute_warmup_probe_v1 import compare_sources, rebuild, timestamp_profile
from price_verification_guarded_run import sha
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = Path(__file__).resolve().parent
OUT = HERE / 'local_minute_snapshot_11_analysis_20261016_v1'
CAP = 5*1024**2


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    refs = {}

    def track(path):
        path = Path(path).resolve()
        refs[str(path)] = sha(path)
        return path

    for name in ('analyze_local_minute_snapshot_11_v1.py', 'minute_warmup_probe_v1.py',
                 'local_minute_snapshot_11_v1.py', 'minute_risk_features_v1.py', 'research_budget_v2.py'):
        track(HERE / name)
    auth = json.loads(track(AUTH).read_bytes())
    verify_inputs(auth['inputs'])
    opening = json.loads(track(BATCH / 'opening.json').read_bytes())
    stop = json.loads(track(BATCH / 'stop_record.json').read_bytes())
    if sha(AUTH) != opening['authorization_sha256'] or stop['status'] != 'completed':
        raise ValueError('BATCH_AUTH_OR_STATUS')
    if opening['queries'] != [r['query'] for r in stop['results']] or stop['calls_started'] != 11:
        raise ValueError('QUERY_LEDGER_MISMATCH')
    grouped, index, times, received = {}, [], [], 0
    for number, result in enumerate(stop['results'], 1):
        raw = track(BATCH / f'{number:02d}_raw.bin')
        meta = json.loads(track(BATCH / f'{number:02d}_meta.json').read_bytes())
        attempt = json.loads(track(BATCH / f'{number:02d}_attempt.json').read_bytes())
        if meta != result or attempt['query'] != result['query']:
            raise ValueError('METADATA_LEDGER_MISMATCH')
        body = raw.read_bytes()
        if len(body) != result['bytes'] or sha(raw) != result['sha256']:
            raise ValueError('RESPONSE_HASH_OR_SIZE')
        rows = decode_rows(body, result['query'])
        if len(rows) != result['row_count']:
            raise ValueError('ROW_LEDGER_MISMATCH')
        received += len(body)
        times.append(datetime.fromisoformat(attempt['attempted_at']))
        query = result['query']
        if query['scope'] == 'STOCK':
            grouped.setdefault(query['code'], {})[query['date']] = rows
        else:
            index.append(dict(query=query, rows=len(rows), verdict=result['result']))
    if received != stop['response_bytes_received'] or received > 4*1024**2:
        raise ValueError('RESPONSE_BUDGET_OR_LEDGER')
    calendar = json.loads(track(HERE / 'historical_stock_day_preparation_v1/manifest.json').read_bytes())['calendar_dates']
    cases = []
    base = HERE.parent / 'green_common_profile/gcp_round_b_small_sample_v1/minute_ext'
    for code, days in sorted(grouped.items()):
        previous, day = sorted(days)
        if calendar[calendar.index(previous)+1] != day:
            raise ValueError('P_E_RELATION')
        rows = days[previous] + days[day]
        old = pq.read_table(track(base / f'{code}.parquet')).to_pylist()
        cases.append(dict(code=code, P=previous, E=day,
            p_profile=timestamp_profile(rows, previous), e_profile=timestamp_profile(rows, day),
            old_E_comparison=compare_sources(old, rows),
            scenarios=[rebuild(rows, code, previous, day, semantics) for semantics in ('START', 'END')],
            source_snapshot='CURRENT_LOCAL_STOCKDB_READ_NOT_2025_ARRIVAL', live_feature_admitted=False))
    terminal_guards = []
    for action in (check, lambda: Batch(BATCH).execute([])):
        try:
            action()
        except ValueError as error:
            if str(error) not in ('BATCH_EXISTS_NO_RESTART', 'BATCH_TERMINATED'):
                raise
            terminal_guards.append(str(error))
        else:
            raise ValueError('TERMINAL_GUARD_FAILED')
    old_count = 0
    for name in ('first_fit_result_audit_v1.json', 'crossyear_fit_result_audit_v1.json'):
        old_files = json.loads(track(HERE / name).read_bytes())['files']
        verify_inputs(old_files)
        old_count += len(old_files)
    for path in (MANIFEST, HERE / 'candidate_resonance_test_20261016_v1/manifest.json',
                 HERE / 'two_stage_risk_acceptance_20261016_v1/manifest.json',
                 HERE / 'intraday_cache_probe_20261016_v1/manifest.json',
                 HERE / 'minute_warmup_probe_20261016_v1/manifest.json',
                 HERE / 'synchronized_context_inventory_20261016_v1/manifest.json'):
        verify_inputs(json.loads(track(path).read_bytes())['inputs'])
    exe = HERE.parents[2] / 'stargazer_desktop/dist/Stargazer.exe'
    if sha(track(exe)) != '06a2c63e513ff131a6f16d7dd74b78811cef6b2625c0c87b23cbfd235bd738d0':
        raise ValueError('PRODUCTION_EXE_CHANGED')
    batch_bytes = sum(p.stat().st_size for p in BATCH.iterdir())
    report = dict(cases=cases, index_queries=index, requests=11, response_bytes=received,
        batch_bytes_actual=batch_bytes, reserve_not_actual_usage=64*1024,
        minimum_recorded_attempt_gap_seconds=min((b-a).total_seconds() for a,b in zip(times,times[1:])),
        terminal_guards=terminal_guards, frozen_old_fit_files_verified=old_count,
        production_exe_unchanged=True, scale_verified=False, source_clock_verified=False,
        historical_arrival_replayed=False, sector_and_leader_bound=False, fit_ready=False)
    for path, digest in refs.items():
        if sha(Path(path)) != digest:
            raise ValueError('INPUT_CHANGED_DURING_ANALYSIS')
    body = encoded(report)
    import hashlib
    manifest = encoded(dict(inputs=[dict(path=p, sha256=h) for p,h in sorted(refs.items())],
        outputs=[dict(path='analysis.json', bytes=len(body), sha256=hashlib.sha256(body).hexdigest())]))
    total = len(body)+len(manifest)
    if batch_bytes + total + 64*1024 > CAP or bytes_used()+total > TOTAL_CAP:
        raise ValueError('ANALYSIS_WRITE_BUDGET')
    OUT.mkdir(exist_ok=False)
    for name, data in (('analysis.json',body),('manifest.json',manifest)):
        with (OUT/name).open('xb') as stream:
            stream.write(data)
    print(json.dumps(dict(cases=len(cases), requests=11, response_bytes=received,
        batch_actual=batch_bytes, analysis_bytes=total, index_returned=sum(v['rows'] for v in index), fit_ready=False)))


if __name__ == '__main__':
    run()
