"""Read-only persisted-matrix audit; --report writes one new immutable audit JSON."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from archived_corpus_replay_v4 import parse_daily
from build_archived_corpus_v4 import OUT, OLD, REUSED
from build_restricted_history_v3 import SCOPE
from historical_risk_core_v1 import (STOCK_FIELDS, MARKET_FIELDS, sample_gate, segment,
                                    label_at, stock_features, market_features)
from market_index_source_v1 import bind_from_experiment_config
from price_verification_guarded_run import sha, encoded
from research_budget_v2 import bytes_used, TOTAL_CAP
from stockdb_adjustment_v2 import rebuild_mult_factors

HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise ValueError(message)


def check():
    plan = json.loads((OUT / 'plan.json').read_bytes())
    terminal = json.loads((OUT / 'stop_record.json').read_bytes())
    evidence = json.loads((OUT / 'evidence_and_gate.json').read_bytes())
    matrix = json.loads((OUT / 'matrix/stop_record.json').read_bytes())
    require(terminal['status'] == matrix['status'] == 'completed', 'BATCH_NOT_COMPLETE')
    for ref in plan['inputs']:
        require(sha(ref['path']) == ref['sha256'], 'FROZEN_INPUT_DRIFT:' + ref['path'])
    require(terminal['local_requests'] == len(terminal['request_ledger']) == 80, 'REQUEST_COUNT')
    require(terminal['response_bytes'] == sum(r['bytes'] for r in terminal['request_ledger'])
            <= plan['response_cap'], 'RESPONSE_ACCOUNTING')
    expected = {(c, t) for c in plan['new_codes'] for t in ('日k', '复权')}
    require({(r['code'], r['table']) for r in terminal['request_ledger']} == expected, 'REQUEST_SCOPE')
    old_terminal = json.loads((OLD / 'stop_record.json').read_bytes())
    for root, ledger in ((OUT, terminal['request_ledger']), (OLD, old_terminal['request_ledger'])):
        for entry in ledger:
            tag = 'daily' if entry['table'] == '日k' else 'actions'
            body = gzip.decompress((root / f"{entry['code']}_{tag}.json.gz").read_bytes())
            require(len(body) == entry['bytes'] and hashlib.sha256(body).hexdigest() == entry['sha256'],
                    'ARCHIVED_RESPONSE_DRIFT')
    rows, files = [], []
    for part in matrix['parts']:
        path = OUT / 'matrix' / part['name']
        require(path.stat().st_size == part['bytes'] and sha(path) == part['sha256'], 'PARQUET_DRIFT')
        rows.extend(pq.read_table(path).to_pylist())
        files.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
    require(len(rows) == evidence['row_count'] == matrix['counts']['rows_seen'], 'PERSISTED_ROW_COUNT')
    keys = {(r['code'], r['P']) for r in rows}
    require(len(keys) == len(rows), 'DUPLICATE_STOCK_DAY')
    require({r['code'] for r in rows} == set(plan['codes']), 'POPULATION_LOSS')
    calendar = json.loads((HERE / 'historical_stock_day_preparation_v1/manifest.json').read_bytes())['calendar_dates']
    index = {d: i for i, d in enumerate(calendar)}
    samples = {}
    counts, labels = Counter(), Counter()
    year_counts, board_counts = {}, {}
    for row in rows:
        require(row['source_evidence_sha256'] == sha(OUT / 'plan.json'), 'ROW_EVIDENCE_REFERENCE')
        require(row['admission_scope'] == SCOPE and row['coverage_verified'] is False, 'SCOPE_DRIFT')
        i = index[row['P']]
        window = calendar[i + 1:i + 6]
        require(len(window) == 5 and row['label_end_date'] == window[-1], 'LABEL_WINDOW_DRIFT')
        seg, within = segment(row['P'], row['label_end_date'])
        require(row['segment'] == seg, 'SEGMENT_DRIFT')
        if row['y_tail_5d'] is None:
            require(row['mae_5d'] is None and not row['training_eligible'], 'NULL_LABEL_MISUSE')
        else:
            require(row['label_status'] == 'LABEL_COMPLETE' and math.isfinite(row['mae_5d']),
                    'LABEL_STATUS_INVALID')
        if row['training_eligible']:
            require(within and row['y_tail_5d'] in (0, 1), 'ADMISSION_INVALID')
            require(all(type(row['features'][f]) in (int, float) and math.isfinite(row['features'][f])
                        for f in STOCK_FIELDS + MARKET_FIELDS), 'ADMITTED_FEATURE_INVALID')
            counts['admitted'] += 1
            labels['positive' if row['y_tail_5d'] else 'nonpositive'] += 1
            name = (row['code'], seg)
            samples.setdefault(name, row)  # Fixed earliest eligible P per stock and segment.
            code = row['code']
            board = ('STAR' if code.startswith('688') else 'GEM' if code.startswith(('300', '301'))
                     else 'SH_MAIN' if code.startswith('6') else 'SZ_MAIN')
            for groups, key in ((year_counts, str(row['P'] // 10000)), (board_counts, board)):
                c = groups.setdefault(key, Counter())
                c['events'] += 1
                c['positive'] += row['y_tail_5d']
                c['nonpositive'] += 1 - row['y_tail_5d']
    require(counts['admitted'] == matrix['counts']['admitted'], 'PERSISTED_ADMISSION_COUNT')
    require(all(labels[k] == matrix['counts'][k] for k in ('positive', 'nonpositive')), 'CLASS_COUNT')
    require({k: dict(v) for k, v in year_counts.items()} == evidence['year_class_counts'], 'YEAR_COUNT')
    require({k: dict(v) for k, v in board_counts.items()} == evidence['board_class_counts'], 'BOARD_COUNT')
    gate = sample_gate(rows)
    require(gate['segments'] == evidence['gate']['segments'], 'GATE_RECOMPUTE_DIFFERENCE')
    closes = bind_from_experiment_config()
    replay_count = 0
    for code in plan['codes']:
        root = OLD if code in REUSED else OUT
        raw = json.loads(gzip.decompress((root / f'{code}_daily.json.gz').read_bytes()))
        bars, _ = parse_daily(code, raw)
        events = [dict(date=e['date'], mult=e['multiplier']) for e in evidence['audit'][code]['events']
                  if e.get('multiplier') is not None]
        factors = rebuild_mult_factors(events, calendar)
        issues = evidence['audit'][code]['issues']
        for row in (v for (c, _), v in samples.items() if c == code):
            p = row['P']
            start, end = calendar[max(0, index[p] - 60)], row['label_end_date']
            require(not any(x['date'] is None or start <= x['date'] <= end for x in issues),
                    'QUARANTINED_WINDOW_ADMITTED')
            label = label_at(bars, calendar, p, factors, scale_verified=True, coverage_verified=True)
            require(label['mae_5d'] == row['mae_5d'] and label['y_tail_5d'] == row['y_tail_5d'],
                    'RAW_LABEL_REPLAY_DIFFERENCE')
            full, _ = stock_features(bars, calendar, p, factors)
            prefix, _ = stock_features({d: b for d, b in bars.items() if d <= p}, calendar, p,
                                       {d: f for d, f in factors.items() if d <= p})
            changed, _ = stock_features({d: b if d <= p else {k: v * 1.5 if v is not None else None
                                         for k, v in b.items()} for d, b in bars.items()}, calendar, p, factors)
            market, _ = market_features(closes, calendar, p)
            market_prefix, _ = market_features({d: v for d, v in closes.items() if d <= p}, calendar, p)
            market_changed, _ = market_features({d: v if d <= p else v * 2 for d, v in closes.items()}, calendar, p)
            require(full == prefix == changed, 'STOCK_PIT_FAILURE')
            require(market == market_prefix == market_changed, 'MARKET_PIT_FAILURE')
            require(dict(full, **market) == row['features'], 'FEATURE_REPLAY_DIFFERENCE')
            replay_count += 1
    legacy = json.loads((HERE / 'real_subset_audit_v2_20261015.json').read_bytes())
    require(all(sha(r['path']) == r['sha256'] for r in legacy['preserved_artifacts']), 'LEGACY_MUTATION')
    actual = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    require(actual <= plan['disk_cap'] and bytes_used() <= TOTAL_CAP, 'DISK_BUDGET')
    return dict(status='PERSISTED_MATRIX_AUDIT_PASS', data_preparation_ready=gate['training_ready'],
                training_ready=False, fit_authorized=False, coverage_verified=False,
                scope=SCOPE, persisted_rows=len(rows), eligible=counts['admitted'],
                isolated_or_incomplete=len(rows)-counts['admitted'], gate=gate,
                raw_label_and_pit_replay_cases=replay_count,
                legacy_artifacts_unchanged=len(legacy['preserved_artifacts']),
                local_requests=80, reused_prior_requests=6, external_requests=0,
                response_bytes=terminal['response_bytes'], batch_actual_bytes=actual,
                registered_actual_bytes=bytes_used(), cumulative_cap=TOTAL_CAP,
                matrix_files=files, year_counts={k: dict(v) for k, v in year_counts.items()},
                board_counts={k: dict(v) for k, v in board_counts.items()},
                audit_code=dict(path=str(Path(__file__)), sha256=sha(Path(__file__))),
                controls=[dict(path=str(p), sha256=sha(p)) for p in
                          (OUT/'plan.json', OUT/'stop_record.json', OUT/'evidence_and_gate.json',
                           OUT/'matrix/stop_record.json')],
                limitations=plan['limitations'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    report = check()
    if args.report:
        data = encoded(report)
        require(len(data) <= 1024**2 and bytes_used() + len(data) <= TOTAL_CAP, 'AUDIT_REPORT_BUDGET')
        with (HERE / 'archived_corpus_v4_acceptance_20261015.json').open('xb') as f:
            f.write(data)
    print(json.dumps({k: v for k, v in report.items() if k not in ('matrix_files', 'controls')},
                     ensure_ascii=False, indent=2))
