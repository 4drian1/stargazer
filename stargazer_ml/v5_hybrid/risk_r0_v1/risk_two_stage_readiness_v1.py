"""Local-only contract acceptance and measured source readiness; no fetch or fit."""
from collections import Counter
import io
import json
from pathlib import Path
import unittest

import numpy as np
import pyarrow.parquet as pq

from research_budget_v2 import bytes_used, TOTAL_CAP
from two_stage_risk_v1 import evaluate
from test_two_stage_risk_v1 import TwoStageTests
from price_verification_guarded_run import sha

HERE = Path(__file__).resolve().parent
OUT = HERE / 'two_stage_risk_acceptance_20261016_v1'
CAP = 2 * 1024**2


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if bytes_used() + CAP > TOTAL_CAP:
        raise ValueError('REGISTERED_BUDGET')
    refs = {}

    def track(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('SOURCE_HASH_DRIFT')
        refs[str(path)] = digest
        return path

    for name in ('risk_two_stage_readiness_v1.py', 'two_stage_risk_v1.py', 'test_two_stage_risk_v1.py',
                 'R0_TWO_STAGE_RISK_V1.md', 'research_budget_v2.py'):
        track(HERE / name)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TwoStageTests)
    tests = unittest.TextTestRunner(stream=stream).run(suite)
    if not tests.wasSuccessful():
        raise ValueError('TWO_STAGE_TESTS_FAILED:' + stream.getvalue())
    acceptance = json.loads(track(HERE / 'archived_corpus_v4_acceptance_20261015.json').read_bytes())
    rows = []
    for ref in acceptance['matrix_files']:
        rows.extend(pq.read_table(track(ref['path'], ref['sha256']), columns=['code', 'P', 'segment']).to_pylist())
    if len(rows) != acceptance['persisted_rows'] or len({(r['code'], r['P']) for r in rows}) != len(rows):
        raise ValueError('POPULATION_MISMATCH')
    membership_path = HERE.parent / 'opportunity_v2/o5_sector_lag/pit_membership.parquet'
    membership_table = pq.read_table(track(membership_path))
    membership = membership_table.to_pylist()
    by_code = {}
    for r in membership:
        by_code.setdefault(r['symbol'], []).append(r)
    counts, years, segments = Counter(), {}, {}
    for code in sorted({r['code'] for r in rows}):
        group = [r for r in rows if r['code'] == code]
        dates = np.array([r['P'] for r in group])
        matched, l1 = np.zeros(len(group), dtype=int), np.zeros(len(group), dtype=int)
        for r in by_code.get(code, []):
            mask = (dates >= r['effective_from']) & (dates < r['effective_to'])
            matched += mask
            if r.get('index_code_l1'):
                l1 += mask
        for row, n, n_l1 in zip(group, matched, l1):
            key = ('SINGLE_EFFECTIVE_L1_MATCH' if n == 1 and n_l1 == 1 else
                   'AMBIGUOUS_EFFECTIVE_MATCH' if n > 1 else 'MISSING_EFFECTIVE_L1_MATCH')
            counts[key] += 1
            years.setdefault(str(row['P'] // 10000), Counter())[key] += 1
            segments.setdefault(row['segment'], Counter())[key] += 1
    has_known_at = any(k in membership_table.column_names for k in ('known_at', 'available_at'))
    overlay = json.loads(track(HERE / 'contracts/historical_input_overlay_20261015.json').read_bytes())
    track(overlay['market_source']['bound_input'], overlay['market_source']['sha256'])
    binding_path = Path(overlay['market_source']['bound_input'])
    track(binding_path.with_name(binding_path.name + '.binding.json'), overlay['market_source']['binding_sha256'])
    fixture = TwoStageTests()
    plan = fixture.plan()
    policy = dict(max_quote_age_seconds=60, max_bar_age_seconds=60)
    before = evaluate(plan, decision_at='2026-10-16T10:00:00+08:00', **policy)
    touched = evaluate(plan, decision_at='2026-10-16T10:02:02+08:00', quote=fixture.quote(), previous=before, **policy)
    reviewed = evaluate(plan, decision_at='2026-10-16T10:15:05+08:00', observations=fixture.facts(), previous=touched, **policy)
    facts = fixture.facts(); facts['leader']['status'] = 'BLOCK'
    blocked = evaluate(plan, decision_at='2026-10-16T10:15:06+08:00', observations=facts, previous=reviewed, **policy)
    trace = dict(kind='SYNTHETIC_ENGINEERING_ONLY_NOT_REAL_PLAN_OR_QUOTES',
                 policy='60-second limits are test fixtures, NOT approved production thresholds',
                 frozen_plan=plan, states=[before, touched, reviewed, blocked])
    report = dict(status='TWO_STAGE_FLOW_ACCEPTED_REAL_SOURCES_PENDING',
        engineering_tests=dict(run=tests.testsRun, failures=len(tests.failures), errors=len(tests.errors)),
        original_population=len(rows), original_unique_codes=len({r['code'] for r in rows}),
        membership=dict(path=str(membership_path), physical_rows=len(membership),
            source_unique_codes=len(by_code), schema=membership_table.column_names,
            effective_date_mapping=dict(counts), effective_mapping_by_year={k: dict(v) for k, v in years.items()},
            effective_mapping_by_segment={k: dict(v) for k, v in segments.items()},
            has_dedicated_known_at_column=has_known_at,
            status='AVAILABLE_AT_NOT_VERIFIED',
            note='Effective/update dates do not certify historical publication time; no membership used as live-safe.'),
        source_readiness=dict(daily_market_bound_identity=overlay['market_source']['symbol'],
            shanghai_composite_15m='NOT_BOUND_IN_R0_EXPERIMENT',
            historical_sector_known_at='NOT_BOUND', sufficient_sector_member_prices='NOT_BOUND',
            historical_leader_selection='NOT_BOUND', intraday_feature_rule_thresholds='NOT_FROZEN',
            scope_note='Bound-source inventory only; not a claim that no such files exist elsewhere.'),
        implemented='Immutable plan and trigger/review state contract; not actual RSI/sector/leader feature extraction',
        deployment_changed=False, market_requests=0, local_http_requests=0, fits=0,
        real_plan_or_quote_records_written=0, training_ready_for_complete_two_stage_model=False)
    manifest = dict(inputs=[dict(path=p, sha256=h) for p, h in sorted(refs.items())],
                    rule_scope='Offline engineering contract, not source admission or FIT authorization')
    blobs = {name: json.dumps(doc, ensure_ascii=False, allow_nan=False, indent=1).encode('utf8')
             for name, doc in [('source_readiness.json', report), ('synthetic_trace.json', trace), ('manifest.json', manifest)]}
    if sum(map(len, blobs.values())) + 65536 > CAP or bytes_used() + sum(map(len, blobs.values())) + 65536 > TOTAL_CAP:
        raise ValueError('ACCEPTANCE_OUTPUT_BUDGET')
    if any(sha(p) != h for p, h in refs.items()):
        raise ValueError('SOURCE_CHANGED')
    OUT.mkdir(exist_ok=False)
    for name, data in blobs.items():
        with (OUT / name).open('xb') as f:
            f.write(data)
    print(json.dumps(dict(status=report['status'], tests=tests.testsRun, population=len(rows),
        membership_rows=len(membership), effective_mapping=dict(counts),
        dedicated_known_at_column=has_known_at, output_bytes=sum(map(len, blobs.values())),
        synthetic_states=[r['content']['state'] for r in trace['states']]), ensure_ascii=False))


if __name__ == '__main__':
    run()
