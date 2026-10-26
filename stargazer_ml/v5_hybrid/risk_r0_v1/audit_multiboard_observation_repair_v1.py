import argparse
import io
import json
import unittest

import numpy as np
import pandas as pd

import repair_multiboard_observation_v1 as repair
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE, OUT = repair.HERE, repair.OUT
ENTRY = HERE / 'multiboard_observation_repair_entry_20261018_v1.json'
RESULT = HERE / 'multiboard_observation_repair_acceptance_20261018_v1.json'


def save(path, record):
    body = repair.prep.encoded(record)
    used = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) if OUT.exists() else 0
    if used + len(body) + repair.RESERVE > repair.CAP or bytes_used() + len(body) + repair.RESERVE > TOTAL_CAP:
        raise ValueError('REPAIR_ACCEPTANCE_BUDGET_STOP')
    with path.open('xb') as stream:
        stream.write(body)


def entry():
    if ENTRY.exists():
        raise ValueError('ENTRY_EXISTS_NO_OVERWRITE')
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromNames(['test_multiboard_observation_repair_v1',
        'test_multiboard_risk_features_v1', 'test_multiboard_context_counts_v1'])
    result = unittest.TextTestRunner(stream=log).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise ValueError('REPAIR_ENTRY_TESTS_FAILED:' + log.getvalue())
    refs = {str(HERE/n): repair.prep.sha(HERE/n) for n in ['repair_multiboard_observation_v1.py',
        'newhigh_common_motion_v1.py', 'test_multiboard_observation_repair_v1.py',
        'audit_multiboard_observation_repair_v1.py']}
    record = dict(status='PASS_OBSERVATION_REPAIR_ENTRY_NOT_FIT', tests=result.testsRun,
                  test_output=log.getvalue(), refs=refs, actual_fit=False, http_requests=0)
    save(ENTRY, record)
    print(json.dumps({k:v for k,v in record.items() if k not in ['test_output', 'refs']}))


def result():
    if RESULT.exists():
        raise ValueError('ACCEPTANCE_EXISTS_NO_OVERWRITE')
    accepted = json.loads(ENTRY.read_bytes())
    for path, expected in accepted['refs'].items():
        if repair.prep.sha(path) != expected:
            raise ValueError('ENTRY_CODE_DRIFT')
    terminal = json.loads((OUT/'stop_record.json').read_bytes())
    manifest = json.loads((OUT/'manifest.json').read_bytes())
    if terminal['status'] != 'COMPLETE_NOT_FIT_OR_DEPLOYMENT' or repair.prep.sha(OUT/'manifest.json') != terminal['manifest_sha256']:
        raise ValueError('REPAIR_NOT_COMPLETE_OR_DRIFTED')
    for name, item in manifest['outputs'].items():
        if repair.prep.sha(OUT/name) != item['sha256']:
            raise ValueError('FROZEN_REPAIR_OUTPUT_DRIFT')
    report = json.loads((OUT/'report.json').read_bytes())
    for path, expected in report['inputs'].items():
        if repair.prep.sha(path) != expected:
            raise ValueError('FROZEN_REPAIR_INPUT_DRIFT:' + path)
    spec = json.loads((OUT/'feature_spec.json').read_bytes())
    attrib = pd.read_parquet(OUT/'coverage_attribution.parquet')
    np.testing.assert_allclose(attrib[list(repair.REASONS)].sum(axis=1), 1., atol=1e-12, rtol=0)
    rows = 0
    for segment in repair.prep.SEGMENTS:
        old = repair.read_segment(segment)
        new = repair.read_context(segment)
        pd.testing.assert_frame_equal(new[['code', repair.TIME]], old[['code', repair.TIME]])
        expected = old[[repair.TIME, 'mb15_member_coverage']].merge(attrib, on=repair.TIME, validate='many_to_one')
        np.testing.assert_allclose(expected.mb15_member_coverage, expected.valid_return, atol=1e-12, rtol=0)
        original = old.copy()
        combined = original.merge(new.drop(columns='segment'), on=['code', repair.TIME], validate='one_to_one', sort=False)
        pd.testing.assert_frame_equal(combined[list(old)], old)
        fields = spec['shared_fields'] + spec['own_fields'] + spec['market_state_fields']
        if np.isinf(combined[fields].to_numpy(dtype=float)).any():
            raise ValueError('INFINITE_NEW_REPAIR_INPUT')
        for frequency in [5, 15]:
            for benchmark in ['sse', 'cyb']:
                high = new[f'nh_{benchmark}{frequency}_close_new_day_high'].to_numpy()
                for scope in ['mb', 'own']:
                    for flag in ['nonpositive', 'falling']:
                        base = f'{scope}{frequency}_{benchmark}_positive_{flag}'
                        r, cov = repair.combine(old[base+'_rate'], old[base+'_coverage'],
                                               old[f'{scope}{frequency}_positive_share'], high)
                        name = f'nh_{scope}{frequency}_{benchmark}_positive_common_{flag}'
                        np.testing.assert_allclose(new[name+'_rate'], r, equal_nan=True, atol=0, rtol=0)
                        np.testing.assert_allclose(new[name+'_coverage'], cov, equal_nan=True, atol=0, rtol=0)
        rows += len(old)
    record = dict(status='PASS_COVERAGE_PROVENANCE_AND_NEW_HIGH_COMMON_MOTION_NOT_FIT',
        tests=accepted['tests'], original_rows=rows, rate_and_coverage_fields=32, market_state_fields=4,
        original_columns_labels_and_model_unchanged=True,
        report_sha256=repair.prep.sha(OUT/'report.json'), manifest_sha256=repair.prep.sha(OUT/'manifest.json'),
        financial_effectiveness_tested=False, fit=False, new_http_requests=0)
    save(RESULT, record)
    print(json.dumps(record))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--entry', action='store_true')
    mode.add_argument('--result', action='store_true')
    args = parser.parse_args()
    entry() if args.entry else result()
