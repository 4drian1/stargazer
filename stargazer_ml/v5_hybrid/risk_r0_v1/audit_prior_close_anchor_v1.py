"""Independent source-key replay of proposed anchors; no population/label writes."""
import io
import json
import unittest

import numpy as np
import pandas as pd

import verify_prior_close_anchor_v1 as run


def audit():
    destination = run.HERE/'prior_close_anchor_acceptance_20261018_v1.json'
    if destination.exists():
        raise ValueError('ACCEPTANCE_EXISTS_NO_OVERWRITE')
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        'test_prior_close_anchor_v1', 'test_multiboard_observation_repair_v1'])
    tests = unittest.TextTestRunner(stream=log).run(suite)
    if not tests.wasSuccessful() or tests.skipped:
        raise ValueError('TESTS_FAILED:' + log.getvalue())
    report = json.loads((run.OUT/'report.json').read_bytes())
    terminal = json.loads((run.OUT/'stop_record.json').read_bytes())
    manifest = json.loads((run.OUT/'manifest.json').read_bytes())
    if terminal['status'] != 'COMPLETE_NOT_FIT' or run.old.prep.sha(run.OUT/'manifest.json') != terminal['manifest_sha256']:
        raise ValueError('INVALID_TERMINAL')
    for name, item in manifest['outputs'].items():
        if run.old.prep.sha(run.OUT/name) != item['sha256']:
            raise ValueError('OUTPUT_DRIFT:' + name)
    for name, expected in report['inputs'].items():
        if run.old.prep.sha(name) != expected:
            raise ValueError('INPUT_DRIFT:' + name)
    clock = pd.read_parquet(run.old.SOURCE/'shared_market_and_leaders.parquet', columns=['cutoff_index', run.TIME])
    clock = clock.sort_values('cutoff_index')
    dates = clock[run.TIME].to_numpy() // 1000000
    if not (np.diff(np.unique(dates)) > 0).all() or not (np.diff(clock[run.TIME]) > 0).all():
        raise ValueError('CLOCK_NOT_STRICTLY_CHRONOLOGICAL')
    patch = pd.read_parquet(run.OUT/'anchor_potential.parquet')
    if patch.duplicated(['code', 'current_session_day']).any() or patch.formal_admitted.any():
        raise ValueError('INVALID_PATCH_KEYS_OR_FORMAL_ADMISSION')
    day_number = pd.Series(np.arange(len(dates)//16), index=dates[::16])
    np.testing.assert_array_equal(patch.prior_cutoff_index,
        patch.current_session_day.map(day_number).to_numpy()*16-1)
    if not (dates[patch.prior_cutoff_index] < patch.current_session_day).all():
        raise ValueError('NONCAUSAL_ANCHOR')
    old_manifest = json.loads((run.old.SOURCE/'manifest.json').read_bytes())
    replayed = 0
    for part in old_manifest['joint_parts']:
        f = pd.read_parquet(run.old.SOURCE/part['path'], columns=['code','cutoff_index','stock_close'])
        p = patch[patch.code.isin(f.code.unique())]
        matched = p.merge(f, left_on=['code','prior_cutoff_index'], right_on=['code','cutoff_index'],
                         how='left', validate='many_to_one')
        np.testing.assert_allclose(matched.reference_close_potential, matched.stock_close, atol=0, rtol=0)
        replayed += len(p)
    if replayed != len(patch) or replayed != report['factorized_code_days']:
        raise ValueError('ANCHOR_REPLAY_COUNT_MISMATCH')
    coverage = pd.read_parquet(run.OUT/'coverage_potential.parquet')
    old_coverage = pd.read_parquet(run.old.OUT/'coverage_attribution.parquet')
    np.testing.assert_array_equal(coverage[run.TIME], old_coverage[run.TIME])
    np.testing.assert_allclose(coverage.old_valid_return_coverage, old_coverage.valid_return, rtol=0, atol=0)
    if not ((coverage.potential_anchor_gain >= 0)
            & (coverage.potential_anchor_gain <= old_coverage.prior_close_unavailable + 1e-12)).all():
        raise ValueError('POTENTIAL_GAIN_INVALID')
    if coverage.formal_added_coverage.any():
        raise ValueError('UNAUTHORIZED_FORMAL_ADMISSION')
    fit_dir = run.HERE/'multiboard_three_arm_fit_20261018_v1'
    fitted = json.loads((fit_dir/'artifact_manifest.json').read_bytes())
    for name, item in fitted['outputs'].items():
        if run.old.prep.sha(fit_dir/name) != item['sha256']:
            raise ValueError('OLD_FIT_CHANGED:' + name)
    d2 = run.HERE/'postclose_repaired_research_fit_20261017_v1/model.txt'
    if run.old.prep.sha(d2) != '5d8402cd530cb83778d901e33eb87788f21ebc8619414f335b6d56905a81e005':
        raise ValueError('D2_MODEL_CHANGED')
    result = dict(status='PASS_STRUCTURAL_ANCHOR_POTENTIAL_NOT_FORMAL_ADMISSION',
        tests=tests.testsRun, test_output=log.getvalue(), independently_replayed_code_days=replayed,
        inputs_verified=len(report['inputs']), old_fit_artifacts_verified=len(fitted['outputs']),
        d2_unchanged=True, http_requests=0, fit=False, formal_admitted_rows=0,
        report_sha256=run.old.prep.sha(run.OUT/'report.json'),
        auditor_sha256=run.old.prep.sha(run.HERE/'audit_prior_close_anchor_v1.py'))
    body = run.old.prep.encoded(result)
    size = sum(p.stat().st_size for p in run.OUT.iterdir() if p.is_file())
    if size + len(body) + run.RESERVE > run.CAP or run.old.bytes_used() + len(body) + run.RESERVE > run.old.TOTAL_CAP:
        raise ValueError('ACCEPTANCE_BUDGET_STOP')
    with destination.open('xb') as stream:
        stream.write(body)
    print(json.dumps({k:v for k,v in result.items() if k != 'test_output'}))


if __name__ == '__main__':
    audit()
