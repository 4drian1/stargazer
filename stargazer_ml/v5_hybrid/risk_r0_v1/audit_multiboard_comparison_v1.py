"""Independent preparation readback; no fit, calibration or network calls."""
import io
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import prepare_multiboard_comparison_v1 as prep
from research_budget_v2 import bytes_used, TOTAL_CAP


def run():
    receipt_path = prep.HERE / 'multiboard_comparison_preparation_acceptance_20261018_v1.json'
    if receipt_path.exists():
        raise ValueError('ACCEPTANCE_EXISTS_NO_OVERWRITE')
    modules = ['test_prepare_multiboard_comparison_v1', 'test_joint_fit_preparation_v1',
               'test_multiboard_context_counts_v1', 'test_read_multiboard_full_features_v1',
               'test_multiboard_risk_features_v1', 'test_build_multiboard_full_features_v1']
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    result = unittest.TextTestRunner(stream=stream).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise ValueError('REGRESSION_FAILED_OR_SKIPPED:' + stream.getvalue())
    checked = prep.check()
    if not checked['preparation_ready'] or checked['training_ready'] or checked['fit_authorized']:
        raise ValueError('PREPARATION_GATE_OR_FIT_BOUNDARY')
    config = json.loads((prep.OUT / 'config.json').read_bytes())
    report = json.loads((prep.OUT / 'report.json').read_bytes())
    masks = pd.read_parquet(prep.OUT / 'population_audit.parquet')
    bound_refs = {str(prep.OUT / 'report.json'): prep.sha(prep.OUT / 'report.json'),
                  str(prep.OUT / 'stop_record.json'): prep.sha(prep.OUT / 'stop_record.json'),
                  str(Path(__file__).resolve()): prep.sha(Path(__file__))}
    bound_refs.update({str(prep.OUT / n): item['sha256'] for n, item in report['outputs'].items()})
    for module in modules:
        bound_refs[str(prep.HERE / (module + '.py'))] = prep.sha(prep.HERE / (module + '.py'))
    total, eligible = 0, 0
    boundaries = {row['name']: row for row in config['segments']}
    for segment in prep.SEGMENTS:
        original = prep.read_segment(segment)
        prep.validate_times_labels(original, segment, boundaries)
        stored = masks[masks.proposed_segment_name.eq(segment)].reset_index(drop=True)
        pd.testing.assert_frame_equal(stored[['code', prep.TIME, 'proposed_segment_name']],
                                     original[['code', prep.TIME, 'proposed_segment_name']])
        common, groups = prep.common_population(original, config['core_observation_groups'])
        np.testing.assert_array_equal(common, stored.common_main_eligible)
        reasons = ['|'.join(g for g, m in groups.items() if not m[i]) for i in range(len(original))]
        if reasons != stored.missing_core_groups.tolist():
            raise ValueError('STORED_EXCLUSION_REASONS_NOT_REPRODUCIBLE')
        if prep.summarize(original, common) != report['segments'][segment]['common_main']:
            raise ValueError('POPULATION_OR_CLASS_COUNTS_DRIFT')
        for fields in config['arms'].values():
            view = original.loc[common, fields]
            if len(view) != int(common.sum()) or np.isinf(view.to_numpy(dtype=float)).any():
                raise ValueError('ARM_POPULATION_OR_VALUE_ERROR')
        total += len(original)
        eligible += int(common.sum())
    model = prep.HERE / 'postclose_repaired_research_fit_20261017_v1/model.txt'
    if prep.sha(model) != '5d8402cd530cb83778d901e33eb87788f21ebc8619414f335b6d56905a81e005':
        raise ValueError('D2_MODEL_DRIFT')
    record = dict(status='PASS_COMPARISON_SPEC_AND_POPULATION_NOT_FIT', tests=result.testsRun,
        test_output=stream.getvalue(), original_rows=total, common_main_rows=eligible,
        retained_original_audit_rows=total, primary_excluded_rows=total - eligible,
        configuration_sha256=prep.sha(prep.OUT / 'config.json'), refs=bound_refs,
        original_model_sha256=prep.sha(model), http_requests=0, fit_performed=False,
        real_preprocessing_fitted=False, risk_effectiveness_proven=False,
        statistical_power_proven=False, full_strategy_implementation_certified=False,
        next_gate='SEPARATE_FIXED_RESEARCH_FIT_AUTHORIZATION', proposed_fit_cap=prep.FIT_CAP)
    body = prep.encoded(record)
    used = sum(p.stat().st_size for p in prep.OUT.rglob('*') if p.is_file())
    if used + len(body) > prep.CAP or bytes_used() + len(body) + prep.FIT_CAP > TOTAL_CAP:
        raise ValueError('PREPARATION_OR_FIT_BUDGET_STOP')
    for path, expected in bound_refs.items():
        if prep.sha(path) != expected:
            raise ValueError('INPUT_CHANGED_DURING_ACCEPTANCE')
    with receipt_path.open('xb') as target:
        target.write(body)
    print(json.dumps({k: v for k, v in record.items() if k not in ('refs', 'test_output')}, ensure_ascii=False))


if __name__ == '__main__':
    run()
