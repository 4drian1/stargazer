"""Pre-fit synthetic entry acceptance, or immutable real-result prediction replay."""
import argparse
import io
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import multiboard_three_arm_fit_v1 as fit
from multiboard_fit_evaluation_v1 import evaluate
from research_budget_v2 import bytes_used, TOTAL_CAP


def save(path, record):
    body = fit.prep.encoded(record)
    existing = sum(p.stat().st_size for p in fit.OUT.rglob('*') if p.is_file()) if fit.OUT.exists() else 0
    if existing + len(body) + fit.RESERVE > fit.CAP or bytes_used() + len(body) + fit.RESERVE > TOTAL_CAP:
        raise ValueError('ACCEPTANCE_BUDGET_STOP')
    with path.open('xb') as target:
        target.write(body)


def entry():
    if fit.ENTRY.exists():
        raise ValueError('ENTRY_ACCEPTANCE_EXISTS_NO_OVERWRITE')
    log = io.StringIO()
    modules = ['test_multiboard_three_arm_fit_v1', 'test_prepare_multiboard_comparison_v1',
               'test_joint_fit_preparation_v1']
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    result = unittest.TextTestRunner(stream=log).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise ValueError('ENTRY_TESTS_FAILED_OR_SKIPPED:' + log.getvalue())
    config, parts, refs = fit.preflight(authorize=False)
    selected_refs = {str(fit.prep.OUT / 'config.json'): fit.CONFIG_SHA}
    for name in ['multiboard_three_arm_fit_v1.py', 'multiboard_fit_evaluation_v1.py',
                 'test_multiboard_three_arm_fit_v1.py', 'audit_multiboard_three_arm_fit_v1.py']:
        selected_refs[str(fit.HERE / name)] = fit.prep.sha(fit.HERE / name)
    if len(parts['train']) != 60712 or len(config['arms']['M1']) != 200:
        raise ValueError('FROZEN_FIT_SCOPE_CHANGED')
    for path, expected in refs.items():
        if fit.prep.sha(path) != expected:
            raise ValueError('ENTRY_INPUT_DRIFT')
    record = dict(status='PASS_FIXED_FIT_ENTRY_NO_REAL_FIT', tests=result.testsRun,
        test_output=log.getvalue(), refs=selected_refs,
        common_rows={s: len(frame) for s, frame in parts.items()},
        synthetic_models_only=True, real_fit_performed=False, http_requests=0)
    save(fit.ENTRY, record)
    print(json.dumps({k: v for k, v in record.items() if k not in ('test_output', 'refs')}))


def result():
    destination = fit.HERE / 'multiboard_three_arm_fit_acceptance_20261018_v1.json'
    if destination.exists():
        raise ValueError('RESULT_ACCEPTANCE_EXISTS_NO_OVERWRITE')
    terminal = json.loads((fit.OUT / 'stop_record.json').read_bytes())
    if terminal['status'] != 'COMPLETE_RESEARCH_NOT_DEPLOYMENT':
        raise ValueError('FIT_NOT_COMPLETE')
    manifest_path = fit.OUT / 'artifact_manifest.json'
    if fit.prep.sha(manifest_path) != terminal['manifest_sha256']:
        raise ValueError('ARTIFACT_MANIFEST_DRIFT')
    manifest = json.loads(manifest_path.read_bytes())
    for name, item in manifest['outputs'].items():
        if fit.prep.sha(fit.OUT / name) != item['sha256'] or (fit.OUT / name).stat().st_size != item['bytes']:
            raise ValueError('FROZEN_FIT_OUTPUT_DRIFT:' + name)
    plan = json.loads((fit.OUT / 'plan.json').read_bytes())
    for path, expected in plan['inputs'].items():
        if fit.prep.sha(path) != expected:
            raise ValueError('FIT_INPUT_OR_CODE_DRIFT:' + path)
    config = plan['config']
    if plan['config_sha256'] != fit.CONFIG_SHA or fit.prep.sha(fit.prep.OUT / 'config.json') != fit.CONFIG_SHA:
        raise ValueError('FROZEN_CONFIG_DRIFT')
    fit.check_authorization(json.loads(fit.AUTH.read_bytes()), fit.CONFIG_SHA, fit.prep.sha(fit.ENTRY))
    masks = pd.read_parquet(fit.prep.OUT / 'population_audit.parquet')
    predictions = pd.read_parquet(fit.OUT / 'predictions.parquet')
    report = json.loads((fit.OUT / 'report.json').read_bytes())
    count, comparisons = 0, 0
    segment_scores = {}
    parts = {}
    for segment in fit.prep.SEGMENTS:
        full = fit.prep.read_segment(segment)
        stored = masks[masks.proposed_segment_name.eq(segment)].reset_index(drop=True)
        common, _ = fit.prep.common_population(full, config['core_observation_groups'])
        np.testing.assert_array_equal(common, stored.common_main_eligible)
        parts[segment] = full.loc[common].reset_index(drop=True)
        actual = predictions[predictions.segment.eq(segment)].reset_index(drop=True)
        cols = ['code', fit.prep.TIME, 'label_end_time', 'y_tail_1200m', 'mae_1200m']
        pd.testing.assert_frame_equal(actual[cols], parts[segment][cols])
        segment_scores[segment] = {}
        count += len(actual)
    for arm, fields in config['arms'].items():
        for algorithm in ['lr', 'lightgbm']:
            name = arm + '_' + algorithm
            preprocessing = json.loads((fit.OUT / (name + '_preprocessing.json')).read_bytes())
            if preprocessing['columns'] != fields:
                raise ValueError('MODEL_COLUMN_ORDER_DRIFT')
            # Recover train-only statistics independently from the selected train view.
            train = parts['train'][fields].to_numpy(dtype=float)
            median = np.nanmedian(train, axis=0)
            filled = np.where(np.isnan(train), median, train)
            scale = filled.std(axis=0)
            for key, expected in [('median', median), ('mean', filled.mean(axis=0)), ('scale', np.where(scale > 0, scale, 1.))]:
                np.testing.assert_allclose(preprocessing[key], expected, atol=1e-12, rtol=0)
            model = json.loads((fit.OUT / (name + '_model.json')).read_bytes()) if algorithm == 'lr' else (fit.OUT / (name + '_model.txt')).read_text(encoding='utf-8')
            calibration = json.loads((fit.OUT / (name + '_calibration.json')).read_bytes())
            restored = fit.replay(parts, fields, algorithm, preprocessing, model, calibration, config['calibration']['epsilon'])
            for segment in fit.prep.SEGMENTS:
                actual = predictions[predictions.segment.eq(segment)]
                for state in ['raw', 'calibrated']:
                    column = name + '_' + state
                    values = restored[segment][state]
                    np.testing.assert_allclose(values, actual[column], atol=1e-11, rtol=0)
                    segment_scores[segment][column] = values
                    comparisons += len(values)
    # Replay the primary paired losses and probability aggregates, not just file hashes.
    for segment, frame in parts.items():
        again = evaluate(frame, segment_scores[segment], report['train_constant_risk'], config)
        for name, values in again['probability_metrics'].items():
            expected = report['metrics'][segment]['probability_metrics'][name]
            for key in ['brier', 'log_loss', 'predicted_risk', 'actual_risk']:
                np.testing.assert_allclose(values[key], expected[key], atol=1e-11, rtol=0)
        for name, values in again['paired_loss_differences'].items():
            expected = report['metrics'][segment]['paired_loss_differences'][name]
            for loss in ['brier', 'log_loss']:
                np.testing.assert_allclose(values[loss]['left_minus_right'], expected[loss]['left_minus_right'], atol=1e-11, rtol=0)
                np.testing.assert_allclose(values[loss]['ci95'], expected[loss]['ci95'], atol=1e-11, rtol=0)
    old_model = fit.HERE / 'postclose_repaired_research_fit_20261017_v1/model.txt'
    old_sha = fit.prep.sha(old_model)
    if old_sha != '5d8402cd530cb83778d901e33eb87788f21ebc8619414f335b6d56905a81e005':
        raise ValueError('D2_MODEL_CHANGED')
    record = dict(status='PASS_FIXED_RESEARCH_FIT_SERIALIZATION_AND_RISK_READBACK',
        models=6, common_rows=count, probability_values_replayed=comparisons,
        report_sha256=fit.prep.sha(fit.OUT / 'report.json'), manifest_sha256=fit.prep.sha(manifest_path),
        plan_sha256=fit.prep.sha(fit.OUT / 'plan.json'), old_d2_sha256=old_sha,
        config_sha256=fit.CONFIG_SHA, promotion_gate_passed=False, deployment_authorized=False,
        independent_blind_test=False, http_requests=0, pnl_evaluated=False)
    save(destination, record)
    print(json.dumps(record))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--entry', action='store_true')
    mode.add_argument('--result', action='store_true')
    args = parser.parse_args()
    entry() if args.entry else result()
