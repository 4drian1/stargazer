"""Saved models/predictions/metrics readback. No FIT, raw-minute replay or network."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import common_candidate_fixed_fit_v1 as fit
from research_budget_common_v3 import EvidenceWriter

HERE = Path(__file__).resolve().parent
OUT = HERE / 'common_candidate_fit_acceptance_20261019_v1'


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    fit.verify_manifest(fit.OUT, 'COMPLETE_NOT_DEPLOYED')
    fit.verify_manifest(fit.ENTRY, 'COMPLETE_ENTRY_NOT_FIT')
    plan = json.loads((fit.OUT / 'plan.json').read_bytes())
    for path, expected in plan['inputs'].items():
        if fit.sha(path) != expected:
            raise ValueError('FROZEN_FIT_INPUT_DRIFT:' + path)
    fit.authorize(plan['authorization'], fit.sha(fit.PREP / 'config.json'),
                  fit.sha(fit.ENTRY / 'manifest.json'), fit.sha(fit.SIDECAR / 'manifest.json'))
    config, parts, _ = fit.load_accepted()
    if plan['config'] != config:
        raise ValueError('FIT_CONFIG_MISMATCH')
    predictions = pd.read_parquet(fit.OUT / 'predictions.parquet')
    baseline = pd.read_parquet(fit.previous.OUT / 'predictions.parquet')
    report = json.loads((fit.OUT / 'report.json').read_bytes())
    if len(predictions) != 86319 or predictions.duplicated(['code', fit.previous.prep.TIME]).any():
        raise ValueError('PREDICTION_POPULATION_DRIFT')
    direct, count = {}, 0
    for algorithm in ('ridge', 'lightgbm'):
        state = json.loads((fit.OUT / (algorithm+'_preprocessing.json')).read_bytes())
        if state['columns'] != config['fields']:
            raise ValueError('PREPROCESSING_FIELD_ORDER_DRIFT')
        train = parts['train'][config['fields']].to_numpy(dtype=float)
        median = np.nanmedian(train, axis=0)
        filled = np.where(np.isnan(train), median, train)
        scale = filled.std(axis=0)
        expected = dict(median=median, mean=filled.mean(axis=0), scale=np.where(scale > 0, scale, 1.))
        for name, values in expected.items():
            np.testing.assert_allclose(state[name], values, atol=1e-12, rtol=0)
        if algorithm == 'ridge':
            model = json.loads((fit.OUT / 'ridge_model.json').read_bytes())
            if len(model['coef']) != 532:
                raise ValueError('RIDGE_TRANSFORM_DIMENSION_MISMATCH')
        else:
            model = fit.gzip.decompress((fit.OUT / 'lightgbm_model.txt.gz').read_bytes()).decode('utf-8')
        replay = fit.previous.replay(parts, config['fields'], algorithm, state, model)
        for segment, frame in parts.items():
            saved = predictions[predictions.segment.eq(segment)].reset_index(drop=True)
            old = baseline[baseline.segment.eq(segment)].reset_index(drop=True)
            for table in (saved, old):
                pd.testing.assert_frame_equal(frame[['code', fit.previous.prep.TIME]],
                                              table[['code', fit.previous.prep.TIME]])
            raw = saved[algorithm+'_raw'].to_numpy()
            np.testing.assert_allclose(replay[segment], raw, atol=1e-11, rtol=0)
            count += len(raw)
            y = frame[fit.previous.prep.TARGET].to_numpy()
            physical = np.clip(raw, 0, 1)
            old_p = np.clip(old[algorithm+'_raw'].to_numpy(), 0, 1)
            direct.setdefault(segment, {})[algorithm] = dict(mse=float(np.mean((physical-y)**2)),
                mae=float(np.mean(abs(physical-y))), mean_prediction=float(physical.mean()),
                mean_actual=float(y.mean()), old_mse=float(np.mean((old_p-y)**2)))
            for name in ('mse', 'mae'):
                np.testing.assert_allclose(direct[segment][algorithm][name],
                    report['metrics'][segment]['physical_amplitude_metrics'][algorithm][name],
                    atol=1e-14, rtol=0)
                delta = np.mean((physical-y)**2 - (old_p-y)**2) if name == 'mse' else (
                    np.mean(abs(physical-y) - abs(old_p-y)))
                np.testing.assert_allclose(delta,
                    report['increment_vs_saved_232'][segment][algorithm][name]['left_minus_right'],
                    atol=1e-14, rtol=0)
            for month, item in report['metrics'][segment]['monthly'].items():
                select = frame[fit.previous.prep.TIME].to_numpy()//100000000 == int(month)
                np.testing.assert_allclose(np.mean(physical[select]-y[select]),
                    item['metrics'][algorithm]['bias'], atol=1e-14, rtol=0)
    if count != 172638 or report['main_model_fits'] != 2 or report['calibration_fits'] != 0:
        raise ValueError('MODEL_OR_PREDICTION_COUNT_MISMATCH')
    writer = EvidenceWriter(OUT, cap=64*1024, reserve=8*1024)
    writer.json('acceptance.json', dict(status='PASS_FIXED_COMMON_FIT_READBACK_NOT_PROMOTION',
        source_manifest_sha256=fit.sha(fit.OUT / 'manifest.json'),
        source_report_sha256=fit.sha(fit.OUT / 'report.json'), audit_code_sha256=fit.sha(Path(__file__)),
        bound_inputs=len(plan['inputs']), replayed_predictions=count,
        train_only_preprocessing_recomputed=True, direct_metrics=direct,
        direct_pairwise_error_and_month_bias='PASS', original_saved_models_unchanged=True,
        new_fits=0, http_requests=0, raw_minute_replays=0, promotion_passed=False))
    writer.json('stop_record.json', dict(status='COMPLETE_READBACK_NOT_PROMOTION',
                receipt_sha256=fit.sha(OUT / 'acceptance.json')), terminal=True)
    return dict(status='PASS_FIXED_COMMON_FIT_READBACK_NOT_PROMOTION',
                replayed_predictions=count, test=direct['test'])


if __name__ == '__main__':
    print(json.dumps(run()))
