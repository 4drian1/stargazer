"""Read/predict-only replay of the frozen first fit; no estimator is fitted here."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from inventory_ths_stockdb_cache_v1 import HERE, digest
from postclose_training_v1 import CAP, check
from research_budget_v2 import TOTAL_CAP, bytes_used

OUT = HERE / 'postclose_first_research_fit_20261017_v1'


def loss(y, prediction):
    residual = np.asarray(y) - np.asarray(prediction)
    return np.maximum(.9 * residual, -.1 * residual)


def verify_rows(actual, expected):
    columns = ['code', 'day', 'label_end_time', 'next_session_adverse_excursion', 'segment']
    if actual[['code', 'day']].duplicated().any():
        raise ValueError('DUPLICATE_PREDICTION')
    a = actual[columns].sort_values(['segment', 'code', 'day']).reset_index(drop=True)
    b = expected[columns].sort_values(['segment', 'code', 'day']).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=True)


def day_block_interval(frame, draws=1000):
    """Descriptive row-weighted loss difference, resampling blocks of 10 dates."""
    daily = frame.groupby('day', sort=True)['loss_difference'].agg(['sum', 'count'])
    blocks = [daily.iloc[i:i + 10].sum().to_numpy() for i in range(0, len(daily), 10)]
    values = np.asarray(blocks)
    rng = np.random.default_rng(20261017)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].sum(axis=1)
    return dict(blocks=len(blocks), draws=draws, interval_95=np.quantile(
        sampled[:, 0] / sampled[:, 1], [.025, .975]).tolist(),
        status='POST_FIT_DESCRIPTIVE_NOT_PREREGISTERED_GATE')


def run():
    import lightgbm as lgb

    plan = json.loads((OUT / 'plan.json').read_bytes())
    original_names = ['plan.json', 'preprocessing.json', 'model.txt', 'linear_model.json',
                      'predictions.parquet', 'report.json', 'stop_record.json']
    fingerprints = {name: digest((OUT / name).read_bytes()) for name in original_names}
    for path, sha in plan['inputs'].items():
        if digest(Path(path).read_bytes()) != sha:
            raise ValueError('FROZEN_INPUT_DRIFT:' + path)
    terminal = json.loads((OUT / 'stop_record.json').read_bytes())
    if terminal['status'] != 'COMPLETE_RESEARCH_FIT_NOT_DEPLOYMENT_READY':
        raise ValueError('FIT_NOT_COMPLETE')
    _, manifest, policy, parts = check()
    if plan['fields'] != manifest['fields'] or plan['policy'] != policy:
        raise ValueError('FROZEN_PLAN_DRIFT')
    predictions = pd.read_parquet(OUT / 'predictions.parquet')
    expected = pd.concat([f.assign(segment=s) for s, f in parts.items()], ignore_index=True)
    verify_rows(predictions, expected)
    fields = plan['fields']
    state = json.loads((OUT / 'preprocessing.json').read_bytes())
    if state['fields'] != fields:
        raise ValueError('PREPROCESSING_FIELD_DRIFT')
    train = parts['train'][fields].to_numpy(dtype=float)
    median = np.nanmedian(train, axis=0)
    filled = np.where(np.isnan(train), median, train)
    mean, scale = filled.mean(axis=0), filled.std(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    for name, actual in [('median', median), ('mean', mean), ('scale', scale)]:
        np.testing.assert_allclose(state[name], actual, rtol=0, atol=1e-12)
    model = lgb.Booster(model_file=str(OUT / 'model.txt'))
    linear = json.loads((OUT / 'linear_model.json').read_bytes())
    report = json.loads((OUT / 'report.json').read_bytes())
    constant = float(np.quantile(parts['train'].next_session_adverse_excursion, .9))
    metrics, raw_by_segment = {}, {}
    for segment, frame in parts.items():
        x = frame[fields].to_numpy(dtype=float)
        missing = np.isnan(x)
        raw = model.predict(np.c_[x, missing.astype(float)], num_threads=2)
        raw_by_segment[segment] = raw
        standardized = (np.where(missing, median, x) - mean) / scale
        lin = np.clip(np.c_[standardized, missing.astype(float)] @ np.asarray(linear['coef'])
                      + linear['intercept'], 0, 1)
        stored = predictions[predictions.segment.eq(segment)]
        pred = np.clip(raw + report['quantile_shift'], 0, 1)
        for column, value in [('q90_raw', raw), ('q90_adverse', pred), ('q90_linear', lin)]:
            np.testing.assert_allclose(stored[column], value, rtol=0, atol=1e-12)
        y = frame.next_session_adverse_excursion.to_numpy()
        facts = dict(rows=len(frame), dates=int(frame.day.nunique()), stocks=int(frame.code.nunique()),
                     pinball_q90=float(loss(y, pred).mean()),
                     linear_pinball_q90=float(loss(y, lin).mean()),
                     constant_pinball_q90=float(loss(y, constant).mean()),
                     empirical_q90_coverage=float((y <= pred).mean()))
        for key, value in facts.items():
            np.testing.assert_allclose(value, report['metrics'][segment][key], rtol=0, atol=1e-12)
        facts.update(raw_pinball_q90=float(loss(y, np.clip(raw, 0, 1)).mean()),
                     decision_day_min=int(frame.day.min()), decision_day_max=int(frame.day.max()),
                     label_end_max=int(frame.label_end_time.max()))
        metrics[segment] = facts
    delta = np.quantile(parts['calibration'].next_session_adverse_excursion.to_numpy()
                        - raw_by_segment['calibration'], .9)
    np.testing.assert_allclose(delta, report['quantile_shift'], rtol=0, atol=1e-12)
    if model.current_iteration() != report['best_iteration']:
        raise ValueError('TREE_COUNT_DRIFT')
    test = predictions[predictions.segment.eq('test')].copy()
    test['loss_difference'] = (loss(test.next_session_adverse_excursion, test.q90_adverse)
                               - loss(test.next_session_adverse_excursion, constant))
    result = dict(status='PASS_STORAGE_PREDICTION_AND_METRIC_REPLAY', actual_refit=False,
                  audited_original_sha256=fingerprints,
                  auditor_sha256=digest(Path(__file__).read_bytes()),
                  authorization_sha256=digest((HERE / 'contracts' /
                      'POSTCLOSE_FIRST_FIT_CHAT_AUTHORIZATION_20261017.json').read_bytes()),
                  train_constant_q90=constant, calibration_shift=float(delta), metrics=metrics,
                  test_model_minus_constant=float(test.loss_difference.mean()),
                  test_relative_worsening_vs_constant=float(test.loss_difference.mean() /
                      metrics['test']['constant_pinball_q90']),
                  test_day_block_interval=day_block_interval(test),
                  caveats=['Linear baseline is uncalibrated; comparison does not isolate nonlinear value.',
                           'Raw model diagnostic cannot replace the frozen calibrated primary pipeline.',
                           'Research reconstruction, not independent blind test or trading PnL.'],
                  decision='NO_INCREMENTAL_VALUE_PROVEN_NO_DEPLOYMENT')
    for name, sha in fingerprints.items():
        if digest((OUT / name).read_bytes()) != sha:
            raise ValueError('AUDIT_CHANGED_ORIGINAL:' + name)
    body = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) + len(body) > CAP or bytes_used() + len(body) > TOTAL_CAP:
        raise ValueError('AUDIT_OUTPUT_BUDGET_STOP')
    with (OUT / 'independent_readback_audit.json').open('xb') as stream:
        stream.write(body)
    return result


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, allow_nan=False, indent=2))
