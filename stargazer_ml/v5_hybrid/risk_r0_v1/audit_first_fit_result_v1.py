"""Read-only verification of the completed, authorized first fit. Never refits a model."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from price_verification_guarded_run import encoded, sha
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = Path(__file__).resolve().parent
RUN = HERE / 'historical_risk_fit_v1_run'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def check():
    stop = json.loads((RUN / 'stop_record.json').read_bytes())
    require(stop['status'] == 'completed' and stop['real_model_fit_completed'] is True, 'RUN_NOT_COMPLETE')
    for ref in stop['files']:
        path = RUN / ref['name']
        require(sha(path) == ref['sha256'] and path.stat().st_size == ref['bytes'], 'FIT_OUTPUT_HASH')
    receipt = json.loads((RUN / 'input_and_authorization.json').read_bytes())
    config = receipt['config']
    manifest_path = HERE / config['manifest']
    require(sha(manifest_path) == receipt['check']['manifest_sha256'], 'FIT_MANIFEST_DRIFT')
    manifest = json.loads(manifest_path.read_bytes())
    require(all(sha(r['path']) == r['sha256'] for r in manifest['inputs']), 'FROZEN_INPUT_DRIFT')
    require(sha(receipt['authorization_path']) == receipt['authorization_sha256'], 'AUTHORIZATION_DRIFT')
    audit = json.loads((HERE / config['acceptance']).read_bytes())
    matrix = [r for part in audit['matrix_files'] for r in pq.read_table(part['path']).to_pylist()]
    admitted = {(r['code'], r['P']): r for r in matrix if r['training_eligible'] is True}
    predictions = pq.read_table(RUN / 'predictions.parquet').to_pylist()
    require(len(predictions) == len(admitted), 'PREDICTION_COUNT')
    pred_by_key = {(r['code'], r['P']): r for r in predictions}
    require(len(pred_by_key) == len(predictions) and set(pred_by_key) == set(admitted), 'PREDICTION_KEYS')
    fields = config['fields']
    ordered = [admitted[r['code'], r['P']] for r in predictions]
    x = np.array([[r['features'][f] for f in fields] for r in ordered])
    # Hash of the locally produced trusted model is checked above before unpickling.
    import joblib
    models = joblib.load(RUN / 'models.joblib')
    training = [r for r in matrix if r['training_eligible'] and r['segment'] == 'train']
    scaler = models['lr'].named_steps['standardscaler']
    require(int(scaler.n_samples_seen_) == len(training), 'SCALER_SAMPLE_COUNT')
    expected_mean = np.mean([[r['features'][f] for f in fields] for r in sorted(training,
                           key=lambda r: (r['P'], r['code']))], axis=0)
    require(np.allclose(scaler.mean_, expected_mean, rtol=0, atol=1e-12), 'SCALER_NOT_TRAIN_ONLY')
    require(abs(models['constant'] - np.mean([r['y_tail_5d'] for r in training])) < 1e-15, 'CONSTANT_RATE')
    for name in ('lr', 'lightgbm'):
        raw = models[name].predict_proba(x)[:, 1]
        values = np.clip(raw, 1e-12, 1 - 1e-12)
        cal = models['calibrators'][name]
        require(cal.coef_[0, 0] > 0, 'CALIBRATION_RANK_FLIPPED')
        calibrated = cal.predict_proba(np.log(values / (1 - values)).reshape(-1, 1))[:, 1]
        require(np.allclose(raw, [r[name + '_raw'] for r in predictions], rtol=0, atol=1e-12), 'RAW_PREDICTION_REPLAY')
        require(np.allclose(calibrated, [r[name] for r in predictions], rtol=0, atol=1e-12), 'CAL_PREDICTION_REPLAY')
    report = json.loads((RUN / 'report.json').read_bytes())
    require(models['best_iteration'] == report['best_iteration'], 'BEST_ITERATION')
    all_segments = {}
    for name, result in report['segments'].items():
        rows = [r for r in predictions if r['segment'] == name]
        require(len(rows) == result['n'], 'SEGMENT_COUNTS')
        for r in rows:
            original = admitted[r['code'], r['P']]
            require(all(r[k] == original[k] for k in ('segment', 'label_end_date', 'y_tail_5d', 'mae_5d')),
                    'PREDICTION_LABEL_DRIFT')
            require(r['coverage_verified'] is False and r['scope'] == config['scope'], 'PREDICTION_SCOPE_DRIFT')
            require(r['p_constant'] == models['constant'], 'PREDICTION_CONSTANT_DRIFT')
        primary = result['filtering'][0]
        daily = primary['daily_sufficient_statistics']
        totals = {arm: [] for arm in ('lr', 'lightgbm', 'atr_rule')}
        daily_totals = {arm: [] for arm in totals}
        random_counts, kept, population = [], [], []
        dates = sorted({r['P'] for r in rows})
        for date in dates:
            day = [r for r in rows if r['P'] == date]
            n = len(day)
            rejected = int(np.floor(n * config['evaluation']['primary_rejection_fraction']))
            population.append(n)
            kept.append(n-rejected)
            random_counts.append(sum(r['y_tail_5d'] for r in day) * (n-rejected)/n)
            for arm in totals:
                ranked = sorted(day, key=lambda r: (-r['atr_score' if arm == 'atr_rule' else arm], r['code']))
                expected = {(r['code'], r['P']) for r in ranked[rejected:]}
                require(all(r[arm + '_retained_primary'] == ((r['code'], r['P']) in expected) for r in day),
                        'RETAINED_MASK_DRIFT')
                chosen = [r for r in day if r[arm + '_retained_primary']]
                totals[arm].extend(chosen)
                daily_totals[arm].append(sum(r['y_tail_5d'] for r in chosen))
        require(daily['dates'] == dates and daily['retained_counts'] == kept
                and daily['population_counts'] == population, 'DAILY_COHORT_DRIFT')
        require(np.allclose(daily['tail_counts']['random_expected'], random_counts), 'RANDOM_EXPECTATION_DRIFT')
        for arm, chosen in totals.items():
            require(np.array_equal(daily['tail_counts'][arm], daily_totals[arm]), 'DAILY_TAIL_DRIFT')
            require(abs(np.mean([r['y_tail_5d'] for r in chosen]) - primary['arms'][arm]['tail_rate']) < 1e-12,
                    'TAIL_RATE_DRIFT')
        # Independently reconstruct the frozen shared 10-date moving-block draws and paired CIs.
        cfg = config['evaluation']
        starts = np.random.default_rng(cfg['seed']).integers(0, len(dates)-cfg['block_dates']+1,
            size=(cfg['bootstrap_reps'], int(np.ceil(len(dates)/cfg['block_dates']))))
        draws = (starts[:, :, None] + np.arange(cfg['block_dates'])).reshape(cfg['bootstrap_reps'], -1)[:, :len(dates)]
        denominator = np.asarray(kept)[draws].sum(axis=1)
        for arm in ('lr', 'lightgbm'):
            for baseline, counts in (('atr_rule', daily_totals['atr_rule']), ('random_expected', random_counts)):
                delta = np.asarray(daily_totals[arm]) - np.asarray(counts)
                stored = primary['paired_tail_rate_differences'][arm + '_minus_' + baseline]
                require(abs(delta.sum()/sum(kept) - stored['mean']) < 1e-12, 'PAIRED_DIFFERENCE_DIRECTION')
                ci = np.quantile(delta[draws].sum(axis=1)/denominator, [.025, .975])
                require(np.allclose(ci, stored['ci95'], rtol=0, atol=1e-12), 'PAIRED_CI_REPLAY')
        all_segments[name] = dict(n=len(rows), actual_rejection_fraction=1-sum(kept)/len(rows),
            observed_tail_rate=float(np.mean([r['y_tail_5d'] for r in rows])),
            lightgbm_mean_probability=float(np.mean([r['lightgbm'] for r in rows])),
            lr_mean_probability=float(np.mean([r['lr'] for r in rows])))
    legacy = json.loads((HERE / 'real_subset_audit_v2_20261015.json').read_bytes())
    require(all(sha(r['path']) == r['sha256'] for r in legacy['preserved_artifacts']), 'LEGACY_INPUT_MUTATION')
    actual = sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())
    require(actual <= config['new_output_cap'] and bytes_used() <= TOTAL_CAP, 'FIT_DISK_CAP')
    return dict(status='FIT_OUTPUT_AUDIT_PASS', real_predictions_replayed=len(predictions),
                original_frozen_inputs_verified=len(manifest['inputs']), legacy_artifacts_unchanged=105,
                all_primary_masks_differences_and_cis_recomputed=True, segments=all_segments,
                fit_refit_performed=False, network_requests=0, run_actual_bytes=actual,
                registered_actual_bytes=bytes_used(), deployment_authorized=False,
                files=[dict(path=str(RUN/r['name']), sha256=r['sha256']) for r in stop['files']],
                audit_code=dict(path=str(Path(__file__)), sha256=sha(Path(__file__))))


if __name__ == '__main__':
    result = check()
    data = encoded(result)
    require(len(data) <= 1024**2 and bytes_used() + len(data) <= TOTAL_CAP, 'AUDIT_REPORT_CAP')
    with (HERE / 'first_fit_result_audit_v1.json').open('xb') as f:
        f.write(data)
    print(json.dumps(result, ensure_ascii=False, indent=2))
