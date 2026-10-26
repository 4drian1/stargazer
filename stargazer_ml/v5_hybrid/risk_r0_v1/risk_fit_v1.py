"""Frozen restricted-history fit entry. `check` never fits; `train` requires separate FIT record."""
import argparse
from datetime import datetime, timezone, timedelta
from importlib.metadata import version
import io
import json
from pathlib import Path
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from historical_risk_core_v1 import STOCK_FIELDS, MARKET_FIELDS, sample_gate, segment
from price_verification_guarded_run import encoded, sha
from research_budget_v2 import bytes_used, TOTAL_CAP
from risk_fit_eval_v1 import evaluate_segment, keep_mask

HERE = Path(__file__).resolve().parent
CONFIG = HERE / 'contracts/restricted_risk_fit_config_v1.json'
CONTRACT = HERE / 'contracts/RESTRICTED_RISK_FIT_CONTRACT_V1.md'
SCOPE = 'CONDITIONAL_RESEARCH_DOUBLE_SOURCE_ORDINARY_ACTIONS'
PACKAGES = ('numpy', 'scipy', 'scikit-learn', 'lightgbm', 'pyarrow', 'joblib')


def require(value, reason):
    if not value:
        raise ValueError(reason)


def validate_rows(rows, fields):
    require(fields == list(STOCK_FIELDS + MARKET_FIELDS), 'FEATURE_WHITELIST_CHANGED')
    keys, groups = set(), {k: [] for k in ('train', 'select', 'calibration', 'test', 'description')}
    for row in rows:
        key = (row['code'], row['P'])
        require(key not in keys, 'DUPLICATE_STOCK_DAY')
        keys.add(key)
        require(row['admission_scope'] == SCOPE and row['coverage_verified'] is False, 'SCOPE_MISMATCH')
        if row['training_eligible'] is not True:
            continue
        seg, within = segment(row['P'], row['label_end_date'])
        require(seg == row['segment'] and within and seg in groups, 'PURGE_OR_SEGMENT_VIOLATION')
        require(row['label_status'] == 'LABEL_COMPLETE' and type(row['y_tail_5d']) is int
                and row['y_tail_5d'] in (0, 1) and type(row['mae_5d']) in (int, float)
                and np.isfinite(row['mae_5d']), 'ADMITTED_LABEL_INVALID')
        require(all(type(row['features'].get(f)) in (int, float)
                    and np.isfinite(row['features'][f]) for f in fields), 'ADMITTED_FEATURE_INVALID')
        groups[seg].append(row)
    for values in groups.values():
        values.sort(key=lambda r: (r['P'], r['code']))
    return groups


def preflight(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_bytes())
    require(config['schema'] == 'RESTRICTED_R0_FIT_V1' and config['scope'] == SCOPE, 'CONFIG_SCOPE')
    manifest_path = HERE / config['manifest']
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest['schema'] == 'RESTRICTED_R0_FIT_MANIFEST_V1', 'MANIFEST_SCHEMA')
    require(manifest['config_sha256'] == sha(config_path), 'CONFIG_DRIFT')
    require(manifest['contract_sha256'] == sha(CONTRACT), 'CONTRACT_DRIFT')
    require(manifest['runtime_versions'] == {p: version(p) for p in PACKAGES}, 'DEPENDENCY_VERSION_DRIFT')
    for ref in manifest['inputs']:
        require(sha(ref['path']) == ref['sha256'], 'FROZEN_INPUT_DRIFT:' + ref['path'])
    acceptance = json.loads((HERE / config['acceptance']).read_bytes())
    require(acceptance['status'] == 'PERSISTED_MATRIX_AUDIT_PASS'
            and acceptance['data_preparation_ready'] is True, 'DATA_ACCEPTANCE_REQUIRED')
    rows = []
    for ref in acceptance['matrix_files']:
        require(sha(ref['path']) == ref['sha256'], 'MATRIX_DRIFT')
        rows.extend(pq.read_table(ref['path']).to_pylist())
    require(len(rows) == acceptance['persisted_rows'], 'POPULATION_COUNT_DRIFT')
    groups = validate_rows(rows, config['fields'])
    gate = sample_gate(rows)
    require(gate['training_ready'] and gate['segments'] == acceptance['gate']['segments'], 'SAMPLE_GATE_FAIL')
    require(sum(len(v) for v in groups.values()) == acceptance['eligible'], 'ADMITTED_COUNT_DRIFT')
    output = HERE / config['output']
    require(output.resolve().parent == HERE and output.name == 'historical_risk_fit_v1_run', 'OUTPUT_SCOPE')
    require(not output.exists(), 'OUTPUT_EXISTS_NO_RESTART')
    require(config['new_output_cap'] == 64 * 1024**2 and bytes_used() + config['new_output_cap'] <= TOTAL_CAP,
            'FIT_OUTPUT_BUDGET')
    report = dict(data_ready=True, fit_ready=False, fit_authorized=False, model_fit_performed=False,
                  scope=SCOPE, eligible=sum(len(v) for v in groups.values()), sample_gate=gate,
                  manifest_sha256=sha(manifest_path), config_sha256=sha(config_path),
                  contract_sha256=sha(CONTRACT), registered_actual_bytes=bytes_used(),
                  output=str(output), coverage_verified=False)
    return config, groups, report


def authorize(path, report):
    if path is None:
        return False, 'FIT_AUTHORIZATION_REQUIRED'
    try:
        record = json.loads(Path(path).read_bytes())
        require(record.get('status') == 'AUTHORIZED' and record.get('scope') == 'FIT_RESTRICTED_R0_V1',
                'FIT_AUTHORIZATION_REQUIRED')
        require(all(isinstance(record.get(f), str) and record[f].strip()
                    for f in ('signer', 'authorized_at', 'approval_reference')), 'AUTHORIZATION_EVIDENCE_REQUIRED')
        timestamp = datetime.fromisoformat(record['authorized_at'])
        require(timestamp.tzinfo is not None, 'AUTHORIZATION_TIMEZONE_REQUIRED')
        require(timestamp <= datetime.now(timezone.utc) + timedelta(minutes=5), 'AUTHORIZATION_TIME_IN_FUTURE')
        for field in ('manifest_sha256', 'config_sha256', 'contract_sha256'):
            require(record.get(field) == report[field], 'AUTHORIZATION_BINDING_MISMATCH')
    except (ValueError, KeyError, OSError, TypeError) as exc:
        return False, str(exc)
    return True, None


def arrays(rows, fields):
    return np.array([[r['features'][f] for f in fields] for r in rows], dtype=float), np.array(
        [r['y_tail_5d'] for r in rows], dtype=int)


def probabilities(model, x):
    p = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'MODEL_PROBABILITY_INVALID')
    return p


def logits(p):
    p = np.asarray(p, dtype=float)
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'CALIBRATION_INPUT_INVALID')
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_models(groups, config):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.exceptions import ConvergenceWarning
    import lightgbm as lgb

    data = {name: arrays(groups[name], config['fields']) for name in ('train', 'select', 'calibration')}
    x, y = data['train']
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        lr = make_pipeline(StandardScaler(), LogisticRegression(**config['lr']))
        lr.fit(x, y)
        gb = lgb.LGBMClassifier(**config['lightgbm'])
        gb.fit(x, y, eval_set=[data['select']], eval_metric='binary_logloss',
               callbacks=[lgb.early_stopping(config['early_stopping_rounds'], first_metric_only=True, verbose=False)])
        require(type(gb.best_iteration_) is int and gb.best_iteration_ > 0, 'EARLY_STOP_ITERATION_INVALID')
        calibrators = {}
        for name, model in (('lr', lr), ('lightgbm', gb)):
            raw = probabilities(model, data['calibration'][0])
            calibrator = LogisticRegression(**config['sigmoid'])
            calibrator.fit(logits(raw), data['calibration'][1])
            require(np.isfinite(calibrator.coef_).all() and calibrator.coef_[0, 0] > 0
                    and np.isfinite(calibrator.intercept_).all(), 'SIGMOID_NONPOSITIVE_OR_NONFINITE_SLOPE')
            calibrators[name] = calibrator
    return dict(lr=lr, lightgbm=gb, calibrators=calibrators,
                constant=float(y.mean()), best_iteration=gb.best_iteration_)


def train(config, groups, check, authorization):
    allowed, reason = authorize(authorization, check)
    require(allowed, reason)
    output = Path(check['output'])
    reserve = 2 * 1024**2
    output.mkdir(exist_ok=False)
    used, error = 0, None
    fit_attempted, fit_complete = False, False

    def save(name, data):
        nonlocal used
        require(used + len(data) + reserve <= config['new_output_cap']
                and bytes_used() + len(data) + reserve <= TOTAL_CAP, 'SERIALIZED_FIT_BUDGET_STOP')
        with (output / name).open('xb') as f:
            f.write(data)
        used += len(data)

    try:
        save('input_and_authorization.json', encoded(dict(check=check,
             authorization_path=str(Path(authorization).resolve()), authorization_sha256=sha(authorization),
             config=config)))
        fit_attempted = True
        models = fit_models(groups, config)
        fit_complete = True
        import joblib
        buffer = io.BytesIO()
        joblib.dump(models, buffer)
        save('models.joblib', buffer.getvalue())
        results, predictions = {}, []
        for name, rows in groups.items():
            if not rows:
                continue
            x, _ = arrays(rows, config['fields'])
            probs = {}
            for model_name in ('lr', 'lightgbm'):
                raw = probabilities(models[model_name], x)
                probs[model_name + '_raw'] = raw
                probs[model_name] = probabilities(models['calibrators'][model_name], logits(raw))
            results[name] = evaluate_segment(rows, probs, models['constant'], config['evaluation'])
            results[name]['role'] = dict(train='MODEL_FIT_IN_SAMPLE', select='GB_EARLY_STOP_SELECTION',
                calibration='CALIBRATOR_FIT_IN_SAMPLE', test='HISTORICAL_HOLDOUT_NOT_NEW_BLIND_TEST',
                description='DESCRIPTIVE_ONLY')[name]
            masks = {k: keep_mask(rows, v, config['evaluation']['primary_rejection_fraction'])
                     for k, v in dict(lr=probs['lr'], lightgbm=probs['lightgbm'],
                         atr_rule=np.array([r['features']['atr14_pct'] for r in rows])).items()}
            for i, row in enumerate(rows):
                predictions.append(dict(code=row['code'], P=row['P'], segment=name,
                    label_end_date=row['label_end_date'], y_tail_5d=row['y_tail_5d'], mae_5d=row['mae_5d'],
                    scope=SCOPE, coverage_verified=False, p_constant=models['constant'],
                    atr_score=row['features']['atr14_pct'], **{k: float(v[i]) for k, v in probs.items()},
                    **{k + '_retained_primary': bool(v[i]) for k, v in masks.items()}))
        sink = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pylist(predictions), sink, compression='zstd')
        save('predictions.parquet', sink.getvalue().to_pybytes())
        test = results['test']
        primary = test['filtering'][0]['paired_tail_rate_differences']
        p = test['probability']
        flags = dict(probability=(p['lightgbm']['brier_minus_constant']['ci95'][1] < 0
                     and p['lightgbm']['log_loss'] <= p['constant']['log_loss']),
                     paired_tail=(primary['lightgbm_minus_atr_rule']['ci95'][1] < 0
                     and primary['lightgbm_minus_random_expected']['ci95'][1] < 0))
        result = dict(scope=SCOPE, coverage_verified=False, best_iteration=models['best_iteration'],
                      constant=models['constant'], segments=results, research_evidence_flags=flags,
                      production_gate_passed=False, deployment_authorized=False,
                      note='One restricted historical holdout; not full multi-year Walk-forward validation.')
        # Before publishing a completed fit, revalidate all data/code and authorization fingerprints.
        manifest = json.loads((HERE / config['manifest']).read_bytes())
        require(all(sha(r['path']) == r['sha256'] for r in manifest['inputs']), 'INPUT_DRIFT_DURING_FIT')
        require(sha(HERE / config['manifest']) == check['manifest_sha256'], 'MANIFEST_DRIFT_DURING_FIT')
        saved_auth = json.loads((output / 'input_and_authorization.json').read_bytes())
        require(sha(authorization) == saved_auth['authorization_sha256'], 'AUTHORIZATION_DRIFT_DURING_FIT')
        save('report.json', encoded(result))
    except (Exception, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}:{exc}'
    files = [dict(name=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in output.iterdir() if p.is_file()]
    terminal = dict(status='failed' if error else 'completed', error=error, files=files,
                    model_fit_attempted=fit_attempted, real_model_fit_completed=fit_complete,
                    scope=SCOPE, production_gate_passed=False, deployment_authorized=False)
    data = encoded(terminal)
    require(len(data) <= reserve and used + len(data) <= config['new_output_cap']
            and bytes_used() + len(data) <= TOTAL_CAP, 'TERMINAL_BUDGET')
    with (output / 'stop_record.json').open('xb') as f:
        f.write(data)
    print(json.dumps(terminal, ensure_ascii=False))
    return 1 if error else 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('check', 'train'))
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--authorization', type=Path)
    args = parser.parse_args(argv)
    try:
        config, groups, report = preflight(args.config)
        allowed, reason = authorize(args.authorization, report)
        report.update(fit_ready=allowed, fit_authorized=allowed, blocked=reason)
        if args.command == 'check' or not allowed:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if allowed else 2
        return train(config, groups, report, args.authorization)
    except Exception as exc:
        print(json.dumps(dict(data_ready=False, fit_ready=False, model_fit_performed=False,
                             error=f'{type(exc).__name__}:{exc}'), ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
