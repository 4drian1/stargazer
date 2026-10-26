"""Separate frozen cross-year entry. Check is read-only; old FIT authorization is invalid here."""
import argparse
from datetime import datetime, timezone, timedelta
from hashlib import sha256
from importlib.metadata import version
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from price_verification_guarded_run import encoded, sha
from research_budget_v2 import bytes_used, TOTAL_CAP
from risk_failure_diagnostics_v1 import base_valid, fold_inventory
from risk_fit_v1 import PACKAGES, arrays, fit_models, logits, probabilities, require, validate_rows
from risk_fit_eval_v1 import keep_mask
from risk_crossyear_eval_v1 import consistency_flags, evaluate_test

HERE = Path(__file__).resolve().parent
CONFIG = HERE/'contracts/crossyear_risk_config_v1.json'
MANIFEST = HERE/'contracts/crossyear_risk_manifest_v1.json'
CONTRACT = HERE/'contracts/R0_CROSSYEAR_EXECUTION_CONTRACT_V1.md'
OUTPUT = HERE/'historical_risk_crossyear_v1_run'
PERIODS = ('train', 'select', 'calibration', 'test')


def verify_refs(refs):
    for ref in refs:
        require(sha(ref['path']) == ref['sha256'], 'FROZEN_INPUT_DRIFT:'+ref['path'])


def assign_fold(rows, fold):
    require(len({(r['code'], r['P']) for r in rows}) == len(rows), 'DUPLICATE_STOCK_DAY')
    inventory = fold_inventory(rows, fold)
    groups = {}
    for i, role in enumerate(PERIODS):
        start, end = fold[role]
        boundary = fold[PERIODS[i+1]][0] if i < 3 else (end//10000+1)*10000+101
        groups[role] = sorted([r for r in rows if start <= r['P'] <= end and base_valid(r)
                              and r['P'] < r['label_end_date'] < boundary], key=lambda r: (r['P'], r['code']))
        # fold_inventory independently rejects revival of non-temporal quarantines.
        require(len(groups[role]) == inventory['segments'][role]['n'], 'FOLD_ASSIGNMENT_MISMATCH')
    return groups, inventory


def population_fingerprint(groups):
    return {name: dict(n=len(rows), keys_sha256=sha256(encoded(
        [[r['code'], r['P']] for r in rows])).hexdigest(), rows_sha256=sha256(encoded(rows)).hexdigest())
        for name, rows in groups.items()}


def load_data():
    config = json.loads(CONFIG.read_bytes())
    require(config['schema'] == 'R0_CROSSYEAR_EXECUTION_V1' and config['fit_authorized'] is False
            and config['new_output_cap'] == 64*1024**2 and config['network_requests'] == 0
            and config['deployment_authorized'] is False, 'CONFIG_SCOPE')
    require(HERE/config['output'] == OUTPUT and HERE/config['manifest'] == MANIFEST
            and HERE/config['contract'] == CONTRACT, 'OUTPUT_OR_MANIFEST_SCOPE')
    original = json.loads((HERE/'contracts/restricted_risk_fit_manifest_v1.json').read_bytes())
    verify_refs(original['inputs'])
    base = json.loads((HERE/config['base_config']).read_bytes())
    folds = json.loads((HERE/config['folds']).read_bytes())['folds']
    require([f['id'] for f in folds] == ['F1', 'F2', 'F3'], 'FOLD_IDS')
    require([f['mode'] for f in folds] == ['NEW_FIT_PENDING_AUTHORIZATION']*2
            + ['REUSE_FROZEN_FIRST_FIT_ONLY'], 'FOLD_EXECUTION_MODES')
    acceptance = json.loads((HERE/config['acceptance']).read_bytes())
    require(acceptance['data_preparation_ready'] is True, 'DATA_ACCEPTANCE_REQUIRED')
    rows = [r for ref in acceptance['matrix_files'] for r in pq.read_table(ref['path']).to_pylist()]
    require(len(rows) == acceptance['persisted_rows'], 'MATRIX_COUNT_DRIFT')
    original_groups = validate_rows(rows, base['fields'])
    require(sum(map(len, original_groups.values())) == acceptance['eligible'], 'OLD_ADMISSION_DRIFT')
    groups, inventories, fingerprints = {}, [], {}
    for fold in folds:
        groups[fold['id']], inventory = assign_fold(rows, fold)
        require(inventory['sample_gate_passed'], 'FOLD_SAMPLE_GATE:'+fold['id'])
        inventories.append(inventory)
        fingerprints[fold['id']] = population_fingerprint(groups[fold['id']])
    previous = json.loads((HERE/'risk_failure_diagnostics_v1_20261015/fold_inventory.json').read_bytes())
    require(inventories == previous['folds'], 'DIAGNOSTIC_INVENTORY_DRIFT')
    for role in PERIODS:
        require(groups['F3'][role] == original_groups[role], 'F3_ORIGINAL_POPULATION_DRIFT')
    run = HERE/config['reuse_run']
    require(run == HERE/'historical_risk_fit_v1_run', 'REUSE_RUN_SCOPE')
    stop = json.loads((run/'stop_record.json').read_bytes())
    require(stop['status'] == 'completed', 'ORIGINAL_FIT_NOT_COMPLETE')
    verify_refs([dict(path=str(run/r['name']), sha256=r['sha256']) for r in stop['files']])
    reused = pq.read_table(run/'predictions.parquet').to_pylist()
    reused = sorted([r for r in reused if r['segment'] == 'test'], key=lambda r: (r['P'], r['code']))
    require(len(reused) == len(groups['F3']['test']), 'F3_PREDICTION_COUNT')
    for prediction, row in zip(reused, groups['F3']['test']):
        require(all(prediction[k] == row[k] for k in ('code','P','label_end_date','mae_5d','y_tail_5d')),
                'F3_PREDICTION_IDENTITY')
    return config, base, groups, reused, inventories, fingerprints


def preflight():
    manifest = json.loads(MANIFEST.read_bytes())
    require(manifest['schema'] == 'R0_CROSSYEAR_MANIFEST_V1', 'MANIFEST_SCHEMA')
    require(manifest['config_sha256'] == sha(CONFIG) and manifest['contract_sha256'] == sha(CONTRACT),
            'CONFIG_OR_CONTRACT_DRIFT')
    require(manifest['runtime_versions'] == {p: version(p) for p in PACKAGES}, 'DEPENDENCY_DRIFT')
    verify_refs(manifest['inputs'])
    config, base, groups, reused, inventories, fingerprints = load_data()
    require(fingerprints == manifest['fold_populations'] and inventories == manifest['fold_inventory'],
            'FOLD_POPULATION_DRIFT')
    require(not OUTPUT.exists(), 'OUTPUT_EXISTS_NO_RESTART')
    require(bytes_used()+config['new_output_cap'] <= TOTAL_CAP, 'CROSSYEAR_BUDGET')
    return config, base, groups, reused, dict(data_ready=True, fit_ready=False, fit_authorized=False,
        model_fit_performed=False, manifest_sha256=sha(MANIFEST), config_sha256=sha(CONFIG),
        contract_sha256=sha(CONTRACT), frozen_inputs=len(manifest['inputs']),
        fold_inventory=inventories, fold_populations=fingerprints, network_requests=0,
        registered_actual_bytes=bytes_used(), new_output_cap=config['new_output_cap'],
        output=str(OUTPUT), coverage_verified=False, deployment_authorized=False)


def authorize(path, check):
    if path is None:
        return False, 'SEPARATE_CROSSYEAR_FIT_AUTHORIZATION_REQUIRED'
    try:
        record = json.loads(Path(path).read_bytes())
        require(record.get('status') == 'AUTHORIZED' and record.get('scope') == 'FIT_R0_CROSSYEAR_V1',
                'SEPARATE_CROSSYEAR_FIT_AUTHORIZATION_REQUIRED')
        require(record.get('new_fit_folds') == ['F1','F2'] and record.get('reuse_only_folds') == ['F3'],
                'AUTHORIZATION_FOLD_SCOPE')
        require(all(isinstance(record.get(k), str) and record[k].strip()
                    for k in ('signer','authorized_at','approval_reference')), 'AUTHORIZATION_EVIDENCE_REQUIRED')
        ts = datetime.fromisoformat(record['authorized_at'])
        require(ts.tzinfo is not None and ts <= datetime.now(timezone.utc)+timedelta(minutes=5),
                'AUTHORIZATION_TIMESTAMP_INVALID')
        require(all(record.get(k) == check[k] for k in ('manifest_sha256','config_sha256','contract_sha256')),
                'AUTHORIZATION_BINDING_MISMATCH')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, str(exc)
    return True, None


def predictions_for(rows, models, base):
    x, _ = arrays(rows, base['fields'])
    p = {}
    for name in ('lr','lightgbm'):
        p[name+'_raw'] = probabilities(models[name], x)
        p[name] = probabilities(models['calibrators'][name], logits(p[name+'_raw']))
    return p


def prediction_records(rows, p, constant, base, fold):
    masks = {m: keep_mask(rows, v, base['evaluation']['primary_rejection_fraction']) for m, v in dict(
        lr=p['lr'], lightgbm=p['lightgbm'], atr_rule=np.array([r['features']['atr14_pct'] for r in rows])).items()}
    return [dict(code=r['code'], P=r['P'], label_end_date=r['label_end_date'],
        y_tail_5d=r['y_tail_5d'], mae_5d=r['mae_5d'], fold=fold, segment='test',
        p_constant=constant, atr_score=r['features']['atr14_pct'],
        scope=r['admission_scope'], coverage_verified=False,
        **{k:float(v[i]) for k,v in p.items()},
        **{k+'_retained_primary':bool(v[i]) for k,v in masks.items()}) for i,r in enumerate(rows)]


def train(config, base, groups, reused, check, authorization):
    allowed, reason = authorize(authorization, check)
    require(allowed, reason)
    verify_refs(json.loads(MANIFEST.read_bytes())['inputs'])
    require(sha(MANIFEST) == check['manifest_sha256'], 'MANIFEST_DRIFT')
    auth_hash = sha(authorization)
    OUTPUT.mkdir(exist_ok=False)
    reserve, used, error = 2*1024**2, 0, None
    attempted, completed, reports = [], [], {}

    def save(name, data):
        nonlocal used
        require(used+len(data)+reserve <= config['new_output_cap']
                and bytes_used()+len(data)+reserve <= TOTAL_CAP, 'CROSSYEAR_SERIALIZED_BUDGET_STOP')
        with (OUTPUT/name).open('xb') as f:
            f.write(data)
        used += len(data)

    try:
        save('input_and_authorization.json', encoded(dict(check=check, config=config,
             authorization_path=str(Path(authorization).resolve()), authorization_sha256=auth_hash)))
        for fold in ('F1','F2','F3'):
            rows = groups[fold]['test']
            if fold == 'F3':
                p = {k:np.array([r[k] for r in reused]) for k in ('lr_raw','lr','lightgbm_raw','lightgbm')}
                constant = reused[0]['p_constant']
                require(all(r['p_constant'] == constant for r in reused), 'F3_CONSTANT_DRIFT')
                require(abs(constant-np.mean([r['y_tail_5d'] for r in groups[fold]['train']])) < 1e-12,
                        'F3_TRAIN_CONSTANT_MISMATCH')
            else:
                attempted.append(fold)
                models = fit_models(groups[fold], base)
                completed.append(fold)
                import joblib
                buffer = io.BytesIO()
                joblib.dump(models, buffer)
                save(fold+'_models.joblib', buffer.getvalue())
                p, constant = predictions_for(rows, models, base), models['constant']
            records = prediction_records(rows, p, constant, base, fold)
            if fold == 'F3':
                require(all(all(r[k] == old[k] for k in old) for r,old in zip(records,reused)), 'F3_REUSE_DRIFT')
            sink = pa.BufferOutputStream()
            pq.write_table(pa.Table.from_pylist(records), sink, compression='zstd')
            save(fold+'_test_predictions.parquet', sink.getvalue().to_pybytes())
            reports[fold] = evaluate_test(rows, p, constant, base['evaluation'])
            reports[fold]['execution'] = 'REUSED_FROZEN_PREDICTIONS' if fold == 'F3' else 'NEW_FIT'
            if fold != 'F3':
                reports[fold]['best_iteration'] = models['best_iteration']
            save(fold+'_report.json', encoded(reports[fold]))
        summary = dict(folds=list(reports), flags=consistency_flags(list(reports.values())),
            scope='CONDITIONAL_RESEARCH_DOUBLE_SOURCE_ORDINARY_ACTIONS', coverage_verified=False,
            new_blind_test=False, network_requests=0, deployment_authorized=False,
            note='Existing 43-stock source-visible corpus only; all years and deep-tail red flags retained.')
        verify_refs(json.loads(MANIFEST.read_bytes())['inputs'])
        require(sha(MANIFEST) == check['manifest_sha256'] and sha(authorization) == auth_hash,
                'MANIFEST_OR_AUTHORIZATION_DRIFT_DURING_FIT')
        save('summary.json', encoded(summary))
    except (Exception, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}:{exc}'
    terminal = dict(status='failed' if error else 'completed', error=error,
        fit_attempted_folds=attempted, fit_completed_folds=completed,
        reused_folds=['F3'] if 'F3' in reports else [], production_gate_passed=False,
        deployment_authorized=False, network_requests=0,
        files=[dict(name=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in OUTPUT.iterdir() if p.is_file()])
    data = encoded(terminal)
    require(len(data) <= reserve and used+len(data) <= config['new_output_cap']
            and bytes_used()+len(data) <= TOTAL_CAP, 'TERMINAL_BUDGET_STOP')
    with (OUTPUT/'stop_record.json').open('xb') as f:
        f.write(data)
    print(json.dumps(terminal, ensure_ascii=False))
    return 1 if error else 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('check','train'))
    parser.add_argument('--authorization', type=Path)
    args = parser.parse_args(argv)
    try:
        config, base, groups, reused, check = preflight()
        allowed, reason = authorize(args.authorization, check)
        check.update(fit_ready=allowed, fit_authorized=allowed, blocked=reason)
        if args.command == 'check' or not allowed:
            print(json.dumps(check, ensure_ascii=False, indent=2))
            return 0 if allowed else 2
        return train(config, base, groups, reused, check, args.authorization)
    except Exception as exc:
        print(json.dumps(dict(data_ready=False, fit_ready=False, model_fit_performed=False,
            error=f'{type(exc).__name__}:{exc}'), ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
