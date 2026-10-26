"""One fixed, separately chat-authorized 266-field study; no raw replay or HTTP."""
import argparse
from datetime import datetime, timezone
import gzip
from importlib.metadata import version
import io
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import auxiliary_regression_fit_v1 as previous
from auxiliary_regression_eval_v1 import evaluate, paired_errors
from common_candidate_aggregation_v1 import attach_sidecars, learning_fields
from prepare_common_candidate_comparison_v1 import sha, frame_sha
from research_budget_common_v3 import EvidenceWriter, ADDENDUM, bytes_used, TOTAL_CAP

HERE = Path(__file__).resolve().parent
PREP = HERE / 'common_candidate_preparation_20261019_v1'
SIDECAR = HERE / 'common_candidate_sidecar_20261019_v1'
RECEIPT = HERE / 'common_candidate_sidecar_acceptance_20261019_v1'
ENTRY = HERE / 'common_candidate_fit_entry_20261019_v1'
OUT = HERE / 'common_candidate_fixed_fit_20261019_v1'
AUTH = HERE / 'contracts/COMMON_CANDIDATE_FIXED_FIT_AUTHORIZATION_20261019.json'
SCOPE = 'COMMON_CANDIDATE_266_FIXED_RESEARCH_V1'
CAP, RESERVE = 3*1024**2, 128*1024


def verify_manifest(root, status):
    stop = json.loads((root / 'stop_record.json').read_bytes())
    if stop['status'] != status or stop['manifest_sha256'] != sha(root / 'manifest.json'):
        raise ValueError('INCOMPLETE_OR_CHANGED_ARTIFACT:' + str(root))
    manifest = json.loads((root / 'manifest.json').read_bytes())
    for name, item in manifest['outputs'].items():
        if sha(root / name) != item['sha256']:
            raise ValueError('ARTIFACT_HASH_MISMATCH:' + name)


def authorize(record, config_sha, entry_sha, sidecar_sha):
    if record.get('status') != 'CHAT_APPROVED_ONE_FIXED_RESEARCH_FIT' or record.get('scope') != SCOPE:
        raise ValueError('SEPARATE_COMMON_FIT_AUTHORIZATION_REQUIRED')
    expected = dict(config_sha256=config_sha, entry_manifest_sha256=entry_sha,
                    sidecar_manifest_sha256=sidecar_sha)
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('AUTHORIZATION_HASH_BINDING_MISMATCH')
    if record.get('models') != ['ridge', 'lightgbm'] or record.get('fits_per_model') != 1:
        raise ValueError('FIXED_FIT_COUNT_REQUIRED')
    if not record.get('user_quote') or not record.get('approval_context'):
        raise ValueError('CHAT_APPROVAL_CONTEXT_REQUIRED')
    if any(record.get(k) is not False for k in ('http_authorized', 'deployment_authorized',
                                               'parameter_search_authorized', 'calibration_authorized')):
        raise ValueError('AUTHORIZATION_SCOPE_EXPANDED')


def load_accepted():
    verify_manifest(SIDECAR, 'COMPLETE_NOT_FIT')
    receipt = json.loads((RECEIPT / 'acceptance.json').read_bytes())
    receipt_stop = json.loads((RECEIPT / 'stop_record.json').read_bytes())
    if (receipt_stop['status'] != 'COMPLETE_NOT_FIT'
            or receipt_stop['receipt_sha256'] != sha(RECEIPT / 'acceptance.json')
            or receipt['status'] != 'PASS_SAVED_REAL_CANDIDATE_SIDECAR_NOT_FIT'
            or receipt['source_manifest_sha256'] != sha(SIDECAR / 'manifest.json')):
        raise ValueError('INDEPENDENT_REAL_SIDECAR_ACCEPTANCE_REQUIRED')
    config = json.loads((PREP / 'config.json').read_bytes())
    base = json.loads((previous.prep.OUT / 'config.json').read_bytes())
    if (config['fields'] != base['fields'] + learning_fields() or len(config['fields']) != 266
            or config['fit_authorized'] is not False):
        raise ValueError('FIXED_266_FIELD_SCHEMA_DRIFT')
    for key in ('models', 'segments', 'target', 'preprocessing', 'selection', 'calibration', 'evaluation'):
        if config[key] != base[key]:
            raise ValueError('PARENT_FIXED_PARAMETER_DRIFT:' + key)
    if {p: version(p) for p in config['package_versions']} != config['package_versions']:
        raise ValueError('RUNTIME_DRIFT')
    refs = dict(json.loads((SIDECAR / 'plan.json').read_bytes())['inputs'])
    for root, state in ((PREP, 'COMPLETE_PREPARATION_NOT_FIT'),
                        (previous.OUT, 'COMPLETE_NOT_DEPLOYED'),
                        (HERE / 'continuous_motion_fixed_fit_20261018_v1', 'COMPLETE_NOT_DEPLOYED')):
        verify_manifest(root, state)
        for name in ('manifest.json', 'stop_record.json'):
            refs[str(root / name)] = sha(root / name)
        for name, item in json.loads((root / 'manifest.json').read_bytes())['outputs'].items():
            refs[str(root / name)] = item['sha256']
    for root in (SIDECAR, RECEIPT):
        for path in root.iterdir():
            if path.is_file():
                refs[str(path)] = sha(path)
    for name in ('common_candidate_fixed_fit_v1.py', 'test_common_candidate_fixed_fit_v1.py',
                 'auxiliary_regression_fit_v1.py', 'auxiliary_regression_eval_v1.py',
                 'joint_fit_preparation_v1.py', 'research_budget_common_v3.py'):
        refs[str(HERE / name)] = sha(HERE / name)
    refs[str(ADDENDUM)] = sha(ADDENDUM)
    for path, expected in refs.items():
        if sha(path) != expected:
            raise ValueError('BOUND_INPUT_DRIFT:' + path)
    parts = previous.prep.load_parts(base)
    old = json.loads((PREP / 'report.json').read_bytes())
    for segment, frame in parts.items():
        labels = ['code', previous.prep.TIME, previous.prep.TARGET, 'label_end_time', 'auxiliary_end_time']
        if (frame_sha(frame[labels]) != old['segments'][segment]['keys_labels_sha256']
                or frame_sha(frame[base['fields']]) != old['segments'][segment]['original_232_sha256']):
            raise ValueError('OLD_KEYS_LABELS_OR_FEATURES_CHANGED')
    parts = attach_sidecars(parts, pd.read_parquet(SIDECAR / 'candidate_features.parquet'), base['fields'])
    new = json.loads((SIDECAR / 'report.json').read_bytes())
    for segment, frame in parts.items():
        for field in learning_fields():
            if frame_sha(frame[[field]]) != new['profiles'][segment][field]['sha256']:
                raise ValueError('REAL_NEW_COLUMN_FINGERPRINT_DRIFT:' + field)
    return config, parts, refs


def check_entry():
    if ENTRY.exists() or OUT.exists():
        raise ValueError('ENTRY_OR_FIT_EXISTS_NO_RESTART')
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromNames(
        ['test_common_candidate_fixed_fit_v1', 'test_common_candidate_preparation_v1']))
    if not tests.wasSuccessful():
        raise ValueError('ENTRY_TEST_FAILED:' + log.getvalue())
    config, parts, refs = load_accepted()
    if bytes_used() + CAP + 128*1024 > TOTAL_CAP:
        raise ValueError('FIT_RESERVATION_UNAVAILABLE')
    writer = EvidenceWriter(ENTRY, 128*1024, 16*1024)
    writer.json('acceptance.json', dict(status='PASS_COMMON_CANDIDATE_ENTRY_NO_FIT',
        refs=refs, config_sha256=sha(PREP / 'config.json'), sidecar_manifest_sha256=sha(SIDECAR / 'manifest.json'),
        rows={s: len(f) for s, f in parts.items()}, logical_fields=266, transformed_fields=532,
        tests=tests.testsRun, test_output=log.getvalue(), real_model_fits=0,
        training_authorized=False, proposed_fit_cap=CAP))
    writer.json('manifest.json', dict(outputs=writer.outputs.copy()))
    writer.json('stop_record.json', dict(status='COMPLETE_ENTRY_NOT_FIT',
                manifest_sha256=sha(ENTRY / 'manifest.json')), terminal=True)
    return dict(engineering_ready=True, fit_performed=False, entry_manifest_sha256=sha(ENTRY / 'manifest.json'))


def train(flag=False):
    if not flag:
        raise ValueError('EXPLICIT_FIXED_RESEARCH_FLAG_REQUIRED')
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    verify_manifest(ENTRY, 'COMPLETE_ENTRY_NOT_FIT')
    entry = json.loads((ENTRY / 'acceptance.json').read_bytes())
    if entry['status'] != 'PASS_COMMON_CANDIDATE_ENTRY_NO_FIT':
        raise ValueError('ENTRY_ACCEPTANCE_REQUIRED')
    record = json.loads(AUTH.read_bytes())
    authorize(record, sha(PREP / 'config.json'), sha(ENTRY / 'manifest.json'), sha(SIDECAR / 'manifest.json'))
    for path, expected in entry['refs'].items():
        if sha(path) != expected:
            raise ValueError('ENTRY_INPUT_DRIFT:' + path)
    config, parts, refs = load_accepted()
    for path in (AUTH, ENTRY / 'acceptance.json', ENTRY / 'manifest.json', ENTRY / 'stop_record.json'):
        refs[str(path)] = sha(path)
    writer = EvidenceWriter(OUT, CAP, RESERVE)
    writer.json('plan.json', dict(config=config, inputs=refs, authorization=record,
                                 proposed_models=2, output_cap=CAP))
    try:
        baseline = pd.read_parquet(previous.OUT / 'predictions.parquet')
        old_raw = {}
        for segment, frame in parts.items():
            saved = baseline[baseline.segment.eq(segment)].reset_index(drop=True)
            pd.testing.assert_frame_equal(frame[['code', previous.prep.TIME]], saved[['code', previous.prep.TIME]])
            old_raw[segment] = {m: saved[m+'_raw'].to_numpy() for m in ('ridge', 'lightgbm')}
        raw = {s: {} for s in parts}; fitted = {}; usage = {}
        for algorithm in ('ridge', 'lightgbm'):
            print(json.dumps(dict(stage='fixed_common_candidate_fit', algorithm=algorithm)), flush=True)
            result = previous.fit_model(parts, config['fields'], config, algorithm)
            state = {k: v.tolist() if isinstance(v, np.ndarray) else v
                     for k, v in result['preprocessing'].items()}
            writer.json(algorithm+'_preprocessing.json', state)
            if algorithm == 'ridge':
                writer.json('ridge_model.json', result['model'])
                model = json.loads((OUT / 'ridge_model.json').read_bytes())
            else:
                writer.write('lightgbm_model.txt.gz', gzip.compress(result['model'].encode('utf-8'), mtime=0))
                model = gzip.decompress((OUT / 'lightgbm_model.txt.gz').read_bytes()).decode('utf-8')
                counts = previous.lgb.Booster(model_str=model).feature_importance(importance_type='split')
                for i, name in enumerate(config['fields']):
                    if name in learning_fields():
                        usage[name] = dict(numeric_splits=int(counts[i]), missing_splits=int(counts[i+266]))
            restored = previous.replay(parts, config['fields'], algorithm,
                json.loads((OUT / (algorithm+'_preprocessing.json')).read_bytes()), model)
            for segment in parts:
                np.testing.assert_allclose(restored[segment], result['raw'][segment], atol=1e-11, rtol=0)
                raw[segment][algorithm] = restored[segment]
            fitted[algorithm] = dict(best_iteration=result['best_iteration'], warnings=result['warnings'], saved_replay='PASS')
        frozen_at = datetime.now(timezone.utc).isoformat()
        mean = float(parts['train'][previous.prep.TARGET].mean())
        median = float(parts['train'][previous.prep.TARGET].median())
        metrics, increment, tables = {}, {}, []
        for segment, frame in parts.items():
            metrics[segment] = evaluate(frame, raw[segment], mean, median, config)
            y = frame[previous.prep.TARGET].to_numpy(); dates = frame[previous.prep.TIME].to_numpy()//1000000
            increment[segment] = {m: paired_errors(y, np.clip(raw[segment][m], 0, 1),
                np.clip(old_raw[segment][m], 0, 1), dates, config['evaluation']['uncertainty'])
                for m in ('ridge', 'lightgbm')}
            table = frame[['code', previous.prep.TIME]].copy(); table['segment'] = segment
            for m in ('ridge', 'lightgbm'):
                table[m+'_raw'] = raw[segment][m]
            tables.append(table)
        sink = io.BytesIO(); pd.concat(tables, ignore_index=True).to_parquet(sink, index=False, compression='zstd')
        writer.write('predictions.parquet', sink.getvalue())
        writer.json('report.json', dict(status='COMPLETE_COMMON_CANDIDATE_FIXED_RESEARCH_NOT_PROMOTION',
            fitted=fitted, metrics=metrics, increment_vs_saved_232=increment, new_field_split_usage=usage,
            baselines=dict(train_mean=mean, train_median=median), model_weights_frozen_at=frozen_at,
            main_model_fits=2, calibration_fits=0, replayed_predictions=172638,
            test_reused=True, deployment=False, promotion_passed=False, http_requests=0,
            raw_minute_replays=0, amplitude_not_probability=True, pnl_evaluated=False))
        for path, expected in refs.items():
            if sha(path) != expected:
                raise ValueError('INPUT_CHANGED_DURING_FIT:' + path)
        writer.json('manifest.json', dict(outputs=writer.outputs.copy()))
        writer.json('stop_record.json', dict(status='COMPLETE_NOT_DEPLOYED',
                    manifest_sha256=sha(OUT / 'manifest.json')), terminal=True)
    except Exception as exc:
        writer.json('stop_record.json', dict(status='FAILED_NO_RESTART',
                    reason=f'{type(exc).__name__}:{exc}'), terminal=True)
        raise
    return dict(status='COMPLETE_NOT_DEPLOYED', rows=86319, logical_fields=266, bytes=writer.used)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true'); mode.add_argument('--train', action='store_true')
    parser.add_argument('--authorize-fixed-research', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check_entry() if args.check else train(args.authorize_fixed_research)))
