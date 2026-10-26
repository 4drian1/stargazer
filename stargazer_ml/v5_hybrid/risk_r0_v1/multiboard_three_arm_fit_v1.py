"""One chat-authorized fixed historical comparison; immutable inputs and outputs."""
import argparse
import gc
from importlib.metadata import version
import io
import json
from pathlib import Path
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import prepare_multiboard_comparison_v1 as prep
from joint_fit_preparation_v1 import fit_lr_preprocessing, transform_lr, native_missing_view
from multiboard_fit_evaluation_v1 import evaluate
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = prep.HERE
OUT = HERE / 'multiboard_three_arm_fit_20261018_v1'
AUTH = HERE / 'contracts/MULTIBOARD_FIXED_FIT_CHAT_AUTHORIZATION_20261018.json'
ENTRY = HERE / 'multiboard_fit_entry_acceptance_20261018_v1.json'
CONFIG_SHA = '1ea339247bcf3f09fc7f3e8634e283c4e8b2b815e9f9ff82ca34c434a105a9ab'
CAP, RESERVE = prep.FIT_CAP, 128 * 1024


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    return value


class EvidenceWriter:
    def __init__(self, directory=OUT, cap=CAP, reserve=RESERVE):
        self.root, self.cap, self.reserve = Path(directory), cap, reserve
        self.used, self.outputs = 0, {}
        if self.root.exists():
            raise ValueError('OUTPUT_EXISTS_NO_RESTART')
        if bytes_used() + cap > TOTAL_CAP:
            raise ValueError('REGISTERED_FIT_BUDGET_RESERVATION_FAILED')

    def write(self, name, body, terminal=False):
        if Path(name).name != name or name in self.outputs:
            raise ValueError('INVALID_OR_DUPLICATE_EVIDENCE_NAME')
        reserve = 0 if terminal else self.reserve
        if self.used + len(body) + reserve > self.cap or bytes_used() + len(body) + reserve > TOTAL_CAP:
            raise ValueError('FIT_OUTPUT_BUDGET_STOP')
        if not self.root.exists():
            self.root.mkdir(exist_ok=False)
        with (self.root / name).open('xb') as target:
            target.write(body)
        self.used += len(body)
        self.outputs[name] = dict(bytes=len(body), sha256=prep.sha(self.root / name))

    def json(self, name, value, terminal=False):
        self.write(name, prep.encoded(serializable(value)), terminal=terminal)


def positive(model, values):
    if list(model.classes_) != [0, 1]:
        raise ValueError('CLASS_ORDER_NOT_BINARY_0_1')
    result = model.predict_proba(values)[:, 1]
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError('INVALID_MODEL_PROBABILITY')
    return result


def linear_record(model):
    return dict(coef=model.coef_[0].tolist(), intercept=float(model.intercept_[0]), classes=[0, 1])


def fit_algorithm(parts, fields, config, algorithm):
    """Only train/select fit the main model, only calibration fits sigmoid."""
    y = {s: parts[s].y_tail_1200m.to_numpy() for s in prep.SEGMENTS}
    state = fit_lr_preprocessing(parts['train'][fields])
    x = {s: transform_lr(parts[s][fields], state) if algorithm == 'lr'
         else native_missing_view(parts[s][fields]) for s in prep.SEGMENTS}
    with warnings.catch_warnings(record=True) as observed, threadpool_limits(limits=2):
        warnings.simplefilter('always')
        if algorithm == 'lr':
            model = LogisticRegression(**config['models']['lr'])
            model.fit(x['train'], y['train'])
            artifact = linear_record(model)
            best_iteration = None
        elif algorithm == 'lightgbm':
            model = lgb.LGBMClassifier(**config['models']['lightgbm'])
            model.fit(x['train'], y['train'], eval_set=[(x['select'], y['select'])],
                eval_metric=config['selection']['lightgbm_metric'],
                callbacks=[lgb.early_stopping(config['selection']['lightgbm_early_stopping_rounds'], verbose=False)])
            artifact = model.booster_.model_to_string(num_iteration=model.best_iteration_)
            best_iteration = int(model.best_iteration_)
        else:
            raise ValueError('UNKNOWN_ALGORITHM')
        raw = {s: positive(model, values) for s, values in x.items()}
        epsilon = config['calibration']['epsilon']
        calibration = LogisticRegression(**config['calibration']['estimator'])
        calibration.fit(logit(np.clip(raw['calibration'], epsilon, 1-epsilon)).reshape(-1, 1), y['calibration'])
        calibrated = {s: positive(calibration, logit(np.clip(values, epsilon, 1-epsilon)).reshape(-1, 1))
                      for s, values in raw.items()}
    messages = [dict(category=type(w.message).__name__, message=str(w.message)) for w in observed]
    if any(issubclass(w.category, ConvergenceWarning) for w in observed):
        raise ValueError('FIXED_MODEL_DID_NOT_CONVERGE_NO_AUTOMATIC_RETUNE:' + json.dumps(messages))
    return dict(raw=raw, calibrated=calibrated, preprocessing=state, model=artifact,
                calibration=linear_record(calibration), best_iteration=best_iteration, warnings=messages)


def replay(parts, fields, algorithm, preprocessing, model, calibration, epsilon):
    """Reconstruct predictions using stored coefficients/text, not fitted objects."""
    state = dict(preprocessing)
    for key in ['median', 'mean', 'scale']:
        state[key] = np.asarray(state[key], dtype=float)
    predictor = lgb.Booster(model_str=model) if algorithm == 'lightgbm' else None
    outputs = {}
    with threadpool_limits(limits=2):
        for segment, frame in parts.items():
            x = transform_lr(frame[fields], state) if algorithm == 'lr' else native_missing_view(frame[fields])
            raw = expit(x @ np.asarray(model['coef']) + model['intercept']) if predictor is None else predictor.predict(x, num_threads=2)
            calibrated = expit(logit(np.clip(raw, epsilon, 1-epsilon)) * calibration['coef'][0] + calibration['intercept'])
            outputs[segment] = dict(raw=raw, calibrated=calibrated)
    return outputs


def check_authorization(record, config_sha, entry_sha):
    if record.get('status') != 'CHAT_APPROVED_ONE_FIXED_RESEARCH_FIT' or record.get('scope') != 'MULTIBOARD_THREE_ARM_RESEARCH_V1':
        raise ValueError('FIXED_RESEARCH_FIT_AUTHORIZATION_REQUIRED')
    if record.get('config_sha256') != config_sha or record.get('entry_acceptance_sha256') != entry_sha:
        raise ValueError('AUTHORIZATION_HASH_BINDING_MISMATCH')
    if record.get('user_quote') != '好，请继续' or not record.get('approval_context'):
        raise ValueError('AUTHORIZATION_CHAT_CONTEXT_REQUIRED')
    if any(record.get(k) is not False for k in ['new_http_authorized', 'deployment_authorized', 'parameter_search_authorized']):
        raise ValueError('AUTHORIZATION_SCOPE_EXPANDED')


def preflight(authorize=False):
    checked = prep.check()
    if not checked['preparation_ready'] or checked['config_sha256'] != CONFIG_SHA:
        raise ValueError('FIXED_PREPARATION_OR_CONFIGURATION_DRIFT')
    accepted = json.loads((HERE / 'multiboard_comparison_preparation_acceptance_20261018_v1.json').read_bytes())
    refs = dict(accepted['refs'])
    refs[str(HERE / 'multiboard_comparison_preparation_acceptance_20261018_v1.json')] = prep.sha(HERE / 'multiboard_comparison_preparation_acceptance_20261018_v1.json')
    refs.update(json.loads((prep.OUT / 'report.json').read_bytes())['inputs'])
    config = json.loads((prep.OUT / 'config.json').read_bytes())
    if {p: version(p) for p in config['package_versions']} != config['package_versions']:
        raise ValueError('FROZEN_RUNTIME_VERSION_DRIFT')
    for filename in ['multiboard_three_arm_fit_v1.py', 'multiboard_fit_evaluation_v1.py',
                     'test_multiboard_three_arm_fit_v1.py', 'audit_multiboard_three_arm_fit_v1.py']:
        refs[str(HERE / filename)] = prep.sha(HERE / filename)
    if authorize:
        entry = json.loads(ENTRY.read_bytes())
        if entry['status'] != 'PASS_FIXED_FIT_ENTRY_NO_REAL_FIT':
            raise ValueError('ENTRY_ACCEPTANCE_REQUIRED')
        for path, expected in entry['refs'].items():
            if prep.sha(path) != expected:
                raise ValueError('ENTRY_ACCEPTANCE_DRIFT:' + path)
        record = json.loads(AUTH.read_bytes())
        check_authorization(record, CONFIG_SHA, prep.sha(ENTRY))
        refs[str(AUTH)], refs[str(ENTRY)] = prep.sha(AUTH), prep.sha(ENTRY)
    for path, expected in refs.items():
        if prep.sha(path) != expected:
            raise ValueError('FROZEN_INPUT_DRIFT:' + path)
    masks = pd.read_parquet(prep.OUT / 'population_audit.parquet')
    parts = {}
    boundaries = {row['name']: row for row in config['segments']}
    for segment in prep.SEGMENTS:
        full = prep.read_segment(segment)
        stored = masks[masks.proposed_segment_name.eq(segment)].reset_index(drop=True)
        pd.testing.assert_frame_equal(full[['code', prep.TIME]], stored[['code', prep.TIME]])
        common, _ = prep.common_population(full, config['core_observation_groups'])
        np.testing.assert_array_equal(common, stored.common_main_eligible)
        prep.validate_times_labels(full, segment, boundaries)
        parts[segment] = full.loc[common].reset_index(drop=True)
        if prep.summarize(full, common) != checked['segments'][segment]['common_main']:
            raise ValueError('COMMON_POPULATION_DRIFT')
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if bytes_used() + CAP > TOTAL_CAP:
        raise ValueError('REGISTERED_FIT_BUDGET_RESERVATION_FAILED')
    return config, parts, refs


def train(authorize=False):
    if not authorize:
        raise ValueError('EXPLICIT_FIXED_RESEARCH_AUTHORIZATION_REQUIRED')
    config, parts, refs = preflight(authorize=True)
    writer = EvidenceWriter()
    writer.json('plan.json', dict(config_sha256=CONFIG_SHA, config=config, inputs=refs,
        authorization='USER_CHAT_CONTINUE_FIXED_RESEARCH_ONLY', output_cap=CAP,
        runtime={p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'pyarrow', 'scipy', 'threadpoolctl']},
        limitations=['snapshot membership hindsight', 'unknown original historical arrivals',
            'restricted ordinary action evidence', 'reseal evidence unknown', 'old test reused for research',
            'labels overlap across minutes and stocks', 'primary common subset is not full population']))
    try:
        scores = {s: {} for s in prep.SEGMENTS}
        fitted = {}
        for arm, fields in config['arms'].items():
            for algorithm in ['lr', 'lightgbm']:
                print(json.dumps(dict(stage='fit_start', arm=arm, algorithm=algorithm, fields=len(fields))), flush=True)
                result = fit_algorithm(parts, fields, config, algorithm)
                prefix = arm + '_' + algorithm
                writer.json(prefix + '_preprocessing.json', result['preprocessing'])
                writer.json(prefix + '_calibration.json', result['calibration'])
                if algorithm == 'lr':
                    writer.json(prefix + '_model.json', result['model'])
                else:
                    writer.write(prefix + '_model.txt', result['model'].encode('utf-8'))
                restored = replay(parts, fields, algorithm,
                    json.loads((OUT / (prefix + '_preprocessing.json')).read_bytes()),
                    json.loads((OUT / (prefix + '_model.json')).read_bytes()) if algorithm == 'lr'
                    else (OUT / (prefix + '_model.txt')).read_text(encoding='utf-8'),
                    json.loads((OUT / (prefix + '_calibration.json')).read_bytes()), config['calibration']['epsilon'])
                for segment in prep.SEGMENTS:
                    for state in ['raw', 'calibrated']:
                        np.testing.assert_allclose(result[state][segment], restored[segment][state], atol=1e-11, rtol=0)
                        scores[segment][prefix + '_' + state] = result[state][segment]
                fitted[prefix] = dict(best_iteration=result['best_iteration'], warnings=result['warnings'],
                    calibration_slope=result['calibration']['coef'][0],
                    negative_calibration_slope_is_not_silently_repaired=result['calibration']['coef'][0] < 0,
                    serialization_replay='PASS')
                print(json.dumps(dict(stage='fit_and_replay_complete', name=prefix, best_iteration=result['best_iteration'])), flush=True)
                del result, restored
                gc.collect()
        predictions, reports = [], {}
        train_rate = float(parts['train'].y_tail_1200m.mean())
        for segment, frame in parts.items():
            print(json.dumps(dict(stage='risk_evaluation', segment=segment)), flush=True)
            reports[segment] = evaluate(frame, scores[segment], train_rate, config)
            item = frame[['code', prep.TIME, 'label_end_time', 'y_tail_1200m', 'mae_1200m']].copy()
            item['segment'] = segment
            for name, values in scores[segment].items():
                item[name] = values
            predictions.append(item)
        sink = io.BytesIO()
        pd.concat(predictions, ignore_index=True).to_parquet(sink, index=False, compression='zstd')
        writer.write('predictions.parquet', sink.getvalue())
        writer.json('report.json', dict(status='COMPLETE_FIXED_HISTORICAL_RESEARCH_NOT_PROMOTION',
            actual_fit=True, fitted=fitted, metrics=reports, config_sha256=CONFIG_SHA,
            train_constant_risk=train_rate, primary_pair='M1_minus_B1', golden_pair='B1_minus_B0',
            train_and_calibration_metrics_are_in_sample_for_their_fitting_steps=True,
            new_blind_test=False, promotion_gate_passed=False, deployment_authorized=False, pnl_evaluated=False))
        for path, expected in refs.items():
            if prep.sha(path) != expected:
                raise ValueError('INPUT_CHANGED_DURING_FIT:' + path)
        writer.json('artifact_manifest.json', dict(outputs=writer.outputs.copy()))
        writer.json('stop_record.json', dict(status='COMPLETE_RESEARCH_NOT_DEPLOYMENT',
            manifest_sha256=prep.sha(OUT / 'artifact_manifest.json'), bytes_before_stop=writer.used), terminal=True)
        return dict(status='COMPLETE_RESEARCH_NOT_DEPLOYMENT', models=6, rows=sum(map(len, parts.values())), bytes=writer.used)
    except Exception as exc:
        writer.json('stop_record.json', dict(status='FAILED_NO_AUTOMATIC_RETRY',
            error=f'{type(exc).__name__}:{exc}', bytes_before_stop=writer.used), terminal=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--train', action='store_true')
    parser.add_argument('--authorize-fixed-research', action='store_true')
    args = parser.parse_args()
    if args.train:
        answer = train(authorize=args.authorize_fixed_research)
    else:
        _, parts, _ = preflight(authorize=args.authorize_fixed_research)
        answer = dict(engineering_ready=True, fit_performed=False,
            fit_authorized=args.authorize_fixed_research, common_rows={s: len(f) for s, f in parts.items()})
    print(json.dumps(answer, ensure_ascii=False))
