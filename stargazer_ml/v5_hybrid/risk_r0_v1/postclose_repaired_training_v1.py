"""Single authorized repair-only fit; first fit and its frozen entry stay unchanged."""
import argparse
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from inventory_ths_stockdb_cache_v1 import HERE, digest
from postclose_training_v1 import check as original_check, preprocessing, transform, pinball, CAP, RESERVE
from read_postclose_daily_context_repair_v1 import read_repaired_population
from research_budget_v2 import TOTAL_CAP, bytes_used

OUT = HERE / 'postclose_repaired_research_fit_20261017_v1'
REPAIR = HERE / 'postclose_daily_context_repair_20261017_v1'
FIRST = HERE / 'postclose_first_research_fit_20261017_v1'
AUTH = HERE / 'contracts/POSTCLOSE_REPAIRED_FIT_AUTHORIZATION_20261017.json'


def prepare():
    _, original, policy, _ = original_check()
    data, manifest = read_repaired_population()
    fields = original['fields']
    accepted = json.loads((REPAIR / 'readback_acceptance.json').read_bytes())
    if accepted['status'] != 'PASS_DATA_REPLAY_NOT_FIT_AUTHORIZATION':
        raise ValueError('REPAIR_NOT_ACCEPTED')
    refs = dict(manifest['inputs'])
    for name, sha in accepted['artifact_sha256'].items():
        refs[str(REPAIR / name)] = sha
    refs[str(HERE / 'read_postclose_daily_context_repair_v1.py')] = accepted['reader_sha256']
    first_audit = json.loads((FIRST / 'independent_readback_audit.json').read_bytes())
    refs.update({str(FIRST / p): h for p,h in first_audit['audited_original_sha256'].items()})
    for path in [AUTH, REPAIR / 'readback_acceptance.json', Path(__file__),
                 HERE/'postclose_training_v1.py', HERE/'postclose_research_policy_v1.json',
                 HERE/'compare_postclose_repaired_fit_v1.py', HERE/'test_postclose_repaired_fit_v1.py']:
        refs[str(path)] = digest(path.read_bytes())
    for path, sha in refs.items():
        if digest(Path(path).read_bytes()) != sha:
            raise ValueError('FROZEN_INPUT_DRIFT:' + path)
    first_plan = json.loads((FIRST/'plan.json').read_bytes())
    for path, sha in first_plan['inputs'].items():
        if digest(Path(path).read_bytes()) != sha:
            raise ValueError('FIRST_FIT_INPUT_DRIFT:' + path)
    if fields != first_plan['fields'] or policy != first_plan['policy']:
        raise ValueError('PARAMETER_OR_FIELD_DRIFT')
    parts = {s:data[data.research_cohort_eligible & data.proposed_segment_name.eq(s)].copy() for s in policy['sample_gates']}
    for segment, frame in parts.items():
        actual = dict(rows=len(frame), dates=int(frame.day.nunique()), stocks=int(frame.code.nunique()))
        if not all(actual[k] >= v for k,v in policy['sample_gates'][segment].items()):
            raise ValueError('SAMPLE_GATE_FAILED')
        if frame[['code','day']].duplicated().any() or not (frame.label_end_time > frame.decision_ts_assumed_end).all():
            raise ValueError('INVALID_SAMPLE_TIME_OR_KEY')
        if not frame.next_session_adverse_excursion.between(0,1,inclusive='left').all() or np.isinf(frame[fields].to_numpy(float)).any():
            raise ValueError('INVALID_TARGET_OR_FEATURE')
    if parts['train'][fields].isna().all().any():
        raise ValueError('ALL_NULL_TRAIN_FIELD')
    return fields, policy, parts, refs


def train(authorize=False):
    if not authorize:
        raise ValueError('EXPLICIT_REPAIR_RESEARCH_FIT_AUTHORIZATION_REQUIRED')
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    fields, policy, parts, refs = prepare()
    if bytes_used() + CAP > TOTAL_CAP:
        raise ValueError('FIT_OUTPUT_BUDGET_RESERVATION_FAILED')
    OUT.mkdir(exist_ok=False)
    def write(name, body, terminal=False):
        reserve = 0 if terminal else RESERVE
        current = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        if current + len(body) + reserve > CAP or bytes_used() + len(body) + reserve > TOTAL_CAP:
            raise ValueError('FIT_OUTPUT_BUDGET_STOP')
        with (OUT/name).open('xb') as stream: stream.write(body)
    plan = dict(status='AUTHORIZED_RESTRICTED_DATA_REPAIR_FIT', inputs=refs, fields=fields,
        policy=policy, target='next-session adverse excursion Q90, not probability or trading PnL',
        comparison='common/new-only/old-only test keys separately; examined test not blind')
    write('plan.json',json.dumps(plan,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8'))
    try:
        import lightgbm as lgb
        from sklearn.linear_model import QuantileRegressor
        state = preprocessing(parts['train'][fields])
        x = {s:transform(f[fields],state,False) for s,f in parts.items()}
        linear_x = {s:transform(f[fields],state,True) for s,f in parts.items()}
        y = {s:f.next_session_adverse_excursion.to_numpy() for s,f in parts.items()}
        baseline = QuantileRegressor(quantile=.9,alpha=.01,solver='highs')
        baseline.fit(linear_x['train'],y['train'])
        parameters = dict(policy['primary_model_proposed']); parameters.pop('name')
        model = lgb.LGBMRegressor(**parameters,verbosity=-1,deterministic=True,force_col_wise=True)
        model.fit(x['train'],y['train'],eval_set=[(x['select'],y['select'])],callbacks=[lgb.early_stopping(50,verbose=False)])
        raw = {s:model.predict(x[s]) for s in parts}
        delta = float(np.quantile(y['calibration']-raw['calibration'],.9))
        constant = float(np.quantile(y['train'],.9))
        predictions, metrics = [], {}
        for s, frame in parts.items():
            pred = np.clip(raw[s]+delta,0,1)
            linear = np.clip(baseline.predict(linear_x[s]),0,1)
            metrics[s] = dict(rows=len(frame),dates=int(frame.day.nunique()),stocks=int(frame.code.nunique()),
                pinball_q90=pinball(y[s],pred),linear_pinball_q90=pinball(y[s],linear),
                constant_pinball_q90=pinball(y[s],np.full(len(frame),constant)),
                empirical_q90_coverage=float((y[s]<=pred).mean()))
            p = frame[['code','day','label_end_time','next_session_adverse_excursion']].copy()
            p['segment']=s; p['q90_adverse']=pred; p['q90_raw']=raw[s]; p['q90_linear']=linear
            predictions.append(p)
        for path, sha in refs.items():
            if digest(Path(path).read_bytes()) != sha: raise ValueError('INPUT_CHANGED_DURING_FIT')
        write('preprocessing.json',json.dumps(state,allow_nan=False,indent=2).encode())
        write('model.txt',model.booster_.model_to_string().encode())
        write('linear_model.json',json.dumps(dict(coef=baseline.coef_.tolist(),intercept=float(baseline.intercept_)),allow_nan=False).encode())
        sink=io.BytesIO(); pd.concat(predictions,ignore_index=True).to_parquet(sink,index=False,compression='zstd')
        write('predictions.parquet',sink.getvalue())
        report=dict(metrics=metrics,quantile_shift=delta,train_constant_q90=constant,best_iteration=model.best_iteration_,
            actual_fit=True,production_gate_passed=False,independent_blind_test=False,deployment_authorized=False)
        write('report.json',json.dumps(report,allow_nan=False,indent=2).encode())
        write('stop_record.json',json.dumps(dict(status='COMPLETE_RESEARCH_FIT_NOT_DEPLOYMENT_READY')).encode(),True)
        return report
    except Exception as exc:
        write('stop_record.json',json.dumps(dict(status='FAILED',error=f'{type(exc).__name__}:{exc}')).encode(),True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--train',action='store_true'); parser.add_argument('--authorize-restricted-research',action='store_true')
    args=parser.parse_args()
    if args.train: print(json.dumps(train(args.authorize_restricted_research)))
    else:
        fields,_,parts,_=prepare()
        print(json.dumps(dict(engineering_ready=True,fields=len(fields),rows={s:len(f) for s,f in parts.items()},fit_run=False)))
