"""Explicit operator-authorized restricted research fit; never a production deploy."""
from pathlib import Path
import argparse
import io
import json
import importlib.util
import numpy as np
import pandas as pd
from inventory_ths_stockdb_cache_v1 import HERE,digest
from finalize_postclose_cohort_v3 import OUT,read_population
from research_budget_v2 import bytes_used,TOTAL_CAP

CAP,RESERVE=8*1024**2,1024**2


def preprocessing(frame):
    x=frame.to_numpy(dtype=float)
    if np.isinf(x).any() or np.isnan(x).all(axis=0).any():raise ValueError('INVALID_TRAIN_FEATURES')
    median=np.nanmedian(x,axis=0);filled=np.where(np.isnan(x),median,x)
    mean=filled.mean(axis=0);scale=filled.std(axis=0);scale=np.where(scale>0,scale,1.)
    return dict(fields=list(frame),median=median.tolist(),mean=mean.tolist(),scale=scale.tolist())


def transform(frame,state,linear):
    if list(frame)!=state['fields']:raise ValueError('FEATURE_ORDER_DRIFT')
    x=frame.to_numpy(dtype=float)
    if np.isinf(x).any():raise ValueError('INFINITE_INPUT')
    missing=np.isnan(x)
    if linear:x=(np.where(missing,state['median'],x)-state['mean'])/state['scale']
    return np.c_[x,missing.astype(float)]


def pinball(y,pred,alpha=.9):
    residual=np.asarray(y)-np.asarray(pred)
    return float(np.maximum(alpha*residual,(alpha-1)*residual).mean())


def check():
    data,manifest=read_population()
    for path,sha in manifest['inputs'].items():
        if digest(Path(path).read_bytes())!=sha:raise ValueError('FROZEN_INPUT_DRIFT:'+path)
    policy=json.loads((HERE/'postclose_research_policy_v1.json').read_bytes())
    parts={s:data[data.research_cohort_eligible&data.proposed_segment_name.eq(s)].copy()
           for s in policy['sample_gates']}
    fields=manifest['fields']
    forbidden=['mae','target','next_session','label','code','reference','available','eligible','end_time']
    if len(fields)!=len(set(fields)) or any(any(word in f for word in forbidden) for f in fields):raise ValueError('UNSAFE_WHITELIST')
    if any('15m' in f or '5m' in f or 'macd_dea' in f for f in fields):raise ValueError('NON_DAILY_OR_REDUNDANT_INPUT')
    if not manifest['engineering_ready']:raise ValueError('SAMPLE_GATE_NOT_MET')
    for name,frame in parts.items():
        threshold=policy['sample_gates'][name]
        actual=dict(rows=len(frame),dates=int(frame.day.nunique()),stocks=int(frame.code.nunique()))
        if not all(actual[k]>=v for k,v in threshold.items()):raise ValueError('SAMPLE_GATE_DRIFT')
        if frame[['code','day']].duplicated().any():raise ValueError('DUPLICATE_SAMPLE')
        if not (frame.label_end_time>frame.decision_ts_assumed_end).all():raise ValueError('TARGET_NOT_FUTURE')
        if not frame.next_session_adverse_excursion.between(0,1,inclusive='left').all():raise ValueError('INVALID_TARGET')
        if np.isinf(frame[fields].to_numpy(dtype=float)).any():raise ValueError('INFINITE_FEATURES')
    if parts['train'][fields].isna().all().any():raise ValueError('ALL_NULL_TRAIN_FIELD')
    if any(importlib.util.find_spec(m) is None for m in ['lightgbm','sklearn']):raise ValueError('DEPENDENCY_MISSING')
    return data,manifest,policy,parts


def train(*,authorize=False):
    if not authorize:raise ValueError('EXPLICIT_RESTRICTED_RESEARCH_FIT_AUTHORIZATION_REQUIRED')
    destination=HERE/'postclose_first_research_fit_20261017_v1'
    if destination.exists():raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    data,manifest,policy,parts=check()
    if bytes_used()+CAP>TOTAL_CAP:raise ValueError('FIT_OUTPUT_BUDGET_RESERVATION_FAILED')
    # Freeze requested configuration and all input fingerprints before any real-data fit.
    refs={str(OUT/'manifest.json'):digest((OUT/'manifest.json').read_bytes()),
          manifest['population']['path']:manifest['population']['sha256'],
          str(OUT/'eligibility.parquet'):manifest['outputs']['eligibility.parquet']['sha256'],
          str(Path(__file__)):digest(Path(__file__).read_bytes()),
          str(HERE/'postclose_research_policy_v1.json'):digest((HERE/'postclose_research_policy_v1.json').read_bytes())}
    destination.mkdir(exist_ok=False);used=0
    def write(name,body,terminal=False):
        nonlocal used
        reserve=0 if terminal else RESERVE
        if used+len(body)+reserve>CAP or bytes_used()+len(body)+reserve>TOTAL_CAP:raise ValueError('FIT_OUTPUT_BUDGET_STOP')
        with (destination/name).open('xb') as stream:stream.write(body)
        used+=len(body)
    plan=dict(status='OPERATOR_CLI_AUTHORIZED_RESTRICTED_RESEARCH_ONLY',
        authorization='--authorize-restricted-research explicitly passed by operator',
        population_limitations=['limited daily-evidence stocks','archived membership hindsight','partial raw leader/sector context',
                                'unknown historical arrivals','amount activity not share volume','reseal unknown'],
        target='next-session adverse excursion Q90, NOT risk probability or fill/PnL',
        fields=manifest['fields'],inputs=refs,policy=policy,output_cap=CAP)
    write('plan.json',json.dumps(plan,ensure_ascii=False,allow_nan=False,indent=2).encode())
    try:
        from sklearn.linear_model import QuantileRegressor
        import lightgbm as lgb
        fields=manifest['fields'];state=preprocessing(parts['train'][fields])
        x={s:transform(f[fields],state,False) for s,f in parts.items()}
        linear_x={s:transform(f[fields],state,True) for s,f in parts.items()}
        y={s:f.next_session_adverse_excursion.to_numpy() for s,f in parts.items()}
        baseline=QuantileRegressor(quantile=.9,alpha=.01,solver='highs')
        baseline.fit(linear_x['train'],y['train'])
        parameters=dict(policy['primary_model_proposed']);parameters.pop('name')
        model=lgb.LGBMRegressor(**parameters,verbosity=-1,deterministic=True,force_col_wise=True)
        model.fit(x['train'],y['train'],eval_set=[(x['select'],y['select'])],
                  callbacks=[lgb.early_stopping(50,verbose=False)])
        constant=float(np.quantile(y['train'],.9));raw={s:model.predict(x[s]) for s in parts}
        delta=float(np.quantile(y['calibration']-raw['calibration'],.9))
        report={};predictions=[]
        for s,f in parts.items():
            pred=np.clip(raw[s]+delta,0,1)
            lin=np.clip(baseline.predict(linear_x[s]),0,1)
            report[s]=dict(rows=len(f),dates=int(f.day.nunique()),stocks=int(f.code.nunique()),
                pinball_q90=pinball(y[s],pred),linear_pinball_q90=pinball(y[s],lin),
                constant_pinball_q90=pinball(y[s],np.full(len(f),constant)),
                empirical_q90_coverage=float((y[s]<=pred).mean()))
            p=f[['code','day','label_end_time','next_session_adverse_excursion']].copy()
            p['segment']=s;p['q90_adverse']=pred;p['q90_raw']=raw[s];p['q90_linear']=lin
            predictions.append(p)
        for path,sha in refs.items():
            if digest(Path(path).read_bytes())!=sha:raise ValueError('FROZEN_INPUT_CHANGED_DURING_FIT')
        write('preprocessing.json',json.dumps(state,allow_nan=False,indent=2).encode())
        write('model.txt',model.booster_.model_to_string().encode())
        write('linear_model.json',json.dumps(dict(coef=baseline.coef_.tolist(),intercept=float(baseline.intercept_)),allow_nan=False).encode())
        sink=io.BytesIO();pd.concat(predictions,ignore_index=True).to_parquet(sink,index=False,compression='zstd')
        write('predictions.parquet',sink.getvalue())
        write('report.json',json.dumps(dict(metrics=report,quantile_shift=delta,best_iteration=model.best_iteration_,
            actual_fit=True,production_gate_passed=False,independent_blind_test=False,deployment_authorized=False),allow_nan=False,indent=2).encode())
        write('stop_record.json',json.dumps(dict(status='COMPLETE_RESEARCH_FIT_NOT_DEPLOYMENT_READY',bytes_before_stop=used)).encode(),True)
    except Exception as exc:
        write('stop_record.json',json.dumps(dict(status='FAILED',error=f'{type(exc).__name__}:{exc}')).encode(),True)
        raise
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--authorize-restricted-research',action='store_true')
    args=parser.parse_args()
    if args.train:
        if args.check:raise ValueError('CHOOSE_CHECK_OR_TRAIN')
        print(json.dumps(train(authorize=args.authorize_restricted_research)))
    else:
        _,manifest,_,_=check()
        print(json.dumps(dict(engineering_ready=True,fit_authorized=False,fit_run=False,
            target='next-session adverse excursion Q90, not probability',fields=len(manifest['fields']),summary=manifest['summary'])))


if __name__=='__main__':main()
