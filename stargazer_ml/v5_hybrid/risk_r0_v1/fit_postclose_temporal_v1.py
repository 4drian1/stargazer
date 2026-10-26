"""Two fixed research fits; D2 is hash-bound reuse, never refitted here."""
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from inventory_ths_stockdb_cache_v1 import HERE,digest
from postclose_repaired_training_v1 import prepare,OUT as D2
from read_postclose_daily_context_repair_v1 import read_repaired_population
from postclose_training_v1 import preprocessing,transform,pinball
from prepare_postclose_temporal_validation_v1 import OUT as PREFLIGHT,SPEC
from research_budget_v2 import TOTAL_CAP,bytes_used

OUT=HERE/'postclose_temporal_research_fit_20261017_v1'
AUTH=HERE/'contracts/POSTCLOSE_TEMPORAL_FIT_AUTHORIZATION_20261017.json'
CAP=8*1024**2


def frozen_parts():
    fields,policy,d2,refs=prepare()
    pre=json.loads((PREFLIGHT/'report.json').read_bytes())
    accepted=json.loads((PREFLIGHT/'readback_acceptance.json').read_bytes())
    if not accepted['D2_exact_reuse_accepted'] or digest((PREFLIGHT/'report.json').read_bytes())!=accepted['preflight_report_sha256']:
        raise ValueError('PREFLIGHT_NOT_ACCEPTED')
    refs.update(pre['inputs'])
    refs.update({str(PREFLIGHT/n):e['sha256'] for n,e in pre['output_hashes'].items()})
    for p in [AUTH,SPEC,PREFLIGHT/'report.json',PREFLIGHT/'readback_acceptance.json',Path(__file__),
              HERE/'test_postclose_temporal_fit_v1.py',HERE/'evaluate_postclose_temporal_v1.py',D2/'comparison_and_readback.json']:
        refs[str(p)]=digest(p.read_bytes())
    reuse=json.loads((D2/'comparison_and_readback.json').read_bytes())
    refs.update({str(D2/n):sha for n,sha in reuse['new_fit_sha256'].items()})
    for p,sha in refs.items():
        if digest(Path(p).read_bytes())!=sha:raise ValueError('FROZEN_INPUT_DRIFT:'+p)
    data,_=read_repaired_population();folds={}
    for name in ['D1','D2','D3']:
        assignment=pd.read_parquet(PREFLIGHT/(name+'_assignment.parquet'))
        pd.testing.assert_frame_equal(assignment[['code','day']],data[['code','day']])
        parts={s:data[assignment.eligible&assignment.segment.eq(s)].copy() for s in policy['sample_gates']}
        for s,f in parts.items():
            facts=dict(rows=len(f),dates=int(f.day.nunique()),stocks=int(f.code.nunique()))
            if not all(facts[k]>=v for k,v in policy['sample_gates'][s].items()):raise ValueError('NUMERIC_GATE_DRIFT')
            if f[['code','day']].duplicated().any() or not (f.label_end_time>f.decision_ts_assumed_end).all():raise ValueError('KEY_OR_TIME_DRIFT')
            end=json.loads(SPEC.read_bytes())['folds'][name][s][1]
            if not f.label_end_time.le(end*1000000+235959).all():raise ValueError('CROSS_BOUNDARY_LABEL')
            if np.isinf(f[fields].to_numpy(float)).any() or not f.next_session_adverse_excursion.between(0,1,inclusive='left').all():raise ValueError('INVALID_FEATURE_OR_TARGET')
        if parts['train'][fields].isna().all().any():raise ValueError('ALL_NULL_TRAIN_FEATURE')
        if name=='D2':
            for s in parts:pd.testing.assert_frame_equal(parts[s],d2[s])
        folds[name]=parts
    return fields,policy,folds,refs


def fit_one(parts,fields,policy):
    import lightgbm as lgb
    from sklearn.linear_model import QuantileRegressor
    state=preprocessing(parts['train'][fields])
    x={s:transform(f[fields],state,False) for s,f in parts.items()}
    lx={s:transform(f[fields],state,True) for s,f in parts.items()}
    y={s:f.next_session_adverse_excursion.to_numpy() for s,f in parts.items()}
    linear=QuantileRegressor(quantile=.9,alpha=.01,solver='highs');linear.fit(lx['train'],y['train'])
    params=dict(policy['primary_model_proposed']);params.pop('name')
    model=lgb.LGBMRegressor(**params,verbosity=-1,deterministic=True,force_col_wise=True)
    model.fit(x['train'],y['train'],eval_set=[(x['select'],y['select'])],callbacks=[lgb.early_stopping(50,verbose=False)])
    raw={s:model.predict(x[s]) for s in parts}
    shift=float(np.quantile(y['calibration']-raw['calibration'],.9));constant=float(np.quantile(y['train'],.9))
    predictions=[];metrics={}
    for s,f in parts.items():
        pred=np.clip(raw[s]+shift,0,1);lin=np.clip(linear.predict(lx[s]),0,1)
        metrics[s]=dict(rows=len(f),dates=int(f.day.nunique()),stocks=int(f.code.nunique()),
            pinball_q90=pinball(y[s],pred),linear_pinball_q90=pinball(y[s],lin),
            constant_pinball_q90=pinball(y[s],np.full(len(f),constant)),empirical_q90_coverage=float((y[s]<=pred).mean()))
        p=f[['code','day','label_end_time','next_session_adverse_excursion']].copy()
        p['segment']=s;p['q90_adverse']=pred;p['q90_raw']=raw[s];p['q90_linear']=lin;predictions.append(p)
    sink=io.BytesIO();pd.concat(predictions,ignore_index=True).to_parquet(sink,index=False,compression='zstd')
    report=dict(metrics=metrics,quantile_shift=shift,train_constant_q90=constant,best_iteration=model.best_iteration_,
        actual_fit=True,production_gate_passed=False,independent_blind_test=False,deployment_authorized=False)
    bodies={'model.txt':model.booster_.model_to_string().encode(),'predictions.parquet':sink.getvalue()}
    for n,value in [('preprocessing.json',state),('linear_model.json',dict(coef=linear.coef_.tolist(),intercept=float(linear.intercept_))),('report.json',report)]:
        bodies[n]=json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8')
    return bodies,report


def run(authorize=False):
    if not authorize:raise ValueError('EXPLICIT_TEMPORAL_FIT_AUTHORIZATION_REQUIRED')
    if OUT.exists():raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    fields,policy,folds,refs=frozen_parts()
    if bytes_used()+CAP>TOTAL_CAP:raise ValueError('BATCH_BUDGET_RESERVATION_FAILED')
    auth=json.loads(AUTH.read_bytes())
    if auth['new_fits']!=['D1','D3']:raise ValueError('AUTHORIZED_SCOPE_DRIFT')
    OUT.mkdir(exist_ok=False)
    def write(name,body,terminal=False):
        reserve=0 if terminal else 1024**2
        if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())+len(body)+reserve>CAP or bytes_used()+len(body)+reserve>TOTAL_CAP:raise ValueError('TEMPORAL_BUDGET_STOP')
        path=OUT/name;path.parent.mkdir(exist_ok=True)
        with path.open('xb') as stream:stream.write(body)
    write('plan.json',json.dumps(dict(inputs=refs,fields=fields,policy=policy,authorization=auth,D2_reuse=str(D2)),ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8'))
    complete=[]
    try:
        summary={'D2':dict(reused=True,path=str(D2),report=json.loads((D2/'report.json').read_bytes()))}
        for name in ['D1','D3']:
            bodies,report=fit_one(folds[name],fields,policy)
            for file,body in bodies.items():write(name+'/'+file,body)
            write(name+'/stop_record.json',b'{"status":"COMPLETE_RESEARCH_FIT"}',True)
            complete.append(name);summary[name]=dict(reused=False,report=report)
            print(json.dumps(dict(fold=name,actual_fit=True,report=report)),flush=True)
        for p,sha in refs.items():
            if digest(Path(p).read_bytes())!=sha:raise ValueError('INPUT_CHANGED_DURING_FIT')
        write('summary.json',json.dumps(summary,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8'))
        write('stop_record.json',json.dumps(dict(status='COMPLETE_RESEARCH_BATCH_NOT_DEPLOYMENT',new_fits=complete,reused=['D2'])).encode(),True)
        return summary
    except Exception as exc:
        write('stop_record.json',json.dumps(dict(status='FAILED_NO_AUTOMATIC_CONTINUATION',completed_folds=complete,error=f'{type(exc).__name__}:{exc}')).encode(),True)
        raise


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--authorize-temporal-research-fits',action='store_true')
    run(p.parse_args().authorize_temporal_research_fits)
