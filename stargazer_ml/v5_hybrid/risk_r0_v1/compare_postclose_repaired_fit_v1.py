"""Replay repair fit and compare on exact common keys, not mixed populations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from inventory_ths_stockdb_cache_v1 import digest, HERE
from postclose_repaired_training_v1 import OUT, FIRST, prepare, CAP
from audit_postclose_first_fit_v1 import loss, verify_rows, day_block_interval
from diagnose_postclose_predictions_v1 import filter_by_day, filter_summary
from research_budget_v2 import TOTAL_CAP, bytes_used

TARGET='next_session_adverse_excursion'


def partitions(old, new):
    if old[['code','day']].duplicated().any() or new[['code','day']].duplicated().any():
        raise ValueError('DUPLICATE_COMPARE_KEYS')
    joined=old.merge(new,on=['code','day'],suffixes=('_old','_new'),how='outer',indicator=True,validate='one_to_one')
    common=joined[joined._merge.eq('both')].copy()
    for field in [TARGET,'label_end_time']:
        if not np.array_equal(common[field+'_old'],common[field+'_new']):
            raise ValueError('COMMON_TARGET_OR_END_DRIFT')
    return common, new.merge(old[['code','day']],on=['code','day'],how='left',indicator=True).query('_merge == "left_only"').drop(columns='_merge'), old.merge(new[['code','day']],on=['code','day'],how='left',indicator=True).query('_merge == "left_only"').drop(columns='_merge')


def metrics(y, pred, constant):
    if not len(y): return dict(rows=0,pinball=None,coverage=None,constant_pinball=None)
    return dict(rows=len(y),pinball=float(loss(y,pred).mean()),coverage=float((np.asarray(y)<=np.asarray(pred)).mean()),
                constant_pinball=float(loss(y,constant).mean()))


def run():
    import lightgbm as lgb
    fields,policy,parts,refs=prepare()
    plan=json.loads((OUT/'plan.json').read_bytes())
    if plan['inputs']!=refs or plan['fields']!=fields or plan['policy']!=policy:
        raise ValueError('FIT_PLAN_DRIFT')
    original_names=['plan.json','report.json','preprocessing.json','linear_model.json','model.txt','predictions.parquet','stop_record.json']
    fingerprints={name:digest((OUT/name).read_bytes()) for name in original_names}
    if json.loads((OUT/'stop_record.json').read_bytes())['status']!='COMPLETE_RESEARCH_FIT_NOT_DEPLOYMENT_READY':
        raise ValueError('FIT_NOT_COMPLETE')
    report=json.loads((OUT/'report.json').read_bytes())
    state=json.loads((OUT/'preprocessing.json').read_bytes())
    if state['fields']!=fields: raise ValueError('PREPROCESSING_ORDER_DRIFT')
    train=parts['train'][fields].to_numpy(float); median=np.nanmedian(train,axis=0)
    filled=np.where(np.isnan(train),median,train); mean=filled.mean(axis=0); scale=filled.std(axis=0); scale=np.where(scale>0,scale,1.)
    for key,value in [('median',median),('mean',mean),('scale',scale)]: np.testing.assert_allclose(state[key],value,rtol=0,atol=1e-12)
    model=lgb.Booster(model_file=str(OUT/'model.txt'))
    linear=json.loads((OUT/'linear_model.json').read_bytes())
    predictions=pd.read_parquet(OUT/'predictions.parquet')
    verify_rows(predictions,pd.concat([f.assign(segment=s) for s,f in parts.items()],ignore_index=True))
    raw={}
    constant=float(np.quantile(parts['train'][TARGET],.9))
    for s,frame in parts.items():
        x=frame[fields].to_numpy(float); missing=np.isnan(x)
        raw[s]=model.predict(np.c_[x,missing.astype(float)],num_threads=2)
        lin=np.clip(np.c_[(np.where(missing,median,x)-mean)/scale,missing.astype(float)]@np.asarray(linear['coef'])+linear['intercept'],0,1)
        stored=predictions[predictions.segment.eq(s)]
        pred=np.clip(raw[s]+report['quantile_shift'],0,1)
        for c,value in [('q90_raw',raw[s]),('q90_adverse',pred),('q90_linear',lin)]: np.testing.assert_allclose(stored[c],value,rtol=0,atol=1e-12)
        calculated=metrics(frame[TARGET],pred,constant)
        for key,ref in [('pinball','pinball_q90'),('coverage','empirical_q90_coverage'),('constant_pinball','constant_pinball_q90')]:
            np.testing.assert_allclose(calculated[key],report['metrics'][s][ref],rtol=0,atol=1e-12)
        np.testing.assert_allclose(loss(frame[TARGET],lin).mean(),report['metrics'][s]['linear_pinball_q90'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(report['quantile_shift'],np.quantile(parts['calibration'][TARGET]-raw['calibration'],.9),rtol=0,atol=1e-12)
    if model.current_iteration()!=report['best_iteration']: raise ValueError('TREE_COUNT_DRIFT')
    oldall=pd.read_parquet(FIRST/'predictions.parquet')
    old=oldall[oldall.segment.eq('test')]; new=predictions[predictions.segment.eq('test')]
    common,added,removed=partitions(old,new)
    old_constant=float(np.quantile(oldall.loc[oldall.segment.eq('train'),TARGET],.9))
    y=common[TARGET+'_new']
    comparison=dict(common_rows=len(common),common_dates=int(common.day.nunique()),old_only_rows=len(removed),new_only_rows=len(added),
        old=metrics(y,common.q90_adverse_old,old_constant),new=metrics(y,common.q90_adverse_new,constant),
        constant_values=dict(old=old_constant,new=constant),
        new_only=metrics(added[TARGET],added.q90_adverse,constant),old_only=metrics(removed[TARGET],removed.q90_adverse,old_constant))
    diff=pd.DataFrame(dict(day=common.day,loss_difference=loss(y,common.q90_adverse_new)-loss(y,common.q90_adverse_old)))
    comparison['paired_pinball_difference_new_minus_old']=dict(mean=float(diff.loss_difference.mean()),**day_block_interval(diff))
    common_context=parts['test'][['code','day','stock_return1d']]
    filtering={}
    for version in ['old','new']:
        current=pd.DataFrame(dict(code=common.code,day=common.day,**{TARGET:y},q90_adverse=common['q90_adverse_'+version],q90_linear=common['q90_linear_'+version]))
        current=current.merge(common_context,on=['code','day'],validate='one_to_one')
        daily,coverage=filter_by_day(current,old_constant)
        filtering[version]=dict(coverage=coverage,result=filter_summary(daily))
    if filtering['old']['coverage']!=filtering['new']['coverage']: raise ValueError('FILTER_POPULATION_DRIFT')
    months={str(month):metrics(g[TARGET],g.q90_adverse,constant) for month,g in new.groupby(new.day.astype(str).str[:6])}
    result=dict(status='PASS_REPLAY_AND_COMMON_KEY_COMPARISON_NOT_DEPLOYMENT',new_fit_sha256=fingerprints,
        old_fit_unchanged=True,comparison=comparison,common_same_day_filter=filtering,new_test_months=months,
        actual_refit_in_audit=False,independent_blind_test=False,deployment_authorized=False)
    body=json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8')
    for p,sha in refs.items():
        if digest(Path(p).read_bytes())!=sha: raise ValueError('INPUT_CHANGED_DURING_AUDIT')
    if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())+len(body)>CAP or bytes_used()+len(body)>TOTAL_CAP: raise ValueError('AUDIT_BUDGET_STOP')
    with (OUT/'comparison_and_readback.json').open('xb') as stream: stream.write(body)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__': run()
