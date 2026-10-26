"""Saved-weight replay, complete fold reporting and common-key D2/D3 comparison."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fit_postclose_temporal_v1 import OUT,D2,frozen_parts,CAP
from audit_postclose_first_fit_v1 import loss,verify_rows,day_block_interval
from diagnose_postclose_predictions_v1 import strata,filter_by_day,filter_summary
from compare_postclose_repaired_fit_v1 import partitions
from inventory_ths_stockdb_cache_v1 import digest
from research_budget_v2 import TOTAL_CAP,bytes_used

TARGET='next_session_adverse_excursion'


def run():
    import lightgbm as lgb
    fields,policy,folds,refs=frozen_parts()
    plan=json.loads((OUT/'plan.json').read_bytes())
    if plan['inputs']!=refs or plan['fields']!=fields or plan['policy']!=policy:raise ValueError('BATCH_PLAN_DRIFT')
    if json.loads((OUT/'stop_record.json').read_bytes())['status']!='COMPLETE_RESEARCH_BATCH_NOT_DEPLOYMENT':raise ValueError('BATCH_NOT_COMPLETE')
    outputs={};results={};predictions={}
    for fold in ['D1','D2','D3']:
        directory=D2 if fold=='D2' else OUT/fold
        names=['model.txt','linear_model.json','preprocessing.json','predictions.parquet','report.json','stop_record.json']
        outputs[fold]={n:digest((directory/n).read_bytes()) for n in names}
        parts=folds[fold];report=json.loads((directory/'report.json').read_bytes())
        state=json.loads((directory/'preprocessing.json').read_bytes())
        if state['fields']!=fields:raise ValueError('PREPROCESSING_ORDER_DRIFT')
        x=parts['train'][fields].to_numpy(float);median=np.nanmedian(x,axis=0)
        filled=np.where(np.isnan(x),median,x);mean=filled.mean(axis=0);scale=filled.std(axis=0);scale=np.where(scale>0,scale,1.)
        for key,value in [('median',median),('mean',mean),('scale',scale)]:np.testing.assert_allclose(state[key],value,rtol=0,atol=1e-12)
        model=lgb.Booster(model_file=str(directory/'model.txt'));linear=json.loads((directory/'linear_model.json').read_bytes())
        p=pd.read_parquet(directory/'predictions.parquet')
        verify_rows(p,pd.concat([f.assign(segment=s) for s,f in parts.items()],ignore_index=True))
        raw={};constant=float(np.quantile(parts['train'][TARGET],.9))
        for s,f in parts.items():
            x=f[fields].to_numpy(float);missing=np.isnan(x)
            raw[s]=model.predict(np.c_[x,missing.astype(float)],num_threads=2)
            predicted=np.clip(raw[s]+report['quantile_shift'],0,1)
            lin=np.clip(np.c_[(np.where(missing,median,x)-mean)/scale,missing.astype(float)]@np.asarray(linear['coef'])+linear['intercept'],0,1)
            stored=p[p.segment.eq(s)]
            for c,v in [('q90_raw',raw[s]),('q90_adverse',predicted),('q90_linear',lin)]:np.testing.assert_allclose(stored[c],v,rtol=0,atol=1e-12)
            metrics=report['metrics'][s]
            for c,v in [('pinball_q90',loss(f[TARGET],predicted).mean()),('linear_pinball_q90',loss(f[TARGET],lin).mean()),
                        ('constant_pinball_q90',loss(f[TARGET],constant).mean()),('empirical_q90_coverage',(f[TARGET]<=predicted).mean())]:
                np.testing.assert_allclose(metrics[c],v,rtol=0,atol=1e-12)
        np.testing.assert_allclose(report['quantile_shift'],np.quantile(parts['calibration'][TARGET]-raw['calibration'],.9),rtol=0,atol=1e-12)
        if model.current_iteration()!=report['best_iteration']:raise ValueError('TREE_COUNT_DRIFT')
        test=p[p.segment.eq('test')].merge(parts['test'][['code','day','stock_return1d']],on=['code','day'],validate='one_to_one')
        predictions[fold]=test
        delta=pd.DataFrame(dict(day=test.day,loss_difference=loss(test[TARGET],test.q90_adverse)-loss(test[TARGET],constant)))
        monthly={}
        for month,g in test.groupby(test.day.astype(str).str[:6]):
            monthly[str(month)]=dict(rows=len(g),dates=int(g.day.nunique()),pinball=float(loss(g[TARGET],g.q90_adverse).mean()),
                constant_pinball=float(loss(g[TARGET],constant).mean()),coverage=float((g[TARGET]<=g.q90_adverse).mean()))
        edges=np.quantile(p.loc[p.segment.eq('train'),'q90_adverse'],[.2,.4,.6,.8])
        daily,coverage=filter_by_day(test,constant)
        results[fold]=dict(reused=fold=='D2',best_iteration=report['best_iteration'],train_constant_q90=constant,
            shift=report['quantile_shift'],test_metrics=report['metrics']['test'],
            raw_test_pinball=float(loss(test[TARGET],test.q90_raw).mean()),
            relative_pinball_improvement_vs_constant=float(1-report['metrics']['test']['pinball_q90']/report['metrics']['test']['constant_pinball_q90']),
            monthly=monthly,fixed_train_strata=strata(test,edges,constant),
            model_minus_constant=dict(mean=float(delta.loss_difference.mean()),**day_block_interval(delta)),
            same_day_filter=dict(coverage=coverage,results=filter_summary(daily)))
    common,added,removed=partitions(predictions['D2'],predictions['D3'])
    diff=pd.DataFrame(dict(day=common.day,loss_difference=loss(common[TARGET+'_new'],common.q90_adverse_new)-loss(common[TARGET+'_old'],common.q90_adverse_old)))
    pair=dict(rows=len(common),dates=int(common.day.nunique()),D3_only=len(added),D2_only=len(removed),
        D2_pinball=float(loss(common[TARGET+'_old'],common.q90_adverse_old).mean()),D3_pinball=float(loss(common[TARGET+'_new'],common.q90_adverse_new).mean()),
        D2_coverage=float((common[TARGET+'_old']<=common.q90_adverse_old).mean()),D3_coverage=float((common[TARGET+'_new']<=common.q90_adverse_new).mean()),
        paired_difference=dict(mean=float(diff.loss_difference.mean()),**day_block_interval(diff)),filters={})
    for name,suffix in [('D2','old'),('D3','new')]:
        frame=pd.DataFrame(dict(code=common.code,day=common.day,**{TARGET:common[TARGET+'_'+suffix]},
            q90_adverse=common['q90_adverse_'+suffix],q90_linear=common['q90_linear_'+suffix],stock_return1d=common['stock_return1d_'+suffix]))
        daily,coverage=filter_by_day(frame,results['D2']['train_constant_q90'])
        pair['filters'][name]=dict(coverage=coverage,metrics=filter_summary(daily))
    for p,sha in refs.items():
        if digest(Path(p).read_bytes())!=sha:raise ValueError('INPUT_CHANGED_DURING_EVALUATION')
    result=dict(status='PASS_SAVED_MODELS_REPLAY_HISTORICAL_ROBUSTNESS_ONLY',folds=results,
        common_D2_D3=pair,artifact_hashes=outputs,independent_blind_test=False,deployment_authorized=False,actual_refit_in_audit=False)
    body=json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8')
    if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())+len(body)>CAP or bytes_used()+len(body)>TOTAL_CAP:raise ValueError('EVALUATION_BUDGET_STOP')
    with (OUT/'evaluation_and_readback.json').open('xb') as stream:stream.write(body)
    print(json.dumps({f:{k:v for k,v in r.items() if k not in ['monthly','same_day_filter','fixed_train_strata']} for f,r in results.items()},ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in pair.items() if k!='filters'},ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':run()
