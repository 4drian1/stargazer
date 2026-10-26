"""No-raw-minute replay of structural coverage and saved-model error limits."""
import io
import gzip
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import continuous_motion_fixed_fit_v1 as fit
from materialize_relative_stall_pilot_v1 import sha
from motion_diagnostic_math_v1 import static_membership,old_linear_contributions,outcome_mix
from multiboard_three_arm_fit_v1 import EvidenceWriter

HERE=fit.HERE;OUT=HERE/'continuous_motion_limits_20261018_v2'
CAP,RESERVE=512*1024,32*1024


def run():
    writer=EvidenceWriter(OUT,CAP,RESERVE)
    frozen=json.loads((fit.OUT/'plan.json').read_bytes());refs=dict(frozen['inputs'])
    for name in ['plan.json','feature_acceptance.json','report.json','predictions.parquet',
                 'ridge_preprocessing.json','ridge_model.json','stop_record.json','manifest.json']:
        refs[str(fit.OUT/name)]=sha(fit.OUT/name)
    for name in ['motion_diagnostic_math_v1.py','test_motion_diagnostic_math_v1.py','diagnose_motion_limits_v1.py',
                 'continuous_motion_fit_acceptance_20261018_v1.json']:
        refs[str(HERE/name)]=sha(HERE/name)
    # Only existing derived evidence is consumed. No bound raw-minute files opened.
    for path,expected in refs.items():
        if sha(path)!=expected:raise ValueError('INPUT_DRIFT:'+path)
    writer.json('plan.json',dict(inputs=refs,cap=CAP,raw_minute_reads=0,http_requests=0,fit=False,
        scope='STRUCTURAL_FROZEN_UNIVERSE_AND_SAVED_RIDGE_BLOCKS_AND_LGBM_MONTHLY_ERRORS',
        no_admission_policy_change=True,new_feature_block_is_algebraic_residual=True))
    try:
        log=io.StringIO();tests=unittest.TextTestRunner(stream=log,verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromName('test_motion_diagnostic_math_v1'))
        if not tests.wasSuccessful():raise ValueError(log.getvalue())
        base=json.loads((fit.previous.prep.OUT/'config.json').read_bytes())
        parts=fit.previous.prep.load_parts(base)
        target=fit.previous.prep.TARGET;time=fit.previous.prep.TIME
        predictions=pd.read_parquet(fit.OUT/'predictions.parquet')
        accept=json.loads((fit.OUT/'feature_acceptance.json').read_bytes())
        spec=json.loads((HERE/'ths_full_concept_scope_20261017_v1.json').read_bytes())
        if sha(spec['source'])!=spec['source_sha256']:raise ValueError('CATALOG_DRIFT')
        boards=[b for b in pq.read_table(spec['source']).to_pylist() if b['source']=='ths' and b['type']=='concept']
        bound={r['code'] for r in json.loads((HERE/'relative_stall_5m_20261018_v1/raw_input_ledger.json').read_bytes())}
        codes=sorted({c for f in parts.values() for c in f.code})
        structure=static_membership(boards,bound,codes)
        population=pd.concat([f[['code',time]].assign(segment=s) for s,f in parts.items()],ignore_index=True)
        counts=population.code.value_counts().to_dict()
        for code,r in structure.items():r['rows']=int(counts[code])
        blocked=[c for c,r in structure.items() if r['related'] and not r['potentially_complete']]
        static_blocked=int(population.code.isin(blocked).sum())
        if static_blocked!=82753 or len(blocked)!=27:raise ValueError('STATIC_DIAGNOSTIC_EXPECTATION_DRIFT')
        state=json.loads((fit.OUT/'ridge_preprocessing.json').read_bytes())
        model=json.loads((fit.OUT/'ridge_model.json').read_bytes());beta=np.asarray(model['coef'])
        intercept=float(model['intercept']);n=len(base['fields'])
        train=parts['train'];train_x=train[base['fields']].to_numpy(dtype=float)
        train_min=np.nanmin(train_x,axis=0);train_max=np.nanmax(train_x,axis=0)
        result={};error_rows=[]
        for s,frame in parts.items():
            pred=predictions[predictions.segment.eq(s)].reset_index(drop=True)
            pd.testing.assert_frame_equal(pred[['code',time]],frame[['code',time]])
            contribution,z=old_linear_contributions(frame,base['fields'],state,beta)
            known=contribution.sum(axis=1);raw=pred.ridge_raw.to_numpy()
            residual=raw-intercept-known
            # Identity, not an independent reconstruction of the new 118 values.
            np.testing.assert_allclose(intercept+known+residual,raw,atol=1e-12,rtol=0)
            means=contribution.mean(axis=0);x=frame[base['fields']].to_numpy(dtype=float)
            drift=[]
            for i,name in enumerate(base['fields']):
                valid=np.isfinite(x[:,i]);outside=valid&((x[:,i]<train_min[i])|(x[:,i]>train_max[i]))
                drift.append(dict(field=name,mean_score_contribution=float(means[i]),
                    train_min=float(train_min[i]),train_max=float(train_max[i]),
                    observed_fraction=float(valid.mean()),outside_train_fraction=float(outside.mean()),
                    mean_standardized_value=float(z[:,i].mean()),
                    saved_scale=float(state['scale'][i]),numeric_coefficient=float(beta[i])))
            monthly={}
            for month,index in frame.groupby(frame[time]//100000000).groups.items():
                idx=np.asarray(index,dtype=int)
                monthly[str(month)]=dict(rows=len(idx),raw_predicted_mean=float(raw[idx].mean()),
                    physical_predicted_mean=float(np.clip(raw[idx],0,1).mean()),actual_mean=float(frame[target].iloc[idx].mean()),
                    intercept=intercept,original_232_mean_contribution=float(known[idx].mean()),
                    added_118_mean_residual=float(residual[idx].mean()),
                    top_original_mean_contributions=sorted([
                        dict(field=name,mean_score_contribution=float(contribution[idx,i].mean()))
                        for i,name in enumerate(base['fields'])],key=lambda v:abs(v['mean_score_contribution']),reverse=True)[:8])
            result[s]=dict(rows=len(frame),static_blocked_rows=int(frame.code.isin(blocked).sum()),
                original_232_mean_contribution=float(known.mean()),added_118_mean_residual=float(residual.mean()),
                intercept=intercept,raw_predicted_mean=float(raw.mean()),actual_mean=float(frame[target].mean()),
                original_feature_drift=drift,monthly=monthly)
            temp=frame[['code',time,target]].copy();temp['segment']=s
            temp['predicted']=pred.lightgbm_raw.to_numpy().clip(0,1);error_rows.append(temp)
        all_errors=pd.concat(error_rows,ignore_index=True)
        monthly_errors={}
        for (segment,month),g in all_errors.groupby(['segment',all_errors[time]//100000000],sort=True):
            y=g[target].to_numpy();p=g.predicted.to_numpy();daily=g.assign(error=p-y).groupby(g[time]//1000000).error.mean()
            per_code=[]
            for c,h in g.groupby('code'):
                per_code.append(dict(code=c,rows=len(h),actual_mean=float(h[target].mean()),
                    predicted_mean=float(h.predicted.mean()),bias=float((h.predicted-h[target]).mean())))
            monthly_errors[f'{segment}_{month}']=dict(rows=len(g),dates=len(daily),codes=len(per_code),
                row_weighted_bias=float((p-y).mean()),date_equal_bias=float(daily.mean()),
                predicted_quantiles=np.quantile(p,[0,.1,.5,.9,1]).tolist(),
                target_quantiles=np.quantile(y,[0,.1,.5,.9,1]).tolist(),per_code=per_code,
                train_code_mixture=outcome_mix(train.code,train[target],g.code,g[target]))
        tiny=[dict(field=name,scale=float(v)) for name,v in zip(state['columns'],state['scale']) if v<1e-10]
        report=dict(status='COMPLETE_READONLY_DIAGNOSIS_NO_FIT',tests=tests.testsRun,test_output=log.getvalue(),
            rows=len(population),boards=len(boards),boards_with_unbound_member=sum(not(set(b['symbols'])<=bound) for b in boards),
            unbound_member_codes=len({c for b in boards for c in b['symbols']}-bound),
            structurally_blocked_candidate_codes=blocked,structurally_blocked_rows=static_blocked,
            static_membership_by_code=structure,saved_finite_profiles=accept['profiles'],
            ridge_blocks=result,ridge_tiny_scales_below_1e_10=tiny,lgbm_monthly_errors=monthly_errors,
            exact_dynamic_missing_slot_attribution=False,raw_minute_reads=0,http_requests=0,fit=False,
            altered_admission=False,new_feature_block_is_residual_not_independent_replay=True,
            causal=False,new_forecast_rule=False)
        body=json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8')
        writer.write('report.json.gz',gzip.compress(body,mtime=0))
        for path,expected in refs.items():
            if sha(path)!=expected:raise ValueError('INPUT_CHANGED:'+path)
        writer.json('manifest.json',dict(outputs=writer.outputs.copy()))
        writer.json('stop_record.json',dict(status='COMPLETE_NO_FIT',manifest_sha256=sha(OUT/'manifest.json')),terminal=True)
        return dict(status=report['status'],rows=len(population),static_blocked_rows=static_blocked,
            blocked_codes=len(blocked),bytes=writer.used)
    except Exception as exc:
        writer.json('stop_record.json',dict(status='FAILED_NO_RESTART',reason=str(exc)),terminal=True)
        raise


if __name__=='__main__':print(json.dumps(run()))
