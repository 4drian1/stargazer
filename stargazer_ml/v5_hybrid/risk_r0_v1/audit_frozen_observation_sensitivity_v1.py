"""Independent coefficient/text inference, sparse completeness and direct loss checks."""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

import evaluate_frozen_observation_sensitivity_v1 as build
from read_conditional_board_observations_v2 import read_segment


def infer(frame,fields,algorithm,conditional):
    prefix='M1_'+algorithm
    state=json.loads((build.FIT/(prefix+'_preprocessing.json')).read_bytes())
    calibration=json.loads((build.FIT/(prefix+'_calibration.json')).read_bytes())
    columns=[('ar_'+f) if conditional and 'ar_'+f in frame else f for f in fields]
    x=np.column_stack([frame[c].to_numpy(dtype=float) for c in columns])
    missing=np.isnan(x)
    if algorithm=='lr':
        z=(np.where(missing,np.array(state['median']),x)-np.array(state['mean']))/np.array(state['scale'])
        transformed=np.column_stack([z,missing.astype(float)])
        model=json.loads((build.FIT/(prefix+'_model.json')).read_bytes())
        logits=transformed@np.asarray(model['coef'])+model['intercept']
        raw=expit(logits)
    else:
        transformed=np.column_stack([x,missing.astype(float)])
        booster=lgb.Booster(model_file=str(build.FIT/(prefix+'_model.txt')))
        raw=booster.predict(transformed,num_threads=2)
        logits=None
    calibrated=expit(logit(np.clip(raw,1e-12,1-1e-12))*calibration['coef'][0]+calibration['intercept'])
    return dict(raw=raw,calibrated=calibrated),transformed,logits


def run():
    destination=build.HERE/'frozen_observation_sensitivity_acceptance_20261018_v1.json'
    if destination.exists():
        raise ValueError('ACCEPTANCE_EXISTS_NO_OVERWRITE')
    report=json.loads((build.OUT/'report.json').read_bytes())
    terminal=json.loads((build.OUT/'stop_record.json').read_bytes())
    manifest=json.loads((build.OUT/'manifest.json').read_bytes())
    if terminal['status']!='COMPLETE_NOT_FIT_OR_DEPLOYMENT' or build.prep.sha(build.OUT/'manifest.json')!=terminal['manifest_sha256']:
        raise ValueError('INVALID_TERMINAL')
    for path,expected in report['inputs'].items():
        if build.prep.sha(path)!=expected:
            raise ValueError('INPUT_DRIFT:'+path)
    for name,item in manifest['outputs'].items():
        if build.prep.sha(build.OUT/name)!=item['sha256']:
            raise ValueError('OUTPUT_DRIFT:'+name)
    population=pd.read_parquet(build.prep.OUT/'population_audit.parquet')
    stored=pd.read_parquet(build.FIT/'predictions.parquet')
    differences=pd.read_parquet(build.OUT/'prediction_differences.parquet')
    config=json.loads((build.prep.OUT/'config.json').read_bytes())
    fields=config['arms']['M1']
    checked=0;delta_count=0;max_error=0.;cases=[]
    with threadpool_limits(limits=2):
        for segment in build.prep.SEGMENTS:
            full=read_segment(segment)
            bound=population[population.proposed_segment_name.eq(segment)].reset_index(drop=True)
            pd.testing.assert_frame_equal(full[['code',build.TIME]],bound[['code',build.TIME]])
            frame=full.loc[bound.common_main_eligible.to_numpy()].reset_index(drop=True)
            target=stored[stored.segment.eq(segment)].reset_index(drop=True)
            y=frame.y_tail_1200m.to_numpy(dtype=float)
            for algorithm in ['lr','lightgbm']:
                old,x0,z0=infer(frame,fields,algorithm,False)
                conditional,x1,z1=infer(frame,fields,algorithm,True)
                for state in ['raw','calibrated']:
                    a,b=old[state],conditional[state]
                    np.testing.assert_allclose(a,target[f'M1_{algorithm}_{state}'],atol=1e-11,rtol=0)
                    take=np.abs(b-a)>1e-12
                    item=differences[(differences.segment==segment)&(differences.algorithm==algorithm)&(differences.state==state)]
                    pd.testing.assert_frame_equal(item[['code',build.TIME]].reset_index(drop=True),
                                                  frame.loc[take,['code',build.TIME]].reset_index(drop=True))
                    np.testing.assert_allclose(item.original_probability,a[take],atol=1e-11,rtol=0)
                    np.testing.assert_allclose(item.conditional_probability,b[take],atol=1e-11,rtol=0)
                    delta_count+=len(item);checked+=len(a)*2
                    metrics=report['segments'][segment]['models'][algorithm]['states'][state]
                    for values,name in [(a,'original'),(b,'conditional')]:
                        clipped=np.clip(values,1e-12,1-1e-12)
                        actual=dict(brier=float(np.mean((values-y)**2)),
                            log_loss=float(np.mean(-y*np.log(clipped)-(1-y)*np.log1p(-clipped))),
                            actual_minus_predicted=float(y.mean()-values.mean()))
                        for metric,value in actual.items():
                            error=abs(value-metrics[name][metric]);max_error=max(max_error,error)
                            if error>1e-12:
                                raise ValueError('DIRECT_METRIC_MISMATCH')
                    if segment in ['train','select']:
                        np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
                if segment=='test' and algorithm=='lr':
                    coef=np.array(json.loads((build.FIT/'M1_lr_model.json').read_bytes())['coef'])
                    contributions=(x1-x0)*coef
                    np.testing.assert_allclose(contributions.sum(axis=1),z1-z0,atol=1e-11,rtol=0)
                    order=np.argsort(-np.abs(conditional['raw']-old['raw']),kind='stable')[:5]
                    for i in order:
                        top=np.argsort(-np.abs(contributions[i]),kind='stable')[:5]
                        case=dict(code=frame.code.iloc[i],decision_ts=int(frame[build.TIME].iloc[i]),
                            original_probability=float(old['raw'][i]),conditional_probability=float(conditional['raw'][i]),
                            logit_delta=float(z1[i]-z0[i]),
                            numeric_logit_delta=float(contributions[i,:len(fields)].sum()),
                            missing_indicator_logit_delta=float(contributions[i,len(fields):].sum()),
                            top_terms=[])
                        for position in top:
                            field=fields[position%len(fields)]
                            before=frame[field].iloc[i]
                            after=frame['ar_'+field].iloc[i] if 'ar_'+field in frame else before
                            case['top_terms'].append(dict(field=field,kind='numeric' if position<len(fields) else 'missing_indicator',
                                old_value=float(before) if pd.notna(before) else None,
                                new_value=float(after) if pd.notna(after) else None,
                                coefficient=float(coef[position]),logit_delta=float(contributions[i,position])))
                        cases.append(case)
    if checked!=86319*8 or delta_count!=report['prediction_difference_cells']:
        raise ValueError('PREDICTION_REPLAY_COUNT_MISMATCH')
    old_manifest=json.loads((build.FIT/'artifact_manifest.json').read_bytes())
    for name,item in old_manifest['outputs'].items():
        if build.prep.sha(build.FIT/name)!=item['sha256']:
            raise ValueError('OLD_MODEL_CHANGED')
    d2=build.HERE/'postclose_repaired_research_fit_20261017_v1/model.txt'
    if build.prep.sha(d2)!='5d8402cd530cb83778d901e33eb87788f21ebc8619414f335b6d56905a81e005':
        raise ValueError('D2_CHANGED')
    result=dict(status='PASS_FIXED_MODEL_INPUT_SENSITIVITY_NOT_FIT',tests=report['tests'],
        independent_probability_values=checked,sparse_prediction_cells=delta_count,
        maximum_direct_metric_error=max_error,old_model_artifacts_unchanged=len(old_manifest['outputs']),d2_unchanged=True,
        report_sha256=build.prep.sha(build.OUT/'report.json'),
        auditor_sha256=build.prep.sha(build.HERE/'audit_frozen_observation_sensitivity_v1.py'),
        posthoc_largest_probability_change_cases_not_used_for_feature_selection=cases,
        fit=False,recalibration=False,http_requests=0)
    body=build.prep.encoded(result)
    used=sum(p.stat().st_size for p in build.OUT.iterdir() if p.is_file())
    if used+len(body)+build.RESERVE>build.CAP or build.bytes_used()+len(body)+build.RESERVE>build.TOTAL_CAP:
        raise ValueError('ACCEPTANCE_BUDGET_STOP')
    with destination.open('xb') as stream:
        stream.write(body)
    print(json.dumps({k:v for k,v in result.items() if not k.startswith('posthoc_')}))


if __name__=='__main__':
    run()
