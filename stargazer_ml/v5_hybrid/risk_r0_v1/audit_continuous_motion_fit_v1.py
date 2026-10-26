"""Independent persisted-artifact / target / incremental-error check. No FIT."""
import gzip
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

import continuous_motion_fixed_fit_v1 as fit
from materialize_relative_stall_pilot_v1 import sha
from research_budget_v2 import bytes_used,TOTAL_CAP


def run():
    out=fit.OUT
    stop=json.loads((out/'stop_record.json').read_bytes())
    if stop['status']!='COMPLETE_NOT_DEPLOYED':raise ValueError('FIXED_FIT_NOT_COMPLETE')
    if sha(out/'manifest.json')!=stop['manifest_sha256']:raise ValueError('TERMINAL_MANIFEST_DRIFT')
    manifest=json.loads((out/'manifest.json').read_bytes())
    for name,item in manifest['outputs'].items():
        if sha(out/name)!=item['sha256']:raise ValueError('ARTIFACT_DRIFT:'+name)
    plan=json.loads((out/'plan.json').read_bytes());config=plan['config']
    for path,expected in plan['inputs'].items():
        if sha(path)!=expected:raise ValueError('INPUT_CHANGED:'+path)
    accept=json.loads((out/'feature_acceptance.json').read_bytes())
    if accept['status']!='PASS_NOT_FIT_YET' or accept['all_null_train']:raise ValueError('FEATURE_ACCEPTANCE_MISSING')
    assert len(accept['column_fingerprints'])==118*4
    assert len(config['fields'])==350 and len(set(config['fields']))==350
    names=config['fields']+[n+'__missing' for n in config['fields']]
    text=gzip.decompress((out/'lightgbm_model.txt.gz').read_bytes()).decode()
    tree=lgb.Booster(model_str=text)
    assert tree.num_feature()==700
    coefficients=json.loads((out/'ridge_model.json').read_bytes())
    assert len(coefficients['coef'])==700 and np.isfinite(coefficients['coef']).all()
    splits=tree.feature_importance('split')
    used=[dict(field=n,splits=int(v)) for n,v in zip(names,splits) if v]
    new=[v for v in used if v['field'].startswith('cm_')]
    predictions=pd.read_parquet(out/'predictions.parquet')
    old=pd.read_parquet(fit.previous.OUT/'predictions.parquet')
    target=pd.read_parquet(fit.previous.prep.auxiliary.OUT/'labels.parquet')
    report=json.loads((out/'report.json').read_bytes());checks=0
    for segment in fit.previous.prep.SEGMENTS:
        p=predictions[predictions.segment.eq(segment)].reset_index(drop=True)
        b=old[old.segment.eq(segment)].reset_index(drop=True)
        y=target[target.segment.eq(segment)].reset_index(drop=True)
        keys=['code',fit.previous.prep.TIME]
        pd.testing.assert_frame_equal(p[keys],b[keys]);pd.testing.assert_frame_equal(p[keys],y[keys])
        if y.training_eligible.any():raise ValueError('SOURCE_LABEL_FLAG_CHANGED')
        actual=y[fit.previous.prep.TARGET].to_numpy()
        for algorithm in ('ridge','lightgbm'):
            raw=p[algorithm+'_raw'].to_numpy();baseline=np.clip(b[algorithm+'_raw'].to_numpy(),0,1)
            assert np.isfinite(raw).all()
            physical=np.clip(raw,0,1)
            metrics=report['metrics'][segment]['physical_amplitude_metrics'][algorithm]
            mse=np.mean((physical-actual)**2);mae=np.mean(abs(physical-actual))
            np.testing.assert_allclose(metrics['mse'],mse,atol=1e-14,rtol=0)
            np.testing.assert_allclose(metrics['mae'],mae,atol=1e-14,rtol=0)
            delta=report['incremental_new_minus_saved_baseline'][segment][algorithm]
            np.testing.assert_allclose(delta['mse']['left_minus_right'],mse-np.mean((baseline-actual)**2),atol=1e-14,rtol=0)
            np.testing.assert_allclose(delta['mae']['left_minus_right'],mae-np.mean(abs(baseline-actual)),atol=1e-14,rtol=0)
            checks+=len(p)
    oldroot=fit.HERE/'multiboard_three_arm_fit_20261018_v1'
    oldmanifest=json.loads((oldroot/'artifact_manifest.json').read_bytes())
    for name,item in oldmanifest['outputs'].items():assert sha(oldroot/name)==item['sha256']
    assert sha(fit.HERE/'postclose_repaired_research_fit_20261017_v1/model.txt')=='5d8402cd530cb83778d901e33eb87788f21ebc8619414f335b6d56905a81e005'
    answer=dict(status='PASS_PERSISTED_TARGET_AND_INCREMENTAL_ERROR_AUDIT_NOT_PROMOTION',
        report_sha256=sha(out/'report.json'),predictions_checked=checks,recipe_input_refs_checked=len(plan['inputs']),
        numeric_fields=350,transformed_fields=700,new_column_fingerprints=472,
        saved_model_prediction_replay_in_training_report=report['replayed_predictions'],
        independent_reconstruction_of_all_new_feature_values=False,
        used_columns=len(used),split_count=sum(v['splits'] for v in used),new_used_columns=new,
        old_three_arm_artifacts_unchanged=len(oldmanifest['outputs']),d2_unchanged=True,
        http_requests=0,fit=False,deployment=False)
    body=json.dumps(answer,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8')
    if bytes_used()+len(body)+32*1024>TOTAL_CAP:raise ValueError('AUDIT_BUDGET_STOP')
    with (fit.HERE/'continuous_motion_fit_acceptance_20261018_v1.json').open('xb') as stream:stream.write(body)
    return answer


if __name__=='__main__':print(json.dumps(run()))
