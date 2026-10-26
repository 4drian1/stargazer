"""Conditional chat-approved fixed full-population feature acceptance then FIT."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import gzip
import hashlib
import io
from importlib.metadata import version
import json
import unittest

import numpy as np
import pandas as pd

import auxiliary_regression_fit_v1 as previous
from auxiliary_regression_eval_v1 import evaluate,paired_errors
from build_continuous_motion_population_v1 import build,bindings,fields
from materialize_relative_stall_pilot_v1 import HERE,sha
from multiboard_three_arm_fit_v1 import EvidenceWriter
from research_budget_v2 import bytes_used,TOTAL_CAP

OUT=HERE/'continuous_motion_fixed_fit_20261018_v1'
CAP,RESERVE=3*1024**2,128*1024
CONTRACT=HERE/'contracts/CONTINUOUS_MOTION_CONDITIONAL_FIT_20261018.json'


def config_and_refs():
    contract=json.loads(CONTRACT.read_bytes())
    if (contract.get('user_quote')!='好，请继续吧，验收通过就直接训练吧'
        or contract.get('scope')!='FULL_FRAMEWORK_CONTINUOUS_MOTION_ACCEPT_THEN_ONE_FIXED_FIT'
        or contract.get('http_authorized') is not False or contract.get('deployment_authorized') is not False
        or contract.get('parameter_search_authorized') is not False):raise ValueError('CONDITIONAL_FIT_AUTHORIZATION_REQUIRED')
    old=previous.prep.OUT/'config.json'
    if sha(old)!='8a3ffb0cb2ea6f04f4f6d9cb32030d6fd7281f96ed92c1fd94c9aadcc7d0baaa':raise ValueError('PARENT_CONFIG_DRIFT')
    base=json.loads(old.read_bytes());config=deepcopy(base)
    config.update(version='CONTINUOUS_MOTION_FIXED_RESEARCH_V1',fields=base['fields']+fields(),
        numeric_dimensions=350,transformed_dimensions=700,status='FIXED_CONDITIONAL_CHAT_AUTHORIZATION',
        prior_model_reference=str(previous.OUT),new_fields=fields(),fit_authorized_after_acceptance=True)
    if len(fields())!=118 or len(set(config['fields']))!=350:raise ValueError('FEATURE_SCHEMA_DRIFT')
    if {p:version(p) for p in config['package_versions']}!=config['package_versions']:raise ValueError('RUNTIME_DRIFT')
    prior_plan=json.loads((previous.OUT/'plan.json').read_bytes());refs=dict(prior_plan['inputs'])
    refs.update(bindings())
    for p in [CONTRACT,old,previous.OUT/'plan.json',previous.OUT/'manifest.json',previous.OUT/'report.json',
              previous.OUT/'predictions.parquet',previous.OUT/'stop_record.json',
              HERE/'auxiliary_fixed_fit_acceptance_20261018_v1.json']:
        refs[str(p)]=sha(p)
    for n in ['continuous_motion_matrix_v1.py','test_continuous_motion_matrix_v1.py',
              'continuous_motion_features_v1.py','test_continuous_motion_features_v1.py',
              'build_continuous_motion_population_v1.py','continuous_motion_fixed_fit_v1.py',
              'CONTINUOUS_MOTION_FEATURE_SPEC_V1.md','auxiliary_regression_eval_v1.py','auxiliary_regression_fit_v1.py']:
        refs[str(HERE/n)]=sha(HERE/n)
    stop=json.loads((previous.OUT/'stop_record.json').read_bytes())
    if stop['status']!='COMPLETE_NOT_DEPLOYED' or stop['manifest_sha256']!=sha(previous.OUT/'manifest.json'):
        raise ValueError('PRIOR_FIT_NOT_ACCEPTED')
    for name,item in json.loads((previous.OUT/'manifest.json').read_bytes())['outputs'].items():
        if sha(previous.OUT/name)!=item['sha256']:raise ValueError('PRIOR_ARTIFACT_DRIFT')
    for path,expected in refs.items():
        if sha(path)!=expected:raise ValueError('INPUT_DRIFT:'+path)
    return base,config,refs


def run(authorize=False):
    if not authorize:raise ValueError('EXPLICIT_CONDITIONAL_FIT_FLAG_REQUIRED')
    if OUT.exists():raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if bytes_used()+CAP>TOTAL_CAP:raise ValueError('COMBINED_FEATURE_AND_FIT_RESERVATION_UNAVAILABLE')
    base,config,refs=config_and_refs()
    writer=EvidenceWriter(OUT,CAP,RESERVE)
    writer.json('plan.json',dict(config=config,inputs=refs,conditional_authorization=json.loads(CONTRACT.read_bytes()),
        cap=CAP,feature_sidecar='IN_MEMORY_REPRODUCIBLE_SOURCE_RECIPE_AND_COLUMN_HASHES',
        actual_http_requests=0,new_main_model_fits_if_accepted=2,calibration_fits=0))
    try:
        log=io.StringIO()
        tests=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames([
            'test_continuous_motion_matrix_v1','test_continuous_motion_features_v1','test_multiboard_risk_features_v1']))
        if not tests.wasSuccessful():raise ValueError('FEATURE_TEST_FAILED:'+log.getvalue())
        parts=previous.prep.load_parts(base)
        original={s:f[base['fields']].copy() for s,f in parts.items()}
        labels={s:f[[previous.prep.TARGET,'label_end_time','auxiliary_end_time']].copy() for s,f in parts.items()}
        parts,feature_report=build(parts)
        for s,frame in parts.items():
            pd.testing.assert_frame_equal(frame[base['fields']],original[s])
            pd.testing.assert_frame_equal(frame[[previous.prep.TARGET,'label_end_time','auxiliary_end_time']],labels[s])
        all_null=[f for f in fields() if feature_report['profiles']['train'][f]['finite']==0]
        writer.json('feature_acceptance.json',dict(status='PASS_NOT_FIT_YET' if not all_null else 'BLOCKED_ALL_NULL_TRAIN',
            tests=tests.testsRun,test_output=log.getvalue(),all_null_train=all_null,**feature_report))
        if all_null:raise ValueError('ALL_NULL_TRAIN_NO_AUTOMATIC_DROP:'+','.join(all_null))
        # Reusing the saved baseline establishes no original inputs were changed.
        baseline=pd.read_parquet(previous.OUT/'predictions.parquet')
        prior_report=json.loads((previous.OUT/'report.json').read_bytes())
        baseline_raw={}
        for s,frame in parts.items():
            old=baseline[baseline.segment.eq(s)].reset_index(drop=True)
            pd.testing.assert_frame_equal(frame[['code',previous.prep.TIME]],old[['code',previous.prep.TIME]])
            baseline_raw[s]={m:old[m+'_raw'].to_numpy() for m in ('ridge','lightgbm')}
        print(json.dumps(dict(stage='feature_acceptance_passed',rows=86319,fields=350,tests=tests.testsRun)),flush=True)
        raw={s:{} for s in parts};fitted={}
        for algorithm in ('ridge','lightgbm'):
            print(json.dumps(dict(stage='fixed_motion_fit',algorithm=algorithm)),flush=True)
            result=previous.fit_model(parts,config['fields'],config,algorithm)
            writer.json(algorithm+'_preprocessing.json',result['preprocessing'])
            if algorithm=='ridge':writer.json('ridge_model.json',result['model'])
            else:writer.write('lightgbm_model.txt.gz',gzip.compress(result['model'].encode('utf-8'),mtime=0))
            state=json.loads((OUT/(algorithm+'_preprocessing.json')).read_bytes())
            model=json.loads((OUT/'ridge_model.json').read_bytes()) if algorithm=='ridge' else gzip.decompress((OUT/'lightgbm_model.txt.gz').read_bytes()).decode()
            restored=previous.replay(parts,config['fields'],algorithm,state,model)
            for s in parts:
                np.testing.assert_allclose(restored[s],result['raw'][s],atol=1e-11,rtol=0)
                raw[s][algorithm]=restored[s]
            fitted[algorithm]=dict(best_iteration=result['best_iteration'],warnings=result['warnings'],saved_replay='PASS')
        frozen=datetime.now(timezone.utc).isoformat()
        mean=prior_report['baselines']['train_mean'];median=prior_report['baselines']['train_median']
        reports={};increment={};tables=[]
        for s,frame in parts.items():
            reports[s]=evaluate(frame,raw[s],mean,median,config)
            y=frame[previous.prep.TARGET].to_numpy();dates=frame[previous.prep.TIME].to_numpy()//1000000
            increment[s]={m:paired_errors(y,np.clip(raw[s][m],0,1),np.clip(baseline_raw[s][m],0,1),
                dates,config['evaluation']['uncertainty']) for m in ('ridge','lightgbm')}
            item=frame[['code',previous.prep.TIME]].copy();item['segment']=s
            for m in ('ridge','lightgbm'):
                p=np.clip(raw[s][m],0,1);metric=reports[s]['physical_amplitude_metrics'][m]
                np.testing.assert_allclose(metric['mse'],np.mean((p-y)**2),atol=1e-14,rtol=0)
                np.testing.assert_allclose(metric['mae'],np.mean(abs(p-y)),atol=1e-14,rtol=0)
                item[m+'_raw']=raw[s][m]
            tables.append(item)
        sink=io.BytesIO();pd.concat(tables,ignore_index=True).to_parquet(sink,index=False,compression='zstd')
        writer.write('predictions.parquet',sink.getvalue())
        writer.json('report.json',dict(status='COMPLETE_CONTINUOUS_MOTION_FIXED_RESEARCH_NOT_PROMOTION',
            fitted=fitted,metrics=reports,incremental_new_minus_saved_baseline=increment,main_model_fits=2,
            calibration_fits=0,model_weights_frozen_at=frozen,replayed_predictions=172638,
            labels_unchanged=True,old_models_unchanged=True,http_requests=0,deployment=False,promotion_passed=False,
            test_reused=True,prospective_validation='NOT_STARTED_NOT_AUTHORIZED',amplitude_not_probability=True))
        for path,expected in refs.items():
            if sha(path)!=expected:raise ValueError('INPUT_CHANGED_DURING_BUILD:'+path)
        writer.json('manifest.json',dict(outputs=writer.outputs.copy()))
        writer.json('stop_record.json',dict(status='COMPLETE_NOT_DEPLOYED',manifest_sha256=sha(OUT/'manifest.json')),terminal=True)
        return dict(status='COMPLETE_NOT_DEPLOYED',rows=86319,fields=350,bytes=writer.used)
    except Exception as exc:
        writer.json('stop_record.json',dict(status='FAILED_NO_RESTART',reason=f'{type(exc).__name__}:{exc}'),terminal=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--accept-then-train',action='store_true');args=parser.parse_args()
    print(json.dumps(run(args.accept_then_train)))
