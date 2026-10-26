"""Freeze full-framework continuous-risk research inputs; no real-data fitting."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

import direct_warning_fixed_fit_v1 as parent
import build_auxiliary_120m_v1 as auxiliary
from direct_warning_comparison_v1 import train_patterns

HERE,TIME,SEGMENTS=parent.HERE,parent.TIME,parent.SEGMENTS
OUT=HERE/'auxiliary_120m_fit_preparation_20261018_v1'
FIT_OUT=HERE/'auxiliary_120m_fixed_fit_20261018_v1'
TARGET='max_adverse_drop_120m'
CAP,RESERVE,FIT_CAP=256*1024,32*1024,3*1024**2


def load_parts(config):
    target=pd.read_parquet(auxiliary.OUT/'labels.parquet')
    parts={}
    for segment in SEGMENTS:
        print(json.dumps(dict(stage='load_auxiliary_research',segment=segment)),flush=True)
        frame=parent.prep.load_arm(segment,'C1')
        labels=target[target.segment.eq(segment)].reset_index(drop=True)
        pd.testing.assert_frame_equal(frame[['code',TIME]],labels[['code',TIME]])
        x=labels[TARGET].to_numpy(dtype=float)
        if not np.isfinite(x).all() or ((x<0)|(x>=1)).any():raise ValueError('INVALID_CONTINUOUS_TARGET')
        if labels.training_eligible.any():raise ValueError('SOURCE_LABEL_AUTHORIZATION_MUTATED')
        frame[TARGET]=x
        frame['auxiliary_end_time']=labels.auxiliary_end_time.to_numpy(dtype=np.int64)
        if not ((frame[TIME]<frame.auxiliary_end_time)&(frame.auxiliary_end_time<=frame.label_end_time)).all():
            raise ValueError('AUXILIARY_TIMES_NOT_STRICTLY_AFTER_DECISION')
        boundary=next(s['end'] for s in config['segments'] if s['name']==segment)
        if (frame.auxiliary_end_time//1000000>boundary).any():raise ValueError('AUXILIARY_LABEL_CROSSES_SEGMENT')
        values=frame[config['fields']].to_numpy(dtype=float)
        if np.isinf(values).any():raise ValueError('INFINITE_FEATURE')
        parts[segment]=frame
    if sum(map(len,parts.values()))!=86319:raise ValueError('FROZEN_POPULATION_CHANGED')
    return parts


def validate_fields(fields,expected):
    if fields!=expected or len(fields)!=232 or len(set(fields))!=232:
        raise ValueError('FULL_FRAMEWORK_WHITELIST_DRIFT')
    if any(f in ['code',TIME,TARGET] or f.startswith(('future_','max_adverse_','auxiliary_','mae_','y_tail','label_')) for f in fields):
        raise ValueError('TARGET_OR_AUDIT_LEAKAGE')


def config_and_refs():
    checked=parent.prep.check()
    if checked['config_sha256']!=parent.CONFIG_SHA or not checked['preparation_ready']:
        raise ValueError('PARENT_SPEC_DRIFT')
    old=json.loads((parent.prep.OUT/'config.json').read_bytes())
    fields=old['arms']['C1'];validate_fields(fields,old['arms']['C1'])
    booster=dict(old['models']['lightgbm']);booster.pop('class_weight',None);booster['objective']='regression_l2'
    config=dict(version='FULL_FRAMEWORK_120M_CONTINUOUS_RESEARCH_V1',status='FIXED_SPEC_NOT_FIT_AUTHORIZATION',
        fit_authorized=False,http_authorized=False,deployment_authorized=False,target=TARGET,
        fields=fields,numeric_dimensions=232,transformed_dimensions=464,segments=old['segments'],
        parent_field_mapping_reference=dict(path=str(parent.prep.OUT/'config.json'),sha256=parent.CONFIG_SHA),
        label_reference=dict(path=str(auxiliary.OUT/'labels.parquet'),sha256=parent.previous.sha(auxiliary.OUT/'labels.parquet')),
        framework='golden price/time + market + full board distribution + current/captured leaders + individual',
        models=dict(ridge=dict(alpha=1.,fit_intercept=True,solver='cholesky'),lightgbm=booster),
        preprocessing=dict(ridge='train-only median/mean/std plus all per-field missing indicators',
            lightgbm='native NaN plus all per-field missing indicators',all_null_train='STOP_NO_COLUMN_DROP'),
        selection=dict(metric='l2',early_stopping_rounds=50,segment='select'),
        calibration=dict(method=None,segment_role='diagnostic_only_no_fit_no_pooling'),
        baselines=dict(mean='train target arithmetic mean',median='train target median'),
        output_policy=dict(primary='clip raw predictions to [0,1], physical amplitude not probability',
            raw='persist float64 and report raw errors/bound violations; never pick representation from test'),
        evaluation=dict(primary='MSE and paired LightGBM minus Ridge / train mean',
            secondary=['MAE','train median baseline','monthly mean bias','same-cutoff continuous concordance',
                       'mean observed downside at equal retention versus analytical random'],
            uncertainty=old['evaluation']['uncertainty'],rejection_fraction=.2,minimum_cross_section=5,
            ties='SHA256(R0_MULTIBOARD_V1|cutoff|code)',pnl=False,binary_threshold=None,
            independent_blind_test=False,test_role='reused historical research, no selection',promotion_gate=None),
        prospective_validation=dict(status='NOT_STARTED_NOT_AUTHORIZED',
            start='first trading day strictly after actual model weight freeze; no retrospective backfill'),
        package_versions=old['package_versions'],fit_output_cap=FIT_CAP,
        artifact_policy='gzip lossless LightGBM text, two float64 raw prediction columns, no copied feature matrix',
        real_preprocessing_fitted=False,real_models_fitted=False)
    if {p:version(p) for p in config['package_versions']}!=config['package_versions']:
        raise ValueError('FROZEN_RUNTIME_DRIFT')
    refs=dict(json.loads((parent.prep.OUT/'report.json').read_bytes())['inputs'])
    acceptance=HERE/'auxiliary_120m_acceptance_20261018_v1.json'
    accepted=json.loads(acceptance.read_bytes())
    if accepted['status']!='PASS_AUXILIARY_120M_LABELS_NOT_FIT' or parent.previous.sha(auxiliary.OUT/'report.json')!=accepted['report_sha256']:
        raise ValueError('AUXILIARY_ACCEPTANCE_REQUIRED')
    for p in [parent.prep.OUT/'config.json',acceptance,auxiliary.CONTRACT,auxiliary.OUT/'labels.parquet',
        auxiliary.OUT/'manifest.json',auxiliary.OUT/'stop_record.json',auxiliary.OUT/'report.json',Path(__file__),
        HERE/'auxiliary_regression_fit_v1.py',HERE/'auxiliary_regression_eval_v1.py',HERE/'test_auxiliary_fixed_fit_v1.py',
        HERE/'audit_auxiliary_fixed_fit_v1.py',HERE/'joint_fit_preparation_v1.py',
        HERE/'diagnose_auxiliary_120m_v1.py',HERE/'multiboard_three_arm_fit_v1.py',HERE/'research_budget_v2.py']:
        refs[str(p)]=parent.previous.sha(p)
    for p,expected in refs.items():
        if parent.previous.sha(p)!=expected:raise ValueError('INPUT_DRIFT:'+p)
    return config,refs


def prepare():
    import io
    import unittest
    log=io.StringIO()
    tests=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_auxiliary_fixed_fit_v1'))
    if not tests.wasSuccessful():raise ValueError(log.getvalue())
    config,refs=config_and_refs()
    writer=parent.base.EvidenceWriter(OUT,CAP,RESERVE)
    writer.json('plan.json',dict(inputs=refs,output_cap=CAP,proposed_fit_cap=FIT_CAP,
        note='Only consume accepted derived snapshots, no raw-minute reparse or real estimator fit'))
    try:
        writer.json('config.json',config)
        parts=load_parts(config);profiles={};segments={}
        train=parts['train'][config['fields']]
        patterns=train_patterns(train)
        if patterns['all_null_columns']:raise ValueError('ALL_NULL_TRAIN_COLUMN')
        for field in config['fields']:
            profiles[field]=dict(known=int(train[field].notna().sum()),missing=int(train[field].isna().sum()),distinct=int(train[field].nunique(dropna=True)))
        for segment,f in parts.items():
            x=f[TARGET].to_numpy()
            segments[segment]=dict(rows=len(f),dates=int((f[TIME]//1000000).nunique()),stocks=int(f.code.nunique()),
                target_mean=float(x.mean()),target_median=float(np.median(x)),target_min=float(x.min()),target_max=float(x.max()),
                all_null_feature_columns=[c for c in config['fields'] if f[c].isna().all()])
        writer.json('train_profiles.json',dict(fields=profiles,patterns=patterns,automatic_column_removal=False))
        writer.json('report.json',dict(status='COMPLETE_FULL_FRAMEWORK_AUXILIARY_PREPARATION_NOT_FIT',
            technical_blockers=[],preparation_ready=True,config_sha256=parent.previous.sha(OUT/'config.json'),
            tests=tests.testsRun,test_output=log.getvalue(),segments=segments,rows=86319,
            real_preprocessing_fitted=False,real_models_fitted=False,calibration_fits=0,
            http_requests=0,fit_authorized=False,original_source_labels_not_mutated=True))
        for p,expected in refs.items():
            if parent.previous.sha(p)!=expected:raise ValueError('INPUT_CHANGED:'+p)
        writer.json('manifest.json',dict(outputs=writer.outputs.copy()))
        writer.json('stop_record.json',dict(status='COMPLETE_NOT_FIT',manifest_sha256=parent.previous.sha(OUT/'manifest.json')),terminal=True)
        return dict(preparation_ready=True,fit_authorized=False,bytes=writer.used,config_sha256=parent.previous.sha(OUT/'config.json'))
    except Exception as exc:
        writer.json('stop_record.json',dict(status='FAILED_NO_RESTART',error=f'{type(exc).__name__}:{exc}'),terminal=True)
        raise


def check():
    stop=json.loads((OUT/'stop_record.json').read_bytes())
    if stop['status']!='COMPLETE_NOT_FIT' or parent.previous.sha(OUT/'manifest.json')!=stop['manifest_sha256']:
        raise ValueError('AUXILIARY_PREPARATION_INCOMPLETE')
    for n,item in json.loads((OUT/'manifest.json').read_bytes())['outputs'].items():
        if parent.previous.sha(OUT/n)!=item['sha256']:raise ValueError('PREPARATION_OUTPUT_DRIFT:'+n)
    for p,expected in json.loads((OUT/'plan.json').read_bytes())['inputs'].items():
        if parent.previous.sha(p)!=expected:raise ValueError('PREPARATION_INPUT_DRIFT:'+p)
    config=json.loads((OUT/'config.json').read_bytes())
    if any(config[k] is not False for k in ['fit_authorized','http_authorized','deployment_authorized']):
        raise ValueError('SPEC_IS_NOT_AUTHORIZATION')
    report=json.loads((OUT/'report.json').read_bytes())
    return dict(preparation_ready=report['preparation_ready'],fit_authorized=False,
        config_sha256=parent.previous.sha(OUT/'config.json'),real_models_fitted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--check',action='store_true')
    args=parser.parse_args();print(json.dumps(prepare() if args.prepare else check()))
