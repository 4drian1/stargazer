"""Guarded fixed full-framework regression entry; real FIT needs separate approval."""
import argparse
from datetime import datetime,timezone
import gzip
import io
import json
from importlib.metadata import version
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import prepare_auxiliary_fixed_fit_v1 as prep
from joint_fit_preparation_v1 import fit_lr_preprocessing,transform_lr,native_missing_view
from auxiliary_regression_eval_v1 import evaluate
from research_budget_v2 import bytes_used,TOTAL_CAP

HERE,OUT=prep.HERE,prep.FIT_OUT
ENTRY=HERE/'auxiliary_fixed_fit_entry_acceptance_20261018_v1.json'
AUTH=HERE/'contracts/AUXILIARY_120M_FIXED_FIT_CHAT_AUTHORIZATION_20261018.json'


def authorization(record,config_sha,entry_sha):
    if record.get('status')!='CHAT_APPROVED_ONE_FIXED_RESEARCH_FIT' or record.get('scope')!='FULL_FRAMEWORK_AUXILIARY_120M_FIXED_RESEARCH_V1':
        raise ValueError('SEPARATE_AUXILIARY_FIT_AUTHORIZATION_REQUIRED')
    if record.get('config_sha256')!=config_sha or record.get('entry_acceptance_sha256')!=entry_sha:
        raise ValueError('AUTHORIZATION_HASH_BINDING_MISMATCH')
    if not record.get('user_quote') or not record.get('approval_context'):raise ValueError('CHAT_APPROVAL_CONTEXT_REQUIRED')
    if any(record.get(k) is not False for k in ['http_authorized','deployment_authorized','parameter_search_authorized']):
        raise ValueError('AUTHORIZATION_SCOPE_EXPANDED')


def fit_model(parts,fields,config,algorithm):
    state=fit_lr_preprocessing(parts['train'][fields])
    x={s:transform_lr(f[fields],state) if algorithm=='ridge' else native_missing_view(f[fields]) for s,f in parts.items()}
    y={s:f[prep.TARGET].to_numpy(dtype=float) for s,f in parts.items()}
    if any(not np.isfinite(v).all() or ((v<0)|(v>=1)).any() for v in y.values()):raise ValueError('INVALID_CONTINUOUS_TARGET')
    with warnings.catch_warnings(record=True) as observed,threadpool_limits(2):
        warnings.simplefilter('always')
        if algorithm=='ridge':
            model=Ridge(**config['models']['ridge']).fit(x['train'],y['train'])
            saved=dict(coef=model.coef_.tolist(),intercept=float(model.intercept_));best=None
        elif algorithm=='lightgbm':
            model=lgb.LGBMRegressor(**config['models']['lightgbm'])
            model.fit(x['train'],y['train'],eval_set=[(x['select'],y['select'])],
                eval_metric=config['selection']['metric'],
                callbacks=[lgb.early_stopping(config['selection']['early_stopping_rounds'],verbose=False)])
            best=int(model.best_iteration_);saved=model.booster_.model_to_string(num_iteration=best)
        else:raise ValueError('UNKNOWN_REGRESSION_MODEL')
        raw={s:model.predict(v) for s,v in x.items()}
    if any(not np.isfinite(v).all() for v in raw.values()):raise ValueError('NONFINITE_PREDICTION')
    if any(issubclass(w.category,ConvergenceWarning) for w in observed):raise ValueError('NONCONVERGENCE_NO_AUTOMATIC_RETUNE')
    return dict(preprocessing=state,model=saved,raw=raw,best_iteration=best,
        warnings=[dict(category=type(w.message).__name__,message=str(w.message)) for w in observed])


def replay(parts,fields,algorithm,state,model):
    state=dict(state)
    for k in ['median','mean','scale']:state[k]=np.asarray(state[k],dtype=float)
    predictor=lgb.Booster(model_str=model) if algorithm=='lightgbm' else None
    with threadpool_limits(2):
        return {s:(predictor.predict(native_missing_view(f[fields]),num_threads=2) if predictor is not None else
            transform_lr(f[fields],state)@np.asarray(model['coef'])+model['intercept']) for s,f in parts.items()}


def preflight(authorized=False):
    checked=prep.check();config=json.loads((prep.OUT/'config.json').read_bytes())
    if not checked['preparation_ready']:raise ValueError('PREPARATION_NOT_READY')
    if {p:version(p) for p in config['package_versions']}!=config['package_versions']:raise ValueError('RUNTIME_DRIFT')
    accepted=HERE/'auxiliary_fixed_preparation_acceptance_20261018_v1.json'
    receipt=json.loads(accepted.read_bytes())
    if receipt['status']!='PASS_FULL_FRAMEWORK_AUXILIARY_SPEC_NOT_FIT' or receipt['config_sha256']!=checked['config_sha256']:
        raise ValueError('PREPARATION_ACCEPTANCE_REQUIRED')
    refs=dict(json.loads((prep.OUT/'plan.json').read_bytes())['inputs'])
    for p in [accepted,prep.OUT/'config.json']:
        refs[str(p)]=prep.parent.previous.sha(p)
    if authorized:
        if not AUTH.exists() or not ENTRY.exists():raise ValueError('SEPARATE_AUXILIARY_FIT_AUTHORIZATION_REQUIRED')
        entry=json.loads(ENTRY.read_bytes())
        if entry['status']!='PASS_AUXILIARY_FIXED_ENTRY_NO_REAL_FIT':raise ValueError('ENTRY_ACCEPTANCE_REQUIRED')
        for p,expected in entry['refs'].items():
            if prep.parent.previous.sha(p)!=expected:raise ValueError('ENTRY_DRIFT:'+p)
        authorization(json.loads(AUTH.read_bytes()),checked['config_sha256'],prep.parent.previous.sha(ENTRY))
        refs[str(AUTH)],refs[str(ENTRY)]=prep.parent.previous.sha(AUTH),prep.parent.previous.sha(ENTRY)
    for p,expected in refs.items():
        if prep.parent.previous.sha(p)!=expected:raise ValueError('INPUT_DRIFT:'+p)
    if OUT.exists():raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if bytes_used()+prep.FIT_CAP>TOTAL_CAP:raise ValueError('PROPOSED_FIT_BUDGET_NOT_AVAILABLE')
    return config,prep.load_parts(config),refs


def train(authorized=False):
    if not authorized:raise ValueError('SEPARATE_AUXILIARY_FIT_AUTHORIZATION_REQUIRED')
    config,parts,refs=preflight(True)
    writer=prep.parent.base.EvidenceWriter(OUT,prep.FIT_CAP,128*1024)
    writer.json('plan.json',dict(config=config,inputs=refs,real_model_fits=2,calibration_fits=0,
        source_labels_training_flag_not_mutated=True,output_cap=prep.FIT_CAP))
    try:
        raw={s:{} for s in prep.SEGMENTS};fitted={}
        for algorithm in ['ridge','lightgbm']:
            print(json.dumps(dict(stage='fixed_auxiliary_fit',algorithm=algorithm)),flush=True)
            result=fit_model(parts,config['fields'],config,algorithm)
            writer.json(algorithm+'_preprocessing.json',result['preprocessing'])
            if algorithm=='ridge':writer.json('ridge_model.json',result['model'])
            else:writer.write('lightgbm_model.txt.gz',gzip.compress(result['model'].encode('utf-8'),mtime=0))
            state=json.loads((OUT/(algorithm+'_preprocessing.json')).read_bytes())
            model=json.loads((OUT/'ridge_model.json').read_bytes()) if algorithm=='ridge' else gzip.decompress((OUT/'lightgbm_model.txt.gz').read_bytes()).decode('utf-8')
            restored=replay(parts,config['fields'],algorithm,state,model)
            for segment in prep.SEGMENTS:
                np.testing.assert_allclose(restored[segment],result['raw'][segment],atol=1e-11,rtol=0)
                raw[segment][algorithm]=restored[segment]
            fitted[algorithm]=dict(best_iteration=result['best_iteration'],warnings=result['warnings'],replay='PASS')
        frozen_at=datetime.now(timezone.utc).isoformat()
        train_mean=float(parts['train'][prep.TARGET].mean());train_median=float(parts['train'][prep.TARGET].median())
        reports={};tables=[]
        for segment,frame in parts.items():
            reports[segment]=evaluate(frame,raw[segment],train_mean,train_median,config)
            item=frame[['code',prep.TIME]].copy();item['segment']=segment
            for model,p in raw[segment].items():item[model+'_raw']=p
            tables.append(item)
        sink=io.BytesIO();pd.concat(tables,ignore_index=True).to_parquet(sink,index=False,compression='zstd')
        writer.write('predictions.parquet',sink.getvalue())
        writer.json('report.json',dict(status='COMPLETE_FIXED_AUXILIARY_RESEARCH_NOT_PROMOTION',fitted=fitted,metrics=reports,
            baselines=dict(train_mean=train_mean,train_median=train_median),main_model_fits=2,calibration_fits=0,
            amplitude_not_probability=True,training_and_select_metrics_not_independent=True,
            model_weights_frozen_at=frozen_at,forward_validation_status='NOT_STARTED_NOT_AUTHORIZED',
            test_reused=True,promotion_passed=False,deployment=False,http_requests=0))
        for p,expected in refs.items():
            if prep.parent.previous.sha(p)!=expected:raise ValueError('INPUT_CHANGED:'+p)
        writer.json('manifest.json',dict(outputs=writer.outputs.copy()))
        writer.json('stop_record.json',dict(status='COMPLETE_NOT_DEPLOYED',manifest_sha256=prep.parent.previous.sha(OUT/'manifest.json')),terminal=True)
        return dict(status='COMPLETE_NOT_DEPLOYED',rows=86319,bytes=writer.used)
    except Exception as exc:
        writer.json('stop_record.json',dict(status='FAILED_NO_RESTART',error=f'{type(exc).__name__}:{exc}'),terminal=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--train',action='store_true')
    parser.add_argument('--authorize-fixed-research',action='store_true');args=parser.parse_args()
    try:
        if args.train:answer=train(args.authorize_fixed_research)
        else:
            _,parts,_=preflight(args.authorize_fixed_research)
            answer=dict(engineering_ready=True,training_authorized=args.authorize_fixed_research,
                fit_performed=False,rows={s:len(f) for s,f in parts.items()})
        print(json.dumps(answer))
    except (ValueError,FileNotFoundError) as exc:
        print(json.dumps(dict(training_ready=False,blocked=str(exc),fit_performed=False)))
        raise SystemExit(2)
