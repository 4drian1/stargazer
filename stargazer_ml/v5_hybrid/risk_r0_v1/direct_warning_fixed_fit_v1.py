"""One authorized fixed incremental research fit; never mutates preparation."""
import argparse
import gc
import io
import json
from importlib.metadata import version
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import prepare_direct_warning_comparison_v1 as prep
import prepare_multiboard_comparison_v1 as previous
import multiboard_three_arm_fit_v1 as base
from direct_warning_fit_evaluation_v1 import evaluate
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE, TIME, SEGMENTS = prep.HERE, prep.TIME, prep.SEGMENTS
OUT = HERE/'direct_warning_fixed_fit_20261018_v1'
ENTRY = HERE/'direct_warning_fit_entry_acceptance_20261018_v1.json'
AUTH = HERE/'contracts/DIRECT_WARNING_FIXED_FIT_CHAT_AUTHORIZATION_20261018.json'
CONFIG_SHA = '7c179a33e5260936e8a6cb084e77e4045df24cf7c97cfc8905c33743b30398d7'
CAP = 8*1024**2


def authorize(record, entry_sha):
    if record.get('scope') != 'DIRECT_WARNING_C0_C1_FIXED_RESEARCH_V1' or record.get('status') != 'CHAT_APPROVED_ONE_FIXED_RESEARCH_FIT':
        raise ValueError('FIXED_RESEARCH_AUTHORIZATION_REQUIRED')
    if record.get('config_sha256') != CONFIG_SHA or record.get('entry_acceptance_sha256') != entry_sha:
        raise ValueError('AUTHORIZATION_HASH_BINDING_MISMATCH')
    if record.get('user_quote') != '好，请继续' or not record.get('approval_context'):
        raise ValueError('CHAT_CONTEXT_REQUIRED')
    if any(record.get(k) is not False for k in ['http_authorized','deployment_authorized','parameter_search_authorized']):
        raise ValueError('AUTHORIZATION_SCOPE_EXPANDED')


def c0_predict(parts, config, algorithm):
    refs = config['C0_base_artifacts'][algorithm]
    state = json.loads(Path(refs[f'M1_{algorithm}_preprocessing.json']['path']).read_bytes())
    reference = Path(refs[f'M1_{algorithm}_model.'+('json' if algorithm=='lr' else 'txt')]['path'])
    model = json.loads(reference.read_bytes()) if algorithm=='lr' else reference.read_text(encoding='utf-8')
    # Identity sigmoid is only a replay placeholder, never the old calibrator.
    replay = base.replay(parts, config['arms']['C0'], algorithm, state, model,
                         dict(coef=[1.], intercept=0.), config['calibration']['epsilon'])
    raw = {s: replay[s]['raw'] for s in SEGMENTS}
    epsilon = config['calibration']['epsilon']
    with warnings.catch_warnings(record=True) as observed, threadpool_limits(2):
        warnings.simplefilter('always')
        sigmoid = LogisticRegression(**config['calibration']['estimator'])
        sigmoid.fit(logit(np.clip(raw['calibration'],epsilon,1-epsilon)).reshape(-1,1),
                    parts['calibration'].y_tail_1200m.to_numpy())
        calibrated = {s: base.positive(sigmoid,logit(np.clip(p,epsilon,1-epsilon)).reshape(-1,1)) for s,p in raw.items()}
    if any(issubclass(w.category,ConvergenceWarning) for w in observed):
        raise ValueError('CALIBRATION_DID_NOT_CONVERGE_NO_RETUNE')
    return dict(raw=raw, calibrated=calibrated, preprocessing=state, model=model,
                calibration=base.linear_record(sigmoid), best_iteration=None,
                warnings=[dict(category=type(w.message).__name__,message=str(w.message)) for w in observed])


def preflight(authorized=False):
    checked = prep.check()
    if not checked['preparation_ready'] or checked['config_sha256'] != CONFIG_SHA:
        raise ValueError('FIXED_PREPARATION_DRIFT')
    config = json.loads((prep.OUT/'config.json').read_bytes())
    if {p:version(p) for p in config['package_versions']} != config['package_versions']:
        raise ValueError('RUNTIME_DRIFT')
    acceptance = HERE/'direct_warning_comparison_acceptance_20261018_v1.json'
    if json.loads(acceptance.read_bytes())['status'] != 'PASS_DIRECT_WARNING_RESEARCH_SPEC_NOT_FIT':
        raise ValueError('PREPARATION_ACCEPTANCE_REQUIRED')
    refs = dict(json.loads((prep.OUT/'report.json').read_bytes())['inputs'])
    for p in [acceptance, prep.OUT/'config.json', Path(__file__),
              HERE/'direct_warning_fit_evaluation_v1.py', HERE/'test_direct_warning_fixed_fit_v1.py',
              HERE/'audit_direct_warning_fixed_fit_v1.py', HERE/'multiboard_three_arm_fit_v1.py',
              HERE/'multiboard_fit_evaluation_v1.py', HERE/'joint_fit_preparation_v1.py',
              HERE/'risk_fit_eval_v1.py',HERE/'prepare_multiboard_comparison_v1.py']:
        refs[str(p)] = previous.sha(p)
    for group in config['C0_base_artifacts'].values():
        for item in group.values():
            refs[item['path']] = item['sha256']
    if authorized:
        entry = json.loads(ENTRY.read_bytes())
        if entry['status'] != 'PASS_DIRECT_WARNING_FIT_ENTRY_NO_REAL_FIT':
            raise ValueError('ENTRY_ACCEPTANCE_REQUIRED')
        for p, expected in entry['refs'].items():
            if previous.sha(p) != expected:
                raise ValueError('ENTRY_CODE_OR_INPUT_DRIFT:'+p)
        authorize(json.loads(AUTH.read_bytes()), previous.sha(ENTRY))
        refs[str(ENTRY)], refs[str(AUTH)] = previous.sha(ENTRY), previous.sha(AUTH)
    for p,expected in refs.items():
        if previous.sha(p) != expected:
            raise ValueError('FROZEN_REFERENCE_DRIFT:'+p)
    parts = {'C0':{},'C1':{}}
    for segment in SEGMENTS:
        print(json.dumps(dict(stage='load_fixed_inputs',segment=segment)),flush=True)
        c1 = prep.load_arm(segment,'C1')
        c0 = c1.drop(columns=config['added_fields'])
        # Check against the original source, not a second copy of the same repaired reader.
        if segment in ['train','select']:
            old = previous.read_segment(segment)
            mask = pd.read_parquet(config['population_reference']['path'])
            mask = mask[mask.proposed_segment_name.eq(segment)].common_main_eligible.to_numpy()
            pd.testing.assert_frame_equal(c0[['code',TIME]+config['arms']['C0']],
                old.loc[mask,['code',TIME]+config['arms']['C0']].reset_index(drop=True))
            del old,mask
        previous.validate_times_labels(c1,segment,{s['name']:s for s in config['segments']})
        if list(c1[config['arms']['C1']]) != config['arms']['C1']:
            raise ValueError('COLUMN_ORDER_DRIFT')
        parts['C0'][segment],parts['C1'][segment] = c0,c1
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if bytes_used()+CAP > TOTAL_CAP:
        raise ValueError('REGISTERED_BUDGET_RESERVATION_FAILED')
    return config,parts,refs


def train(authorized=False):
    if not authorized:
        raise ValueError('EXPLICIT_FIXED_RESEARCH_AUTHORIZATION_REQUIRED')
    config,parts,refs = preflight(authorized=True)
    writer = base.EvidenceWriter(OUT,CAP,128*1024)
    writer.json('plan.json',dict(config=config,config_sha256=CONFIG_SHA,inputs=refs,
        main_fits=dict(C0=0,C1=2),sigmoid_fits=4,output_cap=CAP,
        limitations=['snapshot membership hindsight','unknown original arrival times',
            'restricted ordinary corporate action evidence','reused development test, not blind',
            '16 new coverage fields duplicate old training fields; package comparison not causal',
            'sparse equal-retention support; overlapping targets; no automatic promotion']))
    try:
        scores={s:{} for s in SEGMENTS}
        fitted={}
        for arm in ['C0','C1']:
            for algorithm in ['lr','lightgbm']:
                prefix=f'{arm}_{algorithm}'
                print(json.dumps(dict(stage='reuse_and_calibrate' if arm=='C0' else 'fixed_fit',name=prefix)),flush=True)
                result = c0_predict(parts[arm],config,algorithm) if arm=='C0' else base.fit_algorithm(parts[arm],config['arms'][arm],config,algorithm)
                if arm=='C0':
                    writer.json(prefix+'_base_references.json',config['C0_base_artifacts'][algorithm])
                else:
                    writer.json(prefix+'_preprocessing.json',result['preprocessing'])
                    if algorithm=='lr':
                        writer.json(prefix+'_model.json',result['model'])
                    else:
                        writer.write(prefix+'_model.txt',result['model'].encode('utf-8'))
                writer.json(prefix+'_calibration.json',result['calibration'])
                restored = base.replay(parts[arm],config['arms'][arm],algorithm,result['preprocessing'],
                    result['model'],result['calibration'],config['calibration']['epsilon'])
                for s in SEGMENTS:
                    for state in ['raw','calibrated']:
                        np.testing.assert_allclose(restored[s][state],result[state][s],atol=1e-11,rtol=0)
                        scores[s][prefix+'_'+state]=restored[s][state]
                fitted[prefix]=dict(main_model_refitted=arm=='C1',sigmoid_refitted=True,
                    best_iteration=result['best_iteration'],calibration=result['calibration'],warnings=result['warnings'],
                    negative_slope_not_repaired=result['calibration']['coef'][0]<0,replay='PASS')
                del result,restored
                gc.collect()
        reports,predictions={},[]
        train_rate=float(parts['C0']['train'].y_tail_1200m.mean())
        for s,frame in parts['C1'].items():
            print(json.dumps(dict(stage='evaluate',segment=s)),flush=True)
            reports[s]=evaluate(frame,scores[s],train_rate,config)
            item=frame[['code',TIME,'label_end_time','y_tail_1200m','mae_1200m']].copy()
            item['segment']=s
            for name,values in scores[s].items():
                item[name]=values
            predictions.append(item)
        sink=io.BytesIO()
        pd.concat(predictions,ignore_index=True).to_parquet(sink,index=False,compression='zstd')
        writer.write('predictions.parquet',sink.getvalue())
        writer.json('report.json',dict(status='COMPLETE_FIXED_RESEARCH_NOT_PROMOTION',fitted=fitted,metrics=reports,
            config_sha256=CONFIG_SHA,train_constant_risk=train_rate,actual_main_model_fits=2,actual_sigmoid_fits=4,
            rows=sum(map(len,parts['C0'].values())),new_blind_test=False,promotion_gate_passed=False,
            deployment_authorized=False,pnl_evaluated=False))
        for p,expected in refs.items():
            if previous.sha(p)!=expected:
                raise ValueError('INPUT_CHANGED_DURING_FIT:'+p)
        writer.json('artifact_manifest.json',dict(outputs=writer.outputs.copy()))
        writer.json('stop_record.json',dict(status='COMPLETE_RESEARCH_NOT_DEPLOYMENT',
            manifest_sha256=previous.sha(OUT/'artifact_manifest.json')),terminal=True)
        return dict(status='COMPLETE_RESEARCH_NOT_DEPLOYMENT',rows=86319,bytes=writer.used)
    except Exception as exc:
        writer.json('stop_record.json',dict(status='FAILED_NO_AUTOMATIC_RETRY',error=f'{type(exc).__name__}:{exc}'),terminal=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--train',action='store_true')
    parser.add_argument('--authorize-fixed-research',action='store_true')
    args=parser.parse_args()
    if args.train:
        print(json.dumps(train(args.authorize_fixed_research)))
    else:
        _,parts,_=preflight(args.authorize_fixed_research)
        print(json.dumps(dict(engineering_ready=True,fit_performed=False,rows={s:len(f) for s,f in parts['C0'].items()})))
