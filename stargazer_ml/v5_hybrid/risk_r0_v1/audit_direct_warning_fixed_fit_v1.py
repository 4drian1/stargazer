"""Independent persisted-model replay and direct metric audit."""
import argparse
import contextlib
import io
import json
import unittest

import numpy as np
import pandas as pd

import direct_warning_fixed_fit_v1 as fit
from research_budget_v2 import bytes_used,TOTAL_CAP


def write_exclusive(path,value):
    body=fit.previous.encoded(value)
    if bytes_used()+len(body)>TOTAL_CAP:
        raise ValueError('AUDIT_BUDGET_STOP')
    with path.open('xb') as f:
        f.write(body)


def entry():
    suite=unittest.defaultTestLoader.loadTestsFromName('test_direct_warning_fixed_fit_v1')
    log=io.StringIO()
    with contextlib.redirect_stdout(log):
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise ValueError(log.getvalue())
    config,parts,refs=fit.preflight()
    write_exclusive(fit.ENTRY,dict(status='PASS_DIRECT_WARNING_FIT_ENTRY_NO_REAL_FIT',
        config_sha256=fit.CONFIG_SHA,refs=refs,tests=result.testsRun,test_log=log.getvalue(),
        rows={s:len(f) for s,f in parts['C0'].items()},
        main_fits_permitted=dict(C0=0,C1=2),calibration_fits_permitted=4,
        new_http=False,deployment=False,independent_blind_test=False))
    return dict(status='PASS_DIRECT_WARNING_FIT_ENTRY_NO_REAL_FIT',tests=result.testsRun)


def audit():
    terminal=json.loads((fit.OUT/'stop_record.json').read_bytes())
    if terminal['status']!='COMPLETE_RESEARCH_NOT_DEPLOYMENT' or fit.previous.sha(fit.OUT/'artifact_manifest.json')!=terminal['manifest_sha256']:
        raise ValueError('FIT_TERMINAL_INVALID')
    manifest=json.loads((fit.OUT/'artifact_manifest.json').read_bytes())
    for name,item in manifest['outputs'].items():
        if fit.previous.sha(fit.OUT/name)!=item['sha256']:
            raise ValueError('ARTIFACT_DRIFT:'+name)
    plan=json.loads((fit.OUT/'plan.json').read_bytes())
    for p,expected in plan['inputs'].items():
        if fit.previous.sha(p)!=expected:
            raise ValueError('INPUT_DRIFT:'+p)
    config=plan['config']
    predictions=pd.read_parquet(fit.OUT/'predictions.parquet')
    report=json.loads((fit.OUT/'report.json').read_bytes())
    maximum_error=0.;checked=0
    for arm in ['C0','C1']:
        parts={s:fit.prep.load_arm(s,arm) for s in fit.SEGMENTS}
        for algorithm in ['lr','lightgbm']:
            prefix=f'{arm}_{algorithm}'
            if arm=='C0':
                refs=json.loads((fit.OUT/(prefix+'_base_references.json')).read_bytes())
                paths={name:fit.Path(item['path']) for name,item in refs.items()}
                for name,item in refs.items():
                    if fit.previous.sha(paths[name])!=item['sha256']:
                        raise ValueError('C0_BASE_DRIFT')
                state_path=paths[f'M1_{algorithm}_preprocessing.json']
                model_path=paths[f'M1_{algorithm}_model.'+('json' if algorithm=='lr' else 'txt')]
            else:
                state_path=fit.OUT/(prefix+'_preprocessing.json')
                model_path=fit.OUT/(prefix+'_model.'+('json' if algorithm=='lr' else 'txt'))
            state=json.loads(state_path.read_bytes())
            model=json.loads(model_path.read_bytes()) if algorithm=='lr' else model_path.read_text(encoding='utf-8')
            calibration=json.loads((fit.OUT/(prefix+'_calibration.json')).read_bytes())
            restored=fit.base.replay(parts,config['arms'][arm],algorithm,state,model,calibration,config['calibration']['epsilon'])
            for s in fit.SEGMENTS:
                stored=predictions[predictions.segment.eq(s)].reset_index(drop=True)
                pd.testing.assert_frame_equal(stored[['code',fit.TIME,'label_end_time','y_tail_1200m','mae_1200m']],
                    parts[s][['code',fit.TIME,'label_end_time','y_tail_1200m','mae_1200m']])
                for kind in ['raw','calibrated']:
                    p=stored[prefix+'_'+kind].to_numpy()
                    maximum_error=max(maximum_error,float(np.max(np.abs(p-restored[s][kind]))))
                    np.testing.assert_allclose(p,restored[s][kind],atol=1e-11,rtol=0)
                    y=stored.y_tail_1200m.to_numpy()
                    metrics=report['metrics'][s]['probability_metrics'][prefix+'_'+kind]
                    q=np.clip(p,1e-12,1-1e-12)
                    np.testing.assert_allclose(metrics['brier'],np.mean((p-y)**2),atol=1e-14)
                    np.testing.assert_allclose(metrics['log_loss'],np.mean(-y*np.log(q)-(1-y)*np.log1p(-q)),atol=1e-14)
                    checked+=len(p)
    original=fit.HERE/'multiboard_three_arm_fit_20261018_v1'
    oldmanifest=json.loads((original/'artifact_manifest.json').read_bytes())
    for name,item in oldmanifest['outputs'].items():
        if fit.previous.sha(original/name)!=item['sha256']:
            raise ValueError('OLD_MODEL_ARTIFACT_CHANGED:'+name)
    d2=fit.HERE/'postclose_repaired_research_fit_20261017_v1/model.txt'
    if fit.previous.sha(d2)!='5d8402cd530cb83778d901e33eb87788f21ebc8619414f335b6d56905a81e005':
        raise ValueError('D2_MODEL_CHANGED')
    answer=dict(status='PASS_FIXED_DIRECT_WARNING_RESEARCH_REPLAY_NOT_PROMOTION',
        probability_values_checked=checked,maximum_replay_error=maximum_error,
        rows=len(predictions),old_artifacts_checked=len(oldmanifest['outputs']),D2_unchanged=True,
        output_bytes=sum(p.stat().st_size for p in fit.OUT.iterdir() if p.is_file()),
        registered_bytes=bytes_used(),cap=TOTAL_CAP,new_http=0,deployed=False)
    write_exclusive(fit.HERE/'direct_warning_fixed_fit_acceptance_20261018_v1.json',answer)
    return answer


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--entry',action='store_true')
    args=parser.parse_args();print(json.dumps(entry() if args.entry else audit()))
