"""Preparation/entry audit and later saved-regressor replay; never authorizes FIT."""
import argparse
import gzip
import io
import json
import unittest

import numpy as np
import pandas as pd

import auxiliary_regression_fit_v1 as fit
from audit_direct_warning_fixed_fit_v1 import write_exclusive
from research_budget_v2 import bytes_used,TOTAL_CAP


def preparation():
    checked=fit.prep.check();config=json.loads((fit.prep.OUT/'config.json').read_bytes())
    old=json.loads((fit.prep.parent.prep.OUT/'config.json').read_bytes())
    assert config['fields']==old['arms']['C1'] and len(config['fields'])==232
    assert config['calibration']['method'] is None and not config['fit_authorized']
    assert config['models']['lightgbm']['objective']=='regression_l2'
    parts=fit.prep.load_parts(config);report=json.loads((fit.prep.OUT/'report.json').read_bytes())
    profiles=json.loads((fit.prep.OUT/'train_profiles.json').read_bytes())
    target=pd.read_parquet(fit.prep.auxiliary.OUT/'labels.parquet');checked_cells=0
    for segment,frame in parts.items():
        source=target[target.segment.eq(segment)].reset_index(drop=True)
        np.testing.assert_array_equal(frame[fit.prep.TARGET],source[fit.prep.TARGET])
        assert len(frame)==report['segments'][segment]['rows']
        np.testing.assert_allclose(frame[fit.prep.TARGET].mean(),report['segments'][segment]['target_mean'],atol=1e-14)
        for field in config['fields']:
            values=frame[field].to_numpy(dtype=float)
            assert not np.isinf(values).any();checked_cells+=len(values)
            if segment=='train':
                p=profiles['fields'][field]
                assert p['known']==int(np.isfinite(values).sum()) and p['missing']==int(np.isnan(values).sum())
                assert p['distinct']==frame[field].nunique(dropna=True)
    assert checked_cells==86319*232
    answer=dict(status='PASS_FULL_FRAMEWORK_AUXILIARY_SPEC_NOT_FIT',config_sha256=checked['config_sha256'],
        numeric_cells_checked=checked_cells,rows=86319,fields=232,transformed_dimensions=464,
        segments={s:len(f) for s,f in parts.items()},real_preprocessing_fitted=False,real_model_fits=0,
        refs=dict(config=str(fit.prep.OUT/'config.json'),report=str(fit.prep.OUT/'report.json')))
    write_exclusive(fit.HERE/'auxiliary_fixed_preparation_acceptance_20261018_v1.json',answer)
    return answer


def entry():
    log=io.StringIO();tests=unittest.TextTestRunner(stream=log,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('test_auxiliary_fixed_fit_v1'))
    if not tests.wasSuccessful():raise ValueError(log.getvalue())
    config,parts,refs=fit.preflight()
    answer=dict(status='PASS_AUXILIARY_FIXED_ENTRY_NO_REAL_FIT',tests=tests.testsRun,test_output=log.getvalue(),
        refs=refs,config_sha256=fit.prep.parent.previous.sha(fit.prep.OUT/'config.json'),
        model_fits_later=2,calibration_fits_later=0,real_models_fitted=False,
        source_labels_training_flag_not_mutated=True,proposed_output_cap=fit.prep.FIT_CAP,
        proposed_fit_reservation_available=bytes_used()+fit.prep.FIT_CAP<TOTAL_CAP,
        rows={s:len(f) for s,f in parts.items()})
    write_exclusive(fit.ENTRY,answer)
    return dict(status=answer['status'],tests=tests.testsRun,rows=answer['rows'])


def result():
    if not fit.OUT.exists():return dict(status='REAL_AUXILIARY_FIT_NOT_RUN')
    stop=json.loads((fit.OUT/'stop_record.json').read_bytes());manifest=json.loads((fit.OUT/'manifest.json').read_bytes())
    if stop['status']!='COMPLETE_NOT_DEPLOYED' or fit.prep.parent.previous.sha(fit.OUT/'manifest.json')!=stop['manifest_sha256']:
        raise ValueError('FIT_TERMINAL_INVALID')
    for name,item in manifest['outputs'].items():
        if fit.prep.parent.previous.sha(fit.OUT/name)!=item['sha256']:raise ValueError('OUTPUT_DRIFT:'+name)
    plan=json.loads((fit.OUT/'plan.json').read_bytes())
    for p,expected in plan['inputs'].items():
        if fit.prep.parent.previous.sha(p)!=expected:raise ValueError('INPUT_DRIFT:'+p)
    config=plan['config'];parts=fit.prep.load_parts(config)
    stored=pd.read_parquet(fit.OUT/'predictions.parquet');report=json.loads((fit.OUT/'report.json').read_bytes())
    checked=0
    for algorithm in ['ridge','lightgbm']:
        state=json.loads((fit.OUT/(algorithm+'_preprocessing.json')).read_bytes())
        model=json.loads((fit.OUT/'ridge_model.json').read_bytes()) if algorithm=='ridge' else gzip.decompress((fit.OUT/'lightgbm_model.txt.gz').read_bytes()).decode('utf-8')
        raw=fit.replay(parts,config['fields'],algorithm,state,model)
        for segment,frame in parts.items():
            old=stored[stored.segment.eq(segment)].reset_index(drop=True)
            pd.testing.assert_frame_equal(old[['code',fit.prep.TIME]],frame[['code',fit.prep.TIME]])
            np.testing.assert_allclose(old[algorithm+'_raw'],raw[segment],atol=1e-11,rtol=0)
            y=frame[fit.prep.TARGET].to_numpy();p=np.clip(raw[segment],0,1)
            metric=report['metrics'][segment]['physical_amplitude_metrics'][algorithm]
            np.testing.assert_allclose(metric['mse'],np.mean((p-y)**2),atol=1e-14)
            np.testing.assert_allclose(metric['mae'],np.mean(np.abs(p-y)),atol=1e-14)
            checked+=len(frame)
    answer=dict(status='PASS_SAVED_AUXILIARY_REGRESSION_REPLAY_NOT_PROMOTION',predictions_checked=checked,
        output_bytes=sum(p.stat().st_size for p in fit.OUT.iterdir()),registered_bytes=bytes_used(),cap=TOTAL_CAP,
        http_requests=0,deployed=False)
    write_exclusive(fit.HERE/'auxiliary_fixed_fit_acceptance_20261018_v1.json',answer)
    return answer


if __name__=='__main__':
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--preparation',action='store_true');g.add_argument('--entry',action='store_true');g.add_argument('--result',action='store_true')
    args=parser.parse_args();print(json.dumps(preparation() if args.preparation else entry() if args.entry else result()))
