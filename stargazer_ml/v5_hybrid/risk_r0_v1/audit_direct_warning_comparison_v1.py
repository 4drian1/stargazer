"""Check frozen C0/C1 specification and all real input views; never fit."""
import io
import json
import unittest

import numpy as np
import pandas as pd

import prepare_direct_warning_comparison_v1 as prepare
from direct_warning_comparison_v1 import warning_fields


def run():
    destination=prepare.HERE/'direct_warning_comparison_acceptance_20261018_v1.json'
    if destination.exists():
        raise ValueError('ACCEPTANCE_EXISTS_NO_OVERWRITE')
    checked=prepare.check()
    if not checked['preparation_ready']:
        raise ValueError('PREPARATION_HAS_TECHNICAL_BLOCKERS')
    log=io.StringIO()
    tests=unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromNames([
        'test_direct_warning_comparison_v1','test_frozen_observation_sensitivity_v1']))
    if not tests.wasSuccessful() or tests.skipped:
        raise ValueError('INDEPENDENT_ENTRY_TESTS_FAILED:'+log.getvalue())
    config=json.loads((prepare.OUT/'config.json').read_bytes())
    report=json.loads((prepare.OUT/'report.json').read_bytes())
    profiles=json.loads((prepare.OUT/'field_profiles.json').read_bytes())
    patterns=json.loads((prepare.OUT/'train_patterns.json').read_bytes())
    old_config=json.loads((prepare.previous.OUT/'config.json').read_bytes())
    if config['arms']['C0']!=old_config['arms']['M1'] or config['arms']['C1']!=config['arms']['C0']+warning_fields():
        raise ValueError('ARM_WHITELIST_CHANGED')
    for key in ['models','selection','calibration','segments','label','package_versions']:
        if config[key]!=old_config[key]:
            raise ValueError('FROZEN_EXPERIMENT_PARAMETER_CHANGED:'+key)
    if 'EACH_ARM_NEW_SIGMOID' not in config['calibration_policy']:
        raise ValueError('UNEQUAL_CALIBRATION_OPPORTUNITY')
    population=pd.read_parquet(config['population_reference']['path'])
    inspected=0;arm_rows={};dimension={};new_cov=set(f for f in warning_fields() if f.endswith('_coverage'))
    equivalent=set()
    for group in patterns['exact_duplicate_groups_including_missing_pattern']:
        if set(group)&set(config['arms']['C0']):
            equivalent.update(set(group)&new_cov)
    for segment in prepare.SEGMENTS:
        source=prepare.read_segment(segment)
        stored=population[population.proposed_segment_name.eq(segment)].reset_index(drop=True)
        pd.testing.assert_frame_equal(source[['code',prepare.TIME]],stored[['code',prepare.TIME]])
        mask=stored.common_main_eligible.to_numpy()
        for field in config['arms']['C1']:
            actual=source[config['logical_to_source'][field]].to_numpy(dtype=float)
            for scope,values in [('full',actual),('common_main',actual[mask])]:
                expected=profiles[segment][scope][field]
                if (int(np.isfinite(values).sum())!=expected['finite']
                        or int(np.isnan(values).sum())!=expected['missing']
                        or int(pd.Series(values).nunique(dropna=True))!=expected['distinct_nonmissing']):
                    raise ValueError('FIELD_PROFILE_NOT_REPRODUCED:'+segment+':'+field)
            inspected+=len(actual)
        arm_rows[segment]={}
        for arm in ['C0','C1']:
            frame=prepare.load_arm(segment,arm)
            fields=config['arms'][arm]
            for field in fields:
                wanted=source.loc[mask,config['logical_to_source'][field]].to_numpy(dtype=float)
                np.testing.assert_allclose(frame[field],wanted,atol=0,rtol=0,equal_nan=True)
            meta=['code',prepare.TIME,'label_end_time','y_tail_1200m','mae_1200m','proposed_segment_name']
            pd.testing.assert_frame_equal(frame[meta],source.loc[mask,meta].reset_index(drop=True))
            if len(frame)!=report['segments'][segment]['common_main']['rows']:
                raise ValueError('ARM_POPULATION_CHANGED')
            if segment in ['train','select']:
                np.testing.assert_allclose(frame[config['arms']['C0']],source.loc[mask,config['arms']['C0']],
                                           atol=0,rtol=0,equal_nan=True)
            arm_rows[segment][arm]=len(frame)
            dimension[arm]=len(fields)*2
    fit=prepare.HERE/'multiboard_three_arm_fit_20261018_v1'
    old_manifest=json.loads((fit/'artifact_manifest.json').read_bytes())
    for name,item in old_manifest['outputs'].items():
        if prepare.previous.sha(fit/name)!=item['sha256']:
            raise ValueError('OLD_MODEL_ARTIFACT_CHANGED')
    d2=prepare.HERE/'postclose_repaired_research_fit_20261017_v1/model.txt'
    if prepare.previous.sha(d2)!='5d8402cd530cb83778d901e33eb87788f21ebc8619414f335b6d56905a81e005':
        raise ValueError('D2_CHANGED')
    result=dict(status='PASS_DIRECT_WARNING_RESEARCH_SPEC_NOT_FIT',tests=tests.testsRun,
        test_output=log.getvalue(),config_sha256=checked['config_sha256'],arm_common_rows=arm_rows,
        logical_cells_profiled=inspected,transformed_dimensions=dimension,
        source_aliases={a:sum(config['logical_to_source'][f]!=f for f in fields) for a,fields in config['arms'].items()},
        new_warning_rate_columns=len(warning_fields())-len(new_cov),new_warning_coverage_columns=len(new_cov),
        new_coverage_columns_exactly_equal_to_baseline_in_train=len(equivalent),
        interpretation='PREDICTIVE_FIELD_PACKAGE_COMPARISON_NOT_ISOLATED_CAUSAL_EFFECT_OF_SIGNAL_LOGIC',
        interpretation_limitations=['duplicate columns alter effective LR regularization',
            'duplicate columns may alter LightGBM feature-sampling opportunities',
            'keep registered fields; no posthoc selection from reused test'],
        old_model_artifacts_unchanged=len(old_manifest['outputs']),d2_unchanged=True,
        real_preprocessing_fitted=False,fit_authorized=False,fit=False,calibration_fit=False,http_requests=0,
        report_sha256=prepare.previous.sha(prepare.OUT/'report.json'),
        auditor_sha256=prepare.previous.sha(prepare.HERE/'audit_direct_warning_comparison_v1.py'))
    body=prepare.previous.encoded(result)
    used=sum(p.stat().st_size for p in prepare.OUT.iterdir() if p.is_file())
    if used+len(body)+prepare.RESERVE>prepare.CAP or prepare.bytes_used()+len(body)+prepare.RESERVE>prepare.TOTAL_CAP:
        raise ValueError('ACCEPTANCE_BUDGET_STOP')
    with destination.open('xb') as stream:
        stream.write(body)
    print(json.dumps({k:v for k,v in result.items() if k!='test_output'}))


if __name__=='__main__':
    run()
