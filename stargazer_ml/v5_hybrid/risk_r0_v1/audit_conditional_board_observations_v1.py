"""Read back sparse observation patch against frozen baseline and coverage proof."""
import json
import io
import unittest

import numpy as np
import pandas as pd

import build_conditional_board_observations_v1 as build
from read_conditional_board_observations_v1 import attach


def run():
    destination=build.HERE/'conditional_board_observations_acceptance_20261018_v1.json'
    if destination.exists():
        raise ValueError('ACCEPTANCE_EXISTS_NO_OVERWRITE')
    log=io.StringIO()
    reader_tests=unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromName(
        'test_read_conditional_board_observations_v1'))
    if not reader_tests.wasSuccessful() or reader_tests.skipped:
        raise ValueError('READER_TESTS_FAILED:'+log.getvalue())
    out=build.OUT
    report=json.loads((out/'report.json').read_bytes())
    manifest=json.loads((out/'manifest.json').read_bytes())
    terminal=json.loads((out/'stop_record.json').read_bytes())
    if terminal['status']!='COMPLETE_NOT_FIT' or build.old.prep.sha(out/'manifest.json')!=terminal['manifest_sha256']:
        raise ValueError('INVALID_TERMINAL')
    for name,item in manifest['outputs'].items():
        if build.old.prep.sha(out/name)!=item['sha256']:
            raise ValueError('OUTPUT_DRIFT:'+name)
    for path,expected in report['inputs'].items():
        if build.old.prep.sha(path)!=expected:
            raise ValueError('INPUT_DRIFT:'+path)
    parent,_=build.metadata()
    for path,expected in parent['inputs'].items():
        if build.old.prep.sha(path)!=expected:
            raise ValueError('FROZEN_PARENT_CONTROL_DRIFT:'+path)
    diff=pd.read_parquet(out/'observation_differences.parquet')
    if diff.duplicated(['code',build.TIME,'feature']).any() or len(diff)!=report['changed_cells']:
        raise ValueError('INVALID_SPARSE_KEYS')
    baseline=[]
    for segment in build.old.prep.SEGMENTS:
        f=build.read_segment(segment).assign(segment=segment)
        side=build.old.read_context(segment).drop(columns='segment')
        baseline.append(f.merge(side,on=['code',build.TIME],validate='one_to_one'))
    baseline=pd.concat(baseline,ignore_index=True)
    if len(baseline)!=100704 or baseline.duplicated(['code',build.TIME]).any():
        raise ValueError('POPULATION_CHANGED')
    key_index=pd.MultiIndex.from_frame(baseline[['code',build.TIME]])
    shared=baseline.drop_duplicates(build.TIME).set_index(build.TIME)
    checked=0
    for (code,feature),group in diff.groupby(['code','feature'],sort=False):
        if feature not in baseline or not feature.startswith(('mb','own','nh_mb','nh_own')):
            raise ValueError('PATCH_OUTSIDE_OBSERVATION_SCHEMA')
        if code=='__SHARED__':
            if feature.startswith(('own','nh_own')):
                raise ValueError('OWN_FIELD_IN_SHARED_PATCH')
            expected=shared.loc[group[build.TIME],feature].to_numpy(dtype=float)
        else:
            if not feature.startswith(('own','nh_own')):
                raise ValueError('SHARED_FIELD_IN_OWN_PATCH')
            pos=key_index.get_indexer(pd.MultiIndex.from_frame(group[['code',build.TIME]]))
            if (pos<0).any():
                raise ValueError('PATCH_OUTSIDE_ORIGINAL_POPULATION')
            expected=baseline[feature].to_numpy(dtype=float)[pos]
        np.testing.assert_allclose(group.old_value,expected,atol=1e-12,rtol=0,equal_nan=True)
        if np.isclose(group.old_value,group.new_value,equal_nan=True,atol=1e-12,rtol=0).any():
            raise ValueError('NONCHANGE_IN_SPARSE_PATCH')
        if np.isinf(group.new_value).any():
            raise ValueError('INFINITE_NEW_VALUE')
        checked+=len(group)
    gain=pd.read_parquet(build.binding.OUT/'coverage_increment.parquet')
    d=diff[diff.code.eq('__SHARED__') & diff.feature.eq('mb15_member_coverage')].set_index(build.TIME)
    delta=(d.new_value-d.old_value).reindex(gain[build.TIME],fill_value=0.).to_numpy()
    np.testing.assert_allclose(delta,gain.conditional_research_gain,atol=1e-12,rtol=0)
    fit=build.HERE/'multiboard_three_arm_fit_20261018_v1'
    fitted=json.loads((fit/'artifact_manifest.json').read_bytes())['outputs']
    for name,item in fitted.items():
        if build.old.prep.sha(fit/name)!=item['sha256']:
            raise ValueError('MODEL_CHANGED')
    d2=build.HERE/'postclose_repaired_research_fit_20261017_v1/model.txt'
    if build.old.prep.sha(d2)!='5d8402cd530cb83778d901e33eb87788f21ebc8619414f335b6d56905a81e005':
        raise ValueError('D2_CHANGED')
    focus=[]
    for frequency in (5,15):
        focus.extend([f'mb{frequency}_member_coverage',f'own{frequency}_member_coverage',
            f'nh_mb{frequency}_sse_positive_common_nonpositive_rate',
            f'nh_own{frequency}_sse_positive_common_nonpositive_rate',
            f'own{frequency}_current_change_negative_rate',
            f'own{frequency}_captured_change_negative_rate'])
    small=baseline[['code',build.TIME,*focus]]
    context=attach(small,diff[diff.feature.isin(focus)])
    population=pd.read_parquet(build.old.prep.OUT/'population_audit.parquet')
    context=context.merge(population[['code',build.TIME,'proposed_segment_name','common_main_eligible']],
                          on=['code',build.TIME],validate='one_to_one')
    impact={}
    for segment in build.old.prep.SEGMENTS:
        f=context[context.proposed_segment_name.eq(segment)&context.common_main_eligible]
        impact[segment]={}
        for field in focus:
            after=f['ar_'+field] if 'ar_'+field in f else f[field]
            impact[segment][field]=dict(rows=len(f),old_known=int(f[field].notna().sum()),
                new_known=int(after.notna().sum()),old_positive=int(f[field].gt(0).sum()),
                new_positive=int(after.gt(0).sum()),old_mean=float(f[field].mean()),new_mean=float(after.mean()))
    result=dict(status='PASS_CONDITIONAL_BOARD_OBSERVATIONS_NOT_FIT',tests=report['tests'],
        reader_tests=reader_tests.testsRun,reader_test_output=log.getvalue(),
        reader_sha256=build.old.prep.sha(build.HERE/'read_conditional_board_observations_v1.py'),
        original_population_rows=len(baseline),sparse_old_values_verified=checked,
        input_hashes_verified=len(report['inputs']),parent_controls_verified=len(parent['inputs']),
        coverage_proof_cutoffs=len(gain),real_prefix_frequencies=report['real_prefix_frequencies'],
        fixed_observation_impact=impact,
        old_model_artifacts_unchanged=len(fitted),d2_unchanged=True,fit=False,http_requests=0,
        report_sha256=build.old.prep.sha(out/'report.json'),
        auditor_sha256=build.old.prep.sha(build.HERE/'audit_conditional_board_observations_v1.py'))
    body=build.old.prep.encoded(result)
    size=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    if size+len(body)+build.RESERVE>build.CAP or build.old.bytes_used()+len(body)+build.RESERVE>build.old.TOTAL_CAP:
        raise ValueError('ACCEPTANCE_BUDGET_STOP')
    with destination.open('xb') as stream:
        stream.write(body)
    print(json.dumps(result))


if __name__=='__main__':
    run()
