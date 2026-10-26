"""Freeze C0/C1 direct-warning comparison. No estimator or calibration fit."""
import argparse
from copy import deepcopy
import io
import json
from importlib.metadata import version
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import prepare_multiboard_comparison_v1 as previous
from direct_warning_comparison_v1 import specification, inputs, train_patterns
from read_conditional_board_observations_v2 import read_segment
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE,TIME,SEGMENTS=previous.HERE,previous.TIME,previous.SEGMENTS
OUT=HERE/'direct_warning_comparison_preparation_20261018_v1'
CAP,RESERVE,FIT_CAP=2*1024**2,64*1024,8*1024**2


def prepare():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if bytes_used()+CAP+FIT_CAP>TOTAL_CAP:
        raise ValueError('PREPARATION_AND_PROPOSED_FIT_STORAGE_NOT_FEASIBLE')
    log=io.StringIO()
    tests=unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromNames([
        'test_direct_warning_comparison_v1','test_frozen_observation_sensitivity_v1']))
    if not tests.wasSuccessful() or tests.skipped:
        raise ValueError('ENTRY_TEST_FAILED:'+log.getvalue())
    refs={}

    def bind(path,expected=None):
        path=Path(path);actual=previous.sha(path)
        if expected is not None and actual!=expected:
            raise ValueError('FROZEN_INPUT_DRIFT:'+str(path))
        refs[str(path)]=actual
        return path

    for name in ['prepare_direct_warning_comparison_v1.py','direct_warning_comparison_v1.py',
        'test_direct_warning_comparison_v1.py','test_frozen_observation_sensitivity_v1.py',
        'prepare_multiboard_comparison_v1.py','joint_fit_preparation_v1.py','research_budget_v2.py']:
        bind(HERE/name)
    old_config=json.loads(bind(previous.OUT/'config.json').read_bytes())
    for package,expected in old_config['package_versions'].items():
        if version(package)!=expected:
            raise ValueError('FROZEN_RUNTIME_VERSION_DRIFT:'+package)
    old_report=json.loads(bind(previous.OUT/'report.json').read_bytes())
    population_path=bind(previous.OUT/'population_audit.parquet',
        old_report['outputs']['population_audit.parquet']['sha256'])
    population=pd.read_parquet(population_path)
    receipt=json.loads(bind(HERE/'frozen_observation_sensitivity_acceptance_20261018_v1.json').read_bytes())
    sensitivity=json.loads(bind(HERE/'frozen_observation_sensitivity_20261018_v1/report.json',
        receipt['report_sha256']).read_bytes())
    if receipt['status']!='PASS_FIXED_MODEL_INPUT_SENSITIVITY_NOT_FIT':
        raise ValueError('FIXED_SENSITIVITY_NOT_ACCEPTED')
    reader_receipt=json.loads(bind(HERE/'conditional_board_reader_acceptance_20261018_v2.json').read_bytes())
    bind(HERE/'read_conditional_board_observations_v2.py',reader_receipt['reader_sha256'])
    context_receipt=json.loads(bind(HERE/'conditional_board_observations_acceptance_20261018_v1.json',
        reader_receipt['prior_acceptance_sha256']).read_bytes())
    context=HERE/'conditional_board_observations_20261018_v1'
    context_report=json.loads(bind(context/'report.json',context_receipt['report_sha256']).read_bytes())
    for name,item in context_report['outputs'].items():
        bind(context/name,item['sha256'])
    nh=HERE/'multiboard_observation_repair_20261018_v1'
    nh_report=json.loads(bind(nh/'report.json').read_bytes())
    spec=json.loads(bind(nh/'feature_spec.json',nh_report['outputs']['feature_spec.json']['sha256']).read_bytes())
    new=spec['shared_fields']+spec['own_fields']
    base=old_config['arms']['M1']
    aliases=list(sensitivity['field_mapping'].values())+sensitivity['ignored_untrained_nh_fields']
    arms,mapping=specification(base,new,base+new+aliases)
    # Bind only consumed derived evidence; no repeated raw-minute scan.
    full=HERE/'multiboard_full_features_20261018_v1'
    full_report=json.loads(bind(full/'report.json').read_bytes())
    for name,item in full_report['outputs'].items():
        bind(full/name,item['sha256'])
    for name in ['shared.parquet','own.parquet']:
        bind(nh/name,nh_report['outputs'][name]['sha256'])
    fit=HERE/'multiboard_three_arm_fit_20261018_v1'
    fit_stop=json.loads(bind(fit/'stop_record.json').read_bytes())
    fit_manifest=json.loads(bind(fit/'artifact_manifest.json',fit_stop['manifest_sha256']).read_bytes())
    for name,item in fit_manifest['outputs'].items():
        bind(fit/name,item['sha256'])
    if previous.sha(previous.OUT/'config.json')!=json.loads((fit/'report.json').read_bytes())['config_sha256']:
        raise ValueError('BASELINE_CONFIG_NOT_ORIGINAL')
    reused={}
    for algorithm in ['lr','lightgbm']:
        prefix='M1_'+algorithm
        names=[prefix+'_preprocessing.json',prefix+('_model.json' if algorithm=='lr' else '_model.txt')]
        reused[algorithm]={name:dict(path=str(fit/name),sha256=previous.sha(fit/name)) for name in names}
    bind(Path(old_config['label_contract']))
    config=dict(version='DIRECT_WARNING_FIXED_COMPARISON_V1',status='FIXED_RESEARCH_SPEC_NOT_FIT_AUTHORIZATION',
        fit_authorized=False,http_authorized=False,deployment_authorized=False,
        arms=arms,logical_to_source=mapping,added_fields=new,
        logical_reader='prepare_direct_warning_comparison_v1.load_arm',
        population_reference=dict(path=str(population_path),sha256=previous.sha(population_path),
            policy='KEEP_FROZEN_PRE_REPAIR_COMMON_MASK_FOR_BOTH_ARMS; NO_NEW_FIELD_RESELECTION',
            original_rows=100704,common_rows=86319),
        numeric_dimensions={a:len(f) for a,f in arms.items()},
        numeric_plus_missing_indicator_dimensions={a:len(f)*2 for a,f in arms.items()},
        label=old_config['label'],label_contract=old_config['label_contract'],segments=deepcopy(old_config['segments']),
        framework='daily golden price/time + market + full board distribution/stall + current/captured leaders + individual; human decision',
        core_observation_groups=deepcopy(old_config['core_observation_groups']),
        missing_policy=deepcopy(old_config['missing_policy']),models=deepcopy(old_config['models']),
        selection=deepcopy(old_config['selection']),calibration=deepcopy(old_config['calibration']),
        evaluation=deepcopy(old_config['evaluation']),package_versions=deepcopy(old_config['package_versions']),
        C0_base_policy='REUSE_FROZEN_M1_MAIN_MODELS_AND_PREPROCESSING_ONLY_AFTER_TRAIN_SELECT_VALUE_IDENTITY',
        C0_base_artifacts=reused,C1_base_policy='ONE_FIXED_LR_AND_LIGHTGBM_FIT_ON_TRAIN; SELECT_EARLY_STOP_ONLY',
        calibration_policy='EACH_ARM_NEW_SIGMOID_ON_ITS_OWN_CALIBRATION_PREDICTIONS; SAME_RULE_NO_BASE_REFIT',
        negative_calibration_slope='REPORT_UNCORRECTED; NOT_AUTOMATIC_PROBABILITY_VALIDITY_OR_PROMOTION',
        later_fit_output_cap=FIT_CAP,technical_gate_is_not_statistical_power_or_financial_gate=True)
    config['evaluation']['primary']='paired C1 minus C0 Brier/LogLoss, raw and calibrated; no winner chosen from reused test'
    config['evaluation']['filter_uncertainty']='paired observed-date blocks with common retained denominators; report support dates; no promotion from sparse support'
    config['evaluation']['same_cutoff_ranking']='secondary only; count usable mixed-label cutoffs; not independent duplicated market signals'
    config['evaluation']['old_frozen_calibrator']='audit reference only; not reused as unequal calibration opportunity for C0'
    config['missing_policy']['common_population']=config['population_reference']['policy']
    config['missing_policy']['core_observation_after_repair']='REPORT_LOST_GAINED_ORIGINAL_CORE_GROUPS; KEEP_BOTH_ARMS_PAIRED; NO_POSTHOC_RESELECTION'
    used,outputs=0,{}

    def write(name,body,terminal=False):
        nonlocal used
        reserve=0 if terminal else RESERVE
        if used+len(body)+reserve>CAP or bytes_used()+len(body)+reserve>TOTAL_CAP:
            raise ValueError('SERIALIZED_PREPARATION_BUDGET_STOP')
        if not OUT.exists():
            OUT.mkdir(exist_ok=False)
        with (OUT/name).open('xb') as stream:
            stream.write(body)
        used+=len(body);outputs[name]=dict(bytes=len(body),sha256=previous.sha(OUT/name))

    write('plan.json',previous.encoded(dict(inputs=refs.copy(),tests=tests.testsRun,test_output=log.getvalue(),
        scope='PREPARE_C0_C1_ONLY',cap=CAP,proposed_later_fit_cap=FIT_CAP,fit=False,http_requests=0)))
    try:
        write('config.json',previous.encoded(config))
        profiles,segments,patterns,blockers={}, {}, None, []
        boundaries={r['name']:r for r in config['segments']}
        for segment in SEGMENTS:
            frame=read_segment(segment)
            prior=population[population.proposed_segment_name.eq(segment)].reset_index(drop=True)
            pd.testing.assert_frame_equal(frame[['code',TIME]],prior[['code',TIME]])
            previous.validate_times_labels(frame,segment,boundaries)
            actual_arms,actual_mapping=specification(base,new,list(frame))
            if actual_arms!=arms or actual_mapping!=mapping:
                raise ValueError('READER_SCHEMA_OR_MAPPING_DRIFT')
            mask=prior.common_main_eligible.to_numpy()
            main=previous.summarize(frame,mask)
            if main!=old_report['segments'][segment]['common_main']:
                raise ValueError('FROZEN_POPULATION_CLASS_COUNTS_CHANGED')
            x=inputs(frame,arms['C1'],mapping)
            if segment in ['train','select']:
                np.testing.assert_allclose(x.loc[mask,base].to_numpy(),frame.loc[mask,base].to_numpy(dtype=float),
                                           rtol=0,atol=0,equal_nan=True)
            profile=previous.profile(x,arms['C1'])
            main_profile=previous.profile(x.loc[mask],arms['C1'])
            profiles[segment]=dict(full=profile,common_main=main_profile)
            if segment=='train':
                patterns=train_patterns(x.loc[mask])
                blockers.extend('ALL_NULL_TRAIN:'+f for f in patterns['all_null_columns'])
            if not main['positive'] or not main['negative']:
                blockers.append('NOT_TWO_CLASS:'+segment)
            core_fields=list(dict.fromkeys(f for fields in previous.CORE_GROUPS.values() for f in fields))
            core=frame[core_fields].copy()
            for field in core:
                if field in mapping:
                    core[field]=frame[mapping[field]].to_numpy()
            repaired_core,group_masks=previous.common_population(core)
            support=frame.loc[mask].groupby(TIME).size()
            segments[segment]=dict(full_cohort=previous.summarize(frame,np.ones(len(frame),bool)),common_main=main,
                excluded_from_common=len(frame)-main['rows'],retained_share=main['rows']/len(frame),
                corrected_core_missing_within_frozen_common=int((mask&~repaired_core).sum()),
                corrected_core_gained_outside_frozen_common=int((~mask&repaired_core).sum()),
                missing_by_corrected_core_group={g:int((mask&~m).sum()) for g,m in group_masks.items()},
                comparable_rows=int(support[support>=5].sum()),comparable_cutoffs=int((support>=5).sum()),
                comparable_dates=int(frame.loc[mask&frame[TIME].isin(support[support>=5].index),TIME].floordiv(1000000).nunique()))
        if sum(v['full_cohort']['rows'] for v in segments.values())!=100704 or sum(v['common_main']['rows'] for v in segments.values())!=86319:
            raise ValueError('POPULATION_SIZE_CHANGED')
        write('field_profiles.json',previous.encoded(profiles))
        write('train_patterns.json',previous.encoded(patterns))
        for path,expected in refs.items():
            if previous.sha(path)!=expected:
                raise ValueError('INPUT_CHANGED_DURING_PREPARATION:'+path)
        write('report.json',previous.encoded(dict(status='COMPLETE_FIXED_RESEARCH_PREPARATION_NOT_FIT',
            inputs=refs,outputs=outputs.copy(),tests=tests.testsRun,test_output=log.getvalue(),segments=segments,
            preparation_ready=not blockers,technical_blockers=blockers,
            numeric_dimensions=config['numeric_dimensions'],transformed_dimensions=config['numeric_plus_missing_indicator_dimensions'],
            added_fields=len(new),baseline_source_aliases=sum(mapping[f]!=f for f in base),
            added_source_aliases=sum(mapping[f]!=f for f in new),
            train_constant_columns=len(patterns['constant_nonmissing_columns']),
            train_exact_duplicate_groups=len(patterns['exact_duplicate_groups_including_missing_pattern']),
            real_preprocessing_fitted=False,fit_authorized=False,fit=False,calibration_fit=False,
            labels_changed=False,http_requests=0,proposed_fit_fits_remaining_budget=bytes_used()+FIT_CAP+RESERVE<TOTAL_CAP,
            output_cap=CAP,proposed_later_fit_cap=FIT_CAP)))
        write('manifest.json',previous.encoded(dict(outputs=outputs.copy())))
        write('stop_record.json',previous.encoded(dict(status='COMPLETE_NOT_FIT',
            manifest_sha256=previous.sha(OUT/'manifest.json'))),terminal=True)
        print(json.dumps(dict(preparation_ready=not blockers,blockers=blockers,bytes=used,
            config_sha256=previous.sha(OUT/'config.json'),fit=False)))
    except Exception as exc:
        write('stop_record.json',previous.encoded(dict(status='FAILED_NO_RESTART',error=f'{type(exc).__name__}:{exc}')),terminal=True)
        raise


def check():
    terminal=json.loads((OUT/'stop_record.json').read_bytes())
    if terminal['status']!='COMPLETE_NOT_FIT' or previous.sha(OUT/'manifest.json')!=terminal['manifest_sha256']:
        raise ValueError('PREPARATION_NOT_COMPLETE')
    manifest=json.loads((OUT/'manifest.json').read_bytes())
    for name,item in manifest['outputs'].items():
        if previous.sha(OUT/name)!=item['sha256']:
            raise ValueError('PREPARATION_OUTPUT_DRIFT:'+name)
    report=json.loads((OUT/'report.json').read_bytes())
    for path,expected in report['inputs'].items():
        if previous.sha(path)!=expected:
            raise ValueError('PREPARATION_INPUT_DRIFT:'+path)
    config=json.loads((OUT/'config.json').read_bytes())
    if config['fit_authorized'] or config['http_authorized'] or config['deployment_authorized']:
        raise ValueError('SPEC_IS_NOT_AN_AUTHORIZATION_RECORD')
    return dict(preparation_ready=report['preparation_ready'],technical_blockers=report['technical_blockers'],
                config_sha256=previous.sha(OUT/'config.json'),fit_authorized=False,fit=False)


def load_arm(segment,arm,common=True):
    if not check()['preparation_ready']:
        raise ValueError('PREPARATION_TECHNICAL_BLOCKERS')
    config=json.loads((OUT/'config.json').read_bytes())
    if segment not in SEGMENTS or arm not in config['arms']:
        raise ValueError('UNKNOWN_SEGMENT_OR_ARM')
    reference=config['population_reference']
    if previous.sha(reference['path'])!=reference['sha256']:
        raise ValueError('POPULATION_REFERENCE_DRIFT')
    frame=read_segment(segment)
    population=pd.read_parquet(reference['path'])
    prior=population[population.proposed_segment_name.eq(segment)].reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[['code',TIME]],prior[['code',TIME]])
    mask=prior.common_main_eligible.to_numpy() if common else np.ones(len(frame),bool)
    numeric=inputs(frame.loc[mask],config['arms'][arm],config['logical_to_source']).reset_index(drop=True)
    meta=['code',TIME,'label_end_time','y_tail_1200m','mae_1200m','proposed_segment_name']
    return pd.concat([frame.loc[mask,meta].reset_index(drop=True),numeric],axis=1)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        prepare()
    else:
        answer=check();print(json.dumps(answer));raise SystemExit(0 if answer['preparation_ready'] else 2)
