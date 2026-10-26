"""Fixed M1 weights, train transform and calibration; paired input-only sensitivity."""
from importlib.metadata import version
import io
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import prepare_multiboard_comparison_v1 as prep
from multiboard_three_arm_fit_v1 import replay
from multiboard_fit_evaluation_v1 import probabilities
from frozen_observation_sensitivity_v1 import mapped_inputs, compare
from read_conditional_board_observations_v2 import read_segment
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE, TIME = prep.HERE, prep.TIME
FIT = HERE/'multiboard_three_arm_fit_20261018_v1'
OUT = HERE/'frozen_observation_sensitivity_20261018_v1'
CAP, RESERVE = 4*1024**2, 64*1024


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if bytes_used()+CAP>TOTAL_CAP:
        raise ValueError('REGISTERED_BUDGET_RESERVATION_FAILED')
    log=io.StringIO()
    tests=unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromName(
        'test_frozen_observation_sensitivity_v1'))
    if not tests.wasSuccessful() or tests.skipped:
        raise ValueError('ENTRY_TESTS_FAILED:'+log.getvalue())
    refs={}

    def bind(path,expected=None):
        path=Path(path);actual=prep.sha(path)
        if expected is not None and actual!=expected:
            raise ValueError('FROZEN_INPUT_DRIFT:'+str(path))
        refs[str(path)]=actual
        return path

    for name in ['evaluate_frozen_observation_sensitivity_v1.py','frozen_observation_sensitivity_v1.py',
        'test_frozen_observation_sensitivity_v1.py','multiboard_three_arm_fit_v1.py',
        'multiboard_fit_evaluation_v1.py','joint_fit_preparation_v1.py',
        'prepare_multiboard_comparison_v1.py','risk_fit_eval_v1.py','research_budget_v2.py']:
        bind(HERE/name)
    config=json.loads(bind(prep.OUT/'config.json').read_bytes())
    for package,expected in config['package_versions'].items():
        if version(package)!=expected:
            raise ValueError('FROZEN_RUNTIME_VERSION_DRIFT:'+package)
    terminal=json.loads(bind(FIT/'stop_record.json').read_bytes())
    if terminal['status']!='COMPLETE_RESEARCH_NOT_DEPLOYMENT':
        raise ValueError('FROZEN_FIT_NOT_COMPLETE')
    frozen=json.loads(bind(FIT/'artifact_manifest.json',terminal['manifest_sha256']).read_bytes())
    for name,item in frozen['outputs'].items():
        bind(FIT/name,item['sha256'])
    fit_report=json.loads((FIT/'report.json').read_bytes())
    if prep.sha(prep.OUT/'config.json')!=fit_report['config_sha256']:
        raise ValueError('FROZEN_CONFIG_DRIFT')
    previous=pd.read_parquet(FIT/'predictions.parquet')
    population=pd.read_parquet(bind(prep.OUT/'population_audit.parquet'))
    reader_acceptance=json.loads(bind(HERE/'conditional_board_reader_acceptance_20261018_v2.json').read_bytes())
    if reader_acceptance['status']!='PASS_BATCH_READER_IDENTICAL_TO_V1_NOT_FIT':
        raise ValueError('READER_NOT_ACCEPTED')
    bind(HERE/'read_conditional_board_observations_v2.py',reader_acceptance['reader_sha256'])
    accepted=json.loads(bind(HERE/'conditional_board_observations_acceptance_20261018_v1.json',
        reader_acceptance['prior_acceptance_sha256']).read_bytes())
    context=HERE/'conditional_board_observations_20261018_v1'
    context_report=json.loads(bind(context/'report.json',accepted['report_sha256']).read_bytes())
    context_stop=json.loads(bind(context/'stop_record.json').read_bytes())
    if context_stop['status']!='COMPLETE_NOT_FIT':
        raise ValueError('CONTEXT_NOT_COMPLETE')
    cm=json.loads(bind(context/'manifest.json',context_stop['manifest_sha256']).read_bytes())
    for name,item in cm['outputs'].items():
        bind(context/name,item['sha256'])
    for name in ['read_conditional_board_observations_v1.py','read_multiboard_full_features_v1.py',
                 'repair_multiboard_observation_v1.py']:
        bind(HERE/name)
    # Consume frozen derived snapshots; no raw minute reads or recomputation.
    full=HERE/'multiboard_full_features_20261018_v1'
    full_report=json.loads(bind(full/'report.json').read_bytes())
    for name,item in full_report['outputs'].items():
        bind(full/name,item['sha256'])
    nh=HERE/'multiboard_observation_repair_20261018_v1'
    nh_report=json.loads(bind(nh/'report.json').read_bytes())
    for name in ['shared.parquet','own.parquet']:
        bind(nh/name,nh_report['outputs'][name]['sha256'])
    fields=config['arms']['M1']
    artifacts={}
    for algorithm in ['lr','lightgbm']:
        prefix='M1_'+algorithm
        state=json.loads((FIT/(prefix+'_preprocessing.json')).read_bytes())
        calibration=json.loads((FIT/(prefix+'_calibration.json')).read_bytes())
        model=json.loads((FIT/(prefix+'_model.json')).read_bytes()) if algorithm=='lr' else (FIT/(prefix+'_model.txt')).read_text(encoding='utf-8')
        if state['columns']!=fields or calibration['classes']!=[0,1]:
            raise ValueError('FROZEN_FEATURE_OR_CLASS_ORDER_CHANGED')
        if algorithm=='lr' and model['classes']!=[0,1]:
            raise ValueError('FROZEN_CLASS_ORDER_CHANGED')
        artifacts[algorithm]=(state,model,calibration)
    used,outputs=0,{}

    def write(name,body,terminal=False):
        nonlocal used
        reserve=0 if terminal else RESERVE
        if used+len(body)+reserve>CAP or bytes_used()+len(body)+reserve>TOTAL_CAP:
            raise ValueError('SERIALIZED_BUDGET_STOP')
        if not OUT.exists():
            OUT.mkdir(exist_ok=False)
        with (OUT/name).open('xb') as stream:
            stream.write(body)
        used+=len(body)
        outputs[name]=dict(bytes=len(body),sha256=prep.sha(OUT/name))

    write('plan.json',prep.encoded(dict(inputs=refs.copy(),cap=CAP,tests=tests.testsRun,test_output=log.getvalue(),
        scope='M1_FIXED_WEIGHTS_PREPROCESSING_AND_CALIBRATION_INPUT_ONLY_SENSITIVITY',
        fields=fields,evaluation=config['evaluation'],fit=False,recalibration=False,http_requests=0,
        new_nh_fields_are_not_model_inputs=True,old_test_reused=True)))
    try:
        reports,changed_rows,monthly={},[],{}
        baseline_replayed=0;max_replay_error=0.;mapping=None;ignored=None
        for segment in prep.SEGMENTS:
            full_frame=read_segment(segment)
            bound=population[population.proposed_segment_name.eq(segment)].reset_index(drop=True)
            pd.testing.assert_frame_equal(full_frame[['code',TIME]],bound[['code',TIME]])
            frame=full_frame.loc[bound.common_main_eligible.to_numpy()].reset_index(drop=True)
            target=previous[previous.segment.eq(segment)].reset_index(drop=True)
            identity=['code',TIME,'label_end_time','y_tail_1200m','mae_1200m']
            pd.testing.assert_frame_equal(frame[identity],target[identity])
            old,new,current_mapping,changed=mapped_inputs(frame,fields)
            if mapping is not None and mapping!=current_mapping:
                raise ValueError('SEGMENT_DEPENDENT_FIELD_MAPPING')
            mapping=current_mapping
            ignored=sorted(c for c in frame if c.startswith('ar_nh_'))
            if segment in ['train','select'] and changed.any():
                raise ValueError('TRAIN_OR_SELECT_INPUT_CHANGED')
            # The baseline/golden-only arms have no affected fields, not refitted.
            unaffected={}
            for arm in ['B0','B1']:
                _,_,arm_mapping,arm_change=mapped_inputs(frame,config['arms'][arm])
                if arm_change.any():
                    raise ValueError('UNEXPECTED_BASELINE_ARM_INPUT_CHANGE')
                unaffected[arm]=dict(mapped_fields=len(arm_mapping),changed_cells=0)
            segment_report=dict(rows=len(frame),mapped_fields=len(mapping),changed_input_cells=int(changed.sum()),
                changed_input_rows=int(changed.any(axis=1).sum()),unaffected_arms=unaffected,models={})
            months=frame[TIME].to_numpy()//100000000
            for algorithm,(state,model,calibration) in artifacts.items():
                baseline=replay({segment:old},fields,algorithm,state,model,calibration,config['calibration']['epsilon'])[segment]
                conditional=replay({segment:new},fields,algorithm,state,model,calibration,config['calibration']['epsilon'])[segment]
                segment_report['models'][algorithm]=dict(calibration_slope_unchanged=calibration['coef'][0],states={})
                for status in ['raw','calibrated']:
                    a,b=baseline[status],conditional[status]
                    expected=target[f'M1_{algorithm}_{status}'].to_numpy()
                    np.testing.assert_allclose(a,expected,atol=1e-11,rtol=0)
                    baseline_replayed+=len(a)
                    max_replay_error=max(max_replay_error,float(np.abs(a-expected).max()))
                    if segment in ['train','select']:
                        np.testing.assert_allclose(b,a,atol=1e-12,rtol=0)
                    result=compare(frame,a,b,config['evaluation'])
                    original_metrics=fit_report['metrics'][segment]['probability_metrics'][f'M1_{algorithm}_{status}']
                    for metric in ['brier','log_loss','roc_auc','pr_auc_average_precision','predicted_risk']:
                        np.testing.assert_allclose(result['original'][metric],original_metrics[metric],atol=1e-11,rtol=0)
                    segment_report['models'][algorithm]['states'][status]=result
                    take=np.abs(b-a)>1e-12
                    if take.any():
                        delta=frame.loc[take,['code',TIME]].copy()
                        delta['segment'],delta['algorithm'],delta['state']=segment,algorithm,status
                        delta['original_probability'],delta['conditional_probability']=a[take],b[take]
                        changed_rows.append(delta)
                    for month in np.unique(months):
                        mask=months==month
                        key=f'{segment}:{month}'
                        item=monthly.setdefault(key,dict(segment=segment,month=int(month),rows=int(mask.sum()),
                            dates=int(frame.loc[mask,TIME].floordiv(1000000).nunique()),models={}))
                        item['models'][algorithm+'_'+status]=dict(
                            original=probabilities(frame.loc[mask,'y_tail_1200m'],a[mask]),
                            conditional=probabilities(frame.loc[mask,'y_tail_1200m'],b[mask]),
                            changed_rows=int(take[mask].sum()))
                print(json.dumps(dict(stage='frozen_model_pair_replayed',segment=segment,algorithm=algorithm)),flush=True)
            reports[segment]=segment_report
        if baseline_replayed!=86319*4:
            raise ValueError('COMMON_POPULATION_REPLAY_COUNT_CHANGED')
        delta=pd.concat(changed_rows,ignore_index=True) if changed_rows else pd.DataFrame(columns=[
            'code',TIME,'segment','algorithm','state','original_probability','conditional_probability'])
        if delta.duplicated(['code',TIME,'algorithm','state']).any():
            raise ValueError('DUPLICATE_PREDICTION_DIFFERENCE')
        sink=io.BytesIO();delta.to_parquet(sink,index=False,compression='zstd')
        write('prediction_differences.parquet',sink.getvalue())
        write('monthly.json',prep.encoded(monthly))
        for path,expected in refs.items():
            if prep.sha(path)!=expected:
                raise ValueError('INPUT_CHANGED_DURING_RUN:'+path)
        write('report.json',prep.encoded(dict(status='COMPLETE_FIXED_MODEL_INPUT_SENSITIVITY_NOT_PROMOTION',
            inputs=refs,outputs=outputs.copy(),tests=tests.testsRun,test_output=log.getvalue(),
            common_rows=86319,original_baseline_probabilities_replayed=baseline_replayed,
            maximum_baseline_replay_error=max_replay_error,field_mapping=mapping,
            ignored_untrained_nh_fields=ignored,segments=reports,prediction_difference_cells=len(delta),
            paired_loss_sign='conditional minus original; negative favors conditional',
            filtering_pair_difference_ci_not_computed=True,
            limitations=['restricted historical reconstruction','old test reused for research',
                'overlapping labels','conditional source/action assumptions','no new threshold or model selection'],
            fit=False,recalibration=False,model_parameters_changed=False,labels_changed=False,
            financial_gate_passed=False,deployment_authorized=False,pnl_evaluated=False,http_requests=0)))
        write('manifest.json',prep.encoded(dict(outputs=outputs.copy())))
        write('stop_record.json',prep.encoded(dict(status='COMPLETE_NOT_FIT_OR_DEPLOYMENT',
            manifest_sha256=prep.sha(OUT/'manifest.json'))),terminal=True)
        print(json.dumps(dict(bytes=used,replayed=baseline_replayed,difference_cells=len(delta),
                              mapped_fields=len(mapping),ignored_nh=len(ignored),fit=False)))
    except Exception as exc:
        write('stop_record.json',prep.encoded(dict(status='FAILED_NO_RESTART',error=f'{type(exc).__name__}:{exc}')),terminal=True)
        raise


if __name__=='__main__':
    run()
