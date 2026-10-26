"""Frozen historical development folds, exposure ledger and preflight; no fitting."""
import json
import io
from pathlib import Path
import numpy as np
import pandas as pd
from finalize_postclose_cohort_v3 import corrected_masks
from read_postclose_daily_context_repair_v1 import read_repaired_population
from inventory_ths_stockdb_cache_v1 import HERE,digest
from postclose_repaired_training_v1 import OUT as FIT,FIRST,prepare
from research_budget_v2 import TOTAL_CAP,bytes_used

OUT=HERE/'postclose_temporal_validation_plan_20261017_v1'
SPEC=HERE/'contracts/POSTCLOSE_TEMPORAL_VALIDATION_PLAN_V1.json'
CAP=2*1024**2


def assign_segments(data,bounds,eligible):
    if data[['code','day']].duplicated().any(): raise ValueError('DUPLICATE_POPULATION_KEYS')
    stages=['train','select','calibration','test']
    if list(bounds)!=stages: raise ValueError('SEGMENT_ORDER_DRIFT')
    previous=None
    for start,end in bounds.values():
        if start>end or (previous is not None and start<=previous): raise ValueError('OVERLAPPING_OR_REVERSED_SEGMENTS')
        previous=end
    out=data[['code','day']].copy();out['segment']=None;out['eligible']=False;out['purged']=False
    out['reason']='OUTSIDE_FOLD_DATES'
    for segment,(start,end) in bounds.items():
        assigned=data.day.between(start,end)
        known=assigned&eligible
        future=data.label_end_time.gt(data.decision_ts_assumed_end)
        within=data.label_end_time.le(end*1000000+235959)
        out.loc[assigned,'segment']=segment
        out.loc[assigned,'reason']='LABEL_OR_REQUIRED_FAMILY_UNAVAILABLE'
        out.loc[known&~future,'reason']='INVALID_OR_UNKNOWN_LABEL_END'
        out.loc[known&future&~within,'purged']=True
        out.loc[known&future&~within,'reason']='CROSS_SEGMENT_LABEL'
        out.loc[known&future&within,'eligible']=True
        out.loc[known&future&within,'reason']='ELIGIBLE_FOR_RESEARCH_SEGMENT'
    return out


def run():
    if OUT.exists(): raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    fields,policy,_,_=prepare()
    data,repair=read_repaired_population()
    spec=json.loads(SPEC.read_bytes())
    refs={str(SPEC):digest(SPEC.read_bytes()),str(Path(__file__)):digest(Path(__file__).read_bytes()),
          str(HERE/'test_temporal_validation_preparation_v1.py'):digest((HERE/'test_temporal_validation_preparation_v1.py').read_bytes()),
          str(HERE/'read_postclose_daily_context_repair_v1.py'):digest((HERE/'read_postclose_daily_context_repair_v1.py').read_bytes()),
          str(HERE/'postclose_daily_context_repair_20261017_v1/manifest.json'):digest((HERE/'postclose_daily_context_repair_20261017_v1/manifest.json').read_bytes())}
    expected=json.loads((FIT/'comparison_and_readback.json').read_bytes())
    refs.update({str(FIT/n):h for n,h in expected['new_fit_sha256'].items()})
    first=json.loads((FIRST/'independent_readback_audit.json').read_bytes())
    refs.update({str(FIRST/n):h for n,h in first['audited_original_sha256'].items()})
    refs[str(HERE/'postclose_research_policy_v1.json')]=digest((HERE/'postclose_research_policy_v1.json').read_bytes())
    for path,sha in refs.items():
        if digest(Path(path).read_bytes())!=sha: raise ValueError('FROZEN_INPUT_DRIFT:'+path)
    family=np.logical_and.reduce(list(corrected_masks(data).values()))
    base=data.label_state.eq('CONDITIONAL_NEXT_SESSION_COMPLETE')&family
    plans={};payloads={}
    for name,bounds in spec['folds'].items():
        assigned=assign_segments(data,bounds,base)
        summary={};gates={}
        for segment in bounds:
            frame=data[assigned.eligible&assigned.segment.eq(segment)]
            entry=dict(rows=len(frame),dates=int(frame.day.nunique()),stocks=int(frame.code.nunique()),
                first_day=int(frame.day.min()) if len(frame) else None,
                last_day=int(frame.day.max()) if len(frame) else None,
                label_end_max=int(frame.label_end_time.max()) if len(frame) else None,
                all_null_fields=frame[fields].columns[frame[fields].isna().all()].tolist(),
                any_infinite=bool(np.isinf(frame[fields].to_numpy(float)).any()))
            summary[segment]=entry
            gates[segment]={key:entry[key]>=v for key,v in policy['sample_gates'][segment].items()}
            gates[segment]['finite_features']=not entry['any_infinite']
            if segment=='train': gates[segment]['no_all_null_train_field']=not entry['all_null_fields']
        plans[name]=dict(bounds=bounds,summary=summary,gates=gates,
            engineering_ready=all(all(g.values()) for g in gates.values()),
            reason_counts={str(k):int(v) for k,v in assigned.reason.value_counts().items()},
            historical_development_only=True,independent_blind_test=False,fit_authorized=False)
        sink=io.BytesIO();assigned.to_parquet(sink,index=False,compression='zstd');payloads[name+'_assignment.parquet']=sink.getvalue()
    exposure=data[['code','day']].copy();exposure['reconstruction_inputs_previously_accessed']=True
    for label,directory in [('first',FIRST),('second',FIT)]:
        prior=pd.read_parquet(directory/'predictions.parquet',columns=['code','day','segment'])
        exposure=exposure.merge(prior.rename(columns={'segment':label+'_fit_segment'}),on=['code','day'],how='left',validate='one_to_one')
    exposure['test_prediction_previously_evaluated']=exposure[['first_fit_segment','second_fit_segment']].eq('test').any(axis=1)
    sink=io.BytesIO();exposure.to_parquet(sink,index=False,compression='zstd');payloads['exposure_ledger.parquet']=sink.getvalue()
    report=dict(status='DEVELOPMENT_SPLITS_PREPARED_NO_FIT',inputs=refs,fields=fields,
        field_contract_previously_developed_not_fold_discovered=True,
        folds=plans,old_original_purge_not_inherited=True,
        exposure=dict(population_rows=len(data),accessed_days=int(data.day.nunique()),last_previously_studied_day=int(data.day.max()),
            evaluated_test_key_union=int(exposure.test_prediction_previously_evaluated.sum()),
            note='Accessed reconstruction evidence is not equivalent to model training; previously evaluated test cannot become a new blind test'),
        independent_validation_ready=False,
        independent_blockers=['INDEPENDENT_INPUT_UNBOUND','INDEPENDENCE_AND_AVAILABILITY_EVIDENCE_NOT_ACCEPTED','FINANCIAL_EFFECT_AND_STOPPING_RULES_NOT_SIGNED','SEPARATE_VALIDATION_EXECUTION_AUTHORIZATION_REQUIRED'],
        actual_fit=False,collection_activated=False,deployment_authorized=False,
        output_hashes={n:dict(bytes=len(b),sha256=digest(b)) for n,b in payloads.items()})
    payloads['report.json']=json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8')
    if sum(map(len,payloads.values()))+65536>CAP or bytes_used()+sum(map(len,payloads.values()))+65536>TOTAL_CAP: raise ValueError('PREPARATION_BUDGET_STOP')
    for path,sha in refs.items():
        if digest(Path(path).read_bytes())!=sha: raise ValueError('INPUT_CHANGED_DURING_PREPARATION')
    OUT.mkdir(exist_ok=False)
    for name,body in payloads.items():
        with (OUT/name).open('xb') as stream:stream.write(body)
    for name in spec['folds']:
        actual=pd.read_parquet(OUT/(name+'_assignment.parquet'))
        pd.testing.assert_frame_equal(actual[['code','day']],data[['code','day']])
        for segment,(_,end) in spec['folds'][name].items():
            frame=data[actual.eligible&actual.segment.eq(segment)]
            if not frame.label_end_time.le(end*1000000+235959).all(): raise ValueError('PURGE_READBACK_FAIL')
    print(json.dumps(dict(status=report['status'],folds=plans,exposure=report['exposure'],independent_validation_ready=False),ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':run()
