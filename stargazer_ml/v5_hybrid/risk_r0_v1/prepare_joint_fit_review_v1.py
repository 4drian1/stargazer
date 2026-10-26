"""Materialize the labeled review cohort, not an approved training dataset."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from inventory_ths_stockdb_cache_v1 import HERE,digest
from build_joint_candidate_context_v1 import SOURCE,OUT as CONTEXT
from read_decision_after_labels_v2 import OUT as LABELS,read_stored_part
from read_decision_after_labels_v1 import attach_label_times
from joint_fit_preparation_v1 import whitelist,derive,group_observation_masks,numeric,cutoff_support,DERIVED

CONFIG=HERE/'contracts/JOINT_RISK_FIT_REVIEW_V1.json'
OUT=HERE/'joint_fit_review_20261017_v1'
CAP,RESERVE=32*1024**2,1024**2


def run():
    if OUT.exists():raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    from research_budget_v2 import bytes_used,TOTAL_CAP
    if bytes_used()+RESERVE>TOTAL_CAP:raise ValueError('REGISTERED_BUDGET_STOP')
    spec=json.loads(CONFIG.read_bytes());fields=whitelist(spec)
    if spec['fit_authorized'] or spec['status']!='DRAFT_FOR_REVIEW_NOT_SIGNED':raise ValueError('NOT_A_REVIEW_CONFIG')
    frozen=[CONFIG,SOURCE/'manifest.json',CONTEXT/'manifest.json',LABELS/'manifest.json',
        LABELS/'readback_acceptance.json',Path(__file__),HERE/'joint_fit_preparation_v1.py',
        HERE/'test_joint_fit_preparation_v1.py',HERE/'read_decision_after_labels_v2.py']
    refs={str(p):digest(p.read_bytes()) for p in frozen}
    source,context,labels=[json.loads((p/'manifest.json').read_bytes()) for p in [SOURCE,CONTEXT,LABELS]]
    for directory in [SOURCE,CONTEXT,LABELS]:
        if json.loads((directory/'stop_record.json').read_bytes())['status']!='COMPLETE':raise ValueError('INPUT_NOT_COMPLETE')
    if context['source_manifest_sha256']!=refs[str(SOURCE/'manifest.json')]:raise ValueError('CONTEXT_SOURCE_DRIFT')
    if labels['inputs'][str(SOURCE/'manifest.json')]!=refs[str(SOURCE/'manifest.json')]:raise ValueError('LABEL_SOURCE_DRIFT')
    shared_body=(SOURCE/'shared_market_and_leaders.parquet').read_bytes()
    if digest(shared_body)!=source['output_hashes']['shared_market_and_leaders.parquet']['sha256']:raise ValueError('SHARED_INPUT_DRIFT')
    shared=pq.read_table(pa.BufferReader(shared_body)).to_pandas();times=shared.decision_ts_assumed_end.to_numpy()
    label_contract=HERE/'contracts/INTRADAY_DECISION_AFTER_LABEL_V1.json'
    if digest(label_contract.read_bytes())!=labels['inputs'][str(label_contract)]:raise ValueError('LABEL_CONTRACT_DRIFT')
    refs[str(label_contract)]=digest(label_contract.read_bytes())
    segments=json.loads(label_contract.read_bytes())['proposed_segments']
    extra=[f'sse_15m_time_fraction_distance_{r}' for r in ['0_236','0_382','0_5','0_618','0_786']]
    need=set(fields)-set(DERIVED);need.update(extra);need.update(['code','cutoff_index','decision_ts_assumed_end'])
    OUT.mkdir(exist_ok=False);used=0;outputs={};buffers=defaultdict(list);bound_parts=[]
    def write(name,body,terminal=False):
        nonlocal used
        reserve=0 if terminal else RESERVE
        if used+len(body)+reserve>CAP or bytes_used()+len(body)+reserve>TOTAL_CAP:raise ValueError('REVIEW_BUDGET_STOP')
        with (OUT/name).open('xb') as stream:stream.write(body)
        used+=len(body);outputs[name]=dict(bytes=len(body),sha256=digest(body))
    status,error='COMPLETE',None
    try:
        write('plan.json',json.dumps(dict(inputs=refs,fields=fields,cap=CAP,
            purpose='LABELED_FEATURE_REVIEW_ONLY',fit_authorized=False,real_preprocessing_fitted=False),ensure_ascii=False).encode())
        for index,part in enumerate(labels['parts']):
            lab=read_stored_part(part);keep=(lab.label_state.to_numpy()==0)&~lab.proposed_purged.to_numpy()
            if not keep.any():continue
            source_name=part['source']
            if context['parts'][index]['source']!=source_name or source['joint_parts'][index]['path']!=source_name:
                raise ValueError('INPUT_PART_ORDER_CHANGED')
            body=(SOURCE/source_name).read_bytes()
            if digest(body)!=source['output_hashes'][source_name]['sha256']:raise ValueError('SOURCE_PART_DRIFT')
            file=pq.ParquetFile(pa.BufferReader(body));base_cols=[c for c in file.schema_arrow.names if c in need]
            base=file.read(columns=base_cols).to_pandas()
            if not base[['code','cutoff_index']].equals(lab[['code','cutoff_index']]):raise ValueError('LABEL_FEATURE_KEYS_DIFFER')
            positions=np.flatnonzero(keep);selected=base.iloc[positions].reset_index(drop=True)
            context_name=context['parts'][index]['context'];body=(CONTEXT/context_name).read_bytes()
            if digest(body)!=context['parts'][index]['sha256']:raise ValueError('CONTEXT_PART_DRIFT')
            table=pq.ParquetFile(pa.BufferReader(body));cols=[c for c in table.schema_arrow.names if c in need]
            ctx=table.read(columns=cols).to_pandas()
            if not ctx[['code','cutoff_index']].equals(lab[['code','cutoff_index']]):raise ValueError('CONTEXT_KEYS_DIFFER')
            selected=selected.merge(ctx.iloc[positions],on=['code','cutoff_index'],validate='one_to_one')
            missing=[c for c in need if c not in selected and c in shared]
            selected=selected.merge(shared[['cutoff_index',*missing]],on='cutoff_index',how='left',validate='many_to_one')
            if need-set(selected.columns):raise ValueError('WHITELIST_DEPENDENCY_MISSING:'+str(need-set(selected.columns)))
            selected=derive(selected);numeric(selected[fields])
            meta=attach_label_times(lab.iloc[positions].reset_index(drop=True),times,segments)
            if not selected[['code','cutoff_index']].equals(meta[['code','cutoff_index']]):raise ValueError('COHORT_ORDER_CHANGED')
            result=meta[['code','cutoff_index','decision_ts_assumed_end','label_end_time',
                'y_tail_1200m','mae_1200m','proposed_segment_name']].copy()
            for field in fields:result[field]=selected[field].to_numpy(dtype=float)
            masks=group_observation_masks(result,spec['feature_groups'])
            result['has_any_value_in_each_candidate_group']=np.logical_and.reduce(list(masks.values()))
            result['source_part_index']=index;result['source_row_index']=positions
            for name,group in result.groupby('proposed_segment_name',sort=False):buffers[name].append(group)
            bound_parts.append(dict(index=index,source_sha256=source['output_hashes'][source_name]['sha256'],
                context_sha256=context['parts'][index]['sha256'],rows=len(result)))
            print(json.dumps(dict(stage='review_cohort',source_part=index,rows=sum(p['rows'] for p in bound_parts))),flush=True)
        summary={};field_audit={};group_audit={};split_support={}
        for name,items in buffers.items():
            data=pd.concat(items,ignore_index=True).sort_values(['decision_ts_assumed_end','code']).reset_index(drop=True)
            if data.duplicated(['code','cutoff_index']).any():raise ValueError('DUPLICATE_REVIEW_SAMPLE')
            masks=group_observation_masks(data,spec['feature_groups'])
            group_audit[name]={g:dict(observed_rows=int(mask.sum()),entire_group_missing_rows=int((~mask).sum())) for g,mask in masks.items()}
            field_audit[name]={field:dict(missing=int(data[field].isna().sum()),finite=int(data[field].notna().sum()),
                distinct=int(data[field].nunique(dropna=True))) for field in fields}
            summary[name]=dict(rows=len(data),positive=int(data.y_tail_1200m.sum()),
                nonpositive=int(len(data)-data.y_tail_1200m.sum()),dates=int((data.decision_ts_assumed_end//1000000).nunique()),
                stocks=int(data.code.nunique()),rows_with_any_in_each_group=int(data.has_any_value_in_each_candidate_group.sum()))
            split_support[name]=cutoff_support(data.decision_ts_assumed_end,spec['evaluation_proposed']['ranking_minimum_cross_section'])
            table=pa.Table.from_pandas(data,preserve_index=False);sink=pa.BufferOutputStream()
            pq.write_table(table,sink,compression='zstd',compression_level=9,
                use_dictionary=['code','proposed_segment_name','y_tail_1200m'],
                use_byte_stream_split=[f.name for f in table.schema if pa.types.is_floating(f.type)])
            write(name+'.parquet',sink.getvalue().to_pybytes())
        expected=labels['proposed_split_counts']
        for name,entry in summary.items():
            if entry['rows']!=expected[name]['conditional_unpurged_rows'] or entry['positive']!=expected[name]['conditional_unpurged_positive']:
                raise ValueError('LABEL_COHORT_COUNTS_CHANGED')
        if set(summary)!={s['name'] for s in segments}:raise ValueError('MISSING_SEGMENT')
        for name,sha in refs.items():
            if digest(Path(name).read_bytes())!=sha:raise ValueError('INPUT_CHANGED_DURING_REVIEW')
        report=dict(status='JOINT_FEATURE_REVIEW_MATERIALIZED_NOT_TRAINING_READY',inputs=refs,outputs=outputs.copy(),
            fields=fields,field_count=len(fields),source_parts=bound_parts,summary=summary,
            field_audit=field_audit,group_audit=group_audit,ranking_cross_sections=split_support,
            matrix_rows=sum(s['rows'] for s in summary.values()),row_selection='LABEL_STATE_0_AND_NOT_PURGED_ONLY',
            feature_missing_rows_dropped=0,real_preprocessing_fitted=False,http_requests=0,fit_run=False,
            fit_authorized=False,training_ready=False,full_rule_inputs_ready=False,
            pending_rule_inputs=spec['not_materialized_as_confirmed_rules'],sample_gate_status=spec['sample_gate_status'])
        write('manifest.json',json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2).encode())
    except Exception as exc:status,error='FAILED',f'{type(exc).__name__}:{exc}'
    write('stop_record.json',json.dumps(dict(status=status,error=error,bytes_before_stop=used),allow_nan=False).encode(),terminal=True)
    print(json.dumps(dict(status=status,error=error,bytes_written=used)))
    return status


if __name__=='__main__':raise SystemExit(0 if run()=='COMPLETE' else 2)
