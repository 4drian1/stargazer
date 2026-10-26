"""Stream decision-after labels on frozen joint keys; no HTTP, FIT or key filtering."""
from collections import defaultdict,Counter
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from inventory_ths_stockdb_cache_v1 import HERE,ROOTS,digest
from build_joint_risk_features_v1 import QUARANTINE
from decision_after_labels_v1 import minute_clock,price_grid,aligned_days,label_rows,proposed_split,STATES

SOURCE=HERE/'joint_risk_features_20261017_v3'
CONTRACT=HERE/'contracts/INTRADAY_DECISION_AFTER_LABEL_V1.json'
LEGACY=HERE/'historical_archived_corpus_v4_20261015'
OUT=HERE/'joint_decision_after_labels_20261017_v1'
CAP,RESERVE=128*1024**2,1024**2


def admitted_evidence():
    """Consume frozen prior corroboration; never launch old fetching modules."""
    acceptance=HERE/'archived_corpus_v4_acceptance_20261015.json'
    accepted=json.loads(acceptance.read_bytes())
    for ref in accepted['controls']:
        if digest(Path(ref['path']).read_bytes())!=ref['sha256']:raise ValueError('ORDINARY_EVIDENCE_DRIFT')
    return json.loads((LEGACY/'evidence_and_gate.json').read_bytes())['audit'],acceptance


def evidence_for(code,audit,grid,days,refs):
    if code not in audit:return None,None
    path=LEGACY/f'{code}_daily.json.gz'
    if not path.exists():path=HERE/'historical_archived_corpus_v3_20261015'/path.name
    body=path.read_bytes();refs[str(path)]=digest(body)
    records=json.loads(gzip.decompress(body));bars={}
    for key,row in records:
        if row is None:continue
        if not key.startswith(f'日k:{code}:') or row.get('code')!=code:raise ValueError('DAILY_EVIDENCE_CODE_MISMATCH')
        day=int(key.rsplit(':',1)[1])
        if row.get('date')!=day:raise ValueError('DAILY_EVIDENCE_DATE_MISMATCH')
        values={k:row.get(k) for k in ('open','high','low','close')}
        if day in bars and bars[day]!=values:raise ValueError('DAILY_EVIDENCE_CONFLICT')
        bars[day]=values
    aligned=aligned_days(grid,days,bars)
    factors=np.ones(len(days));events=audit[code]['events'];issues=audit[code]['issues']
    for event in events:
        mult=event.get('multiplier')
        if mult is not None:
            if event['status']!='ORDINARY_PARAMETERS_CORROBORATED' or not np.isfinite(mult) or mult<=0:
                raise ValueError('INVALID_ADMITTED_EVENT')
            factors[days>=event['date']]*=mult
    bad=np.zeros(len(days),dtype=bool)
    for issue in issues:
        if issue.get('date') is None:bad[:]=True
        else:bad|=days==issue['date']
    return dict(aligned=aligned,factors=factors,issues=bad),dict(aligned_days=int(aligned.sum()),
        observed_days=int(np.isfinite(grid[:,3]).reshape(-1,240).any(axis=1).sum()),
        event_issues=len(issues),coverage_verified=False)


def serialize(frame):
    table=pa.Table.from_pandas(frame,preserve_index=False);sink=pa.BufferOutputStream()
    pq.write_table(table,sink,compression='zstd',compression_level=9,
        use_dictionary=['code','label_state','proposed_segment','proposed_purged',
                        'diagnostic_tail_raw_1200m','y_tail_1200m'],
        use_byte_stream_split=['diagnostic_mae_raw_1200m','mae_1200m'])
    return sink.getvalue().to_pybytes()


def run():
    if OUT.exists():raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    from research_budget_v2 import bytes_used,TOTAL_CAP
    if bytes_used()+RESERVE>TOTAL_CAP:raise ValueError('REGISTERED_BUDGET_STOP')
    spec=json.loads(CONTRACT.read_bytes())
    if spec['user_answer']!='采用未来1200交易分钟、8%阈值（推荐）' or spec['threshold']!=.08:
        raise ValueError('LABEL_DEFINITION_NOT_APPROVED')
    if json.loads((SOURCE/'stop_record.json').read_bytes())['status']!='COMPLETE':raise ValueError('JOINT_NOT_COMPLETE')
    manifest=json.loads((SOURCE/'manifest.json').read_bytes())
    frozen=[CONTRACT,SOURCE/'manifest.json',SOURCE/'stop_record.json',QUARANTINE,Path(__file__),
            HERE/'decision_after_labels_v1.py',HERE/'test_decision_after_labels_v1.py']
    refs={str(p):digest(p.read_bytes()) for p in frozen}
    audit,accepted_path=admitted_evidence()
    for path in [accepted_path,LEGACY/'evidence_and_gate.json',LEGACY/'plan.json']:
        refs[str(path)]=digest(path.read_bytes())
    shared_body=(SOURCE/'shared_market_and_leaders.parquet').read_bytes()
    if digest(shared_body)!=manifest['output_hashes']['shared_market_and_leaders.parquet']['sha256']:
        raise ValueError('SHARED_CONTEXT_DRIFT')
    times=pq.read_table(pa.BufferReader(shared_body),columns=['decision_ts_assumed_end']).column(0).to_numpy()
    days=np.unique(times//1000000);clock=minute_clock(days)
    if not np.array_equal(times,clock[14::15]):raise ValueError('CUTOFF_CLOCK_MISMATCH')
    quarantine=defaultdict(set)
    with QUARANTINE.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):quarantine[row['code']].add(int(row['day']))
    OUT.mkdir(exist_ok=False);used=0;parts=[];sources=[];stats=defaultdict(Counter)
    date_sets=defaultdict(set);codes_seen=set();states=Counter();evidence_stats={}
    def write(name,body,terminal=False):
        nonlocal used
        reserve=0 if terminal else RESERVE
        if used+len(body)+reserve>CAP or bytes_used()+len(body)+reserve>TOTAL_CAP:
            raise ValueError('ACTUAL_LABEL_BUDGET_STOP')
        with (OUT/name).open('xb') as stream:stream.write(body)
        used+=len(body)
    status,error='COMPLETE',None
    try:
        write('plan.json',json.dumps(dict(inputs=refs,cap=CAP,opening_registered_bytes=bytes_used(),
            fit_run=False,http_requests=0,formal_label_scope='CONDITIONAL_ORDINARY_MINUTE_DAILY_OHLC_ALIGNED',
            folds_status=spec['folds_status']),ensure_ascii=False,allow_nan=False,indent=2).encode())
        for i,part in enumerate(manifest['joint_parts']):
            body=(SOURCE/part['path']).read_bytes()
            if digest(body)!=manifest['output_hashes'][part['path']]['sha256']:raise ValueError('JOINT_PART_DRIFT')
            rows=pq.read_table(pa.BufferReader(body),columns=['code','cutoff_index','stock_close','source_cache']).to_pandas()
            buffer=[]
            for code,frame in rows.groupby('code',sort=False):
                if code in codes_seen:raise ValueError('STOCK_SPANS_SHARDS')
                codes_seen.add(code);frame=frame.reset_index(drop=True)
                labels=set(frame.source_cache)
                if len(labels)!=1:raise ValueError('MIXED_SOURCE')
                source=ROOTS[labels.pop()]/f'{code}.parquet';raw=source.read_bytes()
                sources.append(dict(code=code,path=str(source),sha256=digest(raw),bytes=len(raw)))
                minute=pq.read_table(pa.BufferReader(raw)).to_pandas()
                grid=price_grid(minute,code,clock,quarantine.get(code,set()))
                evidence,quality=evidence_for(code,audit,grid,days,refs)
                if quality is not None:evidence_stats[code]=quality
                result,end=label_rows(frame,grid,days,evidence)
                end_times=np.full(len(end),np.nan);inside=end<len(clock)
                end_times[inside]=clock[end[inside]]
                positions=frame.cutoff_index.to_numpy(dtype=int)
                segments,purged=proposed_split(times[positions],end_times,spec['proposed_segments'])
                result['proposed_segment']=segments;result['proposed_purged']=purged
                for name,count in result.label_state.value_counts().items():states[STATES[int(name)]]+=int(count)
                for seg in np.unique(segments):
                    key=spec['proposed_segments'][seg]['name'] if seg>=0 else 'OUT_OF_SCOPE'
                    selected=segments==seg;target=stats[key];target['population_rows']+=int(selected.sum())
                    target['proposed_purged_rows']+=int((selected&purged).sum())
                    diagnostic=selected&(result.diagnostic_tail_raw_1200m.to_numpy()>=0)
                    admitted=selected&(result.label_state.to_numpy()==0)
                    unpurged=admitted&~purged
                    target['diagnostic_complete_rows']+=int(diagnostic.sum())
                    target['diagnostic_raw_positives_not_formal']+=int((diagnostic&(result.diagnostic_tail_raw_1200m.to_numpy()==1)).sum())
                    target['conditional_label_rows']+=int(admitted.sum())
                    target['conditional_unpurged_rows']+=int(unpurged.sum())
                    target['conditional_unpurged_positive']+=int(result.loc[unpurged,'y_tail_1200m'].sum())
                    date_sets[key+'|population'].update((times[positions[selected]]//1000000).tolist())
                    date_sets[key+'|conditional_unpurged'].update((times[positions[unpurged]]//1000000).tolist())
                buffer.append(result)
            table=pd.concat(buffer,ignore_index=True)
            if not table[['code','cutoff_index']].equals(rows[['code','cutoff_index']].reset_index(drop=True)):
                raise ValueError('LABEL_ROW_IDENTITY_CHANGED')
            encoded=serialize(table);name=f'{i:04d}.parquet';write(name,encoded)
            parts.append(dict(path=name,source=part['path'],rows=len(table),bytes=len(encoded),sha256=digest(encoded)))
            print(json.dumps(dict(stage='decision_after_labels',parts=len(parts),total=len(manifest['joint_parts']),
                stocks=len(codes_seen),rows=sum(p['rows'] for p in parts),bytes_written=used)),flush=True)
        for name,value in stats.items():
            value['population_dates']=len(date_sets[name+'|population'])
            value['conditional_unpurged_dates']=len(date_sets[name+'|conditional_unpurged'])
            value['conditional_unpurged_nonpositive']=value['conditional_unpurged_rows']-value['conditional_unpurged_positive']
        if any(digest(Path(p).read_bytes())!=sha for p,sha in refs.items()):raise ValueError('INPUT_CHANGED_DURING_BUILD')
        report=dict(status='DECISION_AFTER_LABEL_PREPARATION_NOT_FIT_READY',parts=parts,inputs=refs,
            source_files=sources,rows=sum(p['rows'] for p in parts),stock_codes=len(codes_seen),states=dict(states),
            proposed_split_counts={k:dict(v) for k,v in stats.items()},minute_daily_alignment=evidence_stats,
            label_states=STATES,diagnostic_null_sentinel=-1,diagnostic_storage='FLOAT32; CLASSIFICATION_COMPUTED_FLOAT64',
            data_complete_time=None,clock='ASSUMED_END',coverage_verified=False,
            label_scope='CONDITIONAL_ORDINARY_MINUTE_DAILY_OHLC_ALIGNED',
            fit_authorized=False,training_ready=False,training_eligible=False,fit_run=False,http_requests=0)
        write('manifest.json',json.dumps(report,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode())
    except Exception as exc:status,error='FAILED',f'{type(exc).__name__}:{exc}'
    write('stop_record.json',json.dumps(dict(status=status,error=error,parts=len(parts),bytes_before_stop=used),
        ensure_ascii=False,allow_nan=False).encode(),terminal=True)
    print(json.dumps(dict(status=status,error=error,parts=len(parts),rows=sum(p['rows'] for p in parts),bytes_written=used)))
    return status


if __name__=='__main__':raise SystemExit(0 if run()=='COMPLETE' else 2)
