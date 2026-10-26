"""Independent scalar future-window replay plus full label/key read-back."""
from collections import Counter,defaultdict
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from build_decision_after_labels_v1 import SOURCE,CONTRACT,LEGACY
from read_decision_after_labels_v2 import OUT,read_stored_part
from inventory_ths_stockdb_cache_v1 import digest
from build_joint_risk_features_v1 import QUARANTINE
from decision_after_labels_v1 import minute_clock,price_grid,STATES


def run():
    if json.loads((OUT/'stop_record.json').read_bytes())['status']!='COMPLETE':
        raise ValueError('LABEL_BUILD_NOT_COMPLETE')
    metadata=json.loads((OUT/'manifest.json').read_bytes());spec=json.loads(CONTRACT.read_bytes())
    ledger={}
    plan=json.loads((LEGACY/'plan.json').read_bytes())
    approved_inputs={r['path']:r['sha256'] for r in plan['inputs']}
    old_stop=LEGACY.with_name('historical_archived_corpus_v3_20261015')/'stop_record.json'
    if digest(old_stop.read_bytes())!=approved_inputs[str(old_stop)]:raise ValueError('PRIOR_LEDGER_DRIFT')
    for path in [old_stop,LEGACY/'stop_record.json']:
        ledger.update({(r['code'],r['table']):r for r in json.loads(path.read_bytes())['request_ledger']})
    raw_daily_checked=0
    for name,sha in metadata['inputs'].items():
        if name.endswith('_daily.json.gz'):
            path=Path(name);body=path.read_bytes()
            if digest(body)!=sha:raise ValueError('DAILY_ARCHIVE_DRIFT')
            payload=gzip.decompress(body);ref=ledger[path.name[:6],'日k']
            if len(payload)!=ref['bytes'] or digest(payload)!=ref['sha256']:
                raise ValueError('DAILY_PAYLOAD_NOT_ORIGINAL_APPROVED_EVIDENCE')
            raw_daily_checked+=1
    shared=pq.read_table(SOURCE/'shared_market_and_leaders.parquet',columns=['decision_ts_assumed_end']).column(0).to_numpy()
    days=np.unique(shared//1000000);clock=minute_clock(days)
    audit=json.loads((LEGACY/'evidence_and_gate.json').read_bytes())['audit']
    witnesses={};event_witness=None;states=Counter();rows=0;positive=0;admitted=0
    classes=defaultdict(Counter);dates=defaultdict(set);stocks=defaultdict(set)
    for part in metadata['parts']:
        body=(OUT/part['path']).read_bytes()
        if digest(body)!=part['sha256']:raise ValueError('LABEL_PART_HASH_MISMATCH')
        frame=read_stored_part(part)
        keys=pq.read_table(SOURCE/part['source'],columns=['code','cutoff_index']).to_pandas()
        if not frame[['code','cutoff_index']].equals(keys):raise ValueError('LABEL_KEY_MISMATCH')
        complete=frame.label_state==0;ys=frame.y_tail_1200m
        if ys[~complete].notna().any() or frame.mae_1200m[~complete].notna().any():
            raise ValueError('UNADMITTED_FORMAL_LABEL')
        if not np.isfinite(frame.mae_1200m[complete]).all() or not ys[complete].isin([0,1]).all():
            raise ValueError('INVALID_CONDITIONAL_LABEL')
        if not np.array_equal(ys[complete].to_numpy(dtype=int),(frame.mae_1200m[complete]<=-.08+1e-12).to_numpy(dtype=int)):
            raise ValueError('FORMAL_LABEL_THRESHOLD_MISMATCH')
        raw=frame.diagnostic_tail_raw_1200m.to_numpy()
        if not np.isin(raw,[-1,0,1]).all() or not np.array_equal(raw>=0,np.isfinite(frame.diagnostic_mae_raw_1200m)):
            raise ValueError('RAW_DIAGNOSTIC_NULL_MISMATCH')
        for state,group in frame.groupby('label_state',sort=True):
            states[STATES[int(state)]]+=len(group)
            witnesses.setdefault(int(state),group.iloc[0].to_dict())
        ts=shared[frame.cutoff_index.to_numpy(dtype=int)]
        ends=np.full(len(frame),np.nan);indices=frame.cutoff_index.to_numpy(dtype=int)*15+14+1200
        inside=indices<len(clock);ends[inside]=clock[indices[inside]]
        if event_witness is None:
            for code,group in frame[complete].groupby('code',sort=True):
                start_dates=shared[group.cutoff_index.to_numpy(dtype=int)]//1000000
                end_dates=clock[group.cutoff_index.to_numpy(dtype=int)*15+14+1200]//1000000
                for event in audit[code]['events']:
                    if event.get('multiplier') is None or event['multiplier']==1:continue
                    selected=(start_dates<event['date'])&(end_dates>=event['date'])
                    if selected.any():
                        event_witness=group.iloc[np.flatnonzero(selected)[0]].to_dict()
                        break
                if event_witness is not None:break
        for i,seg in enumerate(spec['proposed_segments']):
            selected=frame.proposed_segment.to_numpy()==i
            if not (((ts[selected]//1000000)>=seg['start'])&((ts[selected]//1000000)<=seg['end'])).all():
                raise ValueError('WRONG_DECISION_SEGMENT')
            if i+1<len(spec['proposed_segments']):
                crossing=(~np.isfinite(ends))|(ends>=spec['proposed_segments'][i+1]['start']*1000000)
                if not frame.loc[selected&crossing,'proposed_purged'].all():raise ValueError('CROSS_BOUNDARY_NOT_PURGED')
            keep=selected&complete.to_numpy()&~frame.proposed_purged.to_numpy()
            name=seg['name'];pos=keep&frame.y_tail_1200m.eq(1).fillna(False).to_numpy(dtype=bool)
            classes[name]['rows']+=int(keep.sum());classes[name]['positive']+=int(pos.sum())
            dates[name].update((ts[keep]//1000000).tolist());dates[name+'|positive'].update((ts[pos]//1000000).tolist())
            stocks[name].update(frame.loc[keep,'code'].tolist())
        rows+=len(frame);admitted+=int(complete.sum());positive+=int(ys[complete].sum())
    if rows!=metadata['rows'] or dict(states)!=metadata['states']:raise ValueError('MANIFEST_COUNTS_MISMATCH')
    for name,counts in classes.items():
        counts['nonpositive']=counts['rows']-counts['positive'];counts['distinct_dates']=len(dates[name])
        counts['positive_dates']=len(dates[name+'|positive']);counts['stocks']=len(stocks[name])
        expected=metadata['proposed_split_counts'][name]
        if (counts['rows']!=expected['conditional_unpurged_rows'] or counts['positive']!=expected['conditional_unpurged_positive']
                or counts['distinct_dates']!=expected['conditional_unpurged_dates']):raise ValueError('SEGMENT_COUNTS_MISMATCH')
    source_by_code={x['code']:x for x in metadata['source_files']}
    quarantine=defaultdict(set)
    with QUARANTINE.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):quarantine[row['code']].add(int(row['day']))
    cases=[]
    replay=list(sorted(witnesses.items()))+([(0,event_witness)] if event_witness is not None else [])
    for state,row in replay:
        source=source_by_code[row['code']]
        body=Path(source['path']).read_bytes()
        if digest(body)!=source['sha256']:raise ValueError('LABEL_SOURCE_CHANGED')
        grid=price_grid(pq.read_table(pa.BufferReader(body)).to_pandas(),row['code'],clock,quarantine.get(row['code'],set()))
        t=int(row['cutoff_index'])*15+14;end=t+1200
        low=grid[t+1:end+1,2]
        complete=len(low)==1200 and np.isfinite(low).all()
        if complete:
            raw_mae=float(min(low)/grid[t,3]-1)
            if abs(raw_mae-row['diagnostic_mae_raw_1200m'])>1e-7 or row['diagnostic_tail_raw_1200m']!=int(raw_mae<=-.08+1e-12):
                raise ValueError('SCALAR_RAW_WINDOW_REPLAY_MISMATCH')
        elif row['diagnostic_tail_raw_1200m']!=-1:raise ValueError('INCOMPLETE_DIAGNOSED_AS_COMPLETE')
        if state==0:
            date=int(clock[t]//1000000);events=audit[row['code']]['events']
            normalized=[]
            for position,price in enumerate(low,t+1):
                d=int(clock[position]//1000000)
                multiplier=math.prod(e['multiplier'] for e in events if e.get('multiplier') is not None and date<e['date']<=d)
                normalized.append(price*multiplier)
            mae=min(normalized)/grid[t,3]-1
            if abs(mae-row['mae_1200m'])>1e-12 or row['y_tail_1200m']!=int(mae<=-.08+1e-12):
                raise ValueError('SCALAR_NORMALIZED_LABEL_REPLAY_MISMATCH')
        cases.append(dict(state=STATES[state],code=row['code'],decision_ts=int(clock[t]),
                          expected_end=int(clock[end]) if end<len(clock) else None,scalar_replay='PASS',
                          witness_kind='ORDINARY_EVENT_WINDOW' if row is event_witness else 'FIRST_OF_STATE'))
    report=dict(status='LABEL_READBACK_AND_SCALAR_REPLAY_PASS_NOT_FIT_READY',rows=rows,
        conditional_label_rows=admitted,conditional_positive=positive,conditional_nonpositive=admitted-positive,
        proposed_unpurged_conditional_classes={k:dict(v) for k,v in classes.items()},
        states=dict(states),scalar_cases=cases,manifest_sha256=digest((OUT/'manifest.json').read_bytes()),
        approved_daily_payloads_verified=raw_daily_checked,
        ordinary_event_window_replay='PASS' if event_witness is not None else 'NOT_TESTED_NO_ADMITTED_EVENT_WINDOW',
        audit_code_sha256=digest(Path(__file__).read_bytes()),
        logical_reader_sha256=digest((OUT.parent/'read_decision_after_labels_v2.py').read_bytes()),
        logical_reader_test_sha256=digest((OUT.parent/'test_decision_after_basis_v2.py').read_bytes()),
        folds_status=spec['folds_status'],coverage_verified=False,training_ready=False,fit_run=False,http_requests=0)
    body=json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2).encode()
    from research_budget_v2 import bytes_used,TOTAL_CAP
    if len(body)>1024**2 or bytes_used()+len(body)>TOTAL_CAP:raise ValueError('AUDIT_BUDGET_STOP')
    with (OUT/'readback_acceptance.json').open('xb') as stream:stream.write(body)
    print(body.decode())


if __name__=='__main__':run()
