"""Two named engineering stocks; user-approved conditional research, never full-coverage certification."""
import json
from pathlib import Path
from bisect import bisect_left
from collections import Counter

from historical_stream_v1 import read_unique,write_stream
from historical_risk_core_v1 import label_at,stock_features,market_features,segment,sample_gate
from stockdb_adjustment_v2 import rebuild_mult_factors
from ordinary_action_reconciliation_v3 import reconcile_event
from market_index_source_v1 import bind_from_experiment_config
from price_verification_guarded_run import sha,encoded
from research_budget_v2 import bytes_used,TOTAL_CAP

HERE=Path(__file__).resolve().parent
PREP=HERE/'historical_stock_day_preparation_v1'
EXTERNAL=HERE/'independent_fetch/batch_b_dividend_probe/standard_event_table_v1.json'
SNAP=PREP/'adjustment_snapshot_v2_20261015'
OUT=HERE/'historical_stock_day_restricted_v3_20261015'
CODES=('000001','002594')
SCOPE='CONDITIONAL_RESEARCH_DOUBLE_SOURCE_ORDINARY_ACTIONS'


def reconcile_code(code,bars,external,native):
    local={};other={};issues=[];events=[];audits=[]
    for key,value in native:
        d=int(key.rsplit(':',1)[1])
        if not key.startswith(f'复权:{code}:'):raise ValueError('SOURCE_CODE_MISMATCH')
        if not 20180102<=d<=20261013:continue
        if value['mult']!=1 or any(value.get(k,0) for k in ('div','give','trans')):
            if d in local and local[d]!=value:raise ValueError('NATIVE_EVENT_CONFLICT')
            local[d]=value
    for row in external:
        if row['code']!=code or row.get('reconcile_status')!='eligible':continue
        ds=row.get('dates',{}).get('除权除息日') or row.get('dates',{}).get('除权日')
        if not ds:
            issues.append(dict(date=None,reason='UNLOCATED_EXTERNAL_EVENT'));continue
        d=int(ds.replace('-',''))
        if not 20180102<=d<=20261013:continue
        if d in other and other[d]!=row:raise ValueError('EXTERNAL_EVENT_CONFLICT')
        other[d]=row
    dates=sorted(bars)
    for d in sorted(set(local)|set(other)):
        i=bisect_left(dates,d)
        if d not in local or d not in other:
            audit=dict(status='SINGLE_SOURCE_EVENT',multiplier=None)
        elif i==0:
            audit=dict(status='PREVIOUS_CLOSE_MISSING',multiplier=None)
        else:
            audit=reconcile_event(local[d],other[d],bars[dates[i-1]]['close'])
        audit=dict(audit,date=d)
        audits.append(audit)
        if audit['multiplier'] is None:issues.append(dict(date=d,reason=audit['status']))
        else:events.append(dict(date=d,mult=audit['multiplier']))
    return events,audits,issues


def run():
    if OUT.exists():raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    if bytes_used()+2*1024**2>TOTAL_CAP:raise ValueError('REGISTERED_BUDGET')
    cal_path=PREP/'manifest.json'
    calendar=json.loads(cal_path.read_bytes())['calendar_dates']
    closes=bind_from_experiment_config()
    external=json.loads(EXTERNAL.read_bytes())
    inputs=[cal_path,EXTERNAL,HERE/'contracts/RESTRICTED_HISTORICAL_RESEARCH_V3.md',
            Path(__file__),HERE/'ordinary_action_reconciliation_v3.py',
            HERE/'historical_risk_core_v1.py',HERE/'historical_stream_v1.py',
            HERE/'contracts/historical_input_overlay_20261015.json']
    cases=[];event_results={}
    for code in CODES:
        price=PREP/'real_binding_v1'/f'prices_{code}.parquet'
        adj=SNAP/f'{code}.json.raw'
        inputs.extend([price,adj])
        bars=read_unique(price,('open','high','low','close'),code)
        native=json.loads(adj.read_bytes())
        events,audits,issues=reconcile_code(code,bars,external,native)
        factor=rebuild_mult_factors(events,calendar)
        event_results[code]=dict(audits=audits,issues=issues)
        cases.append((code,bars,factor,issues))
    refs=[dict(path=str(p),sha256=sha(p)) for p in inputs]
    evidence_hash=__import__('hashlib').sha256(encoded(dict(inputs=refs,events=event_results))).hexdigest()
    rows=[];states=Counter()
    for code,bars,factors,issues in cases:
        for i,p in enumerate(calendar):
            if not 20190101<=p<=20260930:continue
            end=calendar[i+5] if i+5<len(calendar) else None
            start=calendar[max(0,i-60)]
            blocked=(end is None or end>20261013 or any(x['date'] is None or
                     start<=x['date']<=end for x in issues))
            # True here means the explicitly approved conditional-research assumption,
            # not proof of corporate-action completeness; persisted coverage_verified stays false.
            label=label_at(bars,calendar,p,factors,scale_verified=not blocked,
                           coverage_verified=not blocked)
            features,feature_status=stock_features(bars,calendar,p,factors)
            market,market_status=market_features(closes,calendar,p)
            name,within=segment(p,label['label_end_date'])
            values=dict(features,**market)
            ready=bool(not blocked and within and label['status']=='LABEL_COMPLETE'
                       and all(v is not None for v in values.values()))
            if not ready:states['not_eligible']+=1
            if blocked:states['source_interval_or_event_blocked']+=1
            rows.append(dict(code=code,P=p,segment=name,label_end_date=label['label_end_date'],
                             mae_5d=label['mae_5d'],y_tail_5d=label['y_tail_5d'],features=values,
                             label_status=label['status'],feature_status=feature_status,market_status=market_status,
                             training_eligible=ready,source_evidence_sha256=evidence_hash,
                             admission_scope=SCOPE,coverage_verified=False))
    if any(sha(x['path'])!=x['sha256'] for x in refs):raise ValueError('SOURCE_CHANGED')
    gate=sample_gate(rows)
    gate.update(scope=SCOPE,fit_authorized=False,
                note='Named engineering stock subset; coverage is assumed in approved supplier scope, never certified.')
    year={}
    for row in rows:
        if row['training_eligible']:
            y=str(row['P']//10000);r=year.setdefault(y,Counter())
            r['events']+=1;r['positive']+=row['y_tail_5d'];r['nonpositive']+=1-row['y_tail_5d']
    final=dict(scope=SCOPE,user_choice='采用受限历史研究（推荐）',inputs=refs,
               events=event_results,year_class_counts={k:dict(v) for k,v in year.items()},
               row_count=len(rows),isolation_counts=dict(states),sample_gate=gate,
               coverage_verified=False,fit_performed=False)
    report_data=encoded(final)
    # All metadata is reserved before writer output; 2MiB total task cap, no per-file reset.
    reserve=len(report_data)+len(encoded(gate))+4096
    written=write_stream(iter(rows),OUT,bytes_used,total_cap=TOTAL_CAP,
                         new_cap=2*1024**2-reserve,chunk_rows=4096)
    for name,data in [('research_evidence.json',report_data),('sample_gate.json',encoded(gate))]:
        if bytes_used()+len(data)>TOTAL_CAP:raise ValueError('REGISTERED_BUDGET')
        with (OUT/name).open('xb') as f:f.write(data)
    print(json.dumps(dict(output=str(OUT),rows=len(rows),written_part_bytes=written['part_bytes'],
                          gate=gate,year_class_counts={k:dict(v) for k,v in year.items()},
                          event_count=sum(len(v['audits']) for v in event_results.values()),
                          event_issues=sum(len(v['issues']) for v in event_results.values())),ensure_ascii=False))


if __name__=='__main__':run()
