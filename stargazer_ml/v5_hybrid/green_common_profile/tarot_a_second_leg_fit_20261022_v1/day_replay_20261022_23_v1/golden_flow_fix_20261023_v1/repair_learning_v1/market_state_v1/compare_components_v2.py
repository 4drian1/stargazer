"""One frozen factorized-reference comparison on the already selected index-days."""
import gzip
import io
import json
import numpy as np
import pandas as pd
from run_acceptance import HERE, STUDY, RISK, TIME, put, sha, peak
from state_rules import describe, combine
from state_components_v2 import describe_parts
from audit_state_switches import node_at


def replay(index, features, bars, receipt):
    assert len(features) == len(bars)
    np.testing.assert_array_equal(features[TIME], bars.ts)
    states=[]; previous_node=None
    for i, (_, row) in enumerate(features.iterrows()):
        cutoff=int(row[TIME]); f=row.to_dict(); old=describe(index,cutoff,f)
        prefix=bars.iloc[:i+1]
        node,path=node_at(prefix,receipt['bands'],receipt['atr_prior'],old['observed'])
        volumes=prefix.raw_volume.to_numpy()
        current=describe_parts(index,cutoff,f,float(prefix.close.iloc[-1]),
            bool(np.isfinite(volumes).all() and (volumes>0).all()),node,path,previous_node)
        previous_node=current['position']['tracked_node']
        states.append((old,current))
    return states


def count_switches(values):
    return sum(a!=b for a,b in zip(values[:-1],values[1:]))


def main():
    v2=STUDY/'repair_effectiveness_v2'
    audit=json.loads((HERE/'switch_audit_report.json').read_bytes())
    selected=[(r['index'],r['day'],r['selection']) for r in audit['index_days']]
    example_times=[94500,100000,103000,110000,113000,150000]
    snapshot_times={94500,103000,113000,140000,150000}
    paths=[HERE/'compare_components_v2.py',HERE/'state_components_v2.py',HERE/'test_state_components_v2.py',
        HERE/'state_rules.py',HERE/'audit_state_switches.py',HERE/'switch_audit_report.json',
        HERE/'switch_audit_transitions.csv',HERE/'switch_audit_observations.parquet',
        v2/'full_market_features.parquet']
    for index in ['sse','cyb']:
        paths += [RISK/f'index_minutes_v1/{index}_5min.parquet',
            STUDY/f'market_break_repair_v1/{index}_source_receipts.json.gz',
            v2/f'phase_missing_{index}_maps.json.gz',HERE/f'switch_missing_{index}_receipts.json.gz']
    inputs={str(p):sha(p) for p in paths}
    put('components_v2_resume_plan.json',dict(inputs=inputs,index_days=selected,
        previous_failure='components_v2_failure_01.json: pandas Series.asof method collision; no result files written',
        example_20261023='Known example only; not added to comparison or used for tuning',
        rules='state_components_v2.py; current position+RSI+known contraction strong; last-bar low retained as observation',
        reference_only=True, new_structure_low_definition=None, carry_previous_strength=False,
        acceptance='Preserve all nine previously documented old-node breaks; no strong without current path/RSI/known volume; exact prefix; never accuracy by switch count',
        fits=0,paid_calls=0,independent_human_labels=0))
    panel=pd.read_parquet(v2/'full_market_features.parquet')
    rows=[]; snapshots=[]; examples={}; prefix_checks=[]; lookup={}; daily=[]
    for index in ['sse','cyb']:
        native=pd.read_parquet(RISK/f'index_minutes_v1/{index}_5min.parquet').sort_values('ts')
        receipts=[]
        for path in [STUDY/f'market_break_repair_v1/{index}_source_receipts.json.gz',
                     v2/f'phase_missing_{index}_maps.json.gz',HERE/f'switch_missing_{index}_receipts.json.gz']:
            receipts += json.loads(gzip.decompress(path.read_bytes()))
        maps={int(r['day']):r for r in receipts}; assert len(maps)==len(receipts)
        for day,selection in [(d,s) for n,d,s in selected if n==index]+[(20261023,'known_example')]:
            f=panel[(panel['index']==index)&(panel[TIME]//1000000==day)].sort_values(TIME).reset_index(drop=True)
            bars=native[native.day==day].reset_index(drop=True)
            result=replay(index,f,bars,maps[day]); assert len(result)==48
            prefix=replay(index,f.iloc[:24],bars.iloc[:24],maps[day])
            assert prefix==result[:24]
            prefix_checks.append(dict(index=index,day=day,rows=24))
            if selection!='known_example':
                a=[x['state'] for x,_ in result]; b=[y['state'] for _,y in result]
                daily.append(dict(index=index,day=day,selection=selection,old_switches=count_switches(a),new_switches=count_switches(b),
                    old_roundtrips=sum(a[i-1]==a[i+1]!=a[i] for i in range(1,47)),
                    new_roundtrips=sum(b[i-1]==b[i+1]!=b[i] for i in range(1,47))))
            for i,(old,new) in enumerate(result):
                cutoff=new['asof']; p=new['position']; m=new['momentum']; q=new['maturity']
                if old['state']=='STRONG_REPAIR': assert new['state']=='STRONG_REPAIR'
                if new['state']=='STRONG_REPAIR':
                    assert p['current_path']!='NONE' and m['rsi_state']=='FAST_ABOVE' and q['volume_state']=='CONTRACTING'
                if new['state']=='WEAK_REPAIR':assert p['current_path']!='NONE'
                if q['volume_state']=='PENDING_AFTER_LOW':assert q['bars_since_low'] in [0,1]
                if new['state']=='LOCAL_REBOUND': assert p['current_path']=='NONE'
                if old['state'] in ['BREAK_UNREPAIRED','DECLINE_ENHANCING']: assert new['state']==old['state']
                if selection=='known_example':
                    if cutoff%1000000 in example_times:examples.setdefault(cutoff,{})[index]=new
                    continue
                lookup[(index,cutoff)]=new
                rows.append(dict(index=index,day=day,asof=cutoff,selection=selection,old=old['state'],new=new['state'],
                    old_label=old['label'],new_label=new['label'],**p,**m,**q))
                if selection=='fixed_calendar' and cutoff%1000000 in snapshot_times:
                    snapshots.append(dict(old_state=old['state'],**new))
    frame=pd.DataFrame(rows);summary={}
    for selection in ['fixed_calendar','worst_switching']:
        g=frame[frame.selection==selection]; d=[r for r in daily if r['selection']==selection]
        summary[selection]=dict(observations=len(g),index_days=len(d),
            old_counts=g.old.value_counts().to_dict(),new_counts=g.new.value_counts().to_dict(),
            old_switches=sum(r['old_switches'] for r in d),new_switches=sum(r['new_switches'] for r in d),
            old_roundtrips=sum(r['old_roundtrips'] for r in d),new_roundtrips=sum(r['new_roundtrips'] for r in d),
            pending_volume=int((g.volume_state=='PENDING_AFTER_LOW').sum()),
            old_unknown_reclassified=g[g.old=='UNKNOWN'].new.value_counts().to_dict(),
            local_rebound_count=int((g.new=='LOCAL_REBOUND').sum()),
            current_node_breaks=int(g.previous_node_broken.sum()),
            lower_repair_with_previous_break=int(g.lower_repair_with_previous_break.sum()))
    old_changes=pd.read_csv(HERE/'switch_audit_transitions.csv')
    protected=old_changes[(old_changes.kind=='REPAIR_DOWN_CROSS') & old_changes.previous.isin(['STRONG_REPAIR','WEAK_REPAIR'])
        & old_changes.current_price_below_previous_node]
    assert len(protected)==9
    preserved=[]
    for _,r in protected.iterrows():
        new=lookup[(r['index'],int(r['asof']))]; assert new['position']['previous_node_broken']
        preserved.append(dict(index=r['index'],asof=int(r['asof']),old=r.current,new=new['state'],
            old_node=float(r.previous_node),tracked_previous=new['position']['previous_confirmed_node']))
    original=pd.read_parquet(HERE/'switch_audit_observations.parquet')
    joined=frame.merge(original[['index','asof','state']],on=['index','asof'],validate='one_to_one')
    assert len(joined)==1440 and joined.old.eq(joined.state).all()
    for cutoff,pair in examples.items():
        assert set(pair)=={'sse','cyb'}
        examples[cutoff]=dict(indices=pair,market=combine(pair['sse'],pair['cyb']))
    manifest=json.loads((RISK/'live_monitor_fingerprints_20261023_v4_flow.json').read_bytes())
    for mapping in [inputs,manifest]:
        for p,h in mapping.items():assert sha(p)==h,p
    buffer=io.BytesIO();frame.to_parquet(buffer,index=False,compression='zstd')
    put('components_v2_states.parquet',buffer.getvalue())
    put('components_v2_changes.csv',frame[frame.old!=frame.new].to_csv(index=False).encode('utf-8-sig'))
    put('components_v2_snapshots.json',snapshots)
    put('components_v2_known_example.json',examples)
    put('components_v2_report.json',dict(groups=summary,daily=daily,old_node_breaks_preserved=preserved,
        prefix_checks=prefix_checks,observations=1440,human_reviews_completed=0,independent_accuracy=False,
        peak_ram=peak(),production_fingerprints_verified=len(manifest),fits=0,paid_calls=0,production_changed=False))
    print(json.dumps(dict(groups=summary,protected_node_breaks=len(preserved),snapshots=len(snapshots),
        example={str(t):{k:v['state'] for k,v in e['indices'].items()} for t,e in examples.items()}),ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        put('components_v2_failure_02.json',dict(error=repr(exc),status='STOP_NO_AUTORETRY'))
        raise
