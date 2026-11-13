"""Fixed band-upper correction; no lower-node fallback or other rule tuning."""
import gzip
import io
import json
import pandas as pd
from run_acceptance import HERE,STUDY,RISK,TIME,put,sha,peak
from compare_components_v2 import replay
from band_reclaim_v3 import qualify_band
from state_rules import combine


def qualify_series(result,bars,bands):
    output=[]
    for i,(_,parts) in enumerate(result):
        node=parts['position']['current_node']
        band=None if node is None else next(b for b in bands if b['node']==node)
        previous=None if i==0 else float(bars['close'].iloc[i-1])
        output.append(qualify_band(parts,band,previous,float(bars['close'].iloc[i])))
    return output


def main():
    v2=STUDY/'repair_effectiveness_v2'
    paths=[HERE/'compare_band_reclaim_v3.py',HERE/'band_reclaim_v3.py',HERE/'test_band_reclaim_v3.py',
        HERE/'compare_components_v2.py',HERE/'state_components_v2.py',HERE/'state_rules.py',
        HERE/'audit_state_switches.py',HERE/'components_v2_report.json',HERE/'components_v2_states.parquet',
        HERE/'components_v2_llm_facts.json.gz',v2/'full_market_features.parquet']
    for index in ['sse','cyb']:
        paths += [RISK/f'index_minutes_v1/{index}_5min.parquet',
            STUDY/f'market_break_repair_v1/{index}_source_receipts.json.gz',
            v2/f'phase_missing_{index}_maps.json.gz',HERE/f'switch_missing_{index}_receipts.json.gz']
    inputs={str(p):sha(p) for p in paths}
    put('band_v3_plan.json',dict(inputs=inputs,user_confirmed='Two completed 5m closes at/above selected band upper for reclaim branch',
        same_samples=True,selected_node_unchanged=True,lower_node_fallback=False,stop_above_unchanged=True,
        future_20261023_decline_not_label=True,fits=0,paid_calls=0,production_change=False))
    oldreport=json.loads((HERE/'components_v2_report.json').read_bytes())
    oldframe=pd.read_parquet(HERE/'components_v2_states.parquet').set_index(['index','asof'])
    assert oldframe.index.is_unique
    panel=pd.read_parquet(v2/'full_market_features.parquet')
    rows=[];checks=[];examples={};decisions={}
    original_facts=json.loads(gzip.decompress((HERE/'components_v2_llm_facts.json.gz').read_bytes()))
    snapshots={(s['index'],s['asof']):s for s in original_facts['review_index_snapshots']}
    facts=[]
    for index in ['sse','cyb']:
        native=pd.read_parquet(RISK/f'index_minutes_v1/{index}_5min.parquet').sort_values('ts')
        receipts=[]
        for path in [STUDY/f'market_break_repair_v1/{index}_source_receipts.json.gz',
                     v2/f'phase_missing_{index}_maps.json.gz',HERE/f'switch_missing_{index}_receipts.json.gz']:
            receipts+=json.loads(gzip.decompress(path.read_bytes()))
        maps={int(r['day']):r for r in receipts};assert len(maps)==len(receipts)
        selected=[(r['day'],r['selection']) for r in oldreport['daily'] if r['index']==index]+[(20261023,'known_example')]
        for day,selection in selected:
            f=panel[(panel['index']==index)&(panel[TIME]//1000000==day)].sort_values(TIME).reset_index(drop=True)
            bars=native[native['day']==day].reset_index(drop=True);bands=maps[day]['bands']
            result=replay(index,f,bars,maps[day]);qualified=qualify_series(result,bars,bands)
            prefix=qualify_series(replay(index,f.iloc[:24],bars.iloc[:24],maps[day]),bars.iloc[:24],bands)
            assert prefix==qualified[:24]
            checks.append(dict(index=index,day=day,rows=24))
            for i,((_,old),new) in enumerate(zip(result,qualified)):
                cutoff=new['asof'];p=new['position']
                if p['current_path']!='RECLAIM' or p['two_closes_above_band_upper']:
                    assert old['state']==new['state']
                else:assert new['state']=='BAND_RECLAIM_PENDING'
                if new['state']=='STRONG_REPAIR':assert p['strong_position_qualified']
                node=p['current_node'];band=None if node is None else next(b for b in bands if b['node']==node)
                previous=None if i==0 else float(bars['close'].iloc[i-1])
                if selection=='known_example':
                    if str(cutoff) in original_facts['known_example_market_snapshots']:
                        source=original_facts['known_example_market_snapshots'][str(cutoff)]['indices'][index]
                        examples.setdefault(cutoff,{})[index]=qualify_band(source,band,previous,float(bars['close'].iloc[i]))
                    continue
                assert oldframe.loc[(index,cutoff),'new']==old['state']
                decisions[(index,cutoff)]=new
                rows.append(dict(index=index,day=day,asof=cutoff,selection=selection,v2=old['state'],v3=new['state'],
                    node=node,band_upper=p['band_upper'],path=p['current_path'],
                    two_closes_above_band_upper=p['two_closes_above_band_upper'],
                    previous_close=previous,current_close=float(bars['close'].iloc[i])))
                if (index,cutoff) in snapshots:
                    facts.append(qualify_band(snapshots[(index,cutoff)],band,previous,float(bars['close'].iloc[i])))
    frame=pd.DataFrame(rows);assert len(frame)==1440 and len(facts)==120
    for r in oldreport['old_node_breaks_preserved']:
        new=decisions[(r['index'],r['asof'])]
        assert new['position']['previous_node_broken'] and new['state']==r['new']
    assert decisions[('cyb',20261022111500)]['state']=='BAND_RECLAIM_PENDING'
    summary={}
    for selection,g in frame.groupby('selection'):
        summary[selection]=dict(observations=len(g),v2_counts=g['v2'].value_counts().to_dict(),v3_counts=g['v3'].value_counts().to_dict(),
            changed=int(g['v2'].ne(g['v3']).sum()),
            strong_demoted=int(((g['v2']=='STRONG_REPAIR')&(g['v3']=='BAND_RECLAIM_PENDING')).sum()))
    known={str(t):dict(indices=pair,market=combine(pair['sse'],pair['cyb'])) for t,pair in examples.items()}
    manifest=json.loads((RISK/'live_monitor_fingerprints_20261023_v4_flow.json').read_bytes())
    for mapping in [inputs,manifest]:
        for path,h in mapping.items():assert sha(path)==h,path
    buffer=io.BytesIO();frame.to_parquet(buffer,index=False,compression='zstd')
    put('band_v3_states.parquet',buffer.getvalue())
    put('band_v3_changes.csv',frame[frame['v2']!=frame['v3']].to_csv(index=False).encode('utf-8-sig'))
    put('band_v3_llm_facts.json.gz',gzip.compress(json.dumps(dict(review_index_snapshots=facts,
        known_example_market_snapshots=known,protocol=original_facts['protocol']),ensure_ascii=False,allow_nan=False).encode(),mtime=0))
    put('band_v3_report.json',dict(groups=summary,prefix_checks=checks,old_node_breaks_preserved=9,
        production_fingerprints_verified=len(manifest),fits=0,paid_calls=0,production_changed=False,
        human_reviews_completed=0,independent_accuracy=False,peak_ram=peak()))
    print(json.dumps(dict(groups=summary,known={t:{i:s['state'] for i,s in p['indices'].items()} for t,p in known.items()}),ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        put('band_v3_failure_01.json',dict(error=repr(exc),status='STOP_NO_AUTORETRY'))
        raise
