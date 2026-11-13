"""Bind completed-15m context without changing any frozen 5m state."""
import gzip
import json
import pandas as pd
from run_acceptance import HERE,STUDY,RISK,put,sha,peak
from repair_features import _rsi
from five_fifteen_context import aggregate_fifteen,context_at


def with_rsi(native):
    panel=aggregate_fifteen(native)
    panel['rsi6']=_rsi(panel['close'].to_numpy(),6)
    panel['rsi15']=_rsi(panel['close'].to_numpy(),15)
    return panel


def main():
    paths=[HERE/'verify_five_fifteen_context.py',HERE/'five_fifteen_context.py',HERE/'test_five_fifteen_context.py',
        HERE/'band_v3_states.parquet',HERE/'components_v2_states.parquet',HERE/'band_v3_llm_facts.json.gz',
        STUDY/'repair_features.py']
    for index in ['sse','cyb']:
        paths += [RISK/f'index_minutes_v1/{index}_5min.parquet',
            STUDY/f'market_break_repair_v1/{index}_source_receipts.json.gz',
            STUDY/f'repair_effectiveness_v2/phase_missing_{index}_maps.json.gz',
            HERE/f'switch_missing_{index}_receipts.json.gz']
    inputs={str(p):sha(p) for p in paths}
    put('five_fifteen_plan.json',dict(inputs=inputs,primary='Existing frozen completed-5m V3 state',
        auxiliary='Completed-15m price/RSI6/15 facts only; not a new composite repair classifier',
        no_uniform_15m_wait=True,primary_rules_changed=False,fits=0,paid_calls=0))
    five=pd.read_parquet(HERE/'band_v3_states.parquet')
    older=pd.read_parquet(HERE/'components_v2_states.parquet').set_index(['index','asof'])
    assert older.index.is_unique
    payload=json.loads(gzip.decompress((HERE/'band_v3_llm_facts.json.gz').read_bytes()))
    packets=payload['review_index_snapshots']+[p for m in payload['known_example_market_snapshots'].values() for p in m['indices'].values()]
    original_states=[(p['index'],p['asof'],p['state']) for p in packets]
    bindings={};checks=[];sources={};leading_examples=[];support_counts={}
    for index in ['sse','cyb']:
        native=pd.read_parquet(RISK/f'index_minutes_v1/{index}_5min.parquet').sort_values('ts')
        fifteen=with_rsi(native)
        assert len(fifteen)==len(native)//3 and fifteen['complete'].all()
        sources[index]=dict(five_minute_rows=len(native),fifteen_minute_rows=len(fifteen))
        receipts=[]
        for path in [STUDY/f'market_break_repair_v1/{index}_source_receipts.json.gz',
                     STUDY/f'repair_effectiveness_v2/phase_missing_{index}_maps.json.gz',
                     HERE/f'switch_missing_{index}_receipts.json.gz']:
            receipts+=json.loads(gzip.decompress(path.read_bytes()))
        maps={int(r['day']):r for r in receipts}
        def band_for(day,node):
            return None if node is None or pd.isna(node) else next(b for b in maps[day]['bands'] if b['node']==node)
        group=five[five['index']==index].sort_values('asof')
        statuses={};leading=0;strong=0
        for _,r in group.iterrows():
            cutoff=int(r['asof']);day=int(r['day']);node=r['node'];role='CURRENT_SELECTED_NODE'
            if pd.isna(node):node=older.loc[(index,cutoff),'tracked_node'];role='PREVIOUS_TRACKED_REFERENCE'
            band=band_for(day,node);context=context_at(fifteen,cutoff,band)
            context['reference_role']=role if band is not None else 'NO_CONFIRMED_REFERENCE'
            assert context['last_window_end'] is None or context['last_window_end']<=cutoff
            assert all(t<=cutoff for t in context.get('source_5m_ends',[]))
            assert not context['required_for_action']
            bindings[(index,cutoff)]=context
            statuses[context['status']]=statuses.get(context['status'],0)+1
            if r['v3']=='STRONG_REPAIR':
                strong+=1
                if context['supports_price_rsi'] is not True:
                    leading+=1
                    if len(leading_examples)<6:
                        leading_examples.append(dict(index=index,asof=cutoff,primary_state=r['v3'],
                            last_15m_at=context['last_completed_at'],support=context['supports_price_rsi']))
        support_counts[index]=dict(primary_strong_points=strong,strong_without_15m_price_rsi_support=leading,statuses=statuses)
        days=sorted(group['day'].unique())
        for day,clock in [(int(days[0]),95500),(int(days[-1]),131000)]:
            cutoff=day*1000000+clock
            part=with_rsi(native[native['ts']<=cutoff])
            assert context_at(part,cutoff,None)==context_at(fifteen,cutoff,None)
            checks.append(dict(index=index,day=day,cutoff=cutoff,native_source_truncated=True))
        for packet in [p for p in packets if p['index']==index]:
            cutoff=packet['asof'];key=(index,cutoff)
            if key not in bindings:
                node=packet['position']['current_node'] or packet['position']['tracked_node']
                bindings[key]=context_at(fifteen,cutoff,band_for(cutoff//1000000,node))
            packet['confirmation_15m']=bindings[key]
    assert original_states==[(p['index'],p['asof'],p['state']) for p in packets]
    manifest=json.loads((RISK/'live_monitor_fingerprints_20261023_v4_flow.json').read_bytes())
    for mapping in [inputs,manifest]:
        for path,h in mapping.items():assert sha(path)==h,path
    payload['protocol'].update(primary_timeframe='5m',auxiliary_timeframe='15m',
        uniform_15m_wait_required=False,auxiliary_scope='Completed OHLCV/price-band/RSI facts; not full repair confirmation')
    put('five_fifteen_llm_facts.json.gz',gzip.compress(json.dumps(payload,ensure_ascii=False,allow_nan=False).encode(),mtime=0))
    put('five_fifteen_report.json',dict(sources=sources,support_counts=support_counts,leading_examples=leading_examples,
        input_primary_rows_verified=1440,serialized_primary_states_unchanged=len(packets),
        actual_source_prefix_checks=checks,production_fingerprints_verified=len(manifest),
        primary_rules_changed=False,production_changed=False,fits=0,paid_calls=0,peak_ram=peak()))
    print(json.dumps(dict(sources=sources,support_counts=support_counts,leading_examples=leading_examples,prefix_checks=checks),ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        put('five_fifteen_failure_01.json',dict(error=repr(exc),status='STOP_NO_AUTORETRY'))
        raise
