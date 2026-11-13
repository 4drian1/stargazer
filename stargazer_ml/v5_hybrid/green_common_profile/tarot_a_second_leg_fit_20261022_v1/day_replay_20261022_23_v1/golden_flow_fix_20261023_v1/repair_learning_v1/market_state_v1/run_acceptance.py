"""Offline state distribution, fixed-day snapshots, true-source prefix checks."""
import io
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;STUDY=HERE.parent
sys.path[:0]=[str(STUDY),str(STUDY/'market_break_repair_v1')]
from run_research import put as parent_put,sha,RISK,peak,TIME
from repair_features import market_panel
from market_features import build
from state_rules import describe,combine,LABELS


def put(name,value):
    body=value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2).encode()
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())+len(body)<=2*1024**2
    parent_put(HERE.name+'/'+name,body)


def main():
    cache_path=STUDY/'repair_effectiveness_v2/full_market_features.parquet'
    case_path=STUDY.parent/'real_deepseek_acceptance_v1/input.json'
    paths=[HERE/'state_rules.py',HERE/'test_state_rules.py',Path(__file__),cache_path,case_path,
           STUDY/'repair_features.py',STUDY/'market_break_repair_v1/market_features.py']
    paths += [RISK/f'index_minutes_v1/{n}_5min.parquet' for n in ['sse','cyb']]
    inputs={str(p):sha(p) for p in paths}
    cutoffs=[20261023094500,20261023100000,20261023103000,20261023110000,20261023113000,20261023150000]
    put('contract.json',dict(inputs=inputs,labels=LABELS,main_axis='SSE priority, not numeric score average',
        label_status='Engineering rule references awaiting independent review; not ML ground truth',
        strong='confirmed position path + RSI6>15 + known low-relative contraction + last two lows nondecreasing',
        not_buying_rules=True,existing_amp_retained_as_auxiliary=True,cutoffs=cutoffs,fits=0,paid_calls=0))
    panel=pd.read_parquet(cache_path)
    state_rows=[];snapshots={t:{} for t in cutoffs};prefix_checks=[]
    for index in ['sse','cyb']:
        own=panel[panel['index'].eq(index)].sort_values(TIME)
        for _,row in own.iterrows():
            t=int(row[TIME]);state=describe(index,t,row.to_dict())
            state_rows.append(dict(index=index,asof=t,state=state['state'],direction=state['direction']))
            if t in snapshots:snapshots[t][index]=state
        native=pd.read_parquet(RISK/f'index_minutes_v1/{index}_5min.parquet').sort_values('ts')
        for cutoff in [20261023103000,20261023113000]:
            truncated=native[native.ts<=cutoff]
            old,_=market_panel(truncated,index,[20261023]);new,_=build(truncated,index,[20261023])
            recomputed=old.merge(new,on=TIME,validate='one_to_one')
            full=own[(own[TIME]//1000000==20261023)&(own[TIME]<=cutoff)]
            before=recomputed[recomputed[TIME]<=cutoff].reset_index(drop=True)
            pd.testing.assert_frame_equal(before,full[before.columns].reset_index(drop=True))
            for _,row in before.iterrows():
                assert describe(index,int(row[TIME]),row.to_dict())==describe(index,int(row[TIME]),own[own[TIME].eq(row[TIME])].iloc[0].to_dict())
            prefix_checks.append(dict(index=index,cutoff=cutoff,rows=len(before)))
    fixed=[]
    bound=json.loads(case_path.read_bytes());risk=bound['risk'];context=risk['context']
    for cutoff,states in snapshots.items():
        assert set(states)=={'sse','cyb'}
        item=dict(cutoff=cutoff,indices=states,market=combine(states['sse'],states['cyb']),
            board=dict(status='UNKNOWN',reason='本时点未绑定同规格板块修复证据，不由指数代替板块'),
            stock_amplitude_auxiliary=None,automatic_action=False)
        if cutoff==20261023113000:
            assert int(risk['asof_ts'])==cutoff
            item['board']=dict(status='PARTIAL_CROSS_SECTION_ONLY',repair_state='UNKNOWN',
                observed=context['board_states'],scope='关联板块部分覆盖截面，不是科创板指数，不据此生成板块强/弱修复标签')
            item['stock_amplitude_auxiliary']=dict(code='688004',amp_232=context['amp_232'],amp_266=context['amp_266'],
                meaning='未来120交易分钟不利下探幅度点估计，非概率/上界/黄金支撑位')
        fixed.append(item)
    states=pd.DataFrame(state_rows);buf=io.BytesIO();states.to_parquet(buf,index=False,compression='zstd')
    put('reference_states.parquet',buf.getvalue());put('fixed_day_snapshots.json',fixed)
    protected=json.loads((STUDY/'plan.json').read_bytes())['inputs']
    production=json.loads((RISK/'live_monitor_fingerprints_20261023_v4_flow.json').read_bytes())
    for manifest in [inputs,protected,production]:
        for p,h in manifest.items():assert sha(p)==h,p
    result=dict(status='OFFLINE_ENGINEERING_ACCEPTANCE_NOT_MODEL_VALIDATION',rows=len(states),
        days={n:int((states.loc[states['index'].eq(n),'asof']//1000000).nunique()) for n in ['sse','cyb']},
        distributions=states.groupby(['index','state']).size().unstack(fill_value=0).to_dict('index'),
        prefix_checks=prefix_checks,production_fingerprints=len(production),fits=0,paid_calls=0,peak_ram=peak(),
        stock_amp_retained=True,model_trained=False,board_repair_verified=False)
    put('report.json',result)
    print(json.dumps(dict(report=result,fixed=[dict(cutoff=r['cutoff'],sse=r['indices']['sse']['label'],cyb=r['indices']['cyb']['label'],market=r['market']['label']) for r in fixed]),ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        put('failure.json',dict(error=repr(exc),status='STOPPED'))
        raise
