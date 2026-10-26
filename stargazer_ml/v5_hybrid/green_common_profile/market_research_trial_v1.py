"""Frozen 39-field market classifier, explicitly experimental. No fitting or labels."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

GCP = Path(__file__).resolve().parent
FLOW = GCP/'tarot_a_second_leg_fit_20261022_v1/day_replay_20261022_23_v1/golden_flow_fix_20261023_v1'
STATE = FLOW/'repair_learning_v1/market_state_v1'
sys.path[:0] = [str(STATE), str(STATE.parent), str(STATE.parent/'market_break_repair_v1')]
from repair_features import market_panel, index_map
from market_features import build
from compare_components_v2 import replay
from compare_band_reclaim_v3 import qualify_series
from state_training_inputs import OWN, LABELS
from node_events import trace_events
from verify_five_fifteen_context import with_rsi
from state_rules import describe

_model = None


def inputs(native, index, cutoff):
    """Rebuild only this session, using historical prefixes and the frozen recipe."""
    n = native.rename(columns={'date':'ts','volume':'raw_volume'}).copy()
    n = n.loc[n.ts<=cutoff].sort_values('ts').reset_index(drop=True)
    day = cutoff//1000000
    bars = n[n.day==day].reset_index(drop=True)
    if bars.empty or int(bars.ts.iloc[-1])!=cutoff:
        raise ValueError('NO_SAME_CUTOFF_MARKET_PREFIX')
    basic,_ = market_panel(n,index,[day])
    broken,receipts = build(n,index,[day])
    base = basic.merge(broken,on='decision_ts_assumed_end',validate='one_to_one')
    base = base.loc[base.decision_ts_assumed_end<=cutoff].reset_index(drop=True)
    receipt=receipts[0]; bands=receipt['bands']; atr=receipt['atr_prior']
    if atr is None or atr<=0:raise ValueError('PRIOR_ATR_UNKNOWN')
    parts=qualify_series(replay(index,base,bars,receipt),bars,bands)
    records=[];geom=[];current=[]
    for i,p in enumerate(parts):
        pos=p['position']; node=pos['current_node']
        records.append(dict(index=index,day=day,asof=p['asof'],close=float(bars.close.iloc[i]),
            low=float(bars.low.iloc[i]),known_day_low=float(bars.low.iloc[:i+1].min()),
            path=pos['current_path'],node=node))
        def band(ref):
            b=next((b for b in bands if b['node']==ref),None)
            return [np.nan]*3 if b is None else [b['node'],b['lower'],b['upper']]
        geom.append(band(pos['tracked_node']));current.append(band(node))
    trace,_=trace_events(records)
    x=pd.DataFrame({name:base[index+'_'+name] for name in OWN})
    x['is_cyb']=float(index=='cyb');x['event_age_known']=[r['event_age'] or 0 for r in trace]
    t=bars.ts.to_numpy();close=bars.close.to_numpy();prior=bars.close.shift(1).to_numpy()
    low=bars.low.to_numpy();known_low=bars.low.cummin().to_numpy()
    fifteen=with_rsi(n)
    aligned=pd.merge_asof(pd.DataFrame({'asof':t}),fifteen[['ts','day','close','rsi6','rsi15']],
        left_on='asof',right_on='ts',direction='backward')
    same=aligned.day.eq(day).to_numpy()
    def geometry(g):
        g=np.asarray(g,float)
        return dict(close_node_atr=(close-g[:,0])/atr,close_upper_atr=(close-g[:,2])/atr,
            previous_close_upper_atr=(prior-g[:,2])/atr,bar_low_node_atr=(low-g[:,0])/atr,
            known_day_low_node_atr=(known_low-g[:,0])/atr,band_width_atr=(g[:,2]-g[:,1])/atr,
            close15_upper_atr=(aligned.close.to_numpy()-g[:,2])/atr*np.where(same,1,np.nan))
    old=geometry(geom)
    for k,v in old.items():
        if k!='close15_upper_atr':x[k]=v
    x['session_slot']=np.arange(len(bars))
    x['rsi6_15m']=aligned.rsi6.where(same).to_numpy();x['rsi15_15m']=aligned.rsi15.where(same).to_numpy()
    x['close15_upper_atr']=old['close15_upper_atr']
    days=sorted(n.day.unique());prior_days=[d for d in days if d<day]
    mapping=index_map(n.rename(columns={'ts':'date'}),int(prior_days[-1])*1000000+150000) if prior_days else {}
    age=draw=flag=np.nan
    if mapping.get('status')=='AVAILABLE_EXECUTION_REFERENCE_NOT_PYXIS_FEATURE':
        age=days.index(prior_days[-1])-days.index(mapping['high_day'])
        previous_close=float(n.loc[n.day==prior_days[-1],'close'].iloc[-1])
        draw=1-previous_close/mapping['high'];flag=float(age>=15 and draw>(.07 if index=='sse' else .25))
    x['background_primary_age']=age;x['background_primary_drawdown']=draw;x['background_primary_flag']=flag
    x['repair_rsi_gap']=x.repair_rsi6-x.repair_rsi15
    for k,v in geometry(current).items():x['current_'+k]=v
    ratio=x.repair_volume_after_low_ratio
    x['volume_contracting_flag']=np.where(ratio.notna(),(ratio<=1).astype(float),np.nan)
    return x.replace([np.inf,-np.inf],np.nan),base,parts,receipt


def predict(native,index,cutoff):
    global _model
    if _model is None:
        import lightgbm as lgb
        path=STATE/'volume_flag_model.txt.gz'
        _model=lgb.Booster(model_str=gzip.decompress(path.read_bytes()).decode())
    x,base,parts,receipt=inputs(native,index,cutoff)
    names=_model.feature_name()
    if set(names)!=set(x) or len(names)!=39:raise ValueError('RESEARCH_FEATURE_CONTRACT_MISMATCH')
    probability=_model.predict(x[names].iloc[[-1]])[0]
    raw=LABELS[int(np.argmax(probability))+1]; protected=raw
    row=base.iloc[-1];rule=describe(index,cutoff,row.to_dict());o=rule['observed']
    has=parts[-1]['position']['current_path']!='NONE'
    falling=o['bars_since_low']==0 and o['return_3bars'] is not None and o['return_3bars']<0 and o['closes_rising']==0
    if not rule['missing'] and falling and not has and raw not in ('BREAK_UNREPAIRED','DECLINE_ENHANCING'):
        protected='DECLINE_ENHANCING'
    last=x.iloc[-1]
    if o['reclaim']==1 and len(x)>=2 and np.isfinite(last.close_upper_atr) and np.isfinite(last.previous_close_upper_atr):
        if not(last.close_upper_atr>=0 and last.previous_close_upper_atr>=0) and protected in ('STRONG_REPAIR','WEAK_REPAIR','REPAIR_PENDING_EVIDENCE'):
            protected='BAND_RECLAIM_PENDING'
    return dict(asof=cutoff,index=index,status='EXPERIMENTAL_MODEL_OUTPUT',raw_state=raw,protected_state=protected,
        model_sha256=hashlib.sha256((STATE/'volume_flag_model.txt.gz').read_bytes()).hexdigest(),
        acceptance_passed=False,volume_verified=False,volume_assumption='UNVERIFIED_RAW_VOLUME_RESEARCH_ONLY',
        current_path=parts[-1]['position']['current_path'],features=39,
        qualification='用户授权试运行研究输出；未核销量能假设，非确认事实、非未来收益概率',
        resonance_bands=[{k:b[k] for k in ('node','lower','upper','source_waves','peak_groups')} for b in receipt['bands']])
