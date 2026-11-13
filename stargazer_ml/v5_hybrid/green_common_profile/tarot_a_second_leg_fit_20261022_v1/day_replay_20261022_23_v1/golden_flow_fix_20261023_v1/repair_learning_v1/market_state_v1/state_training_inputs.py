"""Frozen causal input recipe. No fitting, imputation or rule-label features."""
import hashlib
import json
import numpy as np
import pandas as pd
from run_acceptance import HERE,STUDY,RISK,TIME,sha
from verify_five_fifteen_context import with_rsi

LABELS=['UNKNOWN','OBSERVE','LOCAL_REBOUND','BAND_RECLAIM_PENDING','REPAIR_PENDING_EVIDENCE',
        'WEAK_REPAIR','STRONG_REPAIR','BREAK_UNREPAIRED','DECLINE_ENHANCING']
SPLITS=['warmup','train','select','calibration','test_reused','known_diagnostic']
OWN=['repair_rsi6','repair_rsi15','repair_cross_up_count','repair_bars_since_cross',
     'repair_bars_since_low','repair_last_two_lows_rising','repair_last_two_closes_rising',
     'repair_volume_after_low_ratio','repair_volume_last3_previous3_ratio',
     'repair_low_rebound_return','repair_return_3bars','break_broken_close_run',
     'break_broken_close_depth_atr','break_broken_nearest_wave_count','break_broken_nearest_peak_groups']
BG=['index','day','previous_day','primary_flag','primary_age','primary_drawdown']


def split_for_day(day):
    if day<20250102:return 0
    if day<=20250930:return 1
    if day<=20251231:return 2
    if day<=20260331:return 3
    if day<=20260928:return 4
    return 5


def key_digest(base):
    hashed=pd.util.hash_pandas_object(base[['index',TIME]],index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def decode_geometry(archive,n):
    starts=archive['geometry_starts'];values=archive['geometry_values']
    assert starts[0]==0 and np.all(np.diff(starts)>0)
    return np.repeat(values,np.diff(np.r_[starts,n]),axis=0)


def feature_view(base,geometry,event_age,native_by_index,panels15=None):
    x=pd.DataFrame(index=base.index)
    for name in OWN:
        x[name]=np.where(base['index']=='sse',base['sse_'+name],base['cyb_'+name])
    x['is_cyb']=base['index'].eq('cyb').astype(float)
    x['event_age_known']=event_age.astype(float)
    days=base[TIME]//1000000
    for index in ['cyb','sse']:
        positions=np.flatnonzero(base['index'].eq(index).to_numpy());n=native_by_index[index].sort_values('ts')
        own=base.iloc[positions];t=own[TIME].to_numpy();day=t//1000000
        observed=n.set_index('ts').loc[t]
        daily=n.groupby('day').agg(high=('high','max'),low=('low','min'),close=('close','last'))
        previous=daily['close'].shift(1)
        tr=pd.concat([daily['high']-daily['low'],(daily['high']-previous).abs(),(daily['low']-previous).abs()],axis=1).max(axis=1,skipna=False)
        atr=tr.rolling(14,min_periods=14).mean().shift(1).reindex(day).to_numpy()
        close=observed['close'].to_numpy();low=observed['low'].to_numpy()
        previous_close=n.groupby('day')['close'].shift(1).to_numpy()
        prior=pd.Series(previous_close,index=n['ts']).loc[t].to_numpy()
        session_low=n.groupby('day')['low'].cummin();known_low=pd.Series(session_low.to_numpy(),index=n['ts']).loc[t].to_numpy()
        ref=geometry[positions]
        for name,value in dict(close_node_atr=(close-ref[:,0])/atr,close_upper_atr=(close-ref[:,2])/atr,
            previous_close_upper_atr=(prior-ref[:,2])/atr,bar_low_node_atr=(low-ref[:,0])/atr,
            known_day_low_node_atr=(known_low-ref[:,0])/atr,band_width_atr=(ref[:,2]-ref[:,1])/atr).items():
            x.loc[positions,name]=value
        x.loc[positions,'session_slot']=n.groupby('day').cumcount().set_axis(n['ts']).loc[t].to_numpy()
        fifteen=with_rsi(n) if panels15 is None else panels15[index]
        aligned=pd.merge_asof(pd.DataFrame({'asof':t}),fifteen[['ts','day','close','rsi6','rsi15']],
            left_on='asof',right_on='ts',direction='backward')
        today=aligned['day'].eq(day)
        x.loc[positions,'rsi6_15m']=aligned['rsi6'].where(today).to_numpy()
        x.loc[positions,'rsi15_15m']=aligned['rsi15'].where(today).to_numpy()
        x.loc[positions,'close15_upper_atr']=((aligned['close'].to_numpy()-ref[:,2])/atr)*np.where(today,1,np.nan)
    bg=pd.read_csv(HERE/'long_decline_daily.csv',usecols=BG).rename(columns={'day':'bg_day'})
    assert (bg['previous_day'].dropna()<bg.loc[bg['previous_day'].notna(),'bg_day']).all()
    merged=base[['index']].assign(bg_day=days.to_numpy()).merge(bg,on=['index','bg_day'],how='left',validate='many_to_one')
    for name in ['primary_age','primary_drawdown']:x['background_'+name]=merged[name].to_numpy()
    x['background_primary_flag']=merged['primary_flag'].map({True:1.,False:0.,'True':1.,'False':0.}).to_numpy()
    x=x.replace([np.inf,-np.inf],np.nan).astype(float)
    assert not any(k in x for k in ['state','target','repair_joint','two_close_reclaim','stop_above','full_band_two_close_reclaim','any_strong'])
    return x


def load_prepared():
    metadata=json.loads((HERE/'state_training_manifest.json').read_bytes())
    for path,h in metadata['inputs'].items():assert sha(path)==h,path
    archive_path=HERE/'state_training_targets.npz';assert sha(archive_path)==metadata['archive_sha256']
    base=pd.read_parquet(STUDY/'repair_effectiveness_v2/full_market_features.parquet').sort_values(['index',TIME],kind='stable').reset_index(drop=True)
    assert key_digest(base)==metadata['row_keys_sha256']
    with np.load(archive_path,allow_pickle=False) as a:
        geometry=decode_geometry(a,len(base));labels=a['labels'].copy();splits=a['splits'].copy()
        ages=a['event_age'].copy();groups=a['index_day'].copy();events=a['events'].copy()
    native={index:pd.read_parquet(RISK/f'index_minutes_v1/{index}_5min.parquet') for index in ['cyb','sse']}
    x=feature_view(base,geometry,ages,native)
    assert list(x)==metadata['features']
    keys=base[['index',TIME]].copy()
    keys['split']=[SPLITS[i] for i in splits];keys['index_day']=groups;keys['node_event']=events
    valid=labels!=LABELS.index('UNKNOWN')
    weights=np.zeros(len(labels));counts=pd.Series(groups[valid]).value_counts()
    weights[valid]=1/pd.Series(groups[valid]).map(counts).to_numpy()
    return x,labels,keys,weights
