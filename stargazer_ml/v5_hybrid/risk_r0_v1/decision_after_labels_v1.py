"""Strict future-slot price observations, not fills/PnL or model authorization."""
import numpy as np
import pandas as pd
from datetime import datetime,timedelta

HORIZON=1200
STATES={0:'CONDITIONAL_LABEL_COMPLETE',1:'CALENDAR_TAIL',2:'SOURCE_TAIL',
        3:'NECESSARY_PRICE_MISSING',4:'PRICE_EVIDENCE_PENDING',
        5:'MINUTE_DAILY_BASIS_UNALIGNED',6:'UNRESOLVED_ACTION_WINDOW'}


def minute_clock(days):
    days=np.asarray(days,dtype=np.int64)
    if len(days)==0 or not np.all(np.diff(days)>0):raise ValueError('INVALID_CALENDAR')
    minutes=np.r_[np.arange(571,691),np.arange(781,901)]
    return (days[:,None]*1000000+(minutes//60*10000+minutes%60*100)[None,:]).ravel()


def price_grid(frame,code,clock,quarantine):
    if not (frame.code.astype(str)==code).all():raise ValueError('PRICE_CODE_MISMATCH')
    fields=['open','high','low','close','volume','amount']
    duplicates=frame[frame.duplicated('date',keep=False)]
    bad=duplicates.groupby('date')[fields].nunique(dropna=False).max(axis=1)
    conflicts=set(bad[bad>1].index)
    data=frame.drop_duplicates('date').set_index('date').reindex(clock)[fields].to_numpy(dtype=float)
    o,h,l,c,v,a=data.T
    valid=(np.isfinite(data).all(axis=1)&(np.minimum.reduce([o,h,l,c])>0)&
           (l<=np.minimum(o,c))&(h>=np.maximum(o,c))&(v>=0)&(a>=0)&
           ~np.isin(clock,list(conflicts))&~np.isin(clock//1000000,list(quarantine)))
    result=data[:,:4].copy();result[~valid]=np.nan
    return result


def aligned_days(grid,days,bars,tolerance=.01):
    shaped=grid.reshape(len(days),240,4)
    complete=np.isfinite(shaped).all(axis=(1,2))
    out=np.zeros(len(days),dtype=bool)
    for i,day in enumerate(days):
        bar=bars.get(int(day))
        if not complete[i] or not bar:continue
        values=np.array([bar.get(k,np.nan) for k in ('open','high','low','close')],dtype=float)
        if not np.isfinite(values).all() or not (0<values[2]<=min(values[0],values[3])<=max(values[0],values[3])<=values[1]):continue
        actual=np.array([shaped[i,0,0],shaped[i,:,1].max(),shaped[i,:,2].min(),shaped[i,-1,3]])
        out[i]=np.all(np.abs(actual-values)<=tolerance+1e-12)
    return out


def future_minimum(low,horizon=HORIZON):
    if not isinstance(horizon,int) or horizon<1:raise ValueError('INVALID_HORIZON')
    # At i, rolling value at i+horizon uses precisely i+1..i+horizon.
    return pd.Series(low,dtype=float).rolling(horizon,min_periods=horizon).min().shift(-horizon).to_numpy()


def label_rows(frame,grid,days,evidence=None,horizon=HORIZON):
    """All input keys retained. Raw diagnosis is separate from admitted labels."""
    t=frame.cutoff_index.to_numpy(dtype=int)*15+14
    reference=frame.stock_close.to_numpy(dtype=float)
    if (t<0).any() or (t>=len(grid)).any() or not np.all(grid[t,3]==reference):
        raise ValueError('FEATURE_REFERENCE_PRICE_CHANGED')
    end=t+horizon;inside=end<len(grid)
    finite=np.flatnonzero(np.isfinite(grid[:,2]))
    last=int(finite[-1]) if len(finite) else -1
    raw_min=future_minimum(grid[:,2],horizon)[t]
    raw_mae=raw_min/reference-1
    complete=inside&(end<=last)&np.isfinite(raw_mae)
    status=np.full(len(t),4,dtype=np.uint8)
    status[~inside]=1
    status[inside&(end>last)]=2
    status[inside&(end<=last)&~complete]=3
    mae=np.full(len(t),np.nan);y=np.full(len(t),np.nan)
    if evidence is not None:
        day_ok=evidence['aligned'];issues=evidence['issues'];factor=evidence['factors']
        if len(day_ok)!=len(days) or len(factor)!=len(days) or len(issues)!=len(days):raise ValueError('EVIDENCE_LENGTH')
        if not (np.isfinite(factor)&(factor>0)).all():raise ValueError('INVALID_FACTOR')
        start_day=t//240;end_day=np.minimum(end//240,len(days)-1)
        bad=np.r_[0,np.cumsum(~day_ok)]
        problem=np.r_[0,np.cumsum(issues)]
        basis=(bad[end_day+1]-bad[start_day])==0
        action=(problem[end_day+1]-problem[start_day])==0
        status[complete&~basis]=5
        status[complete&basis&~action]=6
        normal=np.repeat(factor,240)
        scaled=grid[:,2]*normal
        if not np.isfinite(scaled[np.isfinite(grid[:,2])]).all():raise ValueError('NONFINITE_NORMALIZATION')
        anchors=reference*normal[t]
        if not (np.isfinite(anchors)&(anchors>0)).all():raise ValueError('INVALID_NORMALIZED_REFERENCE')
        normalized_min=future_minimum(scaled,horizon)[t]
        normalized_mae=normalized_min/anchors-1
        eligible=complete&basis&action&np.isfinite(normalized_mae)
        status[eligible]=0;mae[eligible]=normalized_mae[eligible]
        y[eligible]=(mae[eligible]<=-.08+1e-12).astype(int)
    diagnostic=np.full(len(t),-1,dtype=np.int8)
    diagnostic[complete]=(raw_mae[complete]<=-.08+1e-12).astype(np.int8)
    result=frame[['code','cutoff_index']].copy().reset_index(drop=True)
    result['diagnostic_mae_raw_1200m']=raw_mae.astype('float32')
    result['diagnostic_tail_raw_1200m']=diagnostic
    result['mae_1200m']=mae
    result['y_tail_1200m']=pd.array(y,dtype='Int8')
    result['label_state']=status
    return result,end


def proposed_split(decisions,ends,segments):
    """By decision date; crossing endpoints purge without changing ownership."""
    days=np.asarray(decisions)//1000000
    assignment=np.full(len(days),-1,dtype=np.int8);purged=np.ones(len(days),dtype=bool)
    for i,seg in enumerate(segments):
        selected=(days>=seg['start'])&(days<=seg['end'])
        assignment[selected]=i
        following=int((datetime.strptime(str(seg['end']),'%Y%m%d')+timedelta(days=1)).strftime('%Y%m%d'))
        boundary=(segments[i+1]['start'] if i+1<len(segments) else following)*1000000
        purged[selected]=~(np.isfinite(ends[selected])&(ends[selected]<boundary)&(ends[selected]>np.asarray(decisions)[selected]))
    return assignment,purged
