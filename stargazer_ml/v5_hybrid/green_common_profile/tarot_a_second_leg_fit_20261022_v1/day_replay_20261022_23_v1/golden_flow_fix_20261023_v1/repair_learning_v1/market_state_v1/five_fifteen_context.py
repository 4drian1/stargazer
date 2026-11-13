"""Completed 15m context, independent of the leading completed-5m decision."""
import math
import numpy as np
import pandas as pd

ENDS=list(range(9*60+45,11*60+31,15))+list(range(13*60+15,15*60+1,15))


def timestamp(day,minute):
    return int(day)*1000000+(minute//60)*10000+(minute%60)*100


def aggregate_fifteen(native):
    assert (native['ts']//1000000==native['day']).all()
    data=native.set_index('ts');assert data.index.is_unique
    rows=[];last=int(native['ts'].max()) if len(native) else 0
    for day in sorted(native['day'].unique()):
        allowed={timestamp(day,e-delta) for e in ENDS for delta in [10,5,0]}
        assert set(native.loc[native['day']==day,'ts']).issubset(allowed)
        for end in ENDS:
            stop=timestamp(day,end)
            if stop>last:continue
            slots=[timestamp(day,end-delta) for delta in [10,5,0]]
            missing=[t for t in slots if t not in data.index]
            valid=not missing
            if valid:
                bars=data.loc[slots]
                valid=bool(np.isfinite(bars[['open','high','low','close','raw_volume']].to_numpy()).all())
            row=dict(day=int(day),ts=stop,complete=valid,source_5m_ends=slots,missing_slots=missing,
                open=np.nan,high=np.nan,low=np.nan,close=np.nan,raw_volume=np.nan)
            if valid:
                row.update(open=float(bars['open'].iloc[0]),high=float(bars['high'].max()),
                    low=float(bars['low'].min()),close=float(bars['close'].iloc[-1]),
                    raw_volume=float(bars['raw_volume'].sum()))
            rows.append(row)
    return pd.DataFrame(rows,columns=['day','ts','complete','source_5m_ends','missing_slots',
        'open','high','low','close','raw_volume'])


def context_at(panel,cutoff,band):
    known=panel[panel['ts']<=cutoff]
    result=dict(asof=int(cutoff),primary_timeframe='5m',auxiliary_timeframe='15m',
        role='COMPLETED_PRICE_RSI_CONTEXT_NOT_A_UNIFORM_ENTRY_GATE',required_for_action=False,
        last_window_end=None,last_completed_at=None,today_complete=False,price_band_position=None,
        rsi6=None,rsi15=None,rsi_state='UNKNOWN',supports_price_rsi=None)
    if not len(known):return dict(result,status='NO_COMPLETED_15M_CONTEXT')
    row=known.iloc[-1];result['last_window_end']=int(row['ts'])
    result['source_5m_ends']=[int(t) for t in row['source_5m_ends']]
    result['missing_slots']=[int(t) for t in row['missing_slots']]
    if row['complete']:result['last_completed_at']=int(row['ts'])
    if int(row['day'])!=cutoff//1000000:return dict(result,status='TODAY_15M_NOT_COMPLETE')
    if not row['complete']:return dict(result,status='LATEST_15M_SOURCE_INCOMPLETE')
    result['today_complete']=True
    for key in ['open','high','low','close','raw_volume']:
        result[key]=float(row[key])
    for key in ['rsi6','rsi15']:
        value=row.get(key,np.nan)
        result[key]=float(value) if math.isfinite(value) else None
    if result['rsi6'] is not None and result['rsi15'] is not None:
        result['rsi_state']='FAST_ABOVE' if result['rsi6']>result['rsi15'] else 'FAST_NOT_ABOVE'
    if band is not None:
        result['selected_band_lower']=band['lower'];result['selected_band_upper']=band['upper']
        result['price_band_position']=('AT_OR_ABOVE_UPPER' if row['close']>=band['upper'] else
            'INSIDE_BAND' if row['close']>=band['lower'] else 'BELOW_LOWER')
        if result['rsi_state']!='UNKNOWN':
            result['supports_price_rsi']=bool(result['price_band_position']=='AT_OR_ABOVE_UPPER' and result['rsi_state']=='FAST_ABOVE')
    return dict(result,status='COMPLETED_15M_FACTS_ONLY_NOT_COMPOSITE_REPAIR_LABEL')
