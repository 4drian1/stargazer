"""Postclose stock-day evidence and continuous next-session risk, no estimator."""
import numpy as np
import pandas as pd
from multitimeframe_momentum_v1 import MomentumState
from materialize_multitimeframe_momentum_v1 import PARAMETERS
from left_side_candidates_v1 import candidates


def stock_days(code, records, days, audit):
    days = np.asarray(days,dtype=np.int64)
    if len(days)==0 or not np.all(np.diff(days)>0):
        raise ValueError('SORTED_UNIQUE_CALENDAR_REQUIRED')
    bars, conflicts = {}, set()
    fields = ['open','high','low','close','volume','amount']
    for key,row in records:
        if row is None:continue
        if not key.startswith('日k:'+code+':') or row.get('code') != code:
            raise ValueError('SOURCE_CODE_MISMATCH')
        date = int(key.rsplit(':',1)[1])
        if date != row.get('date'):raise ValueError('SOURCE_DATE_MISMATCH')
        values = {f:row.get(f,np.nan) for f in fields}
        if date in bars and bars[date] != values:conflicts.add(date)
        bars.setdefault(date,values)
    raw = pd.DataFrame([bars.get(int(d),{}) for d in days],columns=fields,dtype=float)
    raw['day'] = days
    o,h,l,c = [raw[f] for f in ['open','high','low','close']]
    valid = np.isfinite(raw[['open','high','low','close']]).all(axis=1) & (raw[['open','high','low','close']]>0).all(axis=1)
    valid &= (l<=np.minimum(o,c)) & (h>=np.maximum(o,c)) & ~raw.day.isin(conflicts)
    factors = np.ones(len(days)); issue=np.zeros(len(days),dtype=bool)
    for event in audit['events']:
        mult=event.get('multiplier')
        if mult is None:continue
        if event['status']!='ORDINARY_PARAMETERS_CORROBORATED' or not np.isfinite(mult) or mult<=0:
            raise ValueError('UNACCEPTED_FACTOR')
        factors[days>=event['date']] *= mult
    for item in audit['issues']:
        issue |= True if item.get('date') is None else days == item['date']
    if not (np.isfinite(factors)&(factors>0)).all():raise ValueError('INVALID_CUMULATIVE_FACTOR')
    usable = valid.to_numpy() & ~issue
    comparable = raw['close'].to_numpy()*factors
    comparable[~usable]=np.nan
    state=MomentumState(**PARAMETERS)
    indicators=pd.DataFrame([state.update(float(v)) for v in comparable])
    comparisons = indicators[['rsi14','macd_dif','macd_hist','bbi']].copy()
    comparisons.insert(0,'close',comparable)
    left=candidates(comparisons).add_prefix('stock_')
    features=pd.DataFrame(index=raw.index)
    for k in [1,2,3,5,20]:
        price=pd.Series(comparable)
        complete=price.notna().rolling(k+1,min_periods=k+1).sum().eq(k+1)
        features[f'stock_return{k}d']=(price/price.shift(k)-1).where(complete)
    for name in ['rsi14','bbi_slope','close_over_bbi_minus_one','price_above_bbi']:
        features['stock_daily_'+name]=indicators[name].astype(float)
    for name in ['macd_dif','macd_dea','macd_hist']:
        features['stock_daily_'+name+'_over_close']=indicators[name].astype(float)/comparable
    width=h-l
    features['stock_close_location_in_day_range']=((c-l)/width).where(usable & (width>0))
    for field in ['amount','volume']:
        values=raw[field].where(usable & np.isfinite(raw[field]) & (raw[field]>=0))
        baseline=values.shift(1).rolling(5,min_periods=5).mean()
        ratio=(values/baseline).where(baseline>0)
        # Stock split/cash-adjusted price factor is not a valid share-volume multiplier.
        stable=pd.Series(factors).rolling(6,min_periods=6).apply(lambda x:float(np.all(x==x.iloc[-1])),raw=False).eq(1)
        if field=='volume':ratio=ratio.where(stable)
        features['stock_'+field+'_ratio_prior5d']=ratio
    result=pd.concat([pd.DataFrame(dict(code=code,day=days,reference_close_raw=c,
        conditional_daily_evidence=usable,corporate_action_coverage_verified=False)),features,left],axis=1)
    target=np.full(len(days),np.nan);raw_return=np.full(len(days),np.nan)
    states=np.full(len(days),'SOURCE_OR_ACTION_PENDING',dtype=object)
    for i in range(len(days)-1):
        if usable[i] and usable[i+1]:
            value=1-float(raw.low.iloc[i+1])*factors[i+1]/(float(c.iloc[i])*factors[i])
            if not np.isfinite(value):raise ValueError('INVALID_TARGET')
            target[i]=max(0.,value);raw_return[i]=-value
            states[i]='CONDITIONAL_NEXT_SESSION_COMPLETE'
    states[-1]='CALENDAR_TAIL'
    result['next_session_adverse_excursion']=target
    result['next_session_low_return']=raw_return
    result['label_state']=states
    result['label_end_time']=pd.array([int(d)*1000000+150000 for d in days[1:]]+[None],dtype='Int64')
    result['decision_ts_assumed_end']=days*1000000+150000
    result['training_eligible']=False
    result['actual_available_at']=None
    result['data_complete_time']=None
    return result
