"""Completed five/fifteen minute low comparisons using the existing minute clock."""
import math


def aggregate15(bars):
    groups={}
    for b in bars:
        minute=b['t']//10000*60+b['t']%10000//100
        end=(minute+14)//15*15
        groups.setdefault((b['day'],end),[]).append(b)
    result=[]
    for (day,end),group in sorted(groups.items()):
        group=sorted(group,key=lambda b:b['t'])
        expected=[(end-d)//60*10000+(end-d)%60*100 for d in (10,5,0)]
        if [b['t'] for b in group]!=expected:continue
        vols=[b.get('volume') for b in group]
        result.append(dict(day=day,t=expected[-1],open=group[0]['open'],high=max(b['high'] for b in group),
            low=min(b['low'] for b in group),close=group[-1]['close'],
            volume=sum(vols) if all(v is not None for v in vols) else None))
    return result


def pair(bars,previous_day,day,period):
    import numpy as np
    import pandas as pd
    from golden_ladder_v3 import _rsi
    previous=[(i,b) for i,b in enumerate(bars) if b['day']==previous_day]
    current=[(i,b) for i,b in enumerate(bars) if b['day']==day]
    result=dict(period=period,status='UNKNOWN',basis='COMPLETE_END_BARS; RSI_WILDER; DIF_EMA12_26_ADJUST_FALSE',
                previous=None,current=None)
    if len(previous)!=(48 if period==5 else 16):return dict(result,reason='PREVIOUS_DAY_INCOMPLETE')
    if not current:return dict(result,reason='NO_COMPLETE_CURRENT_BAR')
    old_i,old=min(previous,key=lambda ib:ib[1]['low']);new_i,new=min(current,key=lambda ib:ib[1]['low'])
    closes=[b['close'] for b in bars]
    fast,slow=_rsi(closes,6),_rsi(closes,15)
    series=pd.Series(closes,dtype=float)
    dif=series.ewm(span=12,adjust=False).mean()-series.ewm(span=26,adjust=False).mean()
    dea=dif.ewm(span=9,adjust=False).mean()
    def item(i,b):
        def known(v):return float(v) if np.isfinite(v) else None
        return dict(time=b['day']*1000000+b['t'],price=b['low'],close=b['close'],volume=b.get('volume'),
            rsi6=known(fast[i]),rsi15=known(slow[i]),dif=float(dif.iloc[i]) if i>=25 else None,
            dea=float(dea.iloc[i]) if i>=33 else None,macd_hist=float(2*(dif.iloc[i]-dea.iloc[i])) if i>=33 else None)
    a,b=item(old_i,old),item(new_i,new)
    sufficient=all(x[k] is not None for x in (a,b) for k in ('rsi6','rsi15','dif'))
    return dict(result,status='KNOWN' if sufficient else 'PARTIAL',reason=None if sufficient else 'INDICATOR_WARMUP',
        previous=a,current=b,evidence_time=current[-1][1]['day']*1000000+current[-1][1]['t'],
        dif_divergence=bool(b['price']<a['price'] and b['dif']>a['dif']) if sufficient else None,
        rsi_divergence=bool(b['price']<a['price'] and b['rsi6']>a['rsi6']) if sufficient else None,
        low_volume_ratio=b['volume']/a['volume'] if a['volume'] and b['volume'] is not None else None)


def compare(minutes,code,previous_day,day,cutoff):
    from golden_ladder_v3 import completed_bars
    five,history,_,status=completed_bars(minutes,code,day,cutoff)
    if not five or day*1000000+five[-1]['t']!=cutoff:
        return dict(status='DATA_INSUFFICIENT',reason=status,periods={})
    days=sorted({b['day'] for b in history})
    if any(sum(b['day']==d for b in history)!=48 for d in days):
        return dict(status='DATA_INSUFFICIENT',reason='HISTORY_GAP',periods={})
    all_bars=history+five
    return dict(status='AVAILABLE',asof=cutoff,last_close=five[-1]['close'],last_time=day*1000000+five[-1]['t'],
        periods={'5':pair(all_bars,previous_day,day,5),
        '15':pair(aggregate15(all_bars),previous_day,day,15)},
        note='今日低点指已完成对应周期K线中的最低点首现根；未完成低点不前借RSI/DIF')
