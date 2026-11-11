"""Actual observed partial bars, distinct from completed-bar confirmations."""
from copy import deepcopy
import math
CONTRACT='LIVE_OBSERVED_TWO_OF_THREE_V1'

def bars(minutes,code,P,asof,quote=None):
    from golden_ladder_v3 import completed_bars
    day=asof//1000000
    past=[r for r in minutes if int(r['date'])//1000000<day]
    previous,older,_,status=completed_bars(past,code,P,P*1000000+150000)
    history=older+previous
    if len(previous)!=48:raise ValueError('LIVE_PREVIOUS_DAY_INCOMPLETE')
    current=sorted((r for r in minutes if int(r['date'])//1000000==day and int(r['date'])<=asof),key=lambda r:int(r['date']))
    groups={};latest=0
    for r in current:
        stamp=int(r['date']);clock=stamp%1000000;minute=clock//10000*60+clock%10000//100
        if minute in (570,780):continue
        if not (570<minute<=690 or 780<minute<=900):continue
        end=(minute+4)//5*5
        groups.setdefault(end,[]).append(r);latest=max(latest,stamp)
    quote=quote or {}
    qtime=int(quote.get('time') or 0);price=quote.get('price')
    if (type(price) in (int,float) and math.isfinite(price) and price>0 and
        qtime//1000000==day and latest<=qtime<=asof):
        clock=qtime%1000000;minute=clock//10000*60+clock%10000//100
        end=(minute//5+1)*5 if clock%100 else (minute+4)//5*5
        if minute==570:end=575
        if 575<=end<=690 or 785<=end<=900:
            observation=dict(date=qtime,code=code,open=price,high=price,low=price,close=price,volume=None,
                             quote_observation=True)
            if end==575 and not groups:
                observation.update(open=quote.get('open',price),high=quote.get('high',price),low=quote.get('low',price))
            groups.setdefault(end,[]).append(observation)
    five=[]
    for end,values in sorted(groups.items()):
        values=sorted(values,key=lambda r:int(r['date']))
        nominal=day*1000000+end//60*10000+end%60*100
        complete=nominal<latest and not any(v.get('quote_observation') for v in values)
        # The label of a forming bar is the evidence time, never a future close time.
        t=nominal%1000000 if complete else min(asof,nominal)%1000000
        vols=[v.get('volume') for v in values if not v.get('quote_observation')]
        volume=sum(vols) if vols and all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in vols) else None
        five.append(dict(day=day,t=t,nominal_end=nominal,complete=complete,
            open=float(values[0]['open']),high=max(float(v['high']) for v in values),low=min(float(v['low']) for v in values),
            close=float(values[-1]['close']),volume=volume,volume_complete=complete))
    if not five:raise ValueError('LIVE_NO_PRICE_OBSERVATION')
    # Within one five-minute window, later observations replace that one bar, not add crossings.
    five[-1]['t']=asof%1000000
    five[-1]['complete']=False
    return five,history

def position_live(prefix,band,low_index):
    from golden_resonance_v4 import position
    complete=[b for b in prefix if b.get('complete',True)]
    if not complete or low_index>=len(complete):return None
    latest=prefix[-1]
    if latest['low']<min(b['low'] for b in complete) or latest['close']<band['lower']:return None
    branch=position(complete,band,low_index)
    if branch=='RECLAIM' and latest['close']<band['upper']:return None
    return branch

def signal(row,P,asof,minutes,quote=None,observed_oversold=None):
    from golden_resonance_v4 import evaluate_bars
    five,history=bars(minutes,row['code'],P,asof,quote)
    result=evaluate_bars(row,asof//1000000,asof,five,history,position_fn=position_live,observed_oversold=observed_oversold)
    result.update(evaluation_basis=CONTRACT,provisional=True,observed_asof=asof,
        current_bar_end=five[-1]['nominal_end'],current_bar_complete=False,
        source_note='实时试算、未收盘；后续数值可变，真实收盘资格另列')
    if result.get('status')=='RULE_TRIGGERED':result['trigger_time']=asof
    if result.get('combination',{}).get('votes',{}).get('repair'):
        completed=[b for b in five if b.get('complete')]
        low_i=min(range(len(completed)),key=lambda i:completed[i]['low'])
        result['confirmation_bars']=([completed[low_i],*completed[-2:]] if result['branch']=='STOP_ABOVE' else completed[-2:])
    return result,five,history

def fifteen(observed):
    groups={}
    for b in observed:
        clock=b.get('nominal_end',b['day']*1000000+b['t'])%1000000
        minute=clock//10000*60+clock%10000//100;end=(minute+14)//15*15
        groups.setdefault((b['day'],end),[]).append(b)
    result=[]
    for (day,end),group in sorted(groups.items()):
        complete=len(group)==3 and all(x.get('complete',True) for x in group)
        nominal=day*1000000+end//60*10000+end%60*100
        vols=[b.get('volume') for b in group]
        result.append(dict(day=day,t=nominal%1000000 if complete else group[-1]['t'],nominal_end=nominal,complete=complete,
            open=group[0]['open'],high=max(x['high'] for x in group),low=min(x['low'] for x in group),close=group[-1]['close'],
            volume=sum(vols) if all(v is not None for v in vols) else None))
    return result

def comparisons(minutes,code,P,asof,quote=None):
    from strong_low_indicators_v1 import pair
    five,history=bars(minutes,code,P,asof,quote)
    periods={}
    for name,values in [('5',history+five),('15',fifteen(history+five))]:
        p=pair(values,P,asof//1000000,int(name))
        p.update(provisional=True,evidence_time=asof,label='实时试算、未收盘',basis='OBSERVED_PARTIAL_END_BAR; RSI_WILDER; DIF_EMA12_26_ADJUST_FALSE')
        if p.get('current'):
            current=[b for b in values if b['day']==asof//1000000]
            low=min(current,key=lambda b:b['low'])
            p['current'].update(complete=low.get('complete',True),nominal_end=low.get('nominal_end'))
        periods[name]=p
    return dict(status='AVAILABLE',asof=asof,last_time=asof,last_close=five[-1]['close'],periods=periods,
                provisional=True,source_contract=CONTRACT)
