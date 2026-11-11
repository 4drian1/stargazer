"""Causal price maps and completed-bar observations; never orders or model scores."""
import math
import numpy as np
import pandas as pd

POLICY='GOLDEN_LADDER_STOP_V3'
RATIOS=(.236,.382,.5,.618,.786)
EXTENSIONS=(1.,1.272,1.618,2.,2.618)
STOPS=[h*10000+m*100 for h in (9,10,11,13,14,15) for m in range(0,60,5)
       if 93500<=h*10000+m*100<=113000 or 130500<=h*10000+m*100<=150000]


def wave_levels(low,high):
    if not 0<low<high:raise ValueError('INVALID_WAVE_ANCHORS')
    width=high-low
    return [dict(kind=kind,ratio=ratio,price=round(price,6)) for kind,values in (
        ('RETRACEMENT',[(r,high-r*width) for r in RATIOS]),
        ('REBOUND',[(r,low+r*width) for r in RATIOS]),
        ('DOWNSIDE_EXTENSION_FROM_HIGH',[(k,high-k*width) for k in EXTENSIONS]))
        for ratio,price in values if price>0]


def prior_confirmed_waves(g):
    """Recover confirmed historical swings when a new unconfirmed high hides them."""
    waves=[]
    for peak in range(max(2,len(g)-60),len(g)-2):
        hi=float(g.high.iloc[peak])
        if hi<g.high.iloc[peak-2:peak+3].max():continue
        for low in range(max(2,peak-60),peak):
            lo=float(g.low.iloc[low])
            if lo>g.low.iloc[low-2:low+3].min() or hi/lo-1<=.15:continue
            segment=g.iloc[low:peak+1]
            if hi<segment.high.max() or lo>segment.low.min():continue
            waves.append(dict(low=lo,high=hi,low_day=int(g.date.iloc[low]),high_day=int(g.date.iloc[peak]),
                low_basis='CONFIRMED_SWING_LOW',low_confirmed=True,high_confirmed=True,
                known_at=int(g.date.iloc[max(peak+2,low+2)]),
                source_state='PRIOR_CONFIRMED_HISTORICAL_REFERENCE',
                anchor_low_breached_later=bool((g.low.iloc[peak+1:]<lo).any())))
    return waves


def price_map(daily,atr_ref=None):
    """Recent confirmed lows and completed >15% bullish bodies; no future bars."""
    g=daily.sort_values('date').reset_index(drop=True)
    recent_peak=int(g.high.iloc[max(0,len(g)-60):].idxmax())
    waves=[];start=max(2,recent_peak-60)
    for i in range(start,len(g)):
        confirmed=i+2<len(g) and g.low.iloc[i]<=g.low.iloc[i-2:i+3].min()
        impulse=g.close.iloc[i]/g.open.iloc[i]-1>.15
        if not (confirmed or impulse):continue
        peak=int(g.high.iloc[i:].idxmax());lo=float(g.low.iloc[i]);hi=float(g.high.iloc[peak])
        if hi/lo-1<=.15:continue
        high_confirmed=bool(peak+2<len(g) and g.high.iloc[peak]>=g.high.iloc[max(0,peak-2):peak+3].max())
        waves.append(dict(low=lo,high=hi,low_day=int(g.date.iloc[i]),high_day=int(g.date.iloc[peak]),
            low_basis='CONFIRMED_SWING_LOW' if confirmed else 'COMPLETED_IMPULSE_LOW',
            low_confirmed=bool(confirmed),
            high_confirmed=high_confirmed,
            known_at=int(g.date.iloc[max(peak+2 if high_confirmed else peak,i+2 if confirmed else i)])))
    # Repeated endpoints are one source. Shared highs remain explicitly correlated.
    waves=list({(w['low_day'],w['high_day']):w for w in waves}.values())
    fallback=not any(w['low_confirmed'] and w['high_confirmed'] for w in waves)
    if fallback:
        waves=prior_confirmed_waves(g)+waves
    nodes={}
    close=float(g.close.iloc[-1])
    for wave in waves:
        if not wave['low_confirmed'] or not wave['high_confirmed']:continue
        for item in wave_levels(wave['low'],wave['high']):
            if item['kind']=='REBOUND':continue
            if item['kind']=='DOWNSIDE_EXTENSION_FROM_HIGH' and close>=wave['low']:continue
            node=round(item['price'],4)
            if 0<node<=close:
                nodes.setdefault(node,[]).append(dict(**wave,ratio=item['ratio'],kind=item['kind']))
    if not nodes:raise ValueError('NO_CAUSAL_WAVE_MAP')
    # Bounded-span clustering, not a chain that can bridge distant prices.
    clusters=[]
    for node in sorted(nodes):
        if not clusters or node/clusters[-1][0]-1>=.008:clusters.append([node])
        else:clusters[-1].append(node)
    bands=[]
    for values in clusters:
        representative=min(values,key=lambda x:abs(x-float(np.median(values))))
        sources=[dict(source,level=node) for node in values for source in nodes[node]]
        bands.append(dict(node=representative,lower=min(values),upper=max(values),sources=sources,
            shared_peak_groups=len({s['high_day'] for s in sources})))
    result=dict(policy=POLICY,asof=int(g.date.iloc[-1]),levels=sorted([b['node'] for b in bands],reverse=True),
        sources={str(b['node']):b['sources'] for b in bands},waves=waves,bands=bands,cluster_max_span_fraction=.008,
        primary_wave=next(w for w in reversed(waves) if w['low_confirmed'] and w['high_confirmed']),
        temporary_waves=[w for w in waves if not(w['low_confirmed'] and w['high_confirmed'])],
        resonance_status='SOURCES_DISCLOSED_NOT_INDEPENDENT_VOTES',model_score_reference_unchanged=True)
    if fallback:
        result.update(map_strategy='PRIOR_CONFIRMED_FALLBACK',
            fallback_reason='Latest unconfirmed high must not hide older confirmed swings; current temporary high stays separate')
    return result


def index_map(frame,cutoff):
    """Separate execution map; frozen Pyxis matrices remain untouched."""
    f=frame[frame.date<=cutoff].sort_values('date').copy()
    f['day']=f.date//1000000
    daily=f.groupby('day').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),count=('close','count')).reset_index()
    daily=daily[daily['count']==48].reset_index(drop=True)
    # Current session is included only at its completed 15:00 close.
    limit=cutoff//1000000 if cutoff%1000000==150000 else cutoff//1000000-1
    daily=daily[daily.day<=limit].reset_index(drop=True)
    peaks=[i for i in range(max(2,len(daily)-60),len(daily)-2) if daily.high.iloc[i]>=daily.high.iloc[i-2:i+3].max()]
    if not peaks:return dict(status='UNKNOWN_NO_CONFIRMED_PEAK')
    peak=max(peaks,key=lambda i:daily.high.iloc[i])
    lows=[i for i in range(max(2,peak-60),peak) if daily.low.iloc[i]<=daily.low.iloc[i-2:i+3].min()]
    if not lows:return dict(status='UNKNOWN_NO_CONFIRMED_LOW')
    low=min(lows,key=lambda i:daily.low.iloc[i]);lo=float(daily.low.iloc[low]);hi=float(daily.high.iloc[peak])
    levels=wave_levels(lo,hi)
    today=f[f.day==cutoff//1000000]
    observations=[]
    for item in levels:
        back=today[(today.low<item['price'])&(today.close>=item['price'])]
        observations.append(dict(**item,below_then_close_back_times=back.date.astype(int).tolist(),
            last_close_above=bool(f.close.iloc[-1]>=item['price'])))
    return dict(status='AVAILABLE_EXECUTION_REFERENCE_NOT_PYXIS_FEATURE',asof=cutoff,
        current_close=float(f.close.iloc[-1]),
        low=lo,high=hi,low_day=int(daily.day.iloc[low]),high_day=int(daily.day.iloc[peak]),
        anchors_known_at=int(daily.day.iloc[max(peak+2,low+2)]),
        rule='HIGHEST_CONFIRMED_PEAK_RECENT_60_PRIOR_CONFIRMED_LOW_60',levels=observations)


def completed_bars(rows,code,day,cutoff):
    f=pd.DataFrame(rows)
    required={'date','code','open','high','low','close'}
    if f.empty or not required<=set(f):return [],[],[], 'MINUTES_MISSING'
    f=f[f.date<=cutoff].sort_values('date')
    if f.date.duplicated().any() or not f.code.astype(str).str.zfill(6).eq(code).all():
        return [],[],[],'IDENTITY_OR_DUPLICATE_INVALID'
    p=f[['open','high','low','close']].to_numpy(dtype=float)
    if (not np.isfinite(p).all() or (p<=0).any() or
            (f.low>f[['open','close']].min(axis=1)).any() or (f.high<f[['open','close']].max(axis=1)).any()):
        return [],[],[],'INVALID_MINUTE_PRICE'
    current=[];history=[];history_closes=[]
    for date,group in f.groupby(f.date//1000000):
        native=group.get('source_frequency_minutes')
        if native is not None and native.eq(5).any():
            if not native.eq(5).all():return [],[],[],'MIXED_SOURCE_FREQUENCY'
            clocks=group.date.astype('int64')%1000000
            expected=list(STOPS) if date!=day else [t for t in STOPS if date*1000000+t<=cutoff]
            if clocks.tolist()!=expected:return [],[],[],'NATIVE_5M_DAY_INCOMPLETE'
            for r in group.to_dict('records'):
                b=dict(t=int(r['date'])%1000000,day=int(date),open=float(r['open']),
                       high=float(r['high']),low=float(r['low']),close=float(r['close']),
                       volume=float(r['volume']) if pd.notna(r.get('volume')) else None)
                history_closes.append(b['close'])
                (current if date==day else history).append(b)
            continue
        lookup={int(r['date'])%1000000:r for r in group.to_dict('records')}
        allowed={93000,130000}
        for stop in STOPS:
            end=stop//10000*60+stop%10000//100
            allowed.update(m//60*10000+m%60*100 for m in range(end-4,end+1))
        if set(lookup)-allowed:return [],[],[],'OFF_SESSION_SOURCE'
        for stop in STOPS:
            if date==day and date*1000000+stop>cutoff:break
            end=stop//10000*60+stop%10000//100
            need=[m//60*10000+m%60*100 for m in range(end-4,end+1)]
            if not all(t in lookup for t in need):
                if date==day:return current,history,history_closes,'PREFIX_STOPS_AT_MISSING_MINUTE'
                history_closes.append(np.nan);continue
            values=[lookup[t] for t in need]
            # Opening auction observations are optional and included only when present.
            opening=93000 if stop==93500 else 130000 if stop==130500 else None
            if opening in lookup:values=[lookup[opening]]+values
            volume=[r.get('volume') for r in values]
            valid_volume=all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in volume)
            b=dict(t=stop,day=int(date),open=float(values[0]['open']),high=float(max(r['high'] for r in values)),
                low=float(min(r['low'] for r in values)),close=float(values[-1]['close']),
                volume=float(sum(volume)) if valid_volume else None)
            history_closes.append(b['close'])
            (current if date==day else history).append(b)
    return current,history,history_closes,'COMPLETE_PREFIX_OPTIONAL_OPEN'


def _rsi(values,n):
    x=np.asarray(values,dtype=float);out=np.full(len(x),np.nan)
    edges=np.diff(np.r_[False,np.isfinite(x),False].astype(int))
    for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        if end-start<=n:continue
        d=np.diff(x[start:end]);u=np.maximum(d,0);v=np.maximum(-d,0)
        u=pd.Series(np.r_[u[:n].mean(),u[n:]]).ewm(alpha=1/n,adjust=False).mean().to_numpy()
        v=pd.Series(np.r_[v[:n].mean(),v[n:]]).ewm(alpha=1/n,adjust=False).mean().to_numpy()
        out[start+n:end]=np.divide(100*u,u+v,out=np.full_like(u,50),where=u+v>0)
    return out


def evaluate(row,day,cutoff,rows):
    rule=row['entry_rule'];levels=rule['levels'];atr=rule['atr_ref']
    if levels!=sorted(set(levels),reverse=True) or not atr>0 or any(not 0<x<=row['reference_price'] for x in levels):
        raise ValueError('INVALID_FROZEN_LEVELS')
    five,history,closes,status=completed_bars(rows,row['code'],day,cutoff)
    if not five or day*1000000+five[-1]['t']!=cutoff:
        return dict(status='DATA_INSUFFICIENT',reason=status)
    crosses=[];rsi6=rsi15=None
    if len(history)>=48:
        a,b=_rsi(closes,6),_rsi(closes,15);offset=len(closes)-len(five)
        crosses=[five[i]['t'] for i in range(len(five)) if offset+i>0 and
            a[offset+i-1]<=b[offset+i-1] and a[offset+i]>b[offset+i]]
        rsi6=float(a[-1]) if np.isfinite(a[-1]) else None
        rsi15=float(b[-1]) if np.isfinite(b[-1]) else None
    events=[];low_index=0
    for t,bar in enumerate(five):
        if bar['low']<five[low_index]['low']:low_index=t
        low=five[low_index]['low']
        # An intervening new low invalidates earlier stabilization or reclaim evidence.
        events=[e for e in events if bar['low']>=e['invalidation_low'] and
                (e['branch']!='RECLAIM' or bar['close']>=e['node'])]
        if t>=1:
            for node in levels:
                if (any(x['low']<node for x in five[:t]) and
                        five[t-1]['close']>=node and bar['close']>=node and
                        not any(e['branch']=='RECLAIM' and e['node']==node for e in events)):
                    events.append(dict(node=node,branch='RECLAIM',event_index=t,
                        invalidation_low=low,confirmation_bars=[five[t-1],bar]))
        node=next((n for n in levels if n<=low),None)
        if node is None or low-node>.2*atr or t<low_index+2:continue
        before,after=five[t-1],bar;touch=five[low_index]
        prices=before['low']>=low and after['low']>=before['low'] and after['close']>before['close']>=touch['close']
        vols=[touch['volume'],before['volume'],after['volume']]
        contracting=all(v is not None for v in vols) and (vols[1]+vols[2])/2<=vols[0]
        if prices and contracting:
            events.append(dict(node=node,branch='STOP_ABOVE',event_index=t,
                invalidation_low=low,confirmation_bars=[touch,before,after]))
    best=max(events,key=lambda e:(e['event_index'],-abs(e['node']-five[-1]['close']))) if events else {}
    result=dict(status='RULE_TRIGGERED' if best else 'WAIT_SIGNAL',reason='PRICE_FACTS_REQUIRE_SAME_CUTOFF_RISK',
        policy=POLICY,price_asof=cutoff,trigger_time=day*1000000+five[best['event_index']]['t'] if best else None,
        last_two_closes=[b['close'] for b in five[-2:]],session_low=min(b['low'] for b in five),
        next_lower_node=next((n for n in levels if n<=five[-1]['close']),None),
        pierced_levels=[n for n in levels if min(b['low'] for b in five)<n],
        rsi=dict(periods=[6,15],rsi6=rsi6,rsi15=rsi15,cross_up_times=crosses,
            above=rsi6>rsi15 if rsi6 is not None and rsi15 is not None else None,
            basis='HISTORY_PREFIX' if len(history)>=48 else 'UNKNOWN_NO_HISTORY_SEED'),
        divergence=None,scope='逐档价格事实；非自动买入，须同截止风险和人工判断')
    if best:result.update({k:v for k,v in best.items() if k!='event_index'})
    if best:
        from tarot_live_signal_v2 import adapter
        legacy=adapter().d.prior.kernel.replay(five,[0]*len(five),best['node'],atr)
        divergence=legacy.get('divergence')
        if divergence and all(b['low']>=divergence['event_low'] for b in five[divergence['event_bar']+1:]):
            result['divergence']=divergence
        result['divergence_basis']='LEGACY_RSI14_INTRADAY_SEED_NOT_RSI6_15_CROSS'
    return result
