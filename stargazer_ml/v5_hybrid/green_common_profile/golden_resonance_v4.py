"""P-frozen historical bands and causal cross-session repair facts; never orders."""
import math
import numpy as np
from golden_ladder_v3 import prior_confirmed_waves, wave_levels, price_map, completed_bars, _rsi, STOPS

POLICY='GOLDEN_RESONANCE_CROSSDAY_V4'


def higher(a,b):
    return bool(np.isfinite(a) and np.isfinite(b) and a>b and
                not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12))


def build_map(g,atr):
    g=g.sort_values('date').reset_index(drop=True)
    if not math.isfinite(atr) or atr<=0:raise ValueError('ATR_NOT_AVAILABLE')
    waves=prior_confirmed_waves(g);close=float(g.close.iloc[-1]);nodes={}
    for wave in waves:
        for point in wave_levels(wave['low'],wave['high']):
            if point['kind']=='REBOUND':continue
            if point['kind']=='DOWNSIDE_EXTENSION_FROM_HIGH' and close>=wave['low']:continue
            node=round(point['price'],4)
            if 0<node<=close:nodes.setdefault(node,[]).append(dict(**wave,ratio=point['ratio'],kind=point['kind'],level=node))
    if not nodes:raise ValueError('NO_CAUSAL_WAVE_MAP')
    clusters=[]
    for node in sorted(nodes):
        if not clusters or node/clusters[-1][0]-1>=.008:clusters.append([node])
        else:clusters[-1].append(node)
    bands=[]
    for values in clusters:
        sources=[s for node in values for s in nodes[node]]
        count=len({(s['low_day'],s['high_day']) for s in sources})
        upper=min(close*.96,values[-1]+.2*atr)
        bands.append(dict(node=values[0],lower=values[0],upper=values[-1],sources=sources,
            source_waves=count,shared_peak_groups=len({s['high_day'] for s in sources}),
            zone_upper=upper,observable=count>=3 and values[0]<=upper))
    bands.sort(key=lambda b:-b['node'])
    try:temporary=price_map(g,atr)['temporary_waves']
    except ValueError:temporary=[]
    return dict(policy=POLICY,asof=int(g.date.iloc[-1]),bands=bands,
        levels=[b['node'] for b in bands if b['observable']],waves=waves,
        primary_wave=waves[-1],temporary_waves=temporary,cluster_max_span_fraction=.008,
        map_strategy='HISTORICAL_CONFIRMED_MULTI_BAND_V4',
        history_days=g.date.astype(int).tolist()[-4:],
        source_count_note='端点对数量；共享高低点相关，不是独立票数',
        model_score_reference_unchanged=True)


def position(prefix,band,low_index):
    if len(prefix)<2:return None
    low=prefix[low_index]['low'];before,after=prefix[-2:]
    if after['close']<band['lower']:return None
    if low<band['lower']:
        return 'RECLAIM' if all(b['close']>=band['upper'] for b in (before,after)) else None
    if len(prefix)-1<low_index+2 or low>band['zone_upper']:return None
    touch=prefix[low_index]
    prices=before['low']>=low and after['low']>=before['low'] and after['close']>before['close']>=touch['close']
    volumes=[b.get('volume') for b in (touch,before,after)]
    known=all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in volumes)
    return 'STOP_ABOVE' if prices and known and sum(volumes[1:])/2<=volumes[0] else None


def evaluate_bars(row,day,cutoff,five,history,closes=None):
    rule=row['entry_rule'];bands=rule.get('observation_bands') or []
    levels=[b['lower'] for b in bands]
    if (rule.get('levels')!=sorted(set(levels),reverse=True) or rule.get('p_date',day)>=day or
            any(not (0<b['lower']<=b['upper']<=row['reference_price'] and
                b['lower']<=b['zone_upper']<=min(row['reference_price']*.96,b['upper']+.2*rule['atr_ref'])+1e-10 and
                b['source_waves']>=3) for b in bands)):
        raise ValueError('INVALID_V4_FROZEN_BANDS')
    result=dict(policy=POLICY,status='WAIT_SIGNAL',reason='WAIT_CROSSDAY_TWO_CROSSES_AND_REPAIR',
        price_asof=cutoff,trigger_time=None,scope='试运行价格组合；同截止风险及AI综合后人工决定，不下单',
        node=None,branch=None,divergence=None)
    if not five or day*1000000+five[-1]['t']!=cutoff:
        return dict(result,status='DATA_INSUFFICIENT',reason='CURRENT_COMPLETE_5M_PREFIX_MISSING')
    result.update(last_two_closes=[b['close'] for b in five[-2:]],session_low=min(b['low'] for b in five),
        next_lower_node=next((b['lower'] for b in bands if b['lower']<=five[-1]['close']),None),
        pierced_levels=[b['lower'] for b in bands if min(x['low'] for x in five)<b['lower']])
    if not bands:return dict(result,status='WAIT_POSITION',reason='NO_QUALIFIED_LOW_RESONANCE_BAND')
    if len(history)<48:return dict(result,status='DATA_INSUFFICIENT',reason='HISTORY_SEED_MISSING')
    previous_day=rule['p_date'];previous=[b for b in history if b['day']==previous_day]
    if len(previous)!=48 or [b['t'] for b in previous]!=STOPS or history[-1]['day']!=previous_day:
        return dict(result,status='DATA_INSUFFICIENT',reason='PREVIOUS_SESSION_INCOMPLETE')
    closes=closes if closes is not None else [b['close'] for b in history+five]
    fast,slow=_rsi(closes,6),_rsi(closes,15);six=fast[-len(five):];fifteen=slow[-len(five):]
    anchor_i=min(range(48),key=lambda j:previous[j]['low'])
    anchor_rsi=fast[-len(five)-48+anchor_i]
    if not np.isfinite(anchor_rsi) or not np.isfinite(six[-1]) or not np.isfinite(fifteen[-1]):
        return dict(result,status='DATA_INSUFFICIENT',reason='CROSSDAY_RSI_SEED_UNKNOWN')
    anchor=dict(time=previous_day*1000000+previous[anchor_i]['t'],low=previous[anchor_i]['low'],rsi6=float(anchor_rsi))
    crosses=[i for i in range(len(five)) if len(closes)-len(five)+i>0 and
        fast[len(closes)-len(five)+i-1]<=slow[len(closes)-len(five)+i-1] and higher(six[i],fifteen[i])]
    result['rsi']=dict(rsi6=float(six[-1]),rsi15=float(fifteen[-1]),above=higher(six[-1],fifteen[-1]),
        cross_up_times=[five[i]['t'] for i in crosses],basis='KNOWN_HISTORY_PREFIX')
    result['cross_day_anchor']=anchor
    touches={};low_index=0;active=None
    for i,bar in enumerate(five):
        if bar['low']<five[low_index]['low']:low_index=i;active=None
        low=five[low_index]['low']
        if active and (not higher(six[i],fifteen[i]) or
                (active['branch']=='RECLAIM' and bar['close']<active['band']['upper'])):active=None
        choices=[]
        for n,band in enumerate(bands):
            if n not in touches and bar['low']<=band['zone_upper'] and bar['high']>=band['lower']:touches[n]=i
            if n not in touches:continue
            seen=[j for j in crosses if max(1,touches[n],low_index)<=j<=i]
            if len(seen)<2 or not higher(six[i],fifteen[i]):continue
            if not (low<anchor['low'] and higher(six[low_index],anchor_rsi)):continue
            branch=position(five[:i+1],band,low_index)
            if branch:
                choices.append((max(band['lower']-low,low-band['upper'],0),-band['source_waves'],-band['lower'],n,branch,seen))
        if active is not None or not choices:continue
        _,_,_,n,branch,seen=min(choices);band=bands[n]
        confirmation=[five[low_index],five[i-1],bar] if branch=='STOP_ABOVE' else five[i-1:i+1]
        active=dict(node=band['lower'],band={k:band[k] for k in ('lower','upper','zone_upper','source_waves','shared_peak_groups')},
            branch=branch,trigger_time=day*1000000+bar['t'],invalidation_low=low,
            confirmation_bars=confirmation,touch_time=day*1000000+five[touches[n]]['t'],
            event_cross_times=[day*1000000+five[j]['t'] for j in seen],
            cross_day_anchor=anchor,divergence=dict(basis='CROSS_SESSION_RSI6',anchor_time=anchor['time'],
                anchor_low=anchor['low'],anchor_rsi=anchor['rsi6'],event_time=day*1000000+five[low_index]['t'],
                event_low=low,event_rsi=float(six[low_index])))
    if active:result.update(active,status='RULE_TRIGGERED',reason='CROSSDAY_TWO_CROSSES_POSITION_REPAIR_CONFIRMED')
    return result


def evaluate(row,day,cutoff,rows):
    five,history,closes,status=completed_bars(rows,row['code'],day,cutoff)
    # Previous-session bars retain their identity; no future rows are admitted.
    for bar in history:
        if 'day' not in bar:raise ValueError('HISTORY_DAY_IDENTITY_MISSING')
    return evaluate_bars(row,day,cutoff,five,history,closes)
