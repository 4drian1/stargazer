"""P-frozen historical bands and causal cross-session repair facts; never orders."""
import math
import numpy as np
from golden_watch_groups import downside_limit
from golden_ladder_v3 import prior_confirmed_waves, wave_levels, price_map, completed_bars, _rsi, STOPS

POLICY='GOLDEN_RESONANCE_CROSSDAY_V4'
COMBINATION_POLICY='POSITION_TWO_OF_THREE_O20_CROSS1_V1'


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
        upper=min(downside_limit(close),values[-1]+.2*atr)
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


def evaluate_bars(row,day,cutoff,five,history,closes=None,*,position_fn=None,observed_oversold=None):
    from plan_rule_profiles import validate, policy
    profile=validate(row['entry_rule'].get('execution_profile'))
    required=profile['required_votes'];cross_required=profile['momentum_crosses_required'];ratio=profile['oversold_ratio']
    rule=row['entry_rule'];bands=rule.get('observation_bands') or []
    levels=[b['lower'] for b in bands]
    if (rule.get('levels')!=sorted(set(levels),reverse=True) or rule.get('p_date',day)>=day or
            any(not (0<b['lower']<=b['upper']<=row['reference_price'] and
                b['lower']<=b['zone_upper']<=min(downside_limit(row['reference_price']),b['upper']+.2*rule['atr_ref'])+1e-10 and
                b['source_waves']>=3) for b in bands)):
        raise ValueError('INVALID_V4_FROZEN_BANDS')
    result=dict(policy=POLICY,combination_policy=policy(profile),status='WAIT_SIGNAL',reason='WAIT_POSITION_AND_TWO_OF_THREE',
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
    if observed_oversold is not None:
        e=observed_oversold
        if (e.get('anchor_rsi6')!=anchor_rsi or e.get('ratio')!=ratio or
            not day*1000000<=e.get('time',0)<=cutoff or
            not isinstance(e.get('rsi6'),(int,float)) or not 0<=e['rsi6']<anchor_rsi*ratio):
            raise ValueError('OBSERVED_OVERSOLD_REFERENCE_MISMATCH')
    result['cross_day_anchor']=anchor
    touches={};low_index=0;active=None;oversold=observed_oversold;last_choices=[]
    for i,bar in enumerate(five):
        if bar['low']<five[low_index]['low']:low_index=i;active=None
        low=five[low_index]['low']
        if oversold is None and np.isfinite(six[i]) and six[i]<anchor_rsi*ratio:
            oversold=dict(time=day*1000000+bar['t'],rsi6=float(six[i]),anchor_rsi6=float(anchor_rsi),ratio=ratio)
        choices=[]
        for n,band in enumerate(bands):
            if n not in touches and bar['low']<=band['zone_upper'] and bar['high']>=band['lower']:touches[n]=i
            if n not in touches:continue
            seen=[j for j in crosses if max(1,touches[n],low_index)<=j<=i]
            branch=(position_fn or position)(five[:i+1],band,low_index)
            momentum=(bool(low<anchor['low'] and higher(six[low_index],anchor_rsi) and
                higher(six[i],fifteen[i]) and len(seen)>=cross_required) if np.isfinite(six[low_index]) else None)
            votes=dict(repair=bool(branch),momentum=momentum,oversold=oversold is not None)
            count=sum(v is True for v in votes.values())
            rank=(max(band['lower']-low,low-band['upper'],0),-band['source_waves'],-band['lower'])
            choices.append(dict(rank=rank,n=n,branch=branch or 'MOMENTUM_OVERSOLD',seen=seen,votes=votes,count=count))
        last_choices=choices
        qualified=[c for c in choices if c['count']>=required]
        same=next((c for c in qualified if active and bands[c['n']]['lower']==active['node'] and
            c['branch']==active['branch'] and c['votes']==active['combination']['votes']),None)
        if same is not None:continue
        active=None
        if not qualified:continue
        chosen=min(qualified,key=lambda c:c['rank']);n=chosen['n'];branch=chosen['branch'];seen=chosen['seen'];band=bands[n]
        confirmation=[five[low_index],five[i-1],bar] if branch=='STOP_ABOVE' else five[max(0,i-1):i+1]
        active=dict(node=band['lower'],band={k:band[k] for k in ('lower','upper','zone_upper','source_waves','shared_peak_groups')},
            branch=branch,trigger_time=day*1000000+bar['t'],invalidation_low=low,
            combination=dict(votes=chosen['votes'],count=chosen['count'],required=required,momentum_crosses_required=cross_required,oversold_ratio=ratio),
            oversold_evidence=oversold,
            confirmation_bars=confirmation,touch_time=day*1000000+five[touches[n]]['t'],
            event_cross_times=[day*1000000+five[j]['t'] for j in seen],
            cross_day_anchor=anchor,divergence=dict(basis='CROSS_SESSION_RSI6',anchor_time=anchor['time'],
                anchor_low=anchor['low'],anchor_rsi=anchor['rsi6'],event_time=day*1000000+five[low_index]['t'],
                event_low=low,event_rsi=float(six[low_index]) if np.isfinite(six[low_index]) else None,
                confirmed=bool(low<anchor['low'] and higher(six[low_index],anchor_rsi))))
    if active:result.update(active,status='RULE_TRIGGERED',reason='POSITION_AND_TWO_OF_THREE_CONFIRMED')
    pending=[]
    if not active:
        if not last_choices:pending.append('三重以上买入位置尚未到达')
        else:
            chosen=min(last_choices,key=lambda c:(-c['count'],c['rank']))
            result['combination']=dict(votes=chosen['votes'],count=chosen['count'],required=required,momentum_crosses_required=cross_required,oversold_ratio=ratio)
            names={'repair':'位置修复','momentum':'动能（跨日背离＋'+str(cross_required)+'次上穿＋RSI快线在上）','oversold':str(ratio*100)+'%严重超卖'}
            lacking=[names[k]+('证据未知' if v is None else '未满足') for k,v in chosen['votes'].items() if v is not True]
            pending=['三选'+str(required)+'未通过（'+str(chosen['count'])+'/'+str(required)+'）','还需'+str(required-chosen['count'])+'项：'+'；'.join(lacking)]
    result['oversold_evidence']=oversold
    result['pending_conditions']=pending
    return result


def evaluate(row,day,cutoff,rows):
    five,history,closes,status=completed_bars(rows,row['code'],day,cutoff)
    # Previous-session bars retain their identity; no future rows are admitted.
    for bar in history:
        if 'day' not in bar:raise ValueError('HISTORY_DAY_IDENTITY_MISSING')
    return evaluate_bars(row,day,cutoff,five,history,closes)
