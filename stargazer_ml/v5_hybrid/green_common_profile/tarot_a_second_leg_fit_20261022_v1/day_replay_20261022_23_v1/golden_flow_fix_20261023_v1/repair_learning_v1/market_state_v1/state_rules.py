"""Current-state reference rules; not a trained or forward-return classifier."""
import math
from numbers import Real

LABELS=dict(STRONG_REPAIR='强修复',WEAK_REPAIR='弱修复',BREAK_UNREPAIRED='破位未修复',
    DECLINE_ENHANCING='下跌增强',OBSERVE='观察，方向证据不足',UNKNOWN='未知')


def known(value):
    return isinstance(value,Real) and not isinstance(value,bool) and math.isfinite(value)


def describe(index,cutoff,features):
    def get(group,name):
        value=features.get(index+'_'+group+'_'+name)
        return float(value) if known(value) else None
    p=lambda name:get('repair',name)
    b=lambda name:get('break',name)
    observed=dict(map_known=b('map_known'),reclaim=b('two_close_reclaim'),stop=b('stop_above'),
        fast=p('rsi6'),slow=p('rsi15'),volume_ratio=p('volume_after_low_ratio'),
        lows_rising=p('last_two_lows_rising'),closes_rising=p('last_two_closes_rising'),
        rebound=p('low_rebound_return'),bars_since_low=p('bars_since_low'),
        return_3bars=p('return_3bars'),broken_run=b('broken_close_run'),
        broken_no_repair=b('broken_no_repair'),crosses=p('cross_up_count'),
        source_waves=b('broken_nearest_wave_count'),source_peak_groups=b('broken_nearest_peak_groups'))
    facts={}
    def fact(key,text):facts[index+'.'+key]=dict(asof=cutoff,text=text)
    fact('map','黄金地图可用：'+str(observed['map_known']==1))
    fact('position',f"两根完整5分钟收回={observed['reclaim']}；未破企稳={observed['stop']}")
    fact('rsi',f"5分钟RSI6={observed['fast']}，RSI15={observed['slow']}；日内上穿次数={observed['crosses']}，交叉不等同背离")
    fact('volume',f"低点后两根均量/低点根量={observed['volume_ratio']}；只描述量能，不推断资金流")
    fact('lows',f"末两根低点不降={observed['lows_rising']}；距日内低点根数={observed['bars_since_low']}")
    fact('break',f"上方未收回档持续失守根数={observed['broken_run']}；无修复组合={observed['broken_no_repair']}")
    fact('sources',f"最近失守带来源波段数={observed['source_waves']}；共享高点组数={observed['source_peak_groups']}，不视作独立票数")
    required=['map_known','reclaim','stop','fast','slow','lows_rising','bars_since_low']
    missing=[k for k in required if observed[k] is None]
    if observed['map_known']!=1:missing.append('confirmed_golden_map')
    if missing:
        state='UNKNOWN';basis=['map'];direction='未知'
    else:
        positioned=observed['reclaim']==1 or observed['stop']==1
        rsi=observed['fast']>observed['slow'];contracting=observed['volume_ratio'] is not None and observed['volume_ratio']<=1
        new_low=observed['bars_since_low']==0
        falling=new_low and observed['return_3bars'] is not None and observed['return_3bars']<0 and observed['closes_rising']==0
        if falling and not positioned:
            state='DECLINE_ENHANCING';basis=['lows','break','position'];direction='偏走弱'
        elif observed['broken_no_repair']==1 and not positioned and not (rsi and observed['lows_rising']==1 and observed['rebound'] is not None and observed['rebound']>0):
            state='BREAK_UNREPAIRED';basis=['break','position','rsi'];direction='偏走弱'
        elif observed['volume_ratio'] is None:
            state='UNKNOWN';basis=['volume','position'];direction='未知';missing.append('volume_ratio')
        elif positioned and rsi and contracting and observed['lows_rising']==1:
            state='STRONG_REPAIR';basis=['position','rsi','volume','lows'];direction='偏修复'
        elif positioned or (rsi and observed['lows_rising']==1 and observed['rebound'] is not None and observed['rebound']>0):
            state='WEAK_REPAIR';basis=['position','rsi','volume','lows'];direction='偏修复，配合不完整'
        else:
            state='OBSERVE';basis=['position','rsi','volume'];direction='方向不足'
    return dict(index=index,asof=cutoff,state=state,label=LABELS[state],direction=direction,
        basis_ids=[index+'.'+k for k in basis],facts=facts,observed=observed,missing=missing,
        status='RULE_REFERENCE_PENDING_INDEPENDENT_REVIEW',model_trained=False)


def combine(sse,cyb):
    assert sse['asof']==cyb['asof']
    a,b=sse['state'],cyb['state'];down={'BREAK_UNREPAIRED','DECLINE_ENHANCING'};repair={'STRONG_REPAIR','WEAK_REPAIR'}
    split=(a in down and b in repair) or (a in repair and b in down) or ({a,b}=={'STRONG_REPAIR','WEAK_REPAIR'})
    if a=='UNKNOWN':label='大盘未知，沪指主轴证据不足'
    elif split:label='大盘分化；沪指'+sse['label']+'，创业板'+cyb['label']
    else:label='大盘主轴：沪指'+sse['label']+'；创业板'+cyb['label']
    return dict(asof=sse['asof'],primary='sse',auxiliary='cyb',primary_state=a,
        state='UNKNOWN' if a=='UNKNOWN' else 'DIVERGENT' if split else a,label=label,
        numeric_weight=None,priority='SSE_PRIMARY_NOT_SCORE_AVERAGE',
        action='CONTEXT_ONLY_REQUIRES_STOCK_TRIGGER_AND_HUMAN',
        interpretation='当前状态参考，不保证未来涨跌；个股120分钟幅度另列辅助，不能相减平均')
