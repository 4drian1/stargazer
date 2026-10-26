"""P-frozen multi-level trigger, same endpoint grouping and kernels as paid replay."""
import importlib.util
from pathlib import Path

import pandas as pd

FIT=Path(__file__).resolve().parent/'tarot_a_second_leg_fit_20261022_v1'
VERSION='TAROT_A_SECOND_CASH_V2'


def adapter():
    path=FIT/'stockdb_sdk_adapter_diagnostic_20261022_v1/adapter.py'
    spec=importlib.util.spec_from_file_location('tarot_endpoint_frozen',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def evaluate(row,day,cutoff,minute_rows):
    if row['entry_rule'].get('policy')=='GOLDEN_RESONANCE_CROSSDAY_V4':
        from golden_resonance_v4 import evaluate as current
        return current(row,day,cutoff,minute_rows)
    if row['entry_rule'].get('policy')=='GOLDEN_LADDER_STOP_V3':
        from golden_ladder_v3 import evaluate as current
        return current(row,day,cutoff,minute_rows)
    rule=row['entry_rule'];levels=rule['levels'];close_p=float(row['reference_price'])
    if levels!=sorted(set(levels),reverse=True) or any(not 0<v<=close_p for v in levels):
        raise ValueError('INVALID_FROZEN_LEVELS')
    frame=pd.DataFrame(minute_rows)
    if frame.empty or 'date' not in frame:return dict(status='DATA_INSUFFICIENT',reason='MINUTES_MISSING')
    frame=frame[(frame.date//1000000==day)&(frame.date<=cutoff)].copy()
    native=adapter()
    from golden_ladder_v3 import completed_bars
    five,_,_,source_status=completed_bars(frame.to_dict('records'),row['code'],day,cutoff)
    if not five or day*1000000+five[-1]['t']!=cutoff:
        return dict(status='DATA_INSUFFICIENT',reason=source_status)
    opportunities=[]
    for node in levels:
        evidence=native.d.prior.confirmations(five,[0]*len(five),node,rule['atr_ref'])
        if evidence.get('rsi_index')==len(five)-1:
            replay=native.d.prior.kernel.replay(five,[0]*len(five),node,rule['atr_ref'])
            opportunities.append(dict(node=node,branch='RSI',flags=replay['flags'],divergence=replay.get('divergence')))
        if (len(five)>=2 and all(bar['close']>=node for bar in five[-2:]) and
                any(bar['low']<node for bar in five[:-1])):
            opportunities.append(dict(node=node,branch='RECLAIM',flags={'two_completed_reclaim':True},divergence=None))
    best=sorted(opportunities,key=lambda x:(-x['node'],x['branch']!='RSI'))[0] if opportunities else {}
    return dict(status='RULE_TRIGGERED' if best else 'WAIT_SIGNAL',reason='PRICE_CONFIRMATION_REQUIRES_SAME_CUTOFF_RISK',
        trigger_time=cutoff if best else None,last_two_closes=[b['close'] for b in five[-2:]],
        next_lower_node=next((v for v in levels if v<=five[-1]['close']),None),
        pierced_levels=[v for v in levels if any(b['low']<v for b in five)],
        completed_bar_count=len(five),price_asof=cutoff,**best,
        scope='价格事实；不是买入建议，须同截止Pyxis和DeepSeek复核后由人工决定')
