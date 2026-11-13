"""Research-only current state with separate position, momentum and maturity."""
from state_rules import describe, known

LABELS = dict(STRONG_REPAIR='强修复', WEAK_REPAIR='弱修复',
    REPAIR_PENDING_EVIDENCE='位置已确认，修复配合待确认',
    LOCAL_REBOUND='局部回升，黄金位置未确认',
    BREAK_UNREPAIRED='破位未修复', DECLINE_ENHANCING='下跌增强',
    OBSERVE='观察，方向证据不足', UNKNOWN='关键证据未知')


def describe_parts(index, cutoff, features, close, raw_volume_valid,
                   current_node=None, current_path='NONE', previous_node=None):
    original = describe(index, cutoff, features)
    o = original['observed']
    mapped = o['map_known'] == 1
    positioned = mapped and (o['reclaim'] == 1 or o['stop'] == 1)
    if positioned:
        assert known(current_node) and current_node > 0
        assert current_path == ('RECLAIM' if o['reclaim'] == 1 else 'STOP_ABOVE')
    else:
        assert current_node is None and current_path == 'NONE'
    price_known = known(close) and close > 0
    prior_broken = (price_known and known(previous_node) and close < previous_node)
    tracked = current_node if positioned else previous_node
    relation = ('UNKNOWN' if not price_known else 'NONE' if tracked is None else
                'BELOW' if close < tracked else 'AT_OR_ABOVE')
    rsi_known = o['fast'] is not None and o['slow'] is not None
    rsi_up = rsi_known and o['fast'] > o['slow']
    ratio = o['volume_ratio']
    if not raw_volume_valid:
        volume_state = 'SOURCE_MISSING_OR_INVALID'
    elif ratio is not None:
        volume_state = 'CONTRACTING' if ratio <= 1 else 'NOT_CONTRACTING'
    elif o['bars_since_low'] in [0, 1]:
        volume_state = 'PENDING_AFTER_LOW'
    else:
        volume_state = 'UNKNOWN'
    volume_known = volume_state in ['CONTRACTING', 'NOT_CONTRACTING']
    partial = (rsi_up and o['lows_rising'] == 1 and o['rebound'] is not None and o['rebound'] > 0)
    falling = (o['bars_since_low'] == 0 and o['return_3bars'] is not None
               and o['return_3bars'] < 0 and o['closes_rising'] == 0)
    if not mapped or not price_known:
        state = 'UNKNOWN'
    elif positioned:
        if not rsi_known or not volume_known:
            state = 'REPAIR_PENDING_EVIDENCE'
        elif rsi_up and volume_state == 'CONTRACTING':
            state = 'STRONG_REPAIR'
        else:
            state = 'WEAK_REPAIR'
    elif not rsi_known or o['bars_since_low'] is None:
        state = 'UNKNOWN'
    elif falling:
        state = 'DECLINE_ENHANCING'
    elif o['broken_no_repair'] == 1 and not partial:
        state = 'BREAK_UNREPAIRED'
    elif partial:
        state = 'LOCAL_REBOUND'
    else:
        state = 'OBSERVE'
    return dict(index=index, asof=cutoff, state=state, label=LABELS[state],
        position=dict(map_known=mapped, current_node=current_node, current_path=current_path,
            previous_confirmed_node=previous_node, previous_node_broken=bool(prior_broken),
            tracked_node=tracked, tracked_relation=relation,
            lower_repair_with_previous_break=bool(prior_broken and positioned)),
        momentum=dict(rsi_state='UNKNOWN' if not rsi_known else 'FAST_ABOVE' if rsi_up else 'FAST_NOT_ABOVE',
            rsi6=o['fast'], rsi15=o['slow'], crosses=o['crosses'], lows_non_decreasing=o['lows_rising'],
            closes_rising=o['closes_rising'], low_rebound_return=o['rebound'],
            current_new_low=o['bars_since_low']==0, falling_now=bool(falling)),
        maturity=dict(raw_volume_valid=bool(raw_volume_valid), volume_state=volume_state,
            volume_ratio=ratio, bars_since_low=o['bars_since_low'],
            ratio_definition='last_two_mean_volume / observed_session_low_bar_volume; not monotonic shrinkage'),
        status='RESEARCH_RULE_REFERENCE_PENDING_INDEPENDENT_REVIEW',
        model_trained=False, automatic_action=False)
