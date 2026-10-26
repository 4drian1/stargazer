"""Asymmetric close-confirmed research anchors. Keep v1/frozen replay intact.

Engineering policies (not validated market edge): equal closes reset a run;
each direction run can confirm one contraction; expansion is processed first,
but a separately confirmed opposite-end contraction is not silently discarded.
"""
from copy import deepcopy
from decimal import Decimal

from golden_anchor_rules_v1 import initialize as init_v1, validate


def validate_open(bar):
    validate(bar)
    price = bar.get('open')
    if isinstance(price, bool) or not isinstance(price, (float, int)) or not bar['low'] <= price <= bar['high']:
        raise ValueError('INVALID_OPEN')


def initialize(bars):
    for bar in bars:
        validate_open(bar)
    state = init_v1(bars)
    state.update(rule_version='ASYMMETRIC_3DOWN_1PCT_5WATCH_7UP_V2',
                 previous_bar=deepcopy(bars[-1]), run_direction=0, run_count=0,
                 run_origin=None, run_confirmed=False, run_has_one_percent_drop=False,
                 events=['INITIALIZE'], watch_up=False)
    return state


def close_update(previous, bar):
    validate_open(bar)
    if bar['day'] <= previous['asof_day']:
        raise ValueError('NONINCREASING_DAY')
    state = deepcopy(previous)
    last = previous['previous_bar']
    c, p = bar['close'], last['close']
    direction = 1 if c > p else -1 if c < p else 0
    if direction == 0 or direction != previous['run_direction']:
        state.update(run_count=1 if direction else 0, run_origin=deepcopy(last) if direction else None,
                     run_confirmed=False, run_has_one_percent_drop=False)
    else:
        state['run_count'] += 1
    state['run_direction'] = direction
    drop_one = Decimal(str(p)) - Decimal(str(c)) >= Decimal('.01') * Decimal(str(p))
    state['run_has_one_percent_drop'] |= direction < 0 and drop_one
    state.update(asof_day=bar['day'], previous_bar=deepcopy(bar), events=[], event='HOLD', watch_up=False)

    # Breaks always update at this close; they do not wait for contraction confirmation.
    if c > state['high']:
        state.update(high=c, high_day=bar['day'], high_basis='CONFIRMED_DAY_CLOSE')
        state['events'].append('EXPAND_HIGH')
    elif c < state['low']:
        state.update(low=c, low_day=bar['day'], low_basis='CONFIRMED_DAY_CLOSE')
        state['events'].append('EXPAND_LOW')

    origin = state['run_origin']
    if origin and not state['run_confirmed']:
        inside = state['low'] < origin['close'] < state['high']
        if (direction < 0 and state['run_count'] >= 3 and state['run_has_one_percent_drop']
                and origin['close'] > origin['open'] and inside):
            state.update(high=origin['close'], high_day=origin['day'], high_basis='THREE_DOWN_ORIGIN_CLOSE',
                         run_confirmed=True)
            state['events'].append('SHRINK_HIGH')
        elif direction > 0:
            state['watch_up'] = 5 <= state['run_count'] < 7
            if state['run_count'] >= 7 and inside:
                state.update(low=origin['close'], low_day=origin['day'], low_basis='SEVEN_UP_ORIGIN_CLOSE',
                             run_confirmed=True)
                state['events'].append('SHRINK_LOW')
    if state['events']:
        state['version'] += 1
        state['event'] = '+'.join(state['events'])
    if not state['low'] < state['high']:
        raise AssertionError('INVALID_RANGE')
    return state


def replay(bars):
    if len(bars) < 7:
        return []
    state = initialize(bars[:7])
    states = [state]
    for bar in bars[7:]:
        state = close_update(state, bar)
        states.append(state)
    return states
