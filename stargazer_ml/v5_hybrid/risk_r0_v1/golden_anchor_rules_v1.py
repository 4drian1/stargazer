"""Offline risk-side anchor mechanics, no scores, fitting, I/O or trading actions.

STRICT_GT_LT is an explicit engineering convention pending equality approval.
Observed-day sequencing is not proof of an exchange calendar or live availability.
"""
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal
import math


def validate(bar):
    datetime.strptime(str(bar['day']), '%Y%m%d')
    if not bar.get('completed'):
        raise ValueError('INCOMPLETE_DAY')
    if any(not isinstance(bar.get(k), (int, float)) or isinstance(bar[k], bool)
           or not math.isfinite(bar[k]) or bar[k] <= 0 for k in ('high', 'low', 'close')):
        raise ValueError('INVALID_PRICE')
    if not bar['low'] <= bar['close'] <= bar['high']:
        raise ValueError('INVALID_OHLC')


def initialize(bars):
    if len(bars) != 7 or [b['day'] for b in bars] != sorted({b['day'] for b in bars}):
        raise ValueError('NEED_SEVEN_ORDERED_UNIQUE_COMPLETED_DAYS')
    for bar in bars:
        validate(bar)
    high = max(bars, key=lambda b: b['high'])
    low = min(bars, key=lambda b: b['low'])
    if high['high'] <= low['low']:
        raise ValueError('ZERO_RANGE')
    return dict(high=high['high'], low=low['low'], high_day=high['day'], low_day=low['day'],
                asof_day=bars[-1]['day'], initialized_day=bars[-1]['day'], version=1,
                high_basis='INITIAL_DAY_HIGH', low_basis='INITIAL_DAY_LOW',
                event='INITIALIZE', comparison='STRICT_GT_LT_ENGINEERING_CONVENTION',
                actual_known_at=None, training_ready=False)


def close_update(previous, bar):
    validate(bar)
    if bar['day'] <= previous['asof_day']:
        raise ValueError('NONINCREASING_DAY')
    state = deepcopy(previous)
    state.update(asof_day=bar['day'], event='HOLD')
    side = 'high' if bar['close'] > state['high'] else 'low' if bar['close'] < state['low'] else None
    if side:
        state.update({side: bar['close'], side + '_day': bar['day'], side + '_basis': 'CONFIRMED_DAY_CLOSE',
                      'version': state['version'] + 1, 'event': 'UPDATE_' + side.upper()})
    return state


def observe(previous, price):
    if not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
        raise ValueError('INVALID_PRICE')
    return dict(high=previous['high'], low=previous['low'], version=previous['version'], action='NONE',
                relation='ABOVE' if price > previous['high'] else 'BELOW' if price < previous['low'] else 'WITHIN')


def between_levels(low, high, ratios):
    if not 0 < low < high:
        raise ValueError('INVALID_ANCHORS')
    result = {}
    for ratio in ratios:
        q = Decimal(str(ratio))
        if not q.is_finite() or not 0 <= q <= 1:
            raise ValueError('BETWEEN_RATIO_ONLY')
        result[str(ratio)] = float(Decimal(str(low)) + (Decimal(str(high)) - Decimal(str(low))) * q)
    return result


def time_nodes(day, ratios):
    date = datetime.strptime(str(day), '%Y%m%d')
    nodes = []
    for hour, minute in ((9, 30), (13, 0)):
        start = date.replace(hour=hour, minute=minute)
        for ratio in ratios:
            q = Decimal(str(ratio))
            if not q.is_finite() or not 0 <= q <= 1:
                raise ValueError('INVALID_TIME_RATIO')
            node = start + timedelta(microseconds=int(q * 7200 * 1000000))
            elapsed_minutes = q * 120
            ended = {str(k): ((start + timedelta(minutes=int(elapsed_minutes // k) * k)).isoformat()
                             if elapsed_minutes >= k else None) for k in (5, 15)}
            nodes.append(dict(session_start=start.isoformat(), ratio=str(ratio), node=node.isoformat(),
                              latest_bar_end_under_end_assumption=ended, action='NONE'))
    return nodes
