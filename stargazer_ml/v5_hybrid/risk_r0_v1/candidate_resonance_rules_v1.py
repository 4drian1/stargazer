"""Daily CSI300 candidate hypothesis. Correct execution != validated financial edge."""
import math
from datetime import datetime
import numpy as np

from volume_exhaustion_exploration_v1 import signals, valid, anchor_context

VERSION = 'DAILY_CSI300_ENGINEERING_V1'
RATIOS = (.236, .382, .5, .618, .786)


def market_regime(ret20):
    if type(ret20) not in (int, float) or not math.isfinite(ret20):
        raise ValueError('INVALID_MARKET_RETURN')
    if -.05 - 1e-12 <= ret20 <= .08 + 1e-12:
        return 'ALLOW'
    if ret20 < -.08:
        return 'MAIN_DECLINE_BLOCK'
    return 'WATCH_ONLY'


def known_wave(x, before):
    """A then B, both strictly before the requested decision/shock; never future pivots."""
    if before < 60 or not valid(x[before - 60:before]).all():
        return None
    a = before - 60 + int(np.argmin(x[before - 60:before, 2]))
    if a + 1 >= before:
        return None
    b = a + 1 + int(np.argmax(x[a + 1:before, 1]))
    low, high = float(x[a, 2]), float(x[b, 1])
    return dict(a=a, b=b, low=low, high=high) if high > low else None


def market_location(x, i):
    wave = known_wave(x, i)
    if wave is None:
        return None
    span = wave['high'] - wave['low']
    levels = [(r, wave['high'] - r * span) for r in RATIOS]
    levels += [(r, wave['low'] - (r - 1) * span) for r in (1.272, 1.618)]
    levels = [(r, p) for r, p in levels if p > 0]
    ratio, price = min(levels, key=lambda rp: abs(x[i, 3] / rp[1] - 1))
    distance = float(x[i, 3] / price - 1)
    return dict(wave=wave, ratio=ratio, price=price, distance=distance,
                near=bool(abs(distance) <= .005 + 1e-12))


def first_confirmation(x, s, through):
    return next((j for j in range(s + 1, min(s + 5, through) + 1)
                 if x[j, 3] > x[j, 0] and x[j, 4] > x[j - 5:j, 4].mean()), None)


def assess(stock, market, i):
    """Only slices <= i, including final confirmation close; output is not a BUY action."""
    if i < 68:
        return dict(state='DATA_INSUFFICIENT', candidate=None, reason='NEEDS_69_SESSIONS')
    x = np.asarray(stock[i - 68:i + 1], dtype=float)
    m = np.asarray(market[i - 68:i + 1], dtype=float)
    if len(x) != 69 or len(m) != 69 or not valid(x).all() or not valid(m).all():
        return dict(state='DATA_INSUFFICIENT', candidate=None, reason='PRICE_OR_VOLUME_INVALID')
    p = 68
    ret20 = float(m[p, 3] / m[p - 20, 3] - 1)
    regime = market_regime(ret20)
    location = market_location(m, p)
    active = [(d, s) for d, s in signals(x) if p - 5 <= s <= p]
    if not active:
        return dict(state='NO_PATTERN', candidate=False, market_ret20=ret20, regime=regime,
                    market_location=location, components=dict(volume=False, confirmation=False))
    # Earliest still-active observation locks the event; no best-return choice.
    d, s = min(active, key=lambda ds: (ds[1], ds[0]))
    context = anchor_context(x, d, s)
    confirm = first_confirmation(x, s, p)
    stock_position = (context.get('rise_over_15pct') is True
                      and context.get('in_broad_236_786_band') is True)
    components = dict(market_environment=regime == 'ALLOW',
                      market_position=location['near'] if location else None,
                      stock_position=stock_position if context.get('anchor_status') ==
                      'MECHANICAL_TRAILING60_NOT_SWING_CONFIRMATION' else None,
                      volume=True, confirmation=confirm == p)
    state = ('ALREADY_CONFIRMED' if confirm is not None and confirm < p else
             'WAIT_CONFIRMATION' if confirm is None else 'RULE_REJECTED')
    if any(v is None for v in components.values()):
        state = 'EVIDENCE_PENDING'
    candidate = all(v is True for v in components.values())
    if candidate:
        state = 'NEXT_SESSION_ENTRY_CANDIDATE'
    return dict(state=state, candidate=None if state == 'EVIDENCE_PENDING' else candidate,
                components=components, market_ret20=ret20, regime=regime,
                market_location=location, stock_context=context,
                shock_offset=int(d - p), shrink_offset=int(s - p),
                confirmation_offset=None if confirm is None else int(confirm - p),
                shock_volume_ratio=float(x[d, 4] / x[d - 8:d, 4].mean()),
                shrink_volume_ratio=float(x[s, 4] / x[d, 4]))


def exit_reference(entry_price, signal_low, signal_date, days):
    """Synthetic reference only. can_sell must be explicitly evidenced by real callers.

    No fees/fill inference; ambiguous same-bar paths never receive a guessed return.
    """
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
           for v in (entry_price, signal_low)):
        raise ValueError('INVALID_REFERENCE_PRICE')
    dates = [d['date'] for d in days]
    for d in [signal_date] + dates:
        if type(d) is not int or len(str(d)) != 8:
            raise ValueError('INVALID_DATE')
        datetime.strptime(str(d), '%Y%m%d')
    if not dates or dates != sorted(set(dates)) or dates[0] <= signal_date:
        raise ValueError('INVALID_ENTRY_OR_DATE_ORDER')
    stop, target = entry_price * .95, entry_price * 1.08
    pending_close_break = False

    def result(state, index=None, price=None):
        return dict(state=state, exit_date=dates[index] if index is not None else None,
                    exit_price=price,
                    reference_gross_return=price / entry_price - 1 if price is not None else None,
                    net=None, execution_certified=False)

    for i, day in enumerate(days[:10]):
        prices = np.array([[day.get(k, np.nan) for k in ('open', 'high', 'low', 'close')] + [1.]])
        if not valid(prices).all():
            return result('DATA_INSUFFICIENT')
        if i == 0:  # A-share T+1: entry-day high/low cannot be used to sell.
            pending_close_break = day['close'] < signal_low
            continue
        can_sell = day.get('can_sell')
        if type(can_sell) is not bool:
            return result('METADATA_UNKNOWN')
        if not can_sell:
            pending_close_break |= day['close'] < signal_low
            continue
        if pending_close_break:
            return result('CLOSE_BREAK_NEXT_OPEN', i, day['open'])
        if day['open'] <= stop + 1e-10:
            return result('GAP_STOP_REFERENCE', i, day['open'])
        if day['open'] >= target - 1e-10:
            return result('GAP_TARGET_REFERENCE', i, day['open'])
        hit_stop, hit_target = day['low'] <= stop + 1e-10, day['high'] >= target - 1e-10
        if hit_stop and hit_target:
            return result('AMBIGUOUS_INTRADAY_ORDER')
        if hit_stop:
            return result('STOP_REFERENCE', i, stop)
        if hit_target:
            return result('TARGET_REFERENCE', i, target)
        if i == 9:
            return result('TIME_EXIT_REFERENCE', i, day['close'])
        pending_close_break = day['close'] < signal_low
    return result('POSITION_UNCLOSED' if len(days) >= 10 else 'CENSORED_TAIL')
