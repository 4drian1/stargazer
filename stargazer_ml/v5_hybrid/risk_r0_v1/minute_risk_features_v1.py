"""Explicit-clock 15m diagnostics. Unknown arrival/source semantics never become live facts."""
from datetime import datetime, timedelta
import math
import numpy as np

FIELDS = ('open', 'high', 'low', 'close', 'volume')


def parse_ts(value):
    # Accept Python int, numpy integer, and numpy float (DataFrame.iterrows()
    # promotes mixed int/float columns to float64). Must be integer-valued.
    if isinstance(value, (int, np.integer)):
        ival = int(value)
    elif isinstance(value, float) and value.is_integer():
        ival = int(value)
    else:
        raise ValueError('REQUIRES_14_DIGIT_EXCHANGE_CLOCK')
    if len(str(ival)) != 14:
        raise ValueError('REQUIRES_14_DIGIT_EXCHANGE_CLOCK')
    return datetime.strptime(str(ival), '%Y%m%d%H%M%S')


def encode(dt):
    return int(dt.strftime('%Y%m%d%H%M%S'))


def legacy_panel_clock(date, hhmm, value):
    """Convert only the documented D*10000+HHMM*100 clock, never pad a string."""
    if any(type(x) is not int for x in (date, hhmm, value)):
        raise ValueError('PANEL_CLOCK_TYPE')
    if len(str(date)) != 8 or hhmm % 100 >= 60 or not (930 <= hhmm <= 1130 or 1300 <= hhmm <= 1500):
        raise ValueError('PANEL_CLOCK_SESSION')
    if value != date * 10000 + hhmm * 100:
        raise ValueError('PANEL_CLOCK_RELATION')
    return encode(datetime.strptime(str(date) + f'{hhmm:04d}00', '%Y%m%d%H%M%S'))


def source_end(row, semantics):
    if semantics not in ('START', 'END'):
        raise ValueError('EXPLICIT_TIME_SEMANTICS_REQUIRED')
    dt = parse_ts(row['date'])
    return dt + timedelta(minutes=1) if semantics == 'START' else dt


def aggregate(rows, code, dates, semantics):
    if semantics not in ('START', 'END'):
        raise ValueError('EXPLICIT_TIME_SEMANTICS_REQUIRED')
    if dates != sorted(set(dates)):
        raise ValueError('DATES_NOT_SORTED_UNIQUE')
    records, values, issues = {}, {}, []
    for row in rows:
        dt = parse_ts(row['date'])
        if row['date'] // 1000000 not in dates:
            continue
        if row.get('code') != code:
            raise ValueError('MINUTE_CODE_MISMATCH')
        key = encode(dt)
        if dt.second:
            raise ValueError('NOT_MINUTE_GRID')
        fields = tuple(row.get(k) for k in FIELDS)
        if key in values:
            if values[key] != fields:
                records[key] = None
                issues.append(dict(ts=key, reason='CONFLICTING_DUPLICATE'))
            else:
                issues.append(dict(ts=key, reason='IDENTICAL_DUPLICATE'))
            continue
        values[key] = fields
        finite = all(type(v) in (int, float) and math.isfinite(v) for v in fields)
        o, h, l, c, v = fields
        okay = finite and min(o, h, l, c) > 0 and v >= 0 and l <= min(o, c) <= max(o, c) <= h
        records[key] = dict(zip(FIELDS, fields)) if okay else None
        if not okay:
            issues.append(dict(ts=key, reason='INVALID_OHLC_OR_VOLUME'))
    result = []
    for day in dates:
        d = datetime.strptime(str(day), '%Y%m%d')
        for session_start in (d.replace(hour=9, minute=30), d.replace(hour=13)):
            for offset in range(8):
                start = session_start + timedelta(minutes=15 * offset)
                end = start + timedelta(minutes=15)
                grid = [encode(start + timedelta(minutes=j + (1 if semantics == 'END' else 0))) for j in range(15)]
                missing = [t for t in grid if t not in records]
                invalid = [t for t in grid if t in records and records[t] is None]
                item = dict(day=day, bar_start=encode(start), bar_end=encode(end), slot=len(result) % 16,
                    source_semantics_assumption=semantics, available_at=None, diagnostic_only=True,
                    status='COMPLETE_UNDER_TIME_ASSUMPTION' if not missing and not invalid else 'INCOMPLETE',
                    missing_minutes=missing, invalid_minutes=invalid,
                    open=None, high=None, low=None, close=None, volume=None)
                if not missing and not invalid:
                    group = [records[t] for t in grid]
                    item.update(open=float(group[0]['open']), close=float(group[-1]['close']),
                        high=float(max(r['high'] for r in group)), low=float(min(r['low'] for r in group)),
                        volume=float(sum(r['volume'] for r in group)))
                result.append(item)
    return result, issues


def features(bars):
    """Sequential Wilder14, no missing-bar compression; no bullish/bearish thresholds."""
    result, volume_slots = [], {}
    days = sorted({b['day'] for b in bars})
    prior_day = {d: days[i - 1] if i else None for i, d in enumerate(days)}
    previous, gains, losses, avg_gain, avg_loss, previous_rsi = None, [], [], None, None, None
    current_day, daily_open, high, low, prefix_complete = None, None, None, None, False
    for bar in bars:
        complete = bar['status'] == 'COMPLETE_UNDER_TIME_ASSUMPTION'
        if bar['day'] != current_day:
            current_day = bar['day']
            daily_open = bar['open'] if complete and bar['slot'] == 0 else None
            high, low, prefix_complete = None, None, complete and bar['slot'] == 0
        prefix_complete &= complete
        rsi = delta = day_return = drawdown = recovery = volume_change = same_slot_ratio = None
        if complete:
            close = bar['close']
            high = max(high, bar['high']) if high is not None else bar['high']
            low = min(low, bar['low']) if low is not None else bar['low']
            if prefix_complete and daily_open:
                day_return, drawdown, recovery = close / daily_open - 1, close / high - 1, close / low - 1
            if previous is not None and previous['status'] == 'COMPLETE_UNDER_TIME_ASSUMPTION':
                diff = close - previous['close']
                if avg_gain is None:
                    gains.append(max(diff, 0.)); losses.append(max(-diff, 0.))
                    if len(gains) == 14:
                        avg_gain, avg_loss = sum(gains) / 14, sum(losses) / 14
                else:
                    avg_gain = (avg_gain * 13 + max(diff, 0.)) / 14
                    avg_loss = (avg_loss * 13 + max(-diff, 0.)) / 14
                if avg_gain is not None:
                    rsi = (50. if avg_gain == avg_loss == 0 else 100. if avg_loss == 0
                           else 100. - 100. / (1. + avg_gain / avg_loss))
                    delta = rsi - previous_rsi if previous_rsi is not None else None
                if previous['bar_end'] == bar['bar_start'] and previous['volume'] > 0:
                    volume_change = bar['volume'] / previous['volume'] - 1
            baseline = volume_slots.get((prior_day[bar['day']], bar['slot']))
            if baseline is not None and baseline > 0:
                same_slot_ratio = bar['volume'] / baseline
            volume_slots[bar['day'], bar['slot']] = bar['volume']
        else:
            gains, losses, avg_gain, avg_loss = [], [], None, None
        previous_rsi, previous = rsi, bar
        result.append(dict(bar_end=bar['bar_end'], source_status=bar['status'],
            rsi14=rsi, rsi14_delta1=delta, day_return=day_return,
            drawdown_from_day_high=drawdown, recovery_from_day_low=recovery,
            volume_change_previous_adjacent_bar=volume_change,
            volume_ratio_previous_session_same_slot=same_slot_ratio,
            baseline_scope='ONE_PRIOR_SESSION_DIAGNOSTIC_NOT_60DAY_SAMETIME_BASELINE',
            available_at=None, live_feature_admitted=False))
    return result
