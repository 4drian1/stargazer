"""Conditional-research anchor checks, not complete action/RAW certification."""
import numpy as np


def classify(anchor, previous_day, current_day, bars, audit, source_bound, quarantine=()):
    """Only previous-day daily price is accessed; future events never enter factors."""
    if previous_day >= current_day:
        raise ValueError('ANCHOR_NOT_STRICTLY_PAST')
    if previous_day in quarantine or current_day in quarantine:
        return 'CONFLICT', 'QUARANTINED_DEPENDENCY', None
    if audit is None:
        return 'PENDING', 'NO_BOUND_ORDINARY_ACTION_EVIDENCE', None
    if not source_bound:
        return 'PENDING', 'MINUTE_SOURCE_NOT_BOUND_TO_ACCEPTED_RESEARCH', None
    problems = audit.get('issues', []) + audit.get('missing_records', [])
    for issue in problems:
        day = issue.get('date')
        if day is None or previous_day <= day <= current_day:
            return 'CONFLICT', 'UNRESOLVED_ACTION_DEPENDENCY', None
    bar = bars.get(previous_day)
    if bar is None:
        return 'PENDING', 'PREVIOUS_HTTP_DAILY_BAR_MISSING', None
    try:
        o, h, l, c = [float(bar[k]) for k in ('open', 'high', 'low', 'close')]
    except (KeyError, TypeError, ValueError):
        return 'PENDING', 'PREVIOUS_HTTP_DAILY_BAR_INVALID', None
    if not (np.isfinite([o,h,l,c]).all() and 0 < l <= min(o,c) <= max(o,c) <= h):
        return 'CONFLICT', 'PREVIOUS_HTTP_DAILY_BAR_INVALID', None
    if not np.isfinite(anchor) or anchor <= 0 or abs(anchor-c) > .01 + 1e-12:
        return 'CONFLICT', 'PREVIOUS_CLOSE_BASIS_MISMATCH', None
    multiplier = 1.
    seen = set()
    for event in audit.get('events', []):
        day = event['date']
        if not previous_day < day <= current_day:
            continue
        if day in seen:
            return 'CONFLICT', 'DUPLICATE_ACTION_DATE', None
        seen.add(day)
        m = event.get('multiplier')
        if (event.get('status') != 'ORDINARY_PARAMETERS_CORROBORATED'
                or not isinstance(m, (int, float)) or isinstance(m, bool)
                or not np.isfinite(m) or m <= 0):
            return 'CONFLICT', 'UNRESOLVED_ACTION_PARAMETERS', None
        multiplier *= m
    if not np.isfinite(multiplier) or multiplier <= 0:
        return 'CONFLICT', 'INVALID_NORMALIZATION', None
    return 'CONDITIONAL_RESEARCH', 'PRIOR_CLOSE_MATCH_AND_BOUND_ORDINARY_SCOPE', multiplier


def normalized_return(close, anchor, multiplier):
    values = np.asarray([close, anchor, multiplier], dtype=float)
    if not (np.isfinite(values).all() and (values > 0).all()):
        raise ValueError('INVALID_NORMALIZED_RETURN_INPUT')
    value = close / anchor * multiplier - 1
    if not np.isfinite(value) or value <= -1:
        raise ValueError('INVALID_NORMALIZED_RETURN')
    return float(value)
