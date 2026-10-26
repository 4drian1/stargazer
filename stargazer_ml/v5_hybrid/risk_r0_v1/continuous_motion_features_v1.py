"""Causal continuous motion on accepted entity paths; no IO, labels or fitting.

Existing interval returns / close-high drawdowns are reused, not new evidence.
Compute per board or per (board, stock) BEFORE selecting/aggregating leaders.
"""
import numpy as np
import pandas as pd

from intraday_relative_stall_v1 import TIME, slots

METRICS = ('interval_return', 'deceleration', 'giveback',
           'relative_return', 'relative_deceleration', 'relative_giveback')


def entity_motion(frame, keys, frequency):
    """Positive deceleration = prior interval gain minus current interval gain.

    Input canonical columns: interval_return, close_drawdown (<=0),
    interval_known, peak_known. Caller must positively bind provenance / stable
    members and interval semantics; these booleans never certify price scale.
    Lunch resets adjacent-interval comparison, NOT already-observed day high.
    """
    slots(20250102, frequency)
    if frame.duplicated([*keys, TIME]).any() or frame[[*keys, TIME]].isna().any().any():
        raise ValueError('MISSING_OR_DUPLICATE_ENTITY_TIME')
    f = frame.sort_values([*keys, TIME]).reset_index(drop=True).copy()
    if any(not isinstance(t, (int, np.integer)) for t in f[TIME]):
        raise ValueError('NONINTEGER_CUTOFF')
    allowed = {t for d in f[TIME].floordiv(1000000).unique() for t in slots(int(d), frequency)}
    if not set(f[TIME]) <= allowed:
        raise ValueError('OFF_GRID_CUTOFF')
    for field in ('interval_known', 'peak_known'):
        if not f[field].dropna().map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError('PROVENANCE_FLAG_NOT_BOOLEAN')
    for field in ('interval_return', 'close_drawdown'):
        f[field] = pd.to_numeric(f[field], errors='raise')
        if np.isinf(f[field]).any():
            raise ValueError('INFINITE_PATH_VALUE')
    if ((f.close_drawdown.dropna() > 1e-12).any() or (f.close_drawdown.dropna() <= -1).any()
            or (f.interval_return.dropna() <= -1).any()):
        raise ValueError('INVALID_RETURN_OR_DRAWDOWN')
    f['motion_interval_return'] = f.interval_return.where(f.interval_known.fillna(False))
    f['motion_giveback'] = (-f.close_drawdown).clip(lower=0).where(f.peak_known.fillna(False))
    g = f.groupby(keys, sort=False, dropna=False)
    previous_time = g[TIME].shift()
    minute = (f[TIME] // 10000 % 100) * 60 + f[TIME] // 100 % 100
    prev_minute = (previous_time // 10000 % 100) * 60 + previous_time // 100 % 100
    adjacent = ((f[TIME] // 1000000 == previous_time // 1000000)
                & (minute - prev_minute == frequency) & ((minute < 720) == (prev_minute < 720)))
    f['motion_deceleration'] = (g.motion_interval_return.shift() - f.motion_interval_return).where(adjacent)
    return f[[*keys, TIME, 'motion_interval_return', 'motion_deceleration', 'motion_giveback']]


def relative_motion(entity, market, keys):
    """Same-time return DIFFERENCE; never divide by near-zero market return."""
    if entity.duplicated([*keys, TIME]).any() or market.duplicated(TIME).any():
        raise ValueError('DUPLICATE_RELATIVE_JOIN')
    cols = ['motion_interval_return', 'motion_deceleration', 'motion_giveback']
    m = market[[TIME, *cols]].rename(columns={c: 'market_' + c for c in cols})
    f = entity.merge(m, on=TIME, how='left', validate='many_to_one', sort=False)
    for name in ('interval_return', 'deceleration', 'giveback'):
        target = 'relative_return' if name == 'interval_return' else 'relative_' + name
        f['motion_' + target] = f['motion_' + name] - f['market_motion_' + name]
    return f[[*keys, TIME, *['motion_' + c for c in METRICS]]]


def summarize_motion(frame, selected):
    """One cutoff/group: known-only statistics with selected denominator shown."""
    if len(frame) != len(selected):
        raise ValueError('SELECTION_SHAPE_MISMATCH')
    mask = np.asarray(selected)
    if mask.dtype != bool:
        raise ValueError('SELECTION_NOT_BOOLEAN')
    result = {'selected': int(mask.sum())}
    for metric in METRICS:
        values = frame.loc[mask, 'motion_' + metric].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError('INFINITE_SUMMARY')
        known = values[np.isfinite(values)]
        result[metric] = dict(known=len(known), unknown=len(values)-len(known),
            coverage=len(known)/len(values) if len(values) else None,
            mean=float(known.mean()) if len(known) else None,
            quantiles=np.quantile(known, [.1,.5,.9]).tolist() if len(known) else None)
    return result
