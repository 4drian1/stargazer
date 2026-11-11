"""Native 5-min -> arrays[5]/[15], strict comparison, real kernel (v3).

18.176 fixes on top of 18.175:
  #2  Duplicate timestamps are grouped BEFORE writing. Records for the same ts
      that DISAGREE are a conflict: the slot is invalidated (NaN) and its
      15-min group becomes incomplete. v2 kept the first record, so a conflict
      silently influenced every downstream value.
      `allowed_days` lets the builder itself honour classify_file's
      eligible_days instead of trusting the caller to filter.
  #3  interface_consistency_test builds one group_valid mask over all THREE
      5-min slots x ALL required fields, and applies that same mask to every
      expected array. v2 used the first/third slot directly, so a correctly
      generated NaN (missing middle slot) was reported as inconsistent.

Retained from 18.175: strict compare (shape -> NaN pattern -> inf -> values,
NOT_COMPARABLE), self-rejecting builder, real kernel import via sys.path only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_RISK_R0 = Path(__file__).resolve().parent
if str(_RISK_R0) not in sys.path:
    sys.path.insert(0, str(_RISK_R0))
# Real dependency, no stub: the module exists in risk_r0_v1.
from build_joint_risk_features_v1 import (  # noqa: E402
    features_from_arrays as real_features_from_arrays,
    wilder_rsi as real_wilder_rsi,
)

BARS_PER_DAY = 48
FIELDS = ('open', 'high', 'low', 'close', 'volume')
_VALUE_KEYS = ('open', 'high', 'low', 'close', 'raw_volume')


def build_minute_arrays(rows_by_day: dict, days: list[int],
                        allowed_days=None) -> dict:
    """Build {5: {...}, 15: {...}} on the shared calendar `days`.

    Days absent from rows_by_day stay NaN in place. `allowed_days` (the
    classify_file eligible days) restricts which days may contribute at all.
    """
    if list(days) != sorted(set(days)):
        raise ValueError('DAYS_NOT_UNIQUE_SORTED')
    n_day = len(days)
    day_index = {d: i for i, d in enumerate(days)}
    a5 = {f: np.full(n_day * BARS_PER_DAY, np.nan) for f in FIELDS}
    a15 = {f: np.full(n_day * 16, np.nan) for f in FIELDS}
    rejected = dict(day_mismatch=0, ts_inconsistent=0, duplicate_ts=0,
                    conflicting_ts=0, off_grid=0, bad_values=0,
                    unknown_day=0, day_not_allowed=0)
    allowed = set(allowed_days) if allowed_days is not None else None
    for d, bars in rows_by_day.items():
        if d not in day_index:
            rejected['unknown_day'] += len(bars)
            continue
        if allowed is not None and d not in allowed:
            rejected['day_not_allowed'] += len(bars)
            continue
        di = day_index[d]
        # ---- group by ts first, so duplicates can never "keep the first" ----
        by_ts: dict[int, list[dict]] = {}
        for r in bars:
            if r['day'] != d:
                rejected['day_mismatch'] += 1
                continue
            exp_ts = d * 1000000 + (r['minute'] // 60) * 10000 + (r['minute'] % 60) * 100
            if r['ts'] != exp_ts or not 0 <= r['minute'] < 1440:
                rejected['ts_inconsistent'] += 1
                continue
            if not _values_ok(r):
                rejected['bad_values'] += 1
                continue
            by_ts.setdefault(r['ts'], []).append(r)
        for ts, group in by_ts.items():
            slot5 = _slot(ts // 10000 % 100 * 60 + ts // 100 % 100)
            if slot5 < 0:
                rejected['off_grid'] += 1
                continue
            if len(group) > 1:
                rejected['duplicate_ts'] += 1
                # conflicting records for one timestamp: invalidate the slot
                if _disagree(group):
                    rejected['conflicting_ts'] += 1
                    continue
            r = group[0]
            idx = di * BARS_PER_DAY + slot5
            a5['open'][idx] = float(r['open'])
            a5['high'][idx] = float(r['high'])
            a5['low'][idx] = float(r['low'])
            a5['close'][idx] = float(r['close'])
            a5['volume'][idx] = float(r['raw_volume'])
        for g in range(16):
            k = (di * BARS_PER_DAY + g * 3,
                 di * BARS_PER_DAY + g * 3 + 1,
                 di * BARS_PER_DAY + g * 3 + 2)
            if all(np.isfinite(a5[f][i]) for f in FIELDS for i in k):
                i15 = di * 16 + g
                a15['open'][i15] = a5['open'][k[0]]
                a15['high'][i15] = max(a5['high'][i] for i in k)
                a15['low'][i15] = min(a5['low'][i] for i in k)
                a15['close'][i15] = a5['close'][k[2]]
                a15['volume'][i15] = sum(a5['volume'][i] for i in k)
    return {5: a5, 15: a15}, rejected


def _disagree(group: list[dict]) -> bool:
    """True when records sharing a timestamp disagree on any value."""
    first = tuple(float(group[0][k]) for k in _VALUE_KEYS)
    return any(tuple(float(r[k]) for k in _VALUE_KEYS) != first
               for r in group[1:])


def _slot(minute: int) -> int:
    if minute % 5 != 0:
        return -1
    if 575 <= minute <= 690:
        return (minute - 575) // 5
    if 785 <= minute <= 900:
        return 24 + (minute - 785) // 5
    return -1


def _values_ok(r: dict) -> bool:
    try:
        o, h, l, c = (float(r[f]) for f in ('open', 'high', 'low', 'close'))
        v = float(r['raw_volume'])
    except (TypeError, ValueError, KeyError):
        return False
    if any(x != x or x in (float('inf'), float('-inf')) for x in (o, h, l, c, v)):
        return False
    return min(o, h, l, c) > 0 and v >= 0 and l <= min(o, c) <= max(o, c) <= h


# --------------------------------------------------------------------------- #
# Strict comparison
# --------------------------------------------------------------------------- #
def compare(a, b, atol: float = 1e-9) -> dict:
    """Shape -> finite/NaN pattern -> inf -> values. n==0 => NOT_COMPARABLE."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        return dict(ok=False, reason='SHAPE_MISMATCH', n=0,
                    shape_a=list(x.shape), shape_b=list(y.shape))
    fx, fy = np.isfinite(x), np.isfinite(y)
    if not np.array_equal(fx, fy):
        return dict(ok=False, reason='MISSING_PATTERN_MISMATCH', n=0,
                    n_pattern_diff=int((fx != fy).sum()))
    if np.isinf(x).any() or np.isinf(y).any():
        return dict(ok=False, reason='INF_PRESENT', n=0)
    n = int(fx.sum())
    if n == 0:
        return dict(ok=False, reason='NOT_COMPARABLE', n=0)
    d = float(np.abs(x[fx] - y[fy]).max())
    return dict(ok=bool(d <= atol), max_abs_diff=d, n=n)


# --------------------------------------------------------------------------- #
# Feature kernel + interface consistency
# --------------------------------------------------------------------------- #
def run_features(arrays: dict, code: str, days: list[int], quality: dict):
    return real_features_from_arrays(arrays, code, np.array(days, dtype=np.int64),
                                    quality)


def group_valid_mask(x5, n15: int) -> np.ndarray:
    """One mask per 15-min group: all THREE 5-min slots x ALL fields finite."""
    ok = np.ones(n15, dtype=bool)
    base = np.arange(n15) * 3
    for off in (0, 1, 2):
        idx = base + off
        for f in FIELDS:
            ok &= np.isfinite(x5[f][idx])
    return ok


def interface_consistency_test(arrays: dict, code: str, days: list[int],
                               quality: dict) -> dict:
    """Verify 5->15 aggregation and the real features_from_arrays output.

    Every expected array is built under the SAME group_valid mask, so a
    correctly absent 15-min bar is never reported as an inconsistency.
    """
    rows, _returns, _q = run_features(arrays, code, days, quality)
    x5, x15 = arrays[5], arrays[15]
    n15 = len(x15['close'])
    gv = group_valid_mask(x5, n15)

    def masked(expected):
        return np.where(gv, expected, np.nan)

    checks = {}
    checks['agg_open'] = compare(x15['open'], masked(x5['open'][0::3][:n15]))
    checks['agg_close'] = compare(x15['close'], masked(x5['close'][2::3][:n15]))
    checks['agg_high'] = compare(x15['high'], masked(np.maximum.reduce(
        [x5['high'][0::3][:n15], x5['high'][1::3][:n15], x5['high'][2::3][:n15]])))
    checks['agg_low'] = compare(x15['low'], masked(np.minimum.reduce(
        [x5['low'][0::3][:n15], x5['low'][1::3][:n15], x5['low'][2::3][:n15]])))
    checks['agg_volume'] = compare(x15['volume'], masked(
        x5['volume'][0::3][:n15] + x5['volume'][1::3][:n15]
        + x5['volume'][2::3][:n15]))
    # a group that is not fully valid must be entirely absent, never partial
    checks['incomplete_group_absent'] = compare(
        np.isfinite(x15['close']).astype(float), gv.astype(float), atol=0.0)

    keep = np.isfinite(x15['close'])
    checks['rows_match_kept_bars'] = compare(
        np.asarray(rows['cutoff_index'].to_numpy(), dtype=float),
        np.asarray(np.flatnonzero(keep), dtype=float), atol=0.0)

    # day return: the kernel only forms a reference from a day whose 16 bars
    # are ALL valid (not just the last bar's close)
    day_close = x15['close'].reshape(-1, 16)[:, -1].copy()
    day_close[~np.isfinite(x15['close'].reshape(-1, 16)).all(axis=1)] = np.nan
    ref = np.repeat(np.r_[np.nan, day_close[:-1]], 16)
    exp_ret = np.where(np.isfinite(ref), x15['close'] / ref - 1.0, np.nan)
    checks['day_return'] = compare(rows['stock_day_return_raw'].to_numpy(),
                                   exp_ret[keep])
    rsi15 = real_wilder_rsi(x15['close'])
    checks['rsi15'] = compare(rows['stock_rsi14_15m_raw'].to_numpy(), rsi15[keep])
    rsi5 = real_wilder_rsi(x5['close'])
    pick5 = np.arange(n15) * 3 + 2
    checks['rsi5'] = compare(rows['stock_rsi14_5m_raw'].to_numpy(), rsi5[pick5][keep])
    checks['intrabar15'] = compare(rows['stock_intrabar_return15m_raw'].to_numpy(),
                                   np.where(np.isfinite(x15['open']),
                                            x15['close'] / x15['open'] - 1.0,
                                            np.nan)[keep])

    not_comparable = sorted(k for k, v in checks.items()
                            if v.get('reason') == 'NOT_COMPARABLE')
    failing = sorted(k for k, v in checks.items()
                     if not v['ok'] and v.get('reason') != 'NOT_COMPARABLE')
    passed = (not failing) and (len(not_comparable) < len(checks))
    return dict(passed=passed, checks=checks, failing=failing,
                not_comparable=not_comparable,
                feature_rows=int(len(rows)), array_slots=int(n15),
                kept_bars=int(keep.sum()), group_valid=int(gv.sum()),
                note='kernel drops NaN-close rows; cutoff_index is preserved '
                     'and is the cross-symbol join key')


def build_quality(arrays: dict, anomaly: dict | None = None) -> dict:
    q = dict(conflicting_timestamps=0, invalid_or_excluded_rows=0,
             quarantine_days=0,
             feature_rows=int(np.isfinite(arrays[15]['close']).sum()))
    if anomaly:
        q['conflicting_timestamps'] = int(anomaly.get('duplicate_ts', 0)
                                          + anomaly.get('ts_inconsistent', 0)
                                          + anomaly.get('conflicting_ts', 0))
        q['invalid_or_excluded_rows'] = int(anomaly.get('bad_rows', 0)
                                            + anomaly.get('off_grid', 0)
                                            + anomaly.get('conflicting_ts', 0))
    return q
