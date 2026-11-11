"""Yesterday indicators for monitoring (v3) — route 18.176 fixes.

#1 (18.176) P / E / reference day are now EXPLICIT and distinct:
      * data_day    P = the day whose bars are the data cut-off;
      * monitor_day E = the day being monitored / prepared for;
      * reference_day = the day whose indicators are reported.
    NEXT_DAY_PREP (prepare E from P): reference = P, and prev(E) MUST equal P.
    REPLAY (replay an observation day O intraday): reference = prev(O), so a
    10/29 intraday replay legitimately uses 10/28 — but that is a DIFFERENT
    mode, never reached by naming a "market P day" and taking its predecessor.

Monitoring periods RSI6/RSI15/RSI14 (P1-E of 18.175) and per-point warmup
eligibility (P1-D) are retained. 26/34 bars are MINIMUM OBSERVATION THRESHOLDS
from the monitoring contract, not a proof of EMA convergence.
"""
from __future__ import annotations

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

WARMUP_NOTE = ('26/34 bars are minimum observation thresholds of the monitoring '
               'contract, not a proof of EMA convergence.')
PERIODS = (6, 14, 15)          # 6/15 = monitoring; 14 = Pyxis (kept separate)


# --------------------------------------------------------------------------- #
# P / E / reference-day resolution
# --------------------------------------------------------------------------- #
def prev_trading_day(calendar, day: int):
    """Latest canonical trading day strictly before `day` (None if none)."""
    prior = [d for d in calendar if d < day]
    return max(prior) if prior else None


def next_trading_day(calendar, day: int):
    earliest = [d for d in calendar if d > day]
    return min(earliest) if earliest else None


def resolve_reference_day(calendar, data_day=None, monitor_day=None,
                          observation_day=None, reference_day=None):
    """Resolve the reference day EXPLICITLY. Never infer one meaning from a
    field name that could mean two things.

    NEXT_DAY_PREP  data_day=P, monitor_day=E          -> reference = P
                   (requires prev(E) == P, else P_E_NOT_ADJACENT)
    REPLAY         observation_day=O                  -> reference = prev(O)
    EXPLICIT       reference_day given directly       -> reference = that
    """
    out = dict(mode=None, data_day=data_day, monitor_day=monitor_day,
               observation_day=observation_day, reference_day=None,
               status=None, reason=None)
    if reference_day is not None:
        out.update(mode='EXPLICIT', reference_day=int(reference_day),
                   status='OK')
        return out
    if observation_day is not None:
        # Intraday replay of day O: "yesterday" is the day before O.
        ref = prev_trading_day(calendar, observation_day)
        out.update(mode='REPLAY', reference_day=ref,
                   status='OK' if ref is not None else 'NO_PREVIOUS_TRADING_DAY')
        if ref is None:
            out['reason'] = 'no canonical trading day before %s' % observation_day
        return out
    if data_day is not None and monitor_day is not None:
        ref_expect = int(data_day)
        prev_of_e = prev_trading_day(calendar, monitor_day)
        out.update(mode='NEXT_DAY_PREP', reference_day=ref_expect)
        if prev_of_e != ref_expect:
            out.update(status='P_E_NOT_ADJACENT',
                       reason=('prev(%s)=%s != data_day %s; E and P are not '
                               'adjacent trading days' % (monitor_day, prev_of_e,
                                                          ref_expect)))
        else:
            out['status'] = 'OK'
        return out
    out['status'] = 'NO_P_E_DECLARED'
    out['reason'] = ('declare data_day+monitor_day (next-day prep) or '
                     'observation_day (replay)')
    return out


# --------------------------------------------------------------------------- #
# Wilder RSI (matches monitoring kernel golden_ladder_v3._rsi)
# --------------------------------------------------------------------------- #
def wilder_rsi(values, n: int):
    x = np.asarray(values, dtype=float)
    out = np.full(len(x), np.nan)
    edges = np.diff(np.r_[False, np.isfinite(x), False].astype(int))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if e - s <= n:
            continue
        d = np.diff(x[s:e])
        u, v = np.maximum(d, 0.0), np.maximum(-d, 0.0)
        u = pd.Series(np.r_[u[:n].mean(), u[n:]]).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
        v = pd.Series(np.r_[v[:n].mean(), v[n:]]).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
        out[s + n:e] = np.divide(100 * u, u + v, out=np.full_like(u, 50), where=u + v > 0)
    return out


def _runs(x):
    valid = np.isfinite(np.asarray(x, dtype=float))
    edges = np.diff(np.r_[False, valid, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _ema(seg, span):
    if pd is not None:
        return pd.Series(seg).ewm(span=span, adjust=False).mean().to_numpy()
    alpha = 2.0 / (span + 1)
    out, prev = np.empty(len(seg)), seg[0]
    for i, val in enumerate(seg):
        prev = alpha * val + (1 - alpha) * prev
        out[i] = prev
    return out


def macd(close, fast=12, slow=26, signal=9):
    """Segmented DIF/DEA; unconverged prefix masked. Thresholds match
    strong_low_indicators_v1: dif from index 25, dea from index 33."""
    x = np.asarray(close, dtype=float)
    dif = np.full(len(x), np.nan)
    dea = np.full(len(x), np.nan)
    for s, e in _runs(x):
        seg = x[s:e]
        d = _ema(seg, fast) - _ema(seg, slow)
        dif[s:e] = d
        dea[s:e] = _ema(d, signal)
        dif[s:min(e, s + slow - 1)] = np.nan
        dea[s:min(e, s + slow + signal - 2)] = np.nan
    return dif, dea


def _first_valid(x):
    v = np.flatnonzero(np.isfinite(np.asarray(x, dtype=float)))
    return int(v[0]) if v.size else -1


def _num(v):
    return float(v) if v is not None and np.isfinite(v) else None


# --------------------------------------------------------------------------- #
# Per-frequency snapshot
# --------------------------------------------------------------------------- #
def compute_for_frequency(close, lows, ts, day_arr, freq_label: str,
                          calendar=None, data_day=None, monitor_day=None,
                          observation_day=None, reference_day=None,
                          slow=26, signal=9):
    """Indicator snapshot for the resolved reference day."""
    per = 48 if freq_label == '5m' else 16
    _fin = np.isfinite(np.asarray(close, dtype=float))
    _da = np.asarray(day_arr)

    res = resolve_reference_day(calendar, data_day=data_day,
                                monitor_day=monitor_day,
                                observation_day=observation_day,
                                reference_day=reference_day)
    out = dict(freq=freq_label, mode=res['mode'],
               data_day=res['data_day'], monitor_day=res['monitor_day'],
               observation_day=res['observation_day'],
               reference_day=res['reference_day'],
               # file_last_day = last day that actually carries data
               file_last_day=int(_da[_fin].max()) if _fin.any() else None,
               calendar_last_day=int(max(calendar)) if calendar else None,
               status=res['status'], reason=res['reason'])

    rsi = {p: wilder_rsi(close, p) for p in PERIODS}
    dif, dea = macd(close, slow=slow, signal=signal)
    out['warmup'] = dict(
        rsi6_bars=_first_valid(rsi[6]), rsi14_bars=_first_valid(rsi[14]),
        rsi15_bars=_first_valid(rsi[15]), dif_bars=_first_valid(dif),
        dea_bars=_first_valid(dea), note=WARMUP_NOTE)

    if res['status'] != 'OK':
        return out
    yday = res['reference_day']
    idxs = np.flatnonzero(_da == yday)
    if len(idxs) != per:
        out['status'] = 'WAIT_HISTORY'
        out['reason'] = 'reference day absent from the shared calendar slice'
        return out
    if out['file_last_day'] is not None and out['file_last_day'] < yday:
        out['status'] = 'STALE'
        out['reason'] = ('file ends %s, before reference day %s; not usable as '
                         'market yesterday' % (out['file_last_day'], yday))
        return out
    if not _fin[idxs].all():
        out['status'] = 'INCOMPLETE_DAY'
        out['reason'] = 'reference day has missing bars'
        return out

    last_i = int(idxs[-1])
    low_i = int(idxs[int(np.argmin(np.asarray(lows)[idxs]))])

    def point(i, kind):
        vals = dict(rsi6=_num(rsi[6][i]), rsi14=_num(rsi[14][i]),
                    rsi15=_num(rsi[15][i]), dif=_num(dif[i]), dea=_num(dea[i]))
        vals['macd_hist'] = (2 * (vals['dif'] - vals['dea'])
                             if vals['dif'] is not None and vals['dea'] is not None
                             else None)
        monitoring = all(vals[k] is not None for k in ('rsi6', 'rsi15', 'dif'))
        full = monitoring and vals['rsi14'] is not None and vals['dea'] is not None
        return dict(kind=kind, ts=int(np.asarray(ts)[i]),
                    close=_num(np.asarray(close)[i]), low=_num(np.asarray(lows)[i]),
                    eligible=full, monitoring_eligible=monitoring,
                    missing=[k for k in ('rsi6', 'rsi14', 'rsi15', 'dif', 'dea')
                             if vals[k] is None], **vals)

    out['status'] = 'OK'
    out['yesterday_last_bar'] = point(last_i, 'last_bar')
    out['yesterday_low_bar'] = point(low_i, 'low_bar')
    return out


def compute_yesterday_indicators(arrays, days, calendar=None, data_day=None,
                                 monitor_day=None, observation_day=None,
                                 reference_day=None):
    out = {}
    for step, label in ((5, '5m'), (15, '15m')):
        a = arrays[step]
        day_arr = np.repeat(np.asarray(days, dtype=np.int64),
                            48 if step == 5 else 16)
        out[label] = compute_for_frequency(
            a['close'], a['low'], _build_ts(days, step), day_arr, label,
            calendar=calendar, data_day=data_day, monitor_day=monitor_day,
            observation_day=observation_day, reference_day=reference_day)
    return out


def _build_ts(days, step):
    from datetime import datetime, timedelta
    out = []
    for d in days:
        base = datetime.strptime(str(d), '%Y%m%d')
        for hour, start in ((9, 30), (13, 0)):
            for i in range(step, 121, step):
                t = base.replace(hour=hour) + timedelta(minutes=start + i)
                out.append(int(t.strftime('%Y%m%d%H%M%S')))
    return np.array(out, dtype=np.int64)
