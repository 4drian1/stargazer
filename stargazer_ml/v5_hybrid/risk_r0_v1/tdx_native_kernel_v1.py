"""Production native stock preparation kernel, promoted from route 18.177.
Pure source-derived features; no research-directory dependency or model fitting.
"""
import os,sys,json,hashlib,ctypes
from ctypes import wintypes
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from tdx_native_reader_v1 import read_lc5_file
from tdx_native_arrays_v1 import build_minute_arrays,run_features,build_quality
from tdx_native_indicators_v1 import compute_yesterday_indicators
P=E=CUTOFF=None
GRID=np.r_[np.arange(575,691,5),np.arange(785,901,5)]
def stamp():
    return int(datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d%H%M%S'))

def peak_memory():
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[
            (n,ctypes.c_size_t) for n in ('peak','working','qpp','qp','qnpp','qnp','pf','ppf')]
    c=Counters();c.cb=ctypes.sizeof(c)
    if not ctypes.WinDLL('psapi').GetProcessMemoryInfo(wintypes.HANDLE(-1),ctypes.byref(c),c.cb):
        raise OSError('PROCESS_MEMORY_QUERY_FAILED')
    return int(c.peak)

def memory_guard():
    if peak_memory()>512*1024**2:raise MemoryError('RAM_PEAK_OVER_512_MIB')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def good_days(rows):
    """Vectorized equivalent of validate_day: original within-day order retained."""
    if not rows:
        return set()
    days = np.array([r['day'] for r in rows], dtype=np.int64)
    ts = np.array([r['ts'] for r in rows], dtype=np.int64)
    minute = np.array([r['minute'] for r in rows], dtype=np.int64)
    values = np.array([[r[k] for k in ('open','high','low','close','amount','raw_volume')]
                       for r in rows], dtype=float)
    o,h,l,c,a,v = values.T
    ok = (np.isfinite(values).all(axis=1) & (values[:,:4] > 0).all(axis=1)
          & (a >= 0) & (v >= 0) & (l <= np.minimum(o,c)) & (h >= np.maximum(o,c))
          & np.isin(minute, GRID)
          & (ts == days*1000000 + (minute//60)*10000 + (minute%60)*100))
    order = np.argsort(days, kind='stable')
    d, t = days[order], ts[order]
    unique, counts = np.unique(d, return_counts=True)
    groups = np.repeat(np.arange(len(unique)), counts)
    bad = np.bincount(groups, weights=(~ok[order]).astype(int), minlength=len(unique)) > 0
    nonascending = (np.diff(d) == 0) & (np.diff(t) <= 0)
    if nonascending.any():
        bad[np.unique(groups[1:][nonascending])] = True
    return set(int(x) for x in unique[(counts == 48) & ~bad])

class NativeCache:
    def __init__(self, sources, days, prep):
        self.sources, self.days, self.prep = sources, days, prep
        self.indicators, self.metadata, self.tails = {}, {}, []
        self.returns5, self.returns15, self.return_codes = [], [], []

    def stock(self, code, days, cutoff, info=None):
        if list(days) != self.days or int(cutoff) != CUTOFF:
            raise ValueError('PREP_CALENDAR_OR_CUTOFF_MISMATCH')
        source = self.sources[code]
        if info is None:
            info = read_lc5_file(source['path'])
        if info['sha256'] != source['sha256']:
            raise ValueError('SOURCE_VERSION_CHANGED:' + code)
        raw = [r for r in info['rows'] if r['ts'] <= cutoff]
        by_day = {}
        for row in raw:
            by_day.setdefault(row['day'], []).append(row)
        allowed = good_days(raw)
        if P not in allowed:
            raise ValueError('P_DAY_NOT_COMPLETE:' + code)
        arrays, rejected = build_minute_arrays(by_day, days, allowed_days=allowed)
        rows, ret15, _ = run_features(arrays, code, days, build_quality(arrays, rejected))
        daily = arrays[15]['close'].reshape(-1, 16)
        closes = np.where(np.isfinite(daily).all(axis=1), daily[:,-1], np.nan)
        prev = np.r_[np.nan, closes[:-1]]
        refs = np.repeat(prev, 48)
        ret5 = np.divide(arrays[5]['close'], refs,
                         out=np.full(len(refs), np.nan), where=np.isfinite(refs)&(refs>0))-1
        ret = {5: np.where(np.isfinite(ret5)&(ret5>-1), ret5, np.nan),
               15: np.where(np.isfinite(ret15)&(ret15>-1), ret15, np.nan)}
        if code in self.prep:
            self.indicators[code] = compute_yesterday_indicators(
                arrays, days, calendar=days+[E], data_day=P, monitor_day=E)
        tail = rows[rows.cutoff_index//16 == len(days)-1].copy()
        self.tails.append(tail)
        self.returns5.append(ret[5][-48:].copy())
        self.returns15.append(ret[15][-16:].copy())
        self.return_codes.append(code)
        if sha(source['path']) != source['sha256']:
            raise ValueError('SOURCE_VERSION_CHANGED_POST:' + code)
        self.metadata[code] = dict(allowed_history_days=len(allowed),
                                  feature_rows=len(rows), rejected=rejected)
        if len(self.metadata) % 100 == 0:
            memory_guard()
            print(json.dumps(dict(stage='NATIVE_FEATURES',done=len(self.metadata),code=code,
                                  peak_memory=peak_memory())),flush=True)
        return rows, ret
