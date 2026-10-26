"""Unified RAW input reader from the local stockdb HTTP interface (roadmap Blocker 1 input).

Reads RAW (unadjusted) daily OHLC and adjustment (复权) records, builds a per-date
corporate-action factor series, and records provenance. This is the SINGLE storage
caliber for historical label/feature construction: stockdb HTTP @ 127.0.0.1:7899.

Contract-critical rules enforced here:
- RAW OHLC only. `pre_close` is NEVER used as the raw previous close; the prior-day
  close is derived from the series itself (see historical_risk_core_v1).
- Normalization factor(d) = product of positive `mult` at recorded dates <= d;
  stored `cum` is diagnostic only. Effective dates require independent verification. ratio uses
  raw_price(d)*factor(d)/factor(P), so corporate actions between P and d cancel.
- All access is read-only; no external network, no writes to stockdb.

stockdb `get` returns a list of [key, rowdict] pairs (msgpack/json); `iter_pairs`
normalizes that into (key, rowdict) regardless of transport encoding.
"""
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STEP9 = r'D:\stargazer\stargazer_ml\v5_hybrid\step9_preflight'
if STEP9 not in sys.path:
    sys.path.insert(0, STEP9)
from data_source_provenance import (_http_raw, _decode, CANONICAL_BASE,
                                    CANONICAL_HOST, CANONICAL_PORT)

OHLC = ('open', 'high', 'low', 'close')
RETRY = 4
BACKOFF = 1.0


def provenance():
    return dict(source='local_stockdb_http', base_url=CANONICAL_BASE,
                host=CANONICAL_HOST, port=CANONICAL_PORT,
                adapter='http(cmd=get|vals) free-stockdb v0.3.2',
                price_scale='UNVERIFIED_STOCKDB_STORED_OHLC',
                note='No SDK adjustment applied; RAW/effective dates/coverage require evidence; pre_close not raw prev close')


def _fetch(table, code, date='*', timeout=180):
    """Return decoded stockdb response (list of [key, rowdict] pairs)."""
    last = None
    for attempt in range(RETRY):
        try:
            raw = _http_raw('get', table, str(code).zfill(6), date, timeout=timeout)
            return _decode(raw)
        except Exception as exc:  # transient empty/flaky responses
            last = exc
            time.sleep(BACKOFF * (attempt + 1))
    raise RuntimeError(f'stockdb fetch failed {table}:{code}:{date} -> {last}')


def iter_pairs(obj):
    """Yield (key, rowdict) from a stockdb `get` payload (list of pairs or dict)."""
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                yield item[0], item[1]
            elif isinstance(item, dict):
                yield None, item
    elif isinstance(obj, dict):
        if 'date' in obj or 'mult' in obj:
            yield None,obj
            return
        for k, v in obj.items():
            yield k, v


def read_daily_raw(code):
    """Return (rows, dropped). rows = list of {date(int), open, high, low, close} sorted.

    Only finite/positive OHLC kept; `pre_close` is intentionally ignored.
    """
    obj = _fetch('日k', code)
    out, dropped = [], 0
    seen={}
    for _k, v in iter_pairs(obj):
        if not isinstance(v, dict):
            continue
        try:
            if v.get('code')!=str(code).zfill(6): raise ValueError('stock code mismatch')
            if any(isinstance(v.get(k),bool) for k in OHLC): raise ValueError('boolean price')
            d = int(v['date'])
            rec = dict(date=d, open=float(v['open']), high=float(v['high']),
                       low=float(v['low']), close=float(v['close']))
        except (KeyError, TypeError, ValueError):
            dropped += 1
            continue
        if not all(val > 0 and math.isfinite(val) for val in rec.values()):
            dropped += 1
            continue
        if d in seen and seen[d]!=rec: raise ValueError('CONFLICTING_DAILY_DATE')
        if d not in seen: out.append(rec)
        seen[d]=rec
    out.sort(key=lambda x: x['date'])
    return out, dropped


def read_adjustment(code):
    """Return list of {date(int), div, give, trans, mult, cum} sorted ascending."""
    obj = _fetch('复权', code)
    out = []
    for k, v in iter_pairs(obj):
        if not isinstance(v, dict):
            continue
        # key format: '复权:CODE:YYYYMMDD' (or a bare date)
        date_part = str(k).split(':')[-1] if isinstance(k, str) else None
        try:
            d = int(date_part)
            # Missing/zero/NaN mult is not an identity corporate action.
            mult=v['mult']
            if isinstance(mult,bool): raise ValueError('boolean multiplier')
            mult=float(mult)
            if not math.isfinite(mult) or mult<=0: raise ValueError('invalid multiplier')
            out.append(dict(date=d,
                            div=float(v.get('div') or 0),
                            give=float(v.get('give') or 0),
                            trans=float(v.get('trans') or 0),
                            mult=mult,
                            cum=v.get('cum')))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'INVALID_ADJUSTMENT_RECORD:{code}:{k}') from exc
    out.sort(key=lambda x: x['date'])
    return out


def build_factor_series(adjustments, dates):
    """Canonical product(mult), never cum. This alone does not verify effective dates."""
    from stockdb_adjustment_v2 import rebuild_mult_factors
    return rebuild_mult_factors(adjustments,dates)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_provenance(path):
    Path(path).write_text(json.dumps(provenance(), ensure_ascii=False, indent=1),
                          encoding='utf-8')
