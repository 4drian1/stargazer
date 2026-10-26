"""Acquire a CSI300 market-index source from the local stockdb (roadmap Blocker 3).

The pure CSI300 price index (000300 / 399300) is ABSENT from this stockdb instance
(verified: 0 rows in 日k and in index tables). The genuine CSI300 tracker available
is the 沪深300ETF (510300). Tracking error and distributions require separate handling.
This is NOT a stock-median proxy (forbidden by contract); it is a real CSI300
instrument. It is bound with HONEST identity (SH510300 / CSI300_ETF_TRACKER), not
falsely as the official index (SH000300 / CSI300_PRICE_INDEX). No automatic substitution.

Read-only against 127.0.0.1:7899. Writes a CSV + a .binding.json carrying identity,
source and content hash.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

STEP9 = r'D:\stargazer\stargazer_ml\v5_hybrid\step9_preflight'
if STEP9 not in sys.path:
    sys.path.insert(0, STEP9)
from data_source_provenance import _http_raw, _decode, CANONICAL_BASE
from stockdb_source_v1 import iter_pairs

HERE = Path(__file__).resolve().parent
OUT = HERE / 'historical_stock_day_preparation_v1'
PURE_INDEX = ['000300', '399300']
ETF_TRACKER = '510300'
CSV_NAME = 'csi300_tracker_510300.csv'


def _fetch_rows(code):
    raw = _http_raw('get', '日k', code.zfill(6), '*')
    obj = _decode(raw)
    return [v for _k, v in iter_pairs(obj) if isinstance(v, dict)]


def acquire(allow_etf_tracker=False):
    OUT.mkdir(exist_ok=True)
    # 1) try the pure index first (honest preferred identity)
    for code in PURE_INDEX:
        rows = _fetch_rows(code)
        if rows:
            return _emit(code, rows, semantics='CSI300_PRICE_INDEX',
                         symbol=f'SH{code}', pure=True)
    if not allow_etf_tracker:
        raise RuntimeError('PURE_INDEX_UNAVAILABLE; ETF requires explicit separate scope and distribution adjustment')
    # Raw tracker acquisition is diagnostic only, never an index admission.
    rows = _fetch_rows(ETF_TRACKER)
    if not rows:
        raise RuntimeError('no CSI300 index or ETF tracker available in stockdb')
    return _emit(ETF_TRACKER, rows, semantics='CSI300_ETF_TRACKER',
                 symbol=f'SH{ETF_TRACKER}', pure=False)


def _emit(code, rows, semantics, symbol, pure):
    out_rows = []
    for r in rows:
        try:
            d = int(r['date'])
            c = float(r['close'])
            if not (isinstance(c, float) and c > 0 and __import__('math').isfinite(c)):
                continue
            out_rows.append((d, c))
        except (KeyError, TypeError, ValueError):
            continue
    out_rows.sort()
    import io
    buffer=io.StringIO(newline='')
    with __import__('contextlib').nullcontext(buffer) as f:
        w = csv.writer(f)
        w.writerow(['date', 'close'])
        for d, c in out_rows:
            w.writerow([d, c])
    data=buffer.getvalue().encode('utf-8')
    digest = hashlib.sha256(data).hexdigest()
    stamp=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    csv_path=OUT/f'csi300_{code}_{stamp}.csv'
    binding = dict(
        schema='CSI300_INDEX_BINDING_V1',
        symbol=symbol,
        semantics=semantics,
        adjustment='NONE',
        source=f'local stockdb 日k:{code} (base={CANONICAL_BASE})',
        is_pure_index=pure,
        note=('Official CSI300 price index.' if pure else
              '沪深300ETF close as CSI300 tracker. Pure index 000300/399300 absent '
              'from this stockdb instance; ETF is a genuine CSI300 instrument, NOT a '
              'stock-median proxy. Replaces SH000300 only because the pure index is '
              'unavailable here.'),
        rows=len(out_rows),
        date_min=out_rows[0][0] if out_rows else None,
        date_max=out_rows[-1][0] if out_rows else None,
        sha256=digest,
        retrieved_at=__import__('datetime').datetime.now(
            __import__('datetime').timezone.utc).isoformat(timespec='seconds'))
    sidecar=json.dumps(binding,ensure_ascii=False,indent=1).encode('utf-8')
    from prepare_historical_risk_v1 import bytes_used
    if len(data)+len(sidecar)>2*1024**2 or bytes_used()+len(data)+len(sidecar)>128*1024**2:
        raise ValueError('REGISTERED_TOTAL_BUDGET')
    with csv_path.open('xb') as f: f.write(data)
    with csv_path.with_name(csv_path.name+'.binding.json').open('xb') as f: f.write(sidecar)
    print(f'acquired {symbol} ({semantics}) rows={len(out_rows)} '
          f'{binding["date_min"]}..{binding["date_max"]} sha256={digest[:12]}..')
    print(f'wrote {csv_path.name} + {csv_path.name}.binding.json')
    return binding


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--etf-tracker',action='store_true',help='diagnostic acquisition only; not index admission')
    acquire(allow_etf_tracker=parser.parse_args().etf_tracker)
