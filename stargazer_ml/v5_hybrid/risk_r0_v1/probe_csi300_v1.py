"""Confirm whether the CSI300 price index is available in local stockdb; enumerate index tables."""
import sys
import json
from pathlib import Path

sys.path.insert(0, r'D:\stargazer\stargazer_ml\v5_hybrid\step9_preflight')
from data_source_provenance import _http_raw, _decode

# universe as a list
raw = _http_raw('get', '股票代码', '*', '*')
obj = _decode(raw)
print('universe type', type(obj))
codes = set(str(x) for x in obj) if isinstance(obj, list) else set()
print('universe count', len(codes))
for c in ['000300', '399300', '510300', '000905', '399001', '399006', '000016',
          '000010', '000011', '000903', '000904', '000906']:
    print('  ', c, ('IN' if c in codes else 'out'))

# index tables listing
for tbl in ['指数k', 'idx', '指数', 'index']:
    try:
        r = _http_raw('get', tbl, '*', '*')
        o = _decode(r)
        n = len(o) if hasattr(o, '__len__') else 0
        print(f'get {tbl} * -> type={type(o).__name__} len={n}')
        if isinstance(o, list) and o:
            print('   first:', json.dumps(o[0], ensure_ascii=False)[:240])
    except Exception as e:
        print(f'get {tbl} * ERR {type(e).__name__}: {str(e)[:80]}')

# code-format variants for CSI300 index
for code in ['sh000300', '000300.SH', '000300.XSHG', 'hs300', '000300']:
    try:
        r = _http_raw('get', '日k', code, '*')
        o = _decode(r)
        n = len(o) if hasattr(o, '__len__') else 0
        print(f'日k:{code} -> {n} rows')
    except Exception as e:
        print(f'日k:{code} ERR {str(e)[:60]}')

# 510300 (沪深300ETF) sample to confirm usable as tracker
try:
    r = _http_raw('get', '日k', '510300', '*')
    o = _decode(r)
    rows = [v for v in o if isinstance(v, dict)] if isinstance(o, list) else []
    print('510300 rows', len(rows))
    if rows:
        print('510300 sample:', json.dumps(rows[0], ensure_ascii=False)[:240])
        print('510300 last:', json.dumps(rows[-1], ensure_ascii=False)[:240])
except Exception as e:
    print('510300 ERR', str(e)[:80])
print('DONE')
