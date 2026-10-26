"""Read-only exploration of the live local stockdb HTTP interface.

Confirms schemas for daily RAW bars and adjustment records, lists the instrument
universe, and attempts to locate the CSI300 index. No writes, no external network;
all traffic goes to 127.0.0.1:7899. Provenance-friendly (records host/query).
"""
import sys
import json
from pathlib import Path

STEP9 = r'D:\stargazer\stargazer_ml\v5_hybrid\step9_preflight'
sys.path.insert(0, STEP9)
from data_source_provenance import (get_daily_http, get_adjustment_http,
                                    _http_raw, _decode, CANONICAL_BASE)


def show(title, obj, n=3):
    print('====', title, '====')
    if hasattr(obj, 'columns'):
        print('cols:', list(obj.columns), 'rows:', len(obj))
        print(obj.head(n).to_string())
        print('...')
        print(obj.tail(n).to_string())
    elif isinstance(obj, list):
        print('len', len(obj))
        for x in obj[:n]:
            print(json.dumps(x, ensure_ascii=False)[:300])
    else:
        print(type(obj), str(obj)[:300])


# 1) daily RAW for a known stock
d = get_daily_http('000001')
show('000001 daily RAW', d, 3)

# 2) adjustment records
a = get_adjustment_http('000001')
print('==== 000001 adjustment events:', len(a), '====')
for x in a[:6]:
    print(json.dumps(x, ensure_ascii=False)[:300])

# 3) universe list + index-candidate scan
raw = _http_raw('get', '股票代码', '*', '*')
obj = _decode(raw)
print('==== universe type', type(obj), '====')
codes = []
if isinstance(obj, dict):
    for v in obj.values():
        if isinstance(v, list):
            codes.extend(str(x) for x in v)
elif isinstance(obj, list):
    codes.extend(str(x) for x in obj)
print('universe code count:', len(codes))
cand = ['000001', '000002', '000010', '000011', '000016', '000300', '000688',
        '000905', '399001', '399006', '399300', '510300', '510500']
for c in cand:
    print('  ', c, ('PRESENT' if c in codes else 'absent'))
# codes that begin with 0/3/5/6 but look index-ish (4-6 length bucketing)
from collections import Counter
pref = Counter(c[:3] for c in codes if c.isdigit())
print('prefix histogram (top):', pref.most_common(12))

# 4) try CSI300 via daily table across a historical anchor date
for code in ['000300', '399300', '510300', '000905']:
    dd = get_daily_http(code)
    print(f'get_daily 日k:{code} rows={len(dd)}',
          (f"min={int(dd['date'].min())} max={int(dd['date'].max())}" if hasattr(dd, 'columns') and len(dd) else ''))

# 5) try alternative index tables
for tbl in ['指数k', 'idx', '指数', 'index']:
    try:
        r = _http_raw('get', tbl, '000300', '*')
        o = _decode(r)
        print(f'get {tbl}:000300 ->', type(o).__name__,
              (len(o) if hasattr(o, '__len__') else ''))
    except Exception as e:
        print(f'get {tbl}:000300 ERR {type(e).__name__}: {str(e)[:80]}')

print('DONE probe')
