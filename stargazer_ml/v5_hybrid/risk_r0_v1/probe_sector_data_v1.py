"""INVESTIGATE (read-only) sector-data availability before any download.
1) Local sector_daily.parquet: schema, sectors, coverage.
2) stockdb: does it hold SW-industry index K-lines (801xxx) for daily+5min?
   and does it expose per-stock industry membership (for member lists)?
3) Confirm individual-stock 5-min exists (to synthesize sector 5-min + breadth).
No full download. No writes. No payment/training.
"""
import os, sys, urllib.request, urllib.parse, msgpack, pandas as pd

# ---------- 1) local parquet ----------
print("===== LOCAL sector_daily.parquet =====")
P = r"D:\stargazer\stargazer_ml\pullback_train\sector_daily.parquet"
df = pd.read_parquet(P)
print("rows:", len(df), "| cols:", list(df.columns))
print(df.head(3).to_string())
sec_col = [c for c in df.columns if c.lower() in ('sector','name','板块','sector_name','主题','theme','index_name')]
print("guessed sector col:", sec_col)
if sec_col:
    sc = sec_col[0]
    print("n sectors:", df[sc].nunique(), "| values:", sorted(df[sc].unique().tolist()))
    g = df.groupby(sc)
    for name, sub in g:
        dts = pd.to_datetime(sub.iloc[:,0]) if not pd.api.types.is_datetime64_any_dtype(sub.iloc[:,0]) else sub.iloc[:,0]
        print(f"  {name}: rows={len(sub)} date {dts.min()} -> {dts.max()}")

# ---------- 2) stockdb probe ----------
print("\n===== stockdb probe =====")
BASE = 'http://127.0.0.1:7899'
def get(table, code='*', date='*'):
    t = urllib.parse.quote(f'{table}:{code}:{date}')
    with urllib.request.urlopen(f'{BASE}/?cmd=get&t={t}', timeout=60) as r:
        raw = r.read()
    return msgpack.unpackb(raw, raw=False, strict_map_key=False)

def nrows(obj):
    if isinstance(obj, list):
        return sum(1 for _,v in obj if isinstance(v, dict))
    return 0

# SW industry index K-lines (801xxx)
for code in ['801150','801080','801730','801890']:  # 医药生物/电子/电力设备/机械设备(示例)
    try:
        d = get('日k', code, '*')
        m = get('分钟k', code, '20250102*')
        print(f"  SW {code}: 日k rows={nrows(d)} | 分钟k@20250102={nrows(m)}")
    except Exception as e:
        print(f"  SW {code}: ERR {type(e).__name__} {str(e)[:80]}")

# individual stock 5-min (to synthesize sector lines)
for code in ['600276','300750','000858']:  # 恒瑞/宁德/五粮液 代表
    try:
        m = get('分钟k', code, '20250102*')
        print(f"  stock {code}: 分钟k@20250102 rows={nrows(m)}")
    except Exception as e:
        print(f"  stock {code}: ERR {type(e).__name__} {str(e)[:80]}")

# per-stock industry membership field
try:
    obj = get('股票代码')  # full code table
    if isinstance(obj, list) and obj:
        _, first = obj[0]
        if isinstance(first, dict):
            keys = list(first.keys())
            ind_keys = [k for k in keys if '行业' in k or 'industr' in k.lower() or 'index_code' in k.lower() or '板块' in k or 'sw' in k.lower()]
            print("  股票代码 table keys:", keys[:30])
            print("  industry-ish keys:", ind_keys)
            # show one sample
            print("  sample:", {k: first.get(k) for k in (ind_keys[:6] if ind_keys else keys[:6])})
except Exception as e:
    print("  股票代码 table ERR:", type(e).__name__, str(e)[:80])
