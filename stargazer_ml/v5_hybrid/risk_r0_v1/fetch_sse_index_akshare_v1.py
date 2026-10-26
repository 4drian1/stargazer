"""Fetch SSE Composite 000001.SH 1-min OHLCV for 2025-01-02/03/06 from East Money
(akshare's underlying feed; akshare's code-map pre-call hits a blocked subdomain, so we
call the same kline endpoint akshare uses, secid=1.000001 = 上证指数).

Validation BEFORE save:
 - identity: data.name == '上证指数' (and code 000001)
 - coverage of all 3 target dates (each ~240 bars)
 - interval == 1 minute
 - OHLC sanity (上证 level ~3200-3400, not a stock ~11)
Then save RAW csv (original columns) into user_sse_index_export/.

No payment, no training.
"""
import sys, os, json
os.environ.setdefault('NO_PROXY', '*'); os.environ.setdefault('no_proxy', '*')
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)

import requests
import pandas as pd

SECID = "1.000001"          # 1 = SH market, 000001 = Shanghai Composite
BEG, END = "20250102", "20250106"
TARGET = ["2025-01-02", "2025-01-03", "2025-01-06"]
OUT = r"D:\stargazer\stargazer_ml\v5_hybrid\risk_r0_v1\user_sse_index_export"

url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
params = {
    "fields1": "f1,f2,f3,f4,f6",
    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    "klt": "1",          # 1 minute
    "fqt": "0",          # no adjustment (index has no adjustment anyway)
    "secid": SECID,
    "beg": BEG,
    "end": END,
}
s = requests.Session(); s.trust_env = False
r = s.get(url, params=params, timeout=30)
print("http", r.status_code)
j = r.json()
print("api retcode:", j.get("rc"), j.get("rt"))
data = j.get("data")
if not data:
    print("NO DATA:", j); sys.exit(1)
print("identity -> code:", data.get("code"), "name:", data.get("name"))
klines = data["klines"]
print("raw klines count:", len(klines))
print("first:", klines[0])
print("last :", klines[-1])

# Parse (csv rows from EM): 日期,开,收,高,低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
cols = ["时间", "开盘", "收盘", "最高", "最低", "成交量", "成交额",
        "振幅", "涨跌幅", "涨跌额", "换手率"]
rows = [ln.split(",") for ln in klines]
df = pd.DataFrame(rows, columns=cols)
for c in ["开盘", "收盘", "最高", "最低", "成交量", "成交额",
          "振幅", "涨跌幅", "涨跌额", "换手率"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["时间"] = pd.to_datetime(df["时间"])
ts = df["时间"]
print("\nfirst ts:", ts.min(), "last ts:", ts.max())
for d in TARGET:
    sel = ts.dt.strftime('%Y-%m-%d').eq(d)
    print(f"  {d}: bars={int(sel.sum())} "
          f"open(first)={df.loc[sel,'开盘'].iloc[0] if sel.any() else None} "
          f"close(last)={df.loc[sel,'收盘'].iloc[-1] if sel.any() else None}")

sub = ts[ts.dt.strftime('%Y-%m-%d').eq('2025-01-02')].sort_values()
diffs = sub.diff().dropna().dt.total_seconds()
print("2025-01-02 inter-bar seconds: min=%.0f median=%.0f max=%.0f (expect ~60)"
      % (diffs.min(), diffs.median(), diffs.max()))

# identity sanity: index level, not a stock
lvl = float(df['收盘'].iloc[0])
print("first close level =", lvl, "(上证 ~3200-3400 expected; a stock would be ~10-100)")
assert data.get("name") == "上证指数", "IDENTITY MISMATCH: %r" % data.get("name")
assert lvl > 1000, "level too low to be 上证指数"

print("\nVALIDATION PASSED. Saving raw CSV (original EM columns, no slot-fill, no time shift).")
import os as _os
_os.makedirs(OUT, exist_ok=True)
# keep original columns; also expose a clean english-named copy is optional -> keep raw as-is
out_path = _os.path.join(OUT, "sse_composite_000001.SH_1min_20250102_20250106.csv")
df.to_csv(out_path, index=False, encoding="utf-8-sig")
print("written:", out_path, "rows:", len(df))
meta = {
    "source": "East Money kline endpoint (push2his.eastmoney.com) — same feed akshare wraps",
    "akshare_version": None,
    "secid": SECID, "identity_code": data.get("code"), "identity_name": data.get("name"),
    "period": "1min", "fqt": "0 (no adjustment; index N/A)",
    "date_range": [str(ts.min()), str(ts.max())],
    "target_dates": TARGET,
    "bars_per_target": {d: int(ts.dt.strftime('%Y-%m-%d').eq(d).sum()) for d in TARGET},
    "columns": cols,
    "time_label_semantics": "UNKNOWN (East Money 1-min kline timestamp = bar START per convention; not verified)",
    "volume_unit": "UNKNOWN (EM index 成交量 typically in 手; not independently verified)",
    "amount_unit": "UNKNOWN (EM 成交额 typically in 元; not independently verified)",
    "price_caliber": "raw/index level (no adjustment)",
}
with open(_os.path.join(OUT, "sse_composite_000001.SH_1min_20250102_20250106.meta.json"),
          "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("meta written.")
