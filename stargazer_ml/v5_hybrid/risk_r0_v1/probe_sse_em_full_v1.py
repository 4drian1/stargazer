"""Probe East Money free kline: does it serve HISTORICAL 1-min for 2025-01 (not just latest day)?
Fetch with beg=0 (max history) and check whether target dates exist.
Read-only (no save). No payment, no training.
"""
import os
os.environ.setdefault('NO_PROXY', '*'); os.environ.setdefault('no_proxy', '*')
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)
import requests, pandas as pd

SECID = "1.000001"
url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
params = {
    "fields1": "f1,f2,f3,f4,f6",
    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    "klt": "1", "fqt": "0", "secid": SECID,
    "beg": "0", "end": "20500101",
}
s = requests.Session(); s.trust_env = False
j = s.get(url, params=params, timeout=30).json()
data = j.get("data")
klines = data["klines"]
print("identity:", data.get("code"), data.get("name"), "| total bars:", len(klines))
df = pd.DataFrame([ln.split(",") for ln in klines],
                  columns=["时间", "开盘", "收盘", "最高", "最低", "成交量", "成交额",
                           "振幅", "涨跌幅", "涨跌额", "换手率"])
df["时间"] = pd.to_datetime(df["时间"])
ts = df["时间"]
print("returned range:", ts.min(), "->", ts.max())
print("distinct dates:", sorted(ts.dt.strftime('%Y-%m-%d').unique())[:10], "...")
for d in ["2025-01-02", "2025-01-03", "2025-01-06"]:
    n = int(ts.dt.strftime('%Y-%m-%d').eq(d).sum())
    print(f"  {d}: bars={n}")
# how far back does it go?
print("earliest date:", ts.min())
