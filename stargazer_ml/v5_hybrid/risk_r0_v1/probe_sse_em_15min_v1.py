"""Probe East Money free kline for SSE Composite (secid=1.000001) 15-min historical.
Does the free endpoint serve 2025-01 15-min (where 1-min only returned latest day)?
Read-only, free, no payment, no training. Proxy bypassed (dead Win system proxy).
"""
import os
os.environ.setdefault('NO_PROXY', '*'); os.environ.setdefault('no_proxy', '*')
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)
import requests, pandas as pd

SECID = "1.000001"
url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
TARGET = ["2025-01-02", "2025-01-03", "2025-01-06"]

for klt in ["15", "5", "60"]:
    params = {
        "fields1": "f1,f2,f3,f4,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": klt, "fqt": "0", "secid": SECID,
        "beg": "20250102", "end": "20250106",
    }
    s = requests.Session(); s.trust_env = False
    s.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"})
    last = None
    for attempt in range(4):
        try:
            j = s.get(url, params=params, timeout=30).json()
            break
        except Exception as e:
            last = e
            import time; time.sleep(1.5)
    else:
        print(f"klt={klt}: REQUEST FAILED after retries: {type(last).__name__} {str(last)[:100]}")
        continue
    data = j.get("data")
    if not data or not data.get("klines"):
        print(f"klt={klt}: NO DATA (rc={j.get('rc')})")
        continue
    klines = data["klines"]
    df = pd.DataFrame([ln.split(",") for ln in klines],
                      columns=["时间", "开盘", "收盘", "最高", "最低", "成交量", "成交额",
                               "振幅", "涨跌幅", "涨跌额", "换手率"])
    df["时间"] = pd.to_datetime(df["时间"])
    ts = df["时间"]
    print(f"klt={klt}: identity={data.get('name')} total={len(klines)} "
          f"range=[{ts.min()} -> {ts.max()}]")
    for d in TARGET:
        n = int(ts.dt.strftime('%Y-%m-%d').eq(d).sum())
        print(f"    {d}: bars={n}")
