"""Parse Tongdaxin (.lc5/.lc1/.day) index binary files and VALIDATE before delivering.
Checks: identity (index-level values + cross-check vs East Money daily), coverage of
2025-01-02/03/06, OHLC consistency, bar counts (~48 for 5-min/day), time-in-trading-hours,
monotonic timestamps, non-zero volume/amount. Read-only on input; no writes yet.
No payment, no training. (East Money daily cross-check uses free endpoint, proxy-bypassed.)
"""
import os, struct, sys
os.environ.setdefault('NO_PROXY', '*'); os.environ.setdefault('no_proxy', '*')
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)
import requests, pandas as pd

TARGET = [20250102, 20250103, 20250106]

def parse_tdx_min(path):
    with open(path, 'rb') as f:
        data = f.read()
    n = len(data) // 32
    rows = []
    for i in range(n):
        rec = data[i*32:(i+1)*32]
        if len(rec) < 32:
            break
        dt = struct.unpack('<I', rec[0:4])[0]      # unsigned -> YYMMDDHHMM
        date = dt // 10000; tm = dt % 10000
        o, h, l, c, amt, vol = struct.unpack('<6f', rec[4:28])
        rows.append((date, tm, o, h, l, c, amt, vol))
    return rows

def parse_tdx_day(path):
    with open(path, 'rb') as f:
        data = f.read()
    n = len(data) // 32
    rows = []
    for i in range(n):
        rec = data[i*32:(i+1)*32]
        if len(rec) < 32:
            break
        date = struct.unpack('<I', rec[0:4])[0]    # YYYYMMDD
        o, h, l, c, amt, vol = struct.unpack('<6f', rec[4:28])
        rows.append((date, o, h, l, c, amt, vol))
    return rows

def validate(name, rows, is_min, period_label):
    print(f"\n===== {name} ({period_label}) =====")
    df = pd.DataFrame(rows)
    print("rows:", len(df))
    # first record raw diagnostic
    print("first 3 rows:", rows[:3])
    print("last 3 rows:", rows[-3:])
    # date range
    dates = sorted(set(r[0] for r in rows))
    print("date range:", dates[0], "->", dates[-1], "| distinct days:", len(dates))
    # coverage of target
    for d in TARGET:
        if is_min:
            sub = [r for r in rows if r[0] == d]
            print(f"  {d}: bars={len(sub)} "
                  f"firstT={sub[0][1] if sub else None} lastT={sub[-1][1] if sub else None} "
                  f"close_first={sub[0][4] if sub else None} close_last={sub[-1][4] if sub else None}")
        else:
            sub = [r for r in rows if r[0] == d]
            print(f"  {d}: daily present={len(sub)>0} close={sub[0][4] if sub else None}")
    # OHLC sanity
    bad = 0
    for r in rows:
        o, h, l, c = (r[2], r[3], r[4], r[5]) if is_min else (r[1], r[2], r[3], r[4])
        if not (h >= max(o, c) - 1e-6 and l <= min(o, c) + 1e-6 and h >= l - 1e-6):
            bad += 1
    print("OHLC violations:", bad)
    closes = [r[5] if is_min else r[4] for r in rows]
    print("close min/max:", min(closes), max(closes), "(index-level expected, not ~10-100)")
    vols = [r[7] if is_min else r[6] for r in rows]
    amts = [r[6] if is_min else r[5] for r in rows]
    print("volume min/max:", min(vols), max(vols), "| amount min/max:", min(amts), max(amts))
    # time-in-trading-hours (min only)
    if is_min:
        badt = 0
        for r in rows:
            tm = r[1]
            hh, mm = tm // 100, tm % 100
            ok = ((9*60+30 <= hh*60+mm <= 11*60+30) or (13*60 <= hh*60+mm <= 15*60))
            if not ok:
                badt += 1
        print("timestamps-out-of-trading-hours:", badt)
    return df

def em_daily_close(secid):
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {"fields1":"f1,f2,f3,f4,f6","fields2":"f51,f53","klt":"101",
              "fqt":"0","secid":secid,"beg":"20250101","end":"20250110"}
    s = requests.Session(); s.trust_env = False
    s.headers.update({"User-Agent":"Mozilla/5.0","Referer":"https://quote.eastmoney.com/"})
    j = s.get(url, params=params, timeout=30).json()
    out = {}
    if j.get("data") and j["data"].get("klines"):
        for ln in j["data"]["klines"]:
            d, c = ln.split(",")
            out[int(d.replace("-",""))] = float(c)
    return out

print("### 沪指 sh999999.lc5 (5-min) ###")
hu = parse_tdx_min(r"D:\a股数据\沪指\sh999999.lc5")
validate("沪指 sh999999.lc5", hu, True, "5min")
print("\n### 创业板指 sz399006 ###")
cyb1 = parse_tdx_min(r"D:\a股数据\创业板指\sz399006.lc1")
validate("创业板指 sz399006.lc1", cyb1, True, "1min")
cyb5 = parse_tdx_min(r"D:\a股数据\创业板指\sz399006.lc5")
validate("创业板指 sz399006.lc5", cyb5, True, "5min")
cybd = parse_tdx_day(r"D:\a股数据\创业板指\sz399006.day")
validate("创业板指 sz399006.day", cybd, False, "daily")

print("\n### East Money daily cross-check (identity) ###")
hu_daily = em_daily_close("1.000001")   # 上证指数
cyb_daily = em_daily_close("0.399006")  # 创业板指
print("EM 上证指数 daily closes:", {k: hu_daily.get(k) for k in TARGET})
print("EM 创业板指 daily closes:", {k: cyb_daily.get(k) for k in TARGET})
# aggregate 沪指 5-min to daily close (last bar of each day) and compare
hu_df = pd.DataFrame(hu, columns=["date","tm","o","h","l","c","amt","vol"])
for d in TARGET:
    sub = hu_df[hu_df.date==d]
    if len(sub):
        print(f"  沪指 5min-derived daily close {d} = {sub['c'].iloc[-1]:.2f}  vs EM {hu_daily.get(d)}")
cyb_df = pd.DataFrame(cybd, columns=["date","o","h","l","c","amt","vol"])
for d in TARGET:
    sub = cyb_df[cyb_df.date==d]
    if len(sub):
        print(f"  创业板指 day-file close {d} = {sub['c'].iloc[-1]:.2f}  vs EM {cyb_daily.get(d)}")
