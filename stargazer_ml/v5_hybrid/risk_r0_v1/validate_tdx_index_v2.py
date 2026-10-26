"""Validate Tongdaxin index binaries via pytdx (battle-tested parser). Read-only on input.
Checks identity (index-level + EM daily cross-check), coverage of 2025-01-02/03/06,
OHLC consistency, bar counts, trading-hours, monotonic time, non-zero volume/amount.
No writes, no payment, no training. (EM cross-check = free endpoint, proxy-bypassed.)
"""
import os
os.environ.setdefault('NO_PROXY', '*'); os.environ.setdefault('no_proxy', '*')
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)
import requests, pandas as pd
from pytdx.reader import TdxDailyBarReader, TdxMinBarReader

TARGET = [20250102, 20250103, 20250106]

def load_min(path):
    df = TdxMinBarReader().get_df(path)
    df['date'] = pd.to_datetime(df['date'])
    df['dkey'] = df['date'].dt.strftime('%Y%m%d').astype(int)
    return df

def load_day(path):
    df = TdxDailyBarReader().get_df(path)
    df['date'] = pd.to_datetime(df['date'])
    df['dkey'] = df['date'].dt.strftime('%Y%m%d').astype(int)
    return df

def report(name, df, is_min):
    print(f"\n===== {name} =====")
    print("rows:", len(df), "| cols:", list(df.columns))
    print("first:", df.iloc[0].to_dict())
    print("last :", df.iloc[-1].to_dict())
    print("date range:", df['date'].min(), "->", df['date'].max())
    for d in TARGET:
        sub = df[df['dkey'] == d]
        if len(sub):
            print(f"  {d}: bars={len(sub)} "
                  f"firstT={sub['date'].iloc[0]} lastT={sub['date'].iloc[-1]} "
                  f"open={sub['open'].iloc[0]} close={sub['close'].iloc[-1]}")
        else:
            print(f"  {d}: MISSING")
    # OHLC sanity
    bad = int(((df['high'] < df['open']-1e-6)|(df['high'] < df['close']-1e-6)|
               (df['low'] > df['open']+1e-6)|(df['low'] > df['close']+1e-6)).sum())
    print("OHLC violations:", bad)
    print("close min/max:", df['close'].min(), df['close'].max(), "(index-level expected)")
    print("volume min/max:", df['volume'].min(), df['volume'].max())
    print("amount min/max:", df['amount'].min(), df['amount'].max())
    if is_min:
        hhmm = df['date'].dt.hour*60 + df['date'].dt.minute
        ok = ((9*60+30 <= hhmm)&(hhmm <= 11*60+30))|((13*60 <= hhmm)&(hhmm <= 15*60))
        print("out-of-trading-hours bars:", int((~ok).sum()))
        print("dup timestamps:", int(df['date'].duplicated().sum()))

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
            d, c = ln.split(","); out[int(d.replace("-",""))] = float(c)
    return out

print("### 沪指 sh999999.lc5 (5-min) ###")
hu = load_min(r"D:\a股数据\沪指\sh999999.lc5"); report("沪指 5min", hu, True)
print("\n### 创业板指 sz399006 ###")
cyb1 = load_min(r"D:\a股数据\创业板指\sz399006.lc1"); report("创业板指 1min", cyb1, True)
cyb5 = load_min(r"D:\a股数据\创业板指\sz399006.lc5"); report("创业板指 5min", cyb5, True)
cybd = load_day(r"D:\a股数据\创业板指\sz399006.day"); report("创业板指 daily", cybd, False)

print("\n### East Money daily cross-check (identity) ###")
hu_em = em_daily_close("1.000001"); cyb_em = em_daily_close("0.399006")
print("EM 上证指数:", {k: hu_em.get(k) for k in TARGET})
print("EM 创业板指:", {k: cyb_em.get(k) for k in TARGET})
for d in TARGET:
    sub = hu[hu['dkey']==d]
    print(f"  沪指 5min-derived daily close {d} = {sub['close'].iloc[-1]:.2f} vs EM {hu_em.get(d)}")
    subd = cybd[cybd['dkey']==d]
    print(f"  创业板指 day-file close     {d} = {subd['close'].iloc[-1]:.2f} vs EM {cyb_em.get(d)}")
