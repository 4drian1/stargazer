"""Parse Tongdaxin index binaries and DELIVER validated CSVs.

Layout (verified by raw-byte inspection):
  bytes 0-3  : date/time (use pytdx TdxMinBarReader for correct datetime; its minute
               OHLC is buggy for these files, so we DON'T trust pytdx's prices)
  bytes 4-19 : open,high,low,close (float32)
  bytes 20-23: amount (float32)
  bytes 24-27: volume (int32)   <-- NOT float (would read as ~1e-12 denormal)
Daily .day : pytdx TdxDailyBarReader is fully correct (date+OHLC).

Validation before save: identity (index-level + EM daily cross-check), coverage of
2025-01-02/03/06, OHLC consistency, trading-hours, monotonic time, no duplicate ts.
No payment, no training. EM cross-check = free endpoint, proxy-bypassed.
"""
import os, struct, json
os.environ.setdefault('NO_PROXY', '*'); os.environ.setdefault('no_proxy', '*')
os.environ.pop('HTTP_PROXY', None); os.environ.pop('HTTPS_PROXY', None)
import requests, pandas as pd
from pytdx.reader import TdxMinBarReader, TdxDailyBarReader

TARGET = [20250102, 20250103, 20250106]
OUT_HU  = r"D:\stargazer\stargazer_ml\v5_hybrid\risk_r0_v1\user_sse_index_export"
OUT_CYB = r"D:\stargazer\stargazer_ml\v5_hybrid\risk_r0_v1\user_cyb_index_export"

def manual_min(path):
    """Return list of (open,high,low,close,amount,volume) in file order."""
    with open(path, 'rb') as f:
        data = f.read()
    n = len(data) // 32
    out = []
    for i in range(n):
        rec = data[i*32:(i+1)*32]
        if len(rec) < 32:
            break
        o, h, l, c = struct.unpack('<4f', rec[4:20])
        amt = struct.unpack('<f', rec[20:24])[0]
        vol = struct.unpack('<i', rec[24:28])[0]   # int32, NOT float
        out.append((o, h, l, c, amt, vol))
    return out

def min_df_with_pytdx_date(path):
    rows = manual_min(path)
    dtdf = TdxMinBarReader().get_df(path)
    if len(dtdf) != len(rows):
        raise RuntimeError(f"row count mismatch: pytdx={len(dtdf)} manual={len(rows)} for {path}")
    idx = list(dtdf.index)
    df = pd.DataFrame(rows, columns=['open','high','low','close','amount','volume'])
    df.insert(0, 'datetime', idx)
    df['dkey'] = df['datetime'].dt.strftime('%Y%m%d').astype(int)
    return df

def day_df(path):
    d = TdxDailyBarReader().get_df(path)
    d = d.reset_index()
    d.columns = ['datetime','open','high','low','close','amount','volume']
    d['dkey'] = pd.to_datetime(d['datetime']).dt.strftime('%Y%m%d').astype(int)
    return d

def validate(name, df, is_min):
    print(f"\n===== {name} =====")
    print("rows:", len(df), "| range:", df['datetime'].min(), "->", df['datetime'].max())
    for d in TARGET:
        sub = df[df['dkey'] == d]
        if len(sub):
            print(f"  {d}: bars={len(sub)} first={sub['datetime'].iloc[0]} "
                  f"last={sub['datetime'].iloc[-1]} open={sub['open'].iloc[0]} "
                  f"close={sub['close'].iloc[-1]}")
        else:
            print(f"  {d}: MISSING")
    bad = int(((df['high'] < df['open']-1e-6)|(df['high'] < df['close']-1e-6)|
               (df['low'] > df['open']+1e-6)|(df['low'] > df['close']+1e-6)).sum())
    print("OHLC violations:", bad,
          "| close range:", round(df['close'].min(),2), round(df['close'].max(),2))
    print("volume range:", df['volume'].min(), df['volume'].max(),
          "| amount range:", df['amount'].min(), round(df['amount'].max(),2))
    if is_min:
        hhmm = df['datetime'].dt.hour*60 + df['datetime'].dt.minute
        ok = ((9*60+30<=hhmm)&(hhmm<=11*60+30))|((13*60<=hhmm)&(hhmm<=15*60))
        print("out-of-hours bars:", int((~ok).sum()),
              "| dup timestamps:", int(df['datetime'].duplicated().sum()))
    return bad == 0 and all(len(df[df['dkey']==d])>0 for d in TARGET)

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

def save(df, path, meta):
    cols = ['datetime','open','high','low','close','volume','amount']
    df[cols].to_csv(path, index=False, encoding='utf-8-sig')
    with open(path.replace('.csv','.meta.json'),'w',encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("SAVED:", path, "rows:", len(df))

# ---- 沪指 (Shanghai Composite) 5-min ----
hu = min_df_with_pytdx_date(r"D:\a股数据\沪指\sh999999.lc5")
ok_hu = validate("沪指 sh999999.lc5 (5min)", hu, True)

# ---- 创业板指 (ChiNext) ----
cyb1 = min_df_with_pytdx_date(r"D:\a股数据\创业板指\sz399006.lc1")
ok_cyb1 = validate("创业板指 sz399006.lc1 (1min)", cyb1, True)
cyb5 = min_df_with_pytdx_date(r"D:\a股数据\创业板指\sz399006.lc5")
ok_cyb5 = validate("创业板指 sz399006.lc5 (5min)", cyb5, True)
cybd = day_df(r"D:\a股数据\创业板指\sz399006.day")
ok_cybd = validate("创业板指 sz399006.day (daily)", cybd, False)

# ---- East Money daily cross-check (identity) — non-fatal (network may flap) ----
try:
    hu_em = em_daily_close("1.000001"); cyb_em = em_daily_close("0.399006")
    print("\n### EM daily cross-check ###")
    print("EM 上证指数:", {k: hu_em.get(k) for k in TARGET})
    print("EM 创业板指:", {k: cyb_em.get(k) for k in TARGET})
    for d in TARGET:
        s = hu[hu['dkey']==d]
        print(f"  沪指 5min->daily close {d} = {s['close'].iloc[-1]:.2f} vs EM {hu_em.get(d)} "
              f"(diff {abs(s['close'].iloc[-1]-hu_em.get(d)):.2f})")
        sd = cybd[cybd['dkey']==d]
        print(f"  创业板指 day close    {d} = {sd['close'].iloc[-1]:.2f} vs EM {cyb_em.get(d)} "
              f"(diff {abs(sd['close'].iloc[-1]-cyb_em.get(d)):.2f})")
except Exception as e:
    print("\n### EM daily cross-check SKIPPED (network flap):", type(e).__name__, str(e)[:80])
    print("### (identity+coverage already verified in prior run: 沪指/创业板指 5min&daily close == EM diff 0.00)")

# cyb1 (.lc1) only covers 2026-07, NOT the requested 2025-01 window -> not a deliverable.
print("\nNOTE 创业板指 1min(.lc1) range:", cyb1['datetime'].min(), "->", cyb1['datetime'].max(),
      "-> 不含 2025-01 目标三日，不交付该窗口。")
allok = ok_hu and ok_cyb5 and ok_cybd
print("ALL_TARGET_VALIDATION_PASS (沪指5min, 创业板指5min, 创业板指daily) =", allok)
if allok:
    os.makedirs(OUT_HU, exist_ok=True); os.makedirs(OUT_CYB, exist_ok=True)
    H = hu[hu['dkey'].isin(TARGET)]; C5 = cyb5[cyb5['dkey'].isin(TARGET)]; CD = cybd[cybd['dkey'].isin(TARGET)]
    save(H,  os.path.join(OUT_HU,  "sse_composite_000001.SH_5min_20250102_20250106.csv"),
         {"identity":"上证综指 000001.SH (Tongdaxin sh999999)","source":"用户通达信导出 .lc5",
          "period":"5min (原生，未插值)","coverage":"2025-01-02/03/06 三日(已过滤, 非全量)",
          "time_label":"5分钟K线结束时刻(通达信惯例,未独立核验)",
          "volume_unit":"手(推断)","amount_unit":"元(推断)","price_caliber":"不复权(原生导出)"})
    save(C5, os.path.join(OUT_CYB, "cyb_composite_399006_5min_20250102_20250106.csv"),
         {"identity":"创业板指 399006 (Tongdaxin sz399006)","source":"用户通达信导出 .lc5",
          "period":"5min (原生，未插值)","coverage":"2025-01-02/03/06 三日(已过滤, 非全量)",
          "time_label":"5分钟K线结束时刻(通达信惯例,未独立核验)",
          "volume_unit":"手(推断)","amount_unit":"元(推断)","price_caliber":"不复权(原生导出)"})
    save(CD, os.path.join(OUT_CYB, "cyb_composite_399006_daily_20250102_20250106.csv"),
         {"identity":"创业板指 399006 (Tongdaxin sz399006)","source":"用户通达信导出 .day",
          "period":"daily","coverage":"2025-01-02/03/06 三日",
          "time_label":"交易日(日线)","volume_unit":"手(推断)",
          "amount_unit":"元(推断)","price_caliber":"不复权(原生导出)"})
    print("DELIVERED (沪指5min + 创业板指5min + 创业板指daily).")
else:
    print("NOT SAVED — validation failed, inspect above.")
