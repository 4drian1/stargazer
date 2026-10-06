# -*- coding: utf-8 -*-
"""T7 高位连涨耗散风险特征组（独立组，不改冻结的 T5，§18.32）。

背景（盘前可知，信号日 T 收盘及以前，数据=D:/a股数据 日线，与
d0_daily_features.py 同源同式）：
  runup_5d/10d/20d            收盘 n 日涨幅
  pos20/pos60                 (close-lowN)/(highN-lowN)
  rsi14                       同 D0 算法
  rsi_over75_10d              近10日 rsi14>75 的天数
  limit_up_10d/20d            近10/20日涨停天数（板别 lim：68=20%，30/8/4/92=30%，
                              主板=10%，is_st=5%；ret1>=lim-0.005）
  turnover_sum_5d/10d         累计换手率(%)
  turnover_max_5d             5日最大单日换手
  ht_down_10d                 近10日"高换手阴线"天数（ret1<0 且换手>1.5×20日中位）
  wide_range_10d              近10日宽振幅天数（TR>1.5×ATR14）
  atr14/atr60                 TR 均
  ma10/20_dist_atr            (close/maN-1)/atr_pct
  stage_low_high_20d          high20/low20-1（阶段低点到近期高点）
  days_since_high20           距20日高点天数
  churn_upper_shadow_5d       近5日"高换手长上影"天数（上影占比≥0.4 且高换手）

盘中（dp 截断，分钟 t<=dp*100，与引擎 dp_features 同约定）：
  gap_down_atr                (开盘价/pre_close-1)/atr_pct（dp日，全dp同值）
  below_vwap_min              截至dp位于分钟VWAP下方的时间(分钟)
  vwap_rebreak_flag           曾经站上VWAP、当前又跌破（0/1）
  vwap_rebreak_min            上次站上后又在下方的时间(分钟)
  reclaim_retention_945       仅dp1030：1 - 深度(1030)/深度(945)，深度=低于VWAP的ATR距离
  pm_below_vwap_frac          仅PM dp：13:00~dp 位于VWAP下方的时间占比
  am_best_vwap_dist_atr       仅PM dp：上午(945/1000/1030)最优(最接近/高于)VWAP距离
  recovery_decay_pm_atr       仅PM dp：当前距离 - 上午最优距离（>0=修复衰减）
  second_low_lower            dp>945：当日低点是否低于10:00前低点（0/1）

交互（在消融脚本内由上述列与 M0 shock_atr 组合生成 t7_ix_*）。
003040 只作 E0 案例审计，不参与阈值选择。
输出 t7_features_{year}.parquet，键 (code, dp_date, dp)，带 date(信号日)。
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
SAMPLES = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
OUT = os.path.join(OUT_DIR, "t7_features_2025.parquet")
DATA_GLOB = "D:/a股数据/data_daily_parts/part_*.parquet"
DPS = (945, 1000, 1030, 1330, 1400, 1430, 1450)
AM_DPS = (945, 1000, 1030)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minute_label_engine import load_code  # noqa: E402

BG = ("runup_5d", "runup_10d", "runup_20d", "pos20", "pos60", "rsi14",
      "rsi_over75_10d", "limit_up_10d", "limit_up_20d", "turnover_sum_5d",
      "turnover_sum_10d", "turnover_max_5d", "ht_down_10d",
      "wide_range_10d", "atr14", "atr60", "ma10_dist_atr", "ma20_dist_atr",
      "stage_low_high_20d", "days_since_high20", "churn_upper_shadow_5d")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def lim_of(code, is_st):
    if is_st:
        return 0.05
    if code.startswith("68"):
        return 0.20
    if code.startswith(("30", "8", "4", "92")):
        return 0.30
    return 0.10


def build_background(keys):
    """keys: DataFrame[code,date](信号日) → (code,date) 背景特征表。"""
    cols = ["date", "code", "open", "high", "low", "close", "pre_close",
            "volume", "amount", "turnover", "is_st"]
    frames = [pd.read_parquet(f, columns=cols)
              for f in sorted(glob.glob(DATA_GLOB))]
    df = pd.concat(frames, ignore_index=True)
    del frames
    df = df.dropna(subset=["close", "high", "low", "open"])
    df = df.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
    df["code"] = df["code"].astype(str)
    prev_close = df.groupby("code", sort=False)["close"].shift(1)
    df["pre_close"] = df["pre_close"].fillna(prev_close)
    df = df.dropna(subset=["pre_close"]).reset_index(drop=True)
    wl = df["code"].str[:2].isin(["00", "30", "60", "68"]) \
        | df["code"].str[:1].isin(["8", "4"]) | df["code"].str.startswith("92")
    codes = set(keys["code"])
    df = df[wl & df["code"].isin(codes)].reset_index(drop=True)
    log(f"daily loaded: {len(df):,} rows, {df['code'].nunique()} codes")
    for c in ["open", "high", "low", "close", "pre_close", "turnover"]:
        df[c] = df[c].astype(np.float32)
    df["ret1"] = df["close"] / df["pre_close"] - 1
    df["is_st"] = df["is_st"].fillna(0).astype(int)
    df["lim"] = [lim_of(c, s) for c, s in zip(df["code"], df["is_st"])]
    df["limit_up"] = df["ret1"] >= df["lim"] - 0.005

    g = df.groupby("code", sort=False)
    for n in (5, 10, 20):
        df[f"runup_{n}d"] = (df["close"] / g["close"].shift(n) - 1) \
            .astype(np.float32)
    for n in (20, 60):
        lo = g["low"].transform(lambda s, n=n: s.rolling(n, min_periods=n).min())
        hi = g["high"].transform(lambda s, n=n: s.rolling(n, min_periods=n).max())
        df[f"pos{n}"] = ((df["close"] - lo) / (hi - lo)).astype(np.float32)
    delta = g["close"].diff()
    up = delta.clip(lower=0).groupby(df["code"], sort=False) \
        .transform(lambda s: s.rolling(14).mean())
    dn = (-delta).clip(lower=0).groupby(df["code"], sort=False) \
        .transform(lambda s: s.rolling(14).mean())
    df["rsi14"] = (100 - 100 / (1 + up / dn.replace(0, np.nan))).astype(np.float32)
    df["rsi_over75_10d"] = (df["rsi14"] > 75).groupby(df["code"], sort=False) \
        .transform(lambda s: s.rolling(10).sum()).astype(np.float32)
    df["limit_up_10d"] = df["limit_up"].groupby(df["code"], sort=False) \
        .transform(lambda s: s.rolling(10).sum()).astype(np.float32)
    df["limit_up_20d"] = df["limit_up"].groupby(df["code"], sort=False) \
        .transform(lambda s: s.rolling(20).sum()).astype(np.float32)
    df["turnover_sum_5d"] = g["turnover"].transform(
        lambda s: s.rolling(5).sum()).astype(np.float32)
    df["turnover_sum_10d"] = g["turnover"].transform(
        lambda s: s.rolling(10).sum()).astype(np.float32)
    df["turnover_max_5d"] = g["turnover"].transform(
        lambda s: s.rolling(5).max()).astype(np.float32)
    t_med = g["turnover"].transform(
        lambda s: s.rolling(20, min_periods=10).median())
    ht = (df["turnover"] > 1.5 * t_med)
    df["ht_down_10d"] = ((df["ret1"] < 0) & ht).groupby(df["code"], sort=False) \
        .transform(lambda s: s.rolling(10).sum()).astype(np.float32)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["pre_close"]).abs(),
                    (df["low"] - df["pre_close"]).abs()], axis=1).max(axis=1)
    df["tr"] = tr.astype(np.float32)
    df["atr14"] = g["tr"].transform(
        lambda s: s.rolling(14, min_periods=10).mean()).astype(np.float32)
    df["atr60"] = g["tr"].transform(
        lambda s: s.rolling(60, min_periods=40).mean()).astype(np.float32)
    df["wide_range_10d"] = (df["tr"] > 1.5 * df["atr14"]) \
        .groupby(df["code"], sort=False) \
        .transform(lambda s: s.rolling(10).sum()).astype(np.float32)
    atr_pct = df["atr14"] / df["close"]
    for n in (10, 20):
        ma = g["close"].transform(lambda s, n=n: s.rolling(n).mean())
        df[f"ma{n}_dist_atr"] = ((df["close"] / ma - 1) / atr_pct) \
            .astype(np.float32)
    hi20 = g["high"].transform(lambda s: s.rolling(20, min_periods=20).max())
    lo20 = g["low"].transform(lambda s: s.rolling(20, min_periods=20).min())
    df["stage_low_high_20d"] = (hi20 / lo20 - 1).astype(np.float32)
    df["days_since_high20"] = g["high"].transform(
        lambda s: s.rolling(20, min_periods=20)
        .apply(lambda w: 19 - int(np.argmax(w)), raw=True)).astype(np.float32)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    upper_frac = ((df["high"] - np.maximum(df["open"], df["close"])) / rng)
    churn_shadow = (upper_frac >= 0.4) & ht
    df["churn_upper_shadow_5d"] = churn_shadow.groupby(df["code"], sort=False) \
        .transform(lambda s: s.rolling(5).sum()).astype(np.float32)

    out = df[["code", "date"] + list(BG)].copy()
    out["code"] = out["code"].astype(str)
    out["date"] = out["date"].astype(int)
    out = out.merge(keys, on=["code", "date"], how="inner")
    return out


def intraday_day(cm, day_i, atr_pct):
    """单日分钟一次走查，输出各 dp 的盘中特征 dict（dp→dict）。"""
    m = cm["day_idx"] == day_i
    t = cm["t"][m].astype(int)
    o = cm["o"][m].astype(float)
    h = cm["h"][m].astype(float)
    l = cm["l"][m].astype(float)
    c = cm["c"][m].astype(float)
    v = cm["v"][m].astype(float)
    a = cm["a"][m].astype(float)
    if not len(t):
        return {}
    order = np.argsort(t)
    t, o, h, l, c, v, a = (x[order] for x in (t, o, h, l, c, v, a))
    pc = float(cm["day_close"][day_i - 1]) if day_i > 0 else np.nan
    if not np.isfinite(pc) or pc <= 0 or not np.isfinite(atr_pct) \
            or atr_pct <= 0:
        return {}
    atr_px = pc * atr_pct
    day_open = float(o[0])
    cum_v = np.cumsum(v)
    cum_a = np.cumsum(a)
    with np.errstate(all="ignore"):
        vwap = np.where(cum_v > 0, cum_a / np.where(cum_v > 0, cum_v, np.nan),
                        np.nan)
    below = c < vwap
    # 站上→跌破状态机 + 下方分钟数
    below_min = np.zeros(len(t), dtype=int)
    rebreak_flag = np.zeros(len(t), dtype=int)
    rebreak_min = np.full(len(t), np.nan)
    was_below = below[0]
    reclaimed = False
    rb_start = -1
    cnt = 0
    for i in range(len(t)):
        if below[i]:
            cnt += 1
            if reclaimed:
                if rb_start < 0:
                    rb_start = i
                rebreak_flag[i] = 1
                rebreak_min[i] = i - rb_start
        else:
            reclaimed = True
            rb_start = -1
        below_min[i] = cnt
        was_below = below[i]
    dist = (c - vwap) / atr_px
    gap_down_atr = (day_open / pc - 1) / atr_pct
    res = {}
    am_dists = {}
    low_up_to_1000 = np.nan
    for dp in DPS:
        sel = t <= dp * 100
        if not sel.any():
            continue
        j = int(np.where(sel)[0][-1])
        f = {"gap_down_atr": float(gap_down_atr),
             "below_vwap_min": int(below_min[j]),
             "vwap_rebreak_flag": int(rebreak_flag[j]),
             "vwap_rebreak_min": float(rebreak_min[j])
             if np.isfinite(rebreak_min[j]) else np.nan,
             "vwap_dist_atr_dp": float(dist[j])}
        if dp in AM_DPS:
            am_dists[dp] = float(dist[j])
            if dp == 1000:
                low_up_to_1000 = float(l[:j + 1].min())
        if dp == 1030 and 945 in am_dists:
            d945 = -am_dists[945]          # >0 = 低于VWAP的深度
            d1030 = -float(dist[j])
            f["reclaim_retention_945"] = float(1 - d1030 / d945) \
                if d945 > 0 else np.nan
        if dp not in AM_DPS:
            pm = (t >= 130000) & sel
            f["pm_below_vwap_frac"] = float(below[pm].mean()) if pm.any() \
                else np.nan
            if am_dists:
                best = min(am_dists.values())
                f["am_best_vwap_dist_atr"] = float(best)
                f["recovery_decay_pm_atr"] = float(dist[j] - best)
        if dp > 945 and np.isfinite(low_up_to_1000):
            f["second_low_lower"] = int(float(l[:j + 1].min())
                                        < low_up_to_1000)
        res[dp] = f
    return res


def main():
    global SAMPLES, OUT
    args = sys.argv[1:]
    if "--samples" in args:
        SAMPLES = args[args.index("--samples") + 1]
    if "--out" in args:
        OUT = args[args.index("--out") + 1]
    t0 = time.time()
    sp = pd.read_parquet(SAMPLES, columns=["code", "date", "dp_date",
                                           "dp", "atr_pct", "entry_type"])
    sp = sp[sp["entry_type"] == "vwap"]
    keys = sp[["code", "date"]].drop_duplicates()
    keys["code"] = keys["code"].astype(str)
    keys["date"] = keys["date"].astype(int)
    log(f"信号键 (code,date): {len(keys):,}")

    bg = build_background(keys)
    log(f"背景特征行: {len(bg):,}")

    ev = (sp.groupby(["code", "dp_date"])["atr_pct"].first().reset_index())
    ev["code"] = ev["code"].astype(str)
    ev["dp_date"] = ev["dp_date"].astype(int)
    rows, miss = [], 0
    cur, cm = None, None
    for _, r in ev.iterrows():
        code = str(r["code"])
        if code != cur:
            cur = code
            cm = load_code(code)
        if cm is None:
            miss += 1
            continue
        hit = {int(d): i for i, d in enumerate(cm["days"])}
        ddate = int(r["dp_date"])
        if ddate not in hit:
            miss += 1
            continue
        fs = intraday_day(cm, hit[ddate], float(r["atr_pct"]))
        for dp, f in fs.items():
            rows.append({"code": code, "dp_date": ddate, "dp": dp, **f})
    log(f"盘中特征行: {len(rows):,}（miss {miss}）")
    ic = pd.DataFrame(rows)

    sig = (sp[["code", "date", "dp_date"]].drop_duplicates())
    sig["code"] = sig["code"].astype(str)
    sig["date"] = sig["date"].astype(int)
    sig["dp_date"] = sig["dp_date"].astype(int)
    out = ic.merge(sig, on=["code", "dp_date"], how="left")
    out = out.merge(bg, on=["code", "date"], how="left")
    out = out.drop_duplicates(subset=["code", "dp_date", "dp"], keep="first")
    out.to_parquet(OUT, index=False)
    feats = [c for c in out.columns if c.startswith("t7_") or c in BG
             or c in ("gap_down_atr", "below_vwap_min", "vwap_rebreak_flag",
                      "vwap_rebreak_min", "reclaim_retention_945",
                      "pm_below_vwap_frac", "am_best_vwap_dist_atr",
                      "recovery_decay_pm_atr", "second_low_lower",
                      "vwap_dist_atr_dp")]
    log(f"写入 {OUT}: {out.shape}；用时 {time.time()-t0:.0f}s")
    nan = out[feats].isna().mean().round(4)
    log("NaN率: " + ", ".join(f"{k}:{v}" for k, v in nan.items() if v > 0.01))


if __name__ == "__main__":
    main()
