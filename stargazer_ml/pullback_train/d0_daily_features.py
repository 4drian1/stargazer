# -*- coding: utf-8 -*-
"""D0 日线背景特征（v3 冻结方案 §6.6-1，§18.21）。

给盘中双头（G-SHOCK）加入盘后可知的日线背景：全部用信号日 T 收盘及以前
数据，T 收盘后产出候选 → 无前视。连续软特征，不设 RSI/BOLL/量能硬门槛。

复用定义（与 build_events_train_v1.py / build_event_funnel.py 完全同式）：
  rsi14/rsi_delta3/bbi_distance/bbi_slope5/boll_pct_b/
  down_volume_ratio5/turnover_ratio20/turnover_pct60/
  pos20/pullback_from_high5/adv20/amount_ratio20/atr14/atr_ref
新增（v3）：
  rsi6             6日RSI（同 rsi14 算法，窗口6）
  rsi6_minus_rsi14 rsi6 - rsi14（短期动能背离）
  boll_width20_atr BOLL宽(4*std20)/atr14，ATR单位

输出：d0_daily_features_{year}.parquet，主键 (code, date=信号日T)。
atr_ref 与候选池 barrier 尺度一致（shift(1).rolling(5).median()）。
幂等；--samples/--out 跑 2025 与 2026。
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
SAMPLES = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
OUT = os.path.join(OUT_DIR, "d0_daily_features_2025.parquet")
DATA_GLOB = "D:/a股数据/data_daily_parts/part_*.parquet"
FEATS = ("rsi14", "rsi_delta3", "bbi_distance", "bbi_slope5", "boll_pct_b",
         "down_volume_ratio5", "turnover_ratio20", "turnover_pct60",
         "pos20", "pullback_from_high5", "adv20", "amount_ratio20",
         "atr14", "atr_ref", "rsi6", "rsi6_minus_rsi14", "boll_width20_atr")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    global SAMPLES, OUT
    args = sys.argv[1:]
    if "--samples" in args:
        SAMPLES = args[args.index("--samples") + 1]
    if "--out" in args:
        OUT = args[args.index("--out") + 1]
    t0 = time.time()
    keys = (pd.read_parquet(SAMPLES, columns=["code", "date"])
            .drop_duplicates())
    keys["code"] = keys["code"].astype(str)
    keys["date"] = keys["date"].astype(int)
    codes = set(keys["code"])
    log(f"样本唯一 (code,date): {len(keys):,}，涉及 {len(codes):,} 只")

    cols = ["date", "code", "open", "high", "low", "close", "pre_close",
            "volume", "amount", "turnover", "float_mv", "is_st"]
    files = sorted(glob.glob(DATA_GLOB))
    frames = [pd.read_parquet(f, columns=cols) for f in files]
    df = pd.concat(frames, ignore_index=True)
    del frames
    df = df.dropna(subset=["close", "high", "low", "open"])
    df = df.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
    prev_close = df.groupby("code", sort=False)["close"].shift(1)
    df["pre_close"] = df["pre_close"].fillna(prev_close)
    df = df.dropna(subset=["pre_close"]).reset_index(drop=True)
    p1, p2 = df["code"].str[:1], df["code"].str[:2]
    wl = p2.isin(["00", "30", "60", "68"]) | p1.isin(["8", "4"]) \
        | df["code"].str.startswith("92")
    df = df[wl & df["code"].isin(codes)].reset_index(drop=True)
    log(f"daily loaded: {len(df):,} rows, {df['code'].nunique()} codes "
        f"({df['date'].min()}~{df['date'].max()})")
    for c in ["open", "high", "low", "close", "pre_close", "turnover"]:
        df[c] = df[c].astype(np.float32)
    df["ret1"] = df["close"] / df["pre_close"] - 1

    g = df.groupby("code", sort=False)
    # ---- 与 build_event_funnel.add_features 同式 ----
    df["high5"] = g["high"].transform(lambda s: s.rolling(5).max())
    df["low20"] = g["low"].transform(lambda s: s.rolling(20).min())
    df["high20"] = g["high"].transform(lambda s: s.rolling(20).max())
    df["pos20"] = ((df["close"] - df["low20"])
                   / (df["high20"] - df["low20"])).astype(np.float32)
    df["pullback_from_high5"] = (df["close"] / df["high5"] - 1).astype(np.float32)
    df["adv20"] = g["amount"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=15).mean())
    df["amount_ratio20"] = (df["amount"] / df["adv20"]).astype(np.float32)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["pre_close"]).abs(),
                    (df["low"] - df["pre_close"]).abs()],
                   axis=1).max(axis=1)
    df["tr"] = tr.astype(np.float32)
    df["atr14"] = g["tr"].transform(
        lambda s: s.rolling(14, min_periods=10).mean()).astype(np.float32)
    df["atr_ref"] = g["atr14"].transform(
        lambda s: s.shift(1).rolling(5).median()).astype(np.float32)
    # ---- 与 build_events_train_v1 同式 ----
    delta = g["close"].diff()
    df["_up"] = delta.clip(lower=0)
    df["_dn"] = (-delta).clip(lower=0)
    up14 = g["_up"].transform(lambda s: s.rolling(14).mean())
    dn14 = g["_dn"].transform(lambda s: s.rolling(14).mean())
    df["rsi14"] = (100 - 100 / (1 + up14 / dn14.replace(0, np.nan))) \
        .astype(np.float32)
    up6 = g["_up"].transform(lambda s: s.rolling(6).mean())
    dn6 = g["_dn"].transform(lambda s: s.rolling(6).mean())
    df["rsi6"] = (100 - 100 / (1 + up6 / dn6.replace(0, np.nan))) \
        .astype(np.float32)
    df["rsi6_minus_rsi14"] = (df["rsi6"] - df["rsi14"]).astype(np.float32)
    df["rsi_delta3"] = (df["rsi14"]
                        - df["rsi14"].groupby(df["code"], sort=False)
                        .shift(3)).astype(np.float32)
    ma = {n: g["close"].transform(lambda s, n=n: s.rolling(n).mean())
          for n in (3, 6, 12, 24, 20)}
    bbi = (ma[3] + ma[6] + ma[12] + ma[24]) / 4
    df["bbi_distance"] = (df["close"] / bbi - 1).astype(np.float32)
    df["bbi_slope5"] = (bbi / bbi.groupby(df["code"], sort=False).shift(5)
                        - 1).astype(np.float32)
    std20 = g["close"].transform(lambda s: s.rolling(20).std())
    denom = (4 * std20).replace(0, np.nan)
    df["boll_pct_b"] = ((df["close"] - (ma[20] - 2 * std20)) / denom) \
        .astype(np.float32)
    df["boll_width20_atr"] = (4 * std20 / df["atr14"]).astype(np.float32)
    dn_vol = g["volume"].transform(
        lambda s: s.where(df.loc[s.index, "ret1"] < 0, 0).rolling(5).sum())
    tot_vol = g["volume"].transform(lambda s: s.rolling(5).sum())
    df["down_volume_ratio5"] = (dn_vol / tot_vol.replace(0, np.nan)) \
        .astype(np.float32)
    t_med = g["turnover"].transform(
        lambda s: s.rolling(20, min_periods=10).median())
    df["turnover_ratio20"] = (df["turnover"] / t_med.replace(0, np.nan)) \
        .astype(np.float32)
    # turnover_pct60：当日换手在过去60日（含当日，至少20个）的分位
    to_vals = df["turnover"].values
    tpc = np.full(len(df), np.nan, dtype=np.float32)
    for c, grp in df.groupby("code", sort=False):
        arr = grp["turnover"].values
        idx_pos = grp.index.values
        n = len(arr)
        for j in range(n):
            lo = max(0, j - 59)
            w = arr[lo:j + 1]
            if len(w) >= 20 and not np.isnan(arr[j]):
                tpc[idx_pos[j]] = (w <= arr[j]).mean()
    df["turnover_pct60"] = tpc

    out = df[["code", "date"] + list(FEATS)].copy()
    out["code"] = out["code"].astype(str)
    out["date"] = out["date"].astype(int)
    out = out.merge(keys, on=["code", "date"], how="inner")
    out.to_parquet(OUT, index=False)
    log(f"写入 {OUT}: {out.shape}；用时 {time.time()-t0:.0f}s")
    log(f"NaN率: {out[list(FEATS)].isna().mean().round(4).to_dict()}")


if __name__ == "__main__":
    main()
