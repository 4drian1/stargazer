# -*- coding: utf-8 -*-
"""T1 盘中路径结构特征 builder（盘中路线 T0-T5 消融，第 1 步）。

复用 minute_label_engine.load_code() 与 DP 时间门控，保证窗口口径与标签引擎
完全一致（无未来函数）：窗口 = T+1 当日 [09:30, DP] 的已完成分钟 bar。

输出：minute_path_features_2025.parquet
  code, dp_date, dp,
  pf_ret1m, pf_ret3m, pf_ret5m,        # 最近 1/3/5 分钟速度（ATR 单位，负=向下）
  pf_decline_speed,                   # 窗口内 close 对分钟序号 OLS 斜率（ATR/分钟）
  pf_decel,                           # 后段斜率 − 前段斜率（>0=下杀减速）
  pf_low2_minus_low1,                 # 第2局部低点 − 第1局部低点（ATR 单位，>0=低点抬高）
  pf_has_two_lows                    # 窗口内是否出现 ≥2 个局部低点（标志位，不填充）

ATR 单位换算：atr_ref_price = sig_close × atr_pct（= 信号日冻结 atr_ref 价位）；
价格差 / atr_ref_price 即 ATR 单位。所有「最低/低点」一律取 DP 窗口 [09:30, DP]
内，绝不取全日最低（10:00/10:30 取全日最低会泄漏未来信息）。

局部低点：分钟 close 严格低于左右邻 k=2 根。09:45 窗口仅 15 分钟，双底多为
NaN → 用 pf_has_two_lows 标志，不填充（与「缺失值作为单独特征」约定一致）。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
sys.path.insert(0, OUT_DIR)
from minute_label_engine import load_code  # noqa: E402

SAMPLES = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
OUT = os.path.join(OUT_DIR, "minute_path_features_2025.parquet")
DPS = (945, 1000, 1030, 1330, 1400, 1430, 1450)  # AM+PM 全决策点（§18.17）
K_LOCAL = 2  # 局部低点左右邻根数


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ols_slope(y):
    """y 对 0..n-1 的 OLS 斜率（价格/分钟）；n<2 返回 NaN。"""
    n = len(y)
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def compute_path_features(p, atr_ref_price):
    """p: DP 窗口内分钟 close 数组（已按时间升序）。返回 7 特征 dict。"""
    n = len(p)
    if n == 0 or not np.isfinite(atr_ref_price) or atr_ref_price <= 0:
        return {k: np.nan for k in (
            "pf_ret1m", "pf_ret3m", "pf_ret5m", "pf_decline_speed",
            "pf_decel", "pf_low2_minus_low1", "pf_has_two_lows")}
    ap = atr_ref_price

    # 1-3 速度（相对窗口末 bar）
    pf_ret1m = (p[-1] - p[-2]) / ap if n >= 2 else np.nan
    pf_ret3m = (p[-1] - p[-4]) / ap if n >= 4 else np.nan
    pf_ret5m = (p[-1] - p[-6]) / ap if n >= 6 else np.nan

    # 4 平均下杀速度（窗口内 OLS 斜率，ATR/分钟）
    pf_decline_speed = ols_slope(p) / ap

    # 5 减速 = 后段斜率 − 前段斜率
    mid = n // 2
    if mid >= 2 and (n - mid) >= 2:
        s1 = ols_slope(p[:mid])
        s2 = ols_slope(p[mid:])
        pf_decel = (s2 - s1) / ap if (np.isfinite(s1) and np.isfinite(s2)) else np.nan
    else:
        pf_decel = np.nan

    # 6-7 局部低点结构（k=2）
    mins = []
    lo, hi = K_LOCAL, n - K_LOCAL
    for i in range(lo, hi):
        if (p[i] < p[i - 1] and p[i] < p[i - 2]
                and p[i] < p[i + 1] and p[i] < p[i + 2]):
            mins.append(i)
    if len(mins) >= 2:
        low1 = p[mins[0]]
        low2 = p[mins[1]]
        pf_low2_minus_low1 = (low2 - low1) / ap
        pf_has_two_lows = 1
    else:
        pf_low2_minus_low1 = np.nan
        pf_has_two_lows = 0

    return {"pf_ret1m": pf_ret1m, "pf_ret3m": pf_ret3m, "pf_ret5m": pf_ret5m,
            "pf_decline_speed": pf_decline_speed, "pf_decel": pf_decel,
            "pf_low2_minus_low1": pf_low2_minus_low1,
            "pf_has_two_lows": pf_has_two_lows}


def main():
    global SAMPLES, OUT
    args = sys.argv[1:]
    if "--samples" in args:
        SAMPLES = args[args.index("--samples") + 1]
    if "--out" in args:
        OUT = args[args.index("--out") + 1]
    t0 = time.time()
    df = pd.read_parquet(SAMPLES)
    df = df[df["dp"].isin(DPS)].copy()
    # 每个 (code, dp_date, dp) 的 atr_ref 应只取决于信号日，与 src/entry 无关
    g = (df.groupby(["code", "dp_date", "dp"])
           .agg(atr_pct=("atr_pct", "first"),
                sig_close=("sig_close", "first"),
                n_rows=("atr_pct", "size"))
           .reset_index())
    log(f"唯一 (code,dp_date,dp) 组合: {len(g):,}")
    # 一致性检查：atr_ref 在组内应恒定
    chk = (df.groupby(["code", "dp_date", "dp"])["atr_pct"]
           .nunique().max())
    chk2 = (df.groupby(["code", "dp_date", "dp"])["sig_close"]
            .nunique().max())
    if chk > 1 or chk2 > 1:
        log(f"warn: atr_pct/sig_close 组内不唯一 (max nunique={chk}/{chk2})，"
            f"取 first 可能偏差")
    else:
        log("atr_ref 组内一致 ✓")

    rows = []
    cur_code = None
    cm = None
    miss_cache = 0
    miss_day = 0
    for _, r in g.iterrows():
        code = r["code"]
        dp_date = int(r["dp_date"])
        dp = int(r["dp"])
        if code != cur_code:
            cur_code = code
            cm = load_code(code)
        if cm is None:
            miss_cache += 1
            continue
        hit = np.where(cm["days"] == dp_date)[0]
        if not len(hit):
            miss_day += 1
            continue
        day_i = int(hit[0])
        m = (cm["day_idx"] == day_i) & (cm["t"] <= dp * 100)
        if not m.any():
            miss_day += 1
            continue
        p = cm["c"][m].astype(np.float64)
        # 升序（按 t 已排序，但保险起见）
        p = p[np.argsort(cm["t"][m])] if not np.all(np.diff(cm["t"][m]) >= 0) else p
        atr_ref_price = float(r["atr_pct"]) * float(r["sig_close"])
        feats = compute_path_features(p, atr_ref_price)
        rows.append({"code": code, "dp_date": dp_date, "dp": dp, **feats})

    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    log(f"写入 {OUT}: {out.shape}；缺失 cache={miss_cache:,} 缺失day="
        f"{miss_day:,}；用时 {time.time()-t0:.0f}s")
    # 覆盖与缺失值概览
    for c in ["pf_ret1m", "pf_ret3m", "pf_ret5m", "pf_decline_speed",
              "pf_decel", "pf_low2_minus_low1", "pf_has_two_lows"]:
        miss = out[c].isna().mean()
        log(f"  {c}: NaN率 {miss:.1%}")
    log(f"pf_has_two_lows=1 占比 {out['pf_has_two_lows'].mean():.1%}")


if __name__ == "__main__":
    main()
