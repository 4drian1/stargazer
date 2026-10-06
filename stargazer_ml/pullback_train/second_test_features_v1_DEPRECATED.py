# -*- coding: utf-8 -*-
# ⚠️ 作废存根（v1，评审否决，2026-09-08）：三个关键缺陷——
#   1) 只有二低触及/跌破首低才算回踩，丢掉"低点抬高+缩量承接"优质结构；
#   2) 阴线量方向写反（o<c 是阳线量）；
#   3) dp1030 把 9:45-10:30 合并成一段，混淆回踩/企稳/突破。
# 正式实现见 second_test_features.py（v2，ATR冲击后承接结构）。
"""T2 v1（作废，仅留档，禁止用于训练/评估）。
T2二次回踩量价结构特征（固定时间窗口，无主观局部低点阈值）。

窗口（与评审冻结设计一致）：
  第一段 seg1 = 09:30–09:45；第二段 leg2 = 09:45–DP（dp=1000: 9:45–10:00；
  dp=1030: 9:45–10:30）。dp=945 不计算（路径不足，T2 只评 10:00/10:30）。

定义（atr 单位 = 价格差/(sig_close×atr_pct)，与 path_features 相同口径）：
  first_low          = seg1 最低 low；t_first_low 为其时间
  second_low         = leg2 最低 low；t_second_low 为其时间
  has_second_test    = 1 当 second_low <= first_low×1.0001（回踩到/跌破首低），
                       否则 0；has_second_test=0 时 second_* 全部保持 NaN
                       （"未形成二踩" ≠ "二踩幅度为0"，不填0）
  range/volume 类特征用每分钟口径归一（leg2 与 seg1 时长不同，消除窗口长度偏）：
  second_range_vs_first_range        = leg2每分钟振幅 / seg1每分钟振幅
  second_volume_vs_first_volume      = leg2每分钟量 / seg1每分钟量
  second_down_volume_vs_first_down_volume = leg2每分钟阴线量 / seg1每分钟阴线量
                                       （阴线=close<open）
  second_low_minus_first_low_atr     = (second_low-first_low)/atr（≤0）
  rebound_height_after_first_low_atr = (首低后至DP最高价-first_low)/atr
  second_close_location              = (DP收盘-second_low)/(leg2振幅)
  second_vwap_distance_atr           = (DP收盘-leg2真实VWAP)/atr
  second_leg_return_atr              = (DP收盘-leg2首bar开盘)/atr
  new_low_on_second_leg              = 1 当 second_low < first_low×0.999
  second_leg_recovery_ratio          = (DP收盘-second_low)/(二低后至DP最高价-second_low)
  minutes_between_first_low_and_retest = 首低至二低的分钟数

数据来源：minute_cache_clean（bar级/100修正后；量纲不确定日已被隔离，
load_code 自动剔除）。脚本自身幂等，可用 --samples/--out 跑 2025 与 2026。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minute_label_engine import load_code  # noqa: E402

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
SAMPLES = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
OUT = os.path.join(OUT_DIR, "minute_second_test_2025.parquet")
DPS = (1000, 1030)
EPS = 1e-9


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def t_to_min(t):
    """HHMMSS int -> 当日分钟数（支持跨小时相减）。"""
    return (t // 10000) * 60.0 + ((t // 100) % 100) + (t % 100) / 60.0


def nan_feats():
    keys = ("has_second_test", "second_low_minus_first_low_atr",
            "second_range_vs_first_range", "second_volume_vs_first_volume",
            "second_down_volume_vs_first_down_volume",
            "rebound_height_after_first_low_atr", "second_close_location",
            "second_vwap_distance_atr", "second_leg_return_atr",
            "new_low_on_second_leg", "second_leg_recovery_ratio",
            "minutes_between_first_low_and_retest")
    return {k: np.nan for k in keys}


def compute(cm, day_i, dp, atr_px):
    """单(code,day,dp)的T2特征。atr_px<=0 或无bar返回全NaN。"""
    f = nan_feats()
    if not np.isfinite(atr_px) or atr_px <= 0:
        return f
    tt = cm["t"][cm["day_idx"] == day_i]
    if not len(tt):
        return f
    m1 = (cm["day_idx"] == day_i) & (cm["t"] > 93000) & (cm["t"] <= 94500)
    m2 = (cm["day_idx"] == day_i) & (cm["t"] > 94500) & (cm["t"] <= dp * 100)
    if not m1.any() or not m2.any():
        return f
    o1, h1, l1, c1, v1 = (cm["o"][m1], cm["h"][m1], cm["l"][m1],
                          cm["c"][m1], cm["v"][m1])
    o2, h2, l2, c2, v2, a2 = (cm["o"][m2], cm["h"][m2], cm["l"][m2],
                              cm["c"][m2], cm["v"][m2], cm["a"][m2])
    first_low = float(l1.min())
    i1 = int(np.argmin(l1))
    second_low = float(l2.min())
    i2 = int(np.argmin(l2))
    t1 = t_to_min(int(tt[m1][i1]))
    t2 = t_to_min(int(tt[m2][i2]))
    f["has_second_test"] = 1.0 if second_low <= first_low * 1.0001 else 0.0
    if f["has_second_test"] == 0.0:
        return f  # second_* 保持 NaN
    f["second_low_minus_first_low_atr"] = (second_low - first_low) / atr_px
    r1 = float(h1.max() - l1.min())
    r2 = float(h2.max() - l2.min())
    if r1 > EPS:
        f["second_range_vs_first_range"] = (r2 / m2.sum()) / (r1 / m1.sum())
    vv1, vv2 = float(v1.sum()), float(v2.sum())
    if vv1 > EPS:
        f["second_volume_vs_first_volume"] = (vv2 / m2.sum()) / (vv1 / m1.sum())
    d1 = float(v1[o1 < c1 - EPS].sum()) if (o1 < c1 - EPS).any() else 0.0
    d2 = float(v2[o2 < c2 - EPS].sum()) if (o2 < c2 - EPS).any() else 0.0
    if d1 > EPS:
        f["second_down_volume_vs_first_down_volume"] = (d2 / m2.sum()) / \
            (d1 / m1.sum())
    # leg2 全部在首低之后，首低后反弹高度 = leg2 最高价相对首低
    hi_after1 = float(h2.max())
    f["rebound_height_after_first_low_atr"] = (hi_after1 - first_low) / atr_px
    if r2 > EPS:
        f["second_close_location"] = (float(c2[-1]) - second_low) / r2
    if vv2 > EPS:
        f["second_vwap_distance_atr"] = (float(c2[-1]) - float(a2.sum()) / vv2) \
            / atr_px
    f["second_leg_return_atr"] = (float(c2[-1]) - float(o2[0])) / atr_px
    f["new_low_on_second_leg"] = 1.0 if second_low < first_low * 0.999 else 0.0
    hi_after2 = float(h2[i2:].max()) if i2 < len(h2) else float(h2.max())
    if hi_after2 - second_low > EPS:
        f["second_leg_recovery_ratio"] = (float(c2[-1]) - second_low) / \
            (hi_after2 - second_low)
    f["minutes_between_first_low_and_retest"] = t2 - t1
    return f


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
    g = (df.groupby(["code", "dp_date", "dp"])
           .agg(atr_pct=("atr_pct", "first"), sig_close=("sig_close", "first"))
           .reset_index())
    log(f"唯一 (code,dp_date,dp): {len(g):,}")
    rows, miss_cache, miss_day = [], 0, 0
    cur = None
    cm = None
    for _, r in g.iterrows():
        code = r["code"]
        if code != cur:
            cur = code
            cm = load_code(code)
        if cm is None:
            miss_cache += 1
            continue
        hit = np.where(cm["days"] == int(r["dp_date"]))[0]
        if not len(hit):
            miss_day += 1
            continue
        atr_px = float(r["atr_pct"]) * float(r["sig_close"])
        rows.append({"code": code, "dp_date": int(r["dp_date"]),
                     "dp": int(r["dp"]),
                     **compute(cm, int(hit[0]), int(r["dp"]), atr_px)})
    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    log(f"写入 {OUT}: {out.shape}；miss_cache={miss_cache} miss_day={miss_day}"
        f"；用时 {time.time()-t0:.0f}s")
    log(f"has_second_test=1 占比: "
        f"{out.groupby('dp')['has_second_test'].mean().round(3).to_dict()}")
    log(f"二踩特征 NaN 率（应≈1-has_second_test）：")
    for c in out.columns:
        if c.startswith("second_") or c.startswith("rebound"):
            log(f"  {c}: {out[c].isna().mean():.1%}")


if __name__ == "__main__":
    main()
