# -*- coding: utf-8 -*-
"""T2 v2：ATR冲击后的承接结构特征（评审冻结口径，2026-09-08）。

核心定义：开盘较大ATR波动释放 → 首低形成 → 后续下探不再明显创新低/低点抬高
→ 下跌量收缩 → 价格恢复。**不再要求二低触及首低**，每个候选都计算特征，
NaN 只来自数据缺失（某段无bar），不用"是否回踩"做硬过滤。

固定时间窗口（无主观局部低点阈值）：
  首段 seg1 = 09:30–09:45
  A段   segA = 09:45–10:00
  B段   segB = 10:00–10:30
  dp=945  不计算（路径不足，T2 只评 10:00/10:30）
  dp=1000 特征对 (seg1, segA)，列后缀 _a
  dp=1030 保留两对结构 (seg1, segA)=_a 与 (seg1, segB)=_b，不合并45分钟窗口

特征（atr 单位 = 价格差/(sig_close×atr_pct)，与 path_features 同口径；
pre_close = 样本行 pre_close 列 = 信号日收盘，raw 尺度）：
  首段（两对共用，无前缀）：
    first_leg_shock_atr   首段振幅/ATR = (max h1 - min l1)/atr
    first_low_depth_atr   首低相对昨收的ATR距离 = (first_low - pre_close)/atr
  每对（pfx ∈ {_a,_b}）：
    {pfx}second_low_minus_first_low_atr  二低-首低（正值=抬高低点）
    {pfx}rebound_before_second_test_atr  首低后至二段末的反弹高度/ATR
    {pfx}second_pullback_ratio  二次回撤占首次反弹比例
                                =(rebound_high-second_low)/(rebound_high-first_low)
    {pfx}second_volume_vs_first_volume    每分钟量之比（去窗口时长偏）
    {pfx}second_down_volume_vs_first_down_volume  真·阴线量(close<open)每分钟比
    {pfx}second_range_vs_first_range      每分钟振幅之比
    {pfx}second_close_location  二段末收盘在二段振幅中的位置
    {pfx}second_vwap_distance_atr  二段末收盘相对二段真实VWAP的ATR距离
    {pfx}minutes_between_lows   首低至二低的分钟数
    {pfx}new_low_on_second_leg  1=二段跌破首低(×0.999)；诊断分箱用，非硬过滤

数据来源 minute_cache_clean（bar级/100修正；隔离日 load_code 自动剔除）。
幂等；--samples/--out 跑 2025 与 2026；--audit 输出模板股审计表。
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

PAIR_FEATS = ("second_low_minus_first_low_atr",
              "rebound_before_second_test_atr", "second_pullback_ratio",
              "second_volume_vs_first_volume",
              "second_down_volume_vs_first_down_volume",
              "second_range_vs_first_range", "second_close_location",
              "second_vwap_distance_atr", "minutes_between_lows",
              "new_low_on_second_leg")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def t_to_min(t):
    """HHMMSS int -> 当日分钟数。"""
    return (t // 10000) * 60.0 + ((t // 100) % 100) + (t % 100) / 60.0


def pair_features(cm, day_i, m1, m2, atr_px):
    """(seg1, segX) 对的10个特征。段空/atr非法返回全NaN。"""
    f = {k: np.nan for k in PAIR_FEATS}
    if not np.isfinite(atr_px) or atr_px <= 0:
        return f
    if not m1.any() or not m2.any():
        return f
    h1, l1, v1 = cm["h"][m1], cm["l"][m1], cm["v"][m1]
    o2, h2, l2, c2, v2, a2 = (cm["o"][m2], cm["h"][m2], cm["l"][m2],
                              cm["c"][m2], cm["v"][m2], cm["a"][m2])
    t1_arr, t2_arr = cm["t"][m1], cm["t"][m2]
    first_low = float(l1.min())
    second_low = float(l2.min())
    t_first = t_to_min(int(t1_arr[int(np.argmin(l1))]))
    t_second = t_to_min(int(t2_arr[int(np.argmin(l2))]))
    f["second_low_minus_first_low_atr"] = (second_low - first_low) / atr_px
    # 首低后（含seg1余下部分与segX）至二段末的最高价（限当日）
    m_after = ((cm["day_idx"] == day_i)
               & (cm["t"] > t1_arr[int(np.argmin(l1))])
               & (cm["t"] <= int(t2_arr[-1])))
    hi_idx = np.where(m_after)[0]
    rebound_high = float(cm["h"][hi_idx].max()) if len(hi_idx) else np.nan
    f["rebound_before_second_test_atr"] = (rebound_high - first_low) / atr_px
    denom = rebound_high - first_low
    if np.isfinite(denom) and denom > EPS:
        f["second_pullback_ratio"] = (rebound_high - second_low) / denom
    n1, n2 = float(m1.sum()), float(m2.sum())
    r1 = float(h1.max() - l1.min())
    r2 = float(h2.max() - l2.min())
    if r1 > EPS and n1 > 0 and n2 > 0:
        f["second_range_vs_first_range"] = (r2 / n2) / (r1 / n1)
    vv1, vv2 = float(v1.sum()), float(v2.sum())
    if vv1 > EPS:
        f["second_volume_vs_first_volume"] = (vv2 / n2) / (vv1 / n1)
    d1 = float(v1[cm["c"][m1] < cm["o"][m1] - EPS].sum())
    d2 = float(v2[c2 < o2 - EPS].sum())
    if d1 > EPS:
        f["second_down_volume_vs_first_down_volume"] = (d2 / n2) / (d1 / n1)
    if r2 > EPS:
        f["second_close_location"] = (float(c2[-1]) - second_low) / r2
    if vv2 > EPS:
        f["second_vwap_distance_atr"] = (float(c2[-1]) - float(a2.sum()) / vv2) \
            / atr_px
    f["minutes_between_lows"] = t_second - t_first
    f["new_low_on_second_leg"] = 1.0 if second_low < first_low * 0.999 else 0.0
    return f


def compute(cm, day_i, dp, atr_px, pre_close):
    """单(code,day,dp)：首段特征 + 各对特征。"""
    f = {"first_leg_shock_atr": np.nan, "first_low_depth_atr": np.nan}
    for k in PAIR_FEATS:
        f[f"a_{k}"] = np.nan
        if dp == 1030:
            f[f"b_{k}"] = np.nan
    if not np.isfinite(atr_px) or atr_px <= 0:
        return f
    m1 = (cm["day_idx"] == day_i) & (cm["t"] > 93000) & (cm["t"] <= 94500)
    if not m1.any():
        return f
    r1 = float(cm["h"][m1].max() - cm["l"][m1].min())
    first_low = float(cm["l"][m1].min())
    f["first_leg_shock_atr"] = r1 / atr_px
    if np.isfinite(pre_close):
        f["first_low_depth_atr"] = (first_low - pre_close) / atr_px
    mA = (cm["day_idx"] == day_i) & (cm["t"] > 94500) & (cm["t"] <= 100000)
    for k, v in pair_features(cm, day_i, m1, mA, atr_px).items():
        f[f"a_{k}"] = v
    if dp == 1030:
        mB = (cm["day_idx"] == day_i) & (cm["t"] > 100000) \
            & (cm["t"] <= 103000)
        for k, v in pair_features(cm, day_i, m1, mB, atr_px).items():
            f[f"b_{k}"] = v
    return f


def audit(code, dates):
    """模板股审计：逐日打印首段与两对结构的关键量。"""
    cm = load_code(code)
    if cm is None:
        log(f"{code}: 无缓存")
        return
    for d in dates:
        hit = np.where(cm["days"] == d)[0]
        if not len(hit):
            log(f"{code} {d}: 无数据（可能被隔离或缓存缺失）")
            continue
        day_i = int(hit[0])
        atr_px = None
        m1 = (cm["day_idx"] == day_i) & (cm["t"] > 93000) & (cm["t"] <= 94500)
        log(f"--- {code} {d} ---")
        if not m1.any():
            log("  seg1 无bar")
            continue
        l1, h1, v1 = cm["l"][m1], cm["h"][m1], cm["v"][m1]
        pc = float(cm["c"][cm["day_idx"] == day_i][0])  # 占位（审计打印用）
        first_low = float(l1.min())
        rng1 = float(h1.max() - l1.min())
        log(f"  seg1: low={first_low:.2f} range={rng1:.2f} "
            f"vol={v1.sum():,.0f} bars={m1.sum()}")
        for name, lo, hi in (("A(9:45-10:00)", 94500, 100000),
                             ("B(10:00-10:30)", 100000, 103000)):
            m2 = (cm["day_idx"] == day_i) & (cm["t"] > lo) & (cm["t"] <= hi)
            if not m2.any():
                log(f"  seg{name}: 无bar")
                continue
            l2, h2, c2, v2, a2 = (cm["l"][m2], cm["h"][m2], cm["c"][m2],
                                  cm["v"][m2], cm["a"][m2])
            vv2 = float(v2.sum())
            vwap2 = float(a2.sum()) / vv2 if vv2 > 0 else np.nan
            log(f"  seg{name}: low={l2.min():.2f} "
                f"(Δ首低 {l2.min()-first_low:+.2f}) "
                f"range_pm={float(h2.max()-l2.min())/m2.sum():.4f} "
                f"vol_pm={vv2/m2.sum():,.0f} "
                f"downvol_pm={float(v2[cm['c'][m2] < cm['o'][m2]].sum())/m2.sum():,.0f} "
                f"last={float(c2[-1]):.2f} vwap={vwap2:.2f}")


def main():
    global SAMPLES, OUT
    args = sys.argv[1:]
    if "--samples" in args:
        SAMPLES = args[args.index("--samples") + 1]
    if "--out" in args:
        OUT = args[args.index("--out") + 1]
    if "--audit" in args:
        # --audit CODE:DAY1,DAY2,...
        spec = args[args.index("--audit") + 1]
        code, ds = spec.split(":")
        audit(code, [int(x) for x in ds.split(",")])
        return
    t0 = time.time()
    df = pd.read_parquet(SAMPLES)
    df = df[df["dp"].isin(DPS)].copy()
    g = (df.groupby(["code", "dp_date", "dp"])
           .agg(atr_pct=("atr_pct", "first"), sig_close=("sig_close", "first"),
                pre_close=("pre_close", "first"))
           .reset_index())
    log(f"唯一 (code,dp_date,dp): {len(g):,}")
    rows, miss_cache, miss_day = [], 0, 0
    cur, cm = None, None
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
                     **compute(cm, int(hit[0]), int(r["dp"]), atr_px,
                               float(r["pre_close"]))})
    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    log(f"写入 {OUT}: {out.shape}；miss_cache={miss_cache} miss_day={miss_day}"
        f"；用时 {time.time()-t0:.0f}s")
    key = ["first_leg_shock_atr", "a_second_low_minus_first_low_atr",
           "a_second_volume_vs_first_volume", "a_second_vwap_distance_atr"]
    log(f"核心特征 NaN 率：{out[key].isna().mean().round(4).to_dict()}")
    log(f"new_low_on_second_leg 占比："
        f"{out.groupby('dp')[['a_new_low_on_second_leg','b_new_low_on_second_leg']].mean().round(3).to_dict()}")
    log(f"二低抬高(Δ>0) 占比："
        f"{out.groupby('dp')[['a_second_low_minus_first_low_atr','b_second_low_minus_first_low_atr']].apply(lambda s:(s>0).mean()).round(3).to_dict()}")


if __name__ == "__main__":
    main()
