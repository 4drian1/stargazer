# -*- coding: utf-8 -*-
"""T2-PM 事件对齐特征（v3 冻结方案 §6.6-3，§18.23）。

上午 T2 保留原结果（固定段 9:30-9:45/9:45-10:00/10:00-10:30，§18.15/18.16）。
本脚本为下午决策点（13:30/14:00/14:30/14:50）新增**事件对齐版**：
不假设下杀固定从 13:00 开始，而是在决策点前最近 30/60 个**交易分钟**内
因果寻找最大 ATR 下跌段（峰→谷），并计算承接结构。全部只用 ≤DP 的 bar，
**绝不使用决策点后的收盘下影**；午休不计入交易分钟（用 bar 计数天然满足）。

每 (code, dp_date, dp) 输出（W∈{30,60}，单位 ATR，atr_px=atr_pct×sig_close）：
  t2pm_decline_W_atr          最大下跌段幅度 (峰高−谷低)/atr
  t2pm_decline_speed_W        下跌段速度 = 幅度/持续交易分钟
  t2pm_recovery_W_atr         低点后恢复 = (DP价−谷低)/atr
  t2pm_min_since_low_W        距最低点交易分钟数
  t2pm_second_low_minus_first_W_atr  谷后最低低 − 首低（>0=低点抬高）
  t2pm_dnvol_shrink_W         承接段阴线量/下跌段阴线量（<1=缩量）
  t2pm_vwap_reclaim_atr       DP价 − 自09:30 session VWAP 的 ATR 距离（共用）

下跌段定义：窗口内 low 最低点为谷；谷之前 high 最高点为峰（谷在首bar则峰=谷，
幅度=0）。承接段=谷→DP。
幂等；--samples/--out 跑 2025 与 2026。
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
OUT = os.path.join(OUT_DIR, "t2pm_features_2025.parquet")
DPS = (1330, 1400, 1430, 1450)
WINDOWS = (30, 60)
FEATS = [f"t2pm_{k}_{w}" for w in WINDOWS for k in (
    "decline_atr", "decline_speed", "recovery_atr", "min_since_low",
    "second_low_minus_first_atr", "dnvol_shrink")] + ["t2pm_vwap_reclaim_atr"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute(cm, day_i, dp, atr_px):
    f = {k: np.nan for k in FEATS}
    if not np.isfinite(atr_px) or atr_px <= 0:
        return f
    idx_all = np.where(cm["day_idx"] == day_i)[0]
    if not len(idx_all):
        return f
    t_all = cm["t"][idx_all].astype(int)
    h = cm["h"][idx_all].astype(float)
    l = cm["l"][idx_all].astype(float)
    c = cm["c"][idx_all].astype(float)
    v = cm["v"][idx_all].astype(float)
    a = cm["a"][idx_all].astype(float)
    n = len(t_all)
    i_dp = int(np.searchsorted(t_all, dp * 100, side="right")) - 1
    if i_dp < 5:
        return f
    # session VWAP 至 DP（共用特征）
    cv = np.cumsum(v[:i_dp + 1])
    with np.errstate(all="ignore"):
        vwap_dp = np.cumsum(a[:i_dp + 1])[-1] / cv[-1] if cv[-1] > 0 else np.nan
    if np.isfinite(vwap_dp):
        f["t2pm_vwap_reclaim_atr"] = float((c[i_dp] - vwap_dp) / atr_px)
    down_bar = c < np.concatenate(([c[0]], c[:-1]))  # 收盘低于前收=阴线
    for w in WINDOWS:
        i0 = max(0, i_dp - w + 1)
        wl, wh = l[i0:i_dp + 1], h[i0:i_dp + 1]
        wc, wv = c[i0:i_dp + 1], v[i0:i_dp + 1]
        wd = down_bar[i0:i_dp + 1]
        ni = int(np.argmin(wl))            # 谷（相对索引）
        trough = wl[ni]
        peak = wh[:ni + 1].max()           # 峰=谷及之前的最高
        dur = ni  # 峰→谷交易分钟（保守按索引差）
        decline = (peak - trough) / atr_px
        f[f"t2pm_decline_atr_{w}"] = float(decline)
        f[f"t2pm_decline_speed_{w}"] = float(decline / dur) if dur > 0 else 0.0
        f[f"t2pm_recovery_atr_{w}"] = float((wc[-1] - trough) / atr_px)
        f[f"t2pm_min_since_low_{w}"] = float(i_dp - (i0 + ni))
        if ni < len(wl) - 1:
            second = wl[ni + 1:].min()
            f[f"t2pm_second_low_minus_first_atr_{w}"] = float(
                (second - trough) / atr_px)
            dn_decline = wv[:ni + 1][wd[:ni + 1]].sum()
            dn_rebound = wv[ni + 1:][wd[ni + 1:]].sum()
            if dn_decline > 0:
                f[f"t2pm_dnvol_shrink_{w}"] = float(dn_rebound / dn_decline)
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
           .agg(atr_pct=("atr_pct", "first"),
                sig_close=("sig_close", "first"))
           .reset_index())
    log(f"唯一 (code,dp_date,dp): {len(g):,}")
    rows, miss = [], 0
    cur, cm = None, None
    for _, r in g.iterrows():
        code = str(r["code"])
        if code != cur:
            cur = code
            cm = load_code(code)
        if cm is None:
            miss += 1
            continue
        hit = np.where(cm["days"] == int(r["dp_date"]))[0]
        if not len(hit):
            miss += 1
            continue
        atr_px = float(r["atr_pct"]) * float(r["sig_close"])
        rows.append({"code": code, "dp_date": int(r["dp_date"]),
                     "dp": int(r["dp"]),
                     **compute(cm, int(hit[0]), int(r["dp"]), atr_px)})
    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    log(f"写入 {OUT}: {out.shape}；miss={miss}；用时 {time.time()-t0:.0f}s")
    log(f"NaN率: {out[FEATS].isna().mean().round(4).to_dict()}")


if __name__ == "__main__":
    main()
