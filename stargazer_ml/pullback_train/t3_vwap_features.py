# -*- coding: utf-8 -*-
"""T3：VWAP修复持续性特征（AM+PM，评审冻结口径，§18.17）。

决策点：AM头 10:00/10:30；PM头 13:30/14:00/14:30/14:50（9:45仅M0/T1，无T3）。
Session VWAP = 自09:30起累计 amount/volume（含午休，逐分钟滚动）。
所有特征只用 ≤DP 已完成的bar；"最终收下影"等全日形态只能作标签，不作输入。

特征（atr单位 = 价格差/(sig_close×atr_pct)）：
  v3_first_reclaim_min        首次由VWAP下方站回上方的分钟数（自开盘；未站回NaN）
  v3_hold_after_reclaim_min   站回后至DP连续保持VWAP上方的分钟数（DP处下方=0）
  v3_cross_count              截至DP close穿越VWAP次数
  v3_longest_above_min        截至DP最长连续站上分钟数
  v3_rebreak_flag             站回后是否再次跌破（1/0；未站回NaN）
  v3_rebreak_depth_atr        再次跌破的最大深度/ATR（正=在VWAP下方；无则NaN）
  v3_above_min_frac           截至DP位于VWAP上方的分钟占比
  v3_dp_vwap_dist_atr         DP价与VWAP的ATR距离（正=上方）
  v3_above_vol_frac           VWAP上方bar的成交量占比
  v3_rel_mkt_after_reclaim    站回至DP个股收益 − 510300ETF同时窗收益（无站回NaN）

数据源：minute_cache_clean（load_code 自动剔除隔离日）；市场代理 etf510300_minute.parquet
（量纲已验收 r P1=0.9992/P99=1.0008，独立存储不入候选缓存）。
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
OUT = os.path.join(OUT_DIR, "minute_t3_vwap_2025.parquet")
ETF = os.path.join(OUT_DIR, "etf510300_minute.parquet")
DPS = (1000, 1030, 1330, 1400, 1430, 1450)
FEATS = ("v3_first_reclaim_min", "v3_hold_after_reclaim_min",
         "v3_cross_count", "v3_longest_above_min", "v3_rebreak_flag",
         "v3_rebreak_depth_atr", "v3_above_min_frac",
         "v3_dp_vwap_dist_atr", "v3_above_vol_frac",
         "v3_rel_mkt_after_reclaim")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def t_to_min(t):
    return (t // 10000) * 60.0 + ((t // 100) % 100) + (t % 100) / 60.0


class EtfProxy:
    """510300 分钟收盘价索引：day -> (t_array, cumret dict)"""

    def __init__(self, path):
        d = pd.read_parquet(path, columns=["date", "close"])
        d["day"] = (d["date"].astype(np.int64) // 1000000).astype(int)
        d["t"] = d["date"].astype(np.int64) % 1000000
        self.days = {}
        for day, g in d.groupby("day"):
            g = g.sort_values("t")
            c0 = float(g["close"].iloc[0])
            self.days[int(day)] = (g["t"].values.astype(int),
                                   (g["close"].values / c0).astype(float))

    def cumret_at(self, day, t):
        """day 内时间 <=t 的最后一根 bar 的开盘基准累计收益"""
        rec = self.days.get(int(day))
        if rec is None:
            return np.nan
        ts, cr = rec
        i = np.searchsorted(ts, t, side="right") - 1
        return float(cr[i]) if i >= 0 else np.nan


def compute(cm, day_i, dp, atr_px, etf, day):
    f = {k: np.nan for k in FEATS}
    if not np.isfinite(atr_px) or atr_px <= 0:
        return f
    m = (cm["day_idx"] == day_i) & (cm["t"] > 93000) & (cm["t"] <= dp * 100)
    if not m.any():
        return f
    idx = np.where(m)[0]
    c = cm["c"][idx]
    a = cm["a"][idx]
    v = cm["v"][idx]
    tt = cm["t"][idx]
    with np.errstate(all="ignore"):
        vwap = np.cumsum(a) / np.where(np.cumsum(v) > 0, np.cumsum(v), np.nan)
    above = c > vwap
    f["v3_cross_count"] = float(np.sum(above[1:] != above[:-1])) \
        if len(above) > 1 else 0.0
    f["v3_above_min_frac"] = float(above.mean())
    vv = v.sum()
    if vv > 0:
        f["v3_above_vol_frac"] = float(v[above].sum() / vv)
    f["v3_dp_vwap_dist_atr"] = float((c[-1] - vwap[-1]) / atr_px)
    # 最长连续站上
    longest = cur = 0
    for x in above:
        cur = cur + 1 if x else 0
        longest = max(longest, cur)
    f["v3_longest_above_min"] = float(longest)
    # 首次站回：出现过下方之后第一次上方
    below_seen = False
    reclaim_i = -1
    for i, x in enumerate(above):
        if not x:
            below_seen = True
        elif below_seen and reclaim_i < 0:
            reclaim_i = i
    if reclaim_i >= 0:
        f["v3_first_reclaim_min"] = t_to_min(int(tt[reclaim_i])) - 570.0
        # 当前连续保持
        hold = 0
        for x in above[::-1]:
            if x:
                hold += 1
            else:
                break
        f["v3_hold_after_reclaim_min"] = float(hold)
        after = above[reclaim_i:]
        f["v3_rebreak_flag"] = 1.0 if bool((~after).any()) else 0.0
        below_after = np.where(~after)[0]
        if len(below_after):
            ii = idx[reclaim_i:]
            depth = (vwap[reclaim_i:] - c[reclaim_i:])[below_after]
            f["v3_rebreak_depth_atr"] = float(depth.max() / atr_px)
        # 相对大盘：站回至DP
        er_dp = etf.cumret_at(day, dp * 100)
        er_rc = etf.cumret_at(day, int(tt[reclaim_i]))
        if np.isfinite(er_dp) and np.isfinite(er_rc) and er_rc > 0:
            f["v3_rel_mkt_after_reclaim"] = (float(c[-1]) / float(
                c[reclaim_i]) - 1.0) - (er_dp / er_rc - 1.0)
    return f


def main():
    global SAMPLES, OUT
    args = sys.argv[1:]
    if "--samples" in args:
        SAMPLES = args[args.index("--samples") + 1]
    if "--out" in args:
        OUT = args[args.index("--out") + 1]
    t0 = time.time()
    etf = EtfProxy(ETF)
    df = pd.read_parquet(SAMPLES)
    df = df[df["dp"].isin(DPS)].copy()
    g = (df.groupby(["code", "dp_date", "dp"])
           .agg(atr_pct=("atr_pct", "first"), sig_close=("sig_close", "first"))
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
                     **compute(cm, int(hit[0]), int(r["dp"]), atr_px, etf,
                               int(r["dp_date"]))})
    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    log(f"写入 {OUT}: {out.shape}；miss_cache={miss_cache} miss_day={miss_day}"
        f"；用时 {time.time()-t0:.0f}s")
    log(f"NaN率: {out[list(FEATS)].isna().mean().round(4).to_dict()}")
    rc_rate = out.groupby("dp")["v3_first_reclaim_min"].apply(
        lambda s: s.notna().mean()).round(3).to_dict()
    log(f"站回占比: {rc_rate}")


if __name__ == "__main__":
    main()
