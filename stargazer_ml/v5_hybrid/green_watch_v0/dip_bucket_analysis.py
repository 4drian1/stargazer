"""dip_bucket_analysis.py — 池内事件 OpenDipATR 分桶 × X0 净收益。

目的：验证 H-LATR-01「0.20–0.25 ATR 开盘下杀是最佳低吸深度」是否有数据支持。

OpenDipATR 定义（latr_candidate_engine_spec §2）：
  OpenDipATR = (Open - Low_15m) / ATR_ref
口径：
  - 2025+ 且 minute_cache_clean 有该票分钟数据：用 T+1 入场日 09:30–09:44 真实分钟
    low（MINUTE_EXACT）。
  - 其余（2019–2024 历史段无分钟数据，或 2025+ 缺分钟缓存）：用 T+1 日线
    (open - low)/ATR_ref 近似，标注 DAILY_APPROX。两口径分开统计不混合。
另附「假设以 15 分钟低点成交」的近似入场收益（APPROX，仅探索，非真实口径）：
  approx_dip_net = exit_price*(1-卖侧成本) / (low_15m*(1+买侧成本)) - 1
  成本按冻结口径 买 0.0013 / 卖 0.0018。

输出 dip_buckets.parquet（逐事件）+ 控制台分桶表。
运行：d:/stargazer/.venv/Scripts/python.exe dip_bucket_analysis.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
POOL_EV = BASE / "pool_events.parquet"
OUT = BASE / "dip_buckets.parquet"
MIN_DIR = Path("d:/stargazer/stargazer_ml/pullback_train/minute_cache_clean")
BARS_GLOB = "D:/a股数据/data_daily_parts/part_*.parquet"

BUCKETS = [(-np.inf, 0.05, "NO_DIP/HOLD"), (0.05, 0.10, "SMALL_DIP"),
           (0.10, 0.15, "LIGHT_DIP"), (0.15, 0.20, "NORMAL_DIP"),
           (0.20, 0.25, "HYPOTHESIS_CORE"), (0.25, 0.30, "DEEP_DIP"),
           (0.30, 0.40, "VERY_DEEP_DIP"), (0.40, np.inf, "EXTREME_DIP")]
BUY_COST, SELL_COST = 0.0013, 0.0018


def bucketize(v):
    for lo, hi, name in BUCKETS:
        if lo <= v < hi:
            return name
    return "NA"


def load_daily_ohlc(codes):
    import glob
    cols = ["date", "code", "open", "low"]
    parts = [pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(BARS_GLOB))]
    df = pd.concat(parts, ignore_index=True)
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df[df["code"].isin(codes)]


def minute_open_dip(code, entry_date, atr_ref):
    """T+1 日 09:30–09:44 真实分钟 (open0930 - low15)/ATR_ref；无数据返回 None。"""
    f = MIN_DIR / f"{code}.parquet"
    if not f.exists() or not atr_ref or atr_ref <= 0:
        return None
    df = pd.read_parquet(f, columns=["date", "open", "low"])
    day = df[df["date"] // 1000000 == entry_date]
    if not len(day):
        return None
    m15 = day[day["date"] % 1000000 <= 94400]          # 09:30–09:44
    if not len(m15):
        m15 = day.head(15)
    o0 = float(day["open"].iloc[0])
    lo = float(m15["low"].min())
    if not np.isfinite(o0) or not np.isfinite(lo) or o0 <= 0:
        return None
    return (o0 - lo) / atr_ref, lo, o0


def main():
    print("[1/4] 加载池事件（已匹配 X0）…")
    ev = pd.read_parquet(POOL_EV)
    ev = ev[ev["matched"]].copy()
    ev["atr_ref"] = ev["atr_ref_y"]            # 标签冻结口径 ATR_ref
    ev = ev.drop(columns=["atr_ref_x", "atr_ref_y"])
    ev["code"] = ev["code"].astype(str).str.zfill(6)
    print(f"  {len(ev)} 事件")

    print("[2/4] 日线 open/low（T+1 近似口径）…")
    daily = load_daily_ohlc(set(ev["code"]))
    daily = daily.set_index(["code", "date"])
    doi = pd.MultiIndex.from_frame(ev[["code", "entry_date"]])
    ev["d_open"] = daily["open"].reindex(doi).to_numpy()
    ev["d_low"] = daily["low"].reindex(doi).to_numpy()
    ev["open_dip_atr_daily"] = (ev["d_open"] - ev["d_low"]) / ev["atr_ref"]
    ev.loc[(ev["atr_ref"] <= 0) | (ev["d_open"].isna()), "open_dip_atr_daily"] = np.nan

    print("[3/4] 2025+ 分钟精确口径 …")
    ev["open_dip_atr_minute"] = np.nan
    ev["low_15m"] = np.nan
    sub25 = ev[ev["entry_date"] >= 20250101]
    codes_with_min = {p.stem for p in MIN_DIR.glob("*.parquet")}
    n_hit = 0
    for i, r in enumerate(sub25.itertuples(), 1):
        if r.code not in codes_with_min:
            continue
        res = minute_open_dip(r.code, int(r.entry_date), float(r.atr_ref))
        if res is not None:
            ev.at[r.Index, "open_dip_atr_minute"] = res[0]
            ev.at[r.Index, "low_15m"] = res[1]
            n_hit += 1
        if i % 200 == 0:
            print(f"  {i}/{len(sub25)}（分钟命中 {n_hit}）", flush=True)
    print(f"  2025+ 事件 {len(sub25)}，分钟命中 {n_hit}，"
          f"回落日线近似 {len(sub25) - n_hit}")

    ev["dip_precision"] = np.where(ev["open_dip_atr_minute"].notna(),
                                   "MINUTE_EXACT", "DAILY_APPROX")
    ev["open_dip_atr"] = ev["open_dip_atr_minute"].fillna(ev["open_dip_atr_daily"])
    ev["latr_bucket"] = ev["open_dip_atr"].map(bucketize)

    # 近似「以15分钟低点成交」净收益（APPROX，非真实口径）
    entry_eff = ev["low_15m"].fillna(ev["d_low"])          # 分钟低点优先，否则日线 low
    ok = entry_eff.notna() & (entry_eff > 0) & ev["exit_price"].notna()
    ev["approx_dip_net"] = np.where(
        ok, ev["exit_price"] * (1 - SELL_COST) / (entry_eff * (1 + BUY_COST)) - 1,
        np.nan)

    ev.to_parquet(OUT, index=False)

    print("\n[4/4] 分桶统计（X0 标签 net_return，cost=0.003 旧口径）…")
    for prec, g in ev.groupby("dip_precision"):
        print(f"\n== 口径 {prec}（n={len(g)}）==")
        t = g.groupby("latr_bucket").agg(
            n=("net_return", "size"),
            mean_net=("net_return", "mean"),
            med_net=("net_return", "median"),
            win=("net_return", lambda s: (s > 0).mean()),
            approx_dip_mean=("approx_dip_net", "mean"),
        ).reindex([b[2] for b in BUCKETS]).dropna(how="all")
        print(t.to_string(float_format=lambda x: f"{x:+.4f}"))
    # 按 primary_setup 拆分（混合口径主表，另列分钟/日线分开的附表见报告）
    print("\n== HYPOTHESIS_CORE vs 邻桶（全部事件）==")
    for b in ["NORMAL_DIP", "HYPOTHESIS_CORE", "DEEP_DIP"]:
        g = ev[ev["latr_bucket"] == b]
        if len(g):
            print(f"  {b:16s} n={len(g):5d} mean_net={g['net_return'].mean():+.4f} "
                  f"med={g['net_return'].median():+.4f} win={(g['net_return']>0).mean():.1%} "
                  f"approx_dip_mean={g['approx_dip_net'].mean():+.4f}")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
