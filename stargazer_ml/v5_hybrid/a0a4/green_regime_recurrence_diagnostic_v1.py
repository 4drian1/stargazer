"""GREEN_REGIME_RECURRENCE_DIAGNOSTIC_V1 (预注册 2026-09-23, 见 开发路线).

唯一研究问题: 历史上是否存在重复出现的、事前(PIT)可观察的市场状态,
使 GREEN 弱信号具有一致方向?

纪律:
  - 不问 "什么 regime 能救策略"; 不用 2026-H2 定义任何状态 (防 hindsight overfitting)
  - 状态变量极简 (trend/breadth/vol/liquidity), 无聚类/embedding/深度表征
  - 阈值一律 expanding-window (min 250 交易日) 前向应用, 不做全历史分位回填
  - 2021-2025 为检验床; 2026-H2 仅作对照

被检信号:
  (1) pc_zone_distance_atr 每日截面 spearman IC vs 次日超额收益 (DIP eligible / BOTTOM 两组)
  (2) DIP / BOTTOM 候选池次日超额均值 baseline
  标签: r_h1 = close(H1)/open(D) − 1 (raw, 减全市场同窗均值; 除权噪声由截面去均值+秩相关吸收, 局限已注明)
       另算 r_h3 (H3 close) 稳健版.

预注册出口:
  R1: 多个独立历史时期同一状态方向一致 (非个别年份驱动) → 才允许 Pyxis Y0.5 Market Context
      作为 Context/interaction 做增量验证 (仍不是 Regime B→禁买)
  R2: 无重复关系 → 正式结束 GREEN-WATCH v1.1 (RESEARCH_NEGATIVE), 回上游设计新 hypothesis
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import a0a4_config as C  # noqa
sys.path.insert(0, str(_HERE.parent / "step9_preflight"))
import trading_calendar as TC  # noqa
from data_source_provenance import get_daily_parquet  # noqa

OUT = C.RUN / "green_regime_recurrence_diagnostic_v1"
OUT.mkdir(parents=True, exist_ok=True)
POOL = _HERE.parent / "green_watch_v0" / "pool_daily_v11.parquet"
MIN_POP = 30          # 每日 IC 最小候选数
MIN_STATE_DAYS = 10   # 年×状态 最少天数才报告
EXP_MIN = 250         # expanding 阈值最少历史(交易日)


def expand_state(s: pd.Series) -> pd.Series:
    """PIT 前向状态: s > expanding median (前 EXP_MIN 日), 不足历史 = NaN."""
    med = s.expanding(EXP_MIN).median()
    cnt = s.expanding(EXP_MIN).count()
    return pd.Series(np.where(cnt >= EXP_MIN, (s > med).astype(float), np.nan), index=s.index)


def daily_ic(df: pd.DataFrame, col_ret: str) -> pd.Series:
    """每日截面 spearman(zone_distance_atr, col_ret), 经组内 rank + pearson 向量化."""
    g = df.dropna(subset=["zone_distance_atr", col_ret]).copy()
    g = g[g.groupby("date")[col_ret].transform("size") >= MIN_POP]
    if g.empty:
        return pd.Series(dtype=float)
    g["rz"] = g.groupby("date")["zone_distance_atr"].rank()
    g["rr"] = g.groupby("date")[col_ret].rank()
    g["rzrr"] = g["rz"] * g["rr"]
    g["rz2"] = g["rz"] ** 2
    g["rr2"] = g["rr"] ** 2
    a = g.groupby("date").agg(n=("rr", "size"), rz=("rz", "mean"), rr=("rr", "mean"),
                              rzrr=("rzrr", "mean"), rz2=("rz2", "mean"), rr2=("rr2", "mean"))
    var_rz = a["rz2"] - a["rz"] ** 2
    var_rr = a["rr2"] - a["rr"] ** 2
    cov = a["rzrr"] - a["rz"] * a["rr"]
    ic = cov / np.sqrt(var_rz * var_rr)
    ic[(var_rz <= 0) | (var_rr <= 0)] = np.nan
    return ic


def main():
    t0 = time.time()
    # ---------------- 1. market regime series (PIT at P close) ----------------
    daily = get_daily_parquet()[["date", "code", "open", "close", "pct_chg", "amount"]].copy()
    for c_ in ("date", "open", "close", "pct_chg", "amount"):
        daily[c_] = pd.to_numeric(daily[c_], errors="coerce")
    daily = daily.dropna(subset=["date", "code", "open", "close"]).sort_values(["code", "date"])
    daily = daily[(daily["open"] > 0) & (daily["close"] > 0)]  # open=0 脏行会产出 inf 收益
    daily["ret"] = daily["pct_chg"] / 100.0
    grp = daily.groupby("code")["close"]
    daily["ma20"] = grp.transform(lambda s: s.rolling(20, 20).mean())
    daily["next_close"] = grp.shift(-1)
    daily["close3"] = grp.shift(-3)
    daily["above"] = np.where(daily["ma20"].isna(), np.nan,
                              (daily["close"] > daily["ma20"]).astype(float))
    daily["r_h1"] = np.where(daily["next_close"] > 0, daily["next_close"] / daily["open"] - 1.0, np.nan)
    daily["r_h3"] = np.where(daily["close3"] > 0, daily["close3"] / daily["open"] - 1.0, np.nan)
    per_day = daily.groupby("date").agg(
        mkt_ret=("ret", "mean"), amt=("amount", "sum"), breadth=("above", "mean"),
        mkt_h1=("r_h1", "mean"), mkt_h3=("r_h3", "mean")).sort_index()
    idx = (1.0 + per_day["mkt_ret"].fillna(0)).cumprod()
    reg = pd.DataFrame({
        "trend": (idx > idx.rolling(20, 20).mean()).astype(float),
        "breadth_hi": expand_state(per_day["breadth"]),
        "vol_hi": expand_state(per_day["mkt_ret"].rolling(20, 20).std()),
        "liq_hi": expand_state(per_day["amt"].rolling(20, 20).mean()),
        "mkt_h1": per_day["mkt_h1"], "mkt_h3": per_day["mkt_h3"],
        "breadth": per_day["breadth"], "mkt_ret": per_day["mkt_ret"],
    })
    print(f"[1] regime series {len(reg)} days ({reg.index.min()}-{reg.index.max()}) "
          f"{time.time()-t0:.0f}s")
    op = daily[["code", "date", "open"]].copy()
    cl = daily[["code", "date", "close"]].copy()
    del daily, grp

    # ---------------- 2. calendar next-day map ----------------
    TC.load_calendar()
    cal = TC._CAL if isinstance(TC._CAL, list) else sorted(TC._CAL)
    nxt = {int(cal[i]): int(cal[i + 1]) for i in range(len(cal) - 1)}

    # ---------------- 3. pool candidates + labels ----------------
    pool = pd.read_parquet(POOL)
    pool = pool[["date", "code", "zone_distance_atr", "dip_eligible_v11", "bottom_gate"]]
    pool = pool.dropna(subset=["zone_distance_atr"]).drop_duplicates(["date", "code"])
    pool["dip"] = pool["dip_eligible_v11"].astype(str).isin(("True", "1", "1.0"))
    pool["bottom"] = pool["bottom_gate"].astype(str).isin(("True", "1", "1.0"))
    pool["D"] = pool["date"].astype(int).map(nxt)
    pool = pool.dropna(subset=["D"])
    pool["D"] = pool["D"].astype(int)
    pool["H1"] = pool["D"].map(nxt)
    pool["H2"] = pool["H1"].map(nxt)
    pool["H3"] = pool["H2"].map(nxt)
    pool = pool.dropna(subset=["H1"])
    for c_ in ("H1", "H2", "H3"):
        pool[c_] = pool[c_].astype("Int64")

    op["date"] = op["date"].astype(int)
    cl["date"] = cl["date"].astype(int)
    pool = (pool.merge(op.rename(columns={"date": "D", "open": "open_d"}), on=["code", "D"], how="left")
                 .merge(cl.rename(columns={"date": "H1", "close": "close_h1"}), on=["code", "H1"], how="left")
                 .merge(cl.rename(columns={"date": "H3", "close": "close_h3"}), on=["code", "H3"], how="left"))
    pool = pool[(pool["open_d"] > 0).fillna(False)]
    pool["r_h1"] = np.where(pool["close_h1"] > 0, pool["close_h1"] / pool["open_d"] - 1.0, np.nan)
    pool["r_h3"] = np.where(pool["close_h3"] > 0, pool["close_h3"] / pool["open_d"] - 1.0, np.nan)
    mh1 = reg["mkt_h1"].to_dict()
    mh3 = reg["mkt_h3"].to_dict()
    pool["ex_h1"] = pool["r_h1"] - pool["D"].map(mh1)
    pool["ex_h3"] = pool["r_h3"] - pool["D"].map(mh3)
    pool = pool[np.isfinite(pool["ex_h1"].fillna(np.nan))]
    pool = pool.dropna(subset=["ex_h1"])
    print(f"[3] pool labeled n={len(pool)} dip/day={pool['dip'].groupby(pool['date']).sum().mean():.0f} "
          f"bottom/day={pool['bottom'].groupby(pool['date']).sum().mean():.0f} {time.time()-t0:.0f}s")

    # ---------------- 4. daily signals ----------------
    dip = pool[pool["dip"]]
    bot = pool[pool["bottom"]]
    sig = pd.DataFrame({
        "dip_ic_h1": daily_ic(dip, "ex_h1"), "dip_ic_h3": daily_ic(dip, "ex_h3"),
        "bot_ic_h1": daily_ic(bot, "ex_h1"), "bot_ic_h3": daily_ic(bot, "ex_h3"),
        "dip_base_h1": dip.groupby("date")["ex_h1"].mean(),
        "dip_base_h3": dip.groupby("date")["ex_h3"].mean(),
        "bot_base_h1": bot.groupby("date")["ex_h1"].mean(),
        "bot_base_h3": bot.groupby("date")["ex_h3"].mean(),
        "n_dip": dip.groupby("date").size(), "n_bot": bot.groupby("date").size(),
    })
    S = reg.join(sig, how="inner")
    yr = S.index.astype(str).str[:4]
    S["year"] = np.where(S.index >= 20260701, "2026H2",
                         np.where(yr == "2026", "2026H1", yr))
    S["post2607"] = S.index >= 20260701
    S["state_tb"] = np.select(
        [~np.isnan(S["trend"]) & (S["trend"] == 1) & (S["breadth_hi"] == 1),
         ~np.isnan(S["trend"]) & (S["trend"] == 1) & (S["breadth_hi"] == 0),
         ~np.isnan(S["trend"]) & (S["trend"] == 0) & (S["breadth_hi"] == 1),
         ~np.isnan(S["trend"]) & (S["trend"] == 0) & (S["breadth_hi"] == 0)],
        ["UP_HI", "UP_LO", "DN_HI", "DN_LO"], default=None)
    S.to_parquet(OUT / "daily_series.parquet")
    print(f"[4] daily signals n={len(S)} ({S.index.min()}-{S.index.max()}) {time.time()-t0:.0f}s")

    # ---------------- 5. tables ----------------
    def tab(state_col, metric):
        d = S.dropna(subset=[state_col, metric]).copy()
        t = d.groupby(["year", state_col])[metric].agg(["mean", "count"]).reset_index()
        t = t[t["count"] >= MIN_STATE_DAYS]
        return t.pivot(index=state_col, columns="year", values="mean").round(4)

    report = {"pre_registered_exits": {"R1": "repeated conditional edge -> Pyxis Y0.5 Context",
                                       "R2": "no repeatable relation -> Archive GREEN v1.1 RESEARCH_NEGATIVE"},
              "validation_2025plus": {
                  "dip_ic_h1": round(float(S[S.index >= 20250101]["dip_ic_h1"].mean()), 4),
                  "bot_ic_h1": round(float(S[S.index >= 20250101]["bot_ic_h1"].mean()), 4),
                  "dip_base_h1": round(float(S[S.index >= 20250101]["dip_base_h1"].mean()), 5),
                  "bot_base_h1": round(float(S[S.index >= 20250101]["bot_base_h1"].mean()), 5)},
              "note": "IC 符号: pc_zone_distance_atr 为正=价在支撑上方; proximity-good 预期 IC<0"}

    print("\n===== VALIDATION (a14 epoch 2025-01+, 对照 S0-A1p) =====")
    print(report["validation_2025plus"])
    for metric in ("dip_ic_h1", "bot_ic_h1", "dip_base_h1", "bot_base_h1"):
        print(f"\n===== {metric} | TRENDxBREADTH (rows=state, cols=year) =====")
        t = tab("state_tb", metric)
        print(t.to_string())
        d = S.dropna(subset=["state_tb", metric])
        cnt = d[d.groupby(["year", "state_tb"])[metric].transform("size") >= MIN_STATE_DAYS] \
            .groupby(["year", "state_tb"]).size().unstack(0)
        print("n_days:\n", cnt.to_string())
        t.to_csv(OUT / f"table_{metric}_trend_breadth.csv")
        report[f"tb_{metric}"] = json.loads(t.to_json())
    for st in ("trend", "breadth_hi", "vol_hi", "liq_hi"):
        print(f"\n===== dip_base_h1 | {st} =====")
        t = tab(st, "dip_base_h1")
        print(t.to_string())
        report[f"{st}_dip_base_h1"] = json.loads(t.to_json())
        print(f"===== dip_ic_h1 | {st} =====")
        t2 = tab(st, "dip_ic_h1")
        print(t2.to_string())
        report[f"{st}_dip_ic_h1"] = json.loads(t2.to_json())

    # pooled pre-2026 state diff (简单 t, 日度自相关未调整, 仅诊断)
    pre = S[(S.index < 20260101)]
    diffs = {}
    for st in ("state_tb", "trend", "breadth_hi", "vol_hi", "liq_hi"):
        for metric in ("dip_ic_h1", "dip_base_h1", "bot_base_h1"):
            d = pre.dropna(subset=[st, metric])
            for v in sorted(d[st].dropna().unique()):
                a = d[d[st] == v][metric]
                b = d[d[st] != v][metric]
                if len(a) >= MIN_STATE_DAYS and len(b) >= MIN_STATE_DAYS:
                    se = np.sqrt(a.var() / len(a) + b.var() / len(b))
                    diffs[f"{metric}|{st}={v}|pre2026"] = {
                        "mean": round(float(a.mean() - b.mean()), 5),
                        "t": round(float((a.mean() - b.mean()) / se), 2) if se > 0 else None,
                        "n": int(len(a))}
    report["pre2026_state_diffs"] = diffs
    print("\n===== pre-2026 pooled state-vs-complement diff (mean/t) =====")
    for k, v in diffs.items():
        print(f"  {k}: {v['mean']:+.5f} t={v['t']} n={v['n']}")

    with open(OUT / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[done {time.time()-t0:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main()
