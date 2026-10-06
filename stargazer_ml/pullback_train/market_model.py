# -*- coding: utf-8 -*-
"""市场参与模型（阶段：2024减亏实验 G6/G7 的市场层）。

定位：不选股，只回答"明天该不该做绿盘低吸、最多开几仓"。
每日预测：未来5日低吸候选池平均净收益（回归主目标）与 STOP 比例（辅助），
OOF(2022-2023) 冻结分位 → 槽位建议 R0=0 / R1=5 / R2=10 / R3=20。

输入（全部为 t 日收盘前可知，严格因果）：
- 广度：上涨家数比例、收益中位数、亏>3%/>5%比例
- 涨停/跌停/炸板数量（按各自板块 limit 口径）
- 量能：全市场成交额及 20 日量比
- 指数：上证/创业板 5/20 日趋势、20 日波动率（market_daily.parquet）
- 板块：12 关注板块内离散度、强势板块数、板块均值涨幅（自 df 聚合）
- 低吸池状态：当日候选数、前 5 日候选池已实现净收益/STOP 比例/胜率

标签：
- fwd_pool_ret: t+1..t+5 信号候选的 net_ret(s1.2_t2.0, eligible) 均值
- fwd_stop_ratio: 其中 STOP 占比

训练 2019-2021；OOF 2022-2023 冻结槽位分位；输出 market_slots.csv
（date -> slot_cap），2024 验证、2025 仅描述。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
sys.path.insert(0, OUT_DIR)
from build_event_funnel import load_data, add_features  # noqa: E402
import lightgbm as lgb  # noqa: E402

MARKET_DAILY = os.path.join(OUT_DIR, "market_daily.parquet")
EVENTS = os.path.join(OUT_DIR, "events_train_v1.parquet")
CONS = os.path.join(OUT_DIR, "board_cons.json")
SLOTS_OUT = os.path.join(OUT_DIR, "market_slots.csv")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")

LGB_R = dict(objective="regression", num_leaves=31, learning_rate=0.03,
             min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
             verbose=-1, seed=7)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def board_daily(m):
    """12 关注板块日度聚合（自 df，等权成分）。"""
    import json
    cons = json.load(open(CONS, encoding="utf-8"))
    mb = pd.DataFrame([(c, b) for b, codes in cons.items() for c in codes],
                      columns=["code", "board"])
    x = m.merge(mb, on="code", how="inner")
    lt = x["code"].map(lambda c: 0.195 if c.startswith(("30", "68"))
                       else (0.295 if c.startswith(("8", "4", "92")) else 0.095))
    x["_ld"] = x["ret1"] <= -lt + 0.005
    x["_lu"] = x["ret1"] >= lt - 0.005
    g = x.groupby(["board", "date"], sort=False)
    b = g.agg(sector_ret1=("ret1", "mean"),
              lu=("_lu", "sum"), ld=("_ld", "sum")).reset_index()
    bd = b.groupby("date", sort=False).agg(
        board_disp=("sector_ret1", "std"),
        strong_board_cnt=("sector_ret1", lambda s: int((s > 0.01).sum())),
        watch_board_mean=("sector_ret1", "mean"),
        watch_lu=("lu", "sum"), watch_ld=("ld", "sum"))
    return bd.reset_index()


def main():
    t0 = time.time()
    df = load_data()
    df = add_features(df)
    log(f"loaded {len(df)} rows {time.time()-t0:.0f}s")

    # ---- 全市场日度广度/情绪 ----
    lt = df["code"].map(lambda c: 0.195 if c.startswith(("30", "68"))
                        else (0.295 if c.startswith(("8", "4", "92")) else 0.095))
    touched_up = (df["high"] / df["pre_close"] - 1) >= lt - 0.005
    closed_up = df["ret1"] >= lt - 0.005
    g = df.groupby("date", sort=False)
    day = pd.DataFrame({
        "breadth_up": g.apply(lambda x: float((x["ret1"] > 0).mean()),
                              include_groups=False),
        "med_ret": g["ret1"].median(),
        "loss_ratio": g.apply(lambda x: float((x["ret1"] < -0.03).mean()),
                              include_groups=False),
        "big_loss_ratio": g.apply(lambda x: float((x["ret1"] < -0.05).mean()),
                                  include_groups=False),
        "limit_up_cnt": g.apply(lambda x: int(closed_up.loc[x.index].sum()),
                                include_groups=False),
        "limit_down_cnt": g.apply(lambda x: int((x["ret1"] <= -lt.loc[x.index] + 0.005).sum()),
                                  include_groups=False),
        "zhaban_cnt": g.apply(lambda x: int((touched_up.loc[x.index]
                                             & ~closed_up.loc[x.index]).sum()),
                              include_groups=False),
        "tot_amount": g["amount"].sum(),
    }).reset_index()
    day["amt_ratio20"] = day["tot_amount"] / day["tot_amount"].rolling(20).mean()

    # ---- 指数 ----
    md = pd.read_parquet(MARKET_DAILY)
    md = md[["date", "sh000001_pct_chg", "cyb399006_pct_chg"]].copy()
    md["sh_ret5"] = md["sh000001_pct_chg"].rolling(5).sum()
    md["sh_ret20"] = md["sh000001_pct_chg"].rolling(20).sum()
    md["cyb_ret5"] = md["cyb399006_pct_chg"].rolling(5).sum()
    md["cyb_ret20"] = md["cyb399006_pct_chg"].rolling(20).sum()
    md["mkt_vol20"] = md["sh000001_pct_chg"].rolling(20).std()
    day = day.merge(md.drop(columns=["sh000001_pct_chg", "cyb399006_pct_chg"]),
                    on="date", how="left")

    # ---- 12 板块聚合 ----
    bd = board_daily(df)
    day = day.merge(bd, on="date", how="left")

    # ---- 低吸池状态（A 规则候选，因果：只用 t-5..t-1 已实现 + 当日计数）----
    ev = pd.read_parquet(EVENTS, columns=["event_id", "date", "code", "subtype",
                                          "eligible_s1.2", "outcome_s1.2_t2.0",
                                          "net_ret_s1.2_t2.0"])
    ev = ev.drop_duplicates("event_id")
    pool = ev[(ev["subtype"] == "adj_ready") & (ev["eligible_s1.2"])]
    pday = pool.groupby("date", sort=False).agg(
        pool_ret=("net_ret_s1.2_t2.0", "mean"),
        pool_stop=("outcome_s1.2_t2.0", lambda s: float((s == "STOP").mean())),
        pool_win=("net_ret_s1.2_t2.0", lambda s: float((s > 0).mean())),
        pool_cnt=("event_id", "nunique")).reset_index()
    pday = pday.sort_values("date").reset_index(drop=True)
    for w in (5,):
        pday[f"trail{w}_pool_ret"] = pday["pool_ret"].rolling(w).mean().shift(1)
        pday[f"trail{w}_stop"] = pday["pool_stop"].rolling(w).mean().shift(1)
        pday[f"trail{w}_win"] = pday["pool_win"].rolling(w).mean().shift(1)
    day = day.merge(pday[["date", "pool_cnt", "trail5_pool_ret",
                          "trail5_stop", "trail5_win"]], on="date", how="left")

    # ---- 标签：t+1..t+5 候选池表现 ----
    dates = day["date"].values
    di = {d: i for i, d in enumerate(dates)}
    fwd_ret = np.full(len(day), np.nan)
    fwd_stop = np.full(len(day), np.nan)
    pday = pday.set_index("date")
    for i, d in enumerate(dates):
        nxt = dates[i + 1:i + 6]
        if not len(nxt):
            continue
        sub = pday.reindex(nxt).dropna(subset=["pool_ret"])
        if len(sub) >= 2:
            w = sub["pool_cnt"].fillna(0)
            fwd_ret[i] = np.average(sub["pool_ret"], weights=w + 1)
            fwd_stop[i] = np.average(sub["pool_stop"], weights=w + 1)
    day["fwd_pool_ret"] = fwd_ret
    day["fwd_stop_ratio"] = fwd_stop

    feats = ["breadth_up", "med_ret", "loss_ratio", "big_loss_ratio",
             "limit_up_cnt", "limit_down_cnt", "zhaban_cnt", "amt_ratio20",
             "sh_ret5", "sh_ret20", "cyb_ret5", "cyb_ret20", "mkt_vol20",
             "board_disp", "strong_board_cnt", "watch_board_mean",
             "watch_lu", "watch_ld", "pool_cnt", "trail5_pool_ret",
             "trail5_stop", "trail5_win"]
    d = day.dropna(subset=["fwd_pool_ret"]).copy()
    for c in feats:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    tr = d[(d["date"] >= 20190101) & (d["date"] <= 20211231)]
    oof = d[(d["date"] >= 20220101) & (d["date"] <= 20231231)]
    log(f"train days={len(tr)} oof days={len(oof)} feats={len(feats)}")

    mdl = lgb.LGBMRegressor(**LGB_R, n_estimators=1500)
    mdl.fit(tr[feats], tr["fwd_pool_ret"])
    oof_pred = mdl.predict(oof[feats])
    q25, q50, q75 = np.quantile(oof_pred, [0.25, 0.5, 0.75])
    log(f"OOF 槽位分位冻结: q25={q25:+.4f} q50={q50:+.4f} q75={q75:+.4f}")
    # OOF 诊断：预测 vs 实际
    diag = pd.DataFrame({"pred": oof_pred, "act": oof["fwd_pool_ret"].values,
                         "stop": oof["fwd_stop_ratio"].values})
    ic = diag["pred"].corr(diag["act"], method="spearman")
    hit = float((np.sign(diag["pred"]) == np.sign(diag["act"])).mean())
    log(f"OOF spearman(pred, fwd_ret)={ic:.3f} 符号一致率={hit:.1%}")

    def slots(pred):
        return np.where(pred >= q75, 20, np.where(pred >= q50, 10,
                        np.where(pred >= q25, 5, 0)))
    full_pred = mdl.predict(d[feats])
    out = d[["date"]].copy()
    out["slot_cap"] = slots(full_pred)
    out["pred_fwd_ret"] = full_pred
    # 2024/2025 日期也要输出（用最近可用特征日行；先 merge 全日历再 ffill）
    alld = day[["date"]].copy()
    alld = alld.merge(out, on="date", how="left")
    alld["slot_cap"] = alld["slot_cap"].ffill().fillna(10).astype(int)
    alld.to_csv(SLOTS_OUT, index=False)
    log(f"slots 分布(全期): {alld['slot_cap'].value_counts().sort_index().to_dict()}")

    # ---- 报告 ## 13 前置小节 ----
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n## 13. ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n## 13. 市场参与模型（G6/G7 市场层）\n",
         "定位：不选股，只决定每日最大槽位（R0=0/R1=5/R2=10/R3=20）。"
         "主目标=未来5日低吸候选池加权平均净收益（LightGBM 回归），"
         "特征全部 t 日收盘前可知（广度/涨跌停/炸板/量能/指数趋势/波动/12板块离散度/"
         "候选池滞后状态）。训练2019-2021，OOF2022-2023 冻结分位，2024验证、2025描述。\n",
         f"- 特征 {len(feats)} 个；OOF spearman(预测,实际) = {ic:.3f}，"
         f"符号一致率 {hit:.1%}\n",
         f"- OOF 冻结槽位分位：q25={q25:+.4f} / q50={q50:+.4f} / q75={q75:+.4f}"
         f"（ pred≥q75→20仓, ≥q50→10, ≥q25→5, 否则0）\n",
         f"- 全期槽位分布：{alld['slot_cap'].value_counts().sort_index().to_dict()}\n"]
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
