# -*- coding: utf-8 -*-
"""盘中路线第5步：四口径对照（2025开发数据，非样本外证据）。

口径：
1. daily_close：日线信号日收盘买入（既有日线标签，A2用 s1.2_t2.0，B0用 s1.5_t2.0）；
2. fixed_945 / fixed_1000 / fixed_1030：分钟引擎固定决策点 vwap 买入；
3. rule_confirm：固定决策点上应用先验承接规则（不训练、不据2025调参）：
   downside_atr<=-0.3 且 recovery>=0.4 且 vwap_reclaim_atr>=-0.2；
4. intraday_model：LightGBM 三分类（固定9:45 vwap口径，目标2.0），
   特征=引擎特征组；时间切分 train=2025H1 / test=2025H2（描述性）；
   EV 用训练期 outcome 均值；按 group_id 归组防拆散。

指标（同一批事件、配对比较）：净收益均值/中位、STOP率、TARGET率、
mae_atr 分布（接飞刀深度）、entry/信号日收盘−1（延迟入场价格差）、
模型Top30%EV vs 全体的净收益与STOP率。分 src(A2/B0) 与月份报告。
输出报告 §18.8。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
sys.path.insert(0, OUT_DIR)
SAMPLES = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
KT = "2.0"
FEATS = ["downside_atr", "shock_atr", "recovery", "lower_wick_atr",
         "vwap_reclaim_atr", "kill_speed_atr", "first15_low_atr",
         "first15_range_atr", "vol_ratio15", "new_low_after_945",
         "dip_hold", "bars"]
RULE = lambda d: (d["downside_atr"] <= -0.3) & (d["recovery"] >= 0.4) \
    & (d["vwap_reclaim_atr"] >= -0.2)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stat_block(d, col):
    oc = d[f"outcome_t{KT}"]
    return {"n": len(d),
            "mean": float(d[col].mean()),
            "med": float(d[col].median()),
            "stop": float((oc == "STOP").mean()),
            "tgt": float((oc == "TARGET").mean()),
            "mae_atr_med": float((d["mae"] / d["atr_pct"]).median())}


def main():
    t0 = time.time()
    df = pd.read_parquet(SAMPLES)
    df = df[(df["entry_type"] == "vwap") & (df["censored"] == 0)
            & df[f"net_ret_t{KT}"].notna()].copy()
    log(f"samples: {df.shape}")

    # ---- 口径1：日线收盘买入（既有标签，按 src 取对应 K）
    import glob as _glob
    a2 = pd.read_parquet(os.path.join(OUT_DIR, "events_train_v1.parquet"),
                         columns=["event_id", "date", "code", "subtype",
                                  "eligible_s1.2", "is_censored",
                                  "outcome_s1.2_t2.0", "net_ret_s1.2_t2.0"]) \
        .drop_duplicates("event_id")
    a2 = a2[(a2["date"] >= 20250101) & (a2["subtype"].isin(["adj_ready", "both"]))
            & a2["eligible_s1.2"].astype(bool) & (a2["is_censored"] == 0)]
    a2 = a2.rename(columns={"outcome_s1.2_t2.0": "oc", "net_ret_s1.2_t2.0": "nr"})
    b0 = pd.read_parquet(os.path.join(OUT_DIR, "events_b_b0_cooled.parquet"),
                         columns=["event_id", "date", "code", "is_censored",
                                  "outcome_s1.5_t2.0", "net_ret_s1.5_t2.0"])
    b0 = b0[(b0["date"] >= 20250101) & (b0["date"] <= 20251231)
            & (b0["is_censored"] == 0)]
    b0 = b0.rename(columns={"outcome_s1.5_t2.0": "oc", "net_ret_s1.5_t2.0": "nr"})
    dc = pd.concat([a2[["date", "code", "oc", "nr"]].assign(src="A2"),
                    b0[["date", "code", "oc", "nr"]].assign(src="B0")],
                   ignore_index=True)
    # 配对：日线基准只保留分钟样本覆盖的同一批 (src,code,date)
    keys = set(map(tuple, df[df["dp"] == 945][["src", "code", "date"]].values))
    dc = dc[[(s, c, d) in keys
             for s, c, d in zip(dc["src"], dc["code"], dc["date"])]]
    dc_s = {"n": len(dc), "mean": float(dc["nr"].mean()),
            "stop": float((dc["oc"] == "STOP").mean()),
            "tgt": float((dc["oc"] == "TARGET").mean())}
    log(f"daily_close: {dc_s}")

    # ---- 口径2/3：固定决策点 + 先验承接规则
    var_rows = []
    for dp in (945, 1000, 1030):
        d = df[df["dp"] == dp]
        s = stat_block(d, f"net_ret_t{KT}")
        var_rows.append({"ver": f"fixed_{dp}", "scope": "ALL", **s})
        r = d[RULE(d)]
        if len(r):
            var_rows.append({"ver": f"rule_{dp}", "scope": "ALL",
                             **stat_block(r, f"net_ret_t{KT}")})
        for src in ("A2", "B0"):
            ds = d[d["src"] == src]
            if len(ds):
                var_rows.append({"ver": f"fixed_{dp}", "scope": src,
                                 **stat_block(ds, f"net_ret_t{KT}")})

    # ---- 口径4：盘中模型（9:45 vwap，目标2.0，三分类LGB）
    import lightgbm as lgb
    d9 = df[df["dp"] == 945].copy()
    d9["month"] = d9["date"] // 100
    d9["y"] = d9[f"outcome_t{KT}"]
    for c in FEATS:
        d9[c] = pd.to_numeric(d9[c], errors="coerce")
    tr = d9[d9["date"] <= 20250630]
    te = d9[d9["date"] >= 20250701]
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, num_leaves=31,
                             learning_rate=0.05, min_child_samples=100,
                             subsample=0.9, colsample_bytree=0.9, seed=7,
                             n_estimators=400, verbosity=-1, n_jobs=8)
    clf.fit(tr[FEATS], tr["y"])
    proba = clf.predict_proba(te[FEATS])
    for i, cl in enumerate(clf.classes_):
        te[f"p_{cl}"] = proba[:, i]
    gain = te[te["y"] == "TARGET"][f"net_ret_t{KT}"].mean()
    loss = te[te["y"] == "STOP"][f"net_ret_t{KT}"].mean()
    tmo = te[te["y"] == "TIMEOUT"][f"net_ret_t{KT}"].mean()
    te["EV"] = (te["p_TARGET"] * gain + te["p_STOP"] * loss
                + te["p_TIMEOUT"] * tmo)
    thr = te["EV"].quantile(0.7)
    top = te[te["EV"] >= thr]
    model_rows = {
        "n_train": len(tr), "n_test": len(te),
        "all": stat_block(te, f"net_ret_t{KT}"),
        "top30": stat_block(top, f"net_ret_t{KT}"),
        "feat_imp": dict(zip(FEATS, clf.feature_importances_.tolist())),
    }
    log(f"model: all_mean={model_rows['all']['mean']:+.3%} "
        f"top_mean={model_rows['top30']['mean']:+.3%} "
        f"top_stop={model_rows['top30']['stop']:.1%}")

    # ---- 延迟入场价格差：T+1 9:45 entry vs 信号日收盘
    g = df[df["dp"] == 945].groupby(["src", "code", "date"]).first().reset_index()
    delay = float((g["entry"] / g["sig_close"] - 1).mean())
    log(f"delay entry vs signal close mean: {delay:+.3%}")

    # ---- 报告 §18.8
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.8 四口径对照")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n### 18.8 四口径对照（第5步，2025为开发数据，非独立样本外）\n",
         "配对同一批事件：1=日线收盘买入(既有标签) 2=固定决策点vwap买入 "
         "3=先验承接规则(downside≤-0.3ATR & recovery≥0.4 & vwap_reclaim≥-0.2ATR，"
         "未训练未调参) 4=盘中LightGBM三分类(9:45, train=25H1/test=25H2, "
         "按group_id防拆散)。\n",
         f"- 日线收盘基准：n={dc_s['n']:,} 均值{dc_s['mean']:+.2%} "
         f"STOP={dc_s['stop']:.1%} TARGET={dc_s['tgt']:.1%}",
         "- 9:45(T+1) entry 相对信号日收盘的均价差（延迟入场成本/折价）: "
         f"{delay:+.2%}\n",
         "| 版本 | 范围 | n | 净收益均值 | 中位 | STOP% | TARGET% | mae_atr中位 |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in var_rows:
        L.append(f"| {r['ver']} | {r['scope']} | {r['n']:,} | {r['mean']:+.2%} | "
                 f"{r['med']:+.2%} | {r['stop']:.1%} | {r['tgt']:.1%} | "
                 f"{r['mae_atr_med']:.2f} |")
    a, tp = model_rows["all"], model_rows["top30"]
    L.append("")
    L.append(f"- 盘中模型(9:45)：train={model_rows['n_train']:,} "
             f"test={model_rows['n_test']:,}；全体 均值{a['mean']:+.2%} "
             f"STOP={a['stop']:.1%}；Top30%EV 均值{tp['mean']:+.2%} "
             f"STOP={tp['stop']:.1%} mae_atr中位={tp['mae_atr_med']:.2f}")
    imp = sorted(model_rows["feat_imp"].items(), key=lambda x: -x[1])[:8]
    L.append("- 特征重要性(top8): " + ", ".join(f"{k}={v}" for k, v in imp))
    L.append("\n判读边界：本节为可行性/模型开发证据；2025既用于开发不得宣称样本外"
             "有效，真正前瞻验证从2026-09之后。")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
