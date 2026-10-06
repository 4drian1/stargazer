# -*- coding: utf-8 -*-
"""T3/T4 条件组消融（2025开发，评审冻结协议，§18.19）。

触发条件（§18.18）：T3-only 在 PM 头通过全部硬条件 → 允许测试条件组；
T4-only 未通过。本脚本回答：**已失败的 T1/T2 加回 T3/T4 基座是否退化**。

版本（同一事件/标签t2.0/group_id切分/LGB参数）：
  M0        = 基础分钟模型v1（12个）
  T3+T4     = M0 + v3_* + t4_*（§18.18 基座）
  T1+T3+T4  = + 路径速度特征 pf_*（7个，已扩PM决策点）
  T2+T3+T4  = + T2承接结构 a_*/b_*（**仅AM头**：T2固定段9:30-9:45/
              9:45-10:00/10:00-10:30，PM决策点无新信息，不评估，避免
              把10:30值当PM特征造成假重复）

头/折/过滤/晋级条件与 §18.18 完全相同（Top30%实际净收益为正且>M0基线、
STOP不恶化、Brier不恶化、月度非单月、反向折不反转、src非单一拉动；
条件组额外要求不低于同头 T3+T4 基座）。
输出 ablation_T3T4_cond.csv + §18.19。
"""
import os
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
SAMPLES = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
PF = os.path.join(OUT_DIR, "minute_path_features_2025.parquet")
ST = os.path.join(OUT_DIR, "minute_second_test_2025.parquet")
T3 = os.path.join(OUT_DIR, "minute_t3_vwap_2025.parquet")
T4 = os.path.join(OUT_DIR, "minute_t4_proxy_2025.parquet")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
KT = "2.0"
M0 = ["downside_atr", "shock_atr", "recovery", "lower_wick_atr",
      "vwap_reclaim_atr", "kill_speed_atr", "first15_low_atr",
      "first15_range_atr", "vol_ratio15", "new_low_after_945",
      "dip_hold", "bars"]
T1_ADD = ["pf_ret1m", "pf_ret3m", "pf_ret5m", "pf_decline_speed",
          "pf_decel", "pf_low2_minus_low1", "pf_has_two_lows"]
V3 = ["v3_first_reclaim_min", "v3_hold_after_reclaim_min", "v3_cross_count",
      "v3_longest_above_min", "v3_rebreak_flag", "v3_rebreak_depth_atr",
      "v3_above_min_frac", "v3_dp_vwap_dist_atr", "v3_above_vol_frac",
      "v3_rel_mkt_after_reclaim"]
T4F = ["t4_proxy_available", "t4_basket_cumret_dp", "t4_stock_excess_dp",
       "t4_basket_up_ratio", "t4_basket_h2c", "t4_basket_above_vwap",
       "t4_leader_cumret_dp", "t4_center_cumret_dp", "t4_leader_h2c",
       "t4_stock_excess_leader"]
COMMON_ST = ["first_leg_shock_atr", "first_low_depth_atr"]
A_ST = [f"a_{k}" for k in ("second_low_minus_first_low_atr",
                           "rebound_before_second_test_atr",
                           "second_pullback_ratio",
                           "second_volume_vs_first_volume",
                           "second_down_volume_vs_first_down_volume",
                           "second_range_vs_first_range",
                           "second_close_location",
                           "second_vwap_distance_atr",
                           "minutes_between_lows", "new_low_on_second_leg")]
B_ST = [f"b_{k}" for k in ("second_low_minus_first_low_atr",
                           "rebound_before_second_test_atr",
                           "second_pullback_ratio",
                           "second_volume_vs_first_volume",
                           "second_down_volume_vs_first_down_volume",
                           "second_range_vs_first_range",
                           "second_close_location",
                           "second_vwap_distance_atr",
                           "minutes_between_lows", "new_low_on_second_leg")]
HEADS = {"AM": (1000, 1030), "PM": (1330, 1400, 1430, 1450)}
FOLDS = [("fold1", 20250101, 20250831, 20250901, 20251231),
         ("fold2", 20250901, 20251231, 20250101, 20250831)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def brier(y_true, proba, classes):
    idx = {c: i for i, c in enumerate(classes)}
    oh = np.zeros_like(proba)
    for r, y in enumerate(y_true):
        oh[r, idx[y]] = 1.0
    return float(((proba - oh) ** 2).sum(axis=1).mean())


def run_model(tr, te, feats):
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, num_leaves=31,
                             learning_rate=0.05, min_child_samples=100,
                             subsample=0.9, colsample_bytree=0.9, seed=7,
                             n_estimators=400, verbosity=-1, n_jobs=8)
    clf.fit(tr[feats], tr["y"])
    proba = clf.predict_proba(te[feats])
    ev = np.zeros(len(te))
    for i, cl in enumerate(clf.classes_):
        gain = tr.loc[tr["y"] == cl, "nr"].mean()
        ev += proba[:, i] * gain
    te = te.copy()
    te["EV"] = ev
    thr = te["EV"].quantile(0.7)
    top = te[te["EV"] >= thr]
    oc_top = top[f"outcome_t{KT}"]
    spear = te[["EV", "nr"]].corr(method="spearman").iloc[0, 1]
    return {"n_train": len(tr), "n_test": len(te),
            "all_mean": float(te["nr"].mean()),
            "top_mean": float(top["nr"].mean()),
            "top_stop": float((oc_top == "STOP").mean()),
            "top_n": len(top),
            "brier": brier(te["y"].values, proba, clf.classes_),
            "spearman": float(spear)}


def load_data(dps):
    df = pd.read_parquet(SAMPLES)
    df = df[df["dp"].isin(dps)].copy()
    df = df.merge(pd.read_parquet(PF), on=["code", "dp_date", "dp"], how="left")
    df = df.merge(pd.read_parquet(ST), on=["code", "dp_date", "dp"], how="left")
    df = df.merge(pd.read_parquet(T3), on=["code", "dp_date", "dp"], how="left")
    df = df.merge(pd.read_parquet(T4), on=["code", "dp_date", "dp"], how="left")
    df = df[(df["entry_type"] == "vwap") & (df["censored"] == 0)
            & df[f"net_ret_t{KT}"].notna()].copy()
    if "price_scale_valid" not in df.columns:
        df["price_scale_valid"] = True
    df = df[df["price_scale_valid"].astype(bool)].copy()
    df["nr"] = df[f"net_ret_t{KT}"]
    df["y"] = df[f"outcome_t{KT}"]
    for c in M0 + T1_ADD + V3 + T4F + COMMON_ST + A_ST + B_ST:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def feat_sets(head):
    base = M0 + V3 + T4F
    m = {"M0": M0, "T3+T4": base,
         "T1+T3+T4": base + T1_ADD}
    if head == "AM":
        t2 = COMMON_ST + A_ST + B_ST  # 1000行b_*为NaN，LGBM自动处理
        m["T2+T3+T4"] = base + t2
    return m


def main():
    t0 = time.time()
    all_rows, src_rows, month_rows = [], [], []
    for head, dps in HEADS.items():
        df = load_data(dps)
        models = feat_sets(head)
        log(f"[{head}] 入模 {len(df):,}")
        for fname, ftr0, ftr1, t0d, t1d in FOLDS:
            tr = df[(df["date"] >= ftr0) & (df["date"] <= ftr1)]
            te = df[(df["date"] >= t0d) & (df["date"] <= t1d)]
            for ver, feats in models.items():
                r = run_model(tr, te, feats)
                all_rows.append({"head": head, "fold": fname, "model": ver,
                                 **r})
                log(f"[{head}] {fname} {ver}: top={r['top_mean']:+.3%} "
                    f"stop={r['top_stop']:.1%} brier={r['brier']:.4f}")
        ftr0, ftr1, t0d, t1d = FOLDS[0][1:]
        tr = df[(df["date"] >= ftr0) & (df["date"] <= ftr1)]
        te = df[(df["date"] >= t0d) & (df["date"] <= t1d)]
        for src in ("A2", "B0", "LEGACY"):
            trs, tes = tr[tr["src"] == src], te[te["src"] == src]
            if len(tes) < 500 or len(trs) < 500:
                continue
            for ver, feats in models.items():
                r = run_model(trs, tes, feats)
                src_rows.append({"head": head, "fold": "fold1", "src": src,
                                 "model": ver, **r})
        te2 = te.copy()
        te2["month"] = te2["date"] // 100
        for mo in sorted(te2["month"].unique()):
            tem = te2[te2["month"] == mo]
            for ver in models:
                r = run_model(tr, tem, models[ver])
                month_rows.append({"head": head, "month": int(mo),
                                   "model": ver, **r})
    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(OUT_DIR, "ablation_T3T4_cond.csv"), index=False)
    pd.DataFrame(src_rows).to_csv(
        os.path.join(OUT_DIR, "ablation_T3T4_cond_by_src.csv"), index=False)
    pd.DataFrame(month_rows).to_csv(
        os.path.join(OUT_DIR, "ablation_T3T4_cond_by_month.csv"), index=False)

    # ---- 报告 §18.19 ----
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.19 ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n### 18.19 T3/T4 条件组消融（T1/T2 加回基座，2025开发）\n",
         "触发：§18.18 中 PM T3-only 通过全部硬条件 → 按冻结协议测试条件组；"
         "T4-only 未通过。版本：M0 / T3+T4（基座）/ T1+T3+T4 / "
         "T2+T3+T4（**仅AM头**，T2固定段止于10:30，PM决策点无新信息不评估）。\n",
         "判读规则（冻结）：条件组只有在同头同时满足 §18.18 全部晋级条件"
         "且 fold1 Top30%实际净收益不低于同头 T3+T4 基座时才允许替换基座；"
         "否则维持 §18.18 结论（PM 头仅 T3-only 晋级候选）。\n",
         "| 头 | fold | 版本 | Top30%实际净收益 | Top30%STOP | Brier | Spearman |",
         "|---|---|---|---:|---:|---:|---:|"]
    for _, r in res.iterrows():
        L.append(f"| {r['head']} | {r['fold']} | {r['model']} | "
                 f"{r['top_mean']:+.2%} | {r['top_stop']:.1%} | "
                 f"{r['brier']:.4f} | {r['spearman']:+.3f} |")
    sr = pd.DataFrame(src_rows)
    L.append("")
    L.append("src分层（fold1）：")
    L.append("| 头 | src | 版本 | n_test | Top30%实际净收益 | Top30%STOP |")
    L.append("|---|---|---|---:|---:|---:|")
    for _, r in sr.iterrows():
        L.append(f"| {r['head']} | {r['src']} | {r['model']} | "
                 f"{r['n_test']:,} | {r['top_mean']:+.2%} | "
                 f"{r['top_stop']:.1%} |")
    mr = pd.DataFrame(month_rows)
    L.append("")
    L.append("跨月（fold1，池化模型逐月）：")
    L.append("| 头 | 月份 | 版本 | n_test | Top30%实际净收益 | Top30%STOP |")
    L.append("|---|---:|---|---:|---:|---:|")
    for _, r in mr.iterrows():
        L.append(f"| {r['head']} | {r['month']} | {r['model']} | "
                 f"{r['n_test']:,} | {r['top_mean']:+.2%} | "
                 f"{r['top_stop']:.1%} |")
    L.append("\n明细：ablation_T3T4_cond*.csv。2026复核见 §18.20（只能否定）。")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
