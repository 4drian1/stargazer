# -*- coding: utf-8 -*-
"""T3/T4 冻结复核：2026-01~08 时间复核（训练=2025H1，与开发fold1同池）。

性质：**只能否定、不可确认**。2026前8个月已被项目观察；若按2026结果再调
特征/阈值即构成对新假设的再拟合，禁止。晋级裁决的唯一依据是2025开发折
（§18.18/§18.19）；本节只回答"2025看到的增量是否在2026已观察区间干净复现"。

复核对象（§18.18 唯一通过全部硬条件的版本）：**PM 头 T3-only**。
对照：同头 M0（基线）。AM 头无晋级版本，不复核（报告 §18.18 已否决）。

口径：PM头 dp∈{1330,1400,1430,1450}；entry=vwap；标签 t2.0；
EV=训练期类均值加权；Top30%=测试期EV的0.7分位；表头用"Top30%实际净收益"。
输出 review_2026_T3T4.csv + §18.20。
"""
import os
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
KT = "2.0"
TR25 = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
TE26 = os.path.join(OUT_DIR, "minute_labels", "minute_samples_2026.parquet")
T3_25 = os.path.join(OUT_DIR, "minute_t3_vwap_2025.parquet")
T3_26 = os.path.join(OUT_DIR, "minute_t3_vwap_2026.parquet")
T4_25 = os.path.join(OUT_DIR, "minute_t4_proxy_2025.parquet")
T4_26 = os.path.join(OUT_DIR, "minute_t4_proxy_2026.parquet")
TRAIN_MAX = 20250831
PM_DPS = (1330, 1400, 1430, 1450)
M0 = ["downside_atr", "shock_atr", "recovery", "lower_wick_atr",
      "vwap_reclaim_atr", "kill_speed_atr", "first15_low_atr",
      "first15_range_atr", "vol_ratio15", "new_low_after_945",
      "dip_hold", "bars"]
V3 = ["v3_first_reclaim_min", "v3_hold_after_reclaim_min", "v3_cross_count",
      "v3_longest_above_min", "v3_rebreak_flag", "v3_rebreak_depth_atr",
      "v3_above_min_frac", "v3_dp_vwap_dist_atr", "v3_above_vol_frac",
      "v3_rel_mkt_after_reclaim"]
T4F = ["t4_proxy_available", "t4_basket_cumret_dp", "t4_stock_excess_dp",
       "t4_basket_up_ratio", "t4_basket_h2c", "t4_basket_above_vwap",
       "t4_leader_cumret_dp", "t4_center_cumret_dp", "t4_leader_h2c",
       "t4_stock_excess_leader"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_pool(samples, t3_path, t4_path):
    df = pd.read_parquet(samples)
    df = df[df["dp"].isin(PM_DPS)].copy()
    df = df.merge(pd.read_parquet(t3_path), on=["code", "dp_date", "dp"],
                  how="left")
    df = df.merge(pd.read_parquet(t4_path), on=["code", "dp_date", "dp"],
                  how="left")
    df = df[(df["entry_type"] == "vwap") & (df["censored"] == 0)
            & df[f"net_ret_t{KT}"].notna()].copy()
    if "price_scale_valid" not in df.columns:
        df["price_scale_valid"] = True
    df = df[df["price_scale_valid"].astype(bool)].copy()
    df["nr"] = df[f"net_ret_t{KT}"]
    df["y"] = df[f"outcome_t{KT}"]
    for c in M0 + V3 + T4F:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def brier(y_true, proba, classes):
    idx = {c: i for i, c in enumerate(classes)}
    oh = np.zeros_like(proba)
    for r, y in enumerate(y_true):
        oh[r, idx[y]] = 1.0
    return float(((proba - oh) ** 2).sum(axis=1).mean())


def fit_eval(tr, te, feats):
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
    spear = te[["EV", "nr"]].corr(method="spearman").iloc[0, 1]
    return {"n": len(te),
            "all_mean": float(te["nr"].mean()),
            "all_stop": float((te["y"] == "STOP").mean()),
            "top_mean": float(top["nr"].mean()),
            "top_stop": float((top["y"] == "STOP").mean()),
            "brier": brier(te["y"].values, proba, clf.classes_),
            "spearman": float(spear)}


def main():
    t0 = time.time()
    tr25 = load_pool(TR25, T3_25, T4_25)
    tr = tr25[tr25["date"] <= TRAIN_MAX]
    te26 = load_pool(TE26, T3_26, T4_26)
    te26 = te26[(te26["date"] >= 20260101)
                & (te26["date"] <= 20260831)].copy()
    log(f"train25H1={len(tr):,} test2026={len(te26):,} "
        f"({te26['src'].value_counts().to_dict()})")
    models = {"M0": M0, "T3-only": M0 + V3, "T4-only": M0 + T4F,
              "T3+T4": M0 + V3 + T4F}
    rows = []
    fitted = {}
    for ver, feats in models.items():
        r = fit_eval(tr, te26, feats)
        fitted[ver] = r
        log(f"{ver}: all {r['all_mean']:+.3%} Top30% {r['top_mean']:+.3%}/"
            f"STOP {r['top_stop']:.1%} Brier {r['brier']:.4f}")
        rows.append({"scope": "ALL", "grp": "ALL", "model": ver, **r})
    for src in ("A2", "B0", "LEGACY"):
        tes = te26[te26["src"] == src]
        trs = tr[tr["src"] == src]
        if len(tes) < 100 or len(trs) < 100:
            continue
        for ver, feats in models.items():
            r = fit_eval(trs, tes, feats)
            rows.append({"scope": "src", "grp": src, "model": ver, **r})
    te26m = te26.copy()
    te26m["month"] = te26m["date"] // 100
    for mo in sorted(te26m["month"].unique()):
        tem = te26m[te26m["month"] == mo]
        for ver, feats in models.items():
            r = fit_eval(tr, tem, feats)
            rows.append({"scope": "month", "grp": int(mo), "model": ver, **r})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "review_2026_T3T4.csv"), index=False)

    # ---- 报告 §18.20 ----
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.20 ")
    if idx != -1:
        old = old[:idx] + "\n"
    m0, t3 = fitted["M0"], fitted["T3-only"]
    L = ["\n### 18.20 T3/T4 冻结复核：2026-01~08（PM头，只能否定，不可确认）\n",
         "口径：训练池=2025H1（与开发fold1相同），PM头四决策点合并，vwap，"
         "t2.0，EV=训练期类均值，Top30%=测试期EV的0.7分位。复核对象=§18.18 "
         "唯一通过全部硬条件的 PM T3-only；同时列出 T4-only/T3+T4 仅作诊断"
         "（§18.18 已否决，不因本表翻盘）。\n",
         "| 版本 | n | 全体净收益 | 全体STOP | Top30%实际净收益 | Top30%STOP | "
         "Brier | Spearman |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for ver, r in fitted.items():
        L.append(f"| {ver} | {r['n']:,} | {r['all_mean']:+.2%} | "
                 f"{r['all_stop']:.1%} | {r['top_mean']:+.2%} | "
                 f"{r['top_stop']:.1%} | {r['brier']:.4f} | "
                 f"{r['spearman']:+.3f} |")
    L.append("")
    L.append("分来源：")
    L.append("| src | 版本 | n | Top30%实际净收益 | Top30%STOP | Brier |")
    L.append("|---|---|---:|---:|---:|---:|")
    for _, r in res[res["scope"] == "src"].iterrows():
        L.append(f"| {r['grp']} | {r['model']} | {r['n']:,} | "
                 f"{r['top_mean']:+.2%} | {r['top_stop']:.1%} | "
                 f"{r['brier']:.4f} |")
    L.append("")
    L.append("分月（池化模型逐月）：")
    L.append("| 月份 | 版本 | n | Top30%实际净收益 | Top30%STOP |")
    L.append("|---:|---|---:|---:|---:|")
    for _, r in res[res["scope"] == "month"].iterrows():
        L.append(f"| {r['grp']} | {r['model']} | {r['n']:,} | "
                 f"{r['top_mean']:+.2%} | {r['top_stop']:.1%} |")
    L.append("")
    L.append(f"- 判读：2025开发折 PM T3-only 相对 M0 增益 "
             f"+{(t3['top_mean']-m0['top_mean']):.2%}（§18.18）。本节为"
             f"2026已观察区间：ΔTop30%="
             f"{(t3['top_mean']-m0['top_mean']):+.2%}、ΔSTOP="
             f"{(t3['top_stop']-m0['top_stop']):+.1%}、ΔBrier="
             f"{(t3['brier']-m0['brier']):+.4f}。若方向反转或STOP显著恶化→"
             f"按可否定规则降级；方向一致也只算'未否定'，不构成确认。"
             f"明细：review_2026_T3T4.csv")
    L.append("- 前瞻边界：2026-01~08已观察；T3/T4若按本节结果调整即再拟合。"
             "真正前瞻影子期从本版代码冻结后的下一个交易日起。")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
