# -*- coding: utf-8 -*-
"""T2 v2 冻结复核：2026-01~08 时间复核（训练=2025H1，与开发fold1同池）。

性质：**只能否定、不可确认**。2026前8个月已被项目观察；T2阈值/特征若再按
2026结果调整即构成对新假设的再拟合，禁止。2026-09起数据仅作影子诊断。
晋级裁决的唯一依据是2025开发折（见 §18.15）；本节只回答"2025看到的增量
是否在2026已观察区间干净复现"。

口径：dp=1000/1030 分开；entry=vwap；标签 t2.0；EV=训练期类均值加权；
Top30%=测试期EV的0.7分位；表头用"Top30%实际净收益"。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
KT = "2.0"
TR25 = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
PF25 = os.path.join(OUT_DIR, "minute_path_features_2025.parquet")
ST25 = os.path.join(OUT_DIR, "minute_second_test_2025.parquet")
TE26 = os.path.join(OUT_DIR, "minute_labels", "minute_samples_2026.parquet")
PF26 = os.path.join(OUT_DIR, "minute_features", "minute_path_features_2026.parquet")
ST26 = os.path.join(OUT_DIR, "minute_second_test_2026.parquet")
M0 = ["downside_atr", "shock_atr", "recovery", "lower_wick_atr",
      "vwap_reclaim_atr", "kill_speed_atr", "first15_low_atr",
      "first15_range_atr", "vol_ratio15", "new_low_after_945",
      "dip_hold", "bars"]
T1_ADD = ["pf_ret1m", "pf_ret3m", "pf_ret5m", "pf_decline_speed",
          "pf_decel", "pf_low2_minus_low1", "pf_has_two_lows"]
COMMON_ST = ["first_leg_shock_atr", "first_low_depth_atr"]
_PAIR = ("second_low_minus_first_low_atr", "rebound_before_second_test_atr",
         "second_pullback_ratio", "second_volume_vs_first_volume",
         "second_down_volume_vs_first_down_volume",
         "second_range_vs_first_range", "second_close_location",
         "second_vwap_distance_atr", "minutes_between_lows",
         "new_low_on_second_leg")
TRAIN_MAX = 20250831


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_pool(samples, pf_path, st_path, dp):
    df = pd.read_parquet(samples)
    df = df[df["dp"] == dp].copy()
    df = df.merge(pd.read_parquet(pf_path), on=["code", "dp_date", "dp"],
                  how="left")
    df = df.merge(pd.read_parquet(st_path), on=["code", "dp_date", "dp"],
                  how="left")
    df = df[(df["entry_type"] == "vwap") & (df["censored"] == 0)
            & df[f"net_ret_t{KT}"].notna()].copy()
    if "price_scale_valid" not in df.columns:
        df["price_scale_valid"] = True
    df = df[df["price_scale_valid"].astype(bool)].copy()
    df["nr"] = df[f"net_ret_t{KT}"]
    df["y"] = df[f"outcome_t{KT}"]
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


def feat_sets(dp):
    t2 = COMMON_ST + [f"a_{k}" for k in _PAIR] \
        + ([f"b_{k}" for k in _PAIR] if dp == 1030 else [])
    return {"M0": M0, "T1": M0 + T1_ADD, "T2-only": M0 + t2,
            "T1+T2": M0 + T1_ADD + t2}


def main():
    t0 = time.time()
    rows = []
    for dp in (1000, 1030):
        tr25 = load_pool(TR25, PF25, ST25, dp)
        tr = tr25[tr25["date"] <= TRAIN_MAX]
        te26 = load_pool(TE26, PF26, ST26, dp)
        te26 = te26[(te26["date"] >= 20260101)
                    & (te26["date"] <= 20260831)].copy()
        log(f"dp{dp}: train25H1={len(tr):,} test2026={len(te26):,} "
            f"({te26['src'].value_counts().to_dict()})")
        models = feat_sets(dp)
        fitted = {}
        for ver, feats in models.items():
            fitted[ver] = (feats, fit_eval(tr, te26, feats))
            r = fitted[ver][1]
            log(f"dp{dp} {ver}: all {r['all_mean']:+.3%} "
                f"Top30% {r['top_mean']:+.3%}/STOP {r['top_stop']:.1%} "
                f"Brier {r['brier']:.4f}")
            rows.append({"dp": dp, "scope": "ALL", "grp": "ALL",
                         "model": ver, **r})
        for src in ("A2", "B0", "LEGACY"):
            tes = te26[te26["src"] == src]
            trs = tr[tr["src"] == src]
            if len(tes) < 100 or len(trs) < 100:
                continue
            for ver, feats in models.items():
                r = fit_eval(trs, tes, feats)
                rows.append({"dp": dp, "scope": "src", "grp": src,
                             "model": ver, **r})
        te26m = te26.copy()
        te26m["month"] = te26m["date"] // 100
        for mo in sorted(te26m["month"].unique()):
            tem = te26m[te26m["month"] == mo]
            for ver, feats in models.items():
                r = fit_eval(tr, tem, feats)
                rows.append({"dp": dp, "scope": "month", "grp": int(mo),
                             "model": ver, **r})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "review_2026_T2.csv"), index=False)

    # ---- 报告 §18.16 ----
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.16 ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n### 18.16 T2 v2 冻结复核：2026-01~08（只能否定，不可确认）\n",
         "口径：训练池=2025H1（与开发fold1相同），dp1000/1030分开，vwap，"
         "t2.0，EV=训练期类均值，Top30%=测试期EV的0.7分位，四版本同池。\n",
         "| dp | 版本 | n | 全体净收益 | 全体STOP | Top30%实际净收益 | "
         "Top30%STOP | Brier | Spearman |",
         "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in res[res["scope"] == "ALL"].iterrows():
        L.append(f"| {r['dp']} | {r['model']} | {r['n']:,} | "
                 f"{r['all_mean']:+.2%} | {r['all_stop']:.1%} | "
                 f"{r['top_mean']:+.2%} | {r['top_stop']:.1%} | "
                 f"{r['brier']:.4f} | {r['spearman']:+.3f} |")
    L.append("")
    L.append("分来源：")
    L.append("| dp | src | 版本 | n | Top30%实际净收益 | Top30%STOP | Brier |")
    L.append("|---:|---|---|---:|---:|---:|---:|")
    for _, r in res[res["scope"] == "src"].iterrows():
        L.append(f"| {r['dp']} | {r['grp']} | {r['model']} | {r['n']:,} | "
                 f"{r['top_mean']:+.2%} | {r['top_stop']:.1%} | "
                 f"{r['brier']:.4f} |")
    L.append("")
    L.append("分月（池化模型逐月）：")
    L.append("| dp | 月份 | 版本 | n | Top30%实际净收益 | Top30%STOP |")
    L.append("|---:|---:|---|---:|---:|---:|")
    for _, r in res[res["scope"] == "month"].iterrows():
        L.append(f"| {r['dp']} | {r['grp']} | {r['model']} | {r['n']:,} | "
                 f"{r['top_mean']:+.2%} | {r['top_stop']:.1%} |")
    L.append("")
    L.append("- 判读：对照 §18.15 的2025开发结论——2025晋级版本若在2026"
             "已观察区间反转（方向变负或STOP显著恶化），按'可否定'规则"
             "降级/停止投入；若方向一致也只能算'未否定'，不构成确认。"
             "明细：review_2026_T2.csv")
    L.append("- 前瞻边界：特征设计已使用2026-09-04/07/08三段模板（E0素材），"
             "真正新的前瞻影子期从本版代码冻结后的下一个交易日起。")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
