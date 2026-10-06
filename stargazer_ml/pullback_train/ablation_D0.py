# -*- coding: utf-8 -*-
"""D0 日线背景组同池消融（v3 冻结方案 §6.6-1，§18.21）。

版本：M0（盘中12特征） vs M0+D0（17个日线背景，信号日T收盘前可知，
连续软特征无硬门槛）。头/折/过滤/晋级条件与 §18.18 相同：
AM=1000/1030、PM=1330/1400/1430/1450 分头；fold1 正向为主、fold2 反向
仅诊断；entry=vwap；标签 t2.0；price_scale_valid；Top30%实际净收益为正
且>M0、STOP/Brier 不恶化、月度≥3/4不劣、src 非单一拉动。
D0 按 (code, date=信号日T) 左连 minute_samples。
输出 ablation_D0*.csv + §18.21。
"""
import os
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
SAMPLES = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
D0 = os.path.join(OUT_DIR, "d0_daily_features_2025.parquet")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
KT = "2.0"
M0 = ["downside_atr", "shock_atr", "recovery", "lower_wick_atr",
      "vwap_reclaim_atr", "kill_speed_atr", "first15_low_atr",
      "first15_range_atr", "vol_ratio15", "new_low_after_945",
      "dip_hold", "bars"]
D0F = ["rsi14", "rsi_delta3", "bbi_distance", "bbi_slope5", "boll_pct_b",
       "down_volume_ratio5", "turnover_ratio20", "turnover_pct60",
       "pos20", "pullback_from_high5", "adv20", "amount_ratio20",
       "atr14", "rsi6", "rsi6_minus_rsi14", "boll_width20_atr"]
# 注：atr_ref 未入模（与分钟尺度 atr_pct 重复，atr_pct 已是 barrier 分母）
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
    spear = te[["EV", "nr"]].corr(method="spearman").iloc[0, 1]
    return {"n_train": len(tr), "n_test": len(te),
            "all_mean": float(te["nr"].mean()),
            "all_stop": float((te["y"] == "STOP").mean()),
            "top_mean": float(top["nr"].mean()),
            "top_stop": float((top["y"] == "STOP").mean()),
            "brier": brier(te["y"].values, proba, clf.classes_),
            "spearman": float(spear)}


def load_data(dps):
    df = pd.read_parquet(SAMPLES)
    df = df[df["dp"].isin(dps)].copy()
    d0 = pd.read_parquet(D0)
    d0["code"] = d0["code"].astype(str)
    df["code"] = df["code"].astype(str)
    df = df.merge(d0, on=["code", "date"], how="left")
    df = df[(df["entry_type"] == "vwap") & (df["censored"] == 0)
            & df[f"net_ret_t{KT}"].notna()].copy()
    if "price_scale_valid" not in df.columns:
        df["price_scale_valid"] = True
    df = df[df["price_scale_valid"].astype(bool)].copy()
    df["nr"] = df[f"net_ret_t{KT}"]
    df["y"] = df[f"outcome_t{KT}"]
    for c in M0 + D0F:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    t0 = time.time()
    all_rows, src_rows, month_rows = [], [], []
    summaries = {}
    for head, dps in HEADS.items():
        df = load_data(dps)
        models = {"M0": M0, "D0": M0 + D0F}
        log(f"[{head}] 入模 {len(df):,}；D0特征NaN率: "
            f"{ {c: round(float(df[c].isna().mean()), 4) for c in D0F} }")
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
        res = pd.DataFrame([r for r in all_rows if r["head"] == head])
        f1 = res[res["fold"] == "fold1"].set_index("model")
        f2 = res[res["fold"] == "fold2"].set_index("model")
        mrg = pd.DataFrame([r for r in month_rows if r["head"] == head])
        mm = mrg[mrg["model"] == "D0"].set_index("month")
        m0m = mrg[mrg["model"] == "M0"].set_index("month")
        sr = pd.DataFrame([r for r in src_rows if r["head"] == head
                           and r["fold"] == "fold1"])
        gains = {}
        for src in ("A2", "LEGACY"):
            sub = sr[sr["src"] == src].set_index("model")
            if len(sub) and "D0" in sub.index and "M0" in sub.index:
                gains[src] = float(sub.loc["D0", "top_mean"]
                                   - sub.loc["M0", "top_mean"])
        v1 = f1.loc["D0"]
        summaries[head] = {
            "f1_top": float(v1["top_mean"]),
            "f1_gain": float(v1["top_mean"] - f1.loc["M0", "top_mean"]),
            "f1_stop_delta": float(v1["top_stop"] - f1.loc["M0", "top_stop"]),
            "f1_brier_delta": float(v1["brier"] - f1.loc["M0", "brier"]),
            "f2_gain": float(f2.loc["D0", "top_mean"]
                             - f2.loc["M0", "top_mean"]),
            "months_not_worse": int((mm["top_mean"] >= m0m["top_mean"]).sum()),
            "n_months": int(len(mm)),
            "src_gains": gains,
            "pass": bool(v1["top_mean"] > 0
                         and v1["top_mean"] > f1.loc["M0", "top_mean"]
                         and v1["top_stop"] <= f1.loc["M0", "top_stop"] + 1e-9
                         and v1["brier"] <= f1.loc["M0", "brier"] + 1e-9
                         and gains and all(g > 0 for g in gains.values()))}
        log(f"[{head}] verdict: {summaries[head]}")
    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(OUT_DIR, "ablation_D0.csv"), index=False)
    pd.DataFrame(src_rows).to_csv(os.path.join(OUT_DIR,
                                               "ablation_D0_by_src.csv"),
                                  index=False)
    pd.DataFrame(month_rows).to_csv(os.path.join(OUT_DIR,
                                                 "ablation_D0_by_month.csv"),
                                    index=False)

    # ---- 报告 §18.21 ----
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.21 ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n### 18.21 D0 日线背景组同池消融（v3方案第1阶段，2025开发）\n",
         "D0=17个盘后可知日线特征（信号日T收盘及以前）：rsi14/rsi_delta3、"
         "bbi_distance/bbi_slope5、boll_pct_b/boll_width20_atr、"
         "down_volume_ratio5、turnover_ratio20/turnover_pct60、"
         "pos20/pullback_from_high5、adv20/amount_ratio20、atr14、"
         "rsi6/rsi6_minus_rsi14（新增3个）；atr_ref未入模（与分钟atr_pct重复）。"
         "连续软特征，无硬门槛。AM/PM分头，fold1为主/fold2诊断，"
         "晋级条件同 §18.18（含Top30%为正、Brier不恶化、src双正）。\n",
         "| 头 | fold | 版本 | Top30%实际净收益 | Top30%STOP | 全体净收益 | "
         "Brier | Spearman |",
         "|---|---|---|---:|---:|---:|---:|---:|"]
    for _, r in res.iterrows():
        L.append(f"| {r['head']} | {r['fold']} | {r['model']} | "
                 f"{r['top_mean']:+.2%} | {r['top_stop']:.1%} | "
                 f"{r['all_mean']:+.2%} | {r['brier']:.4f} | "
                 f"{r['spearman']:+.3f} |")
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
    L.append("跨月（fold1）：")
    L.append("| 头 | 月份 | 版本 | n_test | Top30%实际净收益 | Top30%STOP |")
    L.append("|---|---:|---|---:|---:|---:|")
    for _, r in mr.iterrows():
        L.append(f"| {r['head']} | {r['month']} | {r['model']} | "
                 f"{r['n_test']:,} | {r['top_mean']:+.2%} | "
                 f"{r['top_stop']:.1%} |")
    L.append("")
    L.append("晋级判定：")
    for head, v in summaries.items():
        L.append(f"- {head} D0: fold1 Top30% {v['f1_top']:+.2%}（增益 "
                 f"{v['f1_gain']:+.2%}）、ΔSTOP {v['f1_stop_delta']:+.1%}、"
                 f"ΔBrier {v['f1_brier_delta']:+.4f}、fold2增益 "
                 f"{v['f2_gain']:+.2%}、月份不劣 {v['months_not_worse']}/"
                 f"{v['n_months']}、src增益 {v['src_gains']} → "
                 f"{'通过' if v['pass'] else '未通过'}")
    L.append("\n用途：D0 是后续风险头/机会头的基座组成部分（R0=O0=M0+D0），"
             "无论本消融是否通过，D0 进入基座的方案口径不变；本消融只决定"
             "D0 是否带来独立增量。明细：ablation_D0*.csv")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
