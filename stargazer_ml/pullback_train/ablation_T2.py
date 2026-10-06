# -*- coding: utf-8 -*-
"""T2 v2 四组固定消融（2025开发数据，评审冻结设计，2026-09-08）。

消融结构（固定事件、固定决策点(10:00/10:30分开评估)、固定标签t2.0、
固定group_id时序切分、固定LGB参数，只变特征）：
  M0     = 基础分钟模型v1特征（12个）
  T1     = M0 + 路径速度特征（7个 pf_*）
  T2-only= M0 + T2承接结构特征（dp1000: 2首段+10个a_*；dp1030: 2首段+20个a_*/b_*）
  T1+T2  = M0 + pf_* + T2特征
样本集合四版本完全一致；T2特征NaN只来自数据缺失（段无bar），不做硬过滤。

时间折（双向，fold1正向为主要依据，fold2反向仅诊断）：
  fold1: train=20250101-20250831, test=20250901-20251231
  fold2: train=20250901-20251231, test=20250101-20250831
同一事件(group_id)的各决策点行同折（date在组内恒定，按date切分天然成立）。

晋级条件（评审冻结）：Top30%实际净收益>M0 且 STOP不恶化 且 Brier不恶化
且 月度改善非单月贡献 且 反向折不明显反转 且 A2与LEGACY非单一来源拉动。
术语：Top30%按模型EV（事前预测）选样，"Top30%实际净收益"=选中样本的
标签净收益均值（事后结果），表头不用"净EV"以免混淆。

输出：ablation_T2.csv / ablation_T2_by_src.csv / ablation_T2_by_month.csv；
报告写入 §18.15。2025仅开发证据；真正前瞻验证从影子期（代码冻结后）开始。
"""
import os
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
SAMPLES = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
PF = os.path.join(OUT_DIR, "minute_path_features_2025.parquet")
ST = os.path.join(OUT_DIR, "minute_second_test_2025.parquet")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
KT = "2.0"
M0 = ["downside_atr", "shock_atr", "recovery", "lower_wick_atr",
      "vwap_reclaim_atr", "kill_speed_atr", "first15_low_atr",
      "first15_range_atr", "vol_ratio15", "new_low_after_945",
      "dip_hold", "bars"]
T1_ADD = ["pf_ret1m", "pf_ret3m", "pf_ret5m", "pf_decline_speed",
          "pf_decel", "pf_low2_minus_low1", "pf_has_two_lows"]
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
    oc_all = te[f"outcome_t{KT}"]
    spear = te[["EV", "nr"]].corr(method="spearman").iloc[0, 1]
    return {"n_train": len(tr), "n_test": len(te),
            "all_mean": float(te["nr"].mean()),
            "all_stop": float((oc_all == "STOP").mean()),
            "top_mean": float(top["nr"].mean()),
            "top_stop": float((oc_top == "STOP").mean()),
            "top_n": len(top),
            "brier": brier(te["y"].values, proba, clf.classes_),
            "spearman": float(spear)}


def load_data(dp):
    df = pd.read_parquet(SAMPLES)
    pf = pd.read_parquet(PF)
    st = pd.read_parquet(ST)
    df = df[df["dp"] == dp].copy()
    df = df.merge(pf, on=["code", "dp_date", "dp"], how="left")
    df = df.merge(st, on=["code", "dp_date", "dp"], how="left")
    df = df[(df["entry_type"] == "vwap") & (df["censored"] == 0)
            & df[f"net_ret_t{KT}"].notna()].copy()
    if "price_scale_valid" not in df.columns:
        df["price_scale_valid"] = True
    df = df[df["price_scale_valid"].astype(bool)].copy()
    df["nr"] = df[f"net_ret_t{KT}"]
    df["y"] = df[f"outcome_t{KT}"]
    for c in M0 + T1_ADD + COMMON_ST + A_ST + B_ST:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def feat_sets(dp):
    t2 = COMMON_ST + A_ST + (B_ST if dp == 1030 else [])
    return {"M0": M0, "T1": M0 + T1_ADD, "T2-only": M0 + t2,
            "T1+T2": M0 + T1_ADD + t2}


def main():
    t0 = time.time()
    all_rows, src_rows, month_rows = [], [], []
    summaries = {}
    for dp in (1000, 1030):
        df = load_data(dp)
        models = feat_sets(dp)
        nan_chk = {c: round(float(df[c].isna().mean()), 4)
                   for c in models["T1+T2"] if c in df.columns}
        log(f"dp{dp} 入模 {len(df):,}（{df['src'].value_counts().to_dict()}）；"
            f"T2特征NaN率: {nan_chk}")
        for fname, ftr0, ftr1, t0d, t1d in FOLDS:
            tr = df[(df["date"] >= ftr0) & (df["date"] <= ftr1)]
            te = df[(df["date"] >= t0d) & (df["date"] <= t1d)]
            for ver, feats in models.items():
                r = run_model(tr, te, feats)
                all_rows.append({"dp": dp, "fold": fname, "model": ver, **r})
                log(f"dp{dp} {fname} {ver}: top={r['top_mean']:+.3%} "
                    f"stop={r['top_stop']:.1%} brier={r['brier']:.4f}")
        # src 分层（fold1，按来源分别训练+测试，同 T1 协议）
        ftr0, ftr1, t0d, t1d = FOLDS[0][1:]
        tr = df[(df["date"] >= ftr0) & (df["date"] <= ftr1)]
        te = df[(df["date"] >= t0d) & (df["date"] <= t1d)]
        for src in ("A2", "B0", "LEGACY"):
            trs, tes = tr[tr["src"] == src], te[te["src"] == src]
            if len(tes) < 500 or len(trs) < 500:
                continue
            for ver, feats in models.items():
                r = run_model(trs, tes, feats)
                src_rows.append({"dp": dp, "fold": "fold1", "src": src,
                                 "model": ver, **r})
        # 跨月（fold1，池化模型逐月测试）
        te2 = te.copy()
        te2["month"] = te2["date"] // 100
        for mo in sorted(te2["month"].unique()):
            tem = te2[te2["month"] == mo]
            for ver in ("M0", "T2-only", "T1+T2"):
                r = run_model(tr, tem, models[ver])
                month_rows.append({"dp": dp, "month": int(mo), "model": ver,
                                   **r})
        # 晋级判定（fold1 为主，fold2 诊断）
        res = pd.DataFrame([r for r in all_rows if r["dp"] == dp])
        f1 = res[res["fold"] == "fold1"].set_index("model")
        f2 = res[res["fold"] == "fold2"].set_index("model")
        mrg = pd.DataFrame([r for r in month_rows if r["dp"] == dp])
        verdict = {}
        for ver in ("T2-only", "T1+T2"):
            v1, v2 = f1.loc[ver], f2.loc[ver]
            m0_1, m0_2 = f1.loc["M0"], f2.loc["M0"]
            mm = mrg[mrg["model"] == ver].set_index("month")
            m0m = mrg[mrg["model"] == "M0"].set_index("month")
            better_months = int((mm["top_mean"] > m0m["top_mean"]).sum())
            verdict[ver] = {
                "f1_top_gain": float(v1["top_mean"] - m0_1["top_mean"]),
                "f1_stop_delta": float(v1["top_stop"] - m0_1["top_stop"]),
                "f1_brier_delta": float(v1["brier"] - m0_1["brier"]),
                "f2_top_gain": float(v2["top_mean"] - m0_2["top_mean"]),
                "months_better": better_months,
                "n_months": int(len(mm)),
                "pass_f1": bool(v1["top_mean"] > m0_1["top_mean"]
                                and v1["top_stop"] <= m0_1["top_stop"] + 1e-9
                                and v1["brier"] <= m0_1["brier"] + 1e-9)}
        # src 拉动检查（fold1）
        sr = pd.DataFrame([r for r in src_rows if r["dp"] == dp
                           and r["fold"] == "fold1"])
        for ver in ("T2-only", "T1+T2"):
            gains = {}
            for src in ("A2", "LEGACY"):
                sub = sr[sr["src"] == src].set_index("model")
                if len(sub) and ver in sub.index and "M0" in sub.index:
                    gains[src] = float(sub.loc[ver, "top_mean"]
                                       - sub.loc["M0", "top_mean"])
            verdict[ver]["src_gains"] = gains
            verdict[ver]["pass_src"] = bool(
                gains and all(g > 0 for g in gains.values()))
        summaries[dp] = verdict
    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(OUT_DIR, "ablation_T2.csv"), index=False)
    pd.DataFrame(src_rows).to_csv(os.path.join(OUT_DIR,
                                               "ablation_T2_by_src.csv"),
                                  index=False)
    pd.DataFrame(month_rows).to_csv(os.path.join(OUT_DIR,
                                                 "ablation_T2_by_month.csv"),
                                    index=False)
    log(f"verdicts: {summaries}")

    # ---- 报告 §18.15 ----
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.15 ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n### 18.15 T2 v2 承接结构特征四组消融（2025开发，评审冻结设计）\n",
         "设计：固定事件/标签(t2.0)/group_id时序切分/LGB参数，四版本"
         "（M0、T1、T2-only、T1+T2）相同样本集合，10:00 与 10:30 **分开评估**"
         "（9:45 无T2，只报M0/T1）。T2 v2 定义=ATR冲击后的承接结构：首段"
         "9:30–9:45冲击→首低→A段9:45–10:00/B段10:00–10:30（dp1030 保留 "
         "a/b 两段，不合并45分钟）；不再要求二低触及首低，每候选都计算特征，"
         "NaN 只来自段无bar；v1 已作废（second_test_features_v1_DEPRECATED.py，"
         "三缺陷：丢抬高低点结构/阴线量方向反/段合并）。\n",
         "术语：Top30%按模型EV（事前预测）选样，**Top30%实际净收益**=选中样本"
         "标签净收益均值（事后），不再使用\"净EV\"表述。\n",
         "| dp | fold | 版本 | Top30%实际净收益 | Top30%STOP | 全体净收益 | "
         "Brier | Spearman |",
         "|---:|---|---|---:|---:|---:|---:|---:|"]
    for _, r in res.iterrows():
        L.append(f"| {r['dp']} | {r['fold']} | {r['model']} | "
                 f"{r['top_mean']:+.2%} | {r['top_stop']:.1%} | "
                 f"{r['all_mean']:+.2%} | {r['brier']:.4f} | "
                 f"{r['spearman']:+.3f} |")
    sr = pd.DataFrame(src_rows)
    L.append("")
    L.append("src分层（fold1，按来源分别训练+测试）：")
    L.append("| dp | src | 版本 | n_test | Top30%实际净收益 | Top30%STOP | "
             "Brier |")
    L.append("|---:|---|---|---:|---:|---:|---:|")
    for _, r in sr.iterrows():
        L.append(f"| {r['dp']} | {r['src']} | {r['model']} | "
                 f"{r['n_test']:,} | {r['top_mean']:+.2%} | "
                 f"{r['top_stop']:.1%} | {r['brier']:.4f} |")
    mr = pd.DataFrame(month_rows)
    L.append("")
    L.append("跨月（fold1，池化模型逐月测试，M0 vs T2-only vs T1+T2）：")
    L.append("| dp | 月份 | 版本 | n_test | Top30%实际净收益 | Top30%STOP |")
    L.append("|---:|---:|---|---:|---:|---:|")
    for _, r in mr.iterrows():
        L.append(f"| {r['dp']} | {r['month']} | {r['model']} | "
                 f"{r['n_test']:,} | {r['top_mean']:+.2%} | "
                 f"{r['top_stop']:.1%} |")
    L.append("")
    L.append("晋级判定（评审冻结条件，fold1为主/fold2诊断）：")
    for dp, vd in summaries.items():
        for ver, v in vd.items():
            L.append(f"- dp{dp} {ver}: fold1增益 {v['f1_top_gain']:+.2%}、"
                     f"ΔSTOP {v['f1_stop_delta']:+.1%}、ΔBrier "
                     f"{v['f1_brier_delta']:+.4f}、fold2增益 "
                     f"{v['f2_top_gain']:+.2%}、月份改善 "
                     f"{v['months_better']}/{v['n_months']}、"
                     f"src增益 {v['src_gains']} → "
                     f"{'通过' if v['pass_f1'] and v['pass_src'] else '未通过'}")
    L.append("\n判读边界：2025为开发数据；2026-01~08只能作已观察诊断，不能据此"
             "调T2阈值；真正前瞻影子期从代码冻结后的下一个交易日起。"
             "若T2未晋级，不再微调二踩阈值，下一刀切T3（VWAP修复持续性）与"
             "T4（板块分钟代理）。")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
