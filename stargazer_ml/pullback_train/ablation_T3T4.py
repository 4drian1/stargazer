# -*- coding: utf-8 -*-
"""T3/T4 四组固定消融（2025开发数据，评审冻结设计）。

消融结构（固定事件、固定决策点、固定标签t2.0、固定group_id时序切分、
固定LGB参数，只变特征）：
  M0      = 基础分钟模型v1特征（12个）
  T3-only = M0 + VWAP修复持续性特征（10个 v3_*）
  T4-only = M0 + 板块分钟代理软特征（t4_proxy_available + 9个 t4_*）
  T3+T4   = M0 + v3_* + t4_*

时间头分训分验（评审冻结，避免PM再次被遗漏）：
  AM头: dp ∈ {1000, 1030}（9:45 只报 M0/T1，不进本消融）
  PM头: dp ∈ {1330, 1400, 1430, 1450}
两个头各自训练、各自验收，不混合。

时间折（fold1正向为主要依据，fold2反向仅诊断）：
  fold1: train=20250101-20250831, test=20250901-20251231
  fold2: train=20250901-20251231, test=20250101-20250831
同一事件(group_id)的各决策点行同折（date在组内恒定，按date切分天然成立）。

晋级条件（评审冻结，新增必要条件）：Top30%实际净收益 **必须为正**
（不允许在负收益池里"少亏一点"）且 >M0 且 STOP不恶化 且 Brier不恶化
且 月度改善非单月贡献 且 反向折不明显反转 且 A2与LEGACY非单一来源拉动。
术语：Top30%按模型EV（事前预测）选样，"Top30%实际净收益"=选中样本的
标签净收益均值（事后结果）。

输出：ablation_T3T4.csv / ablation_T3T4_by_src.csv / ablation_T3T4_by_month.csv；
报告写入 §18.18。2025仅开发证据；T3/T4 冻结后下一个交易日起才算前瞻影子期。

PM口径说明（冻结）：PM头样本来自与AM头同一的门控前候选池事件
（历史训练总体定义不变）；非候选股票的盘中独立下杀由实时盯盘层生成，
不在本历史训练总体内。
"""
import os
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
SAMPLES = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
T3 = os.path.join(OUT_DIR, "minute_t3_vwap_2025.parquet")
T4 = os.path.join(OUT_DIR, "minute_t4_proxy_2025.parquet")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
KT = "2.0"
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


def load_data(dps):
    df = pd.read_parquet(SAMPLES)
    t3 = pd.read_parquet(T3)
    t4 = pd.read_parquet(T4)
    df = df[df["dp"].isin(dps)].copy()
    df = df.merge(t3, on=["code", "dp_date", "dp"], how="left")
    df = df.merge(t4, on=["code", "dp_date", "dp"], how="left")
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


def main():
    t0 = time.time()
    all_rows, src_rows, month_rows = [], [], []
    summaries = {}
    for head, dps in HEADS.items():
        df = load_data(dps)
        models = {"M0": M0, "T3-only": M0 + V3, "T4-only": M0 + T4F,
                  "T3+T4": M0 + V3 + T4F}
        nan_chk = {c: round(float(df[c].isna().mean()), 4)
                   for c in models["T3+T4"] if c in df.columns}
        log(f"[{head}] dp{dps} 入模 {len(df):,} "
            f"（{df['src'].value_counts().to_dict()}）；特征NaN率: {nan_chk}")
        for fname, ftr0, ftr1, t0d, t1d in FOLDS:
            tr = df[(df["date"] >= ftr0) & (df["date"] <= ftr1)]
            te = df[(df["date"] >= t0d) & (df["date"] <= t1d)]
            for ver, feats in models.items():
                r = run_model(tr, te, feats)
                all_rows.append({"head": head, "fold": fname, "model": ver,
                                 **r})
                log(f"[{head}] {fname} {ver}: top={r['top_mean']:+.3%} "
                    f"stop={r['top_stop']:.1%} brier={r['brier']:.4f}")
        # src 分层（fold1，按来源分别训练+测试）
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
        # 跨月（fold1，池化模型逐月测试）
        te2 = te.copy()
        te2["month"] = te2["date"] // 100
        for mo in sorted(te2["month"].unique()):
            tem = te2[te2["month"] == mo]
            for ver in models:
                r = run_model(tr, tem, models[ver])
                month_rows.append({"head": head, "month": int(mo),
                                   "model": ver, **r})
        # 晋级判定（fold1为主，新增 top_mean>0 必要条件）
        res = pd.DataFrame([r for r in all_rows if r["head"] == head])
        f1 = res[res["fold"] == "fold1"].set_index("model")
        f2 = res[res["fold"] == "fold2"].set_index("model")
        mrg = pd.DataFrame([r for r in month_rows if r["head"] == head])
        verdict = {}
        for ver in ("T3-only", "T4-only", "T3+T4"):
            v1, v2 = f1.loc[ver], f2.loc[ver]
            m0_1, m0_2 = f1.loc["M0"], f2.loc["M0"]
            mm = mrg[mrg["model"] == ver].set_index("month")
            m0m = mrg[mrg["model"] == "M0"].set_index("month")
            better_months = int((mm["top_mean"] > m0m["top_mean"]).sum())
            verdict[ver] = {
                "f1_top": float(v1["top_mean"]),
                "f1_top_gain": float(v1["top_mean"] - m0_1["top_mean"]),
                "f1_stop_delta": float(v1["top_stop"] - m0_1["top_stop"]),
                "f1_brier_delta": float(v1["brier"] - m0_1["brier"]),
                "f2_top_gain": float(v2["top_mean"] - m0_2["top_mean"]),
                "months_better": better_months,
                "n_months": int(len(mm)),
                "pass_f1": bool(v1["top_mean"] > 0
                                and v1["top_mean"] > m0_1["top_mean"]
                                and v1["top_stop"] <= m0_1["top_stop"] + 1e-9
                                and v1["brier"] <= m0_1["brier"] + 1e-9)}
        sr = pd.DataFrame([r for r in src_rows if r["head"] == head
                           and r["fold"] == "fold1"])
        for ver in ("T3-only", "T4-only", "T3+T4"):
            gains = {}
            for src in ("A2", "LEGACY"):
                sub = sr[sr["src"] == src].set_index("model")
                if len(sub) and ver in sub.index and "M0" in sub.index:
                    gains[src] = float(sub.loc[ver, "top_mean"]
                                       - sub.loc["M0", "top_mean"])
            verdict[ver]["src_gains"] = gains
            verdict[ver]["pass_src"] = bool(
                gains and all(g > 0 for g in gains.values()))
        summaries[head] = verdict
    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(OUT_DIR, "ablation_T3T4.csv"), index=False)
    pd.DataFrame(src_rows).to_csv(os.path.join(OUT_DIR,
                                               "ablation_T3T4_by_src.csv"),
                                  index=False)
    pd.DataFrame(month_rows).to_csv(os.path.join(OUT_DIR,
                                                 "ablation_T3T4_by_month.csv"),
                                    index=False)
    log(f"verdicts: {summaries}")

    # ---- 报告 §18.18 ----
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.18 ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n### 18.18 T3/T4 VWAP修复与板块分钟代理四组消融（2025开发，"
         "评审冻结设计）\n",
         "设计：固定事件/标签(t2.0)/group_id时序切分/LGB参数，四版本"
         "（M0、T3-only、T4-only、T3+T4）相同样本集合；**AM/PM 分头训练、"
         "分头验收**（AM=10:00/10:30，PM=13:30/14:00/14:30/14:50；9:45 只报"
         "M0/T1 不进本消融）。\n",
         "T3=VWAP修复持续性（首站回时间/保持/穿越/再跌破深度等10个 v3_*，"
         "session VWAP 自09:30累计，全日形态只作标签）；T4=板块分钟代理软特征"
         "（T-1月度冻结篮子 proxy_baskets.json，t4_proxy_available 为 0 时其余"
         "特征NaN，无覆盖≠板块弱；只作软特征不设硬门）。PM口径：PM头样本与AM头"
         "来自同一门控前候选池，非候选股票的盘中独立下杀由实时盯盘层生成。\n",
         "晋级条件（冻结，含新增必要条件）：Top30%实际净收益**为正**且>M0、"
         "STOP不恶化、Brier不恶化、月度非单月贡献、反向折不明显反转、"
         "A2与LEGACY非单一来源拉动。\n",
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
    L.append("src分层（fold1，按来源分别训练+测试）：")
    L.append("| 头 | src | 版本 | n_test | Top30%实际净收益 | Top30%STOP | "
             "Brier |")
    L.append("|---|---|---|---:|---:|---:|---:|")
    for _, r in sr.iterrows():
        L.append(f"| {r['head']} | {r['src']} | {r['model']} | "
                 f"{r['n_test']:,} | {r['top_mean']:+.2%} | "
                 f"{r['top_stop']:.1%} | {r['brier']:.4f} |")
    mr = pd.DataFrame(month_rows)
    L.append("")
    L.append("跨月（fold1，池化模型逐月测试）：")
    L.append("| 头 | 月份 | 版本 | n_test | Top30%实际净收益 | Top30%STOP |")
    L.append("|---|---:|---|---:|---:|---:|")
    for _, r in mr.iterrows():
        L.append(f"| {r['head']} | {r['month']} | {r['model']} | "
                 f"{r['n_test']:,} | {r['top_mean']:+.2%} | "
                 f"{r['top_stop']:.1%} |")
    L.append("")
    L.append("晋级判定（fold1为主/fold2诊断；pass含 top_mean>0 必要条件）：")
    for head, vd in summaries.items():
        for ver, v in vd.items():
            L.append(f"- {head} {ver}: fold1 Top30% {v['f1_top']:+.2%}"
                     f"（增益 {v['f1_top_gain']:+.2%}）、ΔSTOP "
                     f"{v['f1_stop_delta']:+.1%}、ΔBrier "
                     f"{v['f1_brier_delta']:+.4f}、fold2增益 "
                     f"{v['f2_top_gain']:+.2%}、月份改善 "
                     f"{v['months_better']}/{v['n_months']}、"
                     f"src增益 {v['src_gains']} → "
                     f"{'通过' if v['pass_f1'] and v['pass_src'] else '未通过'}")
    L.append("\n判读边界：2025为开发数据；2026-01~08及9/4、9/7、9/8三段模板已"
             "参与假设形成，只能作否定与诊断；T3/T4冻结后下一个交易日起才算"
             "真正前瞻影子期。若 T3/T4 单独未产生稳定增量，不默认把已失败的"
             "T1/T2 塞回主模型。")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
