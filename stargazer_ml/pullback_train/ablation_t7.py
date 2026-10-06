# -*- coding: utf-8 -*-
"""T7 高位连涨耗散风险组消融（§18.32，独立组，不改冻结 T5）。

标签（三个分别建模，不混合）：
  y0 = r_sbr_t2                 即时接飞刀（P_immediate_stop）
  y3 = exhaust_fail_3d          (o_stop_3d==1) | (o_net_ret_3d<0)（P_exhaust_fail_3d）
  y5 = exhaust_fail_5d          (o_stop_5d==1) | (o_net_ret_5d<0)（P_exhaust_fail_5d）

版本（AM/PM 各自）：R0=M0；R1/R2=冻结原风险头（M0+T1F / M0+T2PM）；
R7=M0+T7R；R17/R27=组合。只验证增量，不改 R1/R2。

闸门（对照各头冻结版本 R1/R2，fold1 为主）：
- y3/y5 模型：高风险组(Top30%)实际 exhaust 率更高；放行组(Bottom30%)净收益
  不低于原风险头；Brier/PR-AUC 不恶化(+1e-4)。
- y0 模型：高风险组 STOP 率不更低；Brier/PR-AUC 不恶化。
- 月份 ≥3/4 不劣；A2/LEGACY 无明显反转；2026 复核（训练=2025 全年）风险
  方向不得反转。003040 仅案例审计，不参与阈值选择。
输出 ablation_t7*.csv + §18.32。
"""
import os
import time

import numpy as np
import pandas as pd

from ablation_risk import M0, T1F, T2PM

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
# 2026 全量样本（含PM决策点）在 minute_labels/ 下；根目录的
# minute_samples_2026.parquet 是早期3dp旧版，勿用。
def samples_path(y):
    if y == 2026:
        return os.path.join(OUT_DIR, "minute_labels",
                            "minute_samples_2026.parquet")
    return os.path.join(OUT_DIR, f"minute_samples_{y}.parquet")
PF = os.path.join(OUT_DIR, "minute_path_features_{y}.parquet")
TPM = os.path.join(OUT_DIR, "t2pm_features_{y}.parquet")
T7 = os.path.join(OUT_DIR, "t7_features_{y}.parquet")
LBL = os.path.join(OUT_DIR, "dual_head_labels_{y}.parquet")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
KT = "2.0"
HEADS = {"AM": (1000, 1030), "PM": (1330, 1400, 1430, 1450)}
FROZEN = {"AM": "R1", "PM": "R2"}
FOLDS = [("fold1", 20250101, 20250831, 20250901, 20251231),
         ("fold2", 20250901, 20251231, 20250101, 20250831)]
BG = ("runup_5d", "runup_10d", "runup_20d", "pos20", "pos60", "rsi14",
      "rsi_over75_10d", "limit_up_10d", "limit_up_20d", "turnover_sum_5d",
      "turnover_sum_10d", "turnover_max_5d", "ht_down_10d",
      "wide_range_10d", "atr14", "atr60", "ma10_dist_atr", "ma20_dist_atr",
      "stage_low_high_20d", "days_since_high20", "churn_upper_shadow_5d")
IC = ("gap_down_atr", "below_vwap_min", "vwap_rebreak_flag",
      "vwap_rebreak_min", "reclaim_retention_945", "pm_below_vwap_frac",
      "am_best_vwap_dist_atr", "recovery_decay_pm_atr", "second_low_lower")
IX = ("t7_ix_runup10_gap", "t7_ix_turnover5_shock", "t7_ix_pos_rebreak",
      "t7_ix_runup_retention")
T7R = list(BG) + list(IC) + list(IX)
LABELS = {"y0": "r_sbr_t2", "y3": "exhaust_fail_3d", "y5": "exhaust_fail_5d"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def brier_bin(y, p):
    return float(((p - y) ** 2).mean())


def pr_auc(y, p):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p))


def load_pool(year, dps):
    df = pd.read_parquet(samples_path(year))
    df = df[df["dp"].isin(dps)].copy()
    for p, on in ((PF.format(y=year), ["code", "dp_date", "dp"]),
                  (TPM.format(y=year), ["code", "dp_date", "dp"]),
                  (LBL.format(y=year), ["code", "dp_date", "dp"])):
        m = pd.read_parquet(p)
        m["code"] = m["code"].astype(str)
        df["code"] = df["code"].astype(str)
        df = df.merge(m, on=on, how="left")
    m = pd.read_parquet(T7.format(y=year))
    m["code"] = m["code"].astype(str)
    df["code"] = df["code"].astype(str)
    m = m.drop(columns=["date"])      # 信号日列与 samples.date 冲突
    df = df.merge(m, on=["code", "dp_date", "dp"], how="left")
    df = df[(df["entry_type"] == "vwap") & (df["censored"] == 0)
            & df[f"net_ret_t{KT}"].notna()
            & df["r_sbr_t2"].notna() & df["o_stop_3d"].notna()
            & df["o_net_ret_5d"].notna()].copy()
    if "price_scale_valid" not in df.columns:
        df["price_scale_valid"] = True
    df = df[df["price_scale_valid"].astype(bool)].copy()
    # 交互项（原始尺度相乘；NaN 传播保留"未发生"语义）
    df["t7_ix_runup10_gap"] = df["runup_10d"] * df["gap_down_atr"]
    df["t7_ix_turnover5_shock"] = df["turnover_sum_5d"] * df["shock_atr"]
    df["t7_ix_pos_rebreak"] = df["pos20"] * df["vwap_rebreak_flag"]
    df["t7_ix_runup_retention"] = df["runup_10d"] * df["reclaim_retention_945"]
    df["exhaust_fail_3d"] = ((df["o_stop_3d"] == 1)
                             | (df["o_net_ret_3d"] < 0)).astype(float)
    df["exhaust_fail_5d"] = ((df["o_stop_5d"] == 1)
                             | (df["o_net_ret_5d"] < 0)).astype(float)
    df["nr"] = df[f"net_ret_t{KT}"]
    return df


def run_model(tr, te, feats, ycol):
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(objective="binary", num_leaves=31,
                             learning_rate=0.05, min_child_samples=100,
                             subsample=0.9, colsample_bytree=0.9, seed=7,
                             n_estimators=400, verbosity=-1, n_jobs=8)
    clf.fit(tr[feats], tr[ycol])
    p = clf.predict_proba(te[feats])[:, 1]
    te = te.copy()
    te["P"] = p
    hi = te[te["P"] >= te["P"].quantile(0.7)]
    lo = te[te["P"] <= te["P"].quantile(0.3)]

    def agg(d, prefix):
        return {
            f"{prefix}_n": len(d),
            f"{prefix}_rate": float(d[ycol].mean()),
            f"{prefix}_stop": float(d["r_sbr_t2"].mean()),
            f"{prefix}_ex3": float(d["exhaust_fail_3d"].mean()),
            f"{prefix}_ex5": float(d["exhaust_fail_5d"].mean()),
            f"{prefix}_nr3": float(d["o_net_ret_3d"].mean()),
            f"{prefix}_nr5": float(d["o_net_ret_5d"].mean())}
    return {"n_train": len(tr), "n_test": len(te),
            "base_rate": float(te[ycol].mean()),
            "brier": brier_bin(te[ycol].values, p),
            "pr_auc": pr_auc(te[ycol].values, p),
            **agg(hi, "hi"), **agg(lo, "lo")}


def feat_sets(head):
    t1 = T1F if head == "AM" else T2PM
    pre = "R1" if head == "AM" else "R2"
    return {"R0": M0, pre: M0 + t1, "R7": M0 + T7R, pre + "7": M0 + t1 + T7R}


def main():
    t0 = time.time()
    all_rows, month_rows, src_rows, chk_rows = [], [], [], []
    for head, dps in HEADS.items():
        df = load_pool(2025, dps)
        log(f"[{head}] 入模 {len(df):,}；y3={df['exhaust_fail_3d'].mean():.1%} "
            f"y5={df['exhaust_fail_5d'].mean():.1%} y0={df['r_sbr_t2'].mean():.1%}")
        models = feat_sets(head)
        for lk, ycol in LABELS.items():
            for fname, ftr0, ftr1, t0d, t1d in FOLDS:
                tr = df[(df["date"] >= ftr0) & (df["date"] <= ftr1)]
                te = df[(df["date"] >= t0d) & (df["date"] <= t1d)]
                for ver, feats in models.items():
                    r = run_model(tr, te, feats, ycol)
                    all_rows.append({"head": head, "label": lk, "fold": fname,
                                     "model": ver, **r})
                    log(f"[{head}] {lk} {fname} {ver}: hi_rate={r['hi_rate']:.1%} "
                        f"hi_ex3={r['hi_ex3']:.1%} hi_ex5={r['hi_ex5']:.1%} "
                        f"lo_nr5={r['lo_nr5']:+.3%} brier={r['brier']:.4f} "
                        f"prAUC={r['pr_auc']:.3f}")
        # 月份 / src（fold1，y3 主标签）
        ftr0, ftr1, t0d, t1d = FOLDS[0][1:]
        tr = df[(df["date"] >= ftr0) & (df["date"] <= ftr1)]
        te = df[(df["date"] >= t0d) & (df["date"] <= t1d)]
        ycol = "exhaust_fail_3d"
        te2 = te.copy()
        te2["month"] = te2["date"] // 100
        for mo in sorted(te2["month"].unique()):
            tem = te2[te2["month"] == mo]
            for ver, feats in models.items():
                r = run_model(tr, tem, feats, ycol)
                month_rows.append({"head": head, "month": int(mo),
                                   "model": ver, **r})
        for src in ("A2", "LEGACY"):
            trs, tes = tr[tr["src"] == src], te[te["src"] == src]
            if len(tes) < 500 or len(trs) < 500:
                continue
            for ver, feats in models.items():
                r = run_model(trs, tes, feats, ycol)
                src_rows.append({"head": head, "src": src, "model": ver, **r})
        # ---- 2026 复核（训练=2025 全年，只能否定） ----
        d26 = load_pool(2026, dps)
        d26 = d26[d26["date"] <= 20260831]
        for lk, ycol in LABELS.items():
            for ver, feats in models.items():
                r = run_model(df, d26, feats, ycol)
                chk_rows.append({"head": head, "label": lk, "model": ver, **r})
                log(f"[{head}] 2026 {lk} {ver}: hi_rate={r['hi_rate']:.1%} "
                    f"hi_ex3={r['hi_ex3']:.1%} hi_ex5={r['hi_ex5']:.1%} "
                    f"brier={r['brier']:.4f}")

    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(OUT_DIR, "ablation_t7.csv"), index=False)
    pd.DataFrame(month_rows).to_csv(
        os.path.join(OUT_DIR, "ablation_t7_by_month.csv"), index=False)
    pd.DataFrame(src_rows).to_csv(
        os.path.join(OUT_DIR, "ablation_t7_by_src.csv"), index=False)
    pd.DataFrame(chk_rows).to_csv(
        os.path.join(OUT_DIR, "ablation_t7_2026.csv"), index=False)

    # ---- 闸门判定（fold1，对照冻结版本 FROZEN[head]） ----
    gates = []
    for head in HEADS:
        ref = FROZEN[head]
        for lk in LABELS:
            sub = res[(res["head"] == head) & (res["label"] == lk)
                      & (res["fold"] == "fold1")]
            b = sub[sub["model"] == ref].iloc[0]
            for ver in feat_sets(head):
                if ver == ref:
                    continue
                r = sub[sub["model"] == ver].iloc[0]
                if lk == "y0":
                    ok = (r["hi_stop"] >= b["hi_stop"] - 1e-4
                          and r["brier"] <= b["brier"] + 1e-4
                          and r["pr_auc"] >= b["pr_auc"] - 1e-4)
                else:
                    ok = (r["hi_rate"] >= b["hi_rate"] + 1e-4
                          and r[f"lo_nr{lk[-1]}"]
                              >= b[f"lo_nr{lk[-1]}"] - 1e-4
                          and r["brier"] <= b["brier"] + 1e-4
                          and r["pr_auc"] >= b["pr_auc"] - 1e-4)
                gates.append({"head": head, "label": lk, "model": ver,
                              "gate": "PASS" if ok else "FAIL",
                              "hi_delta": float(r["hi_rate"] - b["hi_rate"]),
                              "brier_delta": float(r["brier"] - b["brier"])})
    gdf = pd.DataFrame(gates)
    gdf.to_csv(os.path.join(OUT_DIR, "ablation_t7_gates.csv"), index=False)

    # ---- 003040 案例审计（E0，不参与阈值选择） ----
    audit = []
    for year in (2025, 2026):
        try:
            d = load_pool(year, (1000, 1030, 1400))
        except Exception:
            continue
        d = d[d["code"] == "003040"]
        for _, r in d.iterrows():
            audit.append({"year": year, "sig_date": int(r["date"]),
                          "dp": int(r["dp"]),
                          "runup_10d": round(float(r["runup_10d"]), 3),
                          "pos20": round(float(r["pos20"]), 3),
                          "rsi14": round(float(r["rsi14"]), 1),
                          "turnover_sum_5d": round(float(r["turnover_sum_5d"]), 1),
                          "limit_up_10d": r["limit_up_10d"],
                          "gap_down_atr": round(float(r["gap_down_atr"]), 3),
                          "vwap_dist_atr_dp": round(float(r["vwap_dist_atr_dp"]), 3),
                          "vwap_rebreak_flag": int(r["vwap_rebreak_flag"]),
                          "recovery_decay_pm_atr": r.get("recovery_decay_pm_atr"),
                          "r_sbr_t2": int(r["r_sbr_t2"]),
                          "exhaust_fail_5d": int(r["exhaust_fail_5d"]),
                          "o_net_ret_5d": round(float(r["o_net_ret_5d"]), 4)})
    adf = pd.DataFrame(audit)
    adf.to_csv(os.path.join(OUT_DIR, "t7_audit_003040.csv"), index=False)

    # ---- 报告 §18.32 ----
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.32 ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n### 18.32 T7 高位连涨耗散风险组消融（独立组，不改冻结 T5）\n",
         "三标签分别建模：y0=r_sbr_t2（即时接飞刀）、y3=exhaust_fail_3d"
         "（3日先止损或扣费净收益<0）、y5=exhaust_fail_5d。版本：R0=M0；"
         "R1/R2=已冻结原风险头（M0+T1F / M0+T2PM）；R7=M0+T7R（21 背景+9 盘中"
         "+4 交互）；R17/R27=组合。闸门=高风险组 exhaust 率更高、放行组净收益"
         "不低于原头、Brier/PR-AUC 不恶化、月份≥3/4、src/2026 无反转。"
         "003040 仅案例审计，不参与阈值选择。\n",
         "**闸门判定（fold1，对照各头冻结版本）**：\n",
         "| 头 | 标签 | 版本 | 闸门 | hi_rate Δ | brier Δ |",
         "|---|---|---|---|---:|---:|"]
    for _, r in gdf.iterrows():
        L.append(f"| {r['head']} | {r['label']} | {r['model']} | {r['gate']} | "
                 f"{r['hi_delta']:+.2%} | {r['brier_delta']:+.4f} |")
    L.append("\n**明细（fold1/fold2，全版本）**：\n")
    L.append("| 头 | 标签 | fold | 版本 | hi_rate | hi_STOP | hi_ex3 | hi_ex5 | "
             "hi净收益5d | 放行净收益5d | Brier | PR-AUC |")
    L.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in res.iterrows():
        L.append(f"| {r['head']} | {r['label']} | {r['fold']} | {r['model']} | "
                 f"{r['hi_rate']:.1%} | {r['hi_stop']:.1%} | {r['hi_ex3']:.1%} | "
                 f"{r['hi_ex5']:.1%} | {r['hi_nr5']:+.3%} | "
                 f"{r['lo_nr5']:+.3%} | {r['brier']:.4f} | {r['pr_auc']:.3f} |")
    chk = pd.DataFrame(chk_rows)
    L.append("\n**2026 复核**（训练=2025 全年，只能否定；hi_rate 为各标签率）：\n")
    L.append("| 头 | 标签 | 版本 | hi_rate | hi_ex3 | hi_ex5 | Brier | PR-AUC |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|")
    for _, r in chk.iterrows():
        L.append(f"| {r['head']} | {r['label']} | {r['model']} | "
                 f"{r['hi_rate']:.1%} | {r['hi_ex3']:.1%} | {r['hi_ex5']:.1%} | "
                 f"{r['brier']:.4f} | {r['pr_auc']:.3f} |")
    if len(adf):
        L.append("\n**003040 案例审计**（E0）：\n")
        L.append("| 年 | 信号日 | dp | runup10d | pos20 | rsi14 | 换手5d | "
                 "涨停10d | gap | vwap距离 | rebreak | 衰减 | y0 | fail5 | nr5d |")
        L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in adf.iterrows():
            dec = r["recovery_decay_pm_atr"]
            dec = f"{dec:+.2f}" if pd.notna(dec) else "-"
            L.append(f"| {r['year']} | {r['sig_date']} | {r['dp']} | "
                     f"{r['runup_10d']:.1%} | {r['pos20']:.2f} | {r['rsi14']:.0f} | "
                     f"{r['turnover_sum_5d']:.0f}% | {r['limit_up_10d']:.0f} | "
                     f"{r['gap_down_atr']:+.2f} | {r['vwap_dist_atr_dp']:+.2f} | "
                     f"{r['vwap_rebreak_flag']} | {dec} | "
                     f"{r['r_sbr_t2']} | {r['exhaust_fail_5d']} | "
                     f"{r['o_net_ret_5d']:+.2%} |")
    else:
        L.append("\n**003040 案例审计**：池内无 003040 样本（2026-09 案例在分钟"
                 "缓存覆盖/隔离范围之外），仅保留审计脚本入口。")
    L.append(f"\n明细：ablation_t7*.csv / t7_audit_003040.csv。用时 "
             f"{time.time()-t0:.0f}s")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
