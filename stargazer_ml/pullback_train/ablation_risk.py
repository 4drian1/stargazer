# -*- coding: utf-8 -*-
"""T5 风险头消融 v3（2026-09-09 冻结方案：R0/RD0 头内重验 + T6 风险特征组，§18.28）。

主标签：r_sbr_t2（stop_before_repair_through_earliest_sell_day）——
窗口=entry→T+2收盘（T+1盘中买入、T+2最早可卖），含隔夜跳空；
可执行止损=触及且非跌停锁死bar；同bar双触悲观记STOP；T+2收盘后
信息不进标签。辅助：r_stop_by_t2 / r_stop_touched_t2 / r_stop_locked_only_t2
（T+2跌停无法成交计数）、r_same_day_extreme_adverse_move（决策日0.6%旧标签，
仅诊断）。

基座不强制含D0（评审修正：D0在五日EV下无增量，但对STOP子任务可能有用，
头内重验一次，不预设有效）：
  R0  = M0
  RD0 = M0 + D0
  冻结规则（按fold1，逐头独立）：RD0 当选 iff
    brier(RD0) <= brier(R0)+1e-4 且 高风险组真实STOP率(RD0) >= R0，
    且至少一项严格更优；否则 R0。fold2仅诊断，不参与冻结。
  冻结后（B ∈ {R, RD}）：
  {B}1 = base+T1（pf_* 路径速度）
  {B}2 = base+T2（AM: second_test a/b_*；PM: t2pm_* 事件对齐）
  {B}6 = base+T6R（T6 风险特征：中轴/下界破位、停留、假修复、通道宽度）
  {B}12 = base+T1+T2
  R126 = base+T1+T2+T6R（仅当各单组过闸后由 --add-combo 注入）

验收（§6.8 风险头闸门 + 评审修正）：
- 高风险组（P最高20%与30%均报）真实STOP率/MAE 高于基座高风险组；
- 放行组（P最低30%）STOP率低于基座放行组；
- Brier / PR-AUC 不恶化；
- 高风险组"先修复后止损"(o_rbs_t2)比例低（标签自洽）。
通用闸门同 §18.18（fold1为主/fold2诊断、src、月份≥3/4不劣）。
输出 ablation_risk_{tag}*.csv + §18.28。
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
SAMPLES = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
D0 = os.path.join(OUT_DIR, "d0_daily_features_2025.parquet")
PF = os.path.join(OUT_DIR, "minute_path_features_2025.parquet")
ST = os.path.join(OUT_DIR, "minute_second_test_2025.parquet")
TPM = os.path.join(OUT_DIR, "t2pm_features_2025.parquet")
T6 = os.path.join(OUT_DIR, "t6_features_2025.parquet")
LBL = os.path.join(OUT_DIR, "dual_head_labels_2025.parquet")
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
T1F = ["pf_ret1m", "pf_ret3m", "pf_ret5m", "pf_decline_speed",
       "pf_decel", "pf_low2_minus_low1", "pf_has_two_lows"]
COMMON_ST = ["first_leg_shock_atr", "first_low_depth_atr"]
PAIR = ("second_low_minus_first_low_atr", "rebound_before_second_test_atr",
        "second_pullback_ratio", "second_volume_vs_first_volume",
        "second_down_volume_vs_first_down_volume",
        "second_range_vs_first_range", "second_close_location",
        "second_vwap_distance_atr", "minutes_between_lows",
        "new_low_on_second_leg")
T2AM = COMMON_ST + [f"a_{k}" for k in PAIR] + [f"b_{k}" for k in PAIR]
T2PM = [f"t2pm_{k}_{w}" for w in (30, 60) for k in (
    "decline_atr", "decline_speed", "recovery_atr", "min_since_low",
    "second_low_minus_first_atr", "dnvol_shrink")] \
    + ["t2pm_vwap_reclaim_atr"]
T6R = ["t6_dist_mid_s_atr", "t6_dist_mid_l_atr",
       "t6_below_lower_depth_atr", "t6_below_now_min",
       "t6_since_first_below_min", "t6_last_cross_dir",
       "t6_low2_minus_low1_atr", "t6_width_atr", "t6_width_chg_2h_atr",
       "t6_volsur_break"]
HEADS = {"AM": (1000, 1030), "PM": (1330, 1400, 1430, 1450)}
FOLDS = [("fold1", 20250101, 20250831, 20250901, 20251231),
         ("fold2", 20250901, 20251231, 20250101, 20250831)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def brier_bin(y, p):
    return float(((p - y) ** 2).mean())


def pr_auc(y, p):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p))


def load_data(dps):
    df = pd.read_parquet(SAMPLES)
    df = df[df["dp"].isin(dps)].copy()
    for p, on in ((D0, ["code", "date"]), (PF, ["code", "dp_date", "dp"]),
                  (ST, ["code", "dp_date", "dp"]),
                  (TPM, ["code", "dp_date", "dp"]),
                  (T6, ["code", "dp_date", "dp"]),
                  (LBL, ["code", "dp_date", "dp"])):
        m = pd.read_parquet(p)
        m["code"] = m["code"].astype(str)
        df["code"] = df["code"].astype(str)
        df = df.merge(m, on=on, how="left")
    df = df[(df["entry_type"] == "vwap") & (df["censored"] == 0)
            & df[f"net_ret_t{KT}"].notna()
            & df["r_sbr_t2"].notna()].copy()
    if "price_scale_valid" not in df.columns:
        df["price_scale_valid"] = True
    df = df[df["price_scale_valid"].astype(bool)].copy()
    df["nr"] = df[f"net_ret_t{KT}"]
    df["y5"] = df[f"outcome_t{KT}"]
    df["y"] = df["r_sbr_t2"].astype(float)
    return df


def run_model(tr, te, feats):
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(objective="binary", num_leaves=31,
                             learning_rate=0.05, min_child_samples=100,
                             subsample=0.9, colsample_bytree=0.9, seed=7,
                             n_estimators=400, verbosity=-1, n_jobs=8)
    clf.fit(tr[feats], tr["y"])
    p = clf.predict_proba(te[feats])[:, 1]
    te = te.copy()
    te["P"] = p
    q70, q80 = te["P"].quantile(0.7), te["P"].quantile(0.8)
    hi = te[te["P"] >= q70]
    hi20 = te[te["P"] >= q80]
    lo = te[te["P"] <= te["P"].quantile(0.3)]
    return {"n_train": len(tr), "n_test": len(te),
            "base_rate": float(te["y"].mean()),
            "brier": brier_bin(te["y"].values, p),
            "pr_auc": pr_auc(te["y"].values, p),
            "hi_n": len(hi),
            "hi_stop": float(hi["y"].mean()),          # 真实STOP（主标签）
            "hi_stop5d": float((hi["y5"] == "STOP").mean()),
            "hi_mae": float(hi["r_mae_atr"].mean()),
            "hi_mae_q80": float(hi["r_mae_atr"].quantile(0.8)),
            "hi_repair_first": float(hi["o_rbs_t2"].mean()),
            "hi_nr": float(hi["nr"].mean()),
            "hi20_n": len(hi20),
            "hi20_stop": float(hi20["y"].mean()),
            "hi20_mae": float(hi20["r_mae_atr"].mean()),
            "lo_n": len(lo),
            "lo_stop": float(lo["y"].mean()),
            "lo_stop5d": float((lo["y5"] == "STOP").mean()),
            "lo_nr": float(lo["nr"].mean()),
            "lo_repair_first": float(lo["o_rbs_t2"].mean())}


def freeze_base(res_f1):
    """fold1 的 R0/RD0 行 -> 'R0' 或 'RD0'（评审冻结规则）。"""
    r0 = res_f1[res_f1["model"] == "R0"].iloc[0]
    rd0 = res_f1[res_f1["model"] == "RD0"].iloc[0]
    brier_ok = rd0["brier"] <= r0["brier"] + 1e-4
    stop_ok = rd0["hi_stop"] >= r0["hi_stop"]
    strict = (rd0["brier"] < r0["brier"]) or (rd0["hi_stop"] > r0["hi_stop"])
    return ("RD0" if (brier_ok and stop_ok and strict) else "R0",
            float(r0["brier"]), float(rd0["brier"]),
            float(r0["hi_stop"]), float(rd0["hi_stop"]))


def feat_sets(head, base):
    t2 = T2AM if head == "AM" else T2PM
    b = M0 + D0F if base == "RD0" else M0
    pre = "RD" if base == "RD0" else "R"
    return {f"{pre}1": b + T1F, f"{pre}2": b + t2,
            f"{pre}6": b + T6R, f"{pre}12": b + T1F + t2}


def add_combos(models, specs, base_feats):
    """--add-combo "R126:T1,T2AM,T6R" / "R126PM:T1,T2PM,T6R" 形式注入组合。"""
    alias = {"T1": T1F, "T6R": T6R, "D0": D0F,
             "T2AM": T2AM, "T2PM": T2PM}
    for spec in specs or []:
        name, parts = spec.split(":")
        feats = list(base_feats)
        for part in parts.split(","):
            feats += alias[part.strip()]
        models[name] = feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add-combo", action="append",
                    help="如 R126:T1,T2AM,T6R（alias: T1/T2AM/T2PM/T6R/D0）")
    ap.add_argument("--tag", default="v3")
    a = ap.parse_args()
    t0 = time.time()
    all_rows, src_rows, month_rows = [], [], []
    base_notes = []
    for head, dps in HEADS.items():
        df = load_data(dps)
        log(f"[{head}] 入模 {len(df):,}，主标签STOP率 {df['y'].mean():.1%}")
        base_models = {"R0": M0, "RD0": M0 + D0F}
        # ---- 阶段1：基座消融（两折） ----
        f1_rows = []
        for fname, ftr0, ftr1, t0d, t1d in FOLDS:
            tr = df[(df["date"] >= ftr0) & (df["date"] <= ftr1)]
            te = df[(df["date"] >= t0d) & (df["date"] <= t1d)]
            for ver, feats in base_models.items():
                r = run_model(tr, te, feats)
                all_rows.append({"head": head, "fold": fname, "model": ver,
                                 **r})
                if fname == "fold1":
                    f1_rows.append({"model": ver, **r})
                log(f"[{head}] {fname} {ver}: hiSTOP={r['hi_stop']:.1%} "
                    f"hiMAE={r['hi_mae']:.2f} loSTOP={r['lo_stop']:.1%} "
                    f"brier={r['brier']:.4f} prAUC={r['pr_auc']:.3f}")
        base, b_r0, b_rd0, s_r0, s_rd0 = freeze_base(pd.DataFrame(f1_rows))
        base_notes.append(
            f"[{head}] 基座冻结={base}（fold1: R0 brier={b_r0:.4f}/hiSTOP="
            f"{s_r0:.1%}，RD0 brier={b_rd0:.4f}/hiSTOP={s_rd0:.1%}；规则="
            "brier不恶化且hiSTOP不劣且至少一项严格）")
        log(base_notes[-1])
        # ---- 阶段2：冻结基座上的 T1/T2/T6R 消融 ----
        models = feat_sets(head, base)
        add_combos(models, a.add_combo,
                   M0 + D0F if base == "RD0" else M0)
        for fname, ftr0, ftr1, t0d, t1d in FOLDS:
            tr = df[(df["date"] >= ftr0) & (df["date"] <= ftr1)]
            te = df[(df["date"] >= t0d) & (df["date"] <= t1d)]
            for ver, feats in models.items():
                r = run_model(tr, te, feats)
                all_rows.append({"head": head, "fold": fname, "model": ver,
                                 **r})
                log(f"[{head}] {fname} {ver}: hiSTOP={r['hi_stop']:.1%} "
                    f"hiMAE={r['hi_mae']:.2f} loSTOP={r['lo_stop']:.1%} "
                    f"brier={r['brier']:.4f} prAUC={r['pr_auc']:.3f}")
        # ---- src / 月份 细分（fold1 切分，全部版本） ----
        ftr0, ftr1, t0d, t1d = FOLDS[0][1:]
        tr = df[(df["date"] >= ftr0) & (df["date"] <= ftr1)]
        te = df[(df["date"] >= t0d) & (df["date"] <= t1d)]
        all_models = dict(base_models, **models)
        for src in ("A2", "LEGACY"):
            trs, tes = tr[tr["src"] == src], te[te["src"] == src]
            if len(tes) < 500 or len(trs) < 500:
                continue
            for ver, feats in all_models.items():
                r = run_model(trs, tes, feats)
                src_rows.append({"head": head, "src": src, "model": ver,
                                 **r})
        te2 = te.copy()
        te2["month"] = te2["date"] // 100
        for mo in sorted(te2["month"].unique()):
            tem = te2[te2["month"] == mo]
            for ver, feats in all_models.items():
                r = run_model(tr, tem, feats)
                month_rows.append({"head": head, "month": int(mo),
                                   "model": ver, **r})
    res = pd.DataFrame(all_rows)
    tag = a.tag or "v3"
    res.to_csv(os.path.join(OUT_DIR, f"ablation_risk_{tag}.csv"),
               index=False)
    pd.DataFrame(src_rows).to_csv(
        os.path.join(OUT_DIR, f"ablation_risk_by_src_{tag}.csv"),
        index=False)
    pd.DataFrame(month_rows).to_csv(
        os.path.join(OUT_DIR, f"ablation_risk_by_month_{tag}.csv"),
        index=False)

    # ---- 月份闸门统计（fold1，相对基座 hi_stop 不劣的月数） ----
    mo = pd.DataFrame(month_rows)
    month_gate = {}
    for head in HEADS:
        base_name = [n for n in base_notes if n.startswith(f"[{head}]")][0]
        base_name = "RD0" if "基座冻结=RD0" in base_name else "R0"
        sub = mo[mo["head"] == head]
        piv = sub.pivot_table(index="month", columns="model", values="hi_stop")
        for ver in piv.columns:
            if ver == base_name:
                continue
            not_worse = (piv[ver] >= piv[base_name] - 1e-4).sum()
            month_gate[f"{head}:{ver}"] = f"{int(not_worse)}/{len(piv)}"

    # ---- 报告 §18.28 ----
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.28 ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n### 18.28 T5 风险头消融 v3（r_sbr_t2 + T6R，2025开发）\n",
         "主标签 r_sbr_t2=stop_before_repair_through_earliest_sell_day："
         "entry(T+1盘中)→T+2收盘窗口内先触可执行止损(ks×ATR，跌停锁死bar不计)"
         "再修复(+0.3ATR)；同bar双触悲观记STOP。基座头内重验："
         "R0=M0 vs RD0=M0+D0（冻结规则=fold1上RD0的brier不恶化且高风险组真实"
         "STOP率不劣、至少一项严格），逐头独立冻结；"
         "T1=pf_*路径速度、T2=AM second_test a/b_* / PM t2pm_*、"
         "T6R=T6风险特征（中轴/下界破位、下方停留、收复行为、通道宽度与扩张）。"
         f"{'；'.join(base_notes)}。\n",
         "验收（§6.8风险头闸门）：高风险组(P最高20%/30%)真实STOP率/MAE高于"
         "基座高风险组、放行组(P最低30%)STOP低于基座放行组、Brier/PR-AUC不恶化。\n",
         "| 头 | fold | 版本 | hi20%STOP | hi30%STOP | hi30%MAE | hi30%MAEq80 | "
         "hi30%净收益 | 放行STOP | 放行净收益 | Brier | PR-AUC |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in res.iterrows():
        L.append(f"| {r['head']} | {r['fold']} | {r['model']} | "
                 f"{r['hi20_stop']:.1%} | {r['hi_stop']:.1%} | "
                 f"{r['hi_mae']:.2f} | {r['hi_mae_q80']:.2f} | "
                 f"{r['hi_nr']:+.2%} | {r['lo_stop']:.1%} | "
                 f"{r['lo_nr']:+.2%} | {r['brier']:.4f} | {r['pr_auc']:.3f} |")
    L.append("")
    L.append("月份不劣（fold1，hi30%STOP ≥ 基座）：")
    for k, v in month_gate.items():
        L.append(f"- {k}: {v}")
    sr = pd.DataFrame(src_rows)
    L.append("")
    L.append("src分层（fold1）：")
    L.append("| 头 | src | 版本 | hi30%STOP | 放行STOP | Brier | PR-AUC |")
    L.append("|---|---|---|---:|---:|---:|---:|")
    for _, r in sr.iterrows():
        L.append(f"| {r['head']} | {r['src']} | {r['model']} | "
                 f"{r['hi_stop']:.1%} | {r['lo_stop']:.1%} | "
                 f"{r['brier']:.4f} | {r['pr_auc']:.3f} |")
    L.append("\n判读：风险头回答「现在买是不是仍在接飞刀」；通过闸门的版本才"
             "允许进入双头合成（§18.31）。明细："
             f"ablation_risk_{tag}*.csv")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
