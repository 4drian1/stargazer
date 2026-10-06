# -*- coding: utf-8 -*-
"""T5 机会头消融 v3（2026-09-09 冻结：3d/5d 经济账本口径，§18.29）。

主标签：o_t1_bt_5d（T+2~T+6 内 kt=1.0 先止盈先于止损）。机会头回答
"买入后3~5日是否还有经济价值"，验收直接看实际净收益（3d/5d双口径），
"当天收回"类标签已全部降级为辅助/特征（o_rbs_t2 不再作标签）。

基座不强制含D0（头内重验一次）：
  O0  = M0
  OD0 = M0 + D0
  冻结规则（fold1，逐头独立）：OD0 当选 iff brier(OD0)<=brier(O0)+1e-4 且
    高机会组5d净收益不劣 且 t1_bt_5d 率不劣，且至少一项严格；否则 O0。
  冻结后（B ∈ {O, OD}）：
  {B}3 = base+T3(VWAP修复)  {B}4 = base+T4-excess(§18.22 获胜子组)
  {B}6 = base+T6机会特征(中轴收回/通道/距上界剩余空间)
  {B}34 / {B}346 = 组合（346 仅当各单组均过闸后由 --add-combo 注入）

验收（新晋级标准）：Top30% 5d扣费净收益>0 且>基座；3d净收益不恶化；
STOP(5d)不恶化；Brier/PR-AUC 不恶化；≥3/4月份不劣；A2与LEGACY方向一致。
全局诊断（写入报告）：盘中修复成功(o_rbs_t2=1)样本的3d/5d净收益 vs 全体
（"修复成功后3/5日收益确实更好"闸门的直接检验）。
输出 ablation_opp*.csv + §18.29。
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
SAMPLES = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
D0 = os.path.join(OUT_DIR, "d0_daily_features_2025.parquet")
T3 = os.path.join(OUT_DIR, "minute_t3_vwap_2025.parquet")
T4 = os.path.join(OUT_DIR, "minute_t4_proxy_2025.parquet")
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
V3 = ["v3_first_reclaim_min", "v3_hold_after_reclaim_min", "v3_cross_count",
      "v3_longest_above_min", "v3_rebreak_flag", "v3_rebreak_depth_atr",
      "v3_above_min_frac", "v3_dp_vwap_dist_atr", "v3_above_vol_frac",
      "v3_rel_mkt_after_reclaim"]
T4_V2 = ["t4_proxy_available", "t4_stock_excess_dp",
         "t4_stock_excess_leader"]
T6O = ["t6_dist_mid_s_atr", "t6_dist_vwap_atr", "t6_space_to_upper_atr",
       "t6_above_upper_depth_atr", "t6_since_reclaim_min",
       "t6_hold_after_reclaim_min", "t6_upper_touch", "t6_upper_mode",
       "t6_width_atr", "t6_width_chg_2h_atr", "t6_volsur_reclaim"]
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
    for p, on in ((D0, ["code", "date"]), (T3, ["code", "dp_date", "dp"]),
                  (T4, ["code", "dp_date", "dp"]),
                  (T6, ["code", "dp_date", "dp"]),
                  (LBL, ["code", "dp_date", "dp"])):
        m = pd.read_parquet(p)
        m["code"] = m["code"].astype(str)
        df["code"] = df["code"].astype(str)
        df = df.merge(m, on=on, how="left")
    df = df[(df["entry_type"] == "vwap") & (df["censored"] == 0)
            & df["o_t1_bt_5d"].notna()].copy()
    if "price_scale_valid" not in df.columns:
        df["price_scale_valid"] = True
    df = df[df["price_scale_valid"].astype(bool)].copy()
    df["y"] = df["o_t1_bt_5d"].astype(float)
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
    hi = te[te["P"] >= te["P"].quantile(0.7)]
    lo = te[te["P"] <= te["P"].quantile(0.3)]
    return {"n_train": len(tr), "n_test": len(te),
            "base_rate": float(te["y"].mean()),
            "brier": brier_bin(te["y"].values, p),
            "pr_auc": pr_auc(te["y"].values, p),
            "hi_n": len(hi),
            "hi_t1": float(hi["y"].mean()),
            "hi_nr5": float(hi["o_net_ret_5d"].mean()),
            "hi_nr3": float(hi["o_net_ret_3d"].mean()),
            "hi_stop5d": float(hi["o_stop_5d"].mean()),
            "hi_rbl5": float(hi["o_repair_but_loss_5d"].mean()),
            "lo_n": len(lo),
            "lo_t1": float(lo["y"].mean()),
            "lo_nr5": float(lo["o_net_ret_5d"].mean())}


def freeze_base(res_f1):
    o0 = res_f1[res_f1["model"] == "O0"].iloc[0]
    od0 = res_f1[res_f1["model"] == "OD0"].iloc[0]
    brier_ok = od0["brier"] <= o0["brier"] + 1e-4
    nr_ok = od0["hi_nr5"] >= o0["hi_nr5"]
    t1_ok = od0["hi_t1"] >= o0["hi_t1"]
    strict = (od0["brier"] < o0["brier"]) or (od0["hi_nr5"] > o0["hi_nr5"]) \
        or (od0["hi_t1"] > o0["hi_t1"])
    return ("OD0" if (brier_ok and nr_ok and t1_ok and strict) else "O0",
            float(o0["brier"]), float(od0["brier"]),
            float(o0["hi_nr5"]), float(od0["hi_nr5"]),
            float(o0["hi_t1"]), float(od0["hi_t1"]))


def feat_sets(base):
    b = M0 + D0F if base == "OD0" else M0
    pre = "OD" if base == "OD0" else "O"
    return {f"{pre}3": b + V3, f"{pre}4": b + T4_V2, f"{pre}6": b + T6O,
            f"{pre}34": b + V3 + T4_V2}


def add_combos(models, specs, head, base_feats):
    """--add-combo "O346:T3,T4,T6O" / "R126:T1,T2AM,T6R" 形式注入组合。"""
    alias = {"T3": V3, "T4": T4_V2, "T6O": T6O, "T1": None, "T2AM": None,
             "T2PM": None, "T6R": None, "D0": D0F}
    for spec in specs or []:
        name, parts = spec.split(":")
        feats = list(base_feats)
        for part in parts.split(","):
            part = part.strip()
            if part == "T1":
                from ablation_risk import T1F
                feats += T1F
            elif part == "T2AM":
                from ablation_risk import T2AM
                feats += T2AM
            elif part == "T2PM":
                from ablation_risk import T2PM
                feats += T2PM
            elif part == "T6R":
                from ablation_risk import T6R
                feats += T6R
            else:
                feats += alias[part]
        models[name] = feats


def gate_rows(res, base_name):
    """生成每个版本相对基座的闸门判定（fold1）。"""
    b = res[(res["fold"] == "fold1") & (res["model"] == base_name)]
    if not len(b):
        return {}
    b = b.iloc[0]
    out = {}
    for _, r in res[res["fold"] == "fold1"].iterrows():
        out[r["model"]] = {
            "nr5_pos": r["hi_nr5"] > 0,
            "nr5_better": r["hi_nr5"] > b["hi_nr5"],
            "nr3_not_worse": r["hi_nr3"] >= b["hi_nr3"] - 1e-4,
            "stop_ok": r["hi_stop5d"] <= b["hi_stop5d"] + 1e-4,
            "brier_ok": r["brier"] <= b["brier"] + 1e-4,
            "pr_ok": r["pr_auc"] >= b["pr_auc"] - 1e-4}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add-combo", action="append",
                    help="如 O346:T3,T4,T6O（alias: T3/T4/T6O/D0）")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.time()
    all_rows, src_rows, month_rows = [], [], []
    base_notes, gate_notes = [], []
    for head, dps in HEADS.items():
        df = load_data(dps)
        log(f"[{head}] 入模 {len(df):,}，主标签 t1_bt_5d 率 {df['y'].mean():.1%}，"
            f"5d净收益 池均 {df['o_net_ret_5d'].mean():+.3%} / "
            f"3d {df['o_net_ret_3d'].mean():+.3%}")
        # 全局诊断：修复成功后 3/5 日收益
        rep = df[df["o_rbs_t2"] == 1]
        note = (f"[{head}] 修复成功样本 {len(rep):,}：3d "
                f"{rep['o_net_ret_3d'].mean():+.3%} vs 全体 "
                f"{df['o_net_ret_3d'].mean():+.3%}；5d "
                f"{rep['o_net_ret_5d'].mean():+.3%} vs 全体 "
                f"{df['o_net_ret_5d'].mean():+.3%}；repair_but_loss_5d "
                f"{df['o_repair_but_loss_5d'].mean():.1%}")
        gate_notes.append(note)
        log(note)
        base_models = {"O0": M0, "OD0": M0 + D0F}
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
                log(f"[{head}] {fname} {ver}: hiNR5={r['hi_nr5']:+.3%} "
                    f"hiNR3={r['hi_nr3']:+.3%} hiT1={r['hi_t1']:.1%} "
                    f"hiSTOP5d={r['hi_stop5d']:.1%} brier={r['brier']:.4f}")
        base, b_o0, b_od0, n_o0, n_od0, t_o0, t_od0 = freeze_base(
            pd.DataFrame(f1_rows))
        base_notes.append(
            f"[{head}] 基座={base}（fold1: O0 brier={b_o0:.4f}/hiNR5="
            f"{n_o0:+.3%}/hiT1={t_o0:.1%}，OD0 brier={b_od0:.4f}/hiNR5="
            f"{n_od0:+.3%}/hiT1={t_od0:.1%}）")
        log(base_notes[-1])
        models = feat_sets(base)
        add_combos(models, a.add_combo, head,
                   M0 + D0F if base == "OD0" else M0)
        for fname, ftr0, ftr1, t0d, t1d in FOLDS:
            tr = df[(df["date"] >= ftr0) & (df["date"] <= ftr1)]
            te = df[(df["date"] >= t0d) & (df["date"] <= t1d)]
            for ver, feats in models.items():
                r = run_model(tr, te, feats)
                all_rows.append({"head": head, "fold": fname, "model": ver,
                                 **r})
                log(f"[{head}] {fname} {ver}: hiNR5={r['hi_nr5']:+.3%} "
                    f"hiNR3={r['hi_nr3']:+.3%} hiT1={r['hi_t1']:.1%} "
                    f"hiSTOP5d={r['hi_stop5d']:.1%} brier={r['brier']:.4f}")
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
    res.to_csv(os.path.join(OUT_DIR, f"ablation_opp_{tag}.csv"), index=False)
    pd.DataFrame(src_rows).to_csv(
        os.path.join(OUT_DIR, f"ablation_opp_by_src_{tag}.csv"), index=False)
    pd.DataFrame(month_rows).to_csv(
        os.path.join(OUT_DIR, f"ablation_opp_by_month_{tag}.csv"),
        index=False)

    # ---- 月份闸门统计（fold1，相对基座 hi_nr5 不劣的月数） ----
    mo = pd.DataFrame(month_rows)
    month_gate = {}
    for (head, base_name) in (("AM", None), ("PM", None)):
        pass
    for head in HEADS:
        base_name = [n for n in base_notes if n.startswith(f"[{head}]")][0]
        base_name = "OD0" if "基座=OD0" in base_name else "O0"
        sub = mo[mo["head"] == head]
        piv = sub.pivot_table(index="month", columns="model", values="hi_nr5")
        for ver in piv.columns:
            if ver == base_name or ver not in piv:
                continue
            not_worse = (piv[ver] >= piv[base_name] - 1e-4).sum()
            month_gate[f"{head}:{ver}"] = f"{int(not_worse)}/{len(piv)}"

    # ---- 报告 §18.29 ----
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.29 ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = [f"\n### 18.29 T5 机会头消融 v3（3d/5d 经济账本，主标签 o_t1_bt_5d，2025开发）\n",
         "主标签=kt1.0 先止盈(T+2~T+6)；验收看 Top30% 实际净收益 5d(主)/3d(辅)、"
         "STOP(5d)、Brier/PR-AUC，对比基座逐头冻结。{'；'.join(base_notes)}。\n",
         "全局诊断（修复成功=盘中先修复+0.3ATR，即旧机会口径）：",
         *[f"- {n}" for n in gate_notes],
         "\n| 头 | fold | 版本 | hiT1(5d) | hi净收益5d | hi净收益3d | hiSTOP5d | Brier | PR-AUC |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in res.iterrows():
        L.append(f"| {r['head']} | {r['fold']} | {r['model']} | "
                 f"{r['hi_t1']:.1%} | {r['hi_nr5']:+.3%} | "
                 f"{r['hi_nr3']:+.3%} | {r['hi_stop5d']:.1%} | "
                 f"{r['brier']:.4f} | {r['pr_auc']:.3f} |")
    L.append("")
    L.append("月份不劣（fold1，hi净收益5d ≥ 基座）：")
    for k, v in month_gate.items():
        L.append(f"- {k}: {v}")
    sr = pd.DataFrame(src_rows)
    L.append("")
    L.append("src分层（fold1）：")
    L.append("| 头 | src | 版本 | hi净收益5d | hiSTOP5d | Brier |")
    L.append("|---|---|---|---:|---:|---:|")
    for _, r in sr.iterrows():
        L.append(f"| {r['head']} | {r['src']} | {r['model']} | "
                 f"{r['hi_nr5']:+.3%} | {r['hi_stop5d']:.1%} | "
                 f"{r['brier']:.4f} |")
    L.append("\n判读：机会头只回答3~5日经济价值；通过闸门（且月份≥3/4不劣、"
             "src方向一致）的版本进入合成。明细："
             f"ablation_opp*{tag}*.csv")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
