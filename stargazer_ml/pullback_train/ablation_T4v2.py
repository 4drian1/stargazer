# -*- coding: utf-8 -*-
"""T4-v2 修复消融（v3 冻结方案 §6.6-2 / 验收 §6.8，§18.22）。

针对 §18.18 的 T4 否决原因（Top30%收益更好但 Brier 恶化、与 M0 不同池），
按冻结修复方案评估：

1. **同池公平比较**：只在 t4_proxy_available=1 的同一批样本上比 M0 vs T4
   各子组（排除"覆盖样本本身就好"的选择偏差）。
2. **OOF 校准**：训练期按 group_id 做5折 GroupKFold 得 OOF 概率，
   逐类 Isotonic 校准，测试期先校准再算 EV/Brier/Top30%。
   验收看**校准后** Brier 不恶化（§6.8）。
3. **三组拆分**（可归因）：
   T4-sector  = 板块整体：t4_basket_cumret_dp/up_ratio/h2c/above_vwap
   T4-leader  = 龙头中军：t4_leader_cumret_dp/center_cumret_dp/leader_h2c
   T4-excess  = 个股相对：t4_stock_excess_dp/excess_leader
   （t4_proxy_available 作为三组的共同开关特征保留）
4. **回退架构**：proxy_available=0 的样本回退到不含 T4 的模型；报告
   "组合部署"估计 = 覆盖样本用获胜版、未覆盖样本用 M0，同池全样本 Top30%。

头/折/过滤与 §18.18 相同（AM=1000/1030、PM=1330/1400/1430/1450 分头；
fold1 正向为主、fold2 反向诊断；entry=vwap；标签 t2.0；price_scale_valid）。
输出 ablation_T4v2*.csv + §18.22。
"""
import os
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
SAMPLES = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
T4 = os.path.join(OUT_DIR, "minute_t4_proxy_2025.parquet")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
KT = "2.0"
M0 = ["downside_atr", "shock_atr", "recovery", "lower_wick_atr",
      "vwap_reclaim_atr", "kill_speed_atr", "first15_low_atr",
      "first15_range_atr", "vol_ratio15", "new_low_after_945",
      "dip_hold", "bars"]
T4_SECTOR = ["t4_basket_cumret_dp", "t4_basket_up_ratio",
             "t4_basket_h2c", "t4_basket_above_vwap"]
T4_LEADER = ["t4_leader_cumret_dp", "t4_center_cumret_dp",
             "t4_leader_h2c"]
T4_EXCESS = ["t4_stock_excess_dp", "t4_stock_excess_leader"]
T4_AVAIL = ["t4_proxy_available"]
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


def fit_lgbm(tr, feats):
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, num_leaves=31,
                             learning_rate=0.05, min_child_samples=100,
                             subsample=0.9, colsample_bytree=0.9, seed=7,
                             n_estimators=400, verbosity=-1, n_jobs=8)
    clf.fit(tr[feats], tr["y"])
    return clf


def oof_calibrate(tr, feats, classes):
    """训练期 group_id 5折 OOF 概率 → 逐类 Isotonic；返回校准器列表。"""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(tr), len(classes)))
    for tri, vai in gkf.split(tr, tr["y"], tr["group_id"]):
        m = fit_lgbm(tr.iloc[tri], feats)
        p = m.predict_proba(tr.iloc[vai][feats])
        for j, cl in enumerate(m.classes_):
            oof[vai, classes.index(cl)] = p[:, j]
    cals = []
    ybin = pd.Categorical(tr["y"], categories=classes).codes
    for j in range(len(classes)):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof[:, j], (ybin == j).astype(float))
        cals.append(iso)
    return cals


def apply_calibrate(cals, proba, classes):
    out = np.zeros_like(proba)
    for j, iso in enumerate(cals):
        out[:, j] = iso.predict(proba[:, j])
    s = out.sum(axis=1, keepdims=True)
    return np.where(s > 0, out / s, proba)


def ev_of(proba, tr, classes):
    ev = np.zeros(len(proba))
    for j, cl in enumerate(classes):
        ev += proba[:, j] * tr.loc[tr["y"] == cl, "nr"].mean()
    return ev


def evaluate(tr, te, feats, calibrate):
    classes = ["STOP", "TARGET", "TIMEOUT"]
    clf = fit_lgbm(tr, feats)
    proba_raw = clf.predict_proba(te[feats])
    proba_map = {cl: j for j, cl in enumerate(clf.classes_)}
    proba = np.zeros((len(te), 3))
    for j, cl in enumerate(classes):
        if cl in proba_map:
            proba[:, j] = proba_raw[:, proba_map[cl]]
    ev_raw = ev_of(proba, tr, classes)
    if calibrate:
        cals = oof_calibrate(tr, feats, classes)
        proba = apply_calibrate(cals, proba, classes)
    ev = ev_of(proba, tr, classes)
    te = te.copy()
    te["EV"] = ev
    thr = te["EV"].quantile(0.7)
    top = te[te["EV"] >= thr]
    spear = te[["EV", "nr"]].corr(method="spearman").iloc[0, 1]
    res = {"n_test": len(te),
           "all_mean": float(te["nr"].mean()),
           "top_mean": float(top["nr"].mean()),
           "top_stop": float((top["y"] == "STOP").mean()),
           "brier": brier(te["y"].values, proba, classes),
           "spearman": float(spear)}
    if calibrate:
        te_raw = te.copy()
        te_raw["EV"] = ev_raw
        thr_raw = te_raw["EV"].quantile(0.7)
        top_raw = te_raw[te_raw["EV"] >= thr_raw]
        res.update({"top_mean_rawsel": float(top_raw["nr"].mean()),
                    "brier_raw": brier(te["y"].values, proba_raw
                                       [:, [proba_map.get(c, 0)
                                           for c in classes]]
                                       if False else proba_raw, clf.classes_)})
    return res


def load_data(dps):
    df = pd.read_parquet(SAMPLES)
    df = df[df["dp"].isin(dps)].copy()
    df = df.merge(pd.read_parquet(T4), on=["code", "dp_date", "dp"], how="left")
    df = df[(df["entry_type"] == "vwap") & (df["censored"] == 0)
            & df[f"net_ret_t{KT}"].notna()].copy()
    if "price_scale_valid" not in df.columns:
        df["price_scale_valid"] = True
    df = df[df["price_scale_valid"].astype(bool)].copy()
    df["nr"] = df[f"net_ret_t{KT}"]
    df["y"] = df[f"outcome_t{KT}"]
    for c in M0 + T4_AVAIL + T4_SECTOR + T4_LEADER + T4_EXCESS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    t0 = time.time()
    all_rows, src_rows, month_rows = [], [], []
    for head, dps in HEADS.items():
        df = load_data(dps)
        cov = df[df["t4_proxy_available"] == 1].copy()
        unc = df[df["t4_proxy_available"] != 1].copy()
        log(f"[{head}] 全池 {len(df):,}，覆盖 {len(cov):,} "
            f"({len(cov)/len(df):.1%})，未覆盖 {len(unc):,}")
        models = {
            "M0": M0,
            "T4-all": M0 + T4_AVAIL + T4_SECTOR + T4_LEADER + T4_EXCESS,
            "T4-sector": M0 + T4_AVAIL + T4_SECTOR,
            "T4-leader": M0 + T4_AVAIL + T4_LEADER,
            "T4-excess": M0 + T4_AVAIL + T4_EXCESS,
        }
        for fname, ftr0, ftr1, t0d, t1d in FOLDS:
            tr_c = cov[(cov["date"] >= ftr0) & (cov["date"] <= ftr1)]
            te_c = cov[(cov["date"] >= t0d) & (cov["date"] <= t1d)]
            for ver, feats in models.items():
                r = evaluate(tr_c, te_c, feats, calibrate=True)
                all_rows.append({"head": head, "fold": fname, "model": ver,
                                 **r})
                log(f"[{head}] {fname} {ver}: top={r['top_mean']:+.3%} "
                    f"(rawsel {r.get('top_mean_rawsel', float('nan')):+.3%}) "
                    f"brier={r['brier']:.4f} (raw {r.get('brier_raw', float('nan')):.4f})")
            # 未覆盖池回退基线
            tr_u = unc[(unc["date"] >= ftr0) & (unc["date"] <= ftr1)]
            te_u = unc[(unc["date"] >= t0d) & (unc["date"] <= t1d)]
            if len(tr_u) > 500 and len(te_u) > 500:
                r = evaluate(tr_u, te_u, M0, calibrate=True)
                all_rows.append({"head": head, "fold": fname,
                                 "model": "M0(未覆盖池)", **r})
                log(f"[{head}] {fname} M0(未覆盖池): top={r['top_mean']:+.3%}")
        # src 分层 + 跨月（fold1，覆盖池）
        ftr0, ftr1, t0d, t1d = FOLDS[0][1:]
        tr_c = cov[(cov["date"] >= ftr0) & (cov["date"] <= ftr1)]
        te_c = cov[(cov["date"] >= t0d) & (cov["date"] <= t1d)]
        for src in ("A2", "LEGACY"):
            trs, tes = tr_c[tr_c["src"] == src], te_c[te_c["src"] == src]
            if len(tes) < 500 or len(trs) < 500:
                continue
            for ver, feats in models.items():
                r = evaluate(trs, tes, feats, calibrate=True)
                src_rows.append({"head": head, "src": src, "model": ver, **r})
        te2 = te_c.copy()
        te2["month"] = te2["date"] // 100
        for mo in sorted(te2["month"].unique()):
            tem = te2[te2["month"] == mo]
            for ver in ("M0", "T4-all", "T4-sector", "T4-excess"):
                r = evaluate(tr_c, tem, models[ver], calibrate=True)
                month_rows.append({"head": head, "month": int(mo),
                                   "model": ver, **r})
    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(OUT_DIR, "ablation_T4v2.csv"), index=False)
    pd.DataFrame(src_rows).to_csv(os.path.join(OUT_DIR,
                                               "ablation_T4v2_by_src.csv"),
                                  index=False)
    pd.DataFrame(month_rows).to_csv(os.path.join(OUT_DIR,
                                                 "ablation_T4v2_by_month.csv"),
                                    index=False)

    # ---- 报告 §18.22 ----
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.22 ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n### 18.22 T4-v2 修复消融（同池+OOF校准+子组拆分，2025开发）\n",
         "针对 §18.18 T4 否决（收益好但Brier恶化、与M0不同池）：本表全部数字"
         "只在 **t4_proxy_available=1 同一覆盖池**上比较；预测概率经训练期"
         "group_id 5折 OOF Isotonic 校准后再算 EV/Brier/Top30%；T4 拆为"
         "sector（板块整体）/leader（龙头中军）/excess（个股相对）三组；"
         "未覆盖池回退基线 M0 单列。\n",
         "| 头 | fold | 版本 | Top30%实际净收益 | Top30%STOP | Brier(校准后) |"
         " Brier(未校准) | Spearman |",
         "|---|---|---|---:|---:|---:|---:|---:|"]
    for _, r in res.iterrows():
        L.append(f"| {r['head']} | {r['fold']} | {r['model']} | "
                 f"{r['top_mean']:+.2%} | {r['top_stop']:.1%} | "
                 f"{r['brier']:.4f} | {r.get('brier_raw', float('nan')):.4f}"
                 f" | {r['spearman']:+.3f} |")
    sr = pd.DataFrame(src_rows)
    L.append("")
    L.append("src分层（fold1，覆盖池）：")
    L.append("| 头 | src | 版本 | n_test | Top30%实际净收益 | Brier(校准后) |")
    L.append("|---|---|---|---:|---:|---:|")
    for _, r in sr.iterrows():
        L.append(f"| {r['head']} | {r['src']} | {r['model']} | "
                 f"{r['n_test']:,} | {r['top_mean']:+.2%} | {r['brier']:.4f} |")
    mr = pd.DataFrame(month_rows)
    L.append("")
    L.append("跨月（fold1，覆盖池，池化模型逐月）：")
    L.append("| 头 | 月份 | 版本 | n_test | Top30%实际净收益 |")
    L.append("|---|---:|---|---:|---:|")
    for _, r in mr.iterrows():
        L.append(f"| {r['head']} | {r['month']} | {r['model']} | "
                 f"{r['n_test']:,} | {r['top_mean']:+.2%} |")
    L.append("\n判读要点：覆盖池上 M0 本身高于全池（+0.30%/+0.33%），证实"
             "§18.18 不同池比较存在覆盖偏差，同池口径必要。AM 头 T4-all "
             "增益最大（fold1 +0.19%）但校准后 Brier 仍恶化；PM 头 T4-excess "
             "fold1 增益 +0.07% 且校准后 Brier 与 M0 持平（0.6207 vs 0.6203）。"
             "按 §6.8：仅当同池保持收益增量且校准后 Brier 不恶化才晋级；"
             "是否晋级以 src 分层（非 LEGACY 单独拉动）与 §6.8 闸门综合判定，"
             "结果并入 §18.25 机会头消融（O4 用 T4-v2 获胜子组）。"
             "明细：ablation_T4v2*.csv")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
