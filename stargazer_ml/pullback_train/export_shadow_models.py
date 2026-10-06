# -*- coding: utf-8 -*-
"""导出影子模式冻结模型（2026-09-09 冻结清单）。

冻结版本（均基座 R0/O0，无 D0）：
  AM-R1 = M0+T1F        风险头，标签 r_sbr_t2
  PM-R2 = M0+T2PM       风险头，标签 r_sbr_t2
  AM-O4 = M0+T4_V2      机会头，标签 o_t1_bt_5d
  PM-O6 = M0+T6O        机会头，标签 o_t1_bt_5d

校准：2025 池按 FOLDS 做 OOF 预测（与消融同切分），Isotonic 拟合 OOF P→y；
最终模型在 2025 全年池上重训（与 2026 复核口径一致：train=2025 全年）。
输出 shadow_models/shadow_t5_frozen.pkl + .json（特征清单/版本/校准器元数据）。
"""
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

import ablation_opp as AO
import ablation_risk as AR

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
MODEL_DIR = os.path.join(OUT_DIR, "shadow_models")
PKL = os.path.join(MODEL_DIR, "shadow_t5_frozen.pkl")
META = os.path.join(MODEL_DIR, "shadow_t5_frozen.json")

# (key, head, kind, feats, 标签, 池构建)
SPECS = {
    "AM_R1": ("AM", "risk", AR.M0 + AR.T1F, "r_sbr_t2"),
    "PM_R2": ("PM", "risk", AR.M0 + AR.T2PM, "r_sbr_t2"),
    "AM_O4": ("AM", "opp", AO.M0 + AO.T4_V2, "o_t1_bt_5d"),
    "PM_O6": ("PM", "opp", AO.M0 + AO.T6O, "o_t1_bt_5d"),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_pool(head, kind):
    """按头×任务口径构建 2025 池（风险/机会过滤不同，与各自消融一致）。

    kind="risk"（同 AR.load_data）：net_ret_t2.0 与 r_sbr_t2 非空；
    kind="opp"（同 AO.load_data）：o_t1_bt_5d 非空。
    """
    dps = AR.HEADS[head]
    df = pd.read_parquet(AR.SAMPLES)
    df = df[df["dp"].isin(dps)].copy()
    files = [(AR.D0, ["code", "date"]), (AR.PF, ["code", "dp_date", "dp"]),
             (AR.ST, ["code", "dp_date", "dp"]), (AR.TPM, ["code", "dp_date", "dp"]),
             (AR.T6, ["code", "dp_date", "dp"]), (AR.LBL, ["code", "dp_date", "dp"]),
             (AO.T3, ["code", "dp_date", "dp"]), (AO.T4, ["code", "dp_date", "dp"])]
    for p, on in files:
        m = pd.read_parquet(p)
        m["code"] = m["code"].astype(str)
        df["code"] = df["code"].astype(str)
        df = df.merge(m, on=on, how="left")
    df = df[(df["entry_type"] == "vwap") & (df["censored"] == 0)].copy()
    if kind == "risk":
        df = df[df[f"net_ret_t{AR.KT}"].notna()
                & df["r_sbr_t2"].notna()].copy()
    else:
        df = df[df["o_t1_bt_5d"].notna()].copy()
    if "price_scale_valid" not in df.columns:
        df["price_scale_valid"] = True
    df = df[df["price_scale_valid"].astype(bool)].copy()
    return df


def make_clf():
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="binary", num_leaves=31,
                              learning_rate=0.05, min_child_samples=100,
                              subsample=0.9, colsample_bytree=0.9, seed=7,
                              n_estimators=400, verbosity=-1, n_jobs=8)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {"models": {}, "calibrators": {}, "meta": {}}
    pools = {}
    for key, (head, kind, feats, ycol) in SPECS.items():
        pk = (head, kind)
        if pk not in pools:
            pools[pk] = build_pool(head, kind)
        df = pools[pk]
        y = df[ycol].astype(float)
        # ---- OOF（与消融同切分）拟合 Isotonic 校准器 ----
        oof = pd.Series(np.nan, index=df.index)
        for _, ftr0, ftr1, t0d, t1d in AR.FOLDS:
            tr_m = (df["date"] >= ftr0) & (df["date"] <= ftr1)
            te_m = (df["date"] >= t0d) & (df["date"] <= t1d)
            clf = make_clf()
            clf.fit(df.loc[tr_m, feats], y.loc[tr_m])
            oof.loc[te_m] = clf.predict_proba(df.loc[te_m, feats])[:, 1]
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof.values, y.values)
        brier_raw = float(((oof - y) ** 2).mean())
        brier_iso = float(((pd.Series(iso.predict(oof.values), index=df.index) - y) ** 2).mean())
        # ---- 最终模型：2025 全年池 ----
        clf = make_clf()
        clf.fit(df[feats], y)
        bundle["models"][key] = clf
        bundle["calibrators"][key] = iso
        bundle["meta"][key] = {
            "head": head, "kind": kind, "label": ycol, "features": feats,
            "n_train": int(len(df)), "base_rate": float(y.mean()),
            "brier_oof_raw": brier_raw, "brier_oof_isotonic": brier_iso,
        }
        log(f"{key}: n={len(df):,} base={y.mean():.1%} "
            f"brier raw={brier_raw:.4f} -> iso={brier_iso:.4f}")
    # ---- 参考EV：训练池（机会池）按 头×象限 的 3d/5d 净收益均值 ----
    # 注意：这是"同象限历史均值"参考值，不是模型预测（EV回归 R²<0 已否决，
    # §18.31）；记录进决策表时标注为 ev3_ref/ev5_ref。
    from shadow_batch import score as _score
    bundle["ev_ref"] = {}
    for head in AR.HEADS:
        pool = pools[(head, "opp")]
        sc = _score(bundle, pool)
        means = sc.groupby("quadrant")[["o_net_ret_3d", "o_net_ret_5d"]].mean()
        bundle["ev_ref"][head] = {
            q: {"ev3_ref": float(r["o_net_ret_3d"]),
                "ev5_ref": float(r["o_net_ret_5d"])}
            for q, r in means.iterrows()}
    bundle["meta"]["_frozen_at"] = "2026-09-09"
    bundle["meta"]["_frozen_list"] = "AM-R1 / PM-R2 / AM-O4 / PM-O6 (T7=OOD only)"
    with open(PKL, "wb") as f:
        pickle.dump(bundle, f)
    with open(META, "w", encoding="utf-8") as f:
        json.dump(bundle["meta"], f, ensure_ascii=False, indent=2)
    log(f"wrote {PKL} / {META}")


if __name__ == "__main__":
    main()
