"""v2_ablation — TIMING-TRIGGER-v2 的 clean A1-A4 消融 (复用 a0a4 的 split/IC 方法学).

纪律 (与 a0a4 完全一致):
  同一 Universe / 同一 Sample IDs / 同一 Label(timing_return_h1, v2 重算) / 同一 Split /
  同一 Execution / 同一 LGBM params / 同一 seeds; 唯一变量 = Feature Group.

tier 结构 (对齐 a0a4 A1-A4 语义增量):
  a1 = CORE (P_CLOSE 候选特征)
  a2 = a1 + ZONE (ATR_ref / Zone 几何)
  a3 = a2 + DIP (realized/zone-pen/current_* + open_dip/dip_depth/dip_speed, 窗口门控后)
  a4 = a3 + RECOVERY (recovery/reclaim/reclaim_failure/time/wick/vol/vwap)

标签: timing_return_h1 (v2 recomputed, 锚 new actual_fill).
切片: 全量 + 按 trigger_path_v2 (BOTTOM_STABILIZATION/RECOVERY/RECLAIM, DIP_RECOVERY/RECLAIM).
  trigger_path 第一轮只做诊断切片, 不喂模型/不据此筛样本.

产物: datasets/timing_universe_v2_ablation.json (IC 主表 + 按路径 IC + Sample Identity + Feature Delta)
"""
from __future__ import annotations
import json
import sys
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import a0a4_config as C  # noqa

sys.path.insert(0, str(C.PREFLIGHT))
import lightgbm as lgb  # noqa
import a0a4_train as T  # noqa (复用 build_split / split_hash / _spearman / economics / fold_metrics)

LABEL = "timing_return_h1"
ALPHA = "timing_market_alpha_h1"
CAT = {"setup_type", "dip_stage"}
PATHS = ["BOTTOM_STABILIZATION", "BOTTOM_RECOVERY", "BOTTOM_RECLAIM",
         "DIP_RECOVERY", "DIP_RECLAIM"]
TOP_Q = 0.95

# ---- v2 特征分组 (对齐 a0a4 A1-A4 增量语义) ----
V2_CORE = [
    "setup_type", "dip_stage",
    "prior_strength_pass", "structure_intact_flag", "prior_impulse_return",
    "ma20_slope_5", "ret_60d", "ret_20d", "dist_from_60d_high", "range_pos60",
    "range_pos20", "rsi14", "dist_low60", "distribution_score_raw", "support_break_depth_atr",
]
V2_ZONE = [
    "atr_ref", "zone_strength", "zone_distance_atr", "penetration_atr",
    "primary_zone_distance_atr", "zone_band_atr", "n_zones", "zone_n_sources",
]
V2_DIP = [
    "realized_dip_atr", "current_price_dip_atr", "zone_penetration_low_atr",
    "current_zone_distance_atr", "current_break_depth_atr",
    "open_dip_atr", "total_dip_atr", "dip_depth_5m_atr", "dip_depth_30m_atr",
    "dip_speed_5m", "dip_speed_15m", "dip_speed_30m",
]
V2_RECOVERY = [
    "recovery_from_zone_low_atr", "recovery_atr", "dip_recovery_ratio",
    "reclaim_zone_lower", "reclaim_zone_mid", "reclaim_zone_upper", "reclaim_failure_count",
    "time_since_zone_touch", "time_below_zone",
    "lower_wick_ratio", "sell_volume_ratio", "rebound_volume_ratio", "vwap_reclaim",
    # A4 合同补全 (causal at S, B): 冻结 A4_RECOVERY 原有字段
    "close_position_15m", "zone_break_depth_atr", "zone_break_duration",
    "cum_volume_vs_hist_same_time", "cum_turnover_vs_hist_same_time",
]
TIERS = {
    "a1": list(V2_CORE),
    "a2": list(V2_CORE) + list(V2_ZONE),
    "a3": list(V2_CORE) + list(V2_ZONE) + list(V2_DIP),
    "a4": list(V2_CORE) + list(V2_ZONE) + list(V2_DIP) + list(V2_RECOVERY),
}


def _jdef(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def fix_categories(tr, te, feats):
    for f in feats:
        if f in CAT:
            cats = sorted(set(tr[f].astype(str)) | {"NA"})
            tr[f] = pd.Categorical(tr[f].astype(str), categories=cats)
            te[f] = pd.Categorical(te[f].astype(str), categories=cats)
    return tr, te


def train_tier(tier, feats, df, assign, folds):
    dfm = df.merge(assign[["sample_id", "fold"]], on="sample_id", how="left")
    preds = []
    for f in folds:
        tr = dfm[(dfm["fold"] < 0) & (dfm["plan_date"] <= f["train_hi"])].copy()
        te = dfm[dfm["fold"] == f["fold"]].copy()
        tr, te = fix_categories(tr, te, feats)
        for seed in C.SEEDS:
            ds = lgb.Dataset(tr[feats], label=tr[LABEL].astype(float),
                            categorical_feature=[x for x in feats if x in CAT],
                            free_raw_data=False)
            params = C.lgbm_params_with_seed(seed)
            model = lgb.train(params, ds, num_boost_round=params["n_estimators"])
            pr = te[["sample_id", "plan_date", LABEL, ALPHA, "trigger_path_v2"]].copy()
            pr["pred"] = model.predict(te[feats])
            pr["fold"], pr["seed"] = f["fold"], seed
            preds.append(pr)
    pr_all = pd.concat(preds, ignore_index=True)
    oof = pr_all.groupby("sample_id").agg(
        pred=("pred", "mean"), plan_date=("plan_date", "first"),
        fold=("fold", "first")).reset_index()
    y = pr_all.drop_duplicates("sample_id")[
        ["sample_id", LABEL, ALPHA, "trigger_path_v2"]]
    oof = oof.merge(y, on="sample_id")

    ics = []
    for f in folds:
        sub = pr_all[pr_all["fold"] == f["fold"]]
        for seed in C.SEEDS:
            ss = sub[sub["seed"] == seed]
            ic = T._spearman(ss["pred"], ss[LABEL])
            if ic is not None:
                ics.append(ic)
    ic_arr = np.asarray(ics, dtype=float)

    # by-path IC + top-bucket label mean
    path_ic = {}
    for p in PATHS:
        sub = oof[oof["trigger_path_v2"] == p]
        icp = T._spearman(sub["pred"], sub[LABEL])
        path_ic[p] = {"n": int(len(sub)),
                      "ic": (None if icp is None else round(float(icp), 4)),
                      "label_mean": (None if len(sub) == 0 else round(float(sub[LABEL].mean()), 6))}

    pooled = T.economics(oof, LABEL, ALPHA)
    thr = oof["pred"].quantile(TOP_Q)
    top = oof[oof["pred"] >= thr]
    top_e = T.economics(top, LABEL, ALPHA)
    res = {
        "n_oof_test": int(len(oof)),
        "ic_mean": round(float(ic_arr.mean()), 4) if len(ic_arr) else None,
        "ic_std": round(float(ic_arr.std()), 4) if len(ic_arr) else None,
        "ic_ir": round(float(ic_arr.mean() / ic_arr.std()), 4) if len(ic_arr) and ic_arr.std() > 0 else None,
        "pooled_label_mean": pooled.get("label_mean"),
        "pooled_loss_rate": pooled.get("loss_rate"),
        "top_label_mean": top_e.get("label_mean"),
        "top_loss_rate": top_e.get("loss_rate"),
        "path_ic": path_ic,
    }
    return res, oof


def main():
    t0 = time.time()
    df = pd.read_parquet(C.DATASETS / "timing_universe_v2_train.parquet")
    # 准备: 数值化 + 类别
    all_feats = list(V2_CORE) + list(V2_ZONE) + list(V2_DIP) + list(V2_RECOVERY)
    df["setup_type"] = df["primary_setup"].astype(str)
    df["dip_stage"] = df["dip_stage"].fillna("NA").astype(str)
    df["trigger_path_v2"] = df["trigger_path_v2"].astype(str)
    for f in all_feats:
        if f in CAT:
            continue
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors="coerce")

    assign, folds = T.build_split(df)
    shash = T.split_hash(folds, assign)

    out = {}
    id_hashes = {}
    for tier, feats in TIERS.items():
        t1 = time.time()
        res, oof = train_tier(tier, feats, df, assign, folds)
        out[tier] = res
        id_hashes[tier] = hashlib.sha256(
            "|".join(sorted(oof["sample_id"])).encode()).hexdigest()[:16]
        print(f"[v2_ablation] {tier}: ic_mean={res['ic_mean']} ic_ir={res['ic_ir']} "
              f"top_label={res['top_label_mean']} ({time.time()-t1:.0f}s)", flush=True)

    sample_identity = (len({h for h in id_hashes.values()}) == 1)
    delta = {}
    for hi, lo in (("a2", "a1"), ("a3", "a2"), ("a4", "a3")):
        delta[f"{lo}->{hi}"] = sorted(set(TIERS[hi]) - set(TIERS[lo]))

    result = {
        "n_train": int(len(df)),
        "label": LABEL, "alpha": ALPHA,
        "split_hash": shash,
        "folds": [f["fold"] for f in folds],
        "seeds": C.SEEDS,
        "tiers": out,
        "sample_identity_pass": bool(sample_identity),
        "id_hashes": id_hashes,
        "feature_delta": delta,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (C.DATASETS / "timing_universe_v2_ablation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_jdef), encoding="utf-8")

    # ---- 打印主表 ----
    print("\n================ v2 A1-A4 Ablation (timing_return_h1, v2 recomputed) ================")
    print(f"{'tier':5s} {'n':>6s} {'IC':>8s} {'ICstd':>8s} {'IR':>7s} "
          f"{'top_label':>11s} {'pool_label':>11s} {'top_loss':>9s}")
    for t in ("a1", "a2", "a3", "a4"):
        r = out[t]
        print(f"{t:5s} {r['n_oof_test']:6d} {_s(r['ic_mean'],6):>8s} {_s(r['ic_std'],6):>8s} "
              f"{_s(r['ic_ir'],6):>7s} {_p(r['top_label_mean']):>11s} {_p(r['pooled_label_mean']):>11s} "
              f"{_p(r['top_loss_rate']):>9s}")
    print("\n[by-path IC / n / label_mean] (A4)")
    a4 = out["a4"]["path_ic"]
    for p in PATHS:
        v = a4[p]
        print(f"  {p:24s} n={v['n']:5d} ic={_s(v['ic'],4):>7s} label_mean={_p(v['label_mean'])}")
    print(f"\nSAMPLE_IDENTITY_PASS={sample_identity}  split_hash={shash[:12]}")
    print("FEATURE_DELTA:")
    for k, v in delta.items():
        print(f"  {k}: +{v}")
    print(f"[v2_ablation] wrote datasets/timing_universe_v2_ablation.json ({time.time()-t0:.0f}s)")


def _s(x, nd=4):
    if x is None:
        return "n/a"
    return f"{x:+.{nd}f}"


def _p(x):
    if x is None:
        return "n/a"
    return f"{x*100:+.2f}%"


if __name__ == "__main__":
    main()
