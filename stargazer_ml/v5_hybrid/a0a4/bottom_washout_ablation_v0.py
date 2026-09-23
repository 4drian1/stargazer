"""BOTTOM-WASHOUT-v0 消融 (BW-W0 .. BW-W4, 主链; W3 = 独立 challenger).

纪律 (与 A0-A4 完全一致):
  - 同一 BOTTOM 候选 universe (is_bottom==1)
  - 同一 label: s0_return_h3 ; alpha: s0_market_alpha_h3 ; 同一 split / seeds / LGBM params
  - 唯一变化 = 特征组 (BW-W0 -> W1 -> W2 -> W4)
  - 样本不因特征 completeness 被删 (NaN 留给模型; lower_wick_ratio 保持 NaN)
  - W3 (技术指标) 不进入主链, 仅作独立 challenger 报告

BW 组定义 (用户 2026-09-22 冻结):
  BW-W0 = GREEN-WATCH v1.1 baseline (A1_CORE 候选特征)
  BW-W1 = W0 + pressure decay / washout core
  BW-W2 = W1 + support / penetration / reclaim
  BW-W4 = W2 + distribution / breakdown negatives
  BW-W3 (challenger) = 技术指标 (rsi6 / distance_ma20 / macd / boll)  <- 不进主链

增量归因看增量链: (W1-W0), (W2-W1), (W4-W2); 不是 W4>W0 就成功.

产出: bottom_washout_ablation/{metrics.json, predictions_<ver>.parquet, feature_importance.csv}
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import a0a4_config as C  # noqa
sys.path.insert(0, str(_HERE.parent / "step9_preflight"))
import trading_calendar as TC  # noqa
import lightgbm as lgb  # noqa
from a0a4_train import build_split, _spearman  # 复用 A0-A4 切分与 Rank IC

OUT = C.RUN / "bottom_washout_ablation"
OUT.mkdir(parents=True, exist_ok=True)

LABEL = "s0_return_h3"
ALPHA = "s0_market_alpha_h3"
H5 = "return_h5"          # H5 鲁棒性 proxy (s0_return_h5 不存在, 用 raw 5d return)
N_BOOT = 500
N_TOP_Q = 0.95

# ---------------- BW 特征组 (冻结) ----------------
# GREEN-WATCH v1.1 baseline (A1_CORE, 均存在于 a14)
W0_BASE = [
    "prior_strength_pass", "structure_intact_flag", "prior_impulse_return",
    "ma20_slope_5", "ret_60d", "ret_20d", "dist_from_60d_high",
    "range_pos60", "range_pos20", "rsi14", "dist_low60",
    "distribution_score_raw", "support_break_depth_atr",
]
W0_CAT = ["setup_type", "dip_stage"]  # 由 primary_setup / dip_stage 派生

WASHOUT_CORE = [  # W1 additions (P_CLOSE 修正版 zone_distance)
    "pc_zone_distance_atr", "W0_is_bottom",
    "W1_ret_5d_atr", "W1_negative_volume_ratio_5d", "W1_sell_volume_decay",
]
W2_ADD = [  # W2 additions (P_CLOSE 修正版 penetration/lower_wick)
    "pc_penetration_atr",
    "pc_lower_wick_ratio", "W2_low_break_depth_atr", "W2_support_test_count",
]
W4_ADD = [  # W4 additions (P_CLOSE 修正版 breakdown/volume negatives)
    "pc_zone_break_depth_atr", "pc_zone_break_duration", "new_low_flag",
    "pc_sell_volume_ratio", "pc_rebound_volume_ratio",
]
W3_CHALLENGER = [  # 技术指标, 独立 challenger, 不进主链
    "W4_rsi6_current", "W4_distance_ma20_atr", "W4_macd_hist_delta",
    "W4_macd_hist_acceleration", "W4_boll_pct_b",
]

BW = {
    "BW-W0": W0_BASE + W0_CAT,
    "BW-W1": W0_BASE + W0_CAT + WASHOUT_CORE,
    "BW-W2": W0_BASE + W0_CAT + WASHOUT_CORE + W2_ADD,
    "BW-W4": W0_BASE + W0_CAT + WASHOUT_CORE + W2_ADD + W4_ADD,
}
BW_W3 = W0_BASE + W0_CAT + W3_CHALLENGER  # 独立对照


def _rnd(x, nd=6):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def _fix_cat(tr, te, feats):
    for f in feats:
        if f in C.CATEGORICAL_FEATURES:
            cats = sorted(set(tr[f].astype(str)) | set(te[f].astype(str)) | {"NA"})
            tr[f] = pd.Categorical(tr[f].astype(str), categories=cats)
            te[f] = pd.Categorical(te[f].astype(str), categories=cats)
    return tr, te


def _bucket_stats(sub: pd.DataFrame) -> dict:
    if len(sub) == 0:
        return {"n": 0}
    r = {"n": int(len(sub)),
         "label_mean": _rnd(sub[LABEL].mean()),
         "alpha_mean": _rnd(sub[ALPHA].mean()) if ALPHA in sub else None,
         "h5_mean": _rnd(sub[H5].mean()) if H5 in sub else None,
         "mfe_h1_mean": _rnd(sub["mfe_h1"].mean()) if "mfe_h1" in sub else None,
         "mae_h1_mean": _rnd(sub["mae_h1"].mean()) if "mae_h1" in sub else None,
         "loss_rate": _rnd((sub[LABEL] < 0).mean())}
    return r


def _analyze(oof: pd.DataFrame) -> dict:
    """oof: 已含 pred / fold / seed / label / alpha / H5 / mfe / mae."""
    ic_by_fs = []
    for f in C.FOLDS:
        sub = oof[oof["fold"] == f["fold"]]
        for s in C.SEEDS:
            ss = sub[sub["seed"] == s]
            ic = _spearman(ss["pred"], ss[LABEL])
            if ic is not None:
                ic_by_fs.append(ic)
    ic_arr = np.asarray(ic_by_fs, dtype=float)
    ic_pooled = _spearman(oof["pred"], oof[LABEL])
    # quintile monotonicity (pooled oof)
    q = pd.qcut(oof["pred"].rank(method="first"), 5, labels=False)
    means = oof.groupby(q)[LABEL].mean()
    quintile = {f"Q{i + 1}": _rnd(means.get(i)) for i in range(5)}
    q5_q1 = _rnd(means.get(4, np.nan) - means.get(0, np.nan))
    mono = _spearman(list(means.index), list(means.values))
    # top vs pool
    thr = oof["pred"].quantile(N_TOP_Q)
    top = oof[oof["pred"] >= thr]
    pool = _bucket_stats(oof)
    top_s = _bucket_stats(top)
    # Top vs Random P95 (bootstrap 同尺寸)
    n_top = len(top)
    rp95, beats = None, None
    if n_top >= 5:
        rng = np.random.default_rng(123)
        vals = oof[LABEL].to_numpy()
        draws = rng.choice(vals, size=(N_BOOT, n_top), replace=True).mean(axis=1)
        rp95 = _rnd(np.quantile(draws, 0.95))
        beats = bool(top_s.get("label_mean") is not None and top_s["label_mean"] > rp95)
    return dict(
        n_oof=int(len(oof)),
        ic_mean=_rnd(ic_arr.mean()) if len(ic_arr) else None,
        ic_std=_rnd(ic_arr.std()) if len(ic_arr) and ic_arr.std() > 0 else None,
        ic_ir=_rnd(ic_arr.mean() / ic_arr.std()) if len(ic_arr) and ic_arr.std() > 0 else None,
        ic_pooled=_rnd(ic_pooled),
        quintile=quintile, q5_minus_q1=q5_q1,
        quintile_monotonic=_rnd(mono),
        top_n=int(n_top), top=top_s, pool=pool,
        top_minus_pool_label=_rnd(top_s.get("label_mean") - pool["label_mean"])
            if top_s.get("label_mean") is not None else None,
        random_p95_label_mean=rp95, top_beats_random_p95=beats,
    )


# 本地强制切片 (适配 BOTTOM universe: is_bottom==1; 不使用 C.SLICES 的 dual_hit_score40 等列)
SLICES = {
    "ALL": None,
    "BOTTOM_ONLY": (lambda d: d["is_bottom"].astype(bool) & ~d["is_dip"].astype(bool)),
    "BOTH": (lambda d: d["is_bottom"].astype(bool) & d["is_dip"].astype(bool)),
    "PRE_20260701": (lambda d: d["plan_date"] < 20260701),
    "POST_20260701": (lambda d: d["plan_date"] >= 20260701),
}


def _slices(oof: pd.DataFrame) -> dict:
    out = {}
    for name, cond in SLICES.items():
        sub = oof if cond is None else oof[cond(oof)]
        if len(sub) < 30:
            out[name] = {"n": int(len(sub)), "skipped": "n<30"}
            continue
        out[name] = {"n": int(len(sub)),
                     "ic_pooled": _rnd(_spearman(sub["pred"], sub[LABEL])),
                     "top": _bucket_stats(sub[sub["pred"] >= sub["pred"].quantile(N_TOP_Q)]),
                     "pool": _bucket_stats(sub)}
    return out


def train_version(version: str, feats: list, df: pd.DataFrame, assign: pd.DataFrame, folds: list):
    dfm = df.merge(assign[["sample_id", "fold"]], on="sample_id", how="left")
    preds, imp_rows = [], []
    for f in folds:
        tr = dfm[(dfm["fold"] < 0) & (dfm["plan_date"] <= f["train_hi"])]
        te = dfm[dfm["fold"] == f["fold"]]
        tr, te = _fix_cat(tr, te, feats)
        for seed in C.SEEDS:
            ds = lgb.Dataset(tr[feats], label=tr[LABEL].astype(float),
                             categorical_feature=[x for x in feats if x in C.CATEGORICAL_FEATURES],
                             free_raw_data=False)
            params = C.lgbm_params_with_seed(seed)
            model = lgb.train(params, ds, num_boost_round=params["n_estimators"])
            pr = te[["sample_id", "plan_date", LABEL, ALPHA, H5, "mfe_h1", "mae_h1",
                     "is_bottom", "is_dip"]].copy()
            pr["pred"] = model.predict(te[feats])
            pr["fold"], pr["seed"], pr["version"] = f["fold"], seed, version
            preds.append(pr)
            imp = model.feature_importance(importance_type="gain")
            imp_rows.append(pd.DataFrame({"feature": feats, "gain": imp, "seed": seed,
                                          "fold": f["fold"], "version": version}))
    pr_all = pd.concat(preds, ignore_index=True)
    oof = pr_all.groupby("sample_id").agg(
        pred=("pred", "mean"), plan_date=("plan_date", "first"),
        fold=("fold", "first"), seed=("seed", "first")).reset_index()
    y = pr_all.drop_duplicates("sample_id")[
        ["sample_id", LABEL, ALPHA, H5, "mfe_h1", "mae_h1", "is_bottom", "is_dip"]]
    oof = oof.merge(y, on="sample_id")
    return oof, pd.concat(imp_rows, ignore_index=True)


def main():
    t0 = time.time()
    a = pd.read_parquet(C.DATASETS / "a14_universe.parquet")
    f = pd.read_parquet(C.DATASETS / "bottom_washout_v0_features.parquet")
    # 只取 W* 特征 + 审计列 (避免 f 中 REF 副本与 a14 同名列冲突, 如 ma20_slope_5)
    fcols = ["sample_id"] + [c for c in f.columns if c.startswith("W")
             or c in ("daily_feature_source_date", "daily_feature_staleness_days", "suspended_gap_flag")]
    df = a.merge(f[fcols], on="sample_id", how="inner")
    # P_CLOSE 修正版 zone/volume 特征 (替换 a14 中的 D 日泄漏版)
    pc = pd.read_parquet(C.DATASETS / "bottom_washout_pclose_w4.parquet")
    df = df.merge(pc, on="sample_id", how="left")
    # BOTTOM universe (固定)
    df = df[df["is_bottom"].astype(bool)].copy()
    df = df[df["s0_eligible"] == 1].copy()
    df["setup_type"] = df["primary_setup"].astype(str)
    df["dip_stage"] = df["dip_stage"].fillna("NA").astype(str)
    # numeric coercion (非 categorical 特征)
    all_feats = sorted({c for v in BW.values() for c in v} | set(BW_W3))
    for c in all_feats:
        if c not in C.CATEGORICAL_FEATURES and c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"[bw-ablation] BOTTOM universe: {len(df)} rows, label={LABEL}")
    # 同一 split (复用 A0-A4 walk-forward + purge)
    assign, folds = build_split(df)
    shash = hashlib_sha(assign, folds)
    print(f"[bw-ablation] split_hash={shash}  folds={[(f['fold'],f['test_lo'],f['test_hi']) for f in folds]}")

    results = {}
    preds_all = []
    imp_all = []
    for ver, feats in BW.items():
        miss = [c for c in feats if c not in df.columns]
        if miss:
            print(f"[bw-ablation] {ver}: MISSING {miss} -> skip")
            continue
        t1 = time.time()
        oof, imp = train_version(ver, feats, df, assign, folds)
        m = _analyze(oof)
        m["slices"] = _slices(oof)
        results[ver] = m
        preds_all.append(oof.assign(version=ver))
        imp_all.append(imp)
        print(f"[bw-ablation] {ver}: ic_mean={m['ic_mean']} top-pool={m['top_minus_pool_label']} "
              f"({time.time()-t1:.0f}s)")
    # W3 challenger (独立, 不进主链)
    oof3, imp3 = train_version("BW-W3(challenger)", BW_W3, df, assign, folds)
    results["BW-W3(challenger)"] = _analyze(oof3)
    results["BW-W3(challenger)"]["slices"] = _slices(oof3)
    preds_all.append(oof3)
    imp_all.append(imp3)
    print(f"[bw-ablation] BW-W3(challenger): ic_mean={results['BW-W3(challenger)']['ic_mean']}")

    # 增量归因 (主链)
    chain = ["BW-W0", "BW-W1", "BW-W2", "BW-W4"]
    incr = {}
    prev = None
    for v in chain:
        if v not in results:
            continue
        cur = results[v]
        if prev is not None and prev in results:
            incr[f"{v}-vs-{prev}"] = {
                "ic_mean_delta": _rnd(cur["ic_mean"] - results[prev]["ic_mean"]),
                "top_pool_delta": _rnd(cur["top_minus_pool_label"] - results[prev]["top_minus_pool_label"])}
        prev = v
    results["incremental_chain"] = incr

    out = dict(
        universe="is_bottom==1 & s0_eligible==1", n=len(df),
        label=LABEL, alpha=ALPHA, split_hash=shash,
        seeds=C.SEEDS, lgbm_params=C.LGBM_PARAMS,
        groups={k: v for k, v in BW.items()},
        w3_challenger=BW_W3,
        results=results,
    )
    (OUT / "metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=_jdefault), encoding="utf-8")
    pd.concat(preds_all).to_parquet(OUT / "predictions_all.parquet", index=False)
    pd.concat(imp_all).to_csv(OUT / "feature_importance.csv", index=False)
    print(f"\n[bw-ablation] wrote metrics.json + predictions_all.parquet + feature_importance.csv ({time.time()-t0:.0f}s)")
    # 总结表
    print("\n=== BW 主链 IC / Top-Pool / 增量 ===")
    print(f"{'ver':12s} {'ic_mean':>8s} {'ic_std':>8s} {'top-pool':>9s} {'ic_delta':>9s}")
    prev = None
    for v in chain:
        if v not in results:
            continue
        m = results[v]
        d = incr.get(f"{v}-vs-{prev}", {}).get("ic_mean_delta") if prev else None
        print(f"{v:12s} {str(m['ic_mean']):>8s} {str(m['ic_std']):>8s} "
              f"{str(m['top_minus_pool_label']):>9s} {str(d):>9s}")
        prev = v


def hashlib_sha(assign, folds):
    import hashlib
    obj = {"folds": folds, "purge": C.PURGE_TRADING_DAYS,
           "n_by_fold": {int(k): int(v) for k, v in assign["fold"].value_counts().items()}}
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def _jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


if __name__ == "__main__":
    main()
