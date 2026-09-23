"""W1 (P_CLOSE 修正后唯一正增量组) 稳健性 + 家族 leave-one-out 审计。
复用 bottom_washout_ablation_v0 的 build_split / train_version / _analyze / _spearman。
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import a0a4_config as C  # noqa
import bottom_washout_ablation_v0 as M  # noqa

def build_df():
    a = pd.read_parquet(C.DATASETS / "a14_universe.parquet")
    f = pd.read_parquet(C.DATASETS / "bottom_washout_v0_features.parquet")
    fcols = ["sample_id"] + [c for c in f.columns if c.startswith("W")
             or c in ("daily_feature_source_date", "daily_feature_staleness_days", "suspended_gap_flag")]
    df = a.merge(f[fcols], on="sample_id", how="inner")
    pc = pd.read_parquet(C.DATASETS / "bottom_washout_pclose_w4.parquet")
    df = df.merge(pc, on="sample_id", how="left")
    df = df[df["is_bottom"].astype(bool)].copy()
    df = df[df["s0_eligible"] == 1].copy()
    df["setup_type"] = df["primary_setup"].astype(str)
    df["dip_stage"] = df["dip_stage"].fillna("NA").astype(str)
    feats = sorted({c for v in [M.W0_BASE, M.W0_CAT, M.WASHOUT_CORE, M.W2_ADD, M.W4_ADD, M.BW_W3] for c in v})
    for c in feats:
        if c not in C.CATEGORICAL_FEATURES and c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def per_fold_ic(oof):
    out = {}
    for f in C.FOLDS:
        sub = oof[oof["fold"] == f["fold"]]
        ics = [M._spearman(sub[sub["seed"] == s][["pred", M.LABEL]].dropna(), ) for s in C.SEEDS]
        # 上面的简化会出错, 用显式
    return out

def fold_ic_list(oof):
    res = {}
    for f in C.FOLDS:
        sub = oof[oof["fold"] == f["fold"]]
        vals = []
        for s in C.SEEDS:
            ss = sub[sub["seed"] == s]
            ic = M._spearman(ss["pred"], ss[M.LABEL])
            if ic is not None:
                vals.append(ic)
        res[f["fold"]] = (round(float(np.mean(vals)), 4) if vals else None,
                          [round(v, 3) for v in vals])
    return res

def split_ic(oof, mask_fn, name):
    sub = oof[mask_fn(oof)]
    if len(sub) < 30:
        return f"{name}: n={len(sub)} skip"
    # per fold x seed
    vals = []
    for f in C.FOLDS:
        s2 = sub[sub["fold"] == f["fold"]]
        for s in C.SEEDS:
            ss = s2[s2["seed"] == s]
            ic = M._spearman(ss["pred"], ss[M.LABEL])
            if ic is not None:
                vals.append(ic)
    return f"{name}: n={len(sub)} ic={round(float(np.mean(vals)),4) if vals else None}"

def main():
    df = build_df()
    assign, folds = M.build_split(df)
    W1 = M.W0_BASE + M.W0_CAT + M.WASHOUT_CORE
    W1_no_zone = M.W0_BASE + M.W0_CAT + [x for x in M.WASHOUT_CORE if x != "pc_zone_distance_atr"]
    W1_only_zone = M.W0_BASE + M.W0_CAT + ["pc_zone_distance_atr"]  # 去掉 4 个 dynamics
    variants = {"W1": W1, "W1_no_zone_distance": W1_no_zone, "W1_only_zone": W1_only_zone}
    for name, feats in variants.items():
        oof, _ = M.train_version(name, feats, df, assign, folds)
        m = M._analyze(oof)
        print(f"\n=== {name} ===  ic_mean={m['ic_mean']} ic_std={m['ic_std']} "
              f"top-pool={m['top_minus_pool_label']} beats_RND_P95={m['top_beats_random_p95']}")
        print("  per-fold (mean, [seed ICs]):", fold_ic_list(oof))
        print("  " + split_ic(oof, lambda d: d["plan_date"] < 20260701, "PRE_20260701"))
        print("  " + split_ic(oof, lambda d: d["plan_date"] >= 20260701, "POST_20260701"))
        print("  " + split_ic(oof, lambda d: (d["is_bottom"].astype(bool)) & (~d["is_dip"].astype(bool)), "BOTTOM_ONLY"))
        print("  " + split_ic(oof, lambda d: (d["is_bottom"].astype(bool)) & (d["is_dip"].astype(bool)), "BOTH"))
        top = m["top"]
        print(f"  TOP: label={top.get('label_mean')} alpha={top.get('alpha_mean')} "
              f"h5={top.get('h5_mean')} mfe_h1={top.get('mfe_h1_mean')} mae_h1={top.get('mae_h1_mean')} loss={top.get('loss_rate')}")

if __name__ == "__main__":
    main()
