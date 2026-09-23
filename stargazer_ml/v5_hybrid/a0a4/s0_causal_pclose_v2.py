"""S0_CAUSAL_P_CLOSE_V2 — 重建干净的 S0 Selection 证据链 (用户 2026-09-22 裁定).

污染结论 (已冻结):
  BW-W4_LEAKED            = INVALID_FOR_PERFORMANCE_INFERENCE
  OLD_S0_A2_A3_A4        = INVALIDATED_BY_D_DAY_LOOKAHEAD
  原因: a14 的 zone/dip/recovery/volume 特征 feature_asof_ts = D 15:00,
        而 S0 label 从 D open 起 → 未来信息泄漏. 原文件保留, 不覆盖.
  架构裁定: 从此 S0 不再允许 A3/A4 命名/特征组; A3/A4 永久属于 Timing.
            S0 正式只做 A1 (P_CLOSE 日线) / A2-PCLOSE (P_CLOSE zone 表征).

本脚本只跑 P_CLOSE-safe 特征组 (不复现任何 D 日特征):
  S0-A1         = A1_CORE (盘后 quality, 全部 POOL_INHERITED)
  S0-A1p        = A1 + zone proximity (atr_ref + geometry + pc_zone_distance_atr)
  S0-A2-PCLOSE  = A1 + 完整 P_CLOSE zone 表征 (zone_band_atr/n_zones/zone_n_sources/pc_zone_strength)
  —— 无 S0-A3 / S0-A4 ——

复用 bottom_washout_ablation_v0 的 build_split/train_version/_analyze/_spearman/SLICES.
同 universe (全 s0_eligible, 含 BOTTOM+DIP) / 同 label s0_return_h3 / 同 split / 同 seeds / 同 LGBM.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import a0a4_config as C  # noqa
sys.path.insert(0, str(_HERE.parent / "step9_preflight"))
import bottom_washout_ablation_v0 as M  # noqa

OUT = C.RUN / "s0_causal_pclose_v2"
OUT.mkdir(parents=True, exist_ok=True)
LABEL, ALPHA, H5 = M.LABEL, M.ALPHA, M.H5

# ---------- P_CLOSE-safe 特征组 ----------
A1 = [
    "prior_strength_pass", "structure_intact_flag", "prior_impulse_return",
    "ma20_slope_5", "ret_60d", "ret_20d", "dist_from_60d_high",
    "range_pos60", "range_pos20", "rsi14", "dist_low60",
    "distribution_score_raw", "support_break_depth_atr",
    "setup_type", "dip_stage",   # categorical (primary_setup / dip_stage)
]
ZONE_PROX = ["atr_ref", "zone_lower", "zone_upper", "zone_mid", "pc_zone_distance_atr"]
ZONE_FULL = ZONE_PROX + ["zone_band_atr", "n_zones", "zone_n_sources", "pc_zone_strength"]

VERS = {
    "S0-A1":        A1,
    "S0-A1p":       A1 + ZONE_PROX,
    "S0-A2-PCLOSE": A1 + ZONE_FULL,
}


def _derive_zone_strength(df):
    """从 zones_json 取 primary zone 的 zone_strength (plan_date 几何, P_CLOSE-safe)."""
    def _s(z):
        try:
            zz = json.loads(z) if isinstance(z, str) else z
            if not isinstance(zz, list) or not zz:
                return np.nan
            prim = [x for x in zz if isinstance(x, dict) and str(x.get("zone_origin", "")).lower() == "primary"]
            cand = prim[0] if prim else zz[0]
            return float(cand.get("zone_strength", np.nan))
        except Exception:
            return np.nan
    df["pc_zone_strength"] = df["zones_json"].apply(_s)
    return df


def _pct(sub, col, lo=0.5, hi=99.5):
    if col not in sub or sub[col].notna().sum() == 0:
        return None
    return (round(float(sub[col].quantile(lo / 100)), 3),
            round(float(sub[col].quantile(hi / 100)), 3))


def a4_attribution(name, oof):
    """20260701 归因: pre/post 候选构成 + zone_distance 分布 + label/alpha 均值
    + COMMON-COVERAGE universe (pre/post 都出现的 code) 重算."""
    print(f"\n--- A4 20260701 attribution ({name}) ---")
    pre = oof[oof["plan_date"] < 20260701]
    post = oof[oof["plan_date"] >= 20260701]
    for tag, sub in [("PRE", pre), ("POST", post)]:
        ic = M._spearman(sub["pred"], sub[LABEL])
        n_b = int(sub["is_bottom"].sum()); n_d = int(sub["is_dip"].sum())
        zd = _pct(sub, "pc_zone_distance_atr")
        print(f"  {tag}: n={len(sub)} ic={round(ic,4) if ic is not None else None} "
              f"BOTTOM={n_b} DIP={n_d} zone_dist[p50,p99]={zd} "
              f"label_mean={round(float(sub[LABEL].mean()),4)} "
              f"alpha_mean={round(float(sub[ALPHA].mean()),4)}")
    # COMMON-COVERAGE universe: code 在 pre & post 都出现
    pre_codes = set(pre["sample_id"].str.split("|").str[0])
    post_codes = set(post["sample_id"].str.split("|").str[0])
    common = pre_codes & post_codes
    cm = oof[oof["sample_id"].str.split("|").str[0].isin(common)]
    pre_c = cm[cm["plan_date"] < 20260701]
    post_c = cm[cm["plan_date"] >= 20260701]
    ic_pre_c = M._spearman(pre_c["pred"], pre_c[LABEL])
    ic_post_c = M._spearman(post_c["pred"], post_c[LABEL])
    print(f"  COMMON-COVERAGE: codes={len(common)} "
          f"PRE ic={round(ic_pre_c,4) if ic_pre_c is not None else None} "
          f"(n={len(pre_c)})  POST ic={round(ic_post_c,4) if ic_post_c is not None else None} "
          f"(n={len(post_c)})  -> post 翻转=|{round(ic_post_c,4) if ic_post_c is not None else None}|")
    return {"pre_n": len(pre), "post_n": len(post), "common_codes": len(common),
            "pre_ic": ic_pre_c, "post_ic": ic_post_c}


def main():
    t0 = time.time()
    a = pd.read_parquet(C.DATASETS / "a14_universe.parquet")
    a["setup_type"] = a["primary_setup"].astype(str)
    a["dip_stage"] = a["dip_stage"].astype(str).fillna("NA")
    pc = pd.read_parquet(C.DATASETS / "bottom_washout_pclose_w4.parquet")
    df = a.merge(pc, on="sample_id", how="left")
    df = _derive_zone_strength(df)
    df = df[df["s0_eligible"] == 1].copy()  # S0 全候选 universe (含 BOTTOM+DIP)
    df["is_bottom"] = df["is_bottom"].astype(bool)
    df["is_dip"] = df["is_dip"].astype(bool)
    allfeats = sorted({c for v in VERS.values() for c in v})
    for c in allfeats:
        if c not in C.CATEGORICAL_FEATURES and c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    assign, folds = M.build_split(df)
    print(f"[s0-pclose-v2] universe n={len(df)} "
          f"folds={[(f['fold'], int(f['test_lo']), int(f['test_hi'])) for f in folds]}")

    results, a4 = {}, {}
    for name, feats in VERS.items():
        oof, imp = M.train_version(name, feats, df, assign, folds)
        m = M._analyze(oof)
        results[name] = m
        oof.to_parquet(OUT / f"predictions_{name}.parquet", index=False)
        imp.to_csv(OUT / f"feature_importance_{name}.csv", index=False)
        per_fold = {f["fold"]: M._spearman(oof[oof["fold"] == f["fold"]]["pred"],
                                            oof[oof["fold"] == f["fold"]][LABEL]) for f in folds}
        print(f"\n=== {name} (n_feat={len(feats)}) === ic_mean={m['ic_mean']} ic_std={m['ic_std']} "
              f"top-pool={m['top_minus_pool_label']} beats_RND_P95={m['top_beats_random_p95']}")
        print("  per-fold ic:", {k: (round(v, 4) if v is not None else None) for k, v in per_fold.items()})
        for sname, cond in M.SLICES.items():
            sub = oof if cond is None else oof[cond(oof)]
            if len(sub) < 30:
                continue
            print(f"    {sname}: n={len(sub)} ic={round(M._spearman(sub['pred'], sub[LABEL]),4)}")
        if name == "S0-A1p":
            a4[name] = a4_attribution(name, oof)

    import json as _j
    summary = {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)} for k, v in results.items()}
    summary["A4"] = a4
    _j.dump(summary, open(OUT / "metrics.json", "w"), default=str, indent=2)
    print(f"\n[s0-pclose-v2] done {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
