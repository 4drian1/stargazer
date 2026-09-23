"""a0a4_train — SAMPLE_IDENTITY / FEATURE_DELTA 检查 + 3-seed walk-forward 训练 + 指标.

运行前提: datasets/a14_universe.parquet 与 datasets/a0_universe.parquet 已构建。

产出:
  a1/..a4/: feature_manifest.json sample_ids.parquet split_manifest.json
            predictions.parquet metrics.json feature_importance.csv models/*.txt
  a0/:      feature_manifest.json sample_ids.parquet split_manifest.json
            metrics.json (负基线重放经济学, 不训练)
  checks.json (SAMPLE_IDENTITY_PASS / FEATURE_DELTA_PASS)

纪律: 同一 Universe / Sample IDs / Label / Split / Execution / Cost / LGBM params /
      seeds; 唯一变化 = Feature Group。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import a0a4_config as C  # noqa

sys.path.insert(0, str(C.PREFLIGHT))
import trading_calendar as TC  # noqa
import lightgbm as lgb  # noqa

TOP_Q = 0.95          # top bucket 分位 (fold 内)
SLICE_TOP_Q = 0.90    # 切片内 top 分位 (pooled OOF)
N_BOOT = 500


# ---------------------------------------------------------------- split
def build_split(df: pd.DataFrame) -> pd.DataFrame:
    cal = sorted(int(x) for x in TC.load_calendar())
    cal_arr = np.asarray(cal)

    def prev_td(d, n):
        i = int(np.searchsorted(cal_arr, d, side="left"))
        return int(cal_arr[max(0, i - n)])

    folds = []
    for f in C.FOLDS:
        folds.append({"fold": f["fold"], "test_lo": f["test_lo"], "test_hi": f["test_hi"],
                      "train_hi": prev_td(f["test_lo"], C.PURGE_TRADING_DAYS)})
    out = df[["sample_id", "plan_date"]].copy()
    out["fold"] = -1
    for f in folds:
        m = (out["plan_date"] >= f["test_lo"]) & (out["plan_date"] <= f["test_hi"])
        out.loc[m, "fold"] = f["fold"]
    out["role"] = np.where(out["fold"] > 0, "test",
                           np.where(out["plan_date"] <= folds[-1]["train_hi"],
                                    "train_only", "unused_tail"))
    return out, folds


def split_hash(folds, assign: pd.DataFrame) -> str:
    obj = {"folds": folds,
           "n_by_fold": {int(k): int(v) for k, v in
                         assign["fold"].value_counts().items()},
           "purge": C.PURGE_TRADING_DAYS}
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- features
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["setup_type"] = df["primary_setup"].astype(str)
    if "dip_stage" in df:
        df["dip_stage"] = df["dip_stage"].fillna("NA").astype(str)
    else:
        df["dip_stage"] = "NA"
    if "final_state" in df:
        df["final_state"] = df["final_state"].fillna("NA").astype(str)
    for f, _ in C.A1_CORE + C.A2_ZONE + C.A3_DIP + C.A4_RECOVERY:
        if f in C.CATEGORICAL_FEATURES:
            continue
        if f in df:
            df[f] = pd.to_numeric(df[f], errors="coerce")
    return df


def fix_categories(train: pd.DataFrame, test: pd.DataFrame, feats):
    for f in feats:
        if f in C.CATEGORICAL_FEATURES:
            cats = sorted(set(train[f].astype(str)) | {"NA"})
            train[f] = pd.Categorical(train[f].astype(str), categories=cats)
            test[f] = pd.Categorical(test[f].astype(str), categories=cats)
    return train, test


# ---------------------------------------------------------------- metrics
def _spearman(a, b):
    a, b = pd.Series(a), pd.Series(b)
    m = a.notna() & b.notna()
    if m.sum() < 5:
        return None
    return float(a[m].rank().corr(b[m].rank()))


def economics(sub: pd.DataFrame, label: str, alpha: str) -> dict:
    if len(sub) == 0:
        return {"n": 0}
    r = {"n": int(len(sub))}
    r["label_mean"] = _rnd(sub[label].mean())
    r["label_median"] = _rnd(sub[label].median())
    r["alpha_mean"] = _rnd(sub[alpha].mean()) if alpha in sub else None
    for k in ("mfe_30m", "mae_30m", "mfe_h1", "mae_h1"):
        if k in sub:
            r[k] = _rnd(sub[k].mean())
    r["loss_rate"] = _rnd((sub[label] < 0).mean())
    if "failed_rebound" in sub:
        r["failed_rebound_rate"] = _rnd(sub["failed_rebound"].mean())
    return r


def _rnd(x, nd=6):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def fold_metrics(oof: pd.DataFrame, label: str, alpha: str) -> dict:
    """oof: fold 内 test 行 (seed 均值 pred)。"""
    m = {"n_test": int(len(oof))}
    m["ic"] = _spearman(oof["pred"], oof[label])
    try:
        q = pd.qcut(oof["pred"].rank(method="first"), 5, labels=False)
        means = oof.groupby(q)[label].mean()
        m["quintile_label_mean"] = {f"Q{i+1}": _rnd(means.get(i)) for i in range(5)}
        m["q5_minus_q1"] = _rnd(means.get(4, np.nan) - means.get(0, np.nan))
        m["quintile_monotonic_spearman"] = _spearman(list(means.index), list(means.values))
    except Exception:
        m["quintile_label_mean"] = None
        m["q5_minus_q1"] = None
        m["quintile_monotonic_spearman"] = None
    thr = oof["pred"].quantile(TOP_Q)
    top = oof[oof["pred"] >= thr]
    pool = economics(oof, label, alpha)
    m["pool"] = pool
    m["top"] = economics(top, label, alpha)
    m["top_minus_pool_label"] = (_rnd(m["top"].get("label_mean") - pool["label_mean"])
                                 if m["top"].get("label_mean") is not None else None)
    # Top vs Random P95 (bootstrap 同尺寸随机抽取)
    n_top = len(top)
    if n_top >= 5:
        rng = np.random.default_rng(123)
        vals = oof[label].to_numpy()
        draws = rng.choice(vals, size=(N_BOOT, n_top), replace=True).mean(axis=1)
        m["random_p95_label_mean"] = _rnd(np.quantile(draws, 0.95))
        m["top_beats_random_p95"] = bool(
            m["top"].get("label_mean") is not None and
            m["top"]["label_mean"] > m["random_p95_label_mean"])
    m["pred_dispersion"] = _rnd(oof["pred"].std())
    return m


def slice_metrics(oof_test: pd.DataFrame, label: str, alpha: str) -> dict:
    out = {}
    for name, cond in C.SLICES.items():
        sub = oof_test if cond is None else oof_test[cond(oof_test)]
        if len(sub) < 30:
            out[name] = {"n": int(len(sub)), "skipped": "n<30"}
            continue
        r = economics(sub, label, alpha)
        r["ic"] = _spearman(sub["pred"], sub[label])
        thr = sub["pred"].quantile(SLICE_TOP_Q)
        top = sub[sub["pred"] >= thr]
        r["top"] = economics(top, label, alpha)
        out[name] = r
    return out


# ---------------------------------------------------------------- checks
def run_checks(df: pd.DataFrame) -> dict:
    tiers = {t: [f for f, _ in C.TIER_FEATURES[t]] for t in ("a1", "a2", "a3", "a4")}
    delta_ok, delta_detail = True, {}
    for (hi, lo), expected in C.EXPECTED_DELTA.items():
        actual = sorted(set(tiers[hi]) - set(tiers[lo]))
        ok = actual == sorted(expected)
        delta_ok &= ok
        delta_detail[f"{lo}->{hi}"] = {"expected": sorted(expected), "actual": actual,
                                       "ok": ok}
    missing = {t: [f for f in fs if f not in df.columns]
               for t, fs in tiers.items()}
    identity = {"basis": "A1-A4 训练输入为同一 DataFrame (同 Universe/Label/Split), "
                         "训练后逐 tier 对比 predictions 的 sample_id 集合作显式证明"}
    for tr, spec in C.LABEL_TRACKS.items():
        el = df[df[spec["eligibility"]] == 1]
        identity[f"n_eligible_{tr}"] = int(len(el))
        identity[f"sample_id_set_hash_{tr}"] = hashlib.sha256(
            "|".join(sorted(el["sample_id"])).encode()).hexdigest()[:16]
    checks = {"FEATURE_DELTA_PASS": bool(delta_ok and not any(missing.values())),
              "feature_delta_detail": delta_detail,
              "missing_columns": missing,
              "SAMPLE_IDENTITY_PASS": None,   # 训练后由 predictions 显式证明
              "sample_identity_detail": identity}
    return checks


# ---------------------------------------------------------------- training
def train_tier(tier: str, df: pd.DataFrame, assign: pd.DataFrame, folds: list,
               shash: str, ds_meta: dict) -> dict:
    dfm = df.merge(assign[["sample_id", "fold"]], on="sample_id", how="left")
    tdir = C.TIER_DIRS[tier]
    (tdir / "models").mkdir(parents=True, exist_ok=True)
    preds_all, imp_rows, metrics = [], [], {"tier": tier, "tracks": {}}
    for track, spec in C.LABEL_TRACKS.items():
        t0 = time.time()
        feats = C.track_features(tier, track)
        if feats is None:
            # S0 轨仅定义 A1/A2; A3/A4 含 D 日盘中特征, 违反 S0 契约 -> N/A, 不训练
            metrics["tracks"][track] = {
                "status": "N/A_NOT_ELIGIBLE_FOR_S0_ROLE",
                "reason": ("S0 标签锚 D 日 open, 特征必须 as-of <= P_CLOSE; "
                           "A3/A4 的 D 日盘中特征 (open_dip_atr/recovery/reclaim/...) "
                           "违反契约, 故 S0 仅保留 A1/A2, A3/A4 标记为 N/A。")}
            print(f"[train] {tier}/{track}: N/A (not eligible for S0 role)")
            continue
        el = dfm[dfm[spec["eligibility"]] == 1].copy()
        label, alpha = spec["label"], spec["alpha"]
        preds_all_tr = []
        for f in folds:
            tr = el[(el["fold"] < 0) & (el["plan_date"] <= f["train_hi"])]
            te = el[el["fold"] == f["fold"]]
            tr, te = fix_categories(tr, te, feats)
            for seed in C.SEEDS:
                ds = lgb.Dataset(tr[feats], label=tr[label].astype(float),
                                 categorical_feature=[
                                     x for x in feats if x in C.CATEGORICAL_FEATURES],
                                 free_raw_data=False)
                params = C.lgbm_params_with_seed(seed)
                model = lgb.train(params, ds, num_boost_round=params["n_estimators"])
                model.save_model(str(tdir / "models" /
                                     f"{tier}_{track}_f{f['fold']}_s{seed}.txt"))
                pr = te[["sample_id", "plan_date", label, alpha]].copy()
                pr["pred"] = model.predict(te[feats])
                pr["fold"], pr["seed"], pr["tier"], pr["track"] = \
                    f["fold"], seed, tier, track
                for k in ("mfe_30m", "mae_30m", "mfe_h1", "mae_h1", "failed_rebound",
                          "is_bottom", "is_dip", "dual_hit_score40",
                          "v1_rejected_v11_accepted", "dip_stage"):
                    if k in te:
                        pr[k] = te[k].to_numpy()
                preds_all_tr.append(pr)
                imp = model.feature_importance(importance_type="gain")
                imp_rows.append(pd.DataFrame(
                    {"feature": feats, "gain": imp, "track": track,
                     "fold": f["fold"], "seed": seed}))
        pr_tr = pd.concat(preds_all_tr, ignore_index=True)
        preds_all.append(pr_tr)
        # ---- 指标 ----
        oof = pr_tr.groupby("sample_id").agg(
            pred=("pred", "mean"), plan_date=("plan_date", "first"),
            fold=("fold", "first")).reset_index()
        y = pr_tr.drop_duplicates("sample_id")[
            ["sample_id", label, alpha, "mfe_30m", "mae_30m", "mfe_h1", "mae_h1",
             "failed_rebound", "is_bottom", "is_dip", "dual_hit_score40",
             "v1_rejected_v11_accepted", "dip_stage"]]
        oof = oof.merge(y, on="sample_id")
        per_fold = {}
        ics = []
        for f in folds:
            sub = pr_tr[pr_tr["fold"] == f["fold"]]
            for seed in C.SEEDS:
                ss = sub[sub["seed"] == seed]
                ic = _spearman(ss["pred"], ss[label])
                if ic is not None:
                    ics.append(ic)
            oof_f = oof[oof["fold"] == f["fold"]]
            per_fold[f"fold{f['fold']}"] = fold_metrics(oof_f, label, alpha)
        ic_arr = np.asarray(ics, dtype=float)
        pooled = economics(oof, label, alpha)
        oof_test = oof  # OOF 仅含 test 行 (按构造)
        metrics["tracks"][track] = {
            "label": label,
            "n_eligible": int(len(el)),
            "n_oof_test": int(len(oof)),
            "ic_by_fold_seed": [_rnd(x) for x in ics],
            "ic_mean": _rnd(ic_arr.mean()) if len(ic_arr) else None,
            "ic_std": _rnd(ic_arr.std()) if len(ic_arr) else None,
            "ic_ir": _rnd(ic_arr.mean() / ic_arr.std()) if len(ic_arr) and ic_arr.std() > 0 else None,
            "per_fold": per_fold,
            "pooled_oof": pooled,
            "slices": slice_metrics(oof_test, label, alpha),
        }
        print(f"[train] {tier}/{track}: ic_mean={metrics['tracks'][track]['ic_mean']} "
              f"({time.time()-t0:.0f}s)")
    pd.concat(preds_all, ignore_index=True).to_parquet(
        tdir / "predictions.parquet", index=False)
    imp = pd.concat(imp_rows, ignore_index=True)
    imp_g = imp.groupby(["track", "feature"])["gain"].agg(["mean", "std"]).reset_index()
    imp_g.to_csv(tdir / "feature_importance.csv", index=False)
    metrics["hashes"] = dict(ds_meta["hashes"], split_hash=shash,
                             config_fingerprint=C.config_fingerprint())
    (tdir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=_jdefault),
        encoding="utf-8")
    return metrics


# ---------------------------------------------------------------- A0 replay
def a0_replay(df: pd.DataFrame, assign: pd.DataFrame, folds: list, shash: str,
              ds_meta: dict) -> dict:
    dfm = df.merge(assign[["sample_id", "fold"]], on="sample_id", how="left")
    metrics = {"tier": "a0", "mode": "REPLAY_ONLY_NO_MODEL",
               "note": "V0 旧门控宇宙在冻结 Label/Execution 下的负基线; "
                       "A0->A1 = Candidate System Increment",
               "universe": {}, "per_fold": {}, "old_score_top10": {}}
    metrics["universe"]["timing"] = economics(
        dfm[dfm["timing_eligible"] == 1], "timing_return_h1", "timing_market_alpha_h1")
    metrics["universe"]["s0"] = economics(
        dfm[dfm["s0_eligible"] == 1], "s0_return_h3", "s0_market_alpha_h3")
    for f in folds:
        sub = dfm[dfm["fold"] == f["fold"]]
        metrics["per_fold"][f"fold{f['fold']}"] = {
            "timing": economics(sub[sub["timing_eligible"] == 1],
                                "timing_return_h1", "timing_market_alpha_h1"),
            "s0": economics(sub[sub["s0_eligible"] == 1],
                            "s0_return_h3", "s0_market_alpha_h3")}
    # 旧系统自身排序: intraday_dip_score 每日 top10% (历史对照的控制臂)
    for track, spec in C.LABEL_TRACKS.items():
        el = dfm[(dfm[spec["eligibility"]] == 1) & (dfm["fold"] > 0)]
        tops = []
        for p, g in el.groupby("plan_date"):
            thr = g["intraday_dip_score"].quantile(0.90)
            tops.append(g[g["intraday_dip_score"] >= thr])
        tops = pd.concat(tops) if tops else pd.DataFrame()
        metrics["old_score_top10"][track] = economics(
            tops, spec["label"], spec["alpha"]) if len(tops) else {"n": 0}
    metrics["hashes"] = dict(ds_meta["hashes"], split_hash=shash,
                             config_fingerprint=C.config_fingerprint())
    return metrics


# ---------------------------------------------------------------- manifests
def write_feature_manifest(tier: str, df_meta_cols):
    groups = {"a1": C.A1_CORE, "a2": C.A2_ZONE, "a3": C.A3_DIP, "a4": C.A4_RECOVERY}
    feats = []
    if tier == "a0":
        man = {"tier": "a0", "features": [],
               "mode": "REPLAY_ONLY", "note": "A0 无模型; 见 metrics.json"}
    else:
        for t in ("a1", "a2", "a3", "a4"):
            if t == tier:
                feats = groups[t]
                break
        man = {"tier": tier,
               "features": [{"name": f, "origin": o,
                             "group_added": t2,
                             "formula": C.DERIVED_FORMULAS.get(f)}
                            for t2 in ("a1", "a2", "a3", "a4")
                            for f, o in C.TIER_FEATURES[t2]
                            if f in [x for x, _ in feats]],
               "categorical": sorted(C.CATEGORICAL_FEATURES & {f for f, _ in feats}),
               "lgbm_params": C.LGBM_PARAMS, "seeds": C.SEEDS}
    return man


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


# ---------------------------------------------------------------- main
def main():
    t_all = time.time()
    df = pd.read_parquet(C.DATASETS / "a14_universe.parquet")
    df = prepare_features(df)
    print(f"[train] a14 universe: {len(df)} rows")
    checks = run_checks(df)
    import a0a4_gates as G
    gates = G.run_all_gates(df)
    checks["MODEL_ROLE_FEATURE_PASS"] = gates["MODEL_ROLE_FEATURE_PASS"]["pass"]
    checks["TRAINING_TIME_CAUSALITY_PASS"] = gates["TRAINING_TIME_CAUSALITY_PASS"]
    checks["TIMING_TRIGGER_SEMANTICS_PASS"] = gates["TIMING_TRIGGER_SEMANTICS_PASS"]
    (C.RUN / "checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2, default=_jdefault),
        encoding="utf-8")
    print(f"[train] FEATURE_DELTA_PASS={checks['FEATURE_DELTA_PASS']}")
    print(f"[train] MODEL_ROLE_FEATURE_PASS={checks['MODEL_ROLE_FEATURE_PASS']}")
    print(f"[train] TRAINING_TIME_CAUSALITY_PASS={checks['TRAINING_TIME_CAUSALITY_PASS']}")
    print(f"[train] TIMING_TRIGGER_SEMANTICS_PASS={checks['TIMING_TRIGGER_SEMANTICS_PASS']}")
    if not checks["FEATURE_DELTA_PASS"]:
        print("[train] ABORT: feature delta check failed")
        sys.exit(2)

    assign, folds = build_split(df)
    shash = split_hash(folds, assign)
    with open(C.DATASETS / "a14_build_manifest.json", encoding="utf-8") as f:
        ds_meta = json.load(f)
    for t in ("a0", "a1", "a2", "a3", "a4"):
        C.TIER_DIRS[t].mkdir(parents=True, exist_ok=True)

    split_man = {"folds": folds, "purge_trading_days": C.PURGE_TRADING_DAYS,
                 "split_hash": shash, "note": C.SPLIT_HASH_NOTE,
                 "role_counts": assign["role"].value_counts().to_dict()}
    assign.to_parquet(C.TIER_DIRS["a1"] / "sample_ids.parquet", index=False)
    for t in ("a2", "a3", "a4", "a0"):
        C.TIER_DIRS[t].mkdir(parents=True, exist_ok=True)
        assign.to_parquet(C.TIER_DIRS[t] / "sample_ids.parquet", index=False)
    for t in ("a0", "a1", "a2", "a3", "a4"):
        (C.TIER_DIRS[t] / "split_manifest.json").write_text(
            json.dumps(split_man, ensure_ascii=False, indent=2, default=_jdefault),
            encoding="utf-8")

    for t in ("a0", "a1", "a2", "a3", "a4"):
        C.TIER_DIRS[t].mkdir(parents=True, exist_ok=True)
    for tier in ("a1", "a2", "a3", "a4"):
        (C.TIER_DIRS[tier] / "feature_manifest.json").write_text(
            json.dumps(write_feature_manifest(tier, df.columns), ensure_ascii=False,
                       indent=2, default=_jdefault), encoding="utf-8")
        train_tier(tier, df, assign, folds, shash, ds_meta)

    # ---- 训练后显式身份证明: 各 tier 各 track 的 sample_id 集合必须完全一致 ----
    id_proof = {}
    for tr in C.LABEL_TRACKS:
        hashes = {}
        for tier in ("a1", "a2", "a3", "a4"):
            p = pd.read_parquet(C.TIER_DIRS[tier] / "predictions.parquet")
            s = p[p["track"] == tr]["sample_id"].unique()
            hashes[tier] = (len(s), hashlib.sha256(
                "|".join(sorted(s)).encode()).hexdigest()[:16])
        id_proof[tr] = {"per_tier": hashes,
                        "sets_identical": len({h for _, h in hashes.values()}) == 1}
    checks["SAMPLE_IDENTITY_PASS"] = all(v["sets_identical"] for v in id_proof.values())
    checks["sample_identity_proof"] = id_proof
    (C.RUN / "checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2, default=_jdefault),
        encoding="utf-8")
    print(f"[train] SAMPLE_IDENTITY_PASS={checks['SAMPLE_IDENTITY_PASS']}")
    if not checks["SAMPLE_IDENTITY_PASS"]:
        sys.exit(3)
    # A0
    df0 = pd.read_parquet(C.DATASETS / "a0_universe.parquet")
    assign0, folds0 = build_split(df0)
    shash0 = split_hash(folds0, assign0)
    with open(C.DATASETS / "a0_build_manifest.json", encoding="utf-8") as f:
        ds_meta0 = json.load(f)
    assign0.to_parquet(C.TIER_DIRS["a0"] / "sample_ids.parquet", index=False)
    m0 = a0_replay(df0, assign0, folds0, shash0, ds_meta0)
    (C.TIER_DIRS["a0"] / "metrics.json").write_text(
        json.dumps(m0, ensure_ascii=False, indent=2, default=_jdefault),
        encoding="utf-8")
    (C.TIER_DIRS["a0"] / "feature_manifest.json").write_text(
        json.dumps(write_feature_manifest("a0", df0.columns), ensure_ascii=False,
                   indent=2, default=_jdefault), encoding="utf-8")
    print(f"[train] ALL DONE ({time.time()-t_all:.0f}s)")


if __name__ == "__main__":
    main()
