"""drift_audit_preliminary_gate.py — PRELIMINARY 通道历史漂移审计（路线 B 闸门）。

问题（开发路线 #12.3）：v1.0.0 冻结特征中 6/10 原始特征取 label_start(=T+1) 当日数据
（gap_return=T+1 开盘跳空、volume_shock=T+1 量能、is_st/float_mv/total_mv/dist_high_250
=T+1 状态与收盘），完全合规打分时刻 = T+1 收盘后。本审计量化：若用"截至 T 收盘可见"
（早盘口径：gap/is_st 用 T+1 真实值）的合成特征打分，p_materiality 相对真实特征漂移多大。

口径：
- REAL   ：train_final 同款 build_features（全量日线，label_start 行真实值）——对照组
- TCLOSE ：截至 T 收盘合成（gap=NaN→中位数、volume_shock=NaN→中位数、
           is_st/float_mv/total_mv/dist_high 用 T 日值）——"盘后按钮"假想口径
- MORNING：TCLOSE + gap_return 用真实 T+1 跳空 + is_st 用 T+1 真实状态
           （volume_shock 与收盘类仍缺）——"早盘 9:30 后按钮"实际可达口径

预注册闸门（先看数字前冻结；全 FAIL 才放行 PRELIMINARY，任一 FAIL → 回退路线 A）：
- G1  Spearman(p_syn, p_real) ≥ 0.90            （TCLOSE 与 MORNING 各自）
- G2  mean|Δp| ≤ 0.05                          （同上）
- G3  P(|Δp| > 0.15) ≤ 2%                     （同上）
- G4  Jaccard(p_syn ≥ 0.5, p_real ≥ 0.5) ≥ 0.85（同上）
- G5  按年 Spearman ≥ 0.85（各年，告知项，不单独否决）

本脚本不重训、不改模型、不改特征契约；只读 artifacts + out/。
产物：out/drift_audit_preliminary/（audit_report.md / results.parquet / gates.json）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from corvus.materiality_v1.materiality_features_v1 import (  # noqa: E402
    MODEL_FEATURES, TOP_CATEGORIES, build_features)
from _taxonomy import KEYWORD_MATERIAL_WEIGHT, title_keyword_features  # noqa: E402
from c4_shadow import get_model  # noqa: E402

OUT_DIR = ROOT / "out" / "drift_audit_preliminary"

GATES = {
    "G1_spearman_min": 0.90,
    "G2_mean_abs_dp_max": 0.05,
    "G3_p_abs_dp_gt_0.15_max": 0.02,
    "G4_jaccard_p5_min": 0.85,
    "G5_year_spearman_min": 0.85,  # 告知项
}


# ---- 特征行计算（与 build_features 同数学；QC 对拍 build_features 必须 max_abs_diff=0） ----
def row_features(daily: pd.DataFrame, e: pd.Series, *, gap_value,
                 volume_shock_value, override_is_st=None) -> dict | None:
    """daily 需含 label_start 行（REAL）或截至 T 的行（TCLOSE/MORNING，末行=T）。

    gap_value / volume_shock_value: 记录级覆盖（REAL 用存储值；TCLOSE/MORNING 的
    T+1 量能为未知 → NaN 走中位数填充）；override_is_st: MORNING 用 T+1 真实状态。
    """
    daily = daily.copy()
    daily["date"] = daily["date"].astype(int)
    dates = daily["date"].tolist()
    if "label_start" in e and int(e["label_start"]) in set(dates):
        idx = dates.index(int(e["label_start"]))
    else:
        idx = len(daily) - 1  # 合成口径：末行即 T
    row = daily.iloc[idx]
    prev20 = daily.iloc[max(0, idx - 20): idx]
    prev250 = daily.iloc[max(0, idx - 250): idx]
    is_st = row.get("is_st") if override_is_st is None else override_is_st
    f = {
        "is_st_at_event": bool(is_st) if not pd.isna(is_st) else False,
        "log_float_mv": float(np.log(float(row["float_mv"])))
        if not pd.isna(row.get("float_mv")) and float(row.get("float_mv") or 0) > 0 else np.nan,
        "log_total_mv": float(np.log(float(row["total_mv"])))
        if not pd.isna(row.get("total_mv")) and float(row.get("total_mv") or 0) > 0 else np.nan,
        "turnover20": float(pd.to_numeric(prev20.get("turnover"),
                                          errors="coerce").mean()) if len(prev20) >= 10 else np.nan,
        "vol20": float((pd.to_numeric(prev20["pct_chg"], errors="coerce") / 100.0).std())
        if len(prev20) >= 10 and "pct_chg" in prev20.columns else np.nan,
        "dist_high_250": float(row["close"] / float(prev250["high"].max()) - 1)
        if len(prev250) >= 60 else np.nan,
        "gap_return": gap_value,
        "volume_shock": volume_shock_value,
        "n_announcements_in_cluster": int(e["n_announcements_in_cluster"]),
        "sigma250": e.get("sigma250"),
    }
    kw = title_keyword_features(e["title"])
    for k in KEYWORD_MATERIAL_WEIGHT:
        f[f"kw_{k}"] = kw.get(k, 0.0)
    cat = e["category"]
    for c in TOP_CATEGORIES:
        f[f"cat_{c}"] = float(cat == c)
    return f


def load_domain() -> pd.DataFrame:
    ev = pd.read_parquet(ROOT / "out" / "event_cluster.parquet")
    lv = ev["label_valid"] if "label_valid" in ev.columns else pd.Series(True, index=ev.index)
    te = ev["training_eligible"] if "training_eligible" in ev.columns \
        else pd.Series(True, index=ev.index)
    ca = ev["corp_action_within_horizon"].fillna(False).astype(bool) \
        if "corp_action_within_horizon" in ev.columns else pd.Series(False, index=ev.index)
    cond = ((ev["tier"] == "P0") & (lv == True) & (te == True) &
            (ev["attribution_quality"] != "HEAVY") & (ev["category"] != "DISTRIBUTION") & (~ca))
    mat = ev[cond].sort_values("announce_date").drop_duplicates("event_cluster_id")
    return mat.dropna(subset=["materiality"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc-only", action="store_true", help="只跑 QC 小样（~800 事件）")
    ap.add_argument("--qc-n", type=int, default=800)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mat = load_domain()
    print(f"[audit] train domain n = {len(mat)}")
    if args.qc_only:
        # 分层抽样：按年 + 按 category 两层，保证覆盖
        mat["_y"] = mat["announce_date"].astype("int64") // 10000
        parts = []
        per_year = max(1, args.qc_n // mat["_y"].nunique())
        for _, g in mat.groupby("_y"):
            parts.append(g.sample(n=min(per_year, len(g)), random_state=42))
        mat = pd.concat(parts).drop(columns=["_y"]).reset_index(drop=True)
        print(f"[audit] QC sample n = {len(mat)}")

    # ---------- REAL：调共享 build_features（训练同款代码路径） ----------
    feat_real = build_features(mat, str(ROOT / "out"), with_labels=False)
    feat_real = feat_real.set_index("event_cluster_id")
    print(f"[audit] REAL features {len(feat_real)} 行 ({time.time()-t0:.0f}s)")

    # ---------- QC：row_features(REAL 口径) 必须与 build_features 逐位一致 ----------
    cache = ROOT / "out" / "daily_cache"
    qc_rows = []
    for code, g in mat.groupby("code", sort=False):
        p = cache / f"{code}.parquet"
        if not p.exists():
            continue
        daily = pd.read_parquet(p)
        for _, e in g.iterrows():
            f = row_features(daily, e, gap_value=e["gap_return"],
                             volume_shock_value=e["volume_shock"])
            if f is not None:
                qc_rows.append({"event_cluster_id": e["event_cluster_id"], **f})
    qc_df = pd.DataFrame(qc_rows).set_index("event_cluster_id")
    common = qc_df.index.intersection(feat_real.index)
    diff = (qc_df.loc[common, MODEL_FEATURES].astype(float)
            - feat_real.loc[common, MODEL_FEATURES].astype(float)).abs().max().max()
    print(f"[audit] QC parity row_features vs build_features: n={len(common)}, "
          f"max_abs_diff={diff:.3e} (必须为 0)")
    if diff != 0.0:
        raise SystemExit("QC FAIL: 复算特征与共享实现不一致，审计口径无效")

    # ---------- 合成口径（TCLOSE / MORNING） ----------
    model = get_model()
    syn_rows = []
    for code, g in mat.groupby("code", sort=False):
        p = cache / f"{code}.parquet"
        if not p.exists():
            continue
        daily = pd.read_parquet(p)
        daily["date"] = daily["date"].astype(int)
        for _, e in g.iterrows():
            dT = daily[daily["date"] <= int(e["announce_date"])]
            if dT.empty:
                continue
            f_tc = row_features(dT, e, gap_value=np.nan, volume_shock_value=np.nan)
            if f_tc is None:
                continue
            real_row = daily[daily["date"] == int(e["label_start"])]
            is_st_m = real_row.iloc[0].get("is_st") if len(real_row) else None
            f_am = row_features(dT, e, gap_value=e["gap_return"],
                                volume_shock_value=np.nan, override_is_st=is_st_m)
            # 描述性消融：仅替换收盘类（gap/volume 用真实值）——分解漂移来源
            f_cls = row_features(dT, e, gap_value=e["gap_return"],
                                 volume_shock_value=e["volume_shock"])
            syn_rows.append({"event_cluster_id": e["event_cluster_id"],
                             "category": e["category"],
                             "year": int(e["announce_date"]) // 10000,
                             "p_tc": None, "p_am": None,
                             **{f"tc_{k}": v for k, v in f_tc.items()},
                             **{f"am_{k}": v for k, v in (f_am or f_tc).items()},
                             **{f"cls_{k}": v for k, v in (f_cls or f_tc).items()}})
    syn = pd.DataFrame(syn_rows).set_index("event_cluster_id")
    print(f"[audit] synthetic features {len(syn)} 行 ({time.time()-t0:.0f}s)")

    idx = syn.index.intersection(feat_real.index)
    X_real = feat_real.loc[idx, MODEL_FEATURES]
    p_real = model.predict_proba(X_real)
    p_tc = model.predict_proba(syn.loc[idx, [f"tc_{c}" for c in MODEL_FEATURES]]
                               .rename(columns=lambda c: c[3:]))
    p_am = model.predict_proba(syn.loc[idx, [f"am_{c}" for c in MODEL_FEATURES]]
                               .rename(columns=lambda c: c[3:]))
    # 描述性消融（非闸门）：单特征缺失/替换对 p 的影响，分解 TCLOSE 漂移来源
    X_gap = X_real.copy(); X_gap["gap_return"] = np.nan
    X_vol = X_real.copy(); X_vol["volume_shock"] = np.nan
    p_gap = model.predict_proba(X_gap)
    p_vol = model.predict_proba(X_vol)
    p_cls = model.predict_proba(syn.loc[idx, [f"cls_{c}" for c in MODEL_FEATURES]]
                                .rename(columns=lambda c: c[4:]))
    res = pd.DataFrame({"p_real": p_real, "p_tclose": p_tc, "p_morning": p_am,
                        "p_only_gap_missing": p_gap, "p_only_vol_missing": p_vol,
                        "p_only_closeclass_proxy": p_cls,
                        "category": syn.loc[idx, "category"],
                        "year": syn.loc[idx, "year"]}, index=idx)
    print(f"[audit] predictions done ({time.time()-t0:.0f}s)")

    # ---------- 指标与闸门 ----------
    def metrics(a, b):
        d = np.abs(a - b)
        sp = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
        j = len(set(np.where(a >= 0.5)[0]) & set(np.where(b >= 0.5)[0])) / \
            max(1, len(set(np.where(a >= 0.5)[0]) | set(np.where(b >= 0.5)[0])))
        return {"spearman": sp, "mean_abs_dp": float(d.mean()),
                "p_dp_gt_0.15": float((d > 0.15).mean()), "jaccard_p5": float(j)}

    report = {"gates": GATES, "n": int(len(res)),
              "tclose": metrics(res.p_real.values, res.p_tclose.values),
              "morning": metrics(res.p_real.values, res.p_morning.values),
              "by_year": {}, "by_category_top6": {}}
    for y, g in res.groupby("year"):
        report["by_year"][int(y)] = {
            "tclose_spearman": float(pd.Series(g.p_real).corr(g.p_tclose, method="spearman")),
            "morning_spearman": float(pd.Series(g.p_real).corr(g.p_morning, method="spearman"))}
    top6 = res["category"].value_counts().head(6).index.tolist()
    for c in top6:
        g = res[res["category"] == c]
        report["by_category_top6"][c] = {
            "n": int(len(g)),
            "tclose_spearman": float(pd.Series(g.p_real).corr(g.p_tclose, method="spearman")),
            "morning_spearman": float(pd.Series(g.p_real).corr(g.p_morning, method="spearman"))}

    verdict = {}
    for variant in ("tclose", "morning"):
        m = report[variant]
        verdict[variant] = {
            "G1": m["spearman"] >= GATES["G1_spearman_min"],
            "G2": m["mean_abs_dp"] <= GATES["G2_mean_abs_dp_max"],
            "G3": m["p_dp_gt_0.15"] <= GATES["G3_p_abs_dp_gt_0.15_max"],
            "G4": m["jaccard_p5"] >= GATES["G4_jaccard_p5_min"],
        }
    report["verdict"] = verdict
    report["PRELIMINARY_APPROVED"] = all(all(v.values()) for v in verdict.values())
    # 描述性消融（事后分解，不参与闸门判定）
    report["ablation_descriptive"] = {
        "only_gap_missing_mean_abs_dp":
            float(np.abs(res.p_real.values - res.p_only_gap_missing.values).mean()),
        "only_volume_shock_missing_mean_abs_dp":
            float(np.abs(res.p_real.values - res.p_only_vol_missing.values).mean()),
        "only_closeclass_proxy_mean_abs_dp":
            float(np.abs(res.p_real.values - res.p_only_closeclass_proxy.values).mean()),
    }
    report["elapsed_s"] = round(time.time() - t0, 1)

    res.reset_index().to_parquet(OUT_DIR / "results.parquet", index=False)
    (OUT_DIR / "gates.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# PRELIMINARY 漂移审计报告",
             f"n = {report['n']} ｜ 预注册闸门: {json.dumps(GATES)}", "",
             "| 变体 | Spearman | mean|Δp| | P(|Δp|>0.15) | Jaccard(p≥0.5) | 闸门 |",
             "|---|---|---|---|---|---|"]
    for variant, name in (("tclose", "TCLOSE(盘后)"), ("morning", "MORNING(早盘)")):
        m = report[variant]
        v = verdict[variant]
        lines.append(f"| {name} | {m['spearman']:.4f} | {m['mean_abs_dp']:.4f} | "
                     f"{m['p_dp_gt_0.15']:.4f} | {m['jaccard_p5']:.4f} | "
                     f"{'PASS' if all(v.values()) else 'FAIL'} |")
    lines += ["", "## 按年 Spearman", "| 年 | TCLOSE | MORNING |", "|---|---|---|"]
    for y in sorted(report["by_year"]):
        r = report["by_year"][y]
        lines.append(f"| {y} | {r['tclose_spearman']:.4f} | {r['morning_spearman']:.4f} |")
    lines += ["", "## Top-6 事件族 Spearman", "| 族 | n | TCLOSE | MORNING |", "|---|---|---|---|"]
    for c, r in report["by_category_top6"].items():
        lines.append(f"| {c} | {r['n']} | {r['tclose_spearman']:.4f} | {r['morning_spearman']:.4f} |")
    lines += ["", "## 描述性消融（事后分解，非闸门）：单特征近似的 mean|Δp|",
              f"- 仅 gap_return→中位数: {report['ablation_descriptive']['only_gap_missing_mean_abs_dp']:.4f}",
              f"- 仅 volume_shock→中位数: {report['ablation_descriptive']['only_volume_shock_missing_mean_abs_dp']:.4f}",
              f"- 仅收盘类(T 收盘代理): {report['ablation_descriptive']['only_closeclass_proxy_mean_abs_dp']:.4f}",
              "", f"**PRELIMINARY_APPROVED = {report['PRELIMINARY_APPROVED']}** "
                   f"（全 PASS → 登记 DEPLOYMENT_DEVIATION 放行 PRELIMINARY 通道；"
                   f"任一 FAIL → 回退路线 A）"]
    (OUT_DIR / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[audit] DONE ({time.time()-t0:.0f}s) → {OUT_DIR}")
    print(json.dumps({"verdict": verdict, "PRELIMINARY_APPROVED": report["PRELIMINARY_APPROVED"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
