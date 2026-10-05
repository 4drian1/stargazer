"""walk_forward.py — C3 严格时间序列验证（Step 2）。

Expanding walk-forward（事件簇/年不跨 train/test；年份边界 embargo）：
  Train 2019-2021 -> Test 2022
  Train 2019-2022 -> Test 2023
  Train 2019-2023 -> Test 2024
  Train 2019-2024 -> Test 2025

每折比较 rule / LR / LGBM，报告：
  ROC-AUC, PR-AUC, Brier, LogLoss, base_rate,
  PR-AUC - base_rate（lift）, calibration slope/intercept, ECE,
  score decile -> realized materiality（每 bin 报 n）。

注意：每折重新 fit（不用随机切分的全局模型）。
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _stockdb import DEFAULT_BASE
from c3_smoke_train import (rule_baseline_predict, fit_lr,
                            fit_lgbm, pr_auc, brier, calibration_curve,
                            _safe_roc_auc, load_calendar)
from corvus.materiality_v1.materiality_features_v1 import build_features, MODEL_FEATURES


def numeric_cols_of(feat: pd.DataFrame) -> list[str]:
    return [c for c in feat.columns
            if c.startswith(("kw_", "cat_", "log_", "turnover", "vol",
                             "dist_", "gap_", "volume_", "n_ann", "sigma"))
            or c in ("is_st_at_event",)]


def ece(y_true, y_prob, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    tot = 0.0
    for i in range(n_bins):
        m = (y_prob > edges[i]) & (y_prob <= edges[i + 1]) if i else \
            (y_prob >= edges[i]) & (y_prob <= edges[i + 1])
        n = int(m.sum())
        if n == 0:
            continue
        tot += n * abs(float(np.asarray(y_prob)[m].mean()) - float(np.asarray(y_true)[m].mean()))
    return float(tot / len(y_true))


def cal_slope_intercept(y_true, y_prob):
    from sklearn.linear_model import LinearRegression
    x = y_prob.reshape(-1, 1)
    lr = LinearRegression().fit(x, y_true)
    return float(lr.coef_[0]), float(lr.intercept_)


def decile_table(y_true, y_prob, n=10):
    order = np.argsort(y_prob)
    y = np.asarray(y_true)[order]
    p = y_prob[order]
    n_per = max(1, len(y) // n)
    rows = []
    for i in range(n):
        s = i * n_per
        e = (i + 1) * n_per if i < n - 1 else len(y)
        if e <= s:
            continue
        rows.append({"decile": i + 1,
                     "n": int(e - s),
                     "mean_pred": float(p[s:e].mean()),
                     "realized_mat": float(y[s:e].mean())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="out/event_cluster.parquet")
    ap.add_argument("--out", default="out")
    ap.add_argument("--tag", default="")
    ap.add_argument("--embargo-days", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--exclude-corp-action", action="store_true",
                   help="排除 corp_action_within_horizon 的样本(除权守卫 T..T+5 修复生效)")
    ap.add_argument("--exclude-corp-start-day", action="store_true",
                   help="仅排除 label_start 当天除权的样本")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""

    ev = pd.read_parquet(args.events)
    # 容错：旧版 events 无相关列时视为全部有效/无除权
    lv = ev["label_valid"] if "label_valid" in ev.columns else pd.Series(
        True, index=ev.index)
    ca_raw = ev["corp_action_within_horizon"] if "corp_action_within_horizon" in ev.columns else pd.Series(False, index=ev.index)
    cas_raw = ev["corp_action_on_label_start"] if "corp_action_on_label_start" in ev.columns else pd.Series(False, index=ev.index)
    # 该列读为 object 时 ~ 会变成按位取反(恒真)，必须显式转 bool 再取反
    ca = ca_raw.fillna(False).astype(bool)
    cas = cas_raw.fillna(False).astype(bool)
    cond = ((ev["tier"] == "P0") & (ev["label_status"] == "OK") &
            (ev["attribution_quality"] != "HEAVY") &
            (ev["category"] != "DISTRIBUTION") & (lv == True))
    if args.exclude_corp_action:
        cond = cond & (~ca.fillna(False))
    if args.exclude_corp_start_day:
        cond = cond & (~cas.fillna(False))
    mat = ev[cond].copy()
    mat = mat.sort_values("announce_date").drop_duplicates("event_cluster_id")

    # build_features 已合并 materiality / announce_date / event_cluster_id，切勿重复 merge
    feat = build_features(mat, args.out, base=DEFAULT_BASE)
    data = feat.dropna(subset=["materiality"]).copy()
    data["y"] = data["materiality"].astype(int)
    data["year"] = (data["announce_date"].astype("int64") // 10000).astype(int)

    nc = MODEL_FEATURES
    cal = load_calendar(args.out)
    pos = {d: i for i, d in enumerate(cal)}

    folds = [(2019, 2020, 2021, 2022),
             (2019, 2020, 2021, 2022, 2023),
             (2019, 2020, 2021, 2022, 2023, 2024),
             (2019, 2020, 2021, 2022, 2023, 2024, 2025)]
    test_years = [2022, 2023, 2024, 2025]

    all_results = {}
    report = ["# C3 Walk-Forward OOT 验证（Step 2）", ""]
    report.append(f"- 训练域: {len(mat)} 事件; embargo={args.embargo_days} 交易日")
    report.append("- 口径: 每折重新 fit rule/LR/LGBM；事件簇/年不跨集；年份边界 embargo")
    report.append("")

    for fold_i, (test_year,) in enumerate(zip(test_years)):
        train_years = folds[fold_i][:-1]
        # embargo：以 test_year-01-01 在 cal 的位置为界
        test_start = test_year * 10000 + 101
        if test_start not in pos:
            cand = [d for d in cal if d >= test_start]
            test_start = cand[0] if cand else test_start
        p0 = pos[test_start]
        tr_cut = cal[max(0, p0 - args.embargo_days)]
        te_cut = cal[min(len(cal) - 1, p0 + args.embargo_days)]
        tr = data[(data["year"].isin(train_years)) & (data["announce_date"] <= int(tr_cut))]
        te = data[(data["year"] == test_year) & (data["announce_date"] >= int(te_cut))]
        if len(tr) < 50 or len(te) < 10 or tr["y"].nunique() < 2:
            report.append(f"## Fold test={test_year}: 样本不足跳过 (tr={len(tr)} te={len(te)})")
            continue

        med = tr[nc].median()
        X_tr = tr[nc].fillna(med).values
        X_te = te[nc].fillna(med).values
        y_tr, y_te = tr["y"].values, te["y"].values

        p_rule, _ = rule_baseline_predict(tr, te)
        p_lr, _, _ = fit_lr(X_tr, y_tr, X_te)
        p_lgb, _ = fit_lgbm(X_tr, y_tr, X_te, list(range(args.seeds)))

        fold_res = {}
        for name, p in (("rule", p_rule), ("lr", p_lr), ("lgbm", p_lgb)):
            from sklearn.metrics import log_loss
            base = float(np.mean(y_te))
            pr = pr_auc(y_te, p)
            roc = _safe_roc_auc(y_te, p)
            br = brier(y_te, p)
            try:
                ll = float(log_loss(y_te, np.clip(p, 1e-6, 1 - 1e-6)))
            except Exception:
                ll = float("nan")
            slope, intercept = cal_slope_intercept(y_te, np.clip(p, 0, 1))
            fold_res[name] = {
                "n_test": int(len(te)), "base_rate": base,
                "roc_auc": roc, "pr_auc": pr,
                "pr_auc_minus_base": (pr - base) if not (isinstance(pr, float) and np.isnan(pr)) else None,
                "brier": br, "log_loss": ll,
                "cal_slope": slope, "cal_intercept": intercept,
                "ece": ece(y_te, np.clip(p, 0, 1)),
                "decile": decile_table(y_te, np.clip(p, 0, 1)),
            }
        all_results[f"test_{test_year}"] = fold_res

        report.append(f"## Fold: Train {train_years} -> Test {test_year}")
        report.append(f"- n_test={len(te)} base_rate={fold_res['lgbm']['base_rate']:.3f}")
        report.append("")
        report.append("| 模型 | ROC | PR-AUC | PR-base | Brier | LogLoss | cal_slope | ECE |")
        report.append("|---|---|---|---|---|---|---|---|")
        for name in ("rule", "lr", "lgbm"):
            r = fold_res[name]
            report.append(f"| {name} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | "
                          f"{r['pr_auc_minus_base'] if r['pr_auc_minus_base'] is not None else 'NA'} | "
                          f"{r['brier']:.3f} | {r['log_loss']:.3f} | {r['cal_slope']:.3f} | {r['ece']:.3f} |")
        report.append("")
        report.append(f"### LGBM 分位 -> 实际 materiality (test {test_year})")
        report.append("| decile | n | mean_pred | realized_mat |")
        report.append("|---|---|---|---|")
        for d in fold_res["lgbm"]["decile"]:
            report.append(f"| {d['decile']} | {d['n']} | {d['mean_pred']:.3f} | {d['realized_mat']:.3f} |")
        report.append("")

    # 汇总 OOT（四折 test 合并由报告中的各折 + LGBM 分位表体现）
    meta = {
        "tag": args.tag,
        "events_file": args.events,
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }
    if args.tag == "pre_fix":
        meta["STATUS"] = "INVALID_FOR_FINAL_MODEL_SELECTION"
        meta["USE"] = "DIAGNOSTIC_ONLY"
        meta["KNOWN_LABEL_BUG"] = "CORP_ACTION_START_DAY"
    elif args.tag == "post_fix":
        meta["STATUS"] = "VALID_CANDIDATE"
        meta["USE"] = "FINAL_MODEL_SELECTION_CANDIDATE"
        meta["corp_exclusion"] = (
            "corp_action_within_horizon (T..T+5) EXCLUDED from train+test; "
            "label_valid filter is a no-op (backfill never sets it False)")
        meta["comparison_note"] = (
            "PRE_FIX (pre_fix tag) = no corp exclusion = buggy baseline; "
            "this POST_FIX excludes all corp-window samples; "
            "delta between them = pure corporate-action-window exclusion effect (~0).")
    all_results["_meta"] = meta
    out_path = os.path.join(args.out, f"walk_forward{tag}_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    with open(os.path.join(args.out, f"walk_forward{tag}_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=1, default=str)
    print(f"[walk_forward] 已写 {out_path} 与 walk_forward{tag}_results.json")
    for ty, fr in all_results.items():
        if ty == "_meta":
            continue
        l = fr.get("lgbm", {})
        print(f"  {ty}: n={l.get('n_test')} base={l.get('base_rate'):.3f} "
              f"ROC={l.get('roc_auc'):.3f} PR={l.get('pr_auc'):.3f} "
              f"PR-base={l.get('pr_auc_minus_base')}")


if __name__ == "__main__":
    main()
