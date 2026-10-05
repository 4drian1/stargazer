"""c3_smoke_train.py — Corvus-C3 第一代历史信息影响模型（烟基线）。

链路：规则基线（关键词表） -> 正则化 LR -> LightGBM；purged 时间切分 + embargo；
3 seeds；PR-AUC / Brier / 校准曲线 + 分年稳定性。

闸门自检（开发路线 Corvus-C2 段）：
- 烟基线最低条件：≥500 个有效事件（label_status=OK、标签完备、实体明确）且覆盖 ≥150 独立交易日；
- 不满足 -> 只出描述统计，报告明确标 BELOW_GATE，不出模型结论。

标签口径（照抄 corvus + 本包 DEVIATIONS）：
- materiality：|ret_1/3/5d_pc| 任一则超过该股前 250 日收益 std 的 2 倍（初标签）；
- direction_3d：ret_3d_pc 符号（±0.1% 带宽内 FLAT），Secondary。

特征（全部 available_at 前可见；stockdb 日k + 标题关键词）：
- log_float_mv（事件日流通市值）、board（代码前缀板块代理）、turnover20 / vol20 /
  dist_high_250（距 250 日前高的回撤）、amount 分位、category one-hot、标题关键词命中。

用法：
  python c3_smoke_train.py --events out/event_cluster.parquet --out out [--seeds 3] [--embargo-days 5]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _stockdb import DEFAULT_BASE, load_daily, make_opener
from _taxonomy import (board_of, title_keyword_features, KEYWORD_MATERIAL_WEIGHT)
from corvus.materiality_v1.materiality_features_v1 import build_features, MODEL_FEATURES

SMOKE_MIN_EVENTS = 500
SMOKE_MIN_DAYS = 150
CATEGORY_TOP = 12


def log(msg: str):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------- 特征
# build_features 已移至 corvus/materiality_v1/materiality_features_v1.py（共享冻结实现）。
# 本文件只复用该实现；禁止在此另写特征逻辑，否则训练/线上出现 feature drift。


# ---------------------------------------------------------------- 切分
def purged_split(dates: pd.Series, train_frac: float, embargo_days: int,
                 cal: list[int]):
    """按交易日索引切：train 前段 / embargo 缺口 / test 后段（purged 时间切分）。"""
    pos = {d: i for i, d in enumerate(cal)}
    udates = sorted({int(d) for d in dates if int(d) in pos})
    cut = udates[int(len(udates) * train_frac)]
    cut_i = pos[cut]
    train_idx = [d for d in udates if pos[d] < cut_i - embargo_days]
    test_idx = [d for d in udates if pos[d] > cut_i + embargo_days]
    return set(train_idx), set(test_idx)


def load_calendar(out: str) -> list[int]:
    p = Path(out) / "calendar.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    # 退化：用事件 label_start 的并集当日历（精度足够用于 embargo）
    log("WARN: 无 calendar.json，embargo 用事件日并集近似")
    return []


# ---------------------------------------------------------------- 指标
def pr_auc(y_true, y_prob) -> float:
    from sklearn.metrics import average_precision_score
    if len(set(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def brier(y_true, y_prob) -> float:
    return float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))


def calibration_curve(y_true, y_prob, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        m = (y_prob > edges[i]) & (y_prob <= edges[i + 1]) if i else \
            (y_prob >= edges[i]) & (y_prob <= edges[i + 1])
        if m.sum() == 0:
            continue
        out.append({"bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}",
                    "n": int(m.sum()),
                    "mean_pred": float(y_prob[m].mean()),
                    "frac_pos": float(np.asarray(y_true)[m].mean())})
    return out


# ---------------------------------------------------------------- 模型
def rule_baseline_predict(feat_tr, feat_te):
    """关键词权重打分 -> 训练期选阈值 -> 测试期预测概率（仅基线，不作结论）。"""
    kws = [c for c in feat_tr.columns if c.startswith("kw_")]
    w = np.array([KEYWORD_MATERIAL_WEIGHT.get(c[3:], 1.0) for c in kws])
    s_tr = feat_tr[kws].values @ w
    s_te = feat_te[kws].values @ w
    # 阈值 = 训练期正类（materiality）得分中位数（最简单可复现规则）
    pos_med = float(np.median(s_tr[feat_tr["y"].values == 1])) \
        if (feat_tr["y"].values == 1).any() else float(np.median(s_tr))
    p_te = (s_te >= pos_med).astype(float)
    return p_te, {"threshold": pos_med, "weights": dict(zip(kws, w.tolist()))}


def fit_lr(X_tr, y_tr, X_te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_tr)
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    lr.fit(sc.transform(X_tr), y_tr)
    return lr.predict_proba(sc.transform(X_te))[:, 1], lr, sc


def fit_lgbm(X_tr, y_tr, X_te, seeds):
    import lightgbm as lgb
    probs = []
    models = []
    for s in seeds:
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                               min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                               class_weight="balanced", random_state=s, verbose=-1)
        m.fit(X_tr, y_tr)
        probs.append(m.predict_proba(X_te)[:, 1])
        models.append(m)
    return np.mean(probs, axis=0), models


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default="out/event_cluster.parquet")
    ap.add_argument("--out", default="out")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--embargo-days", type=int, default=5)
    ap.add_argument("--train-frac", type=float, default=0.7)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(args.events)
    log(f"{args.events}: {len(ev)} 行")

    # 训练集：P0 事件族 + OK 标签 + 非 HEAVY 重叠（corvus 第7节：HEAVY 排除）
    mat = ev[(ev["tier"] == "P0") & (ev["label_status"] == "OK") &
             (ev["attribution_quality"] != "HEAVY") &
             (ev["category"] != "DISTRIBUTION")].copy()
    # 事件簇级有效样本：同簇多公告不扩充样本量（corvus C2 口径）
    mat = mat.sort_values("announce_date").drop_duplicates("event_cluster_id")
    log(f"训练域 (P0/OK/非HEAVY): {len(mat)} 事件, "
        f"{mat['event_cluster_id'].nunique()} 簇, "
        f"{mat['label_start'].nunique()} 个交易日")

    gate_events = int(len(mat))
    gate_days = int(mat["label_start"].nunique())
    below_gate = gate_events < SMOKE_MIN_EVENTS or gate_days < SMOKE_MIN_DAYS

    desc = descriptive_stats(ev, mat)
    manifest = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "events_file": str(args.events),
        "train_domain": "tier=P0 & label_status=OK & attribution_quality!=HEAVY & category!=DISTRIBUTION",
        "gate": {"min_events": SMOKE_MIN_EVENTS, "min_days": SMOKE_MIN_DAYS,
                 "events": gate_events, "days": gate_days,
                 "status": "BELOW_GATE" if below_gate else "PASS"},
        "descriptive": desc,
    }

    report = ["# C3 烟基线报告（Corvus-C3 第一代历史信息影响模型）", ""]
    report.append(f"- 生成时间: {manifest['generated_at']}")
    report.append(f"- 事件文件: {args.events}")
    report.append(f"- 闸门状态: **{manifest['gate']['status']}** "
                  f"({gate_events}/{SMOKE_MIN_EVENTS} 事件, {gate_days}/{SMOKE_MIN_DAYS} 交易日)")
    report.append("")
    report.append("## 描述统计")
    for k, v in desc.items():
        report.append(f"- {k}: {v}")

    if below_gate:
        report += ["", "## 模型结论",
                   f"**未出模型结论**：有效事件 {gate_events} < {SMOKE_MIN_EVENTS} 或 "
                   f"覆盖交易日 {gate_days} < {SMOKE_MIN_DAYS}（Corvus-C2 烟基线闸门）。",
                   "按开发路线：继续回填，不强行训练下结论。以下为链路自检（pipeline smoke）。",
                   ""]
        log(f"BELOW_GATE: {gate_events} 事件 / {gate_days} 日，只出描述统计")

    if len(mat) >= 30:   # 链路自检最低样本（哪怕低于闸门也跑通管线）
        feat = build_features(mat, args.out, base=args.base)
        log(f"特征表: {len(feat)} 行 x {feat.shape[1]} 列")
        numeric_cols = MODEL_FEATURES
        data = feat.dropna(subset=["materiality"]).copy()
        data["y"] = data["materiality"].astype(int)
        cal = load_calendar(args.out)
        if cal:
            train_d, test_d = purged_split(data["announce_date"], args.train_frac,
                                           args.embargo_days, cal)
        else:
            udates = sorted(data["announce_date"].unique())
            cut = udates[int(len(udates) * args.train_frac)]
            train_d = {d for d in udates if d < cut}
            test_d = {d for d in udates if d > cut}
        tr = data[data["announce_date"].isin(train_d)]
        te = data[data["announce_date"].isin(test_d)]
        log(f"purged 切分: train {len(tr)} / test {len(te)} (embargo {args.embargo_days} 日)")

        results = {}
        if len(tr) >= 20 and len(te) >= 10 and tr["y"].nunique() == 2:
            X_tr = tr[numeric_cols].fillna(tr[numeric_cols].median()).values
            X_te = te[numeric_cols].fillna(tr[numeric_cols].median()).values
            y_tr, y_te = tr["y"].values, te["y"].values

            p_rule, rule_cfg = rule_baseline_predict(tr, te)
            p_lr, lr, sc = fit_lr(X_tr, y_tr, X_te)
            seeds = list(range(args.seeds))
            p_lgb, lgb_models = fit_lgbm(X_tr, y_tr, X_te, seeds)

            for name, p in (("rule", p_rule), ("lr", p_lr), ("lgbm", p_lgb)):
                results[name] = {
                    "pr_auc": pr_auc(y_te, p), "brier": brier(y_te, p),
                    "roc_auc": _safe_roc_auc(y_te, p),
                    "calibration": calibration_curve(y_te, np.clip(p, 0, 1)),
                }
            # direction secondary：LGBM 3 类
            results["direction_lgbm"] = direction_head(tr, te, numeric_cols, seeds)
            # 分年稳定性（全样本 materiality 先验，按公告年）
            years = sorted((data["announce_date"] // 10000).unique())
            results["stability_by_year"] = {
                str(y): {"n": int((data["announce_date"] // 10000 == y).sum()),
                         "prevalence": float(data.loc[data["announce_date"] // 10000 == y, "y"].mean())}
                for y in years}
            # 保存模型
            mdir = out / "models"
            mdir.mkdir(exist_ok=True)
            import joblib
            joblib.dump({"lr": lr, "scaler": sc, "numeric_cols": numeric_cols},
                        mdir / "lr_materiality.joblib")
            for i, m in enumerate(lgb_models):
                m.booster_.save_model(str(mdir / f"lgbm_materiality_s{i}.txt"))
            (mdir / "rule_weights.json").write_text(
                json.dumps(rule_cfg, ensure_ascii=False, indent=1), encoding="utf-8")
            manifest["models"] = [str(p) for p in sorted(mdir.iterdir())]
        else:
            log("样本不足以做 purged 切分评估（仅链路自检）")
            results = {"note": "insufficient sample for purged evaluation"}

        manifest["results"] = results
        report += report_results(results, below_gate)
    else:
        report += ["", "## 模型结论", "样本 < 30，无法做任何切分评估。"]
        manifest["results"] = {"note": "n<30"}

    (out / "c3_smoke_report.md").write_text("\n".join(report), encoding="utf-8")
    (out / "train_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log(f"报告: {out/'c3_smoke_report.md'}; manifest: {out/'train_manifest.json'}")


def _safe_roc_auc(y, p):
    from sklearn.metrics import roc_auc_score
    if len(set(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def direction_head(tr, te, numeric_cols, seeds):
    """Secondary：3 日方向 3 类（UP/FLAT/DOWN）。"""
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, f1_score
    yd_tr = tr["direction_3d"].map({"UP": 0, "FLAT": 1, "DOWN": 2}).values
    yd_te = te["direction_3d"].map({"UP": 0, "FLAT": 1, "DOWN": 2}).values
    if len(set(yd_tr)) < 2:
        return {"note": "direction classes < 2 in train"}
    med = tr[numeric_cols].median()
    m = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, min_child_samples=20,
                           random_state=seeds[0], verbose=-1)
    m.fit(tr[numeric_cols].fillna(med).values, yd_tr)
    pred = m.predict(te[numeric_cols].fillna(med).values)
    return {"accuracy": float(accuracy_score(yd_te, pred)),
            "f1_macro": float(f1_score(yd_te, pred, average="macro")),
            "class_counts": te["direction_3d"].value_counts().to_dict()}


def descriptive_stats(ev, mat):
    ok = ev[ev["label_status"] == "OK"]
    return {
        "announcements_total": int(len(ev)),
        "clusters_total": int(ev["event_cluster_id"].nunique()),
        "tier_counts": ev["tier"].value_counts().to_dict(),
        "label_status": ev["label_status"].value_counts(dropna=False).to_dict(),
        "attribution": ev["attribution_quality"].value_counts(dropna=False).to_dict(),
        "train_domain_events": int(len(mat)),
        "train_domain_days": int(mat["label_start"].nunique()),
        "materiality_rate_train_domain": _rate(mat, "materiality"),
        "direction_dist_train_domain": mat["direction_3d"].value_counts(dropna=False).to_dict(),
        "ret_1d_pc_mean": _f(ok["ret_1d_pc"].mean()),
        "ret_3d_pc_mean": _f(ok["ret_3d_pc"].mean()),
        "ret_5d_pc_mean": _f(ok["ret_5d_pc"].mean()),
        "by_year_train_domain": mat.assign(
            y=mat["announce_date"].astype(str).str[:4])["y"].value_counts().sort_index().to_dict(),
    }


def _rate(df, col):
    s = df[col].dropna()
    return float(s.mean()) if len(s) else None


def _f(x):
    return None if pd.isna(x) else float(x)


def report_results(results, below_gate):
    lines = ["", "## 模型结果（链路自检）" if below_gate else "", ""]
    if "note" in results:
        lines.append(f"_{results['note']}_")
        return lines
    head = "| 模型 | PR-AUC | ROC-AUC | Brier |"
    lines += [head, "|---|---|---|---|"]
    for name in ("rule", "lr", "lgbm"):
        r = results.get(name, {})
        lines.append(f"| {name} | {r.get('pr_auc', float('nan')):.4f} | "
                     f"{r.get('roc_auc', float('nan')):.4f} | {r.get('brier', float('nan')):.4f} |")
    d = results.get("direction_lgbm", {})
    if "accuracy" in d:
        lines += ["", f"direction(3d) LGBM: acc={d['accuracy']:.3f} f1_macro={d['f1_macro']:.3f} "
                  f"counts={d['class_counts']}"]
    lines += ["", "### 分年稳定性（全样本先验）",
              "| 年 | n | materiality 率 |", "|---|---|---|"]
    for y, r in results.get("stability_by_year", {}).items():
        lines.append(f"| {y} | {r['n']} | {r['prevalence']:.3f} |")
    lines += ["", "### 校准曲线（LGBM, 10 bin）",
              "| bin | n | mean_pred | frac_pos |", "|---|---|---|---|"]
    for b in results.get("lgbm", {}).get("calibration", []):
        lines.append(f"| {b['bin']} | {b['n']} | {b['mean_pred']:.3f} | {b['frac_pos']:.3f} |")
    if below_gate:
        lines += ["", "> 以上数值仅证明链路可运行；样本未过闸门，不构成任何模型结论。"]
    return lines


if __name__ == "__main__":
    main()
