# -*- coding: utf-8 -*-
"""2026时间复核第5步：冻结 M0/T1 在 2026-01~08 的时间复核（§18.12）。

评审裁决（路线文档 v2 第1步）：冻结 M0/T1，不改任何特征/阈值/超参，直接在
2026-01~08 复核。性质：**可否定、不可确认**——2026前8个月已被项目观察过，
不是纯锁箱；LEGACY 改善若不重复即停止投入该方向，即便重复也需 2026-09 后
前瞻确认。

冻结口径（与 T1 实验 fold1 完全一致）：
- 特征：M0 = minute_four_way.FEATS 12个；T1 = M0 + 7个 pf_*；
- 训练集 = 2025 样本中 date<=20250831（与 fold1 相同）；
- 模型 = LGBMClassifier 三分类，参数与 ablation_open15.run_fold 相同；
- EV = 训练期各类 net_ret 均值加权；Top30% 阈值 = 测试期 EV 的 0.7 分位
  （与 fold1 相同的规则）；
- 样本过滤：dp=945、entry_type=vwap、price_scale_valid、censored=0、
  net_ret_t2.0 非空。

输出：按 src(A2/B0/LEGACY) × 月(202601~202608) 报告 n / 全体净收益 / Top30%
净收益 / Top30%STOP / Brier / EV-收益Spearman；总表 + LEGACY 聚焦。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
sys.path.insert(0, OUT_DIR)
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
KT = "2.0"
M0 = ["downside_atr", "shock_atr", "recovery", "lower_wick_atr",
      "vwap_reclaim_atr", "kill_speed_atr", "first15_low_atr",
      "first15_range_atr", "vol_ratio15", "new_low_after_945",
      "dip_hold", "bars"]
T1_ADD = ["pf_ret1m", "pf_ret3m", "pf_ret5m", "pf_decline_speed",
          "pf_decel", "pf_low2_minus_low1", "pf_has_two_lows"]
TRAIN_MAX = 20250831


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_pool(samples_path, pf_path):
    df = pd.read_parquet(samples_path)
    df = df[(df["dp"] == 945) & (df["entry_type"] == "vwap")].copy()
    df["nr"] = df[f"net_ret_t{KT}"]
    df["y"] = df[f"outcome_t{KT}"]
    if "price_scale_valid" not in df.columns:
        df["price_scale_valid"] = True
    df = df[df["price_scale_valid"].astype(bool) & (df["censored"] == 0)
            & df["nr"].notna()].copy()
    pf = pd.read_parquet(pf_path)
    df = df.merge(pf, on=["code", "dp_date", "dp"], how="left")
    return df


def brier(y_true, proba, classes):
    idx = {c: i for i, c in enumerate(classes)}
    oh = np.zeros_like(proba)
    for r, y in enumerate(y_true):
        oh[r, idx[y]] = 1.0
    return float(((proba - oh) ** 2).sum(axis=1).mean())


def fit_frozen(tr, feats):
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, num_leaves=31,
                             learning_rate=0.05, min_child_samples=100,
                             subsample=0.9, colsample_bytree=0.9, seed=7,
                             n_estimators=400, verbosity=-1, n_jobs=8)
    clf.fit(tr[feats], tr["y"])
    ev_coef = {cl: float(tr.loc[tr["y"] == cl, "nr"].mean())
               for cl in clf.classes_}
    return clf, ev_coef


def evaluate(clf, ev_coef, te, feats):
    proba = clf.predict_proba(te[feats])
    ev = np.zeros(len(te))
    for i, cl in enumerate(clf.classes_):
        ev += proba[:, i] * ev_coef[cl]
    te = te.copy()
    te["EV"] = ev
    thr = te["EV"].quantile(0.7)
    top = te[te["EV"] >= thr]
    oc_t, oc_a = top[f"outcome_t{KT}"], te[f"outcome_t{KT}"]
    spear = te[["EV", "nr"]].corr(method="spearman").iloc[0, 1] \
        if len(te) > 10 else np.nan
    return {"n": len(te), "all_mean": float(te["nr"].mean()),
            "all_stop": float((oc_a == "STOP").mean()),
            "top_mean": float(top["nr"].mean()),
            "top_stop": float((oc_t == "STOP").mean()),
            "top_n": len(top),
            "brier": brier(te["y"].values, proba, clf.classes_),
            "spearman": float(spear)}


def block(te, keys):
    """按 keys 分组（None=全体）返回 {标签: 子集}。"""
    if keys is None:
        return {"ALL": te}
    out = {}
    for k, g in te.groupby(keys):
        if len(g) >= 30:
            out[str(k)] = g
    return out


def main():
    t0 = time.time()
    tr25 = load_pool(os.path.join(OUT_DIR, "minute_samples_2025.parquet"),
                     os.path.join(OUT_DIR, "minute_path_features_2025.parquet"))
    tr = tr25[tr25["date"] <= TRAIN_MAX]
    log(f"训练池(冻结=25H1): {len(tr):,}（{tr['src'].value_counts().to_dict()}）")
    te26 = load_pool(os.path.join("minute_labels", "minute_samples_2026.parquet"),
                     os.path.join("minute_features", "minute_path_features_2026.parquet"))
    te26 = te26[(te26["date"] >= 20260101) & (te26["date"] <= 20260831)]
    log(f"2026复核池: {len(te26):,}（{te26['src'].value_counts().to_dict()}）")

    models = {}
    for ver, feats in (("M0", M0), ("T1", M0 + T1_ADD)):
        clf, coef = fit_frozen(tr, feats)
        models[ver] = (clf, coef, feats)
        log(f"{ver} trained")

    rows = []
    for scope_keys, scope_name in ((None, "ALL"), ("src", "src"),
                                   (["date"], "month")):
        if scope_keys == ["date"]:
            te26["__m"] = te26["date"] // 100
            groups = block(te26, "__m")
        else:
            groups = block(te26, scope_keys)
        for gname, g in groups.items():
            for ver, (clf, coef, feats) in models.items():
                r = evaluate(clf, coef, g, feats)
                rows.append({"scope": scope_name, "grp": gname, "model": ver,
                             **r})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "review_2026_frozen.csv"), index=False)

    # ---- 控制台要点 ----
    for ver in ("M0", "T1"):
        a = res[(res.scope == "ALL") & (res.model == ver)].iloc[0]
        log(f"2026 ALL {ver}: 全体{a['all_mean']:+.3%}/STOP{a['all_stop']:.1%} "
            f"Top30% {a['top_mean']:+.3%}/STOP{a['top_stop']:.1%} "
            f"Brier={a['brier']:.4f} Spear={a['spearman']:+.3f}")
    for src in ("A2", "B0", "LEGACY"):
        s = res[(res.scope == "src") & (res.grp == src)]
        if not len(s):
            continue
        for ver in ("M0", "T1"):
            r = s[s.model == ver].iloc[0]
            log(f"2026 {src} {ver}: Top30% {r['top_mean']:+.3%}/STOP"
                f"{r['top_stop']:.1%} (n={r['n']:,})")

    # ---- 报告 §18.12 ----
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.13 ")
    if idx != -1:
        old = old[:idx] + "\n"

    def fmt_rows(mask):
        L = []
        for _, r in res[mask].iterrows():
            L.append(f"| {r['grp']} | {r['model']} | {r['n']:,} | "
                     f"{r['all_mean']:+.2%} | {r['all_stop']:.1%} | "
                     f"{r['top_mean']:+.2%} | {r['top_stop']:.1%} | "
                     f"{r['brier']:.4f} | {r['spearman']:+.3f} |")
        return L

    a0 = res[(res.scope == "ALL") & (res.model == "M0")].iloc[0]
    a1 = res[(res.scope == "ALL") & (res.model == "T1")].iloc[0]
    leg = res[(res.scope == "src") & (res.grp == "LEGACY")]
    leg_txt = ("；".join(f"{r.model}: Top30% {r.top_mean:+.2%}/STOP "
                         f"{r.top_stop:.1%}"
                         for r in leg.itertuples()) if len(leg) else "无")
    L = ["\n### 18.13 冻结 M0/T1 的 2026-01~08 时间复核（正确 volume/100 修复后重算）\n",
         "数据基础：分钟数据取自 `minute_cache_clean/`（raw 原始响应只读保存于 "
         "`minute_cache_raw/`；bar 级 r=amount/(volume·close)<0.5 判 100 倍污染并 "
         "volume/100，保存 volume_scale_factor/volume_repaired；无法确定量纲或验收"
         "不合格的 (code,day) 共 12,549 个已隔离于 quarantine_2026.csv，标签引擎与"
         "路径特征自动剔除）。本节取代 §18.12（错误 amount/close 伪修复，作废）。\n",
         "性质（评审裁决）：**可否定、不可确认**。2026前8个月已被项目观察过，"
         "不是纯锁箱；LEGACY 改善若不重复即停止投入，即便重复也需 2026-09 后前瞻确认。\n",
         "冻结口径：特征（M0=12分钟特征；T1=M0+7个pf_*）、训练集（=25H1，与T1实验"
         "fold1完全相同）、LGB参数、EV构造（训练期类均值加权）、Top30%阈值"
         "（测试期EV的0.7分位）全部沿用T1实验，零改动。\n",
         f"- 2026复核池：{len(te26):,} 样本（{te26['src'].value_counts().to_dict()}），"
         f"过滤 dp945/vwap/除权有效/非censored；8月末 censored 事件已剔除",
         f"- 全体：M0 净收益 {a0['all_mean']:+.2%}/STOP {a0['all_stop']:.1%} vs "
         f"T1 {a1['all_mean']:+.2%}/STOP {a1['all_stop']:.1%}",
         f"- Top30%：M0 {a0['top_mean']:+.2%}/STOP {a0['top_stop']:.1%} vs "
         f"T1 {a1['top_mean']:+.2%}/STOP {a1['top_stop']:.1%}",
         f"- LEGACY 聚焦：{leg_txt}",
         "\n**全体（2026-01~08）**：\n",
         "| 范围 | 模型 | n | 全体净收益 | 全体STOP | Top30%净收益 | "
         "Top30%STOP | Brier | Spearman |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|",
         *fmt_rows(res.scope == "ALL"),
         "\n**分来源**：\n",
         "| 范围 | 模型 | n | 全体净收益 | 全体STOP | Top30%净收益 | "
         "Top30%STOP | Brier | Spearman |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|",
         *fmt_rows(res.scope == "src"),
         "\n**分月**：\n",
         "| 月份 | 模型 | n | 全体净收益 | 全体STOP | Top30%净收益 | "
         "Top30%STOP | Brier | Spearman |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|",
         *fmt_rows(res.scope == "month"),
         f"\n明细：review_2026_frozen.csv",
         "\n判读（与 §18.12 错误修复版对照，结论一致）：",
         "\n- 2026前8个月低吸候选池整体转负（全体约 −0.33%/笔，2025为 +0.15~0.22%），"
         "两模型Top30%均为负，市场环境不利。",
         "\n- **T1改善未干净复现**：ALL +0.05pp、A2 +0.04pp 均为微差（可忽略）；"
         "B0 反转为更差（−0.574% vs −0.343%）；Brier 全面略差。",
         "\n- 唯一亮点 LEGACY +0.28pp（−0.163% vs −0.440%）方向重复，但收益仍为负、"
         "STOP 边际变差（29.8% vs 29.1%）。错误修复版该差为 +0.27pp，方向与量级"
         "几乎不变——因 1 分钟 bar 的 VWAP 本就接近收盘价，价格类特征两版几乎相同；"
         "但量价结构（缩量回踩、下跌/反弹量比）只有本版可信。",
         "\n- 裁决（按“可否定不可确认”规则）：**T1 维持观察层、不晋级**；LEGACY 假设"
         "方向重复但强度不足以升级，等 2026-09 后前瞻数据裁决。T1 继续只作盯盘解释"
         "（“下杀速度减缓/仍在加速”），不入交易决策。",
         "\n判读边界：本节只能否定（T1/LEGACY改善不重复→停止投入），不能当样本外"
         "确认；真正前瞻验证从2026-09之后影子运行开始。"]
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
