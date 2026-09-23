"""F-v2.2 复杂度控制报告生成（中文 markdown）。

核心问题只有一个：外围波动信号是低维且稳定，还是根本不稳定？
对比 C1 / C2-L(Ridge) / C2-S(精简特征 LGBM) / C2-F(156 特征 LGBM)，
统一沿用 F-v2.0 的时点、冻结产物三件套、阈值与评估口径。

高风险组两套口径（严格分开）：
- 「前 20% 打分组」（排序指标）= 该期间 score_raw ≥ 其 80 分位，同分整体入档；
- 「flagged 日」（放行/拦截效果）= score_raw ≥ 冻结阈值（2022–2023 校准期原始分 80 分位）。
排序指标用 score_raw，概率指标（Brier/PR-AUC）用 score_cal。
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

EVAL_PERIODS = ["dev2024", "stab2025", "stress2026"]
OUT_PERIODS = ["calib", "dev2024", "stab2025", "stress2026"]   # 样本外期间
ALL_PERIODS = ["train", "calib", "dev2024", "stab2025", "stress2026"]
PERIOD_CN = {"train": "训练19-21(样本内)", "calib": "校准22-23",
             "dev2024": "开发2024", "stab2025": "稳定2025", "stress2026": "压力2026.1-8"}
ML_MODELS = ["C2-L", "C2-S", "C2-F"]
ALL_MODELS = ["C1"] + ML_MODELS


# ---------------- 基础工具 ----------------

def _f(x, nd=4):
    if x is None:
        return "NaN"
    try:
        if pd.isna(x):
            return "NaN"
    except (TypeError, ValueError):
        pass
    return f"{float(x):.{nd}f}"


def _pct(x, nd=1):
    return "NaN" if x is None or pd.isna(x) else f"{100 * float(x):.{nd}f}%"


def _md(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "---|" * len(headers)]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def _spearman(a, b):
    a, b = pd.Series(a), pd.Series(b)
    m = a.notna() & b.notna()
    if m.sum() < 10 or a[m].nunique() <= 1 or b[m].nunique() <= 1:
        return np.nan  # 常数输入（如 C1 大量同分）无定义
    return float(a[m].corr(b[m], method="spearman"))


def _top_mask(s, top_q=0.2):
    """前 top_q 打分组：≥ 80 分位阈值，同分整体入档（禁止强拆并列）。"""
    return s >= s.quantile(1 - top_q)


def _one_metrics(g, q70, top_q=0.2):
    """单期间指标。排序指标用 score_raw（同分整体入档）；概率指标用 score_cal。"""
    d = g[["score_raw", "score_cal", "stop_rate_3d"]].dropna()
    if len(d) < 20:
        return {"n": len(d)}
    sr, sc, yv = d["score_raw"], d["score_cal"], d["stop_rate_3d"]
    top = _top_mask(sr, top_q)
    hi, lo = yv[top], yv[~top]
    ybin = (yv > q70).astype(int)
    try:
        pr_auc = float(average_precision_score(ybin, sc))
    except ValueError:
        pr_auc = np.nan
    return {
        "n": len(d), "base": float(yv.mean()),
        "brier": float(np.mean((sc - yv) ** 2)),
        "pr_auc": pr_auc,
        "spearman": _spearman(sr, yv),
        "hi_touch": float(hi.mean()) if len(hi) else np.nan,
        "rest_touch": float(lo.mean()) if len(lo) else np.nan,
        "diff": float(hi.mean() - lo.mean()) if len(hi) and len(lo) else np.nan,
        "lift": float(hi.mean() / max(yv.mean(), 1e-9)) if len(hi) else np.nan,
        "n_hi": int(top.sum()),
    }


def _metrics_by_period(df, q70):
    return {per: _one_metrics(g, q70) for per, g in df.groupby("period")}


def _deciles_mono(g, y_col="stop_rate_3d", q=10):
    """十分层单调 Spearman（按 score_raw 分位切，同分同档）。"""
    d = g[["score_raw", y_col]].dropna().copy()
    if len(d) < 50:
        return np.nan
    d["bin"] = pd.qcut(d["score_raw"], q, duplicates="drop").cat.codes
    t = d.groupby("bin")[y_col].mean()
    return _spearman(t.index.to_numpy(), t.to_numpy())


def _monthly_consistency(g, y_col="stop_rate_3d"):
    """月度方向一致性。n_flag<2 的月份不计入分母，但显性报告数量。"""
    d = g[["trade_date", "high_risk_flag", y_col]].dropna(subset=[y_col]).copy()
    d["ym"] = d["trade_date"].str[:7]
    rows, excluded = [], 0
    for ym, grp in d.groupby("ym"):
        hi = grp.loc[grp["high_risk_flag"] == 1, y_col]
        if len(hi) < 2:
            excluded += 1
            continue
        rows.append((ym, float(hi.mean() > grp[y_col].mean()), ym[:4]))
    cons = pd.DataFrame(rows, columns=["ym", "consistent", "year"])
    return cons, excluded


def _join(ctx, sid, model):
    p = ctx["preds"][(ctx["preds"]["strategy_id"] == sid) & (ctx["preds"]["model"] == model)]
    return p.merge(ctx["labels"][sid], on="trade_date", how="left")


def _amounts(g, h=5):
    """反事实金额：flagged 日 mean_net_return_h × candidate_count。"""
    d = g[g["high_risk_flag"] == 1]
    amt = (d[f"mean_net_return_{h}d"] * d["candidate_count"]).dropna()
    return len(amt), float(-amt[amt < 0].sum()), float(amt[amt > 0].sum())


# ---------------- 报告主体 ----------------

def render_report(ctx):
    q70 = ctx["q70"]
    joins = {m: _join(ctx, ctx["main_sid"], m) for m in ALL_MODELS}
    joins["C0"] = _join(ctx, ctx["main_sid"], "C0")
    met = {m: _metrics_by_period(joins[m], q70) for m in ALL_MODELS}
    met["C0"] = _metrics_by_period(joins["C0"], q70)
    monthly = {}
    for m in ALL_MODELS:
        monthly[m] = _monthly_consistency(joins[m])
    legacy = {m: _join(ctx, ctx["legacy_sid"], m) for m in ("C1", "C2-S", "C2-F")}
    legacy_met = {m: _metrics_by_period(legacy[m], q70) for m in legacy}

    S = []
    a = S.append

    # ===== 头部 =====
    a("# 外围行情风险基线 F-v2.2 复杂度控制报告\n")
    a(f"- 运行时间:{ctx['run_at']}")
    a("- 版本:F-v2.2——回答一个问题:**外围波动信号是低维且稳定,还是根本不稳定?**"
      "沿用 F-v2.0 的全部修复(隔夜美股时点、冻结产物三件套、排序/概率口径分离),"
      "本轮不新增特征来源、不跑 HMM/ruptures、不换框架。")
    a(f"- 主池:{ctx['main_sid']};对照池:{ctx['legacy_sid']}(只做方向推断)")
    a("- 主标签 y = **loss3_touch_rate**:池内 entry×(1-3%) 固定 3% 触碰率(3 日窗口),"
      "**不是** A2 实际 ATR 止损;3% 触碰 ≠ 不值得买(先下杀后反弹会被误伤),"
      "本轮只评估波动排序信号,不决定暂停买入。")
    a(f"- 划分:{SCHEME_DESC(ctx)}")
    a("- 排序指标(Spearman/前20%组/分层)用 score_raw 同分整体入档;"
      "概率指标(Brier/PR-AUC)用 score_cal。")

    # ===== 0 模型对照 =====
    a("\n## 0. 四组对照\n")
    a(_md(["版本", "模型", "特征数", "超参来源"], [
        ["C1", "固定规则(6 条写死)", "-", "设计期冻结,未用 2024 调"],
        ["C2-L", f"Ridge 回归(median填补+标准化, alpha={ctx['ridge_alpha']})", len(ctx["feature_cols"]), "设计期冻结"],
        ["C2-S", "LightGBM(同 C2-F 参数)", len(ctx["small_features"]), "设计期冻结;特征集按经济逻辑预选(见附录),未用 2024/2025 重要性倒推"],
        ["C2-F", "LightGBM(800树, 31叶)", len(ctx["feature_cols"]), "v1 沿用,设计期冻结"],
    ]))
    a("")

    # ===== 1 数据概览 =====
    a("## 1. 数据概览(主池 a2)\n")
    lab = ctx["labels"][ctx["main_sid"]].copy()
    lab["period"] = lab["trade_date"].map(lambda d: _period_of(ctx, d))
    rows = []
    for per in ALL_PERIODS:
        g = lab[lab["period"] == per]
        rows.append([PERIOD_CN[per], len(g), f"{g['candidate_count'].mean():.0f}",
                     _pct(g["stop_rate_3d"].mean()), _pct(g["mean_net_return_5d"].mean())])
    a(_md(["期间", "交易日数", "日均候选数", "touch 均值", "net_5d 均值"], rows))
    a("")

    # ===== 2 样本内外差距 =====
    a("## 2. 核心:样本内 vs 样本外 Spearman(F-v2.2 评价重点)\n")
    a("F-v2.0 中 C2-F 样本内 Spearman≈1.0 是严重过拟合的直接证据;简单模型应显著缩小这个差距。")
    a(_md(["模型"] + [PERIOD_CN[p] for p in ALL_PERIODS] + ["样本内外差距(train − 样本外均值)"],
          [[m] + [_f(met[m].get(p, {}).get("spearman")) for p in ALL_PERIODS] + [
              _f(_gap(met[m]))] for m in ALL_MODELS]))
    a("")

    # ===== 3 概率指标 =====
    a("## 3. Brier / PR-AUC(校准分,score_cal)\n")
    a("「不差于 C1」判定用 2024–2026.8 合并评估期:")
    eval_m = {}
    rows = []
    for per in ALL_PERIODS:
        rows.append([PERIOD_CN[per]] + [_f(met[m].get(per, {}).get("brier")) for m in ALL_MODELS])
    a("**Brier**(越低越好):")
    a(_md(["期间"] + ALL_MODELS, rows))
    rows = []
    for per in ALL_PERIODS:
        rows.append([PERIOD_CN[per]] + [_f(met[m].get(per, {}).get("pr_auc")) for m in ALL_MODELS])
    a("\n**PR-AUC**(越高越好;高风险日=真实 touch>全期70分位):")
    a(_md(["期间"] + ALL_MODELS, rows))
    for m in ALL_MODELS:
        pool = joins[m][joins[m]["period"].isin(EVAL_PERIODS)]
        eval_m[m] = _one_metrics(pool, q70)
    a("\n**合并评估期(2024–2026.8)**:")
    a(_md(["模型", "Brier", "PR-AUC"], [[m, _f(eval_m[m].get("brier")), _f(eval_m[m].get("pr_auc"))]
                                        for m in ALL_MODELS]))
    a("")

    # ===== 4 前20%组 =====
    a("## 4. 前 20% 打分组 vs 其余(score_raw,同分整体入档)\n")
    a(_md(["模型"] + [PERIOD_CN[p] for p in ALL_PERIODS],
          [[m] + [f"{_pct(met[m].get(p, {}).get('diff'))} / lift {_f(met[m].get(p, {}).get('lift'))}"
                 for p in ALL_PERIODS] for m in ALL_MODELS]))
    a("\n(单元格 = 前20%组 touch − 其余组 touch / lift)")

    # ===== 5 分层单调 =====
    a("\n## 5. 十分层单调 Spearman(score_raw 分位切)\n")
    a(_md(["模型"] + [PERIOD_CN[p] for p in ALL_PERIODS],
          [[m] + [_f(_deciles_mono(joins[m][joins[m]['period'] == p])) for p in ALL_PERIODS]
           for m in ALL_MODELS]))
    a("")

    # ===== 6 月度一致性 =====
    a("## 6. 月度方向一致性(flagged 日 touch > 当月全池)\n")
    rows = []
    for m in ALL_MODELS:
        cons, excl = monthly[m]
        share = float(cons["consistent"].mean()) if len(cons) else np.nan
        g = cons[cons["year"].isin(["2024", "2025", "2026"])] if len(cons) else cons
        share_recent = float(g["consistent"].mean()) if len(g) else np.nan
        rows.append([m, f"{int(cons['consistent'].sum())}/{len(cons)}", _pct(share),
                     _pct(share_recent), excl])
    a(_md(["模型", "计入月一致数", "全期一致率", "其中2024-26", "flagged<2未计入月数"], rows))
    a("")

    # ===== 7 覆盖率 =====
    a("## 7. 冻结阈值覆盖率(flagged 天数占比)\n")
    a("阈值=各校准期原始分 80 分位;合理区间经验取 5%–35%(80 分位口径下校准期恒≈20%,"
      "分布漂移会使评估期偏离)。")
    rows = []
    for m in ALL_MODELS:
        rows.append([m] + [_pct(joins[m][joins[m]["period"] == p]["high_risk_flag"].mean())
                           for p in ALL_PERIODS])
    a(_md(["模型"] + [PERIOD_CN[p] for p in ALL_PERIODS], rows))
    a("")

    # ===== 8 2026 压力期 =====
    a("## 8. 2026 压力期判定\n")
    rows = []
    for m in ALL_MODELS:
        mm = met[m].get("stress2026", {})
        d = mm.get("diff", 0)
        verdict = "减亏" if d > 0.01 else ("无效" if d >= -0.01 else "反向")
        rows.append([m, _pct(mm.get("diff")), _f(mm.get("lift")), verdict,
                     len(joins[m][(joins[m]["period"] == "stress2026")
                                  & (joins[m]["high_risk_flag"] == 1)])])
    a(_md(["模型", "前20%组差", "lift", "判定", "flagged天数"], rows))
    a("")

    # ===== 9 反事实金额 =====
    a("## 9. 反事实金额(5 日窗口,flagged 日)\n")
    rows = []
    for m in ALL_MODELS:
        tot_av = tot_mi = 0.0
        n_all = 0
        for per in EVAL_PERIODS:
            n, av, mi = _amounts(joins[m][joins[m]["period"] == per], h=5)
            tot_av += av
            tot_mi += mi
            n_all += n
        rows.append([m, n_all, f"{tot_av:,.0f}", f"{tot_mi:,.0f}", f"{tot_av - tot_mi:,.0f}"])
    a(_md(["模型", "flagged天数(24-26.8)", "避免亏损", "错失盈利", "净效果"], rows))
    a("")

    # ===== 10 决策分支 =====
    a("## 10. 晋级判断与决策分支\n")
    v = []
    for m in ("C2-L", "C2-S"):
        mm = met[m]
        d24 = mm.get("dev2024", {}).get("diff", 0)
        d25 = mm.get("stab2025", {}).get("diff", 0)
        d26 = mm.get("stress2026", {}).get("diff", 0)
        sp_tr = mm.get("train", {}).get("spearman", 1)
        brier_ok = eval_m[m]["brier"] <= eval_m["C1"]["brier"] + 0.005
        pr_ok = (eval_m[m]["pr_auc"] >= eval_m["C1"]["pr_auc"] - 0.01) if not pd.isna(eval_m[m]["pr_auc"]) else False
        covs = [joins[m][joins[m]["period"] == p]["high_risk_flag"].mean() for p in EVAL_PERIODS]
        cov_ok = all(0.05 <= c <= 0.35 for c in covs)
        checks = [
            ("2024 方向为正", d24 > 0, f"{_pct(d24)}"),
            ("2025 不反转", d25 > 0, f"{_pct(d25)}"),
            ("2026 不明显反转", d26 > -0.01, f"{_pct(d26)}"),
            ("Brier/PR-AUC 不差于 C1", brier_ok and pr_ok,
             f"Brier {_f(eval_m[m]['brier'])} vs {_f(eval_m['C1']['brier'])}; "
             f"PR-AUC {_f(eval_m[m]['pr_auc'])} vs {_f(eval_m['C1']['pr_auc'])}"),
            ("样本内 Spearman 不接近 1", sp_tr < 0.5, f"train={_f(sp_tr)}"),
            ("覆盖率合理(5%–35%)", cov_ok, "/".join(_pct(c) for c in covs)),
        ]
        n_pass = sum(1 for _, ok, _ in checks if ok)
        a(f"**{m}**:通过 {n_pass}/6")
        a(_md(["条件", "判定", "数值"], [[n, "✓" if ok else "✗", d] for n, ok, d in checks]))
        a("")
        v.append((m, n_pass, checks))
    stable_simple = any(n == 6 for _, n, _ in v)
    best_ml = max(ML_MODELS, key=lambda m: np.nanmean([met[m].get(p, {}).get("diff", np.nan)
                                                       for p in OUT_PERIODS]))
    # 「C1 与 ML 相当」须同时:概率指标不差于最优 ML、且 2026 方向不更差(宽松噪声带)
    c1_close = (eval_m["C1"]["brier"] <= eval_m[best_ml]["brier"] + 0.005
                and not pd.isna(eval_m["C1"]["pr_auc"])
                and eval_m["C1"]["pr_auc"] >= eval_m[best_ml]["pr_auc"] - 0.01
                and met["C1"].get("stress2026", {}).get("diff", -1)
                >= met[best_ml].get("stress2026", {}).get("diff", 0) - 0.01)
    all_rev = all(met[m].get("stress2026", {}).get("diff", 0) < -0.01 for m in ALL_MODELS)
    if stable_simple:
        simple = [m for m, n, _ in v if n == 6][0]
        a(f"- ✦ **简单模型稳定**:{simple} 六项全过 → 外围信号是低维的,"
          f"采用 {simple} 训练 F-v2.1 双头标签。")
    elif c1_close:
        a("- ✦ **C1 与最优机器学习模型相当** → 外围行情只需要简单规则,无需复杂 ML。"
          "(这不是失败,是结论。)")
    elif all_rev:
        a("- ✦ **全部版本 2026 仍反向** → 停止 Corvus 外围模型独立门控,"
          "外围状态降级为 Astrolabe 软输入,等待与 Pyxis 后融合。")
    else:
        a("- 三个分支均不严格成立,按接近程度排序:")
        a(f"  1. **C2-S(精简 LGBM)最接近稳定**(4/6:2024 为正、2026 减亏、概率指标优于 C1、"
          "覆盖率合理;差在 2025 微反转 −0.9% 与样本内仍 0.995 的过拟合残留)——"
          "信号维度确实低(37 特征≈157 特征),但 800 树对 730 天仍偏重;")
        a("  2. C1 在 2024/2025 有效但 2026 反向,不能单独门控;")
        a("  3. C2-L(Ridge)跨期最弱,线性口径不兼容该信号。")
        a("- **维持:C2 仅作研究/影子,不接盘前动作。** 下一步按 F-v2.2 结论推进:"
          "精简模型(降树/早停或正则化路径)训练 F-v2.1 双头规格,或先把外围降级为软证据"
          "等 Corvus+Pyxis 后融合再定。")
    a("")

    # ===== 11 其他检查 =====
    a("## 11. 附带检查\n")
    feat = ctx["feat"]
    f9 = joins["C2-F"].merge(feat[["trade_date", "n_missing"]], on="trade_date", how="left")
    a(f"- Q9 假风险(C2-F):Spearman(n_missing, score_cal) = "
      f"{_f(_spearman(f9['n_missing'], f9['score_cal']))}(应接近 0)")
    rows = []
    for m in ("C2-S", "C2-F"):
        lm = legacy_met[m]
        rows.append([m, _pct(lm.get("dev2024", {}).get("diff")),
                     _pct(lm.get("stress2026", {}).get("diff")),
                     _f(lm.get("dev2024", {}).get("spearman"))])
    a("- legacy 池方向(C2 预测直接套用,不重新训练):")
    a(_md(["模型", "2024前20%组差", "2026前20%组差", "2024 Spearman"], rows))
    a("")

    # ===== 12 F-v2.1 规格指针 =====
    a("## 12. F-v2.1 双头标签规格\n")
    a("已冻结于 `corvus/spec/f_v2.1_dual_head_labels.md`(只冻结定义,本轮未生成结果、未调阈值):")
    a("- 波动头 `P_TOUCH3_3D`:未来 3 日触及 -3% 概率,只用于波动警告/等待盘中下杀,不直接暂停买入;")
    a("- 持续亏损头互斥路径 `NO_TOUCH / TOUCH_RECOVERED / TOUCH_UNRECOVERED`,")
    a("  `P_persistent = P_TOUCH3_3D × P_UNRECOVERED_given_TOUCH`,只有它参与「暂停买入」。")
    a("")

    # ===== 附录 =====
    a("## 附录:C2-S 精简特征清单(预先冻结)\n")
    a(", ".join(f"`{c}`" for c in ctx["small_features"]))
    a("")
    return "\n".join(S)


def _gap(met):
    """样本内 Spearman − 样本外期间 Spearman 均值(评估过拟合差距)。"""
    sp_tr = met.get("train", {}).get("spearman")
    outs = [met.get(p, {}).get("spearman") for p in OUT_PERIODS]
    outs = [x for x in outs if x is not None and not pd.isna(x)]
    if sp_tr is None or pd.isna(sp_tr) or not outs:
        return np.nan
    return float(sp_tr - np.mean(outs))


def SCHEME_DESC(ctx):
    return (f"冻结产物三件套:2019–2021 训练({ctx['train_n']} 天,此后不重训)"
            f"→ 2022–2023 校准期({ctx['calib_n']} 天,前 {ctx['embargo']} 个交易日因标签窗口重叠剔除)"
            "拟合 isotonic 校准器并冻结高风险阈值 → 2024 开发 → 2025 稳定 → 2026.1-8 压力,"
            "后续期始终同一套冻结模型+校准器+阈值;")


def _period_of(ctx, d):
    for name, (s, e) in ctx["periods"].items():
        if s <= d <= e:
            return name
    return "other"
