# -*- coding: utf-8 -*-
"""阶段4：B 模式三组消融 LightGBM 基线。

变体：B0=原始规则；B1=放宽缩量；B2=放宽下影/收阳（build_events_b v1 产出）。
协议（与阶段3一致）：
- 训练 2019-2021，OOF 2022-2023 拟合 isotonic 校准 + 冻结 EV 阈值（OOF 校准后 90 分位）；
- 2024 选择保留变体（只此用途）；2025 仅描述，不反向调整；
- K_stop=1.5 统一； LightGBM 三分类（TARGET/STOP/TIMEOUT）×4档 target_atr 展开。
指标：年度事件数/日覆盖率、三分类分布、日Rank IC(t)、Top10%、冻结阈值后真账本
（交易数/平均仓位/年化/回撤/胜率/盈亏比）、月度收益、板块与个股集中度、bootstrap CI。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
sys.path.insert(0, OUT_DIR)
import train_lgbm_baseline as B  # noqa: E402

VARIANTS = ["b0", "b1", "b2"]
KS = "1.5"
LGB_PARAMS = dict(B.LGB_PARAMS)
# B 样本小：降复杂度防过拟合
LGB_PARAMS.update(num_leaves=31, min_child_samples=40, feature_fraction=0.8)

# B 事件额外特征（板块日度聚合，决策时点已知；非泄漏）
B_EXTRA = ["sector_ret1", "sector_ret5", "panic_pct", "limit_down_cnt",
           "limit_up_cnt", "n_cons", "center_ret1"]
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_b(name):
    ev = pd.read_parquet(os.path.join(OUT_DIR, f"events_b_{name}_cooled.parquet"))
    ev["subtype"] = "b1"
    ev["in_watch_sector"] = True  # B 定义上全部来自 12 关注板块
    return ev


def select_features_b(ev):
    feats = list(dict.fromkeys(
        [c for c in B.FEATURE_WHITELIST if c in ev.columns] + B_EXTRA))
    dropped = [c for c in B.FEATURE_WHITELIST if c not in ev.columns]
    X = ev[feats].copy().replace([np.inf, -np.inf], np.nan)
    for c in feats:
        s = X[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s):
            X[c] = s.astype(np.float32)
    return X, feats, dropped


def fit_clf(tr_ev, X, feats):
    tr_ev = tr_ev[tr_ev[f"eligible_s{KS}"]]
    long = B.make_long(tr_ev, KS)
    Xl = X.loc[long.index].copy()
    Xl["target_atr"] = long["target_atr"].values
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=800)
    clf.fit(Xl, long["y"].values, callbacks=[lgb.log_evaluation(0)])
    return clf


def predict_long(clf, ev, X, feats):
    ev = ev[ev[f"eligible_s{KS}"]]
    long = B.make_long(ev, KS)
    Xl = X.loc[long.index].copy()
    Xl["target_atr"] = long["target_atr"].values
    proba = clf.predict_proba(Xl)
    for i, cl in enumerate(clf.classes_):
        long[f"p_{cl}"] = proba[:, i]
    for cl in B.CLASSES:
        if f"p_{cl}" not in long.columns:
            long[f"p_{cl}"] = 0.0
    return long


def bootstrap_ci(nav, trades, n_boot=1000, seed=7):
    rng = np.random.default_rng(seed)
    out = {}
    if len(nav) > 5:
        dr = np.diff(nav) / nav[:-1]
        anns = []
        for _ in range(n_boot):
            s = rng.choice(dr, size=len(dr), replace=True)
            path = np.cumprod(1 + s)
            anns.append(path[-1] ** (252 / len(s)) - 1)
        out["ann_ci"] = tuple(float(np.quantile(anns, q)) for q in (0.025, 0.975))
    if len(trades) >= 10:
        pnl = trades["pnl"].values
        wins = []
        for _ in range(n_boot):
            s = rng.choice(pnl, size=len(pnl), replace=True)
            wins.append((s > 0).mean())
        out["win_ci"] = tuple(float(np.quantile(wins, q)) for q in (0.025, 0.975))
    return out


def portfolio_detail(pf, evd, code2board=None):
    """月度收益、集中度、bootstrap（基于真账本 _trades/_nav/_dates）"""
    d = {}
    if not pf or "_trades" not in pf or not len(pf["_trades"]):
        return d
    t = pf["_trades"].copy()
    nav, dates = pf["_nav"], pf["_dates"]
    pnl = t["pnl"]
    d["pl_ratio"] = float(pnl[pnl > 0].mean() / abs(pnl[pnl < 0].mean())) \
        if (pnl > 0).any() and (pnl < 0).any() else np.nan
    # 月度收益：月内已实现pnl加回月初净值占比（近似盯市）
    t["ym"] = t["date"] // 100
    mon = []
    for ym, g in t.groupby("ym"):
        mon.append((ym, float(g["pnl"].sum())))
    d["monthly"] = mon
    tot_pnl = float(t["pnl"].sum())
    t_sorted = t["pnl"].abs().sort_values(ascending=False)
    d["top10_trade_share"] = float(
        t.reindex(t_sorted.head(min(10, len(t))).index)["pnl"].sum()
        / tot_pnl) if tot_pnl else np.nan
    # 板块集中度（用事件表的 board 映射）
    if code2board is not None:
        t["board"] = t["code"].map(code2board)
        bsum = t.groupby("board")["pnl"].sum()
        d["board_share"] = (bsum.abs().sort_values(ascending=False)
                            .head(5).index.tolist())
        d["top5_board_share"] = float(
            bsum.reindex(d["board_share"]).sum() / tot_pnl) if tot_pnl else np.nan
    d["n_codes"] = int(t["code"].nunique())
    d.update(bootstrap_ci(nav, t))
    return d


def run_variant(name):
    ev = load_b(name)
    X, feats, dropped = select_features_b(ev)
    seg = lambda y0, y1: ev[(ev["date"] >= y0) & (ev["date"] <= y1)
                            & (~ev["is_censored"])
                            & (ev["label_end_date"].notna())
                            & (ev["label_end_date"] <= y1)]
    tr, oof = seg(20190101, 20211231), seg(20220101, 20231231)
    va, t5 = seg(20240101, 20241231), seg(20250101, 20251231)
    log(f"[{name}] feats={len(feats)} (A白名单缺 {len(dropped)} 列) "
        f"tr={len(tr)} oof={len(oof)} va={len(va)} t5={len(t5)}")
    clf = fit_clf(tr, X, feats)
    # OOF 校准 + 冻结阈值
    oof_long = predict_long(clf, oof, X, feats)
    evp = B.EVParams(tr, KS)
    oof_long["EV"] = evp.ev(oof_long)
    cal = B.fit_isotonic(oof_long)
    oof_cal = B.apply_isotonic(oof_long, cal)
    oof_cal["EV"] = evp.ev(oof_cal)
    thr = float(oof_cal["EV"].quantile(0.9))
    const_p = B.const_probs(tr, KS, "b1")
    res = {}
    for seg_ev, tag in [(va, "2024"), (t5, "2025")]:
        raw = predict_long(clf, seg_ev, X, feats)
        raw["EV"] = evp.ev(raw)
        evd = B.event_select(raw)
        cal_seg = B.apply_isotonic(raw, cal)
        cal_seg["EV"] = evp.ev(cal_seg)
        evc = B.event_select(cal_seg)
        # 覆盖全局 PATH_MAP/EVENT_QUOTE 为 B 变体路径
        B.PATH_MAP = pd.read_parquet(
            os.path.join(OUT_DIR, f"events_path_b_{name}_cooled.parquet")
            ).set_index("event_id")
        B.EVENT_QUOTE = ev.drop_duplicates("event_id").set_index("event_id")
        pf_fill = B.portfolio_backtest(evc, ks=KS)
        pf_thr = B.portfolio_backtest(evc, ks=KS, ev_thr=thr) if thr else {}
        dm = B.daily_metrics(evc, thr=thr)
        sm = B.strat_metrics(evd, const_p, ks=KS, thr=thr)
        det = portfolio_detail(pf_thr, evc,
                               code2board=ev.drop_duplicates("code")
                               .set_index("code")["board"])
        lab_fixed = f"outcome_s{KS}_t2.0"
        elig = seg_ev[seg_ev[f"eligible_s{KS}"]]
        res[tag] = {
            "n": len(evd), "sm": sm, "dm": dm, "thr": thr,
            "pf_fill": {k: v for k, v in pf_fill.items() if not k.startswith("_")},
            "pf_thr": {k: v for k, v in pf_thr.items() if not k.startswith("_")},
            "dist_fixed_t2": (elig[lab_fixed].value_counts(normalize=True).to_dict()
                              if len(elig) else {}),
            "det": det,
        }
        log(f"[{name}|{tag}] n={len(evd)} dIC "
            f"{dm.get('d_ic_mean', np.nan):+.4f}(t={dm.get('d_ic_t', np.nan):+.1f}) "
            f"Top10 {sm['top_ret']:+.2%} | 阈值账本 "
            f"{res[tag]['pf_thr'].get('ann_ret', np.nan):+.2%} "
            f" trades={res[tag]['pf_thr'].get('n_trades', 0)}")
    return res


def main():
    t0 = time.time()
    all_res = {}
    for v in VARIANTS:
        all_res[v] = run_variant(v)
    pd.to_pickle(all_res, os.path.join(OUT_DIR, "b_ablation_results.pkl"))

    # ---- 报告 ## 10 ----
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n## 10. ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n## 10. B 模式消融实验（阶段4，LightGBM 基线）\n",
         "协议：训练2019-2021；OOF2022-2023 拟合 isotonic + 冻结 EV 阈值（校准后90分位）；"
         "K_stop=1.5；2024 选择变体，2025 仅描述。真账本允许空仓（20槽）。"
         "bootstrap=日收益重采样1000次（年化CI）/交易重采样（胜率CI），近似口径。\n",
         "| 变体 | 段 | n | 日IC(t) | Top10% | 阈值 | 账本:交易数 | 日均仓位 | 年化 | 最大回撤 | 胜率 | 盈亏比 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v in VARIANTS:
        for tag in ["2024", "2025"]:
            r = all_res[v][tag]
            sm, dm, pf, det = r["sm"], r["dm"], r["pf_thr"], r["det"]
            win = pf.get("trade_win", np.nan)
            plr = det.get("pl_ratio", np.nan)
            L.append(
                f"| {v} | {tag} | {r['n']} | "
                f"{dm.get('d_ic_mean', np.nan):+.4f}({dm.get('d_ic_t', np.nan):+.1f}) | "
                f"{sm['top_ret']:+.2%} | {r['thr']:+.3%} | "
                f"{pf.get('n_trades', 0)} | {pf.get('avg_pos', np.nan):.1f} | "
                f"{pf.get('ann_ret', np.nan):+.2%} | {pf.get('max_dd', np.nan):+.2%} | "
                f"{win:.1%} | {plr:.2f} |")
    for v in VARIANTS:
        r24 = all_res[v]["2024"]
        L.append(f"\n- **{v}** 年化CI(2024阈值口径): "
                 f"{r24['det'].get('ann_ci', ('nan','nan'))[0]:+.2%} ~ "
                 f"{r24['det'].get('ann_ci', (0,0))[1]:+.2%}；"
                 f"top10笔收益贡献占比 {r24['det'].get('top10_trade_share', np.nan):.1%}；"
                 f"前5板块贡献占比 {r24['det'].get('top5_board_share', np.nan):.1%} "
                 f"({', '.join(map(str, r24['det'].get('board_share', [])))})；"
                 f"涉及股票数 {r24['det'].get('n_codes', 0)}")
        mon = r24["det"].get("monthly", [])
        L.append(f"  - 2024月度pnl: " + ", ".join(f"{m}: {p:+.3f}" for m, p in mon))
    L.append("\n- 选择规则：2024 年排序（日IC/Top10/账本）最优的变体保留；"
             "2025 行仅检查方向是否延续，不反向调整。")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
