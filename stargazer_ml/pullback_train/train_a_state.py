# -*- coding: utf-8 -*-
"""阶段3：A 模型市场状态条件化实验（v3.3）。

配置：
- A0 对照：83 特征白名单 + target_atr
- A1 状态交互：A0 + (market_vol_state - 训练中位) × 8 个关键特征
- A2 去市场状态对照：A0 剔除全部市场指数特征

评估（分层 subtype = adj_ready / first_div，冻结 K 来自基线）：
- 校准/阈值：训练期内 rolling——2019-2021 训练 -> 2022/2023 OOF 拟合 isotonic 与
  EV 阈值（全局 90 分位 + 分 mvs 桶 90 分位），2024 只做确认选择，2025 只描述；
- 指标：每日横截面 IC、Top10%、冻结阈值样本收益、真账本组合（强制填仓 / 阈值+空仓）；
- 分 mvs 桶报告每日 IC，检查市场状态变化后排序是否反转；
- first_div 只按风险过滤器评价（Top 层相对基线的减亏幅度），不作收益来源。

只读 events_train_v1.parquet；模型不入盘（实验产物），报告追加 ## 8.9 节。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
sys.path.insert(0, OUT_DIR)

import train_lgbm_baseline as B  # noqa: E402  (main 有 __name__ 守卫，import 安全)

from sklearn.isotonic import IsotonicRegression  # noqa: E402

EV = pd.read_parquet(os.path.join(OUT_DIR, "events_train_v1.parquet"))

INTERACT_BASE = ["quality_score", "amount_ratio20", "rsi14",
                 "pullback_from_high5", "turnover_ratio20", "sector_ret5",
                 "boll_pct_b", "eq_ret20"]
MKT_FEATS = ["eq_ret1", "eq_ret5", "eq_ret20", "market_vol_state",
             "market_pos20", "sh_ret1", "sh_ret5", "sh_ret20",
             "cyb_ret1", "cyb_ret5", "cyb_ret20"]

FREEZE_K = {"adj_ready": "1.2", "first_div": "1.5"}  # 与基线冻结表一致
MVS_Q = [1 / 3, 2 / 3]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_X(ev, feats, mvs_center=None, interact=None):
    X = ev[feats].copy().replace([np.inf, -np.inf], np.nan)
    for c in list(X.columns):
        s = X[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s):
            X[c] = s.astype(np.float32)
    if interact:
        m = ev["market_vol_state"] - (mvs_center if mvs_center is not None else 0.0)
        for f in interact:
            if f in X.columns:
                X[f"ix_{f}"] = (m * X[f]).astype(np.float32)
    return X


def fit_clf(tr_ev, feats, ks, mvs_center=None, interact=None, seed=7):
    tr_ev = tr_ev[tr_ev[f"eligible_s{ks}"]]
    # 内部时间 8:2 尾段做早停（与基线一致）
    es_from = tr_ev["date"].quantile(0.8)
    main = tr_ev[tr_ev["date"] < es_from]
    es = tr_ev[tr_ev["date"] >= es_from]
    long_m = B.make_long(main, ks)
    X_m = build_X(main, feats, mvs_center, interact).reindex(long_m.index)
    X_m["target_atr"] = long_m["target_atr"].values
    long_e = B.make_long(es, ks)
    X_e = build_X(es, feats, mvs_center, interact).reindex(long_e.index)
    X_e["target_atr"] = long_e["target_atr"].values
    clf = B.lgb.LGBMClassifier(**B.LGB_PARAMS, n_estimators=800)
    clf.fit(X_m, long_m["y"].values, eval_set=[(X_e, long_e["y"].values)],
            callbacks=[B.lgb.early_stopping(80, verbose=False),
                       B.lgb.log_evaluation(0)])
    return clf


def predict_long(clf, seg_ev, feats, ks, mvs_center=None, interact=None):
    seg_ev = seg_ev[seg_ev[f"eligible_s{ks}"]]
    long = B.make_long(seg_ev, ks)
    X = build_X(seg_ev, feats, mvs_center, interact).reindex(long.index)
    X["target_atr"] = long["target_atr"].values
    proba = clf.predict_proba(X)
    for i, cl in enumerate(clf.classes_):
        long[f"p_{cl}"] = proba[:, i]
    for cl in B.CLASSES:
        if f"p_{cl}" not in long.columns:
            long[f"p_{cl}"] = 0.0
    # 长表只带 KEYS，市场状态需从事件表映射回来（分桶用）
    m = seg_ev.drop_duplicates("event_id") \
              .set_index("event_id")["market_vol_state"]
    long["market_vol_state"] = long["event_id"].map(m)
    return long


def fit_isotonic_by(long, by=None):
    """全局或分桶 isotonic 校准器。by=None -> 全局; by=Series -> 每桶。"""
    out = {}
    groups = [(None, long)] if by is None else list(long.groupby(by))
    for gk, g in groups:
        cal = {}
        for cl in B.CLASSES:
            y = (g["y"] == cl).astype(float).values
            cal[cl] = IsotonicRegression(out_of_bounds="clip").fit(g[f"p_{cl}"], y)
        out[gk if gk is not None else "ALL"] = cal
    return out


def apply_iso(long, cal, by=None):
    # 长表保留原事件索引（4 行共享），必须先重置，否则 .loc[重复标签] 会展开行
    out = long.reset_index(drop=True).copy()
    if by is None:
        groups = [("ALL", out)]
    else:
        groups = list(out.groupby(by))
    for gk, gdf in groups:
        c = cal[gk if gk in cal else "ALL"]
        idx = gdf.index
        ps = np.zeros((len(idx), 3))
        for i, cl in enumerate(B.CLASSES):
            ps[:, i] = c[cl].predict(out.loc[idx, f"p_{cl}"])
        ps = np.clip(ps, 0, 1)
        ps = ps / ps.sum(axis=1, keepdims=True)
        for i, cl in enumerate(B.CLASSES):
            out.loc[idx, f"p_{cl}"] = ps[:, i]
    return out


def evaluate_config(name, feats, interact, tr, va, t5, mvs_center):
    rows = []
    for st in ["adj_ready", "first_div"]:
        ks = FREEZE_K[st]
        tr_st = tr[tr["subtype"] == st]
        tr_fit = tr_st[tr_st["date"] <= 20211231]
        oof = tr_st[(tr_st["date"] >= 20220101) & (tr_st["date"] <= 20231231)]
        clf = fit_clf(tr_fit, feats, ks, mvs_center, interact)
        oof_long = predict_long(clf, oof, feats, ks, mvs_center, interact)
        oof_long["EV"] = B.EVParams(tr_st, ks).ev(oof_long)
        # 桶标签（训练期三分位）
        q1, q2 = tr_st["market_vol_state"].quantile(MVS_Q).tolist()
        def bkt(s):
            return np.where(s <= q1, 0, np.where(s <= q2, 1, 2))
        # 校准器（全局 + 分桶）在 OOF 上拟合；阈值取校准后 OOF EV 的 90 分位
        cal_all = fit_isotonic_by(oof_long)
        oof_cal = apply_iso(oof_long, cal_all)
        oof_cal["EV"] = B.EVParams(tr_st, ks).ev(oof_cal)
        thr = float(oof_cal["EV"].quantile(0.9))
        oof_long["mvs_b"] = bkt(oof_long["market_vol_state"])
        cal_b = fit_isotonic_by(oof_long, by="mvs_b")
        oof_cal_b = apply_iso(oof_long, cal_b, by="mvs_b")
        oof_cal_b["EV"] = B.EVParams(tr_st, ks).ev(oof_cal_b)
        thr_b = oof_cal_b.groupby("mvs_b")["EV"].quantile(0.9)
        for seg_ev, tag in [(va, "2024"), (t5, "2025")]:
            seg_st = seg_ev[seg_ev["subtype"] == st]
            raw = predict_long(clf, seg_st, feats, ks, mvs_center, interact)
            raw["EV"] = B.EVParams(tr_st, ks).ev(raw)
            evd = B.event_select(raw)
            base = B.daily_metrics(evd)
            base["n"] = len(evd)
            base["top10"] = float(evd[evd["EV"] >= evd["EV"].quantile(0.9)]
                                  ["net_ret"].mean())
            base["mean_ret"] = float(evd["net_ret"].mean())
            # 全局校准
            cal = apply_iso(raw, cal_all)
            cal["EV"] = B.EVParams(tr_st, ks).ev(cal)
            evc = B.event_select(cal)
            pf_fill = B.portfolio_backtest(evc, ks=ks)
            pf_thr = B.portfolio_backtest(evc, ks=ks, ev_thr=thr)
            # 分状态校准 + 分状态阈值
            raw_b = raw.assign(mvs_b=bkt(raw["market_vol_state"]))
            calb = apply_iso(raw_b, cal_b, by="mvs_b")
            calb["EV"] = B.EVParams(tr_st, ks).ev(calb)
            evcb = B.event_select(calb)
            _tb = evcb["mvs_b"].map(thr_b)   # 分状态阈值按桶映射到行
            pf_b = B.portfolio_backtest(
                evcb[evcb["EV"] >= _tb.fillna(np.inf)], ks=ks)
            # 分状态每日 IC（校准后）
            ic_by = evcb.groupby("mvs_b").apply(
                lambda g: B.spearmanr(g["EV"], g["net_ret"]).statistic
                if len(g) >= 30 else np.nan, include_groups=False)
            rows.append({
                "cfg": name, "st": st, "seg": tag, "n": len(evd),
                "d_ic": base.get("d_ic_mean"), "d_ic_t": base.get("d_ic_t"),
                "d_ic_pos": base.get("d_ic_pos"),
                "top10": base["top10"], "mean_ret": base["mean_ret"],
                "thr": thr,
                "pf_fill_ann": pf_fill.get("ann_ret"),
                "pf_thr_ann": pf_thr.get("ann_ret"),
                "pf_b_ann": pf_b.get("ann_ret"),
                "ic_b0": ic_by.get(0, np.nan), "ic_b1": ic_by.get(1, np.nan),
                "ic_b2": ic_by.get(2, np.nan),
            })
            log(f"[{name}|{st}|{tag}] n={len(evd)} dIC "
                f"{base.get('d_ic_mean', np.nan):+.4f}(t={base.get('d_ic_t', np.nan):+.1f}) "
                f"Top10 {base['top10']:+.2%} | 账本 填仓{pf_fill.get('ann_ret', np.nan):+.2%} "
                f"阈值{pf_thr.get('ann_ret', np.nan):+.2%} "
                f"分状态{pf_b.get('ann_ret', np.nan):+.2%}")
    return rows


def main():
    t0 = time.time()
    ev = B.load_event_level()  # 含多板块展开去重（与基线一致）
    tr = B.segment(ev, 20190101, 20231231, 20231231)
    va = B.segment(ev, 20240101, 20241231, 20241231)
    t5 = B.segment(ev, 20250101, 20251231, 20251231)
    feats_all, _ = B.select_features(ev)
    feats = list(feats_all.columns)
    mvs_center = float(tr["market_vol_state"].median())
    log(f"train={len(tr)} feats={len(feats)} mvs_center={mvs_center:.3f}")

    cfgs = [
        ("A0_对照", feats, None),
        ("A1_状态交互", feats, INTERACT_BASE),
        ("A2_去市场状态", [f for f in feats if f not in MKT_FEATS], None),
    ]
    rows = []
    for name, fs, ix in cfgs:
        rows.extend(evaluate_config(name, fs, ix, tr, va, t5, mvs_center))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "state_conditional_results.csv"),
              index=False, encoding="utf-8-sig")

    # ---- 报告 ## 8.9 节 ----
    with open(B.REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n## 8.9 ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n## 8.9 市场状态条件化实验（阶段3，v3.3）\n",
         "配置：A0=白名单基线；A1=A0+(market_vol_state−训练中位)×8关键特征；"
         "A2=A0剔除全部市场指数特征。校准/阈值：2019-2021 训练、2022/2023 OOF 拟合 "
         "isotonic 与 EV 阈值（全局 90 分位 + 分 mvs 桶 90 分位）；2024 确认、2025 仅描述。"
         "账本=v3.3 真账本（B.portfolio_backtest 导入复用）。\n",
         "| 配置 | 子类 | 段 | n | 日IC(t) | Top10% | 基线 | 账本:填仓 | 账本:全局阈值 | 账本:分状态阈值 | IC@mvs低 | IC@中 | IC@高 |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def _pc(x):  # 安全百分比格式化（None/NaN → nan）
        return f"{x:+.2%}" if x is not None and pd.notna(x) else "nan"
    for _, r in df.iterrows():
        L.append(f"| {r['cfg']} | {r['st']} | {r['seg']} | {int(r['n'])} | "
                 f"{r['d_ic']:+.4f}({r['d_ic_t']:+.1f}) | {r['top10']:+.2%} | "
                 f"{r['mean_ret']:+.2%} | {_pc(r['pf_fill_ann'])} | "
                 f"{_pc(r['pf_thr_ann'])} | {_pc(r['pf_b_ann'])} | "
                 f"{r['ic_b0']:+.3f} | {r['ic_b1']:+.3f} | {r['ic_b2']:+.3f} |")
    L.append("\n- first_div 行只按风险过滤器解读：若 Top10% 相对基线减亏且跨段稳定，"
             "保留为风险过滤；不要求正收益。")
    L.append("- 判读：若 A1 在 2025（市场状态与训练期差异最大）相对 A0 的日IC/账本改善，"
             "且 A2 显著变差，说明排序信号依赖市场状态、条件化有价值；"
             "若 A1≈A0 且 A2 不变，说明信号本身与市场状态弱耦合。")
    with open(B.REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
