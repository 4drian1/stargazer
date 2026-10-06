# -*- coding: utf-8 -*-
"""第2步：市场危险日分类 + 第3步：R0硬切（预先冻结三档）。

市场模型 v2（替代 fwd5收益回归）：
- 三分类：UNSUITABLE / NEUTRAL / SUITABLE。标签分位从**训练期2019-2021**冻结
  （fwd_pool_ret 后30%=UNSI，前30%=SUIT，中=NEUT），不用OOF/2024定义标签。
- 特征：market_model v1 全套 + 候选池EV分布/拥挤度/预测STOP比例
  （A2 滚动OOF预测：模型只用 ≤Y-1 数据预测 Y 年，无泄漏）+ 5/10/20日滞后
  + 池状态×广度交互。
- 5日purge/embargo：训练段尾部5个交易日剔除（标签窗口与OOF重叠）。
- 训练2019-2021（EV分布特征2019-20为NaN，树模型容忍并显式标注），
  OOF2022-2023冻结R0分位，2024验证，2025仅描述。

第3步 R0硬切（三档预冻结）：OOF预测UNSI概率最差10%/20%/30%的日期→0仓。
报告：危险日召回、PR-AUC、Brier、校准；R0拦截亏损/错杀盈利/2024减亏/
2025保留/空仓天数/月度分布。
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
sys.path.insert(0, OUT_DIR)
from build_event_funnel import load_data, add_features  # noqa: E402
import train_lgbm_baseline as B  # noqa: E402
import train_a_state as S       # noqa: E402
import lightgbm as lgb          # noqa: E402

MARKET_DAILY = os.path.join(OUT_DIR, "market_daily.parquet")
EVENTS = os.path.join(OUT_DIR, "events_train_v1.parquet")
CONS = os.path.join(OUT_DIR, "board_cons.json")
OUT_CSV = os.path.join(OUT_DIR, "market_danger_day_results.csv")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
CLASSES = ["UNSI", "NEUT", "SUIT"]

LGB_C = dict(objective="multiclass", num_class=3, num_leaves=31,
             learning_rate=0.03, min_child_samples=20, subsample=0.8,
             colsample_bytree=0.8, verbose=-1, seed=7)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def limit_of(code: str) -> float:
    return 0.195 if code.startswith(("30", "68")) \
        else (0.295 if code.startswith(("8", "4", "92")) else 0.095)


def build_day_frame(df):
    lt = df["code"].map(limit_of)
    touched_up = (df["high"] / df["pre_close"] - 1) >= lt - 0.005
    closed_up = df["ret1"] >= lt - 0.005
    g = df.groupby("date", sort=False)
    day = pd.DataFrame({
        "breadth_up": g.apply(lambda x: float((x["ret1"] > 0).mean()),
                              include_groups=False),
        "med_ret": g["ret1"].median(),
        "loss_ratio": g.apply(lambda x: float((x["ret1"] < -0.03).mean()),
                              include_groups=False),
        "big_loss_ratio": g.apply(lambda x: float((x["ret1"] < -0.05).mean()),
                                  include_groups=False),
        "limit_up_cnt": g.apply(lambda x: int(closed_up.loc[x.index].sum()),
                                include_groups=False),
        "limit_down_cnt": g.apply(lambda x: int((x["ret1"] <= -lt.loc[x.index] + 0.005).sum()),
                                  include_groups=False),
        "zhaban_cnt": g.apply(lambda x: int((touched_up.loc[x.index]
                                             & ~closed_up.loc[x.index]).sum()),
                              include_groups=False),
        "tot_amount": g["amount"].sum(),
    }).reset_index()
    day["amt_ratio20"] = day["tot_amount"] / day["tot_amount"].rolling(20).mean()
    md = pd.read_parquet(MARKET_DAILY)[["date", "sh000001_pct_chg", "cyb399006_pct_chg"]]
    md["sh_ret5"] = md["sh000001_pct_chg"].rolling(5).sum()
    md["sh_ret20"] = md["sh000001_pct_chg"].rolling(20).sum()
    md["cyb_ret5"] = md["cyb399006_pct_chg"].rolling(5).sum()
    md["cyb_ret20"] = md["cyb399006_pct_chg"].rolling(20).sum()
    md["mkt_vol20"] = md["sh000001_pct_chg"].rolling(20).std()
    day = day.merge(md.drop(columns=["sh000001_pct_chg", "cyb399006_pct_chg"]),
                    on="date", how="left")
    # 12 板块聚合
    cons = json.load(open(CONS, encoding="utf-8"))
    mb = pd.DataFrame([(c, b) for b, codes in cons.items() for c in codes],
                      columns=["code", "board"])
    x = df.merge(mb, on="code", how="inner")
    lt2 = x["code"].map(limit_of)
    x["_ld"] = x["ret1"] <= -lt2 + 0.005
    x["_lu"] = x["ret1"] >= lt2 - 0.005
    bg = x.groupby(["board", "date"], sort=False)
    bday = bg.agg(sr=("ret1", "mean"), lu=("_lu", "sum"), ld=("_ld", "sum")).reset_index()
    bd = bday.groupby("date", sort=False).agg(
        board_disp=("sr", "std"),
        strong_board_cnt=("sr", lambda s: int((s > 0.01).sum())),
        watch_board_mean=("sr", "mean"), watch_lu=("lu", "sum"),
        watch_ld=("ld", "sum")).reset_index()
    day = day.merge(bd, on="date", how="left")
    return day


def pool_daily():
    ev = pd.read_parquet(EVENTS, columns=["event_id", "date", "code", "subtype",
                                          "eligible_s1.2", "outcome_s1.2_t2.0",
                                          "net_ret_s1.2_t2.0"])
    ev = ev.drop_duplicates("event_id")
    pool = ev[(ev["subtype"] == "adj_ready") & (ev["eligible_s1.2"])]
    pday = pool.groupby("date", sort=False).agg(
        pool_ret=("net_ret_s1.2_t2.0", "mean"),
        pool_stop=("outcome_s1.2_t2.0", lambda s: float((s == "STOP").mean())),
        pool_win=("net_ret_s1.2_t2.0", lambda s: float((s > 0).mean())),
        pool_cnt=("event_id", "nunique")).reset_index().sort_values("date")
    return pday, pool


def a2_rolling_ev(pool):
    """A2 滚动OOF：模型只用 ≤Y-1 的训练数据预测 Y 年候选的 EV 分布。
    返回 date -> (n_cand, mean_ev, p_stop_mean)。"""
    ev = B.load_event_level()
    feats_all, _ = B.select_features(ev)
    feats = [f for f in feats_all.columns if f not in S.MKT_FEATS]
    out = []
    for y in range(2021, 2026):
        tr = B.segment(ev, 20190101, (y - 1) * 10000 + 1231, (y - 1) * 10000 + 1231)
        tr = tr[(tr["subtype"] == "adj_ready") & (tr["date"] <= (y - 1) * 10000 + 1231)]
        te = B.segment(ev, y * 10000 + 101, y * 10000 + 1231, y * 10000 + 1231)
        te = te[te["subtype"] == "adj_ready"]
        if not len(tr) or not len(te):
            continue
        clf = S.fit_clf(tr, feats, "1.2")
        raw = S.predict_long(clf, te, feats, "1.2")
        evp = B.EVParams(tr, "1.2")
        raw["EV"] = evp.ev(raw)
        sel = B.event_select(raw)
        out.append(sel.groupby("date", sort=False).agg(
            ev_n=("event_id", "nunique"), ev_mean=("EV", "mean"),
            ev_q10=("EV", lambda s: float(s.quantile(0.1))),
            pstop_mean=("p_STOP", "mean")).reset_index())
        log(f"  A2 rolling {y}: train={len(tr)} pred={len(sel)}")
    return pd.concat(out, ignore_index=True)


def main():
    t0 = time.time()
    df = load_data()
    df = add_features(df)
    day = build_day_frame(df)
    pday, pool = pool_daily()
    log(f"day={len(day)} pday={len(pday)}")

    # ---- A2 EV 分布（滚动，无泄漏）----
    evdist = a2_rolling_ev(pool)
    day = day.merge(evdist, on="date", how="left")
    for w in (5, 10, 20):
        pday[f"trail{w}_pool_ret"] = pday["pool_ret"].rolling(w).mean().shift(1)
        pday[f"trail{w}_stop"] = pday["pool_stop"].rolling(w).mean().shift(1)
    pday["trail5_win"] = pday["pool_win"].rolling(5).mean().shift(1)
    day = day.merge(pday[["date", "pool_cnt", "trail5_pool_ret", "trail5_stop",
                          "trail5_win", "trail10_pool_ret", "trail20_pool_ret",
                          "trail10_stop", "trail20_stop"]], on="date", how="left")
    # 交互：池状态 × 广度
    day["ix_pool_breadth"] = day["trail5_pool_ret"] * day["breadth_up"]
    day["ix_stop_breadth"] = day["trail5_stop"] * day["breadth_up"]
    # 更长滞后
    day["breadth_up_lag5"] = day["breadth_up"].shift(5)
    day["med_ret_lag5"] = day["med_ret"].shift(5)

    # ---- 标签：t+1..t+5 候选池加权表现，分位从训练期冻结 ----
    dates = day["date"].values
    fwd_ret = np.full(len(day), np.nan)
    fwd_stop = np.full(len(day), np.nan)
    pday = pday.set_index("date")
    for i, d in enumerate(dates):
        nxt = dates[i + 1:i + 6]
        if not len(nxt):
            continue
        sub = pday.reindex(nxt).dropna(subset=["pool_ret"])
        if len(sub) >= 2:
            w = sub["pool_cnt"].fillna(0)
            fwd_ret[i] = np.average(sub["pool_ret"], weights=w + 1)
            fwd_stop[i] = np.average(sub["pool_stop"], weights=w + 1)
    day["fwd_pool_ret"] = fwd_ret
    tr_mask = (day["date"] >= 20190101) & (day["date"] <= 20211231)
    q30, q70 = day.loc[tr_mask, "fwd_pool_ret"].quantile([0.3, 0.7])
    day["y"] = np.where(day["fwd_pool_ret"] <= q30, "UNSI",
                        np.where(day["fwd_pool_ret"] >= q70, "SUIT", "NEUT"))
    log(f"训练期标签分位冻结: q30={q30:+.4f} q70={q70:+.4f}; "
        f"分布 {day.loc[tr_mask, 'y'].value_counts().to_dict()}")

    feats = ["breadth_up", "med_ret", "loss_ratio", "big_loss_ratio",
             "limit_up_cnt", "limit_down_cnt", "zhaban_cnt", "amt_ratio20",
             "sh_ret5", "sh_ret20", "cyb_ret5", "cyb_ret20", "mkt_vol20",
             "board_disp", "strong_board_cnt", "watch_board_mean",
             "watch_lu", "watch_ld", "pool_cnt", "trail5_pool_ret",
             "trail5_stop", "trail5_win", "trail10_pool_ret", "trail20_pool_ret",
             "trail10_stop", "trail20_stop", "ev_n", "ev_mean", "ev_q10",
             "pstop_mean", "ix_pool_breadth", "ix_stop_breadth",
             "breadth_up_lag5", "med_ret_lag5"]
    d = day.dropna(subset=["fwd_pool_ret"]).copy()
    for c in feats:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    # 5日 purge/embargo：训练段尾部5个交易日剔除
    tr_days = sorted(d.loc[d["date"] <= 20211231, "date"])
    embargo_from = tr_days[-5]
    tr = d[(d["date"] < embargo_from)]
    oof = d[(d["date"] >= 20220101) & (d["date"] <= 20231231)]
    log(f"train={len(tr)}(purge≥{embargo_from}) oof={len(oof)}")

    from sklearn.metrics import brier_score_loss, precision_recall_curve, auc
    mdl = lgb.LGBMClassifier(**LGB_C, n_estimators=800)
    mdl.fit(tr[feats], tr["y"])
    oof_p = mdl.predict_proba(oof[feats])
    for i, cl in enumerate(mdl.classes_):
        oof[f"p_{cl}"] = oof_p[:, i]

    # ---- OOF 评估 ----
    y_bin = (oof["y"] == "UNSI").astype(float)
    prec, rec, _ = precision_recall_curve(y_bin, oof["p_UNSI"])
    pr_auc = auc(rec, prec)
    brier = brier_score_loss(y_bin, oof["p_UNSI"])
    # 危险日召回：按 OOF p_UNSI 最差30%作R0
    r0_thr30 = float(oof["p_UNSI"].quantile(0.70))
    sel30 = oof["p_UNSI"] >= r0_thr30
    recall30 = float(y_bin[sel30].sum() / max(y_bin.sum(), 1))
    log(f"OOF: PR-AUC(UNSI)={pr_auc:.3f} Brier={brier:.4f} "
        f"R0最差30%召回={recall30:.1%}")

    # ---- 第3步：R0硬切三档（OOF冻结）----
    ev = B.load_event_level()
    feats_all, _ = B.select_features(ev)
    feats_a2 = [f for f in feats_all.columns if f not in S.MKT_FEATS]
    tr_all = B.segment(ev, 20190101, 20231231, 20231231)
    tr_st = tr_all[tr_all["subtype"] == "adj_ready"]
    tr_fit = tr_st[tr_st["date"] <= 20211231]
    oof_a = tr_st[(tr_st["date"] >= 20220101) & (tr_st["date"] <= 20231231)]
    clf_a = S.fit_clf(tr_fit, feats_a2, "1.2")
    oof_long = S.predict_long(clf_a, oof_a, feats_a2, "1.2")
    evp = B.EVParams(tr_fit, "1.2")
    oof_long["EV"] = evp.ev(oof_long)
    cal = S.fit_isotonic_by(oof_long)
    oof_cal = S.apply_iso(oof_long, cal)
    oof_cal["EV"] = evp.ev(oof_cal)
    thr_ev = float(oof_cal["EV"].quantile(0.9))

    r0_thrs = {f"R0-bot{p}": float(oof["p_UNSI"].quantile(1 - p / 100))
               for p in (10, 20, 30)}
    full_pred = mdl.predict_proba(d[feats])
    for i, cl in enumerate(mdl.classes_):
        d[f"p_{cl}"] = full_pred[:, i]
    p_unsi = d.set_index("date")["p_UNSI"]

    WINDOWS = {"2024": (20240101, 20241231), "2025": (20250101, 20251231)}
    rows = []
    for seg, tag in [(B.segment(ev, 20240101, 20241231, 20241231), "2024"),
                     (B.segment(ev, 20250101, 20251231, 20251231), "2025")]:
        seg_adj = seg[seg["subtype"] == "adj_ready"]
        raw = S.predict_long(clf_a, seg_adj, feats_a2, "1.2")
        raw["EV"] = evp.ev(raw)
        cal_seg = S.apply_iso(raw, cal)
        cal_seg["EV"] = evp.ev(cal_seg)
        evc = B.event_select(cal_seg)
        # G0 基准
        pf0 = B.portfolio_backtest(evc, ks="1.2", ev_thr=thr_ev, window=WINDOWS[tag])
        base_pnl = pf0.get("_trades")
        rows.append({"ver": "G0", "seg": tag,
                     "ann": pf0.get("ann_ret"), "mdd": pf0.get("max_dd"),
                     "n": pf0.get("n_trades")})
        for name, th in r0_thrs.items():
            r0_days = set(p_unsi[p_unsi >= th].index)
            blocked = evc[evc["date"].isin(r0_days)]
            pf = B.portfolio_backtest(evc[~evc["date"].isin(r0_days)], ks="1.2",
                                      ev_thr=thr_ev, window=WINDOWS[tag])
            # 反事实口径（评审修正）：R0 是整日门——被拦日全部候选移除后当日无候选可
            # 递补，故"移除候选"与"当日禁开新仓"两种实现逐单一致（已对账验证），
            # R0 版本账户即"取消 G0 当日成交、不递补"的反事实账户。
            # 账户级反事实贡献 = G0总收益 - R0版本总收益（即 G0 在 R0 日实际亏掉的钱）；
            # blocked_pnl_sum（全部被拦候选净收益合计）含未成交候选，不代表反事实，仅参考。
            kill_acct = pf0.get("total_ret", 0.0) - pf.get("total_ret", 0.0)
            killed = blocked["net_ret"].sum() if len(blocked) else 0.0
            r0_in_seg = sorted(d for d in r0_days
                               if WINDOWS[tag][0] <= d <= WINDOWS[tag][1])
            rows.append({"ver": name, "seg": tag, "ann": pf.get("ann_ret"),
                         "mdd": pf.get("max_dd"), "n": pf.get("n_trades"),
                         "r0_days": len(r0_in_seg),
                         "blocked_pnl_sum": float(killed),
                         "blocked_n": len(blocked),
                         "kill_acct": float(kill_acct)})
            log(f"[{name}|{tag}] ann={pf.get('ann_ret'):+.2%} "
                f"mdd={pf.get('max_dd'):+.2%} r0days={len(r0_in_seg)} "
                f"blocked_n={len(blocked)} killed_pnl={killed:+.3f} "
                f"kill_acct={kill_acct:+.3f}")
    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    write_report(pr_auc, brier, recall30, r0_thrs, rdf)
    log(f"done {time.time()-t0:.0f}s")


def write_report(pr_auc, brier, recall30, r0_thrs, rdf):
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n## 16. ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n## 16. 市场危险日分类 + R0硬切（第2/3步）\n",
         "市场模型v2：三分类 UNSUITABLE/NEUTRAL/SUITABLE，标签分位从训练期2019-2021冻结"
         "（fwd5候选池加权净收益后30%=UNSI）；特征=广度/涨跌停/炸板/量能/指数趋势/波动/"
         "板块离散度/候选池5-10-20日滞后 + A2滚动OOF的EV分布(仅≤Y-1数据) + 池×广度交互；"
         "5日purge/embargo（训练段尾5日剔除）。训练19-21、OOF22-23冻结R0分位。"
         "2024已被多轮观察，本节为新假设。\n",
         f"- OOF：PR-AUC(UNSI)={pr_auc:.3f}，Brier={brier:.4f}，"
         f"最差30%档危险日召回={recall30:.1%}\n",
         f"- R0硬切阈值（OOF冻结）：{ {k: round(v, 3) for k, v in r0_thrs.items()} }\n",
         "| 版本 | 段 | 年化 | 回撤 | 交易数 | R0天数 | 被拦事件数 | 被拦池净收益合计(参考) | R0账户反事实贡献 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def _i(x):
        return "—" if pd.isna(x) else str(int(x))
    for _, r in rdf.iterrows():
        bp = r.get("blocked_pnl_sum")
        bps = f"{bp:+.3f}" if pd.notna(bp) else "—"
        ka = r.get("kill_acct")
        kas = f"{ka:+.3f}" if pd.notna(ka) else "—"
        L.append(f"| {r['ver']} | {r['seg']} | {r['ann']:+.2%} | {r['mdd']:+.2%} | "
                 f"{_i(r['n'])} | {_i(r.get('r0_days'))} | "
                 f"{_i(r.get('blocked_n'))} | {bps} | {kas} |")
    L.append("\n- 判读口径（评审修正）：**被拦池净收益合计不是反事实**——其中含大量本就不会"
             "成交的候选（低于阈值/超槽位），不代表 R0 真实价值。「R0账户反事实贡献」"
             "= G0总收益 − R0版本总收益，即 G0 在被拦日实际成交交易所亏/赚的钱："
             "为负=R0 拦的是真实亏损（门控有价值）；为正=错杀真实盈利。")
    L.append("- 实现注记：R0 是整日门（被拦日候选全部移除后当日无候选可递补），"
             "故「移除候选」与「当日禁开新仓(portfolio_backtest no_buy_dates)」"
             "两种实现逐单一致、已对账验证；候选级门控不适用此结论。")
    L.append("- 2024多轮观察，R0三档只按OOF冻结，未用2024寻优。")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
