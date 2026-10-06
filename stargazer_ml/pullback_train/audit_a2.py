# -*- coding: utf-8 -*-
"""A2 专项核查（阶段5前置）：2024/2025 分别输出
- 实际交易笔数、日均及最高仓位
- 月度收益（真账本 NAV 月末口径）
- 前10笔交易贡献占比、前5板块贡献占比
- 去掉最佳月份后的年化、去掉最佳10笔后的年化
- 阈值附近样本（±10%带）收益稳定性
- EV 阈值 ±10% 敏感性
口径与阶段3一致：A2 特征（去市场指数）、训练2019-2021、OOF2022-2023 校准+冻结阈值。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
sys.path.insert(0, OUT_DIR)

import train_lgbm_baseline as B  # noqa: E402
import train_a_state as S       # noqa: E402  (复用 fit_clf/predict_long/校准工具)

REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
ST = "adj_ready"
KS = S.FREEZE_K[ST]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def monthly_nav(nav, dates):
    s = pd.Series(nav, index=pd.to_datetime(dates, format="%Y%m%d"))
    mlast = s.resample("ME").last()
    mfirst = s.resample("ME").first()
    prev = s.iloc[0]
    rows = []
    for ym, last in mlast.items():
        base = prev
        rows.append((int(ym.strftime("%Y%m")), float(last / base - 1)))
        prev = last
    return rows


def audit_seg(clf, feats, seg_ev, evp, cal, thr, code2sector):
    raw = S.predict_long(clf, seg_ev, feats, KS)
    raw["EV"] = evp.ev(raw)
    evd = B.event_select(raw)
    cal_seg = S.apply_iso(raw, cal)
    cal_seg["EV"] = evp.ev(cal_seg)
    evc = B.event_select(cal_seg)

    pf = B.portfolio_backtest(evc, ks=KS, ev_thr=thr)
    r = {"n": len(evd), "thr": thr,
         "n_trades": pf.get("n_trades", 0),
         "avg_pos": pf.get("avg_pos", np.nan),
         "max_pos": pf.get("max_pos_held", np.nan),
         "ann": pf.get("ann_ret", np.nan),
         "mdd": pf.get("max_dd", np.nan),
         "win": pf.get("trade_win", np.nan)}
    if not pf or "_trades" not in pf or not len(pf["_trades"]):
        return r, evc
    t = pf["_trades"].copy()
    nav, dates = pf["_nav"], pf["_dates"]
    # 月度收益
    mon = monthly_nav(nav, dates)
    r["monthly"] = mon
    best_m = max(mon, key=lambda x: x[1])
    r["best_month"] = best_m
    # 去掉最佳月份：月收益序列剔除最佳月后复合（按21交易日/月近似年化）
    rest = [m for m in mon if m[0] != best_m[0]]
    cum_rest = float(np.prod([1 + m[1] for m in rest]) - 1)
    r["ann_ex_best_month"] = (1 + cum_rest) ** (252 / max(len(nav), 1)) - 1 \
        if rest else np.nan
    # 前10笔贡献
    tot_pnl = float(t["pnl"].sum())
    r["top10_trade_share"] = float(
        t["pnl"].nlargest(10).sum() / tot_pnl) if tot_pnl else np.nan
    t["sector"] = t["code"].map(code2sector)
    bsum = t.groupby("sector")["pnl"].sum()
    r["top5_sector_share"] = float(
        bsum.nlargest(5).sum() / tot_pnl) if tot_pnl else np.nan
    r["top5_sectors"] = bsum.nlargest(5).index.tolist()
    # 去掉最佳10笔
    adj = nav[-1] - t["pnl"].nlargest(10).sum()
    r["ann_ex_best10"] = float(adj ** (252 / len(nav)) - 1)
    # 阈值附近样本稳定性（事件级 ±10% 带）
    band = evc["EV"].abs().max() * 0.0 + thr  # noqa
    lo, hi = thr * 0.9, thr * 1.1
    near = evc[(evc["EV"] >= lo) & (evc["EV"] <= hi)]
    above = evc[evc["EV"] > hi]
    r["near_band_n"] = len(near)
    r["near_band_ret"] = float(near["net_ret"].mean()) if len(near) else np.nan
    r["above_ret"] = float(above["net_ret"].mean()) if len(above) else np.nan
    # 阈值 ±10% 敏感性
    sens = {}
    for tag2, th2 in [("-10%", thr * 0.9), ("+10%", thr * 1.1)]:
        p2 = B.portfolio_backtest(evc, ks=KS, ev_thr=th2)
        sens[tag2] = (p2.get("n_trades", 0), p2.get("ann_ret", np.nan))
    r["sens"] = sens
    return r, evc


def main():
    t0 = time.time()
    ev = B.load_event_level()
    feats_all, _ = B.select_features(ev)
    feats = [f for f in feats_all.columns if f not in S.MKT_FEATS]
    tr = B.segment(ev, 20190101, 20231231, 20231231)
    va = B.segment(ev, 20240101, 20241231, 20241231)
    t5 = B.segment(ev, 20250101, 20251231, 20251231)
    tr_st = tr[tr["subtype"] == ST]
    tr_fit = tr_st[tr_st["date"] <= 20211231]
    oof = tr_st[(tr_st["date"] >= 20220101) & (tr_st["date"] <= 20231231)]
    log(f"feats={len(feats)} tr_fit={len(tr_fit)} oof={len(oof)}")

    clf = S.fit_clf(tr_fit, feats, KS)
    oof_long = S.predict_long(clf, oof, feats, KS)
    evp = B.EVParams(tr_fit, KS)
    oof_long["EV"] = evp.ev(oof_long)
    cal = S.fit_isotonic_by(oof_long)
    oof_cal = S.apply_iso(oof_long, cal)
    oof_cal["EV"] = evp.ev(oof_cal)
    thr = float(oof_cal["EV"].quantile(0.9))
    log(f"frozen thr={thr:+.4%}")

    ev_full = pd.read_parquet(os.path.join(OUT_DIR, "events_train_v1.parquet"),
                              columns=["code", "sector_name"]) \
        .drop_duplicates("code").set_index("code")["sector_name"]
    res = {}
    evc_store = {}
    for seg_ev, tag in [(va, "2024"), (t5, "2025")]:
        seg_st = seg_ev[seg_ev["subtype"] == ST]
        r, evc = audit_seg(clf, feats, seg_st, evp, cal, thr, ev_full)
        res[tag] = r
        evc_store[tag] = evc
        log(f"[{tag}] n={r['n']} trades={r['n_trades']} avg_pos={r['avg_pos']:.1f} "
            f"max_pos={r['max_pos']} ann={r['ann']:+.2%} ex_best_month="
            f"{r.get('ann_ex_best_month', np.nan):+.2%} ex_best10="
            f"{r.get('ann_ex_best10', np.nan):+.2%}")
    pd.to_pickle({"res": res, "thr": thr}, os.path.join(OUT_DIR, "a2_audit.pkl"))

    # ---- 报告 ## 11 ----
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n## 11. ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n## 11. A2 专项核查（阶段5前置）\n",
         "口径：A2 特征（去市场指数），adj_ready，K=1.2；训练2019-2021、OOF2022-2023 "
         "isotonic 校准 + 冻结 EV 阈值（校准后90分位）。真账本允许空仓。年化剔除项均为近似口径。\n"]
    for tag in ["2024", "2025"]:
        r = res[tag]
        L.append(f"### {tag}（阈值 {r['thr']:+.3%}）\n")
        L.append(f"- 事件 {r['n']}，实际交易 {r['n_trades']} 笔，日均仓位 "
                 f"{r['avg_pos']:.1f}，最高仓位 {r['max_pos']}，年化 {r['ann']:+.2%}，"
                 f"回撤 {r['mdd']:+.2%}，胜率 {r['win']:.1%}")
        mon = r.get("monthly", [])
        if mon:
            L.append("- 月度收益: " + ", ".join(f"{m}: {v:+.2%}" for m, v in mon))
            L.append(f"- 最佳月 {r['best_month'][0]}（{r['best_month'][1]:+.2%}）；"
                     f"去掉最佳月年化 {r['ann_ex_best_month']:+.2%}")
        if "top10_trade_share" in r:
            L.append(f"- 前10笔贡献占比 {r['top10_trade_share']:+.1%}；去掉最佳10笔年化 "
                     f"{r['ann_ex_best10']:+.2%}")
            L.append(f"- 前5板块贡献占比 {r['top5_sector_share']:+.1%} "
                     f"({', '.join(map(str, r['top5_sectors']))})")
            L.append(f"- 阈值±10%带样本 n={r['near_band_n']}，带内均收益 "
                     f"{r['near_band_ret']:+.2%}，带上方均收益 {r['above_ret']:+.2%}（应单调递减）")
            s = r["sens"]
            L.append(f"- 阈值敏感性: -10% → {s['-10%'][0]}笔/{s['-10%'][1]:+.2%}；"
                     f"+10% → {s['+10%'][0]}笔/{s['+10%'][1]:+.2%}")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
