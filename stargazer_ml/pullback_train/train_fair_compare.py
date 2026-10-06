# -*- coding: utf-8 -*-
"""阶段5：公平对比（统一真账本 / T+1 / 费用 / 20槽允许空仓 / 同标签引擎）。

版本：
  RULE   纯规则基线：A 规则候选(adj_ready+first_div)，无模型，按时间填仓
  LEGACY 旧绿盘低吸模型：desktop sector_pullback A模式规则+RR排序 历史复现
  A0     白名单全特征模型（adj_ready, K=1.2）
  A2     去市场指数特征模型（adj_ready, K=1.2）
  A2FD   A2 + first_div 风险过滤（剔除疑似首次分歧事件）
  B0     阶段4胜出 B 变体（K=1.5，独立分支）
  COMBO  A2 ∥ B0 并行组合（共享20槽，按EV统一排序）

口径1 共同候选池：adj_ready 规则池上比较 RULE/A0/A2/LEGACY 排序（日IC）
口径2 完整策略：各自候选+各自排序，真账本执行
协议：模型训练2019-2021、OOF2022-2023 isotonic校准+冻结EV阈值；2024确认、2025描述。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
sys.path.insert(0, OUT_DIR)

import train_lgbm_baseline as B  # noqa: E402
import train_a_state as S       # noqa: E402

REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
K_A = "1.2"
K_B = "1.5"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def reset_globals():
    B.PATH_MAP = None
    B.EVENT_QUOTE = None


def pl_ratio(t):
    if t is None or not len(t):
        return np.nan
    p = t["pnl"]
    return float(p[p > 0].mean() / abs(p[p < 0].mean())) \
        if (p > 0).any() and (p < 0).any() else np.nan


def pf_summary(pf):
    if not pf:
        return {}
    return {"ann": pf.get("ann_ret", np.nan), "mdd": pf.get("max_dd", np.nan),
            "n": pf.get("n_trades", 0), "pos": pf.get("avg_pos", np.nan),
            "win": pf.get("trade_win", np.nan),
            "plr": pl_ratio(pf.get("_trades"))}


def train_model(ev, feats, st, ks):
    """训练+OOF校准+冻结阈值。返回 (clf, cal, thr, evp)"""
    tr_all = B.segment(ev, 20190101, 20231231, 20231231)
    tr_st = tr_all[tr_all["subtype"] == st]
    tr_fit = tr_st[tr_st["date"] <= 20211231]
    oof = tr_st[(tr_st["date"] >= 20220101) & (tr_st["date"] <= 20231231)]
    clf = S.fit_clf(tr_fit, feats, ks)
    oof_long = S.predict_long(clf, oof, feats, ks)
    evp = B.EVParams(tr_fit, ks)
    oof_long["EV"] = evp.ev(oof_long)
    cal = S.fit_isotonic_by(oof_long)
    oof_cal = S.apply_iso(oof_long, cal)
    oof_cal["EV"] = evp.ev(oof_cal)
    thr = float(oof_cal["EV"].quantile(0.9))
    return clf, cal, thr, evp


def predict_evc(clf, feats, seg_ev, ks, evp, cal):
    raw = S.predict_long(clf, seg_ev, feats, ks)
    raw["EV"] = evp.ev(raw)
    cal_seg = S.apply_iso(raw, cal)
    cal_seg["EV"] = evp.ev(cal_seg)
    return B.event_select(cal_seg), B.event_select(raw)


# ---------------------------------------------------------------- first_div flag（事件行口径，legacy定义）
def fd_flag(ev):
    return ((ev["ret5"] >= 0.15)
            & (ev["_ret1_prev_min5"] > -0.03)
            & (((ev["high"] - ev["pre_close"]) / ev["pre_close"] - ev["ret1"]) >= 0.03)
            & (ev["amount_ratio20"] >= 1.0))


# ---------------------------------------------------------------- legacy score（事件行口径）
def legacy_score(ev):
    entry = ev["close"]
    risk_dist = 1.5 * ev["atr14"]
    risk_pct = risk_dist / entry + 0.002
    h5 = ev["high5"].where(ev["high5"] > entry, np.nan)
    h20 = ev["high20"].where(ev["high20"] > entry, np.nan)
    resist = pd.concat([h5, h20], axis=1).min(axis=1)
    target = resist.where(resist.notna(), entry + 2.0 * risk_dist)
    gain = (target - entry) / entry - 0.002
    rr = gain / risk_pct
    q = np.where(ev["ret20"] > 0, np.minimum(ev["ret20"], 0.5) / 0.5 * 0.4, 0.0)
    q += np.minimum(ev["limit60"].fillna(0), 5) / 5 * 0.4
    q += np.where((ev["float_mv"] >= 10e8) & (ev["float_mv"] <= 100e8), 0.2, 0.0)
    return rr + 0.5 * q, rr


def main():
    t0 = time.time()
    ev = B.load_event_level()
    feats_all, _ = B.select_features(ev)
    feats_a0 = list(feats_all.columns)
    feats_a2 = [f for f in feats_a0 if f not in S.MKT_FEATS]
    va = B.segment(ev, 20240101, 20241231, 20241231)
    t5 = B.segment(ev, 20250101, 20251231, 20251231)
    adj_va = va[va["subtype"] == "adj_ready"]
    adj_t5 = t5[t5["subtype"] == "adj_ready"]

    log("训练 A0 / A2 ...")
    clf0, cal0, thr0, evp0 = train_model(ev, feats_a0, "adj_ready", K_A)
    clf2, cal2, thr2, evp2 = train_model(ev, feats_a2, "adj_ready", K_A)
    log(f"thr A0={thr0:+.4%} A2={thr2:+.4%}")

    results = {}
    # ---------------- RULE / A0 / A2 / A2FD（共同 adj_ready 池执行）----------------
    reset_globals()
    for seg_adj, tag in [(adj_va, "2024"), (adj_t5, "2025")]:
        pool = seg_adj[seg_adj[f"eligible_s{K_A}"]]
        # RULE：无模型，按时间填仓
        evr = pool[["event_id", "date", "code", "subtype", "in_watch_sector",
                    f"outcome_s{K_A}_t2.0", f"net_ret_s{K_A}_t2.0"]].copy()
        evr.columns = ["event_id", "date", "code", "subtype", "in_watch_sector",
                       "y", "net_ret"]
        evr["target_atr"] = 2.0
        evr["EV"] = 0.0
        # A0 / A2 / A2FD
        _, evd0 = predict_evc(clf0, feats_a0, seg_adj, K_A, evp0, cal0)
        evc0, _ = predict_evc(clf0, feats_a0, seg_adj, K_A, evp0, cal0)
        evc2, evd2 = predict_evc(clf2, feats_a2, seg_adj, K_A, evp2, cal2)
        # A2FD：A2 剔除疑似首次分歧
        fdv = fd_flag(seg_adj.drop_duplicates("event_id").set_index("event_id"))
        drop_ids = fdv[fdv].index
        evc2fd = evc2[~evc2["event_id"].isin(drop_ids)]
        leg_score, leg_rr = legacy_score(pool)
        res = {}
        for name, evc, thr in [("RULE", evr, None), ("A0", evc0, thr0),
                               ("A2", evc2, thr2), ("A2FD", evc2fd, thr2)]:
            pf_f = B.portfolio_backtest(evc, ks=K_A)
            pf_t = B.portfolio_backtest(evc, ks=K_A, ev_thr=thr) if thr else {}
            dm = B.daily_metrics(evc)
            res[name] = {"n": len(evc), "dIC": dm.get("d_ic_mean", np.nan),
                         "dIC_t": dm.get("d_ic_t", np.nan),
                         "fill": pf_summary(pf_f), "thr": pf_summary(pf_t)}
        # 共同池排序比较（口径1）：日IC of A0/A2/legacy score on RULE pool
        pool_sc = pool.rename(columns={f"net_ret_s{K_A}_t2.0": "net_ret"}).copy()
        pool_sc["EV"] = 0.0
        ic_cmp = {}
        leg_s, _ = legacy_score(pool_sc)
        for nm, sc in [("A0", evd0.set_index("event_id")["EV"]),
                       ("A2", evd2.set_index("event_id")["EV"]),
                       ("LEGACY", pd.Series(leg_s.values, index=pool_sc["event_id"].values))]:
            p2 = pool_sc.copy()
            p2["EV"] = p2["event_id"].map(sc).fillna(0.0)
            ic_cmp[nm] = B.daily_metrics(p2).get("d_ic_mean", np.nan)
        res["IC_common"] = ic_cmp
        results[tag] = res
        log(f"[{tag}] " + " ".join(
            f"{k}: thr_ann={v['thr'].get('ann', np.nan):+.2%}({v['thr'].get('n', 0)})"
            if k != "IC_common" else f"IC_common={v}"
            for k, v in res.items()))

    # ---------------- LEGACY（独立候选池）----------------
    leg = pd.read_parquet(os.path.join(OUT_DIR, "events_legacy_cooled.parquet"))
    leg["in_watch_sector"] = True
    # legacy 风险上限7%（production口径）
    leg["eligible_s1.5"] = leg["stop_pct_s1.5"] <= 0.07
    B.PATH_MAP = pd.read_parquet(os.path.join(OUT_DIR, "events_path_legacy_cooled.parquet")).set_index("event_id")
    B.EVENT_QUOTE = leg.drop_duplicates("event_id").set_index("event_id")
    for seg, tag in [(leg[leg["date"].between(20240101, 20241231)], "2024"),
                     (leg[leg["date"].between(20250101, 20251231)], "2025")]:
        seg = seg[seg["eligible_s1.5"] & (~seg["is_censored"])]
        evl = seg[["event_id", "date", "code", "subtype", "in_watch_sector",
                   "outcome_s1.5_t2.0", "net_ret_s1.5_t2.0"]].copy()
        evl.columns = ["event_id", "date", "code", "subtype", "in_watch_sector",
                       "y", "net_ret"]
        evl["target_atr"] = 2.0
        evl["EV"] = evl["event_id"].map(leg.set_index("event_id")["score"])
        pf_f = B.portfolio_backtest(evl, ks=K_B)
        results[tag]["LEGACY"] = {"n": len(evl),
                                  "fill": pf_summary(pf_f), "thr": {}}
        log(f"[LEGACY|{tag}] n={len(evl)} fill_ann={pf_summary(pf_f).get('ann', np.nan):+.2%}")

    # ---------------- B0 + COMBO ----------------
    import train_b_ablation as BA
    b0 = BA.load_b("b0")
    Xb, feats_b, _ = BA.select_features_b(b0)
    trb = b0[(b0["date"] >= 20190101) & (b0["date"] <= 20211231) & (~b0["is_censored"])]
    oofb = b0[(b0["date"] >= 20220101) & (b0["date"] <= 20231231) & (~b0["is_censored"])]
    clfb = BA.fit_clf(trb, Xb, feats_b)
    oof_long = BA.predict_long(clfb, oofb, Xb, feats_b)
    evpb = B.EVParams(trb, K_B)
    oof_long["EV"] = evpb.ev(oof_long)
    calb = B.fit_isotonic(oof_long)
    oof_cal = B.apply_isotonic(oof_long, calb)
    oof_cal["EV"] = evpb.ev(oof_cal)
    thr_b = float(oof_cal["EV"].quantile(0.9))

    path_a = pd.read_parquet(os.path.join(OUT_DIR, "events_path_cooled.parquet"))
    quote_a = pd.read_parquet(os.path.join(OUT_DIR, "events_train_v1.parquet"),
                              columns=["event_id", "ret1", "_limit", "high", "close"]
                              ).drop_duplicates("event_id")
    path_b = pd.read_parquet(os.path.join(OUT_DIR, "events_path_b_b0_cooled.parquet"))
    quote_b = b0.drop_duplicates("event_id")[["event_id", "ret1", "_limit", "high", "close"]]
    off = int(path_a["event_id"].max()) + 1

    for seg_b, seg_adj, tag in [(b0[b0["date"].between(20240101, 20241231) & (~b0["is_censored"])], adj_va, "2024"),
                                (b0[b0["date"].between(20250101, 20251231) & (~b0["is_censored"])], adj_t5, "2025")]:
        raw_b = BA.predict_long(clfb, seg_b, Xb, feats_b)
        raw_b["EV"] = evpb.ev(raw_b)
        calb_seg = B.apply_isotonic(raw_b, calb)
        calb_seg["EV"] = evpb.ev(calb_seg)
        evc_b = B.event_select(calb_seg)
        # B0 独立账本
        B.PATH_MAP = path_b.set_index("event_id")
        B.EVENT_QUOTE = quote_b.set_index("event_id")
        pf_b = B.portfolio_backtest(evc_b, ks=K_B, ev_thr=thr_b)
        results[tag]["B0"] = {"n": len(evc_b), "thr": pf_summary(pf_b),
                              "fill": pf_summary(B.portfolio_backtest(evc_b, ks=K_B))}
        # COMBO：A2 ∥ B0（B 行限 eligible_s1.2：共享账本统一 K=1.2 口径）
        evc_a2, _ = predict_evc(clf2, feats_a2, seg_adj, K_A, evp2, cal2)
        b_ok = seg_b.drop_duplicates("event_id").set_index("event_id")["eligible_s1.2"]
        eb = evc_b[evc_b["event_id"].map(b_ok).fillna(False)].copy()
        eb["event_id"] = eb["event_id"] + off
        comb = pd.concat([evc_a2, eb], ignore_index=True)
        B.PATH_MAP = pd.concat([path_a, path_b.assign(event_id=path_b["event_id"] + off)], ignore_index=True).set_index("event_id")
        qb = quote_b.copy()
        qb["event_id"] = qb["event_id"] + off
        B.EVENT_QUOTE = pd.concat([quote_a, qb], ignore_index=True).drop_duplicates("event_id").set_index("event_id")
        pf_c = B.portfolio_backtest(comb, ks=K_A, ev_thr=min(thr2, thr_b))
        results[tag]["COMBO"] = {"n": len(comb), "thr": pf_summary(pf_c),
                                 "fill": pf_summary(B.portfolio_backtest(comb, ks=K_A))}
        log(f"[{tag}] B0 thr_ann={results[tag]['B0']['thr'].get('ann', np.nan):+.2%} "
            f"COMBO thr_ann={pf_summary(pf_c).get('ann', np.nan):+.2%}")

    reset_globals()
    pd.to_pickle(results, os.path.join(OUT_DIR, "fair_compare_results.pkl"))
    write_report(results)
    log(f"done in {time.time()-t0:.0f}s")


def fmt(x, pct=True):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "nan"
    return f"{x:+.2%}" if pct else f"{x}"


def write_report(results):
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n## 12. ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n## 12. 公平对比（阶段5，统一真账本）\n",
         "统一口径：close 入场、同一 barrier 标签引擎、T+1、费用/滑点一致、20槽、"
         "允许空仓（模型版用 OOF2022-2023 冻结 EV 阈值）；RULE/LEGACY 无 EV 阈值只报填仓。"
         "LEGACY=desktop sector_pullback A模式规则+RR排序复现（12关注板块成分内，"
         "有幸存者偏差；legacy风险上限7%）。A0/A2/A2FD=adj_ready K=1.2；B0 K=1.5；"
         "COMBO=A2∥B0 共享20槽按EV排序（阈值=min(A2,B0)）。\n",
         "| 版本 | 段 | n | 日IC(t) | 填仓年化 | 阈值年化 | 阈值交易数 | 日均仓位 | 回撤 | 胜率 | 盈亏比 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    order = ["RULE", "LEGACY", "A0", "A2", "A2FD", "B0", "COMBO"]
    for tag in ["2024", "2025"]:
        for name in order:
            r = results[tag].get(name)
            if not r:
                continue
            f_, t_ = r.get("fill", {}), r.get("thr", {})
            L.append(f"| {name} | {tag} | {r.get('n', 0)} | "
                     f"{r.get('dIC', np.nan):+.4f}({r.get('dIC_t', np.nan):+.1f}) | "
                     f"{fmt(f_.get('ann'))} | {fmt(t_.get('ann'))} | "
                     f"{t_.get('n', 0)} | {t_.get('pos', np.nan):.1f} | "
                     f"{fmt(t_.get('mdd'))} | {fmt(t_.get('win'))} | "
                     f"{t_.get('plr', np.nan):.2f} |")
        ic = results[tag].get("IC_common", {})
        if ic:
            L.append(f"\n- 共同候选池日IC（{tag}，adj_ready池）: "
                     + ", ".join(f"{k} {v:+.4f}" for k, v in ic.items()))
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
