# -*- coding: utf-8 -*-
"""第1步：板块日级软确认（替代 7 条件 AND，修 G5 样本坍缩）。

sector_gate_score（每个板块每天一条）=
  相对强度 + 上涨广度 + 涨停梯队 + 龙头状态 + 中军状态
  - 跌停扩散 - 冲高回落风险
实现：7 个成分按日横截面 rank→z 后等权求和（先验固定权重，不用2024调参）；
单成分缺失记 0（中性），当日有效成分 <4 的板块日不评分。
事件使用规则层指定的信号板块（sector_name），不事后挑板块。
无板块数据的事件：不判失败，绕过板块门（作为单独特征记录）。

通过率档：2022-2023 OOF 冻结 score 分位 q60/q70/q80（=Top40/30/20% 通过）。
2024 只评估；板块成分仍是当前快照，有幸存者偏差（报告持续标注）。
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
sys.path.insert(0, OUT_DIR)

import train_lgbm_baseline as B  # noqa: E402
import train_a_state as S       # noqa: E402

REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
EVENTS = os.path.join(OUT_DIR, "events_train_v1.parquet")
SCORES_OUT = os.path.join(OUT_DIR, "sector_day_scores.parquet")
THR_OUT = os.path.join(OUT_DIR, "sector_gate_thresholds.json")
K_A = "1.2"

# (列名, 符号, 方向说明)
COMPONENTS = [
    ("sector_excess_ret_5d", +1, "相对强度"),
    ("sector_up_ratio", +1, "上涨广度"),
    ("sector_limit_up_count", +1, "涨停梯队"),
    ("leader_high_to_close", -1, "龙头冲高回落(取负)"),
    ("center_ret1", +1, "中军状态"),
    ("sector_limit_down_count", -1, "跌停扩散(取负)"),
    ("sector_high_to_close", -1, "冲高回落风险(取负)"),
]
MIN_COMP = 4


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_scores():
    cols = ["event_id", "date", "code", "sector_name", "in_watch_sector"] \
        + [c for c, _, _ in COMPONENTS]
    ev = pd.read_parquet(EVENTS, columns=cols)
    have = ev[ev["sector_name"].notna()].copy()
    # 板块日级：成分在板块内相同，取首行
    sd = have.groupby(["sector_name", "date"], sort=False).first().reset_index()
    sd["n_comp"] = 0
    zsum = np.zeros(len(sd))
    for c, sign, _ in COMPONENTS:
        r = sd.groupby("date")[c].rank(pct=True)  # 日横截面分位
        z = norm.ppf(r.clip(1e-4, 1 - 1e-4)) * sign
        ok = sd[c].notna()
        sd.loc[~ok, c + "_z"] = 0.0
        sd.loc[ok, c + "_z"] = z[ok]
        sd["n_comp"] += ok.astype(int)
    zcols = [c + "_z" for c, _, _ in COMPONENTS]
    sd["sector_gate_score"] = np.where(
        sd["n_comp"] >= MIN_COMP, sd[zcols].sum(axis=1) / MIN_COMP, np.nan)
    out = sd[["sector_name", "date", "sector_gate_score", "n_comp"]]
    out.to_parquet(SCORES_OUT, index=False)
    log(f"sector-days scored: {len(out)} "
        f"(未评分 {int(out['sector_gate_score'].isna().sum())})")
    return out


def pl_ratio(t):
    if t is None or not len(t):
        return np.nan
    p = t["pnl"]
    return float(p[p > 0].mean() / abs(p[p < 0].mean())) \
        if (p > 0).any() and (p < 0).any() else np.nan


def m_of(pf):
    if not pf:
        return {}
    t = pf.get("_trades")
    tot = float(t["pnl"].sum()) if t is not None and len(t) else 0.0
    return {"total": pf.get("total_ret", np.nan), "ann": pf.get("ann_ret", np.nan),
            "mdd": pf.get("max_dd", np.nan), "n": pf.get("n_trades", 0),
            "pos": pf.get("avg_pos", np.nan), "util": pf.get("util", np.nan),
            "win": pf.get("trade_win", np.nan), "plr": pl_ratio(t),
            "top10share": float(t["pnl"].nlargest(10).sum() / tot) if tot else np.nan}


def main():
    t0 = time.time()
    sd = build_scores()
    oof_sd = sd[(sd["date"] >= 20220101) & (sd["date"] <= 20231231)]
    thrs = {f"S{int((1-q)*100)}": float(oof_sd["sector_gate_score"].quantile(q))
            for q in (0.60, 0.70, 0.80)}
    import json
    with open(THR_OUT, "w", encoding="utf-8") as f:
        json.dump({"quantiles_oof22_23": thrs,
                   "note": "q60/q70/q80 -> Top40/30/20% 通过"}, f, ensure_ascii=False)
    log(f"OOF 冻结板块门阈值 {thrs}")

    # ---- A2（协议同前）----
    ev = B.load_event_level()
    feats_all, _ = B.select_features(ev)
    feats = [f for f in feats_all.columns if f not in S.MKT_FEATS]
    tr = B.segment(ev, 20190101, 20231231, 20231231)
    va = B.segment(ev, 20240101, 20241231, 20241231)
    t5 = B.segment(ev, 20250101, 20251231, 20251231)
    tr_st = tr[tr["subtype"] == "adj_ready"]
    tr_fit = tr_st[tr_st["date"] <= 20211231]
    oof = tr_st[(tr_st["date"] >= 20220101) & (tr_st["date"] <= 20231231)]
    clf = S.fit_clf(tr_fit, feats, K_A)
    oof_long = S.predict_long(clf, oof, feats, K_A)
    evp = B.EVParams(tr_fit, K_A)
    oof_long["EV"] = evp.ev(oof_long)
    cal = S.fit_isotonic_by(oof_long)
    oof_cal = S.apply_iso(oof_long, cal)
    oof_cal["EV"] = evp.ev(oof_cal)
    thr_ev = float(oof_cal["EV"].quantile(0.9))

    score_map = sd.dropna(subset=["sector_gate_score"]) \
        .set_index(["sector_name", "date"])["sector_gate_score"]
    # 事件板块名映射
    ev_sec = pd.read_parquet(EVENTS, columns=["event_id", "sector_name"])
    ev_sec = ev_sec.drop_duplicates("event_id").set_index("event_id")["sector_name"]

    WINDOWS = {"2024": (20240101, 20241231), "2025": (20250101, 20251231)}
    rows = []
    for seg, tag in [(va, "2024"), (t5, "2025")]:
        seg_adj = seg[seg["subtype"] == "adj_ready"]
        raw = S.predict_long(clf, seg_adj, feats, K_A)
        raw["EV"] = evp.ev(raw)
        cal_seg = S.apply_iso(raw, cal)
        cal_seg["EV"] = evp.ev(cal_seg)
        evc = B.event_select(cal_seg)
        evc["sector_name"] = evc["event_id"].map(ev_sec)
        evc["sg_score"] = [score_map.get((s, d), np.nan)
                           for s, d in zip(evc["sector_name"], evc["date"])]
        no_data = evc["sg_score"].isna()
        pf_g0 = B.portfolio_backtest(evc, ks=K_A, ev_thr=thr_ev,
                                     window=WINDOWS[tag])
        r = {"gate": "G0", "seg": tag, **m_of(pf_g0)}
        rows.append(r)
        log(f"[G0|{tag}] ann={r['ann']:+.2%} n={r['n']}")
        for name, th in thrs.items():
            keep = no_data | (evc["sg_score"] >= th)
            pf = B.portfolio_backtest(evc[keep], ks=K_A, ev_thr=thr_ev,
                                      window=WINDOWS[tag])
            r = {"gate": f"G5-{name}", "seg": tag, **m_of(pf),
                 "pass_rate": float(keep.mean()),
                 "n_total": len(evc)}
            rows.append(r)
            log(f"[G5-{name}|{tag}] ann={r['ann']:+.2%} n={r['n']} "
                f"pass={r['pass_rate']:.1%}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "sector_gate_results.csv"),
              index=False, encoding="utf-8-sig")
    write_report(df, thrs)
    log(f"done {time.time()-t0:.0f}s")


def write_report(df, thrs):
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n## 15. ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n## 15. 板块日级软确认（第1步，替代7条件AND）\n",
         "score=7成分日横截面rank-z等权和（先验固定权重）：相对强度+上涨广度+涨停梯队"
         "+龙头(负向)+中军-跌停扩散-冲高回落(负向)；单成分缺失记0中性，当日有效成分<4"
         "不评分；每板块每天一条记录，事件用规则层指定信号板块；无板块数据事件不判失败、"
         "绕过板块门。通过率档由 OOF2022-2023 冻结（q60/q70/q80=Top40/30/20%通过）。"
         "**板块成分为当前快照，有幸存者偏差。** 2024已被多轮观察，本节为新假设。\n",
         f"- OOF 冻结阈值：{ {k: round(v, 3) for k, v in thrs.items()} }\n",
         "| 版本 | 段 | 通过率 | 总收益 | 年化 | 回撤 | 交易数 | 日均仓位 | 利用率 | 胜率 | top10笔贡献 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        pr = f"{r.get('pass_rate', np.nan):.1%}" if pd.notna(r.get("pass_rate")) else "—"
        L.append(f"| {r['gate']} | {r['seg']} | {pr} | {r['total']:+.2%} | "
                 f"{r['ann']:+.2%} | {r['mdd']:+.2%} | {int(r['n'])} | "
                 f"{r['pos']:.1f} | {r['util']:.1%} | {r['win']:.1%} | "
                 f"{r['top10share']:+.1%} |")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
