# -*- coding: utf-8 -*-
"""2024减亏实验 = 市场参与/空仓模型实验（用户冻结设计 G0–G7）。

不改 A2 选股特征。A2 只负责"同一天里哪只更好"；本实验回答"今天该不该做、做几仓"。
G0 A2原始90分位 / G1 92.5 / G2 95 / G3 97.5（全为 OOF2022-2023 分位）
G4 EV>0
G5 90分位 + 板块确认（准入层，冻结启发式，不用2024调参）
G6 90分位 + 市场参与模型槽位（market_slots.csv，OOF冻结分位）
G7 G6+G5 全部
验收（两级）：研究成功=2024亏损较G0(-25.34%)减≥50% + 回撤降≥30% +
2025保留≥70%(≥+18%) + 不依赖少数交易/单一板块 + 共同池日IC不降；
可执行候选=2024不亏+2025正+两年合正+回撤可接受。
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
sys.path.insert(0, OUT_DIR)

import train_lgbm_baseline as B  # noqa: E402
import train_a_state as S       # noqa: E402
from contracts import CandidateAssessment, GateDecision  # noqa: E402

REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
K_A = "1.2"
MODEL_VERSION = "a2_lgbm_k1.2@v3.3"

SECTOR_RULE_COLS = ["sector_excess_ret_5d", "sector_up_ratio",
                    "sector_limit_up_count", "sector_limit_down_count",
                    "leader_high_to_close", "center_ret1", "sector_high_to_close"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sector_confirm(ev):
    """板块确认（冻结启发式，准入层）。数据缺失=不通过。规则全部来自领域定义，
    未用2024/2025调参。"""
    ok = ((ev["sector_excess_ret_5d"] > 0)
          & (ev["sector_up_ratio"] >= 0.5)
          & (ev["sector_limit_up_count"] >= 1)
          & (ev["sector_limit_down_count"] <= 2)
          & (ev["leader_high_to_close"] <= 0.05)
          & (ev["center_ret1"] >= -0.03)
          & (ev["sector_high_to_close"] <= 0.06))
    return ok.fillna(False)


def pl_ratio(t):
    if t is None or not len(t):
        return np.nan
    p = t["pnl"]
    return float(p[p > 0].mean() / abs(p[p < 0].mean())) \
        if (p > 0).any() and (p < 0).any() else np.nan


def metrics(pf):
    if not pf:
        return {"ann": np.nan, "mdd": np.nan, "n": 0, "pos": np.nan,
                "win": np.nan, "plr": np.nan, "top10share": np.nan,
                "total": np.nan, "util": np.nan}
    t = pf.get("_trades")
    tot = float(t["pnl"].sum()) if t is not None and len(t) else 0.0
    top10 = float(t["pnl"].nlargest(10).sum() / tot) if tot else np.nan
    return {"ann": pf.get("ann_ret", np.nan), "mdd": pf.get("max_dd", np.nan),
            "n": pf.get("n_trades", 0), "pos": pf.get("avg_pos", np.nan),
            "win": pf.get("trade_win", np.nan), "plr": pl_ratio(t),
            "top10share": top10, "total": pf.get("total_ret", np.nan),
            "util": pf.get("util", np.nan)}


def main():
    t0 = time.time()
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
    qs = {g: float(oof_cal["EV"].quantile(q)) for g, q in
          [("G0", 0.90), ("G1", 0.925), ("G2", 0.95), ("G3", 0.975)]}
    log(f"OOF 冻结阈值 {qs}")

    slots = pd.read_csv(os.path.join(OUT_DIR, "market_slots.csv"))
    cap_of = slots.set_index("date")["slot_cap"]

    # 事件行板块确认需要全列（load_event_level 只带白名单，重读原始事件表补板块列）
    ev_full = pd.read_parquet(os.path.join(OUT_DIR, "events_train_v1.parquet"))
    conf_map = ev_full.drop_duplicates("event_id").set_index("event_id")
    conf_map = sector_confirm(conf_map[SECTOR_RULE_COLS])

    rows = []
    assessments = []
    WINDOWS = {"2024": (20240101, 20241231), "2025": (20250101, 20251231)}
    for seg, tag in [(va, "2024"), (t5, "2025")]:
        seg_adj = seg[seg["subtype"] == "adj_ready"]
        raw = S.predict_long(clf, seg_adj, feats, K_A)
        raw["EV"] = evp.ev(raw)
        cal_seg = S.apply_iso(raw, cal)
        cal_seg["EV"] = evp.ev(cal_seg)
        evc = B.event_select(cal_seg)
        evc["conf"] = evc["event_id"].map(conf_map).fillna(False)

        variants = {
            "G0": dict(thr=qs["G0"]),
            "G1": dict(thr=qs["G1"]),
            "G2": dict(thr=qs["G2"]),
            "G3": dict(thr=qs["G3"]),
            "G4": dict(thr=0.0),
            "G5": dict(thr=qs["G0"], conf=True),
            "G6": dict(thr=qs["G0"], slots=True),
            "G7": dict(thr=qs["G0"], conf=True, slots=True),
        }
        for g, kw in variants.items():
            src = evc[evc["conf"]] if kw.get("conf") else evc
            pf = B.portfolio_backtest(src, ks=K_A, ev_thr=kw["thr"],
                                      slot_cap=cap_of if kw.get("slots") else None,
                                      window=WINDOWS[tag])
            m = metrics(pf)
            m.update({"gate": g, "seg": tag})
            rows.append(m)
            log(f"[{g}|{tag}] total={m['total']:+.2%} ann={m['ann']:+.2%} "
                f"mdd={m['mdd']:+.2%} n={m['n']} pos={m['pos']:.1f} "
                f"util={m['util']:.1%} top10share={m['top10share']:+.1%}")
            # 结构化契约样例（Parquet，仅接口测试用，非训练数据源）
            if tag == "2025":
                for _, r in src.iterrows():
                    if r["EV"] < kw["thr"]:
                        continue
                    assessments.append({
                        "candidate_id": str(int(r["event_id"])),
                        "strategy_id": "A/adj_ready",
                        "model_version": MODEL_VERSION,
                        "opportunity_score": float(r["EV"]),
                        "p_target": float(r["p_TARGET"]),
                        "p_stop": float(r["p_STOP"]),
                        "p_timeout": float(r["p_TIMEOUT"]),
                        "expected_value": float(r["EV"]),
                        "evidence_ids": json.dumps([f"event:{int(r['event_id'])}"]),
                        "risk_flags": json.dumps(
                            [] if r["conf"] else ["sector_not_confirmed"]),
                    })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "gate_experiment_results.csv"),
              index=False, encoding="utf-8-sig")
    pd.DataFrame(assessments).to_parquet(
        os.path.join(OUT_DIR, "candidate_assessments_2025.parquet"), index=False)
    log(f"assessments dumped: {len(assessments)} (parquet)")
    write_report(df)
    log(f"done {time.time()-t0:.0f}s")


def write_report(df):
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n## 14. ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n## 14. 市场参与/空仓实验（G0–G7，2024减亏，v2 统一年度窗口）\n",
         "第0步修订：所有版本统一按完整自然年度计净值（2024/2025 全程，闲置现金收益0，"
         "期末强平未退出持仓），同一起始资金与交易日历；报总收益、年化、日均仓位与"
         "资金利用率（util=日均invested/equity），稀疏策略可直接比较。"
         "设计冻结：分位全部来自 OOF2022-2023；A2 选股特征未改；板块确认为准入层"
         "（相对强度>0、上涨占比≥50%、有涨停、跌停≤2、龙头非冲高回落、中军企稳、"
         "板块无长上影；数据缺失=不通过）；市场参与层=market_slots.csv（R0=0/R1=5/"
         "R2=10/R3=20）。两级验收：研究成功=2024亏损较G0减≥50%+回撤降≥30%+2025保留"
         "≥70%(≥+18%)+不依赖少数交易/板块+共同池IC不降；可执行=2024不亏+2025正+两年合正。\n",
         "| 版本 | 段 | 总收益 | 年化 | 回撤 | 交易数 | 日均仓位 | 利用率 | 胜率 | 盈亏比 | top10笔贡献 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        L.append(f"| {r['gate']} | {r['seg']} | {r['total']:+.2%} | {r['ann']:+.2%} | "
                 f"{r['mdd']:+.2%} | {int(r['n'])} | {r['pos']:.1f} | "
                 f"{r['util']:.1%} | {r['win']:.1%} | {r['plr']:.2f} | "
                 f"{r['top10share']:+.1%} |")
    g0 = df[(df.gate == "G0")].set_index("seg")
    for tag in ["2024", "2025"]:
        base = g0.loc[tag]
        L.append(f"\n### 验收检查（{tag}，基准 G0 年化 {base['ann']:+.2%} / 回撤 {base['mdd']:+.2%}）")
        for g in ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]:
            r = df[(df.gate == g) & (df.seg == tag)].iloc[0]
            if tag == "2024":
                loss_cut = (r["ann"] - base["ann"]) / abs(base["ann"])
                dd_cut = (abs(r["mdd"]) - abs(base["mdd"])) / abs(base["mdd"])
                L.append(f"- {g}: 亏损减少 {loss_cut:+.0%}（需≥50%），回撤变化 {dd_cut:+.0%}"
                         f"（需≤-30%），交易 {int(r['n'])} 笔，top10贡献 {r['top10share']:+.0%}")
            else:
                keep = r["ann"] / base["ann"] if base["ann"] else np.nan
                L.append(f"- {g}: 收益保留 {keep:.0%}（需≥70%），年化 {r['ann']:+.2%}")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
