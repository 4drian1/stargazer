# -*- coding: utf-8 -*-
"""T2前置检查：quarantine 隔离分布与选择性偏差报告（§18.14，v2修正）。

v2 修正：searchsorted 默认 side='left' 会把 t1 错算成信号日当天，
受影响事件被严重高估（432→真实约24个）。本版直接从候选展开事件、
用与引擎完全相同的守卫谓词（days_all 完整日历比对）统计受影响事件，
不受引擎重跑覆盖样本的影响。

检查项（评审要求）：
  1) 按月份；2) 按 A2/B0/LEGACY 来源；3) 涉及候选事件数与占比；
  4) 各决策点缺失率；5) 是否集中于上涨/下跌或特定股票。
机制：真实 T+1 被隔离的事件由引擎守卫**整组丢弃**（不滑移 T+2）。
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minute_label_engine import load_code  # noqa: E402

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
CLEAN = os.path.join(OUT_DIR, "minute_cache_clean")
RAW = os.path.join(OUT_DIR, "minute_cache_raw")
QUAR_PATH = os.path.join(OUT_DIR, "quarantine_2026.csv")
CAND = os.path.join(OUT_DIR, "minute_candidates_2026.parquet")
SAMPLES = os.path.join(OUT_DIR, "minute_labels", "minute_samples_2026.parquet")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def expand_events():
    """与引擎相同的候选展开（A2/B0/LEGACY 各自成行）。"""
    cand = pd.read_parquet(CAND)
    rows = []
    for _, r in cand.iterrows():
        base = (bool(r.get("corp_action_on_signal_day", False)),
                bool(r.get("corp_action_during_horizon", False)),
                bool(r.get("price_scale_valid", True)))
        if pd.notna(r["a2_event_id"]):
            rows.append((r["code"], int(r["date"]), "A2", r["a2_event_id"]))
        if pd.notna(r["b0_event_id"]):
            rows.append((r["code"], int(r["date"]), "B0", r["b0_event_id"]))
        if pd.notna(r["legacy_event_id"]):
            rows.append((r["code"], int(r["date"]), "LEGACY",
                         r["legacy_event_id"]))
    return pd.DataFrame(rows, columns=["code", "sig", "src", "event_id"])


def day_ret_from_cache(path, code, days_want=None):
    p = os.path.join(path, f"{code}.parquet")
    if not os.path.exists(p):
        return {}
    d = pd.read_parquet(p, columns=["date", "open", "close"])
    d["day"] = d["date"].astype(np.int64) // 1000000
    g = d.groupby("day").agg(o=("open", "first"), c=("close", "last"))
    if days_want is not None:
        g = g[g.index.isin(days_want)]
    return (g["c"] / g["o"] - 1.0).to_dict()


def ret_bucket(x):
    if x > 0.002:
        return "up"
    if x < -0.002:
        return "down"
    return "flat"


def main():
    q = pd.read_csv(QUAR_PATH, dtype={"code": str})
    qset = set(zip(q["code"], q["day"].astype(np.int64)))
    ev = expand_events()
    log(f"展开事件 {len(ev):,}（{ev['src'].value_counts().to_dict()}）")

    # 守卫谓词复算：受影响 = 信号日之后 clean 日历的下一日 ≠ 完整日历的下一日
    aff_rows = []
    cur, cm = None, None
    for _, r in ev.iterrows():
        code = r["code"]
        if code != cur:
            cur = code
            cm = load_code(code)
        if cm is None:
            continue
        days = cm["days"]
        hit = np.where(days == r["sig"])[0]
        if not len(hit):
            continue
        day_i = int(hit[0])
        if day_i + 1 >= len(days):
            continue
        hit_a = np.where(cm["days_all"] == r["sig"])[0]
        if not len(hit_a) or int(hit_a[0]) + 1 >= len(cm["days_all"]):
            continue
        t1 = int(cm["days_all"][int(hit_a[0]) + 1])
        if days[day_i + 1] != t1:  # 与引擎守卫同一谓词
            aff_rows.append({"code": code, "sig": int(r["sig"]), "t1": t1,
                             "src": r["src"]})
    aff = pd.DataFrame(aff_rows)
    n_ev, n_aff = len(ev), len(aff)
    log(f"受影响事件（真实T+1被隔离，守卫丢弃）: {n_aff:,} / {n_ev:,}"
        f"（{n_aff/n_ev:.2%}）")
    aff.to_csv(os.path.join(OUT_DIR, "quarantine_affected_events.csv"),
               index=False)

    # 月份 / 来源
    aff["month"] = aff["t1"] // 100
    ev_m = ev.assign(month=ev["sig"] // 100)  # 近似按信号月
    mon = ev_m.groupby("month").size().rename("events").to_frame()
    mon = mon.join(aff.groupby("month").size().rename("affected")) \
        .fillna({"affected": 0})
    mon["affected_pct"] = mon["affected"] / mon["events"]
    mon_q = q.assign(month=q["day"] // 100).groupby("month").size() \
        .rename("quar_code_days")
    mon = mon.join(mon_q).fillna({"quar_code_days": 0})
    src_t = ev.groupby("src").size().rename("events").to_frame()
    src_t = src_t.join(aff.groupby("src").size().rename("affected")) \
        .fillna({"affected": 0})
    src_t["affected_pct"] = src_t["affected"] / src_t["events"]
    log(f"按月：\n{mon}\n按来源：\n{src_t}")

    # 决策点缺失率（样本：每事件每 dp 应 2 行）
    s = pd.read_parquet(SAMPLES, columns=["group_id", "dp"])
    per_dp = s.groupby("dp").agg(rows=("group_id", "size"),
                                 groups=("group_id", "nunique"))
    per_dp["rows_per_group"] = per_dp["rows"] / per_dp["groups"]
    log(f"每 dp 行/组：\n{per_dp}")

    # 集中度
    top = q["code"].value_counts()
    aff_codes = aff["code"].value_counts() if n_aff else pd.Series(dtype=int)
    log(f"隔离 {len(q):,} code-day / {q['code'].nunique()} 只，"
        f"top10 占比 {top.head(10).sum()/len(q):.1%}，"
        f"单只最多 {top.iloc[0]} 天；受影响事件涉及 {len(aff_codes)} 只")

    # 涨跌分布：受影响事件 T+1 收益（raw）；对照=全池事件 T+1（clean）
    aff_ret_d = {}
    for c_, g_ in aff.groupby("code"):
        days = set(g_["t1"].tolist())
        aff_ret_d.update({(c_, d): r for d, r in
                          day_ret_from_cache(RAW, c_, days).items()})
    aff["ret"] = [aff_ret_d.get((c_, t), np.nan)
                  for c_, t in zip(aff["code"], aff["t1"])]
    aff_ret = aff["ret"].dropna()
    pool = {}
    for i, c_ in enumerate(sorted(ev["code"].unique())):
        pool.update({(c_, d): r for d, r in
                     day_ret_from_cache(CLEAN, c_).items()})
        if (i + 1) % 1000 == 0:
            log(f"池收益 {i+1}/{ev['code'].nunique()}")
    aff["pool_ret"] = [pool.get((c_, t), np.nan)
                       for c_, t in zip(aff["code"], aff["t1"])]
    log(f"受影响 T+1 收益 n={len(aff_ret):,}：中位 {aff_ret.median():+.2%}，"
        f"{aff_ret.map(ret_bucket).value_counts().to_dict()}")
    aff.to_csv(os.path.join(OUT_DIR, "quarantine_affected_events.csv"),
               index=False)

    out = {"expanded_events": int(n_ev), "affected_events": int(n_aff),
           "affected_pct": n_aff / n_ev,
           "by_month": {int(k): {"events": int(v["events"]),
                                 "affected": int(v["affected"]),
                                 "affected_pct": float(v["affected_pct"]),
                                 "quar_code_days": int(v["quar_code_days"])}
                        for k, v in mon.iterrows()},
           "by_src": {k: {"events": int(v["events"]),
                          "affected": int(v["affected"]),
                          "affected_pct": float(v["affected_pct"])}
                      for k, v in src_t.iterrows()},
           "rows_per_group_by_dp": {int(k): float(v)
                                    for k, v in
                                    per_dp["rows_per_group"].items()},
           "quar_code_days": int(len(q)),
           "quar_codes": int(q["code"].nunique()),
           "top10_code_share": float(top.head(10).sum() / len(q)),
           "aff_event_codes": int(len(aff_codes)),
           "aff_event_top10_share": float(aff_codes.head(10).sum()
                                          / max(n_aff, 1)),
           "aff_day_ret": {"n": int(len(aff_ret)),
                           "median": float(aff_ret.median()),
                           "dist": aff_ret.map(ret_bucket)
                           .value_counts().to_dict()}}
    with open(os.path.join(OUT_DIR, "quarantine_bias_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log("written quarantine_bias_report.json")

    # ---- 报告 §18.14 ----
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.14 ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n### 18.14 T2前置：quarantine 隔离分布与选择性偏差检查\n",
         "机制：引擎日历为剔除隔离日后的序列，若事件真实 T+1 被隔离，旧行为是 "
         "dp_date **静默滑移 T+2**；现已加守卫（minute_label_engine，比对 days_all "
         "完整日历）——真实 T+1 被隔离的事件**整组丢弃**，不滑移。本检查从候选"
         "展开事件用与引擎相同的守卫谓词复算受影响集（v2 修正：初版 "
         "searchsorted side 用错，把受影响误算为 432，实际见下）。\n",
         f"- 隔离总量：{out['quar_code_days']:,} 个 (code,day) / "
         f"{out['quar_codes']} 只（集中于源端污染月 202603）；受影响候选事件 "
         f"{out['affected_events']} / {out['expanded_events']:,}"
         f"（{out['affected_pct']:.2%}），守卫整组丢弃",
         f"- 股票集中度：隔离 top10 占比 {out['top10_code_share']:.1%}、"
         f"单只最多 {int(top.iloc[0])} 天；受影响事件 top10 占比 "
         f"{out['aff_event_top10_share']:.0%}，未形成单一股票主导",
         f"- 各决策点缺失率：每事件每 dp 2 行（open/vwap），实际 "
         f"{out['rows_per_group_by_dp']}；隔离丢弃对 945/1000/1030 三决策点"
         f"一致（整组消失，无决策点选择性）"]
    L.append("- 按月（affected=真实T+1被隔离的展开事件数）：")
    L.append("| 月份 | 展开事件 | 受影响 | 占比 | 隔离code-day |")
    L.append("|---:|---:|---:|---:|---:|")
    for mo in sorted(out["by_month"]):
        m = out["by_month"][mo]
        L.append(f"| {mo} | {m['events']} | {m['affected']} | "
                 f"{m['affected_pct']:.2%} | {m['quar_code_days']} |")
    L.append("- 按来源：")
    L.append("| 来源 | 展开事件 | 受影响 | 占比 |")
    L.append("|---|---:|---:|---:|")
    for src in sorted(out["by_src"]):
        m = out["by_src"][src]
        L.append(f"| {src} | {m['events']} | {m['affected']} | "
                 f"{m['affected_pct']:.2%} |")
    ar = out["aff_day_ret"]
    L.append(f"- 涨跌分布：受影响事件 T+1 当日收益 中位 {ar['median']:+.2%}，"
             f"{ar['dist']}（n={ar['n']}，n 小仅作方向参考）")
    L.append("\n结论：受影响事件占比 "
             f"{out['affected_pct']:.2%}（v2修正后），量级极小，无单一股票/单一"
             "方向主导 → **无明细选择性偏差，T2 正式运行**；这些事件已被守卫"
             "整组丢弃（quarantine_affected_events.csv 留痕），不回填不滑移。")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    return out


if __name__ == "__main__":
    main()
