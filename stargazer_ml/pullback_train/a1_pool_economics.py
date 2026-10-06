# -*- coding: utf-8 -*-
"""真A1池第3步：裸池经济学对比（不训练任何模型）。

问题：不使用模型排序、仅改变候选池定义，2024 的底层 EV 是否改善？

口径：
- 全部用同一主标签组合 K_stop=1.5 × K_target=2.0，eligible(stop_pct<=6%) 子集；
- 裸EV/三分类分布 = eligible 且非 CENSORED 事件的 outcome 统计；
- 机械组合 = 合格事件全部强制填仓（ev_thr=None，容量压力测试口径），
  20槽、T+1、真账本、统一自然年度窗口；EV 列置 0（无排序，当日按原始顺序填槽，
  不代表任何策略收益，只代表池子底层经济学）；
- 宽池对照 = events_a1_cooled 中 subtype∈{adj_ready,both}（与真A1同形态口径）；
- 每个池子的 event_id/path 文件切换时重置 B 模块全局缓存。

输出：报告 §17 续表（裸池经济学 + 机械组合）。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
sys.path.insert(0, OUT_DIR)
import train_lgbm_baseline as B  # noqa: E402

REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
KS = "1.5"
KT = "2.0"
TAG = f"s{KS}_t{KT}"
WINDOWS = {"2024": (20240101, 20241231), "2025": (20250101, 20251231)}

POOLS = {
    "宽池(adj)": ("events_a1_cooled.parquet", "events_path_cooled.parquet"),
    "A1-s40": ("events_a1s40_cooled.parquet", "events_path_a1s40_cooled.parquet"),
    "A1-s30": ("events_a1s30_cooled.parquet", "events_path_a1s30_cooled.parquet"),
    "A1-s20": ("events_a1s20_cooled.parquet", "events_path_a1s20_cooled.parquet"),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def use_pool(events_file, path_file):
    B.EVENT_PATH = os.path.join(OUT_DIR, events_file)
    B.EVENTS_PATH = B.EVENT_PATH
    B.EVENT_QUOTE = None
    B.PATH_COOLED_PATH = os.path.join(OUT_DIR, path_file)
    B.PATH_MAP = None


def load_pool(events_file, path_file):
    ev = pd.read_parquet(os.path.join(OUT_DIR, events_file))
    if "subtype" in ev.columns:
        ev = ev[ev["subtype"].isin(["adj_ready", "both"])]
    return ev


def dist_stats(ev):
    """eligible + 非CENSORED 的 outcome 分布与裸EV"""
    ev = ev[ev[f"eligible_s{KS}"]]
    oc = ev[f"outcome_{TAG}"]
    counted = ~ev["is_censored"] & (oc != "NO_FUTURE")
    oc = oc[counted]
    nr = ev[f"net_ret_{TAG}"][counted]
    n = int(counted.sum())
    if not n:
        return {"n_elig": int(len(ev)), "n": 0, "tgt": np.nan, "stp": np.nan,
                "tmo": np.nan, "ev": np.nan}
    return {"n_elig": int(len(ev)), "n": n,
            "tgt": float((oc == "TARGET").mean() * 100),
            "stp": float((oc == "STOP").mean() * 100),
            "tmo": float((oc == "TIMEOUT").mean() * 100),
            "ev": float(nr.mean())}


def mech_portfolio(ev, window):
    """强制填仓机械组合：合格事件全部参与，无排序"""
    src = ev[ev[f"eligible_s{KS}"] & ~ev["is_censored"]
             & (ev[f"outcome_{TAG}"] != "NO_FUTURE")].copy()
    src["target_atr"] = float(KT)
    src["net_ret"] = src[f"net_ret_{TAG}"].astype(np.float32)
    src["EV"] = 0.0
    pf = B.portfolio_backtest(src[["date", "code", "event_id", "EV",
                                   "net_ret", "target_atr"]],
                              ks=KS, ev_thr=None, window=window)
    return pf


def main():
    t0 = time.time()
    dist_rows, pf_rows = [], []
    for name, (ef, pf) in POOLS.items():
        ev = load_pool(ef, pf)
        log(f"[{name}] events={len(ev)}")
        for y in range(2019, 2026):
            ey = ev[ev["year"] == y]
            if not len(ey):
                continue
            s = dist_stats(ey)
            dist_rows.append({"pool": name, "year": y, **s})
        use_pool(ef, pf)
        for tag, win in WINDOWS.items():
            evw = ev[(ev["date"] >= win[0]) & (ev["date"] <= win[1])]
            if not len(evw):
                continue
            r = mech_portfolio(evw, win)
            pf_rows.append({"pool": name, "seg": tag,
                            "ann": r.get("ann_ret"), "mdd": r.get("max_dd"),
                            "n": r.get("n_trades"), "pos": r.get("avg_pos"),
                            "util": r.get("util"),
                            "win": r.get("trade_win")})
            log(f"[{name}|{tag}] mech ann={r.get('ann_ret'):+.2%} "
                f"mdd={r.get('max_dd'):+.2%} n={r.get('n_trades')} "
                f"util={r.get('util'):.0%}")
    ddf = pd.DataFrame(dist_rows)
    pdf = pd.DataFrame(pf_rows)

    # ------------------------------------------------ 报告 §17 续
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 17.1 裸池经济学")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n### 17.1 裸池经济学（第3步，不训练模型）\n",
         f"主标签组合 K_stop={KS}×K_target={KT}，eligible(止损≤6%) 且非CENSORED 口径。"
         "关键判读：**只看池子定义改变（强势板块限定）是否改善 2024 底层 EV**，"
         "不涉及任何排序模型。机械组合=合格事件全填仓真账本（20槽、T+1、费用、"
         "统一自然年度窗口），EV=0 无排序，仅表征池子底层经济学，不代表策略收益。\n",
         "| 池 | 年 | 合格事件 | 有效事件 | TARGET% | STOP% | TIMEOUT% | 裸EV |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in ddf.iterrows():
        L.append(f"| {r['pool']} | {r['year']} | {int(r['n_elig']):,} | {int(r['n']):,} | "
                 f"{r['tgt']:.1f} | {r['stp']:.1f} | {r['tmo']:.1f} | {r['ev']*100:+.2f}% |")
    L.append("")
    L.append("| 池 | 段 | 机械组合年化 | 回撤 | 交易数 | 日均仓位 | 利用率 | 胜率 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in pdf.iterrows():
        L.append(f"| {r['pool']} | {r['seg']} | {r['ann']:+.2%} | {r['mdd']:+.2%} | "
                 f"{int(r['n'])} | {r['pos']:.1f} | {r['util']:.0%} | {r['win']:.0%} |")
    L.append("")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
