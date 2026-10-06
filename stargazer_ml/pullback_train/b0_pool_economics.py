# -*- coding: utf-8 -*-
"""B0裸池经济学（第17.3节）：同口径一次性确认，不调参。

口径与 §17.1 完全一致：K_stop=1.5 × K_target=2.0，eligible 列
（B模式的eligible上限为8%，见§10），非CENSORED；机械组合=合格事件
全填仓真账本（20槽、T+1、费用、统一自然年度窗口），EV=0无排序。
目的：确定B0是独立收益分支 / 低回撤辅助分支 / 仅市场年份暴露，
不是升级B为主线。结果与 §17.1 的宽池A、真A1三档并列比较。
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
KS, KT, TAG = "1.5", "2.0", "s1.5_t2.0"
WINDOWS = {"2024": (20240101, 20241231), "2025": (20250101, 20251231)}
EVENTS = "events_b_b0_cooled.parquet"
PATH = "events_path_b_b0_cooled.parquet"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    ev = pd.read_parquet(os.path.join(OUT_DIR, EVENTS))
    log(f"B0 events={len(ev)}")
    B.EVENT_PATH = os.path.join(OUT_DIR, EVENTS)
    B.EVENTS_PATH = B.EVENT_PATH
    B.EVENT_QUOTE = None
    B.PATH_COOLED_PATH = os.path.join(OUT_DIR, PATH)
    B.PATH_MAP = None

    dist_rows, pf_rows = [], []
    for y in range(2019, 2026):
        ey = ev[ev["year"] == y]
        if not len(ey):
            continue
        e = ey[ey[f"eligible_s{KS}"]]
        oc_all = e[f"outcome_{TAG}"]
        counted = ~e["is_censored"] & (oc_all != "NO_FUTURE")
        oc, nr = oc_all[counted], e[f"net_ret_{TAG}"][counted]
        dist_rows.append({"year": y, "n_elig": int(len(e)), "n": int(counted.sum()),
                          "tgt": float((oc == "TARGET").mean() * 100),
                          "stp": float((oc == "STOP").mean() * 100),
                          "tmo": float((oc == "TIMEOUT").mean() * 100),
                          "ev": float(nr.mean())})
    for tag, win in WINDOWS.items():
        evw = ev[(ev["date"] >= win[0]) & (ev["date"] <= win[1])]
        src = evw[evw[f"eligible_s{KS}"] & ~evw["is_censored"]
                  & (evw[f"outcome_{TAG}"] != "NO_FUTURE")].copy()
        src["target_atr"] = float(KT)
        src["net_ret"] = src[f"net_ret_{TAG}"].astype(np.float32)
        src["EV"] = 0.0
        r = B.portfolio_backtest(src[["date", "code", "event_id", "EV",
                                      "net_ret", "target_atr"]],
                                 ks=KS, ev_thr=None, window=win)
        pf_rows.append({"seg": tag, "ann": r.get("ann_ret"),
                        "mdd": r.get("max_dd"), "n": r.get("n_trades"),
                        "pos": r.get("avg_pos"), "util": r.get("util"),
                        "win": r.get("trade_win")})
        log(f"[B0|{tag}] mech ann={r.get('ann_ret'):+.2%} "
            f"mdd={r.get('max_dd'):+.2%} n={r.get('n_trades')} "
            f"util={r.get('util'):.0%}")

    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 17.3 B0裸池经济学")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n### 17.3 B0裸池经济学（同口径一次性确认，不调参）\n",
         f"口径与§17.1完全一致：K_stop={KS}×K_target={KT}，eligible（B模式上限8%）"
         "且非CENSORED；机械组合=合格事件全填仓真账本，EV=0无排序。"
         "目的：判定B0是独立收益分支 / 低回撤辅助分支 / 仅市场年份暴露。\n",
         "| 年 | 合格事件 | 有效事件 | TARGET% | STOP% | TIMEOUT% | 裸EV |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for r in dist_rows:
        L.append(f"| {r['year']} | {r['n_elig']:,} | {r['n']:,} | {r['tgt']:.1f} | "
                 f"{r['stp']:.1f} | {r['tmo']:.1f} | {r['ev']*100:+.2f}% |")
    L.append("")
    L.append("| 段 | 机械组合年化 | 回撤 | 交易数 | 日均仓位 | 利用率 | 胜率 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in pf_rows:
        L.append(f"| {r['seg']} | {r['ann']:+.2%} | {r['mdd']:+.2%} | {int(r['n'])} | "
                 f"{r['pos']:.1f} | {r['util']:.0%} | {r['win']:.0%} |")
    L.append("")
    L.append("并列对照（机械组合年化/回撤，K=1.5×2.0全填仓）：宽池(adj) 2024 "
             "-14.1%/-27.9%、2025 +5.7%/-14.5%；真A1三档 2024 -23.8%~-27.5%、"
             "2025 +16.0%~+22.4%（§17.1）。B0带模型阈值口径见§11/§12：2024阈值"
             "年化约-5.92%、公平对比约-8.81%，2025 +6.18%~+6.50%。")
    L.append("\n判读规则（评审预定）：裸池2024仍为负→B0保持辅助观察，不承担主模型"
             "角色；不据本节结果反向调B规则。")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
