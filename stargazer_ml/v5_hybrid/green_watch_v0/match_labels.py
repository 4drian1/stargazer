"""match_labels.py — 池内 (code, signal_date=T) × 退出标签 X0_CONTROL 匹配。

匹配键：(code, signal_date)。标签取 strategy_variant == X0_CONTROL（最简口径：
硬止损 + 时间退出），净收益直接用标签自带 net_return（cost=0.003 旧口径，
与账本冻结口径 0.0013/0.0018 的差异在报告中注明，不重算）。

输出 pool_events.parquet：池事件 + entry/exit/net_return/exit_reason + label_precision。
幂等：已存在则复用。
运行：d:/stargazer/.venv/Scripts/python.exe match_labels.py
"""
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
POOL_F = BASE / "pool_daily.parquet"
OUT = BASE / "pool_events.parquet"
LABELS = [
    Path("d:/stargazer/stargazer_ml/v5_hybrid/green_exit_v1/labels/exit_labels_daily_approx.parquet"),
    Path("d:/stargazer/stargazer_ml/v5_hybrid/green_exit_v1/labels/exit_labels_minute_exact.parquet"),
]


def load_x0():
    frames = [pd.read_parquet(f) for f in LABELS]
    lab = pd.concat(frames, ignore_index=True)
    lab = lab[lab["strategy_variant"] == "X0_CONTROL"].copy()
    lab["code"] = lab["code"].astype(str).str.zfill(6)
    # 2025 年日线/分钟标签可能同 event_id 并存：分钟版优先（后读的文件覆盖）
    lab = lab.sort_values("label_precision").drop_duplicates("event_id", keep="last")
    lab["exit_date"] = lab["exit_time"].astype(str).str[:8].astype(int)
    return lab


def main(force=False):
    if OUT.exists() and not force:
        print(f"[skip] {OUT} 已存在（--force 可重建）")
        return
    t0 = time.time()
    print("[1/3] 加载池 …")
    pool = pd.read_parquet(POOL_F)
    pool["signal_date"] = pool["date"].astype(int)
    print(f"  池行数 {len(pool)}，日期 {pool['signal_date'].min()}..{pool['signal_date'].max()}")

    print("[2/3] 加载 X0_CONTROL 标签 …")
    lab = load_x0()
    print(f"  X0 标签 {len(lab)} 行，{lab['code'].nunique()} 票，"
          f"{lab['signal_date'].min()}..{lab['signal_date'].max()}")

    print("[3/3] 匹配 …")
    keep_cols = ["event_id", "code", "signal_date", "entry_date", "entry_price",
                 "exit_date", "exit_time", "exit_price", "exit_reason", "exit_blocked",
                 "gross_return", "cost", "net_return", "holding_days",
                 "label_precision", "atr_ref", "initial_stop_price"]
    lab_k = lab[keep_cols].rename(columns={
        "code": "label_code", "signal_date": "label_signal_date"})
    ev = pool.merge(lab_k, left_on=["code", "signal_date"],
                    right_on=["label_code", "label_signal_date"], how="left")
    ev["matched"] = ev["event_id"].notna()
    ev = ev.drop(columns=["label_code", "label_signal_date"])

    # ---- 匹配率统计 ----
    n_all = len(ev)
    n_match = int(ev["matched"].sum())
    print(f"\n全池匹配率：{n_match}/{n_all} = {n_match/n_all:.2%}")
    for tier in ["HIGH_PRIORITY_WATCH", "WATCH", "RESEARCH", "NEAR_MISS", "DROP"]:
        sub = ev[ev["primary_tier"] == tier]
        if len(sub):
            print(f"  {tier:20s} {int(sub['matched'].sum()):6d}/{len(sub):6d} "
                  f"= {sub['matched'].mean():.2%}")
    for setup in ["BOTTOM_SETUP", "INTRADAY_DIP_SETUP"]:
        sub = ev[(ev["primary_setup"] == setup) & (ev["primary_tier"].isin(
            ["WATCH", "HIGH_PRIORITY_WATCH"]))]
        if len(sub):
            print(f"  {setup} WATCH+ {int(sub['matched'].sum()):6d}/{len(sub):6d} "
                  f"= {sub['matched'].mean():.2%}")
    # 未匹配的主因：票不在标签覆盖集（标签只有 184 票的旧 S0 宇宙）
    lab_codes = set(lab["code"])
    ev["code_in_label_universe"] = ev["code"].isin(lab_codes)
    un = ev[~ev["matched"]]
    print(f"未匹配中 code 属于标签宇宙的：{un['code_in_label_universe'].mean():.2%}")

    ev.to_parquet(OUT, index=False)
    print(f"\n→ {OUT}（{len(ev)} 行，匹配 {n_match}），用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
