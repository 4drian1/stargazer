"""从 green_exit_v1 退出标签派生月度收益代理（等权、按入场月归组）。

口径说明（非完整组合账本）：
  - 每个标签事件 = 一笔已完成交易，含 net_return（已扣双边成本）。
  - 按 entry_date 的 年月 归组；持仓仅 1~5 个交易日，入场月≈出场月。
  - 月度收益 = 该月所有交易 net_return 的均值（等权代理，未做仓位叠加/再平衡）。
  - 同时给出笔数、净收益合计，便于判断样本量与幅度。

运行：python monthly_returns.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
LABELS = [
    os.path.join(ROOT, "labels", "exit_labels_daily_approx.parquet"),
    os.path.join(ROOT, "labels", "exit_labels_minute_exact.parquet"),
]


def load():
    frames = [pd.read_parquet(p) for p in LABELS if os.path.exists(p)]
    return pd.concat(frames, ignore_index=True)


def monthly(df, variant):
    d = df[df["strategy_variant"] == variant].copy()
    d["entry_date"] = d["entry_date"].astype("int64")
    d["ym"] = d["entry_date"] // 100
    d["year"] = d["entry_date"] // 10000
    d["month"] = d["entry_date"] // 100 % 100
    g = d.groupby("ym").agg(n=("net_return", "size"),
                            mean_net=("net_return", "mean"),
                            sum_net=("net_return", "sum")).reset_index()
    g["ym_str"] = g["ym"].astype(str)
    return d, g


def main():
    df = load()
    print("标签总行数:", len(df), " | 字段:", list(df.columns)[:20])
    for variant in ["X1", "X0_CONTROL", "X1_RF", "X2_STAGED"]:
        d, g = monthly(df, variant)
        print(f"\n========== 变体 {variant} (样本 {len(d)}) ==========")
        # 每年 5/6/7 月
        sub = g.copy()
        sub["year"] = sub["ym"] // 100
        sub["month"] = sub["ym"] % 100
        focus = sub[sub["month"].isin([5, 6, 7])][["ym", "month", "n", "mean_net"]]
        print("每年 5/6/7 月（mean_net=月均净收益）：")
        for _, r in focus.iterrows():
            ym = str(r["ym"])
            flag = "  <-- 关注" if r["month"] in (5, 6, 7) else ""
            print(f"  {ym}: n={int(r['n']):4d}  mean_net={r['mean_net']:+.4f}{flag}")
        # 日历月整体均值（跨年对同月取平均）
        cal = sub.groupby("month")["mean_net"].mean()
        print("各日历月平均净收益（跨年均值）：")
        row = "  "
        for m in range(1, 13):
            row += f"{m:2d}月={cal.get(m, float('nan')):+.4f}  "
            if m % 4 == 0:
                print(row); row = "  "
        if row.strip():
            print(row)
        # 五月/六月 vs 七月
        m56 = sub[sub["month"].isin([5, 6])]["mean_net"].mean()
        m7 = sub[sub["month"] == 7]["mean_net"].mean()
        print(f"\n5/6月平均净收益 = {m56:+.4f}  |  7月平均净收益 = {m7:+.4f}")
        verdict = "七月更优" if m7 > m56 else ("五月/六月更优" if m56 > m7 else "无显著差异")
        print(f"  结论: {verdict}")


if __name__ == "__main__":
    main()
