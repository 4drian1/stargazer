# -*- coding: utf-8 -*-
"""T4第一步：板块分钟代理篮子（T-1 可知、按月冻结，§18.17）。

防未来信息/幸存者偏差规则（评审冻结）：
  - 成分仅用 T-1 月最后一个交易日及之前的数据选定，每月首日冻结当月篮子；
  - 禁止用"最终龙头"回填历史；
  - 北交所(8/4/92)无分钟覆盖，剔除；
  - 板块成分为 board_cons.json 当前快照（幸存者偏差方向未知，沿用项目标注）。

选定规则（每板块每月）：
  leader  = 板块内 ret20 最高（20日动量最强的龙头；历史不足20日则≥5日）
  center  = 板块内 float_mv 最大（中军）
  liquid  = 板块内近20日日均成交额 top5（高流动性篮子）
输出 proxy_baskets.json: {board: {YYYYMM: {leader, center, liquid[5], frozen_at}}}
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
CONS = os.path.join(OUT_DIR, "board_cons.json")
DAILY_GLOB = "D:/a股数据/data_daily_parts/part_*.parquet"
OUT = os.path.join(OUT_DIR, "proxy_baskets.json")
MONTHS = [f"2025{m:02d}" for m in range(1, 13)] \
    + [f"2026{m:02d}" for m in range(1, 10)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    cons = json.load(open(CONS, encoding="utf-8"))
    all_codes = sorted({c for v in cons.values() for c in v})
    all_codes = [c for c in all_codes if not c.startswith(("8", "4", "92"))]
    log(f"12板块成分（剔北交所） {len(all_codes)} 只")
    cols = ["date", "code", "close", "amount", "float_mv", "is_st"]
    frames = []
    for f in sorted(glob.glob(DAILY_GLOB)):
        d = pd.read_parquet(f, columns=cols)
        d = d[(d["code"].isin(all_codes)) & (d["date"] >= 20240901)
              & (~d["is_st"])]
        if len(d):
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    log(f"日线 {len(df):,} 行，{df['code'].nunique()} 只，至 {df['date'].max()}")
    df = df.sort_values(["code", "date"])
    g = df.groupby("code")
    df["ret20"] = g["close"].pct_change(20)
    df["amt20"] = g["amount"].transform(lambda s: s.rolling(20).mean())

    # 每月最后一个交易日（全市场日历）
    cal = np.sort(df["date"].unique())
    month_last = {}
    for d in cal:
        month_last[str(d)[:6]] = int(d)
    code2board = {}
    for b, codes in cons.items():
        for c in codes:
            code2board.setdefault(c, b)

    out = {b: {} for b in cons}
    for mo in MONTHS:
        prev_mos = [m for m in month_last if m < mo]
        if not prev_mos:
            continue
        t1_date = month_last[max(prev_mos)]  # T-1 月最后交易日
        snap = df[df["date"] == t1_date].copy()
        snap["board"] = snap["code"].map(code2board)
        for b, sb in snap.groupby("board"):
            sb = sb[(sb["ret20"].notna()) & (sb["amt20"] > 0)]
            if len(sb) < 3:
                continue
            leader = sb.loc[sb["ret20"].idxmax(), "code"]
            center = sb.loc[sb["float_mv"].idxmax(), "code"]
            liquid = sb.nlargest(5, "amt20")["code"].tolist()
            out[b][mo] = {"leader": leader, "center": center,
                          "liquid": liquid, "frozen_at": int(t1_date)}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"篮子已冻结: {OUT}")
    for b in out:
        months = sorted(out[b])
        if not months:
            log(f"  {b}: 无覆盖")
            continue
        l_changes = len({out[b][m]["leader"] for m in months})
        log(f"  {b}: {len(months)} 个月，龙头更换 {l_changes} 次，"
            f"末日示例 {out[b][months[-1]]['leader']}/"
            f"{out[b][months[-1]]['center']}")


if __name__ == "__main__":
    main()
