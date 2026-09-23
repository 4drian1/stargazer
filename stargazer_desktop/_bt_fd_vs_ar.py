# -*- coding: utf-8 -*-
"""首次分歧 vs 调整到位：历史胜率快速回测（近2年，当前12板块部分成分股）"""
import statistics as st
from datetime import date, timedelta

import numpy as np
import pandas as pd

import sector_pullback as sp
from tools import market

END = date.today().isoformat()
START = (date.today() - timedelta(days=730)).isoformat()

# 逐日判定（与 _compute_first_divergence / _compute_adjustment_ready 同逻辑）
def flags_per_day(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    fd = pd.Series(False, index=d.index)
    ar = pd.Series(False, index=d.index)
    ret5, ret1 = d["ret5"], d["ret1"]
    pb = d["pullback_from_high5"]
    aratio = d["amount_ratio20"]
    pos20 = d["pos20"]
    body, lshadow = d["body"], d["lower_shadow"]
    # first_div
    no_drop5 = (~(ret1 < -0.03)).rolling(5).apply(lambda x: x.all(), raw=True).astype(bool)
    high_pct = (d["high"] - d["prev_close"]) / d["prev_close"]
    fd = ((ret5 >= 0.15) & no_drop5 & ((high_pct - ret1) >= 0.03) & (aratio >= 1.0))
    # adj_ready (4/5)
    c1 = (pb <= -0.03) & (pb >= -0.18)
    c2 = aratio < 1.0
    c3 = pos20 > 0.30
    c4 = d["amp_contract"].astype(bool)
    body_prev = body.shift(1)
    c5 = (lshadow > 0.005) | ((body_prev > 0) & (body < body_prev * 0.7))
    ar = (c1.astype(int) + c2.astype(int) + c3.astype(int) + c4.astype(int) + c5.astype(int)) >= 4
    out = pd.DataFrame({"fd": fd.fillna(False), "ar": ar.fillna(False)}, index=d.index)
    return out


def simulate(df, i, entry):
    """信号日 i（收盘后确认），次日开盘价买入。返回 (结果, 收益率)。
    结果: WIN / LOSS / TIMEOUT"""
    if i + 1 >= len(df):
        return None, 0.0
    nxt = df.iloc[i + 1]
    entry = float(nxt["open"]) if nxt["open"] > 0 else entry
    sig = df.iloc[i]
    atr = sig.get("atr14")
    if pd.isna(atr) or atr <= 0:
        return None, 0.0
    stop = entry - 1.5 * float(atr)
    if (entry - stop) / entry > 0.05:
        return "ABANDON", 0.0
    resists = [t for t in (sig.get("high5"), sig.get("high20"))
               if pd.notna(t) and t > entry]
    target = min(resists) if resists else entry + 2 * (entry - stop)
    cost = sp.ROUND_TRIP_COST
    for j in range(i + 1, min(i + 11, len(df))):
        row = df.iloc[j]
        hit_stop = row["low"] <= stop
        hit_target = row["high"] >= target
        if hit_stop and hit_target:
            return "LOSS", -(entry - stop) / entry - cost  # 同日双触保守记亏
        if hit_stop:
            return "LOSS", -(entry - stop) / entry - cost
        if hit_target:
            return "WIN", (target - entry) / entry - cost
    exit_p = float(df.iloc[min(i + 10, len(df) - 1)]["close"])
    return "TIMEOUT", (exit_p - entry) / entry - cost


BOARDS = ["BK1128", "BK0490", "BK1137", "BK1036", "BK0922", "BK0989", "BK0545", "BK1106"]
results = {"fd": [], "ar": []}
abandon = {"fd": 0, "ar": 0}
for bk in BOARDS:
    cons = sp._board_cons_cached(bk)
    cons = dict(sorted(cons.items(), key=lambda kv: -(kv[1].get("amount_yi") or 0))[:60])
    for code, info in cons.items():
        c, df = sp._kline_one(code, START, END)
        if df is None or len(df) < 120:
            continue
        fmc = float(info.get("fmc") or 0)
        if not (sp.MIN_CAP_YI <= fmc <= sp.MAX_CAP_YI):
            continue
        df = df.copy()
        df["float_mv"] = fmc
        qf = sp._compute_quality_flag(df)  # 用最新质量关近似（幸存者口径，注意偏差）
        if not qf:
            continue
        fl = flags_per_day(df)
        sig_idx = [i for i in range(60, len(df) - 1)
                   if fl["fd"].iloc[i] or fl["ar"].iloc[i]]
        # 同一票 10 日内只取第一个信号，防重叠
        last_t = -99
        for i in sig_idx:
            if i - last_t < 10:
                continue
            kind = "fd" if fl["fd"].iloc[i] else "ar"
            r, ret = simulate(df, i, float(df.iloc[i]["close"]))
            last_t = i
            if r == "ABANDON":
                abandon[kind] += 1
                continue
            if r:
                results[kind].append((r, round(ret * 100, 2)))

for kind, label in (("fd", "首次分歧"), ("ar", "调整到位")):
    rows = results[kind]
    print(f"== {label} == 信号数(不含放弃): {len(rows)}  被ATR>5%放弃: {abandon[kind]}")
    if not rows:
        continue
    wins = [r for r in rows if r[0] == "WIN"]
    losses = [r for r in rows if r[0] == "LOSS"]
    touts = [r for r in rows if r[0] == "TIMEOUT"]
    all_ret = [r[1] for r in rows]
    print(f"   胜率(达标止赢): {len(wins)/len(rows)*100:.1f}%  止损: {len(losses)/len(rows)*100:.1f}%  超时: {len(touts)/len(rows)*100:.1f}%")
    print(f"   平均收益: {st.mean(all_ret):+.2f}%  中位: {st.median(all_ret):+.2f}%")
    if wins or losses:
        expct = (len(wins)/len(rows))*st.mean([r[1] for r in wins] or [0]) + \
                (len(losses)/len(rows))*st.mean([r[1] for r in losses] or [0]) + \
                (len(touts)/len(rows))*st.mean([r[1] for r in touts] or [0])
        print(f"   期望值: {expct:+.2f}%/笔")
    print(f"   赢时均赚: {st.mean([r[1] for r in wins]) if wins else 0:+.2f}%  亏时均亏: {st.mean([r[1] for r in losses]) if losses else 0:+.2f}%")
