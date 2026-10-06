# -*- coding: utf-8 -*-
"""旧绿盘低吸模型（desktop sector_pullback A 模式规则 + RR 排序）历史复现。

目的：阶段5公平对比的"旧模型参考曲线"。在 v3.3 清洗数据链上向量化复现
stargazer_desktop/sector_pullback.py 的完整 A 模式选股：
  质量关(流通10-100亿/总<100亿/adv20>=5000万/非涨停) + 趋势关(MA20向上或距20日低<=6%)
  + 调整到位(4/5放宽版)/首次分歧(legacy定义) + RR(1.5ATR止损<=7%,阻力目标,>=1.5)
  + STRONG优先, 档内 rr+0.5*妖股基因分 排序。
限制在 12 关注板块成分内（legacy 只扫关注板块；成分当前快照有幸存者偏差）。
标签/冷却/路径复用 A 引擎（统一口径：5日冷却、close 入场、同 barrier 标签）。
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
sys.path.insert(0, OUT_DIR)
from build_event_funnel import (load_data, add_features, compute_labels,
                                apply_cooldown, K_STOPS, K_TARGETS)  # noqa: E402

CONS_PATH = os.path.join(OUT_DIR, "board_cons.json")
EVENTS_OUT = os.path.join(OUT_DIR, "events_legacy_cooled.parquet")
PATH_OUT = os.path.join(OUT_DIR, "events_path_legacy_cooled.parquet")
ROUND_TRIP_COST = 0.002
MAX_STOP_PCT = 0.07


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def limit_thresh(code: str) -> float:
    if code.startswith(("30", "68")):
        return 0.195
    if code.startswith(("8", "4", "92")):
        return 0.295
    return 0.095


def main():
    t0 = time.time()
    df = load_data()
    df = add_features(df)
    df["pos"] = df.groupby("code", sort=False).cumcount()
    log(f"loaded {len(df)} rows {time.time()-t0:.0f}s")

    cons = json.load(open(CONS_PATH, encoding="utf-8"))
    mb = pd.DataFrame([(c, b) for b, codes in cons.items() for c in codes],
                      columns=["code", "board"])
    m = df.merge(mb, on="code", how="inner")
    log(f"board-member rows: {len(m)}")

    g = m.groupby("code", sort=False)
    m["ma20"] = g["close"].transform(lambda s: s.rolling(20).mean())
    m["ma20_prev"] = g["ma20"].transform(lambda s: s.shift(5))
    m["llv20"] = g["low"].transform(lambda s: s.rolling(20, min_periods=20).min())
    m["prev5_ret1_min"] = g["ret1"].transform(lambda s: s.shift(1).rolling(5).min())
    m["adv20_l"] = g["amount"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=15).mean())

    # ---- 质量关 ----
    cond_q = (m["float_mv"] >= 10e8) & (m["float_mv"] <= 100e8) \
        & (m["total_mv"] <= 100e8) & (m["adv20_l"] >= 5000e4) \
        & (m["ret1"] < m["_limit"] - 0.005)

    # ---- 趋势关（legacy 放宽版）----
    cond_trend = ((m["ma20"] > m["ma20_prev"]) & (m["close"] > m["ma20"])) \
        | ((m["close"] / m["llv20"] - 1) <= 0.06)

    # ---- 调整到位（legacy 4/5）----
    pb_ok = m["pullback_from_high5"].between(-0.18, -0.03)
    vol_ok = m["amount_ratio20"] < 1.0
    pos_ok = m["pos20"] > 0.30
    amp_ok = m["amp_contract"].fillna(False)
    tail_ok = (m["lower_shadow"] > 0.005) \
        | ((m["body_prev"] > 0) & (m["body"] < m["body_prev"] * 0.7))
    adj_ready = (pb_ok.astype(int) + vol_ok.astype(int) + pos_ok.astype(int)
                 + amp_ok.astype(int) + tail_ok.fillna(False).astype(int)) >= 4

    # ---- 首次分歧（legacy）----
    fd = (m["ret5"] >= 0.15) & (m["prev5_ret1_min"] > -0.03) \
        & (((m["high"] - m["pre_close"]) / m["pre_close"] - m["ret1"]) >= 0.03) \
        & (m["amount_ratio20"] >= 1.0)

    # ---- RR（legacy ATR 口径）----
    entry = m["close"]
    stop = entry - 1.5 * m["atr14"]
    risk_dist = entry - stop
    risk_pct = risk_dist / entry + ROUND_TRIP_COST
    h5 = m["high5"].where(m["high5"] > entry, np.nan)
    h20 = m["high20"].where(m["high20"] > entry, np.nan)
    resist = pd.concat([h5, h20], axis=1).min(axis=1)
    target = resist.where(resist.notna(), entry + 2.0 * risk_dist)
    gain_pct = (target - entry) / entry - ROUND_TRIP_COST
    rr = gain_pct / risk_pct
    cond_rr = (risk_dist > 0) & (risk_dist / entry <= MAX_STOP_PCT) \
        & (gain_pct > 0) & (rr >= 1.5)

    sel = m[cond_q & cond_trend & (adj_ready | fd) & cond_rr].copy()
    log(f"legacy pre-cooldown: {len(sel)}")
    if not len(sel):
        log("no events, abort")
        return

    # ---- 妖股基因分 + 排序分 ----
    lu60 = g["ret1"].transform(lambda s: s.rolling(60).apply(
        lambda x: float((x >= 0.095).sum()), raw=True))
    q = np.where(m["ret20"] > 0, np.minimum(m["ret20"], 0.5) / 0.5 * 0.4, 0.0)
    q += np.minimum(lu60.fillna(0), 5) / 5 * 0.4
    q += np.where((m["float_mv"] >= 10e8) & (m["float_mv"] <= 100e8), 0.2, 0.0)
    m["qscore"] = q
    sel["qscore"] = m.loc[sel.index, "qscore"]
    sel["rr"] = rr.loc[sel.index]
    sel["strong"] = (sel["rr"] >= 3.0).values
    sel["score"] = sel["rr"] + 0.5 * sel["qscore"]

    # ---- 标签（A 引擎）----
    sel = sel.sort_values(["code", "pos"], kind="mergesort").reset_index(drop=True)
    sel["entry"] = sel["close"]
    sel["event_id"] = np.arange(len(sel))
    sel["subtype"] = "legacy"
    bar_cols = ["open", "high", "low", "close", "ret1", "_lock", "date"]
    bar_map = {code: {c: grp[c].values for c in bar_cols}
               for code, grp in df.groupby("code", sort=False)}
    labels, path = compute_labels(sel, bar_map)
    sel = pd.concat([sel.reset_index(drop=True), labels], axis=1)
    path.insert(0, "event_id", sel["event_id"].values)
    for ks in K_STOPS:
        sel[f"stop_pct_s{ks}"] = (float(ks) * sel["atr_ref"] / sel["entry"]).astype(np.float32)
        sel[f"eligible_s{ks}"] = sel[f"stop_pct_s{ks}"] <= 0.06
    exit_cols = [f"exit_day_s{ks}_t{kt}" for ks in K_STOPS for kt in K_TARGETS]
    path = path.merge(sel[["event_id"] + exit_cols], on="event_id", how="left")

    keep = apply_cooldown(sel)
    sel = sel[keep].reset_index(drop=True)
    sel["event_id"] = np.arange(len(sel))
    sel.to_parquet(EVENTS_OUT, index=False)
    id_map = pd.DataFrame({"eid_raw": np.arange(len(keep))[keep]})
    id_map["event_id_new"] = np.arange(len(id_map))
    pc = path.merge(id_map, left_on="event_id", right_on="eid_raw")
    pc = pc.drop(columns=["event_id", "eid_raw"]).rename(columns={"event_id_new": "event_id"})
    pc = pc[["event_id", "entry", "c1", "c2", "c3", "c4", "c5"] + exit_cols] \
        .sort_values("event_id").reset_index(drop=True)
    pc.to_parquet(PATH_OUT, index=False)
    log(f"cooled={len(sel)} wrote events+path {time.time()-t0:.0f}s")

    yc = sel.assign(y=sel["date"] // 10000).groupby("y").size()
    log("年度: " + ", ".join(f"{y}: {n}" for y, n in yc.items()))
    log(f"STRONG占比 {(sel['strong']).mean():.1%} rr中位数 {sel['rr'].median():.2f}")


if __name__ == "__main__":
    main()
