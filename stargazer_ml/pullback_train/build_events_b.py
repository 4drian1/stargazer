# -*- coding: utf-8 -*-
"""B 模式（恐慌板块独立低吸）日线样本构建 v0。

口径（日线近似）：恐慌板块（12 关注板块内，panic_pct≤20% + 跌幅减速或跌停减少 + 中军修复）
中的独立止跌股（硬过滤 + downside_capture<0.5 + 停止创新低 + 下影/收阳 + 缩量 + 距20日低点≤8%）。
标签/路径/冷却完全复用 A 模式的 build_event_funnel 引擎；B 的 ATR 风险上限为 8%（A 为 6%）。
成分股为当前快照 → 有幸存者偏差（报告显著标注）。幂等可重跑。
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
sys.path.insert(0, OUT_DIR)
from build_event_funnel import (load_data, add_features, hard_filter_mask,
                                compute_labels, apply_cooldown,
                                K_STOPS, K_TARGETS, MAX_HOLD)  # noqa: E402

BCAP = 0.08  # B 模式 ATR 止损距离上限
EVENTS_B_RAW = os.path.join(OUT_DIR, "events_b1_raw.parquet")
EVENTS_B_COOLED = os.path.join(OUT_DIR, "events_b1_cooled.parquet")
PATH_B_COOLED = os.path.join(OUT_DIR, "events_path_b_cooled.parquet")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
CONS_PATH = os.path.join(OUT_DIR, "board_cons.json")


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
    df["year"] = (df["date"] // 10000).astype(np.int16)
    log(f"features done {time.time()-t0:.0f}s")

    cons = json.load(open(CONS_PATH, encoding="utf-8"))
    mb = pd.DataFrame([(c, b) for b, codes in cons.items() for c in codes],
                      columns=["code", "board"])
    m = df.merge(mb, on="code", how="inner")
    log(f"board-member rows: {len(m)}")

    # ---- 板块日度聚合（成分等权，当前快照口径）----
    lt = m["code"].map(limit_thresh)
    m["_ld"] = m["ret1"] <= -lt + 0.005
    m["_lu"] = m["ret1"] >= lt - 0.005
    g = m.groupby(["board", "date"], sort=False)
    bday = g.agg(sector_ret1=("ret1", "mean"), n_cons=("code", "nunique"),
                 limit_down_cnt=("_ld", "sum"), limit_up_cnt=("_lu", "sum"),
                 amount=("amount", "sum")).reset_index()
    center = m.loc[g["amount"].idxmax(), ["board", "date", "ret1"]] \
             .rename(columns={"ret1": "center_ret1"})
    bday = bday.merge(center, on=["board", "date"], how="left")
    bday = bday.sort_values(["board", "date"], kind="mergesort")
    bg = bday.groupby("board", sort=False)
    bday["sector_ret5"] = bg["sector_ret1"].transform(lambda s: s.rolling(5, min_periods=3).sum())
    # panic_pct：sector_ret5 在自身历史（截至昨日）的分位
    def _pct(s):
        out = np.full(len(s), np.nan)
        vals = s.values
        for i in range(len(s)):
            h = vals[max(0, i - 250):i]
            h = h[~np.isnan(h)]
            if len(h) >= 60:
                out[i] = (h <= vals[i]).mean()
        return out
    bday["panic_pct"] = bg["sector_ret5"].transform(_pct)
    bday["decel"] = (bday["sector_ret5"] < 0) & \
        (bday["sector_ret5"].abs() < bg["sector_ret5"].shift(5).abs())
    bday["ld_decline"] = bday["limit_down_cnt"] < \
        bg["limit_down_cnt"].transform(lambda s: s.rolling(3, min_periods=2).mean().shift(1))
    bday["center_fix"] = bday["center_ret1"] > bday["sector_ret1"] + 0.005
    bday["board_ok"] = (bday["panic_pct"] <= 0.20) & (bday["sector_ret5"] < 0) \
        & (bday["decel"] | bday["ld_decline"]) & bday["center_fix"]

    keep_b = ["board", "date", "sector_ret1", "sector_ret5", "panic_pct",
              "limit_down_cnt", "limit_up_cnt", "n_cons", "center_ret1", "board_ok"]
    m = m.merge(bday[keep_b], on=["board", "date"], how="left")
    log(f"board daily merged {time.time()-t0:.0f}s")

    # ---- 个股止跌条件（三消融变体，阶段4）----
    # B0=原始规则；B1=放宽缩量(去 cond_vol)；B2=放宽下影线/收阳(去 cond_shadow)
    m["prev_low5"] = m.groupby("code", sort=False)["low"] \
        .transform(lambda s: s.shift(1).rolling(5, min_periods=3).min())
    cond_hard = hard_filter_mask(m)
    cond_panic = m["board_ok"].fillna(False)
    cond_cap = (m["ret5"].abs() / m["sector_ret5"].abs().replace(0, np.nan)) < 0.5
    cond_nolow = (m["low"] > m["prev_low5"]) | (m["close"] > m["open"])
    cond_shadow = (m["lower_shadow"] > 0.005) | (m["close"] > m["open"])
    cond_vol = m["amount_ratio20"] < 1.2
    cond_sup = (m["close"] / m["low20"] - 1) <= 0.08

    base = cond_hard & cond_panic & cond_cap.fillna(False) \
        & cond_nolow.fillna(False) & cond_sup.fillna(False)
    masks = {
        "b0": base & cond_shadow.fillna(False) & cond_vol.fillna(False),
        "b1": base & cond_shadow.fillna(False),                 # 放宽缩量
        "b2": base & cond_vol.fillna(False),                    # 放宽下影/收阳
    }
    union = masks["b0"] | masks["b1"] | masks["b2"]
    log(f"B1 pre-cooldown: b0={int(masks['b0'].sum())} b1={int(masks['b1'].sum())} "
        f"b2={int(masks['b2'].sum())} union={int(union.sum())}")
    if not union.any():
        log("no events, abort")
        return

    # ---- 标签在并集上计算一次（复用 A 引擎）----
    b_all = m[union].copy()  # 保留 m 原索引，供变体子集取值
    b_all = b_all.sort_values(["code", "pos"], kind="mergesort")
    _orig_idx = b_all.index  # reset 前捕获，变体 masks 按 m 原索引对齐
    b_all["entry"] = b_all["close"]
    b_all["event_id"] = np.arange(len(b_all))
    bar_cols = ["open", "high", "low", "close", "ret1", "_lock", "date"]
    if "_lock" not in df.columns:
        df["_lock"] = (df["ret1"] <= -df["code"].map(limit_thresh) + 0.005) \
            & (df["close"] == df["low"])
    bar_map = {code: {c: grp[c].values for c in bar_cols}
               for code, grp in df.groupby("code", sort=False)}
    labels, path = compute_labels(b_all.reset_index(drop=True), bar_map)
    b_all = pd.concat([b_all.reset_index(drop=True), labels], axis=1)
    path.insert(0, "event_id", b_all["event_id"].values)
    for ks in K_STOPS:
        b_all[f"stop_pct_s{ks}"] = (float(ks) * b_all["atr_ref"] / b_all["entry"]).astype(np.float32)
        b_all[f"eligible_s{ks}"] = b_all[f"stop_pct_s{ks}"] <= BCAP
    exit_cols = [f"exit_day_s{ks}_t{kt}" for ks in K_STOPS for kt in K_TARGETS]
    path = path.merge(b_all[["event_id"] + exit_cols], on="event_id", how="left")

    # ---- 各变体独立冷却并落盘 ----
    flags = pd.DataFrame({k: v for k, v in masks.items()})
    a = pd.read_parquet(os.path.join(OUT_DIR, "events_train_v1.parquet"),
                        columns=["code", "date"])
    a_set = set(zip(a["code"], a["date"]))
    stats = {}
    for name in ["b0", "b1", "b2"]:
        keep = flags.loc[_orig_idx, name].values
        sub = b_all[keep].sort_values(["code", "pos"], kind="mergesort")
        cd = apply_cooldown(sub)
        sub = sub[cd].reset_index(drop=True)
        sub["event_id"] = np.arange(len(sub))
        sub.to_parquet(os.path.join(OUT_DIR, f"events_b_{name}_cooled.parquet"),
                       index=False)
        id_map = pd.DataFrame({"eid_raw": b_all["event_id"].values[keep][cd]})
        id_map["event_id_new"] = np.arange(len(id_map))
        pc = path.merge(id_map, left_on="event_id", right_on="eid_raw")
        pc = pc.drop(columns=["event_id", "eid_raw"]) \
            .rename(columns={"event_id_new": "event_id"})
        pc = pc[["event_id", "entry", "c1", "c2", "c3", "c4", "c5"] + exit_cols] \
            .sort_values("event_id").reset_index(drop=True)
        pc.to_parquet(os.path.join(OUT_DIR, f"events_path_b_{name}_cooled.parquet"),
                      index=False)
        ov = float(np.mean([(c, d) in a_set
                            for c, d in zip(sub["code"], sub["date"])]))
        stats[name] = (sub, ov)
        log(f"{name}: cooled={len(sub)} overlap_with_A={ov:.1%}")

    # ---- 统计 ----
    L = ["\n## 9. B 模式样本（恐慌板块独立低吸，v1 三消融）\n",
         "口径：12 关注板块内 panic_pct(自身历史分位)≤20% + 跌幅减速或跌停减少 + 中军修复；"
         "个股 downside_capture<0.5 + 停止创新低 + 距20日低点≤8%。消融变体："
         "B0=原始(另加 下影/收阳 + 缩量(量比20<1.2))；B1=B0放宽缩量；B2=B0放宽下影/收阳。"
         "标签/冷却/路径与 A 完全一致；eligible 上限 8%。"
         "**成分股为当前快照，有幸存者偏差**；板块聚合为成分等权。\n",
         "| 变体 | 年份 | 事件(冷却后) |",
         "|---|---|---:|"]
    for name in ["b0", "b1", "b2"]:
        sub, ov = stats[name]
        yc = sub.assign(y=lambda d: (d["date"] // 10000)).groupby("y").size()
        for y, n in yc.items():
            L.append(f"| {name} | {y} | {n} |")
        elig = float(sub["eligible_s1.5"].mean())
        lab = "outcome_s1.5_t2.0"
        dist = sub[sub["eligible_s1.5"]][lab].value_counts(normalize=True)
        ev = sub[sub["eligible_s1.5"]].groupby(lab)["net_ret_s1.5_t2.0"].mean()
        L.append(f"\n- **{name}** eligible(K=1.5)：{elig:.1%}；"
                 f"K=1.5/t2.0 三分类：" +
                 ", ".join(f"{k} {v:.1%}" for k, v in dist.items()) +
                 f"；各状态净收益：" +
                 ", ".join(f"{k} {v:+.2%}" for k, v in ev.items()) +
                 f"；与A重叠率 {ov:.1%}；裸EV " +
                 f"{float(sub[sub['eligible_s1.5']]['net_ret_s1.5_t2.0'].mean()):+.3%}"
                 f"（t2.0档全eligible均值）")
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n## 9. B 模式样本")
    if idx != -1:
        old = old[:idx] + "\n"
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log("report appended")


if __name__ == "__main__":
    main()
