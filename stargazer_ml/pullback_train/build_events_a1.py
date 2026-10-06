# -*- coding: utf-8 -*-
"""真A1候选池重建（评审"真A1池"流程第1-2步）。

与宽候选池的本质区别：从板块开始生成事件，不是事后给宽池事件贴板块。
流程：
  每日12关注板块 → 板块日级评分(7成分rank-z，与sector_gate同公式) →
  OOF2022-2023冻结Top20/30/40%三档强势板块 → 强势板块成分中找
  adj_ready形态股票 → 硬过滤(复用) → 个股相对板块无明显破位
  (rel_ret_5d >= -0.10，先验冻结) → signal_sector=当日评分最高板块 →
  按(code,date)去重 → 三分类标签(16组K) → eligible → 按股5日冷却。

三档各自落盘：
  events_a1s{20,30,40}_cooled.parquet + events_path_a1s{q}_cooled.parquet

口径变更（相对宽池，均为先验、不用2024调参）：
  - 不设宽池的 trend 关（强势板块+adj_ready+未破位替代其功能）；
  - 只用 adj_ready 分支（含 both），first_div-only 不进真A1；
  - 无板块身份=不属于A1策略定义，直接不进池（不是"绕过板块门"）。

边界：板块成分为当前快照，有幸存者偏差；2024/2025已多轮观察，本节为E0研究；
真正前瞻证据从2026-09之后。

幂等覆盖写。裸池经济学对比见 a1_pool_economics.py（第3步）。
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
sys.path.insert(0, OUT_DIR)

from build_event_funnel import (  # noqa: E402
    load_data, add_features, hard_filter_mask, pattern_mask, compute_labels,
    apply_cooldown,
    COOLDOWN_DAYS, MAX_HOLD, K_STOPS, STOP_PCT_CAP,
)

SECTOR_DAILY_PATH = os.path.join(OUT_DIR, "sector_daily.parquet")
CONS_PATH = os.path.join(OUT_DIR, "board_cons.json")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
THR_OUT = os.path.join(OUT_DIR, "a1_sector_thresholds.json")

# 与 sector_gate.py 完全相同的 7 成分与权重（先验固定）
COMPONENTS = [
    ("sector_excess_ret_5d", +1, "相对强度"),
    ("sector_up_ratio", +1, "上涨广度"),
    ("sector_limit_up_count", +1, "涨停梯队"),
    ("leader_high_to_close", -1, "龙头冲高回落(取负)"),
    ("center_ret1", +1, "中军状态"),
    ("sector_limit_down_count", -1, "跌停扩散(取负)"),
    ("sector_high_to_close", -1, "冲高回落风险(取负)"),
]
MIN_COMP = 4
TIERS = {"s20": 0.80, "s30": 0.70, "s40": 0.60}   # OOF 分位 q -> 通过 Top(1-q)
REL_STOP_MIN = -0.10   # 无明显破位先验：rel_ret_5d >= -10%


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sector_day_table(df):
    """重建板块日级特征+评分表（不依赖任何事件，全 sector-day 覆盖）。

    逻辑复刻 build_events_train_v1.py 的 sec_all + sector_gate.py 的评分公式。
    """
    cons_dict = json.load(open(CONS_PATH, encoding="utf-8"))
    memb = pd.DataFrame(
        [(code, sec) for sec, codes in cons_dict.items() for code in codes],
        columns=["code", "sector_name"]).drop_duplicates()
    member_codes = set(memb["code"])
    log(f"membership: {len(memb)} pairs, {len(member_codes)} codes")

    df = df.copy()
    df["_lim_up"] = df["ret1"] >= df["_limit"] - 0.005
    mdf = df[df["code"].isin(member_codes)].copy()
    mdf = mdf.merge(memb, on="code", how="inner")
    mdf["htclose"] = (mdf["high"] / mdf["close"] - 1).astype(np.float32)

    agg = mdf.groupby(["sector_name", "date"]).agg(
        sector_up_ratio=("ret1", lambda s: float((s > 0).mean())),
        sector_limit_up_count=("_lim_up", "sum"),
        sector_amount=("amount", "sum"),
        sector_n_cons=("code", "nunique"),
    ).reset_index()
    ld = mdf["ret1"] <= -mdf["_limit"] + 0.005
    agg["sector_limit_down_count"] = mdf[ld].groupby(
        ["sector_name", "date"]).size().reindex(
        pd.MultiIndex.from_frame(agg[["sector_name", "date"]]),
        fill_value=0).values

    big = mdf[mdf["float_mv"] >= 5e9].sort_values(
        ["sector_name", "date", "ret20"], ascending=[True, True, False],
        kind="mergesort")
    leader = big.drop_duplicates(["sector_name", "date"])[
        ["sector_name", "date", "ret1", "htclose"]]
    leader.columns = ["sector_name", "date", "leader_ret1",
                      "leader_high_to_close"]
    ctr = mdf.sort_values(["sector_name", "date", "amount"],
                          ascending=[True, True, False], kind="mergesort") \
        .drop_duplicates(["sector_name", "date"])[["sector_name", "date", "ret1"]]
    ctr.columns = ["sector_name", "date", "center_ret1"]
    agg = agg.merge(leader, on=["sector_name", "date"], how="left") \
        .merge(ctr, on=["sector_name", "date"], how="left")

    # 板块指数特征（东财板块指数日线）
    sec = pd.read_parquet(SECTOR_DAILY_PATH)
    sec = sec.sort_values(["sector", "date"], kind="mergesort")
    sg = sec.groupby("sector", sort=False)
    sec["sector_ret1"] = sg["close"].pct_change()
    sec["sector_ret3"] = sg["close"].pct_change(3)
    sec["sector_ret5"] = sg["close"].pct_change(5)
    sec["sector_high_to_close"] = (sec["high"] / sec["close"] - 1).astype(np.float32)
    nsec = sec.groupby("date")["sector"].nunique()
    sec["sector_return_rank"] = sec.groupby("date")["sector_ret5"] \
        .rank(ascending=False, method="min")
    sec["sector_return_pct"] = (1 - sec["sector_return_rank"]
                                / sec["date"].map(nsec)).astype(np.float32)
    eq = df.groupby("date")["ret1"].median().rename("eq_ret1").reset_index()
    eq["eq_ret5"] = eq["eq_ret1"].rolling(5).sum()
    sec_feat = sec.rename(columns={"sector": "sector_name"})[
        ["sector_name", "date", "sector_ret1", "sector_ret3", "sector_ret5",
         "sector_high_to_close", "sector_return_rank", "sector_return_pct"]]
    sec_feat = sec_feat.merge(eq[["date", "eq_ret5"]], on="date", how="left")
    sec_feat["sector_excess_ret_5d"] = (sec_feat["sector_ret5"]
                                        - sec_feat["eq_ret5"]).astype(np.float32)

    sd = agg.merge(sec_feat, on=["sector_name", "date"], how="outer")
    log(f"sector-day table: {len(sd)} rows, {sd['sector_name'].nunique()} sectors, "
        f"{sd['date'].min()}~{sd['date'].max()}")

    # 评分：7 成分按日横截面 rank->z 等权（与 sector_gate.py 同公式）
    sd = sd.sort_values(["date", "sector_name"], kind="mergesort") \
        .reset_index(drop=True)
    zsum = np.zeros(len(sd))
    n_comp = np.zeros(len(sd), dtype=int)
    for c, sign, _ in COMPONENTS:
        r = sd.groupby("date")[c].rank(pct=True)
        z = norm.ppf(r.clip(1e-4, 1 - 1e-4)) * sign
        ok = sd[c].notna()
        zsum += np.where(ok, z, 0.0)
        n_comp += ok.astype(int)
    sd["n_comp"] = n_comp
    sd["sector_gate_score"] = np.where(
        n_comp >= MIN_COMP, zsum / MIN_COMP, np.nan)
    return sd


def main():
    t0 = time.time()
    df = load_data()
    df = add_features(df)
    df["pos"] = df.groupby("code", sort=False).cumcount()
    df["year"] = (df["date"] // 10000).astype(np.int16)
    log(f"features done {time.time()-t0:.0f}s")

    sd = sector_day_table(df)
    oof_sd = sd[(sd["date"] >= 20220101) & (sd["date"] <= 20231231)]
    thrs = {k: float(oof_sd["sector_gate_score"].quantile(q))
            for k, q in TIERS.items()}
    with open(THR_OUT, "w", encoding="utf-8") as f:
        json.dump({"quantiles_oof22_23_full_sector_days": thrs,
                   "note": "s20/s30/s40 = OOF q80/q70/q60 通过(Top20/30/40%板块日); "
                           "全sector-day口径重建(非事件条件化)",
                   "rel_stop_min": REL_STOP_MIN}, f, ensure_ascii=False)
    log(f"OOF 冻结强势板块阈值(全板块日口径): {thrs}")

    # 板块日分位映射：sector_name,date -> score / sector_ret5
    sd_valid = sd.dropna(subset=["sector_gate_score"])
    score_map = sd_valid.set_index(["sector_name", "date"])["sector_gate_score"]
    ret5_map = sd.dropna(subset=["sector_ret5"]).set_index(
        ["sector_name", "date"])["sector_ret5"]

    # ------------------------------------------------ 候选生成（从板块出发）
    # 1) adj_ready 形态事件（含 both），复用宽池同一判定
    pool = df[hard_filter_mask(df)].copy()
    pat, subtype = pattern_mask(pool)
    adj = pool[pat].copy()
    adj["subtype"] = np.asarray(subtype)[pat.values]
    adj = adj[adj["subtype"].isin(["adj_ready", "both"])]
    adj = adj[adj["atr_ref"].notna() & (adj["atr_ref"] > 0)]
    log(f"adj_ready 事件(全市场, 未过滤板块): {len(adj)}")

    # 2) 只保留 12 板块成员（无板块身份 = 不属于 A1 定义）
    cons_dict = json.load(open(CONS_PATH, encoding="utf-8"))
    memb = pd.DataFrame(
        [(code, sec) for sec, codes in cons_dict.items() for code in codes],
        columns=["code", "sector_name"]).drop_duplicates()
    adj = adj.merge(memb, on="code", how="inner")
    log(f"其中板块成员: {len(adj)} (涉及 {adj['code'].nunique()} 股)")

    # 3) 强势板块日过滤 + 破位过滤 + signal_sector 选取
    adj["sg_score"] = [score_map.get((s, d), np.nan)
                       for s, d in zip(adj["sector_name"], adj["date"])]
    adj = adj.dropna(subset=["sg_score"])
    adj["sector_ret5"] = [ret5_map.get((s, d), np.nan)
                          for s, d in zip(adj["sector_name"], adj["date"])]
    adj["rel_ret_5d"] = (adj["ret5"] - adj["sector_ret5"]).astype(np.float32)
    n_pre_rel = len(adj)
    adj = adj[adj["rel_ret_5d"] >= REL_STOP_MIN]
    log(f"强势板块日匹配: {n_pre_rel}; 破位过滤后(rel>= {REL_STOP_MIN}): {len(adj)}")

    # signal_sector = 当日评分最高板块；每(code,date)一条
    adj = adj.sort_values(["code", "date", "sg_score"],
                          ascending=[True, True, False], kind="mergesort")
    adj = adj.drop_duplicates(["code", "date"]).reset_index(drop=True)
    adj["in_watch_sector"] = True
    log(f"signal_sector 去重后(未冷却): {len(adj)}")

    # ------------------------------------------------ 标签 + 分档冷却落盘
    log("building bar map for labels ...")
    bar_cols = ["open", "high", "low", "close", "ret1", "_lock", "date"]
    bar_map = {code: {c: grp[c].values for c in bar_cols}
               for code, grp in df.groupby("code", sort=False)}
    adj = adj.sort_values(["code", "pos"], kind="mergesort").reset_index(drop=True)
    adj["entry"] = adj["close"]

    funnel = {}
    for tier, th in thrs.items():
        ev = adj[adj["sg_score"] >= th].copy()
        log(f"[{tier}] thr={th:.3f} 未冷却事件: {len(ev)}")
        labels, path = compute_labels(ev, bar_map)
        ev = pd.concat([ev.reset_index(drop=True), labels], axis=1)
        for ks in K_STOPS:
            ev[f"stop_pct_s{ks}"] = (ks * ev["atr_ref"] / ev["entry"]).astype(np.float32)
            ev[f"eligible_s{ks}"] = ev[f"stop_pct_s{ks}"] <= STOP_PCT_CAP
        ev["event_id"] = np.arange(len(ev))
        path.insert(0, "event_id", ev["event_id"].values)
        keep = apply_cooldown(ev)
        ev_c = ev[keep].reset_index(drop=True)
        ev_c["event_id"] = np.arange(len(ev_c))
        id_map = pd.DataFrame({"event_id": np.arange(len(ev_c))})
        id_map["eid_raw"] = ev["event_id"].values[keep]
        pth = path.merge(id_map, left_on="event_id", right_on="eid_raw")
        pth = pth.drop(columns=["event_id_x", "eid_raw"]) \
                 .rename(columns={"event_id_y": "event_id"})[["event_id", "entry",
                                                              "c1", "c2", "c3", "c4", "c5"]]
        pth = pth.sort_values("event_id").reset_index(drop=True)
        exit_cols = [f"exit_day_s{ks}_t{kt}" for ks in K_STOPS
                     for kt in ("1.0", "1.5", "2.0", "3.0")]
        pth = pth.merge(ev_c[["event_id"] + exit_cols], on="event_id", how="left")
        evp = os.path.join(OUT_DIR, f"events_a1{tier}_cooled.parquet")
        pvp = os.path.join(OUT_DIR, f"events_path_a1{tier}_cooled.parquet")
        ev_c.to_parquet(evp, index=False)
        pth.to_parquet(pvp, index=False)
        log(f"wrote {evp} {len(ev_c)} rows (+{pvp})")
        funnel[tier] = (ev, ev_c)

    # ------------------------------------------------ 报告 §17 上半（构建+漏斗）
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n## 17. ")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n## 17. 真A1候选池重建（第1-2步：板块出发生成事件）\n",
         "与宽候选池的本质区别：从板块开始生成事件，不是事后给宽池贴板块。"
         "流程：每日12关注板块日级评分(7成分rank-z，公式与§15相同) → "
         "OOF2022-2023冻结Top20/30/40%三档 → 强势板块成分中 adj_ready 形态股票"
         "（复用宽池同一判定，含 both）→ 硬过滤（复用）→ 个股相对板块无明显破位"
         f"（rel_ret_5d≥{REL_STOP_MIN}，先验冻结）→ signal_sector=当日评分最高板块，"
         "每(code,date)一条 → 16组K标签 → 按股5日冷却。\n",
         "口径变更（先验，不用2024调参）：不设宽池 trend 关（强势板块+adj_ready+未破位"
         "替代其功能）；只用 adj_ready 分支，first_div-only 不进真A1；"
         "**无板块身份=不属于A1策略定义，直接不进池**（不是\"绕过板块门\"，"
         "修正§15宽池63.5%自动放行的问题）。\n",
         f"- OOF 冻结阈值（全 sector-day 口径重建，非事件条件化）："
         f"{ {k: round(v, 3) for k, v in thrs.items()} }\n",
         "- 板块成分为当前快照，有幸存者偏差；2024/2025已多轮观察，本节为 E0 研究。\n",
         "| 档 | 未冷却事件 | 冷却后事件 | 2024冷却后 | 2025冷却后 | 2019-23冷却后 |",
         "|---|---:|---:|---:|---:|---:|"]
    for tier in TIERS:
        ev, ev_c = funnel[tier]
        y24 = int((ev_c["year"] == 2024).sum())
        y25 = int((ev_c["year"] == 2025).sum())
        y1923 = int(((ev_c["year"] >= 2019) & (ev_c["year"] <= 2023)).sum())
        L.append(f"| {tier} | {len(ev):,} | {len(ev_c):,} | {y24:,} | {y25:,} | {y1923:,} |")
    L.append("\n裸池经济学（事件数/裸EV/三分类分布/机械组合）见 "
             "a1_pool_economics.py 输出的本节续表——先判读裸池EV，再决定是否重训。")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
