# -*- coding: utf-8 -*-
"""绿盘低吸 pullback_train - 第3.1步第9条：给宽候选池事件接入板块/市场/龙头特征

输入:
  events_a1_cooled.parquet  (v3.1 冷却后事件, 正式训练用)
  sector_daily.parquet / market_daily.parquet / board_cons.json (fetch_sector_data.py 产物)
  D:/a股数据/data_daily_parts/part_*.parquet (本地全市场日线)

输出:
  events_train_v1.parquet  = cooled 事件 + 板块/市场/龙头/个股补充特征
    板块特征多对多展开: 每个事件在其所属各板块各一行 (sector_name 列);
    不属于 12 板块任何之一的事件保留一行, 板块特征列 NaN, in_watch_sector=False

注意:
  - 所有特征只用 T 日及以前数据 (板块特征即 T 日收盘值, 无前视)
  - board_cons.json 是**当前成分快照**, 应用于历史存在幸存者偏差 (报告已标注)
  - 幂等覆盖写
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
EVENTS_PATH = os.path.join(OUT_DIR, "events_a1_cooled.parquet")
OUT_PATH = os.path.join(OUT_DIR, "events_train_v1.parquet")
SECTOR_DAILY_PATH = os.path.join(OUT_DIR, "sector_daily.parquet")
MARKET_DAILY_PATH = os.path.join(OUT_DIR, "market_daily.parquet")
CONS_PATH = os.path.join(OUT_DIR, "board_cons.json")
META_PATH = os.path.join(OUT_DIR, "fetch_sector_meta.json")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
DATA_GLOB = "D:/a股数据/data_daily_parts/part_*.parquet"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def board_limit(p2):
    return np.where(p2.isin(["60", "00"]), 0.095,
                    np.where(p2.isin(["30", "68"]), 0.195, 0.295))


def main():
    t0 = time.time()
    events = pd.read_parquet(EVENTS_PATH)
    log(f"events: {len(events)}")
    events = events.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
    events["event_id"] = np.arange(len(events))

    # ------------------------------------------------ local daily (whitelist)
    cols = ["date", "code", "open", "high", "low", "close", "pre_close",
            "volume", "amount", "turnover", "float_mv", "is_st"]
    files = sorted(glob.glob(DATA_GLOB))
    log(f"loading {len(files)} parquet files ...")
    frames = [pd.read_parquet(f, columns=cols) for f in files]
    df = pd.concat(frames, ignore_index=True)
    del frames
    df = df.dropna(subset=["close", "high", "low", "open"])
    df = df.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
    prev_close = df.groupby("code", sort=False)["close"].shift(1)
    df["pre_close"] = df["pre_close"].fillna(prev_close)
    df = df.dropna(subset=["pre_close"]).reset_index(drop=True)
    p1, p2 = df["code"].str[:1], df["code"].str[:2]
    wl = p2.isin(["00", "30", "60", "68"]) | p1.isin(["8", "4"]) \
        | df["code"].str.startswith("92")
    df = df[wl].reset_index(drop=True)
    log(f"daily loaded: {len(df)} rows, {df['code'].nunique()} codes")
    for c in ["open", "high", "low", "close", "pre_close", "turnover"]:
        df[c] = df[c].astype(np.float32)
    df["ret1"] = (df["close"] / df["pre_close"] - 1).astype(np.float32)
    df["_limit"] = board_limit(df["code"].str[:2]).astype(np.float32)
    df["_lim_up"] = df["ret1"] >= df["_limit"] - 0.005
    df["pos"] = df.groupby("code", sort=False).cumcount()

    # ------------------------------------------------ market features (per date)
    log("computing market features ...")
    eq = df.groupby("date")["ret1"].median().rename("eq_ret1").reset_index()
    eq["eq_ret5"] = eq["eq_ret1"].rolling(5).sum()
    eq["eq_ret20"] = eq["eq_ret1"].rolling(20).sum()
    vol20 = eq["eq_ret1"].rolling(20).std()
    eq["market_vol_state"] = vol20.expanding(60).apply(
        lambda x: (x.iloc[:-1] <= x.iloc[-1]).mean(), raw=False)
    mkt_idx = pd.read_parquet(MARKET_DAILY_PATH)
    sh_c = mkt_idx["sh000300_close"].astype(float)
    mkt_idx["market_pos20"] = (sh_c - sh_c.rolling(20).min()) \
        / (sh_c.rolling(20).max() - sh_c.rolling(20).min())
    for pre, col in [("sh", "sh000300"), ("cyb", "cyb399006")]:
        c = mkt_idx[f"{col}_close"].astype(float)
        mkt_idx[f"{pre}_ret1"] = c.pct_change()
        mkt_idx[f"{pre}_ret5"] = c.pct_change(5)
        mkt_idx[f"{pre}_ret20"] = c.pct_change(20)
    mkt = eq.merge(mkt_idx[["date", "market_pos20", "sh_ret1", "sh_ret5", "sh_ret20",
                            "cyb_ret1", "cyb_ret5", "cyb_ret20"]], on="date", how="left")
    log(f"market_daily: {len(mkt)} days {int(mkt['date'].min())}~{int(mkt['date'].max())}")

    # ------------------------------------------------ sector membership
    with open(CONS_PATH, encoding="utf-8") as f:
        cons_dict = json.load(f)
    memb = pd.DataFrame(
        [(code, sec) for sec, codes in cons_dict.items() for code in codes],
        columns=["code", "sector_name"])
    memb = memb.drop_duplicates()
    member_codes = set(memb["code"])
    log(f"membership: {len(memb)} code-sector pairs, {len(member_codes)} unique codes")

    # ------------------------------------------------ sector-day aggregates (from local daily)
    log("computing sector-day aggregates (member daily) ...")
    mdf = df[df["code"].isin(member_codes)].copy()
    g = mdf.groupby("code", sort=False)
    mdf["ret20"] = g["close"].pct_change(20)
    mdf = mdf.merge(memb, on="code", how="inner")
    mdf["htclose"] = (mdf["high"] / mdf["close"] - 1).astype(np.float32)

    agg = mdf.groupby(["sector_name", "date"]).agg(
        sector_up_ratio=("ret1", lambda s: float((s > 0).mean())),
        sector_limit_up_count=("_lim_up", "sum"),
        sector_limit_down_count=("ret1", lambda s: 0),
        sector_amount=("amount", "sum"),
        sector_n_cons=("code", "nunique"),
    ).reset_index()
    ld = mdf["ret1"] <= -mdf["_limit"] + 0.005
    agg["sector_limit_down_count"] = mdf[ld].groupby(["sector_name", "date"]).size() \
        .reindex(pd.MultiIndex.from_frame(agg[["sector_name", "date"]]), fill_value=0).values

    big = mdf[mdf["float_mv"] >= 5e9].sort_values(
        ["sector_name", "date", "ret20"], ascending=[True, True, False],
        kind="mergesort")
    leader = big.drop_duplicates(["sector_name", "date"])[
        ["sector_name", "date", "code", "ret1", "htclose"]]
    leader.columns = ["sector_name", "date", "leader_code", "leader_ret1",
                      "leader_high_to_close"]
    ctr = mdf.sort_values(["sector_name", "date", "amount"],
                          ascending=[True, True, False], kind="mergesort") \
        .drop_duplicates(["sector_name", "date"])[
        ["sector_name", "date", "code", "ret1"]]
    ctr.columns = ["sector_name", "date", "center_code", "center_ret1"]
    agg = agg.merge(leader, on=["sector_name", "date"], how="left") \
        .merge(ctr, on=["sector_name", "date"], how="left")
    log(f"sector-day agg: {len(agg)} rows")

    # ------------------------------------------------ sector index features
    log("computing sector index features ...")
    sec = pd.read_parquet(SECTOR_DAILY_PATH)
    sec = sec.sort_values(["sector", "date"], kind="mergesort")
    sg = sec.groupby("sector", sort=False)
    sec["sector_ret1"] = sg["close"].pct_change()
    sec["sector_ret3"] = sg["close"].pct_change(3)
    sec["sector_ret5"] = sg["close"].pct_change(5)
    sec["sector_high_to_close"] = (sec["high"] / sec["close"] - 1).astype(np.float32)
    sec["sector_return_rank"] = sec.groupby("date")["sector_ret5"] \
        .rank(ascending=False, method="min")
    nsec = sec.groupby("date")["sector"].nunique()
    sec["sector_return_pct"] = (1 - sec["sector_return_rank"]
                                / sec["date"].map(nsec)).astype(np.float32)
    sec_feat = sec[["sector", "date", "sector_ret1", "sector_ret3", "sector_ret5",
                    "sector_high_to_close", "sector_return_rank",
                    "sector_return_pct"]].merge(
        eq[["date", "eq_ret5"]], on="date", how="left")
    sec_feat["sector_excess_ret_5d"] = (sec_feat["sector_ret5"]
                                        - sec_feat["eq_ret5"]).astype(np.float32)
    sec_feat = sec_feat.rename(columns={"sector": "sector_name"})
    sec_all = agg.merge(sec_feat, on=["sector_name", "date"], how="outer")
    log(f"sector feature table: {len(sec_all)} rows")

    # ------------------------------------------------ event stock features
    log("computing event stock features ...")
    ev_codes = set(events["code"])
    sdf = df[df["code"].isin(ev_codes)].copy()
    gs = sdf.groupby("code", sort=False)
    sdf["ret3"] = gs["close"].pct_change(3)
    sdf["ret20"] = gs["close"].pct_change(20)
    sdf["limit60"] = gs["_lim_up"].transform(lambda s: s.rolling(60, min_periods=20).sum())
    delta = gs["close"].diff()
    sdf["_up"] = delta.clip(lower=0)
    sdf["_dn"] = (-delta).clip(lower=0)
    up = sdf.groupby("code", sort=False)["_up"].transform(lambda s: s.rolling(14).mean())
    dn = sdf.groupby("code", sort=False)["_dn"].transform(lambda s: s.rolling(14).mean())
    rs = up / dn.replace(0, np.nan)
    sdf["rsi14"] = (100 - 100 / (1 + rs)).astype(np.float32)
    ma = {n: gs["close"].transform(lambda s, n=n: s.rolling(n).mean())
          for n in (3, 6, 12, 24, 20)}
    bbi = (ma[3] + ma[6] + ma[12] + ma[24]) / 4
    sdf["bbi_distance"] = (sdf["close"] / bbi - 1).astype(np.float32)
    sdf["bbi_slope5"] = (bbi / bbi.groupby(sdf["code"], sort=False).shift(5)
                         - 1).astype(np.float32)
    std20 = gs["close"].transform(lambda s: s.rolling(20).std())
    denom = (4 * std20).replace(0, np.nan)
    sdf["boll_pct_b"] = ((sdf["close"] - (ma[20] - 2 * std20)) / denom).astype(np.float32)
    dn_vol = gs["volume"].transform(
        lambda s: s.where(sdf.loc[s.index, "ret1"] < 0, 0).rolling(5).sum())
    tot_vol = gs["volume"].transform(lambda s: s.rolling(5).sum())
    sdf["down_volume_ratio5"] = (dn_vol / tot_vol.replace(0, np.nan)).astype(np.float32)
    t_med = gs["turnover"].transform(lambda s: s.rolling(20, min_periods=10).median())
    sdf["turnover_ratio20"] = (sdf["turnover"] / t_med.replace(0, np.nan)).astype(np.float32)
    mv = sdf["float_mv"]
    sdf["quality_score"] = (
        sdf["ret20"].clip(0, 0.6) / 0.6 * 50
        + sdf["limit60"].clip(0, 5) / 5 * 30
        + np.where((mv >= 1e9) & (mv <= 1e10), 20,
                   np.where(mv <= 3e10, 10, 0))
    ).astype(np.float32)
    sdf["rsi_delta3"] = (sdf["rsi14"]
                         - sdf["rsi14"].groupby(sdf["code"], sort=False).shift(3)
                         ).astype(np.float32)

    # turnover_pct60: 事件行处计算（当日换手在过去60日的分位）
    to_map = {c: grp["turnover"].values for c, grp in sdf.groupby("code", sort=False)}
    tpc = np.full(len(events), np.nan, dtype=np.float32)
    ev_code = events["code"].values
    ev_pos = events["pos"].values
    for i in range(len(events)):
        arr = to_map[ev_code[i]][max(0, ev_pos[i] - 59):ev_pos[i] + 1]
        if len(arr) >= 20 and not np.isnan(arr[-1]):
            tpc[i] = (arr <= arr[-1]).mean()
    events["turnover_pct60"] = tpc

    stock_feat = sdf[["code", "date", "ret3", "limit60", "rsi14",
                      "rsi_delta3", "bbi_distance", "bbi_slope5", "boll_pct_b",
                      "down_volume_ratio5", "turnover_ratio20", "quality_score"]]
    events = events.merge(stock_feat, on=["code", "date"], how="left")
    events = events.merge(mkt, on="date", how="left")
    log(f"stock+market features merged: {len(events)}")

    # ------------------------------------------------ sector expansion
    log("expanding events by sector membership ...")
    ev_sec = events.merge(memb, on="code", how="inner")[
        ["event_id", "sector_name"]]
    base = events.drop(columns=["event_id"]).copy()
    base.insert(0, "event_id", events["event_id"].values)
    expanded = ev_sec.merge(base, on="event_id", how="left")
    expanded["in_watch_sector"] = True
    non_member = events[~events["event_id"].isin(set(ev_sec["event_id"]))].copy()
    non_member["sector_name"] = None
    non_member["in_watch_sector"] = False
    out = pd.concat([expanded, non_member], ignore_index=True)

    sec_cols = ["sector_name", "date", "sector_up_ratio", "sector_limit_up_count",
                "sector_limit_down_count", "sector_amount", "sector_n_cons",
                "leader_code", "leader_ret1", "leader_high_to_close",
                "center_code", "center_ret1", "sector_ret1", "sector_ret3",
                "sector_ret5", "sector_high_to_close", "sector_return_rank",
                "sector_return_pct", "sector_excess_ret_5d"]
    out = out.merge(sec_all[sec_cols], on=["sector_name", "date"], how="left")
    out["rel_ret_1d"] = (out["ret1"] - out["sector_ret1"]).astype(np.float32)
    out["rel_ret_3d"] = (out["ret3"] - out["sector_ret3"]).astype(np.float32)
    out["rel_ret_5d"] = (out["ret5"] - out["sector_ret5"]).astype(np.float32)

    f32cols = ["sector_up_ratio", "sector_amount", "leader_ret1",
               "leader_high_to_close", "center_ret1", "rel_ret_1d", "rel_ret_3d",
               "rel_ret_5d"]
    for c in f32cols:
        out[c] = out[c].astype(np.float32)

    out = out.sort_values(["event_id", "sector_name"],
                          na_position="last", kind="mergesort").reset_index(drop=True)
    out.to_parquet(OUT_PATH, index=False)
    log(f"wrote {OUT_PATH}: {out.shape[0]} rows x {out.shape[1]} cols")

    # ------------------------------------------------ report section 7
    # 先移除旧版第7节（幂等追加）
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n## 7. 板块特征接入")
    if idx != -1:
        old = old[:idx] + "\n"
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(old.rstrip("\n") + "\n")
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    n_ev = len(events)
    n_member_ev = int(events["event_id"].isin(set(ev_sec["event_id"])).sum())
    share = n_member_ev / n_ev * 100
    lines = []
    lines.append("\n## 7. 板块特征接入（第3.1步第9条）\n")
    lines.append("数据源：东方财富接口（curl_cffi 直连），板块指数日线 2018-01 至今；"
                 "**成分为当前快照，应用于历史存在幸存者偏差**"
                 "（已退市/被调出个股在历史期内不可见，板块统计系统性偏乐观），"
                 "训练时需注意。\n")
    lines.append("板块匹配结果（同花顺口径 → 东财名称）：")
    lines.append("")
    lines.append("| 板块 | 东财名称 | 类型 | 成分股数 | 指数历史覆盖 |")
    lines.append("|---|---|---|---:|---|")
    for canon, info in meta["matched"].items():
        rng = meta["hist_range"].get(canon, ["?", "?"])
        lines.append(f"| {canon} | {info['em_name']} | {info['src']} "
                     f"| {meta['cons_count'].get(canon, 0)} | {rng[0]}~{rng[1]} |")
    if meta["failed"]:
        lines.append(f"\n**拉取失败板块**：{', '.join(meta['failed'])}")
    lines.append("")
    lines.append(f"- 事件-板块展开：{n_ev:,} 个冷却后事件中 {n_member_ev:,} 个 "
                 f"（{share:.1f}%）属于至少一个关注板块（in_watch_sector=True，"
                 f"板块特征列非 NaN）；其余 {n_ev - n_member_ev:,} 个保留单行，"
                 "板块特征列 NaN。**大部分事件不属于 12 板块是正常的**："
                 "板块特征是排序信号，不是过滤条件。")
    n_expand_rows = len(ev_sec)
    n_multi_sector = int((ev_sec.groupby("event_id").size() > 1).sum())
    lines.append(f"- 多板块归属：{n_expand_rows:,} 行事件-板块展开来自 {n_member_ev:,} 个事件，"
                 f"其中 {n_multi_sector:,} 个事件同属 2 个以上板块。")
    sec_cov = sec_all["date"].agg(["min", "max"])
    lines.append(f"- sector_daily 特征表覆盖：{int(sec_cov['min'])}~{int(sec_cov['max'])}，"
                 f"{sec_all['sector_name'].nunique()} 个板块。")
    mkt_cov = mkt["date"].agg(["min", "max"])
    lines.append(f"- market_daily 覆盖：{int(mkt_cov['min'])}~{int(mkt_cov['max'])}；"
                 "eq_ret1 为全市场（白名单）当日 ret1 中位数。")
    lines.append("- 龙头/中军口径：leader = 板块内 ret20 最大且 float_mv≥50亿 的成分股；"
                 "center = 板块内当日成交额最大的成分股；均用本地日线计算。")
    lines.append("- 新增个股特征：quality_score（ret20 加分 + 60日涨停次数 + 市值达标）、"
                 "rsi14/rsi_delta3、bbi_distance/bbi_slope5、boll_pct_b、"
                 "down_volume_ratio5、turnover_ratio20、turnover_pct60。")
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"appended report section 7")
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
