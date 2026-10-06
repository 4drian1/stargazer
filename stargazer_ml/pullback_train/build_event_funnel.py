# -*- coding: utf-8 -*-
"""绿盘低吸训练数据 - 第3步 v3.1：年度事件漏斗统计（日线近似口径）

重要措辞：本脚本产出的是"全市场宽股票候选池"事件，**不是真正的 A1 事件**。
真正的 A1 还需要板块强势/梯队/龙头中军等特征（需在线拉板块指数历史，
属于后续单独步骤，本脚本不接入）。

口径：信号日 T 收盘后出信号，entry=T 日收盘价（尾盘近似），
退出评估从 T+1 开始，最长持有 MAX_HOLD 个交易日。

v3.1 变更（评审 8 条）：
  1. MAX_HOLD 5 -> 主标签口径（10 日仅用于消融实验）
  2. 冷却顺序：硬过滤 -> 趋势 -> 形态 -> 宽候选池事件 -> 按股 5 日冷却
  3. 双输出：events_a1_raw.parquet（冷却前，审计/漏斗）/ events_a1_cooled.parquet（冷却后，正式训练）
  4. ATR 最大风险资格：stop_pct_k = K_stop*ATR_ref/entry，eligible_k = stop_pct_k<=6%
  5. 删失：label_end_date / available_future_days / is_censored，正式统计单列 CENSORED
  6. 白名单 00/30/60/68 + 北交所(8/4/92 开头)，修复北交所涨跌停 startswith 口径
  7. 连续跌停顺延：按板块口径 lock 判定，恢复可卖日按实际可成交价退出
  8. 命名与措辞统一为"宽候选池"

幂等运行，覆盖写输出。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

DATA_GLOB = "D:/a股数据/data_daily_parts/part_*.parquet"
OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
EVENTS_RAW_PATH = os.path.join(OUT_DIR, "events_a1_raw.parquet")
EVENTS_COOLED_PATH = os.path.join(OUT_DIR, "events_a1_cooled.parquet")
PATH_RAW_PATH = os.path.join(OUT_DIR, "events_path_raw.parquet")
PATH_COOLED_PATH = os.path.join(OUT_DIR, "events_path_cooled.parquet")

COOLDOWN_DAYS = 5          # 同一只股票 5 个交易日内只算第一次事件
MAX_HOLD = 5               # 主标签口径：最长持有 5 个交易日（10 日仅用于消融实验）
COST = 0.002               # 往返成本 0.2%
K_STOPS = [1.2, 1.5, 1.8, 2.0]
K_TARGETS = [1.0, 1.5, 2.0, 3.0]
STOP_PCT_CAP = 0.06        # A 模式最大风险上限：stop_pct <= 6% 才 eligible

os.makedirs(OUT_DIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def board_limit(p2):
    """按代码前缀返回涨跌停阈值：60/00=9.5%，30/68=19.5%，北交所(8/4/92)=29.5%"""
    return np.where(p2.isin(["60", "00"]), 0.095,
                    np.where(p2.isin(["30", "68"]), 0.195, 0.295))


# ---------------------------------------------------------------- load
def load_data():
    files = sorted(glob.glob(DATA_GLOB))
    log(f"loading {len(files)} parquet files ...")
    cols = ["date", "code", "open", "high", "low", "close", "pre_close",
            "amount", "is_st", "float_mv", "total_mv"]
    frames = [pd.read_parquet(f, columns=cols) for f in files]
    df = pd.concat(frames, ignore_index=True)
    del frames
    df = df.dropna(subset=["close", "high", "low", "open"])
    df = df.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
    # v3.3 数据质量修复：源数据为前复权 OHLC，但源 pre_close 是未复权价
    # （2018 年 83% 行偏差>1%，4,774/5,509 只股票受影响，随时间收敛至 2026=0）。
    # 直接用前收盘序列推导 pre_close，保证 ret1/振幅/下影/涨跌停判定全链一致。
    df["pre_close"] = df.groupby("code", sort=False)["close"].shift(1)
    still_na = int(df["pre_close"].isna().sum())
    df = df.dropna(subset=["pre_close"]).reset_index(drop=True)
    log(f"pre_close 弃用源字段，改由前复权前收盘推导; "
        f"{still_na} rows dropped (first bar per code)")
    # v3.3-2 数据质量断言（前复权口径；违反即 abort，禁止带病下游）
    _r = (df["close"] / df["pre_close"] - 1).astype(np.float32)
    # 涨跌停规则下任何负向超跌幅都是数据错误（多为复权缺口），按板块口径剔除
    _p2 = df["code"].str[:2]
    _lim = np.where(_p2.isin(["30", "68"]), 0.195,
                    np.where(df["code"].str[:1].isin(["8", "4"])
                             | df["code"].str.startswith("92"), 0.295, 0.095))
    _err = _r.values < (-_lim - 0.005)
    if _err.any():
        _yr = (df["date"] // 10000).astype(np.int16)[_err]
        _cnt = _yr.value_counts().sort_index()
        log(f"剔除负向超跌幅数据错误行 {int(_err.sum())} 行(占 {_err.mean():.4%}): "
            + ", ".join(f"{y}:{n}" for y, n in _cnt.items()))
        df = df[~_err].reset_index(drop=True)
        _r = (df["close"] / df["pre_close"] - 1).astype(np.float32)
    _r_min = float(_r.min())
    assert _r_min >= -0.305 - 1e-3, f"ret1 最小值异常: {_r_min:.3f}"
    _yr_med = _r.groupby((df["date"] // 10000).astype(np.int16)).median()
    assert bool((_yr_med.abs() < 0.015).all()), \
        f"年度 ret1 中位数异常(应≈0): {_yr_med.round(4).to_dict()}"
    _n_na = int(_r.isna().sum())
    assert _n_na == 0, f"ret1 存在 {_n_na} 个 NaN"
    log("数据质量断言通过 年度ret1中位数: "
        + ", ".join(f"{y}:{v:+.2%}" for y, v in _yr_med.items()))
    # v3.1-6 白名单：00/30/60/68 + 北交所(8/4/92 开头)，剔除 15/51/56/58 基金等
    p1, p2 = df["code"].str[:1], df["code"].str[:2]
    wl = p2.isin(["00", "30", "60", "68"]) | p1.isin(["8", "4"]) \
        | df["code"].str.startswith("92")
    n_nonwl = int((~wl).sum())
    df = df[wl].reset_index(drop=True)
    log(f"whitelist kept {len(df)} rows, removed {n_nonwl} non-A-share rows "
        f"(funds/bonds 15/51/56/58/...)")
    log(f"loaded {len(df)} rows, {df['code'].nunique()} codes, "
        f"{df['date'].min()} ~ {df['date'].max()}")
    return df


# ---------------------------------------------------------------- features
def add_features(df):
    g = df.groupby("code", sort=False)
    fcols = ["open", "high", "low", "close", "pre_close", "amount"]
    for c in fcols:
        df[c] = df[c].astype(np.float32)

    df["len"] = g.cumcount() + 1
    df["ret1"] = df["close"] / df["pre_close"] - 1

    df["ret5"] = g["close"].pct_change(5)
    df["ret20"] = g["close"].pct_change(20)
    df["high5"] = g["high"].transform(lambda s: s.rolling(5).max())
    df["low5"] = g["low"].transform(lambda s: s.rolling(5).min())
    df["high20"] = g["high"].transform(lambda s: s.rolling(20).max())
    df["low20"] = g["low"].transform(lambda s: s.rolling(20).min())
    df["high60"] = g["high"].transform(lambda s: s.rolling(60, min_periods=40).max())
    df["low60"] = g["low"].transform(lambda s: s.rolling(60, min_periods=40).min())

    df["pos20"] = (df["close"] - df["low20"]) / (df["high20"] - df["low20"])
    df["pullback_from_high5"] = df["close"] / df["high5"] - 1
    df["adv20"] = g["amount"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=15).mean())
    df["amount_ratio20"] = df["amount"] / df["adv20"]

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["pre_close"]).abs(),
        (df["low"] - df["pre_close"]).abs(),
    ], axis=1).max(axis=1)
    df["tr"] = tr.astype(np.float32)
    df["atr14"] = g["tr"].transform(lambda s: s.rolling(14, min_periods=10).mean())
    df["atr_ref"] = g["atr14"].transform(lambda s: s.shift(1).rolling(5).median())

    df["body"] = (df["close"] - df["open"]).abs()
    df["body_prev"] = g["body"].shift(1)
    df["lower_shadow"] = (np.minimum(df["open"], df["close"]) - df["low"]) / df["pre_close"]
    df["amplitude"] = (df["high"] - df["low"]) / df["pre_close"]
    df["amp_mean5"] = g["amplitude"].transform(lambda s: s.shift(1).rolling(5).mean())
    df["amp_contract"] = df["amplitude"] < df["amp_mean5"] * 0.8

    df["ma20"] = g["close"].transform(lambda s: s.rolling(20).mean())

    ret1_prev_min5 = g["ret1"].transform(lambda s: s.shift(1).rolling(5).min())
    df["_ret1_prev_min5"] = ret1_prev_min5

    # v3.1-6/7 跌停锁死标记（按板块口径）：ret1 <= -limit+0.5% 且 close==low
    lim = board_limit(df["code"].str[:2])
    df["_limit"] = lim.astype(np.float32)
    df["_lock"] = (df["ret1"] <= -df["_limit"] + 0.005) & (df["close"] == df["low"])

    f32 = ["ret1", "ret5", "ret20", "high5", "low5", "high20", "low20",
           "high60", "low60", "pos20", "pullback_from_high5", "adv20",
           "amount_ratio20", "atr14", "atr_ref", "body", "body_prev",
           "lower_shadow", "amplitude", "amp_mean5", "ma20", "_ret1_prev_min5"]
    for c in f32:
        df[c] = df[c].astype(np.float32)
    return df


# ---------------------------------------------------------------- filters
def hard_filter_mask(df):
    m = (
        (df["len"] >= 60)
        & (~df["is_st"].astype(bool))
        & (df["float_mv"] >= 1e9) & (df["float_mv"] <= 1e10)
        & (df["total_mv"] <= 1e10)
        & (df["adv20"] >= 5e7)
    )
    # v3.1-6 涨停剔除：用 _limit 列（60/00=9.5%, 30/68=19.5%, 北交所=29.5%）
    m = m & (df["ret1"] < df["_limit"])
    return m


def trend_mask(df):
    cond_a = (df["ma20"] > df["ma20"].groupby(df["code"], sort=False).shift(5)) \
        & (df["close"] > df["ma20"])
    cond_b = ((df["close"] - df["low20"]) / df["low20"]) <= 0.06
    m = cond_a | cond_b
    return m.fillna(False), cond_a.fillna(False)


def pattern_mask(df):
    c1 = (df["pullback_from_high5"] >= -0.18) & (df["pullback_from_high5"] <= -0.03)
    c2 = df["amount_ratio20"] < 1.0
    c3 = df["pos20"] > 0.30
    c4 = df["amp_contract"].fillna(False)
    c5 = (df["lower_shadow"] > 0.005) \
        | ((df["body_prev"] > 0) & (df["body"] < df["body_prev"] * 0.7))
    adj_cnt = (c1.astype(np.int8) + c2.astype(np.int8) + c3.astype(np.int8)
               + c4.astype(np.int8) + c5.astype(np.int8))
    adj_ready = adj_cnt >= 4

    fd1 = df["ret5"] >= 0.15
    fd2 = df["_ret1_prev_min5"] >= -0.03
    fd3 = (df["high"] - df["pre_close"]) / df["pre_close"] - df["ret1"] >= 0.03
    fd4 = df["amount_ratio20"] >= 1.0
    first_div = (fd1 & fd2 & fd3 & fd4).fillna(False)

    m = (adj_ready | first_div).fillna(False)
    subtype = np.where(first_div & adj_ready, "both",
                       np.where(first_div, "first_div", "adj_ready"))
    return m, subtype


def apply_cooldown(events, pos_col="pos", days=COOLDOWN_DAYS):
    """同一股票 pos 差 < days 的事件只保留第一个。events 需按 code,pos 排序。"""
    keep = np.ones(len(events), dtype=bool)
    codes = events["code"].values
    poss = events[pos_col].values
    last_code, last_pos = None, -10**9
    for i in range(len(events)):
        c, p = codes[i], poss[i]
        if c == last_code and (p - last_pos) < days:
            keep[i] = False
        else:
            last_code, last_pos = c, p
    return keep


# ---------------------------------------------------------------- labels
def compute_labels(events, bar_map):
    """对每个宽候选池事件，16 组 (K_stop, K_target) 一次性算标签（MAX_HOLD 日主口径）。

    bar_map: code -> dict of numpy arrays (open/high/low/close/ret1/_lock/date)。
    返回 DataFrame: 每事件各 K 组合的 outcome / net_ret / ambiguous / fill_note，
    以及 label_end_date / available_future_days / is_censored（与 K 无关）。
    """
    n = len(events)
    combos = [(ks, kt) for ks in K_STOPS for kt in K_TARGETS]
    outcomes = {c: np.full(n, "", dtype=object) for c in combos}
    net_ret = {c: np.full(n, np.nan, dtype=np.float32) for c in combos}
    ambig = {c: np.zeros(n, dtype=np.int8) for c in combos}
    fills = {c: np.full(n, "", dtype=object) for c in combos}
    timeout_ret = np.full(n, np.nan, dtype=np.float32)
    label_end_date = np.full(n, np.nan)
    avail_days = np.zeros(n, dtype=np.int16)
    # v3.2: 组合回测所需的每日路径与退出日
    crel = np.full((n, MAX_HOLD), np.nan, dtype=np.float32)   # c[d]/entry-1, d=0..MAX_HOLD-1
    exit_days = {c: np.full(n, np.nan, dtype=np.float32) for c in combos}

    codes = events["code"].values
    poss = events["pos"].values
    entries = events["entry"].values
    atrs = events["atr_ref"].values

    for i in range(n):
        bars = bar_map[codes[i]]
        p = poss[i]
        entry = entries[i]
        atr = atrs[i]
        sl = slice(p + 1, p + 1 + MAX_HOLD)
        o = bars["open"][sl]
        h = bars["high"][sl]
        l = bars["low"][sl]
        c = bars["close"][sl]
        lk = bars["_lock"][sl]
        m = len(o)
        avail_days[i] = m
        if m >= MAX_HOLD:
            label_end_date[i] = bars["date"][p + MAX_HOLD]
        crel[i, :m] = (c / entry - 1).astype(np.float32)
        if m == 0:
            for cb in combos:
                outcomes[cb][i] = "NO_FUTURE"
                fills[cb][i] = "none"
            continue
        t_end = c[min(m, MAX_HOLD) - 1]
        timeout_ret[i] = np.float32(t_end / entry - 1 - COST)

        for cb in combos:
            ks, kt = cb
            S = entry - ks * atr
            Tk = entry + kt * atr
            stop_pending = False
            res, ret, fill, am = "TIMEOUT", timeout_ret[i], "timeout_close", 0
            ex = float(min(m, MAX_HOLD))  # v3.2: 默认超时在第 min(m,MAX_HOLD) 日退出
            for d in range(m):
                if stop_pending:
                    # v3.1-7 连续跌停顺延：恢复可卖日按实际可成交价退出
                    if o[d] < S and lk[d]:
                        continue  # 仍锁死，继续顺延
                    if o[d] < S:
                        res, ret, fill = "STOP", np.float32(o[d] / entry - 1 - COST), "stop_deferred_open"
                    else:
                        res, ret, fill = "STOP", np.float32(S / entry - 1 - COST), "stop_deferred_stop"
                    stop_pending = False
                    ex = d + 1
                    break
                if o[d] > Tk:
                    res, ret, fill = "TARGET", np.float32(o[d] / entry - 1 - COST), "target_open_gap"
                    ex = d + 1
                    break
                if o[d] < S:
                    if lk[d]:
                        # 跌停锁死，不可成交，顺延
                        stop_pending = True
                        continue
                    res, ret, fill = "STOP", np.float32(o[d] / entry - 1 - COST), "stop_open"
                    ex = d + 1
                    break
                if l[d] <= S:
                    if h[d] >= Tk:
                        am = 1  # 同 K 线双触, 悲观记 STOP
                    res, ret, fill = "STOP", np.float32(S / entry - 1 - COST), "stop_level"
                    ex = d + 1
                    break
                if h[d] >= Tk:
                    res, ret, fill = "TARGET", np.float32(Tk / entry - 1 - COST), "target_level"
                    ex = d + 1
                    break
            if stop_pending:
                # 窗口最后一日仍锁死，按最后可得 close 近似成交并标注
                res, ret, fill = "STOP", np.float32(c[m - 1] / entry - 1 - COST), "stop_approx_close"
            outcomes[cb][i] = res
            net_ret[cb][i] = ret
            ambig[cb][i] = am
            fills[cb][i] = fill
            exit_days[cb][i] = ex

    out = {
        "label_end_date": pd.array(label_end_date, dtype="Int64"),
        "available_future_days": avail_days,
        "is_censored": avail_days < MAX_HOLD,
        "timeout_return": timeout_ret,
    }
    for cb in combos:
        ks, kt = cb
        tag = f"s{ks}_t{kt}"
        out[f"outcome_{tag}"] = outcomes[cb]
        out[f"net_ret_{tag}"] = net_ret[cb]
        out[f"ambig_{tag}"] = ambig[cb]
        out[f"fill_{tag}"] = fills[cb]
        out[f"exit_day_{tag}"] = exit_days[cb]
    path = {"entry": events["entry"].values}
    for d in range(MAX_HOLD):
        path[f"c{d+1}"] = crel[:, d]
    return pd.DataFrame(out), pd.DataFrame(path)


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    df = load_data()
    df = add_features(df)
    log(f"features done, {time.time()-t0:.0f}s, {df.memory_usage(deep=True).sum()/1e9:.2f} GB")

    df["pos"] = df.groupby("code", sort=False).cumcount()
    df["year"] = (df["date"] // 10000).astype(np.int16)

    # v3.1-2 顺序：硬过滤 -> 趋势 -> 形态 -> 宽候选池 -> 冷却
    pool = df[hard_filter_mask(df)].copy()
    log(f"pool (hard filter): {len(pool)}")

    tr, cond_a = trend_mask(pool)
    tp = pool[tr]
    log(f"trend pass: {len(tp)}")

    pat, subtype = pattern_mask(tp)
    a1 = tp[pat].copy()
    a1["subtype"] = np.asarray(subtype)[pat.values]
    a1["trend_cond_a"] = cond_a[tr].values[pat.values]
    a1 = a1[a1["atr_ref"].notna() & (a1["atr_ref"] > 0)]
    log(f"broad candidate events (pre-cooldown): {len(a1)}")

    # ------------------------------------------------ labels (on pre-cooldown events)
    log("building bar map for label computation ...")
    bar_cols = ["open", "high", "low", "close", "ret1", "_lock", "date"]
    bar_map = {code: {c: grp[c].values for c in bar_cols}
               for code, grp in df.groupby("code", sort=False)}
    a1 = a1.sort_values(["code", "pos"], kind="mergesort").reset_index(drop=True)
    a1["entry"] = a1["close"]
    log(f"computing labels for {len(a1)} events (MAX_HOLD={MAX_HOLD}) ...")
    labels, path = compute_labels(a1, bar_map)
    a1 = pd.concat([a1.reset_index(drop=True), labels], axis=1)
    a1["event_id"] = np.arange(len(a1))  # 原始（冷却前）事件 id
    path.insert(0, "event_id", a1["event_id"].values)

    # v3.1-4 ATR 最大风险资格
    for ks in K_STOPS:
        a1[f"stop_pct_s{ks}"] = (ks * a1["atr_ref"] / a1["entry"]).astype(np.float32)
        a1[f"eligible_s{ks}"] = a1[f"stop_pct_s{ks}"] <= STOP_PCT_CAP

    a1.to_parquet(EVENTS_RAW_PATH, index=False)
    path.to_parquet(PATH_RAW_PATH, index=False)
    log(f"wrote {EVENTS_RAW_PATH}, {len(a1)} rows (+{PATH_RAW_PATH})")

    keep = apply_cooldown(a1)
    a1_cooled = a1[keep].reset_index(drop=True)
    # 冷却后事件 id：与 build_events_train_v1.py 的 (code,date) 排序 + arange 约定一致
    a1_cooled["event_id"] = np.arange(len(a1_cooled))
    a1_cooled.to_parquet(EVENTS_COOLED_PATH, index=False)
    id_map = a1_cooled[["event_id"]]  # 新 id（行号）
    id_map["eid_raw"] = a1["event_id"].values[keep]
    path_cooled = path.merge(id_map, left_on="event_id", right_on="eid_raw")
    path_cooled = path_cooled.drop(columns=["event_id_x", "eid_raw"]) \
        .rename(columns={"event_id_y": "event_id"})[["event_id", "entry",
                                                     "c1", "c2", "c3", "c4", "c5"]]
    path_cooled = path_cooled.sort_values("event_id").reset_index(drop=True)
    # exit_day 各组合列随路径表落盘（组合回测结算用）
    exit_cols = [f"exit_day_s{ks}_t{kt}" for ks in K_STOPS for kt in K_TARGETS]
    path_cooled = path_cooled.merge(a1_cooled[["event_id"] + exit_cols],
                                    on="event_id", how="left")
    path_cooled.to_parquet(PATH_COOLED_PATH, index=False)
    log(f"wrote {EVENTS_COOLED_PATH}, {len(a1_cooled)} rows (+{PATH_COOLED_PATH})")

    # 旧顺序（冷却在硬过滤后）参照值，用于与 v3.0 口径对比说明
    pool_sorted = pool.sort_values(["code", "pos"], kind="mergesort")
    keep_old = apply_cooldown(pool_sorted)
    pool_cooled_old = pool_sorted[keep_old]
    tr_old, _ = trend_mask(pool_cooled_old)
    tp_old = pool_cooled_old[tr_old]
    pat_old, _ = pattern_mask(tp_old)
    a1_old = tp_old[pat_old]
    a1_old = a1_old[a1_old["atr_ref"].notna() & (a1_old["atr_ref"] > 0)]
    log(f"[reference] old-order (cooldown before trend/pattern) A1: {len(a1_old)}")

    # ------------------------------------------------ report helpers
    years = list(range(2019, 2027))
    trade_days = df.groupby("year")["date"].nunique()

    def yr_counts(frame):
        return frame.groupby("year").size()

    pool_y = yr_counts(pool)
    trend_y = yr_counts(tp)
    a1_raw_y = yr_counts(a1)
    a1_cooled_y = yr_counts(a1_cooled)

    TAG15 = "s1.5_t2.0"

    def dist_stats(ev, mask_eligible=None):
        """(N, tgt%, stop%, tmo%, censored%, ev) ；CENSORED/NO_FUTURE 不计入占比与 EV"""
        if mask_eligible is not None:
            ev = ev[mask_eligible]
        oc = ev[f"outcome_{TAG15}"]
        cens = ev["is_censored"] | (oc == "NO_FUTURE")
        counted = ~cens
        n = len(ev)
        n_cens = int(cens.sum())
        oc_c = oc[counted]
        nr = ev[f"net_ret_{TAG15}"][counted]
        n_ct = max(int(counted.sum()), 1)
        return (n,
                float((oc_c == "TARGET").sum() / n_ct * 100),
                float((oc_c == "STOP").sum() / n_ct * 100),
                float((oc_c == "TIMEOUT").sum() / n_ct * 100),
                n_cens / n * 100 if n else 0.0,
                float(nr.mean()) if n_ct else np.nan)

    # ------------------------------------------------ report
    lines = []
    lines.append("# 绿盘低吸 - 年度事件漏斗统计（第3步 v3.1）\n")
    lines.append("口径：日线近似。信号日 T 收盘后出信号，entry=T 日收盘价（尾盘近似），"
                 f"退出评估从 T+1 开始，最长持有 {MAX_HOLD} 个交易日。\n")
    lines.append("**重要：本报告产出的是全市场宽股票候选池事件，不是真正的 A1 事件**"
                 "（真正的 A1 还需板块强势/梯队/龙头中军特征，属后续步骤）。"
                 "以下行文简称「宽候选池」。\n")

    lines.append("## 1. 数据概览\n")
    lines.append(f"- 行数：{len(df):,}（白名单 00/30/60/68 + 北交所 8/4/92 开头）")
    lines.append(f"- 股票数：{df['code'].nunique():,}")
    lines.append(f"- 日期范围：{df['date'].min()} ~ {df['date'].max()}")
    lines.append("- 年份分布（行数）：")
    lines.append("")
    lines.append("| 年份 | 行数 | 交易日数 |")
    lines.append("|---|---:|---:|")
    for y in sorted(df["year"].unique()):
        lines.append(f"| {y} | {int((df['year']==y).sum()):,} | {trade_days.get(y,0)} |")
    lines.append("")

    lines.append("## 2. 年度事件漏斗（v3.1 顺序：硬过滤→趋势→形态→冷却）\n")
    lines.append("宽松池 = 硬过滤（len>=60、非ST、float_mv 10~100亿、total_mv<=100亿、"
                 "adv20>=5000万、非涨停日）。冷却 = 同股 5 个交易日内只算第一次事件，"
                 "应用在形态关之后。\n")
    lines.append("| 年份 | 交易日数 | 宽松池 | 过趋势关 | 宽候选池(未冷却) | 宽候选池(冷却后) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    tot = [0, 0, 0, 0]
    for y in years:
        a = int(pool_y.get(y, 0)); b = int(trend_y.get(y, 0))
        c2 = int(a1_raw_y.get(y, 0)); d = int(a1_cooled_y.get(y, 0))
        tot[0] += a; tot[1] += b; tot[2] += c2; tot[3] += d
        lines.append(f"| {y} | {trade_days.get(y,0)} | {a:,} | {b:,} | {c2:,} | {d:,} |")
    lines.append(f"| 合计 | {int(trade_days.reindex(years).fillna(0).sum())} "
                 f"| {tot[0]:,} | {tot[1]:,} | {tot[2]:,} | {tot[3]:,} |")
    lines.append("")
    old_1923 = int(a1_old[(a1_old["year"] >= 2019) & (a1_old["year"] <= 2023)].shape[0])
    new_1923 = int(a1_cooled[(a1_cooled["year"] >= 2019) & (a1_cooled["year"] <= 2023)].shape[0])
    lines.append(f"> 口径对照：旧顺序（冷却在硬过滤后、趋势/形态前）冷却后 2019-2023 合计 "
                 f"{old_1923:,}；新顺序（冷却在形态关后）冷却后 2019-2023 合计 {new_1923:,}。"
                 "冷却移后只会增多不会减少：旧顺序每个 5 日窗口只在首个宽松池日判定形态，"
                 "该日不过形态则整个窗口作废；新顺序在窗口内每个形态通过日均可成事件。")
    lines.append("")

    lines.append(f"## 3. 三分类标签分布（K_stop=1.5, K_target=2.0，冷却后）\n")
    lines.append("eligible 口径 = stop_pct = K_stop×ATR_ref/entry ≤ 6%（A 模式最大风险上限），"
                 "为正式口径；标签对全部事件计算，不因 eligible 删除。"
                 "CENSORED = 信号后未来交易日不足 5 天（数据末尾删失），不计入 TIMEOUT、不进 EV。\n")
    lines.append("| 年份 | 口径 | 事件数 | TARGET% | STOP% | TIMEOUT% | CENSORED数 | 裸EV |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    rows_1923_note = []
    for y in years + ["合计"]:
        if y == "合计":
            ev = a1_cooled
        else:
            ev = a1_cooled[a1_cooled["year"] == y]
            if len(ev) == 0:
                continue
        for lab, mk in [("全部样本", None), ("eligible", ev[f"eligible_s1.5"].values)]:
            n, tgt, stp, tmo, cens, evr = dist_stats(ev, mk)
            lines.append(f"| {y} | {lab} | {n:,} | {tgt:.1f} | {stp:.1f} "
                         f"| {tmo:.1f} | {int(round(cens/100*n)):,} | {evr*100:.2f}% |")
    lines.append("")
    elig_share_all = float(a1_cooled["eligible_s1.5"].mean() * 100)
    lines.append(f"- 冷却后 eligible(K_stop=1.5) 占比：{elig_share_all:.1f}%"
                 f"（各 K 占比："
                 + ", ".join(f"K={ks}: {float(a1_cooled[f'eligible_s{ks}'].mean()*100):.1f}%"
                             for ks in K_STOPS)
                 + "）")
    lines.append("")

    lines.append("## 4. 分年 × K_stop 三分类分布（冷却后，eligible 口径，K_target=2.0）\n")
    lines.append("| 年份 | K_stop | eligible事件数 | TARGET% | STOP% | TIMEOUT% | CENSORED% | ambiguous% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for y in years:
        ev = a1_cooled[a1_cooled["year"] == y]
        if len(ev) == 0:
            continue
        for ks in K_STOPS:
            t = f"s{ks}_t2.0"
            sub = ev[ev[f"eligible_s{ks}"]]
            oc = sub[f"outcome_{t}"]
            cens = sub["is_censored"] | (oc == "NO_FUTURE")
            counted = ~cens
            n0 = len(sub)
            n_ct = max(int(counted.sum()), 1)
            tgt = float((oc[counted] == "TARGET").sum() / n_ct * 100)
            stp = float((oc[counted] == "STOP").sum() / n_ct * 100)
            tmo = float((oc[counted] == "TIMEOUT").sum() / n_ct * 100)
            csp = float(cens.sum() / n0 * 100) if n0 else 0.0
            amb = float(sub[f"ambig_{t}"][counted].mean() * 100)
            lines.append(f"| {y} | {ks} | {n0:,} | {tgt:.1f} | {stp:.1f} "
                         f"| {tmo:.1f} | {csp:.1f} | {amb:.2f} |")
    lines.append("")

    sub_cnt = a1_cooled["subtype"].value_counts()
    lines.append(f"- 冷却后宽候选池形态子类型分布："
                 + ", ".join(f"{k}={v:,}" for k, v in sub_cnt.items()))
    n_cens_total = int(a1_cooled["is_censored"].sum())
    lines.append(f"- 冷却后 CENSORED 总数：{n_cens_total:,}"
                 f"（占 {n_cens_total/max(len(a1_cooled),1)*100:.1f}%）")
    lines.append("")

    lines.append("## 5. 板块分布说明\n")
    lines.append("本数据无板块/行业字段，以上为**全市场宽股票候选池口径、未按板块过滤**，"
                 "也不是真正的 A1 事件（缺板块强势/梯队/龙头中军特征）；"
                 "实际按板块扫描时数量会更少。板块特征接入是后续步骤。\n")

    lines.append("## 6. 主要近似口径声明\n")
    lines.append(f"- 主标签持有窗口 MAX_HOLD={MAX_HOLD} 个交易日（10 日口径仅用于消融实验）。")
    lines.append("- v3.3：源数据为前复权 OHLC，源 pre_close 为未复权价（2018 年 83% 行偏差>1%，"
                 "随时间收敛），已弃用源 pre_close，统一由同代码前一交易日复权 close 推导；"
                 "首日无前收盘的行剔除。除权日收益因此按复权口径计，与 OHLC 一致。")
    lines.append("- 白名单只保留 00/30/60/68 + 北交所（8/4/92 开头）；"
                 "涨跌停与跌停锁死按板块口径：60/00=9.5%，30/68=19.5%，北交所=29.5%"
                 "（锁死 = ret1 ≤ -limit+0.5% 且 close==low）。")
    lines.append("- entry = T 日收盘价（尾盘近似），无分钟数据，无法模拟真实排队成交。")
    lines.append("- 连续跌停顺延：锁死日记不可成交，恢复可卖日按实际可成交价退出"
                 "（open 低于止损按 open，否则按止损价）；窗口末日仍锁死按最后可得 close 近似成交"
                 "（fill 列标注 stop_approx_close）。")
    lines.append("- 同一根 K 线同时触及止损 S 与目标 T_k 时，按悲观口径记 STOP_FIRST 并计 ambiguous。")
    lines.append("- 目标/止损/超时收益均扣除往返成本 0.2%。")
    lines.append("- 年度归属按信号年；2023 年末信号的 label_end_date 会落在 2024，"
                 "正式切分时需按 label_end_date 过滤（切分逻辑属训练阶段，本脚本只落字段）。")
    lines.append("- CENSORED（available_future_days<5）事件的标签仍计算供审计，"
                 "但正式统计单列 CENSORED，不计入 TIMEOUT、不进 EV。")
    lines.append("- ATR_ref = 截至 T-1 的前 5 日 ATR14 中位数，防当日失真。")
    lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"wrote {REPORT_PATH}")
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
