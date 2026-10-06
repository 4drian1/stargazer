# -*- coding: utf-8 -*-
"""盘中触发式 T5 打分（v2，冻结方案）单文件实现。

三件事：
  1. compute_features()：从当日（截至决策点 dp）分钟 K 线 + 历史分钟
     重建冻结四头（AM_R1/PM_R2/AM_O4/PM_O6）所需的全部特征。
     特征公式一律复用离线 builder（minute_label_engine / path_features /
     t2pm_features / t6_features），不重写，保证口径一致。
  2. --audit：从离线真值表随机抽样 (code, dp_date, dp)，用分钟缓存 replay
     当日分钟、截断到 dp、重算特征，与离线值逐列比对（放行门槛）。
  3. --score-live：盘中实时打分，触发时间向下对齐到最近已收盘决策点
     （09:30-10:00→1000，10:00-10:30→1030，13:00-14:00→1400，
       14:00-14:30→1430，14:30-14:50→1450，其余 NO_SCORE），
     取数走 stockdb 本地服务（模块内置最小取数函数），打分写 shadow_decisions。
  4. --backfill-live：盘中 LIVE 触发行的盘后收益回填。对 shadow_decisions 中
     src='LIVE' 且未回填（nr_5d/outcome_t2 均 NULL）且 dp_date 距今 ≥7 自然日
     的行，从 stockdb 取 dp_date 起的分钟，复用 minute_label_engine.scan_exit()
     扫 T+2 起 3/5 个可卖交易日（kt=2.0 主场景：同分钟双触悲观记 STOP、跌停
     锁死日递延、缺口成交、超时按最后收盘、扣 COST——与离线
     dual_head_labels 的 o_net_ret_3d/5d、o_stop_5d、outcome_t2、censored
     同一口径），UPDATE 回 nr_3d/nr_5d/stop_5d/outcome_t2/censored/filled_at。
     3 日账本 = 把 cm 截断到 dp_date+3 个交易日再调 scan_exit（与离线
     scan_ledger(n_days=3) 逐条等价）。--dry-run 只算不写。
  5. --selftest：本地缓存 replay 一个已知键走完整 score 路径（不连网不写库）。

== 冻结口径要点 ==

- 模型特征共 46 个（open15 十二 + pf 七 + t2pm 十三 + t6 十一 + t4 三；
  任务书称 47，按 bundle["meta"][头]["features"] 实际并集为 46）。
- t4 三列：v2.0 盘中不算板块代理，固定用"不可用"编码，照抄离线表
  minute_t4_proxy_2026.parquet 中 t4_proxy_available==0 的行的编码：
  t4_proxy_available=0，t4_stock_excess_dp=NaN，t4_stock_excess_leader=NaN
  （已核验：不可用行这两列 100% 为 NaN，不是 0）。
- ATR_ref（= 特征归一的 atr_px，元/ATR 单位）：与 d0_daily_features.py
  同式 —— tr=max(h-l,|h-pc|,|l-pc|)，atr14=tr.rolling(14,min_periods=10).mean()，
  atr_ref = atr14.shift(1).rolling(5).median()。离线池在信号日 T 冻结、用于
  T+1（= Median(ATR14_{T-5:T-1})）；**盘中 D0 触发时按冻结方案取
  Median(ATR14_{D-5:D-1})**（只用 D-1 及之前日线）。注意离线窗口比决策日早
  两天（shift(1) 再取 5 日中位），盘中窗口只比决策日早一天，二者天然错位
  一个交易日——这是方案定义如此，已记录。
- **分布差异（与离线池不同，已记录）**：离线 OOD/展示层字段（runup_5d/10d/20d、
  pos20、rsi14、turnover_sum_5d、turnover_max_5d、limit_up_10d）用信号日 T
  （= D0 的前一交易日）收盘计算；盘中 D0 收盘不可得，本模块一律用 **D-1 及之前
  日线**计算（runup/turnover 类指标比离线池"旧一天"，attatch_ood 的阈值仍沿用
  ood_thresholds.json 冻结值，这是有意接受的分布偏移）。
- 象限规则照抄 shadow_batch.score()：低风险=P_risk≤base_rate、
  高风险=P_risk≥1.5×base_rate、高机会=P_opp≥0.5 → FOCUS/WAIT/WATCH/REJECT。
- OOD 判定照抄 shadow_batch.attach_ood()：runup_10d>p99_5 或 ≥3 个高热指标
  >各自 p99_5（高热集 8 个，含 rsi_over75_10d；NaN 视为未超）。
- **LIVE 回填口径（与离线标签的固定差异，已记录）**：离线标签 entry = DP 后
  第一根 bar 的 vwap，LIVE 回填 entry = 触发价（shadow_decisions.entry，
  即 score-live 的 --price）——按冻结 v2 方案，触发价即假定成交价；
  atr_pct = atr_ref/pre_close（离线 = atr_ref/信号日收盘，同为"ATR 价位 /
  Entry 日前一收盘"语义）；ks 用 1.5（离线 B0/LEGACY 口径；A2 的 1.2 仅
  训练池 A2 来源使用，LIVE 无来源可分，取保守档 1.5；shadow_decisions
  无 ks 列，此选择已固化进 nr_3d/nr_5d 语义）。

== 数据源 ==

- 分钟缓存（audit/selftest）：MINUTE_CACHE_DIR 环境变量，默认
  minute_cache_clean（与全部离线 builder 的 CACHE 默认一致；
  minute_cache 与 minute_cache_clean 的 volume 在多数票上不同源，
  audit 模式 --cache-dir 可切换复核）。
- 日线（live ATR_ref 与 OOD 字段）：STOCKDB_DATA_ROOT 环境变量，默认
  D:/a股数据，读 data_daily_parts/part_*.parquet（pyarrow 按 code 过滤）。
- 盘中分钟/日线（live）：stockdb 本地服务。注意 stockdb 的线上协议是
  **原生 TCP 二进制**（stockdb.pyd），并非 HTTP；本模块内嵌最小取数函数
  （service_available/fetch_minute/fetch_daily，实现思路照抄
  v4_2_stockdb.py 但**不 import 它**），用 stdlib importlib 直接从
  STOCKDB_ROOT（默认 D:/a股数据/free-stockdb-win/stockdb）的 pybao 目录
  加载 stock_sdk，只依赖标准库 + numpy + pandas。
  ⚠ 风险记录：服务未运行时无法核验 SDK 1m 返回的 volume 单位（股/手）。
  若 volume 为手，vwap（amount/volume）与离线口径差 100 倍；上线前须用
  --score-live 对一只已知票与 minute_cache_clean 同一日对账一次。
"""
import argparse
import glob
import importlib.util
import json
import os
import pickle
import socket
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 离线 builder（特征公式的唯一来源，直接 import 复用）
import minute_label_engine as mle          # dp_features / hist_vol_ratio / open15 / load_code
import path_features                          # pf_ret1m..pf_has_two_lows
import t2pm_features                          # t2pm_*（PM 决策点）
import t6_features                            # t6_*（中轴/VWAP 特征组）

OUT_DIR = _HERE
PKL = os.path.join(OUT_DIR, "shadow_models", "shadow_t5_frozen.pkl")
OOD_PATH = os.path.join(OUT_DIR, "ood_thresholds.json")
SAMPLES_2026 = os.path.join(OUT_DIR, "minute_labels", "minute_samples_2026.parquet")
PF_2026 = os.path.join(OUT_DIR, "minute_path_features_2026.parquet")
TPM_2026 = os.path.join(OUT_DIR, "t2pm_features_2026.parquet")
T6_2026 = os.path.join(OUT_DIR, "t6_features_2026.parquet")

MINUTE_CACHE_DIR = os.environ.get(
    "MINUTE_CACHE_DIR", os.path.join(OUT_DIR, "minute_cache_clean"))
STOCKDB_DATA_ROOT = os.environ.get("STOCKDB_DATA_ROOT", "D:/a股数据")
DATA_GLOB = os.path.join(STOCKDB_DATA_ROOT, "data_daily_parts", "part_*.parquet")


def _discover_stockdb_root() -> str:
    """stockdb 服务目录定位：env > 本机默认 > 包同级 free-stockdb* 发行版。
    兼容两种目录形态（<root>/stockdb/pybao 与 <root>/pybao）。"""
    def _has_sdk(root: str) -> bool:
        return os.path.isfile(os.path.join(root, "pybao", "stock_sdk.py"))
    env = os.environ.get("STOCKDB_ROOT")
    if env and _has_sdk(env):
        return env
    # 本机常见形态：引擎数据根直接 <STOCKDB_DATA_ROOT>/stockdb（如 D:/a股数据/stockdb）
    direct = os.path.join(STOCKDB_DATA_ROOT, "stockdb")
    if _has_sdk(direct):
        return direct
    local = os.path.join(STOCKDB_DATA_ROOT, "free-stockdb-win", "stockdb")
    if _has_sdk(local):
        return local
    # 包根（pullback_train 的上上级，如 .../stargazer）的同级/trading 目录
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for base in (here, os.path.dirname(here)):
        for cand in sorted(glob.glob(os.path.join(base, "free-stockdb*"))) \
                + sorted(glob.glob(os.path.join(base, "free-stockdb*", "stockdb"))):
            if _has_sdk(cand):
                return cand
    return local  # 都找不到时给默认（service_available 会报不可用）


STOCKDB_ROOT = _discover_stockdb_root()
STOCKDB_HOST = "127.0.0.1"
STOCKDB_PORT = 7899

HEADS = {"AM": (1000, 1030), "PM": (1330, 1400, 1430, 1450)}
# t4 三列"不可用"编码（照抄 minute_t4_proxy_2026.parquet available==0 的行）
T4_UNAVAILABLE = {"t4_proxy_available": 0.0,
                  "t4_stock_excess_dp": np.nan,
                  "t4_stock_excess_leader": np.nan}
OPEN15_12 = ["downside_atr", "shock_atr", "recovery", "lower_wick_atr",
             "vwap_reclaim_atr", "kill_speed_atr", "first15_low_atr",
             "first15_range_atr", "vol_ratio15", "new_low_after_945",
             "dip_hold", "bars"]
PF_7 = ["pf_ret1m", "pf_ret3m", "pf_ret5m", "pf_decline_speed", "pf_decel",
        "pf_low2_minus_low1", "pf_has_two_lows"]
T2PM_13 = t2pm_features.FEATS  # 13 列
PM_T6_11 = ["t6_dist_mid_s_atr", "t6_dist_vwap_atr", "t6_space_to_upper_atr",
            "t6_above_upper_depth_atr", "t6_since_reclaim_min",
            "t6_hold_after_reclaim_min", "t6_upper_touch", "t6_upper_mode",
            "t6_width_atr", "t6_width_chg_2h_atr", "t6_volsur_reclaim"]
T6_ALL = [c for c in (
    "t6_dist_mid_s_atr", "t6_dist_mid_l_atr", "t6_dist_vwap_atr",
    "t6_below_lower_depth_atr", "t6_above_upper_depth_atr",
    "t6_space_to_upper_atr", "t6_width_atr", "t6_width_chg_2h_atr",
    "t6_mid_slope_atr_h", "t6_below_now_min", "t6_since_first_below_min",
    "t6_since_reclaim_min", "t6_hold_after_reclaim_min", "t6_last_cross_dir",
    "t6_low2_minus_low1_atr", "t6_upper_touch", "t6_upper_mode",
    "t6_volsur_break", "t6_volsur_reclaim")]
HEAT8 = ("runup_10d", "turnover_sum_5d", "rsi14", "pos20", "limit_up_10d",
         "turnover_max_5d", "runup_20d", "rsi_over75_10d")
# audit 比对的列 = 全部非 t4 模型特征（open15 十二 + pf 七 + t2pm 十三 + t6 十一）
AUDIT_COLS = OPEN15_12 + PF_7 + T2PM_13 + PM_T6_11


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- cm 结构
def _cm_from_arrays(dates, o, h, l, c, v, a):
    """与 minute_label_engine.load_code 输出同结构的 cm dict（单代码）。

    dates: int64 YYYYMMDDHHMMSS（须已按时间升序）。不含隔离剔除
    （隔离是离线治理概念；live 无隔离；audit 走 load_code 自带隔离）。
    """
    days = (dates // 1000000).astype(int)
    uniq, inv = np.unique(days, return_inverse=True)
    out = {"days": uniq, "days_all": uniq, "day_idx": inv,
           "t": dates % 1000000,
           "o": o.astype(np.float64), "h": h.astype(np.float64),
           "l": l.astype(np.float64), "c": c.astype(np.float64),
           "v": v.astype(np.float64), "a": a.astype(np.float64)}
    dc = np.full(len(uniq), np.nan)
    do = np.full(len(uniq), np.nan)
    dh = np.full(len(uniq), np.nan)
    dl = np.full(len(uniq), np.nan)
    for i in range(len(uniq)):
        m = inv == i
        dc[i] = out["c"][m][-1]
        do[i] = out["o"][m][0]
        dh[i] = out["h"][m].max()
        dl[i] = out["l"][m].min()
    out["day_close"] = dc
    out["day_open"] = do
    out["day_high"] = dh
    out["day_low"] = dl
    return out


def cm_from_minutes_df(df, trade_date, dp):
    """minutes_df（minute_cache 列结构：date,open,high,low,close,volume,amount；
    可含历史交易日 + 当日未截断）→ (cm, day_i)。当日被截断到 t<=dp*100。

    历史交易日用于 t6 的 15 分钟槽尾窗/20 日量能中位基线与 hist_vol_ratio，
    至少应含 20 个 prior 交易日（live 取数建议 ≥25 个交易日）。
    """
    df = df.sort_values("date", kind="mergesort")
    dates = df["date"].to_numpy(np.int64)
    days = dates // 1000000
    keep = (days != int(trade_date)) | (dates % 1000000 <= dp * 100)
    if not (days[keep] == int(trade_date)).any():
        return None, None
    cm = _cm_from_arrays(dates[keep],
                         df["open"].to_numpy(np.float64)[keep],
                         df["high"].to_numpy(np.float64)[keep],
                         df["low"].to_numpy(np.float64)[keep],
                         df["close"].to_numpy(np.float64)[keep],
                         df["volume"].to_numpy(np.float64)[keep],
                         df["amount"].to_numpy(np.float64)[keep])
    day_i = int(np.where(cm["days"] == int(trade_date))[0][0])
    return cm, day_i


def replay_cm(code, dp_date, dp, cache_dir=None):
    """audit 路径：从分钟缓存整载 code（走 minute_label_engine.load_code，
    自带 quarantine 剔除，与离线 builder 完全同源），截断 dp_date 到 dp。"""
    prev_cache = mle.CACHE
    if cache_dir:
        mle.CACHE = cache_dir
        mle._code_cache.clear()
    try:
        cm = mle.load_code(str(code))
    finally:
        if cache_dir:
            mle.CACHE = prev_cache
            mle._code_cache.clear()
    if cm is None:
        return None, None
    hit = np.where(cm["days"] == int(dp_date))[0]
    if not len(hit):
        return None, None
    day_i = int(hit[0])
    keep = (cm["day_idx"] != day_i) | (cm["t"] <= dp * 100)
    out = {k: cm[k][keep] for k in ("day_idx", "t", "o", "h", "l", "c", "v", "a")}
    out["days"] = cm["days"]
    out["days_all"] = cm["days_all"]
    for k in ("day_close", "day_open", "day_high", "day_low"):
        out[k] = cm[k]          # 目标日的这些聚合值不在本模块特征路径上使用
    return out, day_i


# ---------------------------------------------------------------- 特征
def compute_features(code, trade_date, dp, minutes_df, pre_close,
                     atr_px=None, daily_hist=None, atr_pct=None):
    """46 个冻结特征（t4 三列为"不可用"编码）。

    minutes_df: 见 cm_from_minutes_df（当日 + ≥20 个 prior 交易日分钟，
    当日会被内部截断到 ≤dp）。pre_close = 前一交易日收盘（raw 尺度）。
    atr_px: ATR_ref 价位（元/ATR）。不传则从 daily_hist 按冻结公式
    Median(ATR14_{D-5:D-1}) 计算（d0_daily_features 同式）。
    atr_pct: t6 组归一用的 ATR 比例。**口径差异（照抄离线 builder）**：
    离线 t6_features 用 atr_pct=atr_ref/sig_close（sig_close 为 qfq 尺度），
    而 pf/t2pm/open15 用 atr_px=atr_ref=atr_pct×sig_close；盘中无 qfq 概念，
    live 应传 atr_px/pre_close（raw 自洽）。审计时传离线表 atr_pct 以逐位复刻。
    返回 None 表示当日 dp 前无分钟数据，无法打分。
    """
    dp = int(dp)
    if atr_px is None:
        if daily_hist is None:
            raise ValueError("atr_px 与 daily_hist 至少提供一个")
        atr_px = atr_ref_asof(daily_hist, trade_date)
    if atr_pct is None:
        atr_pct = float(atr_px) / float(pre_close)
    if not (np.isfinite(atr_px) and atr_px > 0):
        return None
    cm, day_i = cm_from_minutes_df(minutes_df, trade_date, dp)
    if cm is None:
        return None
    feats = {}

    # ---- open15 十二列（dp_features + hist_vol_ratio，口径与引擎一致）----
    f = mle.dp_features(cm, day_i, dp, float(pre_close), float(atr_px))
    if f is None:
        return None
    for k in OPEN15_12:
        if k == "vol_ratio15":
            continue
        feats[k] = f[k]
    feats["vol_ratio15"] = mle.hist_vol_ratio(cm, day_i, f["first15_vol"])

    # ---- pf 七列（path_features.compute_path_features）----
    m = (cm["day_idx"] == day_i) & (cm["t"] <= dp * 100)
    p = cm["c"][m]                       # cm 已按时间升序
    feats.update(path_features.compute_path_features(p, float(atr_px)))

    # ---- t2pm 十三列（仅 PM 决策点；AM 头不用）----
    if dp >= 1330:
        feats.update(t2pm_features.compute(cm, day_i, dp, float(atr_px)))

    # ---- t6（中轴/VWAP 组；normalize 的 atr_pct 见 docstring 口径说明）----
    raw = t6_features.compute_code(cm, {day_i})
    t6 = {k: np.nan for k in T6_ALL}
    if len(raw):
        norm = t6_features.normalize(raw, {int(trade_date): float(atr_pct)})
        row = norm[norm["dp"] == dp]
        if len(row):
            r = row.iloc[0]
            for k in T6_ALL:
                t6[k] = r[k]
    feats.update(t6)

    # ---- t4 三列：v2.0 不可用编码 ----
    feats.update(T4_UNAVAILABLE)
    return feats


# ---------------------------------------------------------------- 日线（ATR_ref + OOD 字段）
def load_daily_hist(code, data_glob=None):
    """STOCKDB_DATA_ROOT/data_daily_parts/part_*.parquet 按 code 过滤读出
    全部日线（与 d0_daily_features/t7_features 同数据源）。返回按 date 升序
    的 DataFrame（open/high/low/close/pre_close/volume/amount/turnover/is_st）。"""
    files = sorted(glob.glob(data_glob or DATA_GLOB))
    if not files:
        raise FileNotFoundError(data_glob or DATA_GLOB)
    cols = ["date", "code", "open", "high", "low", "close", "pre_close",
            "volume", "amount", "turnover", "is_st"]
    try:
        import pyarrow.dataset as ds
        t = ds.dataset(files, format="parquet").to_table(
            filter=ds.field("code") == str(code), columns=cols)
        df = t.to_pandas()
    except ImportError:
        frames = [pd.read_parquet(f, columns=cols) for f in files]
        df = pd.concat(frames, ignore_index=True)
        df = df[df["code"].astype(str) == str(code)]
    df = df.dropna(subset=["close", "high", "low", "open"])
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    df["code"] = df["code"].astype(str)
    prev_close = df.groupby("code", sort=False)["close"].shift(1)
    df["pre_close"] = df["pre_close"].fillna(prev_close)
    df = df.dropna(subset=["pre_close"]).reset_index(drop=True)
    return df


def _board_lim(code, is_st):
    """照抄 t7_features.lim_of。"""
    if is_st:
        return 0.05
    if str(code).startswith("68"):
        return 0.20
    if str(code).startswith(("30", "8", "4", "92")):
        return 0.30
    return 0.10


def daily_asof(code, daily, trade_date):
    """截至 D-1（含）的日线背景：ATR_ref（Median(ATR14_{D-5:D-1})）+
    OOD/展示层八字段（runup_5d/10d/20d、pos20、rsi14、rsi_over75_10d、
    limit_up_10d、turnover_sum_5d、turnover_max_5d）+ pre_close。

    公式逐行照抄 t7_features.build_background / d0_daily_features，
    全部只用 ≤D-1 的数据（与离线池用信号日 T=D-1 收盘一致；盘中无 D0 收盘，
    分布差异见模块 docstring）。返回 dict；历史不足处为 NaN。"""
    sub = daily[daily["date"] <= _prev_date(daily, trade_date)]
    out = {"asof_date": int(sub["date"].iloc[-1]) if len(sub) else None,
           "pre_close": float(sub["close"].iloc[-1]) if len(sub) else np.nan}
    if not len(sub):
        for k in ("atr_ref", "runup_5d", "runup_10d", "runup_20d", "pos20",
                  "rsi14", "rsi_over75_10d", "limit_up_10d",
                  "turnover_sum_5d", "turnover_max_5d"):
            out[k] = np.nan
        return out
    d = sub.copy()
    for c in ["open", "high", "low", "close", "pre_close", "turnover"]:
        d[c] = d[c].astype(np.float32)
    d["ret1"] = d["close"] / d["pre_close"] - 1
    d["is_st"] = d["is_st"].fillna(0).astype(int) if "is_st" in d \
        else pd.Series(0, index=d.index)
    d["lim"] = [_board_lim(code, s) for s in d["is_st"]]
    d["limit_up"] = d["ret1"] >= d["lim"] - 0.005
    g = d.groupby("code", sort=False)

    for n in (5, 10, 20):
        d[f"runup_{n}d"] = d["close"] / g["close"].shift(n) - 1
    lo = g["low"].transform(lambda s: s.rolling(20, min_periods=20).min())
    hi = g["high"].transform(lambda s: s.rolling(20, min_periods=20).max())
    d["pos20"] = (d["close"] - lo) / (hi - lo)
    delta = g["close"].diff()
    up = delta.clip(lower=0).groupby(d["code"], sort=False) \
        .transform(lambda s: s.rolling(14).mean())
    dn = (-delta).clip(lower=0).groupby(d["code"], sort=False) \
        .transform(lambda s: s.rolling(14).mean())
    d["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    d["rsi_over75_10d"] = (d["rsi14"] > 75).groupby(d["code"], sort=False) \
        .transform(lambda s: s.rolling(10).sum())
    d["limit_up_10d"] = d["limit_up"].groupby(d["code"], sort=False) \
        .transform(lambda s: s.rolling(10).sum())
    d["turnover_sum_5d"] = g["turnover"].transform(lambda s: s.rolling(5).sum())
    d["turnover_max_5d"] = g["turnover"].transform(lambda s: s.rolling(5).max())
    tr = pd.concat([d["high"] - d["low"],
                    (d["high"] - d["pre_close"]).abs(),
                    (d["low"] - d["pre_close"]).abs()], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14, min_periods=10).mean()
    last = d.iloc[-1]
    atr14_tail = d["atr14"].dropna().tail(5)
    out["atr_ref"] = float(atr14_tail.median()) if len(atr14_tail) else np.nan
    for k in ("runup_5d", "runup_10d", "runup_20d", "pos20", "rsi14",
              "rsi_over75_10d", "limit_up_10d", "turnover_sum_5d",
              "turnover_max_5d"):
        out[k] = float(last[k]) if pd.notna(last[k]) else np.nan
    return out


def _prev_date(daily, trade_date):
    """严格早于 trade_date 的最后一个交易日（D-1）。"""
    d = daily[daily["date"] < int(trade_date)]
    if not len(d):
        raise ValueError(f"日线历史中无 {trade_date} 之前的交易日")
    return int(d["date"].iloc[-1])


def atr_ref_asof(daily, trade_date):
    """Median(ATR14_{D-5:D-1})（= d0_daily_features 的 atr_ref 在决策日
    当行的取值，tr 公式与滚动参数完全一致）。"""
    d = daily[daily["date"] < int(trade_date)].copy()
    tr = pd.concat([d["high"] - d["low"],
                    (d["high"] - d["pre_close"]).abs(),
                    (d["low"] - d["pre_close"]).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=10).mean()
    tail = atr14.dropna().tail(5)
    return float(tail.median()) if len(tail) else np.nan


# ---------------------------------------------------------------- stockdb 最小取数
_sdk = None


def service_available(timeout=1.0):
    """stockdb 本地服务存活探测（照抄 v4_2_stockdb.service_available）。"""
    try:
        with socket.create_connection((STOCKDB_HOST, STOCKDB_PORT), timeout=timeout):
            return True
    except OSError:
        return False


def _load_sdk():
    """从 STOCKDB_ROOT/pybao 动态加载 stock_sdk（不 import v4_2_stockdb，
    不依赖 stargazer_ml 根目录）。stockdb 线上协议为原生 TCP 二进制，
    须经 stockdb.pyd 通信，无法用裸 HTTP 替代。"""
    global _sdk
    if _sdk is not None:
        return _sdk
    pybao = os.path.join(STOCKDB_ROOT, "pybao")
    sdk_path = os.path.join(pybao, "stock_sdk.py")
    if not os.path.isfile(sdk_path):
        raise FileNotFoundError(sdk_path)
    if not service_available():
        raise ConnectionError(
            f"free-stockdb 未在 {STOCKDB_HOST}:{STOCKDB_PORT} 监听；"
            r"请先启动 stockdb.exe")
    if pybao not in sys.path:
        sys.path.insert(0, pybao)
    spec = importlib.util.spec_from_file_location("stock_sdk", sdk_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.init(host=STOCKDB_HOST, port=STOCKDB_PORT)
    _sdk = mod
    return mod


def _norm_code(code):
    return str(code).lower().removeprefix("sh").removeprefix("sz") \
        .removeprefix("bj").zfill(6)


def fetch_minute(code, start, end, fq=None):
    """stockdb 1 分钟 K 线 → minute_cache 同构 DataFrame
    （date:int64 YYYYMMDDHHMMSS, open/high/low/close:float64,
      volume/amount:float64）。start/end 为 'YYYYMMDD'。"""
    sdk = _load_sdk()
    df = sdk.rd.get_data(code=_norm_code(code), start=start, end=end,
                         frequency="1m", fq=fq, as_df=True)
    return _norm_bars_df(df)


def fetch_daily(code, start=None, end=None, fq=None):
    """stockdb 日线 → DataFrame（date:int YYYYMMDD + OHLCV/amount，
    保留 pre_close/turnover/is_st 供 daily_asof 使用）。"""
    sdk = _load_sdk()
    df = sdk.rd.get_data(code=_norm_code(code), start=start, end=end,
                         frequency="1d", fq=fq, as_df=True)
    return _norm_bars_df(df, extra=("pre_close", "turnover", "is_st"))


def _norm_bars_df(df, extra=()):
    """SDK 返回的 K 线 DataFrame 列名归一。date 归一为 int：
    分钟 → YYYYMMDDHHMMSS（14 位），日线 → YYYYMMDD（8 位）。"""
    if df is None or not len(df):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close",
                                     "volume", "amount"])
    df = df.copy()
    lowmap = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in lowmap:
                return df[lowmap[n]]
        return None
    if "date" in lowmap:
        dcol = df[lowmap["date"]]
    elif "time" in lowmap:
        dcol = df[lowmap["time"]]
    else:
        dcol = df.iloc[:, 0]
    if np.issubdtype(dcol.dtype, np.datetime64):
        dt = pd.to_datetime(dcol)
        dates = dt.dt.strftime("%Y%m%d%H%M%S").astype(np.int64)
    elif np.issubdtype(dcol.dtype, np.number):
        raw = dcol.to_numpy(np.int64)
        if raw.max() <= 99999999:            # 日线 YYYYMMDD
            dates = raw
        elif raw.max() < 10 ** 11:           # 秒级 epoch 兜底
            dates = (pd.to_datetime(dcol, unit="s")
                     .dt.strftime("%Y%m%d%H%M%S").astype(np.int64))
        else:                                # 分钟 YYYYMMDDHHMMSS
            dates = raw
    else:
        s = dcol.astype(str)
        dates = s.str.replace(r"[-:\s]", "", regex=True).str[:14] \
                 .astype(np.float64).astype(np.int64)
    out = pd.DataFrame({"date": dates})
    for dst, names in (("open", ("open",)), ("high", ("high",)),
                       ("low", ("low",)), ("close", ("close",)),
                       ("volume", ("volume", "vol")),
                       ("amount", ("amount", "amt"))):
        s = pick(*names)
        out[dst] = pd.to_numeric(s, errors="coerce").to_numpy(np.float64) \
            if s is not None else 0.0
    for e in extra:
        s = pick(e)
        if s is not None:
            out[e] = s.to_numpy()
    return out.sort_values("date", kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------- 打分
def load_bundle(pkl=PKL):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def score_features(bundle, dp, feats):
    """照抄 shadow_batch.score() 的单行版：返回 dict(P_risk/P_opp/quadrant/
    base_rate/ev3_ref/ev5_ref/risk_head/opp_head)。特征列缺列/全 NaN 时
    P 值为 NaN。"""
    dp = int(dp)
    head = "AM" if dp < 1200 else "PM"
    rk = f"{head}_R1" if head == "AM" else f"{head}_R2"
    ok = f"{head}_O4" if head == "AM" else f"{head}_O6"
    X = pd.DataFrame([feats])
    p_risk = p_opp = np.nan
    if len(X[bundle["meta"][rk]["features"]].dropna(how="all")):
        p_risk = float(bundle["calibrators"][rk].predict(
            bundle["models"][rk].predict_proba(
                X[bundle["meta"][rk]["features"]])[:, 1])[0])
    if len(X[bundle["meta"][ok]["features"]].dropna(how="all")):
        p_opp = float(bundle["calibrators"][ok].predict(
            bundle["models"][ok].predict_proba(
                X[bundle["meta"][ok]["features"]])[:, 1])[0])
    base = float(bundle["meta"][rk]["base_rate"])
    if not np.isfinite(p_opp):
        quad = "UNSCORED"
    else:
        lo_r, hi_r = p_risk <= base, p_risk >= 1.5 * base
        hi_o = p_opp >= 0.5
        quad = "FOCUS" if (lo_r and hi_o) else \
               "WAIT" if (hi_r and hi_o) else \
               "WATCH" if lo_r else "REJECT"
    ref = bundle.get("ev_ref", {}).get(head, {}).get(quad, {})
    return {"head": head, "risk_head": rk, "opp_head": ok,
            "P_risk": p_risk, "P_opp": p_opp, "base_rate": base,
            "quadrant": quad,
            "ev3_ref": ref.get("ev3_ref"), "ev5_ref": ref.get("ev5_ref")}


def attach_ood(feat_asof, ood_path=OOD_PATH):
    """照抄 shadow_batch.attach_ood()：runup_10d>p99_5 或 ≥3 个高热指标
    >各自 p99_5 → OOD。NaN 视为未超（离线为 fillna(False)）。"""
    with open(ood_path, encoding="utf-8") as f:
        th = json.load(f)["features"]
    hits = 0
    for c in HEAT8:
        if c in th:
            v = feat_asof.get(c, np.nan)
            if np.isfinite(v) and v > th[c]["p99_5"]:
                hits += 1
    v = feat_asof.get("runup_10d", np.nan)
    primary = bool(np.isfinite(v) and v > th["runup_10d"]["p99_5"])
    return {"ood": bool(primary or hits >= 3), "ood_hits": int(hits)}


def align_dp(hhmm):
    """触发时间(HHMM) → 最近已收盘决策点；None 表示无头覆盖（NO_SCORE）。"""
    hhmm = int(hhmm)
    if 930 <= hhmm <= 1000:
        return 1000
    if 1000 < hhmm <= 1030:
        return 1030
    if 1300 <= hhmm <= 1400:
        return 1400
    if 1400 < hhmm <= 1430:
        return 1430
    if 1430 < hhmm <= 1450:
        return 1450
    return None


# ---------------------------------------------------------------- audit
def load_truth_2026(n, seed=7):
    """从离线真值表抽样 n 个 (code, dp_date, dp)，返回 (keys, truth dict)。"""
    s = pd.read_parquet(SAMPLES_2026, columns=["code", "date", "dp_date", "dp",
                                               "atr_pct", "sig_close",
                                               "pre_close"] + OPEN15_12)
    s["code"] = s["code"].astype(str)
    key_cols = ["code", "dp_date", "dp"]
    keys = s[key_cols].drop_duplicates()
    pf = pd.read_parquet(PF_2026); pf["code"] = pf["code"].astype(str)
    tpm = pd.read_parquet(TPM_2026); tpm["code"] = tpm["code"].astype(str)
    t6 = pd.read_parquet(T6_2026); t6["code"] = t6["code"].astype(str)
    for m in (pf, tpm, t6):
        keys = keys.merge(m, on=key_cols, how="left")
    s1 = s.drop_duplicates(key_cols)
    keys = keys.merge(s1, on=key_cols, how="left")
    keys = keys.sample(n=min(n, len(keys)), random_state=seed) \
        .sort_values(key_cols).reset_index(drop=True)
    return keys


def run_audit(n=300, cache_dir=None, seed=7, verbose=True):
    keys = load_truth_2026(n, seed=seed)
    cache_dir = cache_dir or MINUTE_CACHE_DIR
    log(f"audit: 抽样 {len(keys)} 个 (code,dp_date,dp)；缓存目录 {cache_dir}")
    stat = {c: {"diffs": [], "hit": 0, "tot": 0} for c in AUDIT_COLS}
    miss_cache = 0
    no_bars = 0
    done = 0
    for _, r in keys.iterrows():
        code, dp_date, dp = str(r["code"]), int(r["dp_date"]), int(r["dp"])
        atr_px = float(r["atr_pct"]) * float(r["sig_close"])  # = 离线冻结 atr_ref
        cm, day_i = replay_cm(code, dp_date, dp, cache_dir=cache_dir)
        if cm is None:
            miss_cache += 1
            continue
        f = mle.dp_features(cm, day_i, dp, float(r["pre_close"]), atr_px)
        if f is None:
            no_bars += 1
            continue
        feats = {}
        for k in OPEN15_12:
            feats[k] = (mle.hist_vol_ratio(cm, day_i, f["first15_vol"])
                        if k == "vol_ratio15" else f[k])
        m = (cm["day_idx"] == day_i) & (cm["t"] <= dp * 100)
        feats.update(path_features.compute_path_features(cm["c"][m], atr_px))
        if dp >= 1330:
            feats.update(t2pm_features.compute(cm, day_i, dp, atr_px))
        raw = t6_features.compute_code(cm, {day_i})
        for k in T6_ALL:
            feats[k] = np.nan
        if len(raw):
            norm = t6_features.normalize(raw, {dp_date: float(r["atr_pct"])})
            row = norm[norm["dp"] == dp]
            if len(row):
                for k in T6_ALL:
                    feats[k] = row.iloc[0][k]
        done += 1
        for c in AUDIT_COLS:
            exp = r.get(c)
            got = feats.get(c, np.nan)
            if pd.isna(exp):
                continue
            stat[c]["tot"] += 1
            if pd.isna(got):
                continue
            d = abs(float(got) - float(exp))
            stat[c]["diffs"].append(d)
            if d <= 1e-6:
                stat[c]["hit"] += 1
    rows = []
    all_pass = True
    for c in AUDIT_COLS:
        s_ = stat[c]
        diffs = sorted(s_["diffs"], reverse=True)
        top2 = diffs[:2] if diffs else [np.nan, np.nan]
        cov = s_["hit"] / s_["tot"] if s_["tot"] else 0.0
        ok = (not diffs or top2[0] <= 1e-6) and cov >= 0.95
        all_pass &= ok
        rows.append({"feature": c, "max_diff": top2[0],
                     "second_diff": top2[1], "matched": s_["hit"],
                     "total": s_["tot"], "coverage": cov, "pass": ok})
    rep = pd.DataFrame(rows)
    if verbose:
        print(rep.to_string(index=False,
                            formatters={"max_diff": "{:.3e}".format,
                                        "second_diff": "{:.3e}".format,
                                        "coverage": "{:.1%}".format}))
        print(f"\nreplay 成功 {done}/{len(keys)}（缺缓存 {miss_cache}，"
              f"dp 前无 bar {no_bars}）")
        print("AUDIT_PASS" if all_pass else "AUDIT_FAIL")
    return rep, {"done": done, "n": len(keys), "miss_cache": miss_cache,
                 "no_bars": no_bars, "pass": all_pass}


# ---------------------------------------------------------------- score-live
def cmd_score_live(a):
    hhmm = int(str(a.time).replace(":", ""))
    dp = align_dp(hhmm)
    if dp is None:
        print(f"NO_SCORE: 时间 {a.time} 无已收盘决策点覆盖（"
              f"可打分窗口 09:30-10:30 / 13:00-14:50）")
        return 0
    code = _norm_code(a.code)
    trade_date = int(a.date)
    if not service_available():
        print("STOCKDB_UNAVAILABLE: 本机 stockdb 服务未运行"
              f"（{STOCKDB_HOST}:{STOCKDB_PORT}），无法取盘中分钟")
        return 3
    # 历史分钟：≥25 个交易日（t6 需 20 日量能基线 + 3 日槽尾窗）
    start = (pd.Timestamp(str(trade_date)) - pd.Timedelta(days=60)) \
        .strftime("%Y%m%d")
    minutes = fetch_minute(code, start=start, end=str(trade_date), fq=None)
    if not len(minutes):
        print(f"NO_DATA: stockdb 无 {code} {trade_date} 的分钟数据")
        return 4
    daily_start = (pd.Timestamp(str(trade_date)) - pd.Timedelta(days=400)) \
        .strftime("%Y%m%d")
    daily = fetch_daily(code, start=daily_start, end=str(trade_date), fq=None)
    if not len(daily):
        print(f"NO_DATA: stockdb 无 {code} 日线数据")
        return 4
    daily["code"] = code
    if "pre_close" not in daily or daily["pre_close"].isna().all():
        daily["pre_close"] = daily["close"].shift(1)
    asof = daily_asof(code, daily, trade_date)
    pre_close, atr_px = asof["pre_close"], asof["atr_ref"]
    if not (np.isfinite(pre_close) and np.isfinite(atr_px)):
        print("NO_SCORE: 日线历史不足，pre_close/ATR_ref 不可得")
        return 5
    feats = compute_features(code, trade_date, dp, minutes, pre_close,
                             atr_px=atr_px, daily_hist=daily)
    if feats is None:
        print(f"NO_SCORE: {code} {trade_date} dp={dp} 前无分钟数据")
        return 6
    bundle = load_bundle()
    sc = score_features(bundle, dp, feats)
    ood = attach_ood(asof)
    model_version = str(bundle["meta"].get("_frozen_at") or "")
    row = {"date": trade_date, "dp_date": trade_date, "code": code, "dp": dp,
           "P_risk": sc["P_risk"], "P_opp": sc["P_opp"],
           "quadrant": sc["quadrant"], "ev3_ref": sc["ev3_ref"],
           "ev5_ref": sc["ev5_ref"], "atr_ref": atr_px, "entry": float(a.price),
           "src": "LIVE", "subtype": "LIVE", "model_version": model_version,
           **{k: asof[k] for k in ("runup_5d", "runup_10d", "runup_20d",
                                   "pos20", "rsi14", "turnover_sum_5d",
                                   "turnover_max_5d", "limit_up_10d")},
           **ood}
    print(f"{code} {trade_date} dp={dp} ({sc['head']}) "
          f"P_risk={sc['P_risk']:.4f} P_opp={sc['P_opp']:.4f} "
          f"quadrant={sc['quadrant']} ood={row['ood']} "
          f"(model_version={model_version})")
    if a.db:
        import shadow_log          # pullback_train 内，允许依赖
        n = shadow_log.upsert_decisions(a.db, [row],
                                        model_version=model_version)
        log(f"upsert {n} 行 -> {a.db}（src=LIVE, entry={a.price}）")
    return 0


# ---------------------------------------------------------------- backfill-live
LIVE_KS = 1.5          # 离线 B0/LEGACY 口径（A2=1.2 仅训练池 A2 来源；LIVE 取保守档）
LIVE_KT = 2.0          # 主场景档位（= shadow_log.KT / dual_head_labels 主标签）


def _truncate_cm_days(cm, max_day_i):
    """cm 截断到 day_idx <= max_day_i（用于 3 日账本的 scan_exit 复用）。"""
    keep = cm["day_idx"] <= max_day_i
    out = {k: cm[k][keep] for k in ("day_idx", "t", "o", "h", "l", "c", "v", "a")}
    nd = max_day_i + 1
    out["days"] = cm["days"][:nd]
    out["days_all"] = cm["days"][:nd]
    for k in ("day_close", "day_open", "day_high", "day_low"):
        out[k] = cm[k][:nd]
    return out


def _pcb_of(cm):
    """每日 pre_close 序列（scan_exit 的 pre_close_by_day 参数）。"""
    pcb = np.full(len(cm["days"]), np.nan)
    pcb[1:] = cm["day_close"][:-1]
    return pcb


def live_ledger(code, dp_date, entry, atr_ref, minutes, daily):
    """单条 LIVE 行的 3/5 日账本。minutes = stockdb 取的 dp_date 起分钟
    （可含后续 ≥5 个交易日），daily = 含 dp_date 前收盘的日线。
    返回 dict(nr_3d/nr_5d/stop_5d/outcome_t2/censored/avail/need)；
    5 日窗口数据不全时返回 None（等以后回填）。口径见模块 docstring。"""
    entry, atr_ref = float(entry), float(atr_ref)
    d = daily[daily["date"] < int(dp_date)]
    if not len(d) or not (np.isfinite(entry) and entry > 0
                          and np.isfinite(atr_ref) and atr_ref > 0):
        raise ValueError("entry/atr_ref/pre_close 缺失")
    pre_close = float(d["close"].iloc[-1])
    atr_pct = atr_ref / pre_close
    cm, day_i = cm_from_minutes_df(minutes, dp_date, 1450)  # 当日全量（扫描从次日开始）
    if cm is None or day_i != 0 or len(cm["days"]) < 2:
        return None
    lim = mle.LIMIT_OF(str(code))
    pcb = _pcb_of(cm)
    sc5 = mle.scan_exit(cm, day_i, cm["days"], entry, atr_pct, LIVE_KS,
                        pcb, lim, str(code))
    if sc5 is None or sc5["avail_days"] < sc5["need_days"]:
        return None                                  # 5 日窗口不完整，跳过
    cm3 = _truncate_cm_days(cm, min(day_i + 3, len(cm["days"]) - 1))
    sc3 = mle.scan_exit(cm3, day_i, cm3["days"], entry, atr_pct, LIVE_KS,
                        _pcb_of(cm3), lim, str(code))
    r5 = sc5["res"][LIVE_KT]
    censored = int(sc5["avail_days"] < sc5["need_days"])
    if sc3 is None or sc3["avail_days"] < sc3["need_days"]:
        nr3 = np.nan                                 # 3 日缺损但 5 日完整（极少见）
    else:
        nr3 = float(sc3["res"][LIVE_KT]["net_ret"])
    return {"nr_3d": nr3,
            "nr_5d": float(r5["net_ret"]),
            "stop_5d": int(r5["outcome"] == "STOP"),
            "outcome_t2": r5["outcome"],
            "censored": censored,
            "avail": sc5["avail_days"], "need": sc5["need_days"]}


def cmd_backfill_live(a):
    import sqlite3
    conn = sqlite3.connect(a.db, timeout=20)
    rows = pd.read_sql(
        "SELECT sig_date, dp_date, code, dp, entry, atr_ref "
        "FROM shadow_decisions WHERE src='LIVE' "
        "AND nr_5d IS NULL AND outcome_t2 IS NULL", conn)
    today = pd.Timestamp.today().normalize()
    daily_cache = {}
    n_ok = n_skip = n_fail = 0
    fails, done_rows = [], []
    for _, r in rows.iterrows():
        code, dp = str(r["code"]), int(r["dp"])
        dp_date = int(str(r["dp_date"])[:8])
        key = f"{r['sig_date']}/{code}/dp{dp}"
        try:
            if pd.Timestamp(str(dp_date)) + pd.Timedelta(days=7) > today:
                n_skip += 1                          # 观察窗未满 7 自然日
                continue
            if code not in daily_cache:
                daily_cache[code] = fetch_daily(
                    code, start="19900101",
                    end=(today + pd.Timedelta(days=1)).strftime("%Y%m%d"),
                    fq=None)
            daily = daily_cache[code]
            end = (pd.Timestamp(str(dp_date)) + pd.Timedelta(days=25)) \
                .strftime("%Y%m%d")
            minutes = fetch_minute(code, start=str(dp_date), end=end, fq=None)
            if not len(minutes):
                raise ValueError("stockdb 无该日分钟数据")
            res = live_ledger(code, dp_date, r["entry"], r["atr_ref"],
                              minutes, daily)
            if res is None:
                n_skip += 1                          # T+6 窗口不完整，等以后
                print(f"  SKIP {key}: 5日窗口数据不完整，留待以后")
                continue
            res = {k: (None if (isinstance(v, float) and not np.isfinite(v))
                       else v) for k, v in res.items()}
            done_rows.append((str(r["sig_date"]), code, dp, key, res))
            if a.dry_run:
                print(f"  DRY  {key}: nr_3d={res['nr_3d']:.4f} "
                      f"nr_5d={res['nr_5d']:.4f} stop_5d={res['stop_5d']} "
                      f"outcome={res['outcome_t2']} "
                      f"(avail={res['avail']}/{res['need']}, "
                      f"ks={LIVE_KS}, kt={LIVE_KT})")
            n_ok += 1
        except Exception as e:                       # 单行失败不中断
            n_fail += 1
            fails.append((key, f"{type(e).__name__}: {e}"))
            print(f"  FAIL {key}: {type(e).__name__}: {e}")
    if not a.dry_run:
        filled_at = time.strftime("%Y-%m-%d %H:%M:%S")
        for sig_date, code, dp, key, res in done_rows:
            conn.execute(
                "UPDATE shadow_decisions SET nr_3d=?, nr_5d=?, stop_5d=?, "
                "outcome_t2=?, censored=?, filled_at=? "
                "WHERE sig_date=? AND code=? AND dp=?",
                (res["nr_3d"], res["nr_5d"], res["stop_5d"],
                 res["outcome_t2"], res["censored"], filled_at,
                 sig_date, code, dp))
        conn.commit()
    conn.close()
    print(f"\nbackfill-live: 处理 {len(rows)} 行（src=LIVE 未回填），"
          f"成功 {n_ok}、跳过 {n_skip}、失败 {n_fail}"
          f"（{'dry-run 未写库' if a.dry_run else '已 UPDATE ' + str(len(done_rows)) + ' 行'}）")
    return 1 if n_fail else 0


# ---------------------------------------------------------------- selftest
def cmd_selftest():
    """本地缓存 replay 一个已知 (code,dp_date,dp) 走完整 score 路径：
    不连 stockdb、不写库；并与离线特征表直算的 P_risk/P_opp 对账。"""
    keys = load_truth_2026(100000, seed=7)   # 全量键池
    pick = None
    for _, r in keys.iterrows():
        if int(r["dp"]) == 1000 and pd.notna(r.get("pf_ret1m")) \
                and pd.notna(r.get("t6_dist_mid_s_atr")):
            pick = r
            break
    if pick is None:
        pick = keys.iloc[0]
    code, dp_date, dp = str(pick["code"]), int(pick["dp_date"]), int(pick["dp"])
    atr_px = float(pick["atr_pct"]) * float(pick["sig_close"])
    log(f"selftest: replay {code} dp_date={dp_date} dp={dp} "
        f"atr_px={atr_px:.4f} pre_close={pick['pre_close']}")
    cm, day_i = replay_cm(code, dp_date, dp, cache_dir=MINUTE_CACHE_DIR)
    assert cm is not None, "replay 失败：缓存缺数据"
    # 复刻 score-live 的取数形态：历史 + 当日未截断，交给 compute_features 截断
    dates = cm["t"] + (cm["days"][cm["day_idx"]] * 1000000).astype(np.int64)
    df = pd.DataFrame({"date": dates, "open": cm["o"], "high": cm["h"],
                       "low": cm["l"], "close": cm["c"],
                       "volume": cm["v"], "amount": cm["a"]})
    feats = compute_features(code, dp_date, dp, df, float(pick["pre_close"]),
                             atr_px=atr_px, atr_pct=float(pick["atr_pct"]))
    assert feats is not None
    bundle = load_bundle()
    sc = score_features(bundle, dp, feats)
    print("== 特征值（replay）==")
    for c in bundle["meta"][sc["risk_head"]]["features"]:
        print(f"  {c:36s} {feats.get(c)}")
    print(f"== 打分: P_risk={sc['P_risk']:.4f} P_opp={sc['P_opp']:.4f} "
          f"quadrant={sc['quadrant']} base_rate={sc['base_rate']:.4f}")
    # 与离线特征表直算对账（同键取离线列 → 同一 bundle 打分）
    off = {}
    for c in bundle["meta"][sc["risk_head"]]["features"] \
            + bundle["meta"][sc["opp_head"]]["features"]:
        off[c] = pick.get(c, np.nan)
    off.update(T4_UNAVAILABLE)
    sc_off = score_features(bundle, dp, off)
    print(f"== 离线表直算: P_risk={sc_off['P_risk']:.4f} "
          f"P_opp={sc_off['P_opp']:.4f} quadrant={sc_off['quadrant']}")
    d = max(abs(sc["P_risk"] - sc_off["P_risk"]),
            abs(sc["P_opp"] - sc_off["P_opp"]))
    ok = d <= 1e-9
    print("SELFTEST", "PASS" if ok else f"FAIL (ΔP={d:.3e})")
    return 0 if ok else 1


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="盘中触发式 T5 打分 v2")
    ap.add_argument("--audit", action="store_true", help="离线对账审计")
    ap.add_argument("--n", type=int, default=300, help="audit 抽样数")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cache-dir", default=None,
                    help="audit 分钟缓存目录（默认 minute_cache_clean）")
    ap.add_argument("--score-live", action="store_true", help="盘中实时打分")
    ap.add_argument("--code", default=None)
    ap.add_argument("--date", default=None, help="交易日 YYYYMMDD")
    ap.add_argument("--time", default=None, help="触发时间 HH:MM")
    ap.add_argument("--price", type=float, default=None, help="触发价（entry）")
    ap.add_argument("--db", default=None, help="shadow_decisions sqlite 路径")
    ap.add_argument("--selftest", action="store_true",
                    help="本地缓存 replay 完整 score 路径（不连网不写库）")
    ap.add_argument("--backfill-live", action="store_true",
                    help="LIVE 触发行的盘后收益回填（需 --db）")
    ap.add_argument("--dry-run", action="store_true",
                    help="backfill-live 只算不写")
    a = ap.parse_args()
    if a.audit:
        rep, info = run_audit(n=a.n, cache_dir=a.cache_dir, seed=a.seed)
        return 0 if info["pass"] else 1
    if a.score_live:
        for k in ("code", "date", "time", "price"):
            if getattr(a, k) is None:
                ap.error(f"--score-live 需要 --{k}")
        return cmd_score_live(a)
    if a.backfill_live:
        if not a.db:
            ap.error("--backfill-live 需要 --db")
        return cmd_backfill_live(a)
    if a.selftest:
        return cmd_selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
