"""latr_builder — LATR Candidate Engine v0 Builder (Gate 4 接入, Step 9A).

按冻结契约实现, 不重新设计:
  - latr_v0/latr_candidate_engine_spec.md  (ATR 三拆 / 流动性区域 / 承接特征 / PASS-NEAR_MISS-CONTROL-INVALIDATED)
  - step9_preflight/label_contract_v1.yaml  (S0/H1/H3/H5, market_proxy alpha)
  - step9_preflight/adjustment_chain.py     (ANALYTIC 价格走 canonical mult 链, 不用存储 cum)

价格口径 (用户裁决):
  EXECUTION PRICE = 原始真实市场价 (fill / slippage / ledger)
  ANALYTIC  PRICE = PIT-safe 连续复权价 (ATR / return / swing / MA / structural)
  -> ATR_ref / GapATR / OpenDipATR / TotalDipATR / Zone / 承接特征 全部基于 ANALYTIC 价,
     因此除权除息日不会产生假低开。

Sector/Leader 在 Gate 1B 前硬阻塞: alpha 一律用 MARKET_PROXY-V1 (market_proxy_v1.py),
不回填当前板块成分。

运行:
  python latr_builder.py --demo --n 20 --start 20250105 --end 20250130
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timedelta, timezone

import msgpack
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adjustment_chain import build_chains  # noqa
from build_step9_labels import (s0_returns, s0_alpha,  # noqa
                                timing_return_h1, timing_alpha,
                                build_step9_label)
from market_proxy_v1 import MarketProxyV1  # noqa
from zone_state_engine import (build_zones, select_primary_zone,  # ④
                              run_state_engine, volume_baseline)
from dataclasses import asdict  # ④ zone 序列化
from data_source_provenance import (get_daily, get_minute_http,  # noqa
                                    get_adjustment_http, get_universe)
from green_watch_candidate_loader import (load_green_watch_candidates,  # ③.2
                                          CandidateRow)
from execution_engine import execute  # ⑥ Real Execution / Actual Fill

BASE = "http://127.0.0.1:7899"
DATA_ROOT = os.environ.get("STARGAZER_DATA_ROOT", r"D:/a股数据")
DAILY_PARTS = os.path.join(DATA_ROOT, "data_daily_parts")
MINUTE_START = 20250102
ATR_LOOK = 14
ATR_REF_DAYS = 5
SNAPSHOT_MINUTE = 150000  # 15:00:00 (14 位 HHMMSS 后缀, 与分钟 ts 同尺度)


def _num_or_none(x, nd: int = 5):
    """None / 非有限 -> None; 其余 round(nd)。

    修复 (2026-09-18, exc:TypeError 归因, attribute_typeerror.py):
    旧写法 `round(x, 5) if not np.isnan(x) else None` 在 x=None 时抛
    `np.isnan(None) -> TypeError`。触发场景 = proxy alpha 不可用:
      (a) CENSORED_TAIL 分支 alpha_out 全为 None (H3 超出数据末);
      (b) 非删失但该 symbol proxy alpha 覆盖缺口。
    后果: 样本被当作异常丢弃 -> 违反「CENSORED_TAIL 样本必须保留」契约。
    """
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return round(xf, nd) if np.isfinite(xf) else None


def lookback_start(d_date: int, years: int = 2) -> int:
    """ATR/MA 需要历史窗口; 用过去数据计算 (PIT-safe, 不产生未来泄漏)。"""
    y = d_date // 10000
    return (y - years) * 10000 + 101
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ---------------- stockdb HTTP (与 gate1a 同款) ----------------
def _http_vals(table, code="*", date="*", timeout=180):
    t = urllib.parse.quote(f"{table}:{code}:{date}")
    url = f"{BASE}/?cmd=vals&t={t}"
    with _opener.open(url, timeout=timeout) as r:
        raw = r.read()
    try:
        obj = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    except Exception:
        obj = json.loads(raw.decode("utf-8", "replace"))
    if isinstance(obj, list):
        return [it for it in obj if isinstance(it, dict)]
    return []


def _http_pairs(table, code="*", date="*", timeout=300):
    """cmd=get 返回 [[key, {...}], ...]; 复权等日期在 key 中, 必须配对。"""
    t = urllib.parse.quote(f"{table}:{code}:{date}")
    url = f"{BASE}/?cmd=get&t={t}"
    with _opener.open(url, timeout=timeout) as r:
        raw = r.read()
    try:
        obj = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    except Exception:
        obj = json.loads(raw.decode("utf-8", "replace"))
    if isinstance(obj, list):
        return [(str(k), v) for k, v in obj
                if isinstance(k, str) and isinstance(v, dict)]
    return []


def _field(rec, *names, default=np.nan):
    for n in names:
        if n in rec and rec[n] is not None:
            return rec[n]
    return default


def load_universe():
    return get_universe()


def load_daily_window(code, start, end):
    """CANONICAL 日线 (经 data_source_provenance.get_daily, parquet 优先 + HTTP 兜底)。"""
    df = get_daily(code)
    if df.empty:
        return df
    df = df[(df["date"] >= int(start)) & (df["date"] <= int(end))]
    df = df.sort_values("date").reset_index(drop=True)
    for c in ("open", "high", "low", "close", "pre_close", "float_mv"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_adjustment_chain(codes):
    """CANONICAL 复权 (经 data_source_provenance.get_adjustment_http, HTTP)。
    直接传 (date_str, rec) 给 build_chains (它自行提取 mult)。"""
    by = defaultdict(list)
    for code in codes:
        for d, rec in get_adjustment_http(code):
            by[code.zfill(6)].append((str(d), rec))
    return build_chains(by)


def load_minute(code, date):
    """CANONICAL 分钟 (经 data_source_provenance.get_minute_http, HTTP)。

    返回含 ts(14位时间戳, 兼容字段)/open/high/low/close/volume 的 DataFrame。
    get_minute_http 原始列名为 date, 此处重命名为 ts 以兼容 build_one_sample / dry_replay。
    """
    df = get_minute_http(code, date)
    if df.empty:
        return df
    if "date" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"date": "ts"})
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------- ATR / 价格口径 ----------------
def _true_range(hi, lo, cl):
    prev = np.roll(cl, 1)
    prev[0] = cl[0]
    return np.maximum.reduce([hi - lo, np.abs(hi - prev), np.abs(lo - prev)])


def atr_ref_from_daily(hi, lo, cl, entry_idx):
    if entry_idx < ATR_LOOK or entry_idx < ATR_REF_DAYS:
        return None
    tr = _true_range(hi.to_numpy(float), lo.to_numpy(float), cl.to_numpy(float))
    atr = pd.Series(tr).rolling(ATR_LOOK, min_periods=ATR_LOOK).mean().to_numpy()
    window = atr[entry_idx - ATR_REF_DAYS: entry_idx]
    window = window[~np.isnan(window)]
    if window.size == 0:
        return None
    return float(np.median(window))


# ---------------- 单票单日 LATR 样本 ----------------
def build_one_sample(candidate, chains, proxy: MarketProxyV1, lookback_days=60,
                     reject_log=None):
    """消费一个已冻结 CandidateRow, 做 D 日 (effective_trade_date) 盘中重放。

    绝不重新扫描股票池 / 重算候选资格 / 重算 ATR_ref (用户 ③.3-③.5)。

    样本口径 (用户 2026-09-18 裁定):
      - candidate_status 是 D 日 replay 得出的「策略状态」(研究信息), 绝不可当作
        数据有效性, 也不可据其过滤样本 (否则 = sample-selection future leakage)。
      - 数据/工程有效性走各自 *valid 标志; 各模型 eligibility 独立成列。
      - 不可成交 (涨停/跌停/停牌) 不再丢弃样本: execution_data_valid=0;
        S0 锚 D 日 open 不依赖成交, s0_eligible 仍可为 1; timing_eligible=0。
      - reject_log (可选 list): 记录真正不可 replay 的工程性拒绝 (含原因), 不静默吞。
      - 右删失 (h1/h3 未来 bar 超出数据末) -> label_status=CENSORED_TAIL, 不伪造。
    """
    # 一律继承 CandidateRow 字段, 不重算
    code = candidate.symbol
    plan_date = candidate.plan_date
    d_date = candidate.effective_trade_date  # D 日重放日

    def _reject(reason, detail=""):
        if reject_log is not None:
            reject_log.append({"symbol": str(code), "plan_date": int(plan_date),
                               "d_date": int(d_date), "reason": str(reason),
                               "detail": str(detail)[:200]})
        return None

    # ---- 契约断言 (用户 ③.4) ----
    assert candidate.effective_trade_date == d_date
    assert candidate.plan_date < candidate.effective_trade_date
    frozen_atr_ref = candidate.atr_ref           # ③.5: 继承, 绝不重算
    assert frozen_atr_ref and frozen_atr_ref > 0
    start = lookback_start(d_date)
    end_label = int((datetime.strptime(str(d_date), "%Y%m%d") + timedelta(days=20)).strftime("%Y%m%d"))
    daily = load_daily_window(code, start, end_label)  # 含 H1/H3/H5 标签所需未来日线
    if daily.empty or len(daily) < ATR_LOOK + ATR_REF_DAYS:
        return _reject("daily_empty_or_short",
                       f"rows={0 if daily is None else len(daily)}")
    chain = chains.get(code.zfill(6))
    if chain is None:
        return _reject("no_adjustment_chain")
    # ANALYTIC 价: 以 d_date 为锚点回测复权 (factor_at(date)/factor_at(d_date)),
    # 量级正确且消除除权跳变; 分钟 bar 在 d_date 上 scale=1, 与日线口径一致。
    dates_arr = daily["date"].to_numpy(int)
    ref_fac = chain.factor_at(int(d_date)) or 1.0

    def _adj(col):
        raw = daily[col].to_numpy(float)
        fac = np.array([chain.factor_at(int(dt)) for dt in dates_arr], dtype=float)
        return raw * (fac / ref_fac)

    adj_close = _adj("close")
    adj_open = _adj("open")
    adj_low = _adj("low")
    adj_high = _adj("high")

    dates = daily["date"].to_numpy(int)
    d_idx = int(np.where(dates == d_date)[0][0]) if d_date in dates else len(dates) - 1
    if d_idx < ATR_LOOK + ATR_REF_DAYS:
        return _reject("history_too_short_for_atr", f"d_idx={d_idx}")
    atr_ref = frozen_atr_ref  # 继承 CandidateRow ATR_ref (③.5: LATR 不重算另一份)

    # 日内分钟
    m = load_minute(code, d_date)
    if m.empty:
        return _reject("minute_empty", f"d_date={d_date}")
    # 开盘后 15 分钟窗口 (09:30-09:45): ts 末6位为 HHMMSS, 15分钟 = +1500
    open_ts = int(m["ts"].min())
    # ---- ⑥ Real Execution: snapshot(D日Open) -> signal -> next legal minute -> raw fill ----
    signal_ts = open_ts  # 候选在 D 日开盘触发
    exec = execute(signal_ts, m, side="buy")
    actual_fill = exec["actual_fill"]
    # 不可成交 (涨停/跌停/停牌/无流动性): 不再丢弃样本 (用户 2026-09-18 裁定)。
    # 显式记 execution_data_valid=0 + reason; S0 锚 D 日 open, 仍然可算。
    execution_data_valid = 1 if exec["exec_status"] == "FILLED" else 0
    # raw 日线价: 成交/PnL 用原始价; 特征/结构用 ANALYTIC 复权价
    raw_close = daily["close"].to_numpy(float)
    raw_open = daily["open"].to_numpy(float)
    cut = open_ts + 1500
    m15 = m[m["ts"] <= cut]
    if m15.empty:
        m15 = m.iloc[: max(1, len(m) // 10)]
    low_15 = m15["low"].min()
    high_15 = m15["high"].max()
    close_15 = m15["close"].iloc[-1]
    day_low = m["low"].min()
    day_high = m["high"].max()
    last_close = m["close"].iloc[-1]

    prev_close_adj = adj_close[d_idx - 1]
    open_adj = adj_open[d_idx]
    gap_atr = (open_adj - prev_close_adj) / atr_ref
    open_dip_atr = (open_adj - low_15) / atr_ref
    total_dip_atr = (prev_close_adj - day_low) / atr_ref

    # ATR 三拆分桶
    if open_dip_atr < 0.10:
        bucket = "NO_DIP"
    elif open_dip_atr < 0.15:
        bucket = "LIGHT_DIP"
    elif open_dip_atr < 0.20:
        bucket = "NORMAL_DIP"
    elif open_dip_atr < 0.25:
        bucket = "HYPOTHESIS_CORE"
    elif open_dip_atr < 0.30:
        bucket = "DEEP_DIP"
    elif open_dip_atr < 0.40:
        bucket = "VERY_DEEP_DIP"
    else:
        bucket = "EXTREME_DIP"

    # 流动性区域: 日线支撑 (前低 / MA20下 / BOLL下) 与日内低点合并 (0.35 ATR 内)
    prev_low = adj_low[d_idx - 1]
    ma20 = adj_close[max(0, d_idx - 19): d_idx + 1].mean()
    roll_std = adj_close[max(0, d_idx - 19): d_idx + 1].std()
    boll_lower = ma20 - 2 * roll_std if not np.isnan(roll_std) else prev_low
    supports = [prev_low, ma20, boll_lower, day_low]
    # 0.35 ATR 内合并 -> zone_mid
    cand = [s for s in supports if abs(s - day_low) <= 0.35 * atr_ref]
    zone_mid = float(np.mean(cand)) if cand else float(day_low)
    zone_lower = float(min(cand)) if cand else float(day_low)
    zone_upper = float(max(cand)) if cand else float(day_low)
    zone_distance_atr = (open_adj - zone_mid) / atr_ref if False else (day_low - zone_mid) / atr_ref
    penetration_atr = (zone_lower - day_low) / atr_ref
    zone_strength = len(cand)

    # 承接特征
    close_position_15m = ((close_15 - low_15) / (high_15 - low_15)
                           if (high_15 - low_15) > 0 else np.nan)
    recovery_atr = (last_close - day_low) / atr_ref
    new_low_flag = 1 if day_low <= min(adj_low[max(0, d_idx - 19): d_idx]) else 0

    # 状态机
    latr_watch = True  # 由候选引擎传入; demo 视 dip 存在即为 watch
    latr_armed = (0.10 <= open_dip_atr <= 0.40) and (zone_distance_atr <= 0.20)

    # 标签 (label_contract_v1, ⑥修正: S0 与 Timing 锚点严格分离)
    #   S0     : entry = D 日 legal open (snapshot)  -> 公司行为安全 analytic 口径
    #   Timing : entry = actual_fill (Execution Engine 真实成交, 含 slippage/fee)
    # 两者不得共用 fill_price, 否则 S0 会被污染成 "Timing fill -> H3" (用户 ⑥ 明文禁止)。
    h3_idx = d_idx + 3
    h1_idx = d_idx + 1
    h5_idx = d_idx + 5
    s0_open = adj_open[d_idx]   # D 日 open, analytic (factor 比例=1 于 D 锚点)
    h1 = adj_close[h1_idx] / s0_open - 1 if h1_idx < len(adj_close) else np.nan
    h3 = adj_close[h3_idx] / s0_open - 1 if h3_idx < len(adj_close) else np.nan
    h5 = adj_close[h5_idx] / s0_open - 1 if h5_idx < len(adj_close) else np.nan
    # 纯 raw 执行 PnL (锚 actual_fill), 供 ② 对齐/复权校验, 与 S0 区分存放
    if actual_fill is not None:
        raw_h1 = raw_close[h1_idx] / actual_fill - 1 if h1_idx < len(raw_close) else np.nan
        raw_h3 = raw_close[h3_idx] / actual_fill - 1 if h3_idx < len(raw_close) else np.nan
        raw_h5 = raw_close[h5_idx] / actual_fill - 1 if h5_idx < len(raw_close) else np.nan
    else:
        raw_h1 = raw_h3 = raw_h5 = np.nan  # 不可成交 -> 无实际成交 PnL (不伪造)
    # S0 (D-open 锚点) 公司行为安全口径 (与 h3 同源), 不依赖是否成交
    s0_return_h3_adj = (adj_close[h3_idx] / s0_open - 1) if h3_idx < len(adj_close) else np.nan
    # Timing 锚定 actual_fill (投影到 analytic 以保证公司行为安全), entry 严格 != S0 锚点
    raw_open_d = raw_open[d_idx] if d_idx < len(raw_open) else np.nan
    if actual_fill is not None and raw_open_d:
        timing_fill_analytic = actual_fill * (adj_open[d_idx] / raw_open_d)
    else:
        timing_fill_analytic = None      # 不可成交 -> timing 无合法 entry
    timing_ret = timing_return_h1(timing_fill_analytic, adj_close[h1_idx] if h1_idx < len(adj_close) else None)
    # ---- ② MARKET_PROXY-V1: 公司行为安全代理收益 (S0: D-open->H3-close; Timing: D-open->H1-close) ----
    # 与 stock label 完全同区间 (start/end 一致), leave-one-out 剔除自身, P 日权重冻结。
    h1_date = int(dates[h1_idx]) if h1_idx < len(dates) else None
    h3_date = int(dates[h3_idx]) if h3_idx < len(dates) else None
    alpha_in = {
        "symbol": code, "plan_date": plan_date, "d_date": int(d_date),
        "h1_date": h1_date, "h3_date": h3_date,
        "s0_return_h3": h3, "timing_return_h1": timing_ret,
        "s0_start_ts": int(f"{int(d_date)}093000"),
        "s0_end_ts": int(f"{h3_date}150000") if h3_date is not None else None,
        "actual_fill_ts": exec["fill_ts"],
        "timing_end_ts": int(f"{h1_date}150000") if h1_date is not None else None,
    }
    if h3_date is None or h1_date is None:
        # 右删失: h1/h3 未来 bar 超出数据末 (如 20260914-16 vs DATA_END=20260917)。
        # 真实不可得, 不伪造 (不得 NaN->0), 也不物理删样本:
        #   label_status=CENSORED_TAIL -> 不进训练集, 但保留在主样本库 (可审计)。
        alpha_out = {
            "s0_market_alpha_h3": None, "timing_market_alpha_h1": None,
            "proxy_s0_return": None, "proxy_timing_return": None,
            "alpha_label_available": 0, "proxy_weight_date": None,
            "proxy_weight_coverage": None, "proxy_constituent_count": None,
            "proxy_excluded_symbol": code,
        }
        label_status = "CENSORED_TAIL"
    else:
        alpha_out = proxy.attach_alpha_labels(alpha_in)
        label_status = "OK" if execution_data_valid else "NO_FILL"
    # Sector 在 Gate1B 前硬阻塞 -> alpha 一律走 MARKET_PROXY (s0_alpha_h3 即市场 alpha)
    s0_alpha_h3 = alpha_out["s0_market_alpha_h3"]
    timing_alpha_h1 = alpha_out["timing_market_alpha_h1"]

    # ---- ④ Zone Builder + Minute State Engine + A4 连续特征 (PIT 冻结, snapshot-time causal) ----
    zone_state = {}
    try:
        z_zones = build_zones(daily, int(plan_date), atr_ref, chain)
        z_primary = select_primary_zone(z_zones, float(adj_close[d_idx]))
        z_scale = (chain.factor_at(int(d_date)) or 1.0) / (chain.factor_at(int(plan_date)) or 1.0)
        z_snap = int(f"{int(d_date)}{SNAPSHOT_MINUTE}")
        z_out = (run_state_engine(z_primary, m, z_scale, candidate.primary_setup, atr_ref,
                                 z_snap, open_dip_atr,
                                 z_primary.zone_distance_atr if z_primary else 0.0,
                                 dip_stage=getattr(candidate, "dip_stage", None))
                if z_primary else {})
        # 同分钟量能基准: 过去 60 完整交易日同 minute-of-day (不读 D 日未来分钟)
        past_dates = [int(x) for x in dates if int(x) < int(d_date)][-60:]
        past_min = []
        for pd_ in past_dates:
            try:
                pm = load_minute(code, pd_)
                if not pm.empty:
                    past_min.append(pm)
            except Exception:  # noqa
                continue
        vb = volume_baseline(int(d_date), m, past_min, z_snap)
        zone_state = {
            "zones_json": json.dumps([asdict(z) for z in z_zones], ensure_ascii=False) if z_zones else "[]",
            "primary_zone_id": z_primary.zone_id if z_primary else None,
            "primary_zone_source": z_primary.zone_source if z_primary else None,
            "primary_zone_mid": round(z_primary.zone_mid, 4) if z_primary else None,
            "primary_zone_lower": round(z_primary.zone_lower, 4) if z_primary else None,
            "primary_zone_upper": round(z_primary.zone_upper, 4) if z_primary else None,
            "primary_zone_distance_atr": round(z_primary.zone_distance_atr, 4) if z_primary else None,
            "final_state": z_out.get("final_state"),
            "state_transitions_json": json.dumps(z_out.get("state_transitions", []), ensure_ascii=False),
        }
        for k in ("zone_break_depth_atr", "zone_break_duration", "recovery_from_zone_low_atr",
                  "recovery_atr", "dip_recovery_ratio", "reclaim_zone_lower", "reclaim_zone_mid",
                  "reclaim_zone_upper", "reclaim_failure_count", "second_test_flag",
                  "second_test_higher_low", "time_since_zone_touch", "time_below_zone",
                  "lower_wick_ratio", "sell_volume_ratio", "rebound_volume_ratio", "vwap_reclaim"):
            zone_state[k] = z_out.get(k)
        zone_state["cum_volume_vs_hist_same_time"] = vb.get("cum_volume_vs_hist_same_time")
        zone_state["cum_turnover_vs_hist_same_time"] = vb.get("cum_turnover_vs_hist_same_time")
        zone_state["zone_error"] = None
        zone_state["zone_valid"] = 1   # ④ 成功: 允许进入 Model Matrix
    except Exception as e:  # noqa: ④ 失败必须显式 invalid, 不能静默带着缺失特征进训练
        zone_state = {"zone_error": str(e)[:200], "zone_valid": 0}

    # 状态分类 (全存, 不只有 rule_pass)
    if latr_armed and open_dip_atr >= 0.20:
        cand_status = "PASS"
    elif latr_armed:
        cand_status = "NEAR_MISS"
    elif 0.10 <= open_dip_atr <= 0.40:
        cand_status = "CONTROL"
    else:
        cand_status = "INVALIDATED"

    row = {
        "symbol": code, "plan_date": plan_date, "effective_trade_date": d_date,
        "primary_setup": candidate.primary_setup, "d_date": d_date,
        "snapshot_time": int(f"{d_date}{SNAPSHOT_MINUTE}"),
        "atr_ref": round(atr_ref, 5),
        "gap_atr": round(gap_atr, 4), "open_dip_atr": round(open_dip_atr, 4),
        "total_dip_atr": round(total_dip_atr, 4), "latr_bucket": bucket,
        "zone_mid": round(zone_mid, 4), "zone_lower": round(zone_lower, 4),
        "zone_upper": round(zone_upper, 4), "zone_strength": zone_strength,
        "zone_distance_atr": round(zone_distance_atr, 4),
        "penetration_atr": round(penetration_atr, 4),
        "close_position_15m": _num_or_none(close_position_15m, 4),
        "recovery_atr": round(recovery_atr, 4),
        "new_low_flag": new_low_flag,
        "latr_watch": latr_watch, "latr_armed": latr_armed,
        "candidate_status": cand_status,
        "actual_fill": round(float(actual_fill), 4),
        "exec_status": exec["exec_status"],
        "fill_ts": exec["fill_ts"],
        "signal_time": int(signal_ts),
        "alpha_label_available": alpha_out["alpha_label_available"],
        "return_h1": _num_or_none(h1),
        "return_h3": _num_or_none(h3),
        "return_h5": _num_or_none(h5),
        "return_h1_raw": _num_or_none(raw_h1),
        "return_h3_raw": _num_or_none(raw_h3),
        "return_h5_raw": _num_or_none(raw_h5),
        "s0_return_h3": _num_or_none(s0_return_h3_adj),
        "s0_alpha_h3": _num_or_none(s0_alpha_h3),
        "timing_return_h1": _num_or_none(timing_ret),
        "timing_alpha_h1": _num_or_none(timing_alpha_h1),
        # ② MARKET_PROXY-V1 审计字段
        "s0_market_alpha_h3": round(alpha_out["s0_market_alpha_h3"], 6) if alpha_out["s0_market_alpha_h3"] is not None else None,
        "timing_market_alpha_h1": round(alpha_out["timing_market_alpha_h1"], 6) if alpha_out["timing_market_alpha_h1"] is not None else None,
        "market_proxy_return_h3": round(alpha_out["proxy_s0_return"], 6) if alpha_out["proxy_s0_return"] is not None else None,
        "proxy_timing_return": round(alpha_out["proxy_timing_return"], 6) if alpha_out["proxy_timing_return"] is not None else None,
        "proxy_weight_date": alpha_out["proxy_weight_date"],
        "proxy_weight_coverage": alpha_out["proxy_weight_coverage"],
        "proxy_constituent_count": alpha_out["proxy_constituent_count"],
        "proxy_excluded_symbol": alpha_out["proxy_excluded_symbol"],
        # ④ Zone + Minute State + A4 连续特征
        **zone_state,
        # PIT 守卫: 所有源时间戳必须 <= snapshot_time
        "_max_source_ts": int(f"{d_date}{SNAPSHOT_MINUTE}"),
    }
    return _finalize_sample(row, candidate, exec, zone_state, label_status,
                            h3_date, h1_date, open_dip_atr, zone_mid)


def _finalize_sample(row, candidate, exec, zone_state, label_status,
                     h3_date, h1_date, open_dip_atr, zone_mid):
    """显式拆开「数据/工程有效性」与「策略状态」(用户 2026-09-18 裁定)。

    原则:
      - candidate_status 只是 D 日 replay 的研究分层, 绝不参与样本取舍;
        按它删除样本 = sample-selection future leakage (即使特征本身无 D 日信息)。
      - 各模型 eligibility 独立成列, 由下游自己取 mask, 不从主样本库物理删除。
      - 每一处 invalid 必须带显式 reason, 不允许留 NULL/未知桶进 Final Freeze。
    """
    d_date = int(row["d_date"])

    # ---- 数据/工程有效性 (决定能否进入相应数据集) ----
    candidate_contract_valid = 1
    candidate_contract_invalid_reason = None
    if getattr(candidate, "atr_ref", None) in (None, 0):
        candidate_contract_valid = 0
        candidate_contract_invalid_reason = "atr_ref_missing"

    zone_data_valid = int(zone_state.get("zone_valid", 0) == 1)
    zone_invalid_reason = None if zone_data_valid else str(zone_state.get("zone_error"))[:200]

    exec_valid = int(exec["exec_status"] == "FILLED")
    execution_invalid_reason = None if exec_valid else f"exec_status={exec['exec_status']}"

    # S0 标签 = D 日 open -> H3 close (analytic, 公司行为安全), 与是否成交无关
    s0_lab_ok = row["s0_return_h3"] is not None
    if h3_date is None or h1_date is None:
        label_status_, label_invalid_reason = "CENSORED_TAIL", "h3_h1_future_bar_beyond_data_end"
    elif not s0_lab_ok:
        label_status_, label_invalid_reason = "MISSING", "s0_return_h3_null"
    else:
        label_status_, label_invalid_reason = label_status, None
    label_data_valid = int(s0_lab_ok)

    proxy_data_valid = int(row["s0_alpha_h3"] is not None)
    proxy_invalid_reason = None if proxy_data_valid else (
        "censored_tail" if h3_date is None else "proxy_alpha_null")

    feature_data_valid = int(np.isfinite(zone_mid) and np.isfinite(open_dip_atr))
    feature_invalid_reason = None if feature_data_valid else "zone_mid/open_dip_atr_non_finite"

    sample_data_valid = int(candidate_contract_valid and zone_data_valid and feature_data_valid
                            and label_data_valid and proxy_data_valid)
    reasons = [r for r in (candidate_contract_invalid_reason, zone_invalid_reason,
                           label_invalid_reason, proxy_invalid_reason,
                           feature_invalid_reason) if r]
    sample_invalid_reason = ";".join(reasons) if reasons else None

    # ---- 各模型 eligibility (公式中不得出现 candidate_status) ----
    s0_eligible = int(sample_data_valid and label_data_valid)
    s0_ineligible_reason = None if s0_eligible else (sample_invalid_reason or "no_s0_label")

    timing_eligible = int(sample_data_valid and exec_valid
                          and row["timing_return_h1"] is not None
                          and row["timing_alpha_h1"] is not None)
    if timing_eligible:
        timing_ineligible_reason = None
    elif not exec_valid:
        timing_ineligible_reason = execution_invalid_reason
    elif row["timing_return_h1"] is None or row["timing_alpha_h1"] is None:
        timing_ineligible_reason = "timing_label_or_alpha_null"
    else:
        timing_ineligible_reason = sample_invalid_reason

    # R0 / Portfolio 的 exit-label 层在 A0 才生成 -> 此处只给「前置条件 mask」并显式标注 basis
    r0_eligible = int(sample_data_valid and exec_valid)
    portfolio_eligible = int(sample_data_valid and exec_valid)

    row.update({
        # 数据/工程有效性
        "candidate_contract_valid": candidate_contract_valid,
        "candidate_contract_invalid_reason": candidate_contract_invalid_reason,
        "zone_data_valid": zone_data_valid,
        "zone_invalid_reason": zone_invalid_reason,
        "execution_data_valid": exec_valid,
        "execution_invalid_reason": execution_invalid_reason,
        "label_data_valid": label_data_valid,
        "label_invalid_reason": label_invalid_reason,
        "proxy_data_valid": proxy_data_valid,
        "proxy_invalid_reason": proxy_invalid_reason,
        "feature_data_valid": feature_data_valid,
        "feature_invalid_reason": feature_invalid_reason,
        "sample_data_valid": sample_data_valid,
        "sample_invalid_reason": sample_invalid_reason,
        "sample_valid_formula": ("candidate_contract_valid&zone_data_valid&feature_data_valid"
                                 "&label_data_valid&proxy_data_valid"),
        "label_status": label_status_,
        # 各模型 eligibility (下游自行取 mask; 主样本库全量保留)
        "s0_eligible": s0_eligible,
        "s0_ineligible_reason": s0_ineligible_reason,
        "timing_eligible": timing_eligible,
        "timing_ineligible_reason": timing_ineligible_reason,
        "r0_eligible": r0_eligible,
        "portfolio_eligible": portfolio_eligible,
        "exit_eligibility_basis": "PRECONDITION_ONLY__EXIT_LABEL_PENDING_A0",
        # 语义守卫: 明确 candidate_status 不是有效性
        "candidate_status_role": "RESEARCH_TIER_NOT_VALIDITY",
        # 特征时点审计 (供 Gate4 检查特征源上界)
        "feature_asof_ts": int(f"{d_date}{SNAPSHOT_MINUTE}"),
        "pclose_feature_asof_ts": int(f"{int(row['plan_date'])}150000"),
    })
    return row


def build_samples(plan_dates=None, setup_type=None, sample_n=None, max_per_symbol=5):
    """消费 GREEN-WATCH 候选 (build_pool 产出), 不再自扫股票池 (用户 ③.3)。

    plan_dates: None(全部) | int | (lo,hi) | list[int]
    setup_type: None | "BOTTOM_SETUP" | "INTRADAY_DIP_SETUP"
    每个 CandidateRow 仅做 D 日 (effective_trade_date) 盘中重放, 不重新决定候选资格。
    """
    candidates = load_green_watch_candidates(plan_date=plan_dates, setup_type=setup_type)
    if not candidates:
        print("[latr_builder] 无候选 (检查 pool_daily.parquet 是否存在 / plan_dates 范围)")
        return pd.DataFrame()
    if sample_n:
        candidates = candidates[:sample_n]
    codes = sorted({c.symbol for c in candidates})
    chains = load_adjustment_chain(codes)
    proxy = MarketProxyV1(weight="float", leave_one_out=True)
    out = []
    reject_log = []
    exc_log = []
    for cand in candidates:
        try:
            s = build_one_sample(cand, chains, proxy, reject_log=reject_log)
            if s:
                out.append(s)
        except Exception as e:  # noqa
            import traceback
            exc_log.append({"symbol": getattr(cand, "symbol", None),
                            "plan_date": getattr(cand, "plan_date", None),
                            "d_date": getattr(cand, "effective_trade_date", None),
                            "exc_type": type(e).__name__, "exc_msg": str(e)[:300],
                            "traceback": traceback.format_exc()[-2000:]})
            continue
    build_samples.last_rejects = reject_log
    build_samples.last_exceptions = exc_log
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--n", type=int, default=20, help="最多取前 N 个候选")
    ap.add_argument("--plan-date", default=None,
                   help="候选 plan_date: int / lo-hi (如 20260901-20260917) / 留空=全部")
    ap.add_argument("--setup", default=None, choices=["BOTTOM_SETUP", "INTRADAY_DIP_SETUP"])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "latr_samples.csv"))
    args = ap.parse_args()
    plan_dates = None
    if args.plan_date:
        if "-" in args.plan_date:
            lo, hi = args.plan_date.split("-")
            plan_dates = (int(lo), int(hi))
        else:
            plan_dates = int(args.plan_date)
    print(f"[latr_builder] plan_dates={plan_dates} setup={args.setup} n={args.n}")
    df = build_samples(plan_dates=plan_dates, setup_type=args.setup, sample_n=args.n)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[latr_builder] wrote {len(df)} samples -> {args.out}")
    if not df.empty:
        print(df[["symbol", "d_date", "open_dip_atr", "latr_bucket",
                  "latr_armed", "candidate_status", "return_h3", "s0_alpha_h3"]].head(10).to_string())


if __name__ == "__main__":
    main()
