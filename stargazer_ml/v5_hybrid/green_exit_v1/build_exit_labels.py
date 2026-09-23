"""动态退出标签引擎 v1（绿盘低吸）。

双轨：
  - 2019-2024  -> DAILY_APPROX（日线保守模拟）
  - 2025-2026  -> MINUTE_EXACT（分钟精确模拟，缺缓存则回落 DAILY_APPROX）

四变体：X0_CONTROL / X1 / X1_RF / X2_STAGED
所有参数与状态机冻结于 exit_dyn_spec_v1.md。
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 项目根 d:/stargazer/stargazer_ml

import numpy as np
import pandas as pd

ROOT = Path("D:/stargazer/stargazer_ml/v5_hybrid/green_exit_v1")
GREEN = Path("D:/stargazer/stargazer_ml/v4_2_green_leader_retest/20260830_validation_run2/green_full_labels.parquet")
MINUTE_DIR = Path("D:/stargazer/stargazer_ml/pullback_train/minute_cache_clean")
DAILY = Path("D:/stargazer/stargazer_ml/v5_hybrid/v5_9_live/recent_daily_raw.parquet")
MODE_ID = "G0"
ROUND_TRIP_COST = 0.003  # 与 v4_2_train 对齐（如导入失败用此默认）

# ---------- 冻结参数 ----------
ATR_LOOK = 14
ATR_REF_DAYS = 5
STOP_MULT = 1.5
STOP_MIN, STOP_MAX = 0.03, 0.06
# 第一档：普通反弹保护
T1_PEAK, T1_PEAK_ATR = 0.02, 0.8
T1_DD, T1_DD_ATR = 0.01, 0.45
# 第二档：盈利趋势保护
T2_PEAK, T2_PEAK_ATR = 0.05, 1.5
T2_TRAIL, T2_TRAIL_ATR = 0.015, 0.75
# 反弹失败 REBOUND_FAILURE-v1
RF_DAMAGE, RF_REBOUND, RF_FALL = 0.5, 0.3, 0.2
# 双轨分界年
PRECISION_MIN_YEAR = 2025
VARIANTS = ["X0_CONTROL", "X1", "X1_RF", "X2_STAGED"]


# ---------- 工具 ----------
def parse_ts(ts: int):
    s = str(ts)
    ymd = int(s[:8])
    hhmmss = int(s[8:]) if len(s) > 8 else 0
    return ymd, hhmmss


def _minutes(ts1: int, ts2: int) -> int:
    a = pd.to_datetime(str(int(ts1)), format="%Y%m%d%H%M%S")
    b = pd.to_datetime(str(int(ts2)), format="%Y%m%d%H%M%S")
    return max(1, int((b - a).total_seconds() // 60))


def true_range(hi, lo, cl):
    prev = np.roll(cl, 1)
    prev[0] = cl[0]
    return np.maximum.reduce([hi - lo, np.abs(hi - prev), np.abs(lo - prev)])


def compute_atr_ref(px: pd.DataFrame, entry_idx: int):
    """入场前 5 日 ATR14 中位数。px 须按 date 升序且含 raw_high/low/close。"""
    hi = px["raw_high"].to_numpy(dtype=float)
    lo = px["raw_low"].to_numpy(dtype=float)
    cl = px["raw_close"].to_numpy(dtype=float)
    if entry_idx < ATR_LOOK:
        return None
    tr = true_range(hi, lo, cl)
    atr = pd.Series(tr).rolling(ATR_LOOK, min_periods=ATR_LOOK).mean().to_numpy()
    need = np.arange(entry_idx - ATR_REF_DAYS, entry_idx)  # T-4..T
    valid = need[need >= ATR_LOOK - 1]
    if valid.size == 0:
        return None
    return float(np.median(atr[valid]))


class ExitState:
    """跨 bar 持久化状态；step() 每根 bar/day 调用一次，返回 (reason, price, dt) 或 None。

    同 bar 悲观顺序（关键修正）：
      读取本 bar 开始前的旧高水位(old_hw)
      -> 硬止损
      -> 用旧高水位的移动止盈
      -> 反弹失败（用 entry 基准，与 hw 无关）
      -> 本 bar 结束后才更新高水位
      -> 本 bar 内新激活的止盈线从下一 bar 生效(pending 机制)
    """

    def __init__(self, entry, atr_ref, stop, variant):
        self.entry = entry
        self.atr_ref = atr_ref
        self.stop = stop
        self.variant = variant
        self.hw = entry
        self.activated1 = False
        self.activated2 = False
        self.t1_act = None
        self.t2_act = None
        self.pending1 = False
        self.pending2 = False
        self.damaged = False
        self.reb_low = None
        self.reb_high = None
        self.rf_disabled = False
        self.staged = False
        self._last_dt = None

    def thresholds(self):
        e = self.entry
        a = self.atr_ref
        return (
            max(T1_PEAK, T1_PEAK_ATR * a / e),
            max(T1_DD, T1_DD_ATR * a / e),
            max(T2_PEAK, T2_PEAK_ATR * a / e),
            max(T2_TRAIL, T2_TRAIL_ATR * a / e),
            RF_DAMAGE * a / e,
            RF_REBOUND * a / e,
            RF_FALL * a / e,
        )

    def step(self, o, h, l, c, dt, can_sell):
        self._last_dt = dt
        e = self.entry
        t1p, t1d, t2p, t2t, rfd, rfr, rff = self.thresholds()
        old_hw = self.hw  # 本 bar 开始前的高水位（来自之前 bar）
        peak = old_hw / e - 1
        # 激活判定基于旧高水位；新激活从下一 bar 生效
        if not self.activated1 and peak >= t1p:
            self.pending1 = True
        if not self.activated2 and peak >= t2p:
            self.pending2 = True
        # 反弹失败追踪（用 entry 作基准，与 hw 顺序无关）
        rf_trig = False
        if self.variant in ("X1_RF", "X2_STAGED") and not self.rf_disabled:
            if not self.damaged and l <= e * (1 - rfd):
                self.damaged = True
                self.reb_low = l
                self.reb_high = h
            elif self.damaged:
                self.reb_low = min(self.reb_low, l)
                self.reb_high = max(self.reb_high, h)
                if (self.reb_high - self.reb_low) >= rfr and (self.reb_high - c) >= rff and c < e:
                    rf_trig = True
        if not can_sell:
            self._commit(old_hw, h)
            return None
        # 1) 硬止损（悲观优先）
        if o <= self.stop:
            return ("STOP", o, dt)
        if l <= self.stop:
            return ("STOP", self.stop, dt)
        # 2) 反弹失败
        if rf_trig:
            return ("REBOUND_FAILURE_STAGED" if self.variant == "X2_STAGED"
                    else "REBOUND_FAILURE", c, dt)
        # 3) 移动止盈（用旧高水位；新激活线下一 bar 才生效）
        # 变体门控（exit_dyn_spec_v1 §8 冻结表）：X0_CONTROL = 仅硬止损+时间退出，
        # 无 TRAIL1/TRAIL2；移动止盈只服务 X1 / X1_RF / X2_STAGED。
        if self.variant in ("X1", "X1_RF", "X2_STAGED"):
            if self.activated2:
                trail = old_hw * (1 - t2t)
                if o <= trail:
                    return ("TRAIL2", o, dt)
                if l <= trail:
                    return ("TRAIL2", trail, dt)
            elif self.activated1:
                dd = old_hw * (1 - t1d)
                if o <= dd:
                    return ("TRAIL1", o, dt)
                if l <= dd:
                    return ("TRAIL1", dd, dt)
        self._commit(old_hw, h)
        return None

    def _commit(self, old_hw, h):
        self.hw = max(old_hw, h)
        if self.pending1:
            self.activated1 = True
            self.pending1 = False
            self.t1_act = self._last_dt
        if self.pending2:
            self.activated2 = True
            self.pending2 = False
            self.t2_act = self._last_dt


def _record(event_id, code, signal_date, entry_date, entry_time, entry_price,
            atr_ref, stop, hw, t1_act, t2_act, exit_signal_time, exit_time,
            exit_price, reason, blocked, gross, cost, net, hold_min, hold_days,
            precision, variant):
    return {
        "event_id": event_id, "code": code,
        "signal_date": int(signal_date) if signal_date is not None else 0,
        "entry_date": int(entry_date), "entry_time": int(entry_time),
        "entry_price": float(entry_price), "atr_ref": float(atr_ref),
        "initial_stop_price": float(stop), "high_watermark": float(hw),
        "activation_time": int(t1_act) if t1_act is not None else 0,
        "exit_signal_time": int(exit_signal_time) if exit_signal_time else 0,
        "exit_time": int(exit_time), "exit_price": float(exit_price),
        "exit_reason": reason, "exit_blocked": int(blocked),
        "gross_return": float(gross), "cost": float(cost), "net_return": float(net),
        "holding_minutes": int(hold_min), "holding_days": int(hold_days),
        "label_precision": precision, "strategy_variant": variant,
    }


# ---------- 涨跌停 / 停牌 边界（价格近似；ST 5%/普通10%，20%/30%板已知简化） ----------
LIMIT_UP_MULT = 1.098       # 普通板涨停 ~ +9.8%（锁死：收盘=最高）
LIMIT_DOWN_MULT = 0.902     # 普通板跌停 ~ -9.8%（锁死：收盘=最低）
ST_LIMIT_UP_MULT = 1.05     # ST 板 +5%
ST_LIMIT_DOWN_MULT = 0.95   # ST 板 -5%
BLOCK_MAX_DELAY = 3  # 跌停锁死后最多延迟的交易日数


def load_st_flags(codes, start, end):
    """返回 {code: {date: is_st}}；失败则降级为普通 10% 板（返回空 dict）。"""
    try:
        from v4_2_stockdb import daily as _sd
        df = _sd(list(codes), start=start, end=end, fields="is_st,date,code")
        out = {}
        for code, g in df.groupby("code"):
            out[str(code).zfill(6)] = {int(r["date"]): bool(r["is_st"]) for _, r in g.iterrows()}
        return out
    except Exception:
        return {}


def _is_limit_up(close_p, high_p, prev_close, st):
    if prev_close is None or prev_close <= 0:
        return False
    mult = ST_LIMIT_UP_MULT if st else LIMIT_UP_MULT
    return close_p >= prev_close * mult - 1e-9 and abs(close_p - high_p) < 1e-6


def _is_limit_down(close_p, low_p, prev_close, st):
    if prev_close is None or prev_close <= 0:
        return False
    mult = ST_LIMIT_DOWN_MULT if st else LIMIT_DOWN_MULT
    return close_p <= prev_close * mult + 1e-9 and abs(close_p - low_p) < 1e-6


def run_exit(px, signal_idx, atr_ref, variant, minute_df, prev_close_map, st_map,
             code=None, signal_date=None, event_id=None, entry_idx_override=None):
    """统一退出引擎（日线 + 分钟双轨，交易日历推进，涨跌停/停牌/延迟出场）。

    - 入场：从 signal_idx+1 起按交易日推进（T+1..T+5），遇涨停开盘则顺延；
            连续涨停无法买入 -> 返回 None（放弃）。
    - 卖出：T+2 起可卖；T+5 为最长观察期（封顶）。
    - 跌停锁死 / 无成交 -> 标记 blocked，延迟到下一可卖交易日以开盘价出场。
    - holding_days 以交易日数计（exit_idx - entry_idx）。
    """
    # 1) 入场交易日选择
    entry_idx = None
    if entry_idx_override is not None:
        entry_idx = entry_idx_override
    else:
        for k in range(1, 6):  # T+1 .. T+5
            cand = signal_idx + k
            if cand >= len(px):
                break
            cdate = int(px.iloc[cand]["date"])
            copen = float(px.iloc[cand]["raw_open"])
            pc = prev_close_map.get(cdate)
            stf = bool(st_map.get(cdate, False)) if st_map else False
            mult = ST_LIMIT_UP_MULT if stf else LIMIT_UP_MULT
            if pc and copen >= pc * mult - 1e-9:
                continue  # 涨停开盘无法买入 -> 顺延
            entry_idx = cand
            break
    if entry_idx is None or entry_idx + 1 >= len(px):
        return None
    entry_date = int(px.iloc[entry_idx]["date"])
    t5_idx = min(entry_idx + 4, len(px) - 1)
    if t5_idx <= entry_idx:
        return None
    entry = float(px.iloc[entry_idx]["raw_open"])
    if not np.isfinite(entry) or entry <= 0:
        return None
    stop_dist = min(max(STOP_MULT * atr_ref / entry, STOP_MIN), STOP_MAX)
    stop = entry * (1 - stop_dist)
    st = ExitState(entry, atr_ref, stop, variant)
    first_half = None

    def finish(reason, price, dt, d, blocked=0, precision="DAILY_APPROX"):
        if variant == "X2_STAGED" and st.staged and first_half is not None:
            second = price
            blended = 0.5 * (first_half / entry - 1) + 0.5 * (second / entry - 1)
            net = blended - ROUND_TRIP_COST
            hold = d - entry_idx
            return _record(event_id, code, signal_date, entry_date, entry_date, entry,
                           atr_ref, stop, st.hw, st.t1_act, st.t2_act, dt, dt, second,
                           "REBOUND_FAILURE_STAGED", blocked, blended, ROUND_TRIP_COST,
                           net, hold * 390, hold, precision, variant)
        gross = price / entry - 1
        net = gross - ROUND_TRIP_COST
        hold = d - entry_idx
        return _record(event_id, code, signal_date, entry_date, entry_date, entry,
                       atr_ref, stop, st.hw, st.t1_act, st.t2_act, dt, dt, price,
                       reason, blocked, gross, ROUND_TRIP_COST, net, hold * 390,
                       hold, precision, variant)

    # 入场日：仅更新状态，不可卖
    ed = px.iloc[entry_idx]
    st.step(float(ed["raw_open"]), float(ed["raw_high"]), float(ed["raw_low"]),
            float(ed["raw_close"]), int(ed["date"]), can_sell=False)

    has_minute = minute_df is not None and not minute_df.empty
    precision = "MINUTE_EXACT" if has_minute else "DAILY_APPROX"
    blocked = None  # (reason, price, day_idx)

    for d in range(entry_idx + 1, t5_idx + 1):
        cdate = int(px.iloc[d]["date"])
        pc = prev_close_map.get(cdate)
        stf = bool(st_map.get(cdate, False)) if st_map else False
        cclose = float(px.iloc[d]["raw_close"])
        clow = float(px.iloc[d]["raw_low"])
        sellable = not _is_limit_down(cclose, clow, pc, stf)
        day_bars = None
        if has_minute:
            mb = minute_df[minute_df["date"] // 1000000 == cdate]
            if len(mb):
                day_bars = mb.sort_values("date").reset_index(drop=True)
        if day_bars is not None:
            done = False
            for i in range(len(day_bars)):
                b = day_bars.iloc[i]
                ts = int(b["date"])
                o, h, l, c = float(b["open"]), float(b["high"]), float(b["low"]), float(b["close"])
                ymd, hhmmss = parse_ts(ts)
                if cdate == int(px.iloc[t5_idx]["date"]) and hhmmss >= 145000:
                    vwap_bars = day_bars[(day_bars["date"] % 1000000 >= 145100) &
                                        (day_bars["date"] % 1000000 <= 145500)]
                    vwap = (vwap_bars["amount"].sum() / vwap_bars["volume"].sum()
                            if (len(vwap_bars) and vwap_bars["volume"].sum() > 0) else c)
                    if not sellable:
                        blocked = ("TIME_EXIT", vwap, d)
                        done = True
                        break
                    return finish("TIME_EXIT", vwap, ts, d, 0, "MINUTE_EXACT")
                res = st.step(o, h, l, c, ts, can_sell=True)
                if res:
                    reason, price, rdt = res
                    if variant == "X2_STAGED" and reason == "REBOUND_FAILURE_STAGED" and not st.staged:
                        first_half = price
                        st.staged = True
                        st.rf_disabled = True
                        st.activated2 = True
                        res = None
                    if res is None:
                        continue
                    if not sellable:
                        blocked = (res[0], res[1], d)
                        done = True
                        break
                    return finish(res[0], res[1], rdt, d, 0, "MINUTE_EXACT")
            if done:
                break
        else:
            copen = float(px.iloc[d]["raw_open"])
            chigh = float(px.iloc[d]["raw_high"])
            clow2 = float(px.iloc[d]["raw_low"])
            res = st.step(copen, chigh, clow2, cclose, cdate, can_sell=True)
            if res:
                reason, price, rdt = res
                if variant == "X2_STAGED" and reason == "REBOUND_FAILURE_STAGED" and not st.staged:
                    first_half = price
                    st.staged = True
                    st.rf_disabled = True
                    st.activated2 = True
                    res = None
                if res is None:
                    pass
                elif not sellable:
                    blocked = (res[0], res[1], d)
                    break
                else:
                    return finish(res[0], res[1], rdt, d, 0, precision)

    if blocked is not None:  # 延迟到下一可卖交易日，以开盘价出场（保守）
        reason, price, day_idx = blocked
        for dd in range(day_idx + 1, min(day_idx + 1 + BLOCK_MAX_DELAY, len(px))):
            ndate = int(px.iloc[dd]["date"])
            npc = prev_close_map.get(ndate)
            nstf = bool(st_map.get(ndate, False)) if st_map else False
            if _is_limit_down(float(px.iloc[dd]["raw_close"]), float(px.iloc[dd]["raw_low"]), npc, nstf):
                continue
            return finish(reason, float(px.iloc[dd]["raw_open"]), ndate, dd, 1, precision)
        dd = min(day_idx + BLOCK_MAX_DELAY, len(px) - 1)
        return finish(reason, float(px.iloc[dd]["raw_open"]), int(px.iloc[dd]["date"]), dd, 1, precision)

    # 时间退出（窗口内未触发任何退出）
    d = t5_idx
    cdate = int(px.iloc[d]["date"])
    pc = prev_close_map.get(cdate)
    stf = bool(st_map.get(cdate, False)) if st_map else False
    if _is_limit_down(float(px.iloc[d]["raw_close"]), float(px.iloc[d]["raw_low"]), pc, stf):
        for dd in range(d + 1, min(d + 1 + BLOCK_MAX_DELAY, len(px))):
            ndate = int(px.iloc[dd]["date"])
            npc = prev_close_map.get(ndate)
            nstf = bool(st_map.get(ndate, False)) if st_map else False
            if _is_limit_down(float(px.iloc[dd]["raw_close"]), float(px.iloc[dd]["raw_low"]), npc, nstf):
                continue
            return finish("TIME_EXIT", float(px.iloc[dd]["raw_open"]), ndate, dd, 1, precision)
    if has_minute:
        mb = minute_df[minute_df["date"] // 1000000 == cdate]
        if len(mb):
            vwap_bars = mb[(mb["date"] % 1000000 >= 145100) & (mb["date"] % 1000000 <= 145500)]
            price = (vwap_bars["amount"].sum() / vwap_bars["volume"].sum()
                     if (len(vwap_bars) and vwap_bars["volume"].sum() > 0)
                     else float(px.iloc[d]["raw_close"]))
            return finish("TIME_EXIT", price, int(mb.iloc[-1]["date"]), d, 0, "MINUTE_EXACT")
    price = float(px.iloc[d]["raw_close"])
    return finish("TIME_EXIT", price, cdate, d, 0, precision)


def replay_case(code, explicit_entry_date, variant):
    """审计专用入口：直接指定入场日，绕过 G0 候选筛选，仅验证退出引擎。"""
    code = str(code).zfill(6)
    ed = int(explicit_entry_date)
    try:
        from v4_2_train import _raw_and_qfq
        start = (pd.to_datetime(str(ed)) - pd.Timedelta(days=120)).strftime("%Y%m%d")
        end = (pd.to_datetime(str(ed)) + pd.Timedelta(days=45)).strftime("%Y%m%d")
        prices = _raw_and_qfq([code], start, end)
    except Exception:
        prices = load_daily([code]).get(code)
        prices = prices if prices is not None else pd.DataFrame()
    if isinstance(prices, dict):
        prices = prices.get(code)
    if prices is None or (isinstance(prices, pd.DataFrame) and prices.empty):
        return None
    px = prices.sort_values("date").reset_index(drop=True)
    dates = px["date"].to_numpy(dtype=np.int64)
    loc = {int(d): i for i, d in enumerate(dates)}
    i = loc.get(ed)
    if i is None or i + 1 >= len(px):
        return None
    atr_ref = compute_atr_ref(px, i)
    if atr_ref is None:
        return None
    year = ed // 10000
    minute_df = None
    if year >= PRECISION_MIN_YEAR:
        f = MINUTE_DIR / f"{code}.parquet"
        if f.exists():
            minute_df = pd.read_parquet(f)
    prev_close_map = {int(px.iloc[k]["date"]): float(px.iloc[k - 1]["raw_close"]) if k > 0
                      else float(px.iloc[k]["raw_close"]) for k in range(len(px))}
    st_map = load_st_flags([code], start, end).get(code)
    return run_exit(px, i, atr_ref, variant, minute_df, prev_close_map, st_map,
                   code, ed, None, entry_idx_override=i)


def sim_event(code, signal_date, variant):
    """单事件重放（验收脚本复用同一套引擎）。返回 rec dict 或 None。"""
    code = str(code).zfill(6)
    signal_date = int(signal_date)
    try:
        from v4_2_train import _raw_and_qfq
        start = (pd.to_datetime(str(signal_date)) - pd.Timedelta(days=120)).strftime("%Y%m%d")
        end = (pd.to_datetime(str(signal_date)) + pd.Timedelta(days=45)).strftime("%Y%m%d")
        prices = _raw_and_qfq([code], start, end)
    except Exception:
        prices = load_daily([code]).get(code)
        prices = prices if prices is not None else pd.DataFrame()
    if prices is None or (isinstance(prices, pd.DataFrame) and prices.empty):
        return None
    if isinstance(prices, dict):
        prices = prices.get(code)
    px = prices.sort_values("date").reset_index(drop=True)
    dates = px["date"].to_numpy(dtype=np.int64)
    loc = {int(d): i for i, d in enumerate(dates)}
    i = loc.get(signal_date)
    if i is None or i + 1 >= len(px):
        return None
    atr_ref = compute_atr_ref(px, i + 1)
    if atr_ref is None:
        return None
    year = signal_date // 10000
    minute_df = None
    if year >= PRECISION_MIN_YEAR:
        f = MINUTE_DIR / f"{code}.parquet"
        if f.exists():
            minute_df = pd.read_parquet(f)
    prev_close_map = {int(px.iloc[k]["date"]): float(px.iloc[k - 1]["raw_close"])
                      if k > 0 else float(px.iloc[k]["raw_close"]) for k in range(len(px))}
    st_map = load_st_flags([code], start, end).get(code)
    return run_exit(px, i, atr_ref, variant, minute_df, prev_close_map, st_map,
                   code, signal_date, None)


def load_daily(codes):
    """从本地 raw daily parquet 装载（自带 pre_close，免 SDK）。返回 {code: DataFrame}。"""
    d = pd.read_parquet(DAILY)
    if "code" in d.columns:
        d["code"] = d["code"].astype(str).str.zfill(6)
    else:
        d["code"] = d["symbol"].astype(str).str[-6:]
    d = d[d["code"].isin(codes)]
    d = d.rename(columns={"open": "raw_open", "high": "raw_high",
                          "low": "raw_low", "close": "raw_close"})
    keep = ["code", "date", "raw_open", "raw_high", "raw_low", "raw_close"]
    if "pre_close" in d.columns:
        keep.append("pre_close")
    d = d[keep]
    d["date"] = d["date"].astype(int)
    d = d.sort_values(["code", "date"]).reset_index(drop=True)
    return {c: g.reset_index(drop=True) for c, g in d.groupby("code")}


def build(limit=None, codes=None):
    ev = pd.read_parquet(GREEN)
    ev = ev[ev["mode_id"] == MODE_ID].copy()
    ev = ev.dropna(subset=["date", "code", "event_id"])
    ev["date"] = ev["date"].astype(int)
    ev["code"] = ev["code"].astype(str).str.zfill(6)
    if codes:
        ev = ev[ev["code"].isin(codes)]
    if ev.empty:
        print("no G0 events match the given codes; nothing to build")
        return
    start = "20100101"  # 固定早期起点，确保 ATR_ref 的过去窗口不被截断（与 sim_event 一致）
    end = (pd.to_datetime(str(int(ev["date"].max()))) + pd.Timedelta(days=45)).strftime("%Y%m%d")
    print(f"loading daily prices {start}..{end} for {ev['code'].nunique()} codes...", flush=True)
    try:
        from v4_2_train import _raw_and_qfq
        prices = _raw_and_qfq(sorted(ev["code"].unique()), start, end)
        print(f"loaded raw+qfq via stockdb: {prices.shape[0]} rows")
    except Exception as e:
        print(f"stockdb unavailable ({e}); falling back to local daily parquet")
        prices = load_daily(sorted(ev["code"].unique()))
    if isinstance(prices, pd.DataFrame):
        grouped = {c: g for c, g in prices.groupby("code")}
    else:
        grouped = prices
    by_code = {c: g.sort_values("date").reset_index(drop=True) for c, g in grouped.items()}
    print(f"by_code loaded: {len(by_code)} codes", flush=True)
    st_flags = load_st_flags(sorted(ev["code"].unique()), start, end)
    print(f"st_flags loaded: {len(st_flags)} codes", flush=True)
    minute_cache = {}

    def get_minute(code):
        if code in minute_cache:
            return minute_cache[code]
        f = MINUTE_DIR / f"{code}.parquet"
        df = pd.read_parquet(f) if f.exists() else None
        minute_cache[code] = df
        return df

    recs = []
    skipped = 0
    total = len(ev)
    for idx, row in enumerate(ev.itertuples(index=False)):
        if idx % 500 == 0:
            print(f"progress {idx}/{total} recs={len(recs)} skipped={skipped}", flush=True)
        code = str(row.code).zfill(6)
        px = by_code.get(code)
        if px is None or px.empty:
            skipped += 1
            continue
        dates = px["date"].to_numpy(dtype=np.int64)
        loc = {int(d): i for i, d in enumerate(dates)}
        i = loc.get(int(row.date))
        if i is None or i + 1 >= len(px):
            skipped += 1
            continue
        entry_idx = i + 1
        atr_ref = compute_atr_ref(px, entry_idx)
        if atr_ref is None:
            skipped += 1
            continue
        year = int(row.date) // 10000
        minute_df = get_minute(code) if year >= PRECISION_MIN_YEAR else None
        prev_close_map = {int(px.iloc[k]["date"]): float(px.iloc[k - 1]["raw_close"])
                          if k > 0 else float(px.iloc[k]["raw_close"])
                          for k in range(len(px))}
        st_map = st_flags.get(code)
        for variant in VARIANTS:
            rec = run_exit(px, i, atr_ref, variant, minute_df, prev_close_map, st_map,
                           code, int(row.date), row.event_id)
            if rec is None:
                continue
            rec["event_id"] = row.event_id
            rec["code"] = code
            rec["signal_date"] = int(row.date)
            recs.append(rec)
    df = pd.DataFrame(recs)
    if df.empty:
        print("no records produced")
        return
    for prec in ("DAILY_APPROX", "MINUTE_EXACT"):
        sub = df[df["label_precision"] == prec]
        out = ROOT / "labels" / f"exit_labels_{prec.lower()}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(out, index=False)
        print(f"{prec}: {len(sub)} rows -> {out}")
    print(f"skipped(不可交易/无ATR): {skipped}; total events: {len(ev)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--codes", default="")
    args = ap.parse_args()
    codes = [c for c in args.codes.split(",") if c] or None
    build(limit=args.limit, codes=codes)


if __name__ == "__main__":
    main()
