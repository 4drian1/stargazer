"""zone_state_engine.py — ④ Zone Builder + Minute State Engine + A4 连续特征 + 同分钟量能基准.

PIT 约束 (用户裁定):
- 所有 Zone 在 P_CLOSE 冻结: 每个 zone source 的 timestamp <= P_CLOSE.
- confirmed swing 的 confirm_time <= P_CLOSE (不能因 D 日跌到某位就回说 P 日已有支撑).
- Minute State Engine 只扫描 ts <= snapshot_ts 的 D 日分钟; 所有 A4 特征 snapshot-time causal.
- 同分钟量能基准只用过去完整交易日同 minute-of-day, 不读 D 日未来分钟, 不用 "今日量/过去20日量".
- POC/VAL/VAH = OPTIONAL_DISABLED (本模块不实现); Sector/Leader = BLOCKED (不在本模块).

组件:
  build_zones(daily, P, atr_ref, chain=None) -> list[Zone]
  select_primary_zone(zones, price) -> Zone
  run_state_engine(zone, minute, price_scale, setup_type, atr_ref, snapshot_ts,
                   open_dip_atr, zone_distance_atr, dip_stage=None) -> dict
  volume_baseline(D, minute_d, past_minute_dfs, snapshot_ts) -> dict
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

MERGE_BAND_ATR = 0.35      # 用户指定: <=0.35 ATR 合并 cluster
APPROACH_BAND_ATR = 1.5    # ZONE_APPROACH: 距 zone 上沿 <=1.5 ATR 且尚未触及
BREAKDOWN_BARS = 3         # 连续 >=3 根收盘低于 lower 且无 reclaim -> BREAKDOWN


def _mod(ts: int) -> int:
    return ts % 1_000_000   # 取 HHMMSS


def _hhmmss_to_min(x: int) -> float:
    x %= 1_000_000
    hh = x // 10000
    mm = (x % 10000) // 100
    ss = x % 100
    return hh * 60 + mm + ss / 60.0


def _factor_series(dates, chain):
    if chain is None:
        return np.ones(len(dates), dtype=float)
    return np.array([float(chain.factor_at(int(d)) or 1.0) for d in dates], dtype=float)


@dataclass
class Zone:
    zone_id: str
    zone_source: str
    zone_lower: float
    zone_upper: float
    zone_mid: float
    zone_timestamp: int       # max source date (必须 <= P_CLOSE)
    zone_width_atr: float
    zone_strength: int
    zone_source_count: int
    zone_distance_atr: float  # (P_close - mid)/atr_ref
    source_levels: list = field(default_factory=list)  # [(price, source, date)]
    zone_origin: str = "FROZEN"   # FROZEN=已冻结 PIT 来源; PROXY=近似代理(非正式口径)


def build_zones(daily: pd.DataFrame, P: int, atr_ref: float,
                chain=None) -> List[Zone]:
    """在 P_CLOSE 冻结 Zone. daily 必须含 date/open/high/low/close/volume/amount 且覆盖 <=P 的历史."""
    assert atr_ref and atr_ref > 0, "atr_ref 必须为正"
    d = daily[daily["date"] <= int(P)].copy()
    if d.empty or len(d) < 20:
        return []
    d = d.sort_values("date").reset_index(drop=True)
    for c in ("open", "high", "low", "close", "volume", "amount"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    dates = d["date"].to_numpy(int)
    fac = _factor_series(dates, chain)
    fac_p = float(chain.factor_at(int(P)) or 1.0) if chain is not None else 1.0
    scl = fac / fac_p  # analytic-at-P: 在 P 日等价于 raw
    close = d["close"].to_numpy(float) * scl
    low = d["low"].to_numpy(float) * scl
    high = d["high"].to_numpy(float) * scl
    vol = d["volume"].to_numpy(float) if "volume" in d else np.zeros(len(d))
    amt = d["amount"].to_numpy(float) if "amount" in d else np.zeros(len(d))
    p_close = float(close[-1])
    n = len(d)

    levels = []  # (price, source, date, origin)

    # 1) MA20
    ma20 = float(np.mean(close[-20:]))
    levels.append((ma20, "MA20", int(dates[-1]), "FROZEN"))
    # 2) BOLL lower
    roll_std = float(np.std(close[-20:]))
    boll_lower = ma20 - 2 * roll_std
    levels.append((boll_lower, "BOLL_LOWER", int(dates[-1]), "FROZEN"))
    # 3) confirmed swing lows (右 2 根确认, confirm_time<=P)
    for i in range(2, n - 2):
        if low[i] < low[i - 1] and low[i] < low[i - 2] and low[i] < low[i + 1] and low[i] < low[i + 2]:
            levels.append((float(low[i]), "SWING_LOW", int(dates[i]), "FROZEN"))
    # 4) previous support = 最近 20 日最低 low
    prev_support = float(np.min(low[-20:]))
    levels.append((prev_support, "PREV_SUPPORT", int(dates[-1]), "FROZEN"))
    # 5) breakout platform = 最近 30 日最低波动 10 日窗口
    win = 10
    best_range = math.inf
    best = None
    for s in range(max(0, n - 30), max(0, n - win + 1)):
        e = s + win
        if e > n:
            continue
        lo = float(np.min(low[s:e]))
        hi = float(np.max(high[s:e]))
        rg = hi - lo
        if rg < best_range:
            best_range = rg
            best = (lo, hi, int(dates[e - 1]))
    if best is not None:
        lo, hi, dte = best
        levels.append(((lo + hi) / 2.0, "PLATFORM", dte, "FROZEN"))
        levels.append((lo, "PLATFORM_LOW", dte, "FROZEN"))
        levels.append((hi, "PLATFORM_HIGH", dte, "FROZEN"))
    # 6) AVWAP: 正式口径 = 从最近确认 Daily Swing Low anchor 起的 anchored VWAP (需可靠分钟价量).
    #    当前仅日级数据, 只能做 VWAP 近似 -> 改名 AVWAP_PROXY_DAILY, origin=PROXY, 正式 AVWAP 暂 DISABLED.
    w = vol[-20:]
    if w.sum() > 0:
        avwap = float(np.sum(close[-20:] * w) / w.sum())
    elif amt[-20:].sum() > 0:
        avwap = float(np.sum(close[-20:] * amt[-20:]) / amt[-20:].sum())
    else:
        avwap = float(np.mean(close[-20:]))
    levels.append((avwap, "AVWAP_PROXY_DAILY", int(dates[-1]), "PROXY"))
    # 7) integer price levels near p_close
    for step in (1.0, 5.0, 10.0):
        lvl = round(p_close / step) * step
        if abs(lvl - p_close) <= 2 * atr_ref:
            levels.append((float(lvl), "INTEGER_PRICE", int(dates[-1]), "FROZEN"))

    # 断言: 所有 source timestamp <= P_CLOSE
    max_ts = max(dt for _, _, dt, _ in levels)
    assert max_ts <= int(P), f"zone source {max_ts} > P_CLOSE {P}"

    # 合并 cluster (<=0.35 ATR)
    band = MERGE_BAND_ATR * atr_ref
    levels_sorted = sorted(levels, key=lambda x: x[0])
    clusters: List[list] = []
    cur = [levels_sorted[0]]
    cur_max = levels_sorted[0][0]
    for lv in levels_sorted[1:]:
        if lv[0] - cur_max <= band:
            cur.append(lv)
            cur_max = max(cur_max, lv[0])
        else:
            clusters.append(cur)
            cur = [lv]
            cur_max = lv[0]
    clusters.append(cur)

    zones = []
    for zi, cl in enumerate(clusters):
        prices = [x[0] for x in cl]
        srcs = [x[1] for x in cl]
        dts = [x[2] for x in cl]
        origins = [x[3] for x in cl]
        lower = float(min(prices))
        upper = float(max(prices))
        mid = float(np.mean(prices))
        zones.append(Zone(
            zone_id=f"Z{zi + 1}",
            zone_source="|".join(sorted(set(srcs))),
            zone_lower=lower, zone_upper=upper, zone_mid=mid,
            zone_timestamp=int(max(dts)),
            zone_width_atr=(upper - lower) / atr_ref,
            zone_strength=len(cl),
            zone_source_count=len(cl),
            zone_distance_atr=(p_close - mid) / atr_ref,
            source_levels=cl,
            zone_origin="PROXY" if "PROXY" in origins else "FROZEN",
        ))
    return zones


def freeze_check(zones: List[Zone], P: int):
    """返回 (all_frozen, max_source_ts). 用于契约测试断言 zone 全部冻结于 P_CLOSE."""
    if not zones:
        return (True, 0)
    max_ts = max(z.zone_timestamp for z in zones)
    return (max_ts <= int(P), max_ts)


def select_primary_zone(zones: List[Zone], price: float) -> Optional[Zone]:
    if not zones:
        return None
    return sorted(zones, key=lambda z: (-z.zone_strength, abs(z.zone_distance_atr)))[0]


def _arm_condition(setup_type, open_dip_atr, zone_distance_atr, dip_stage, touch_happened,
                   structure_broken=False, hold_zone_flag=False, recovery_flag=False):
    """恢复冻结合同的 BOTTOM / DIP 不同 ARM 逻辑.

    BOTTOM_ARMED (冻结):
        zone_distance_atr <= 0.20
        AND not structure_broken
        AND (open_dip_atr >= 0.05 OR hold_zone_flag OR recovery_flag)
      —— 强调 "本来就在 Zone 附近、轻微下杀/守住/直接修复" 都算, 不强制触 Zone;
         0.05–0.10 ATR 轻微下杀保留; 显式 not structure_broken.

    DIP_ARMED (冻结):
        zone_touch (touch_happened)
        AND 0.10 <= open_dip_atr <= 0.40
        AND not structure_broken
      —— PRE_PULLBACK 未真到 Zone 绝不 ARM (touch_happened=False).

    hold_zone_flag = 已触 Zone 且未破位; recovery_flag = 已 reclaim/首次 recovery.
    """
    if setup_type == "BOTTOM_SETUP":
        return (zone_distance_atr <= 0.20 and not structure_broken and
                (open_dip_atr >= 0.05 or hold_zone_flag or recovery_flag))
    if setup_type == "INTRADAY_DIP_SETUP":
        if not touch_happened:
            return False  # PRE_PULLBACK 未到 Zone 绝不 ARM
        return 0.10 <= open_dip_atr <= 0.40 and not structure_broken
    return touch_happened and zone_distance_atr <= 0.20 and not structure_broken


def run_state_engine(zone: Optional[Zone], minute: pd.DataFrame, price_scale: float,
                     setup_type: str, atr_ref: float, snapshot_ts: int,
                     open_dip_atr: float, zone_distance_atr: float,
                     dip_stage: Optional[str] = None) -> Dict[str, Any]:
    """扫描 D 日分钟 (ts<=snapshot_ts), 输出状态机转移 + A4 连续特征.

    price_scale 将 raw 分钟价投影到 zone 尺度 (analytic-at-P),
    由调用方传 chain.factor_at(D)/chain.factor_at(P).
    """
    if zone is None or minute is None or minute.empty:
        return {"final_state": "INVALIDATED", "transitions": [],
                "trigger_reason": "no_zone_or_minute"}
    m = minute[minute["ts"] <= snapshot_ts].copy()
    if m.empty:
        return {"final_state": "INVALIDATED", "transitions": [],
                "trigger_reason": "no_minute_before_snapshot"}
    m = m.sort_values("ts").reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        if c in m:
            m[c] = m[c].to_numpy(float) * price_scale

    lower, upper, mid = zone.zone_lower, zone.zone_upper, zone.zone_mid
    ts = m["ts"].to_numpy(int)
    h = m["high"].to_numpy(float)
    l = m["low"].to_numpy(float)
    c = m["close"].to_numpy(float)
    v = m["volume"].to_numpy(float) if "volume" in m else np.zeros(len(m))
    n = len(m)
    last_close = float(c[-1])

    transitions = []

    def transition_to(new_state, t, reason):
        nonlocal state, enter_ts
        if new_state == state:
            return
        transitions.append({"state": state, "state_enter_ts": int(enter_ts),
                            "state_exit_ts": int(t), "trigger_reason": reason,
                            "zone_id": zone.zone_id, "snapshot_ts": int(snapshot_ts)})
        state = new_state
        enter_ts = t

    state = "WATCH"
    enter_ts = int(ts[0])

    touched = False
    touch_ts = None
    touch_low = math.inf
    touch_high = -math.inf
    approach_ts = None
    armed = False
    broke = False
    break_ts = None
    break_low = math.inf
    reclaimed = False
    reclaim_ts = None
    reclaim_price = None
    reclaim_failures = 0
    recovered_once = False
    second_tested = False
    second_test_ts = None
    second_test_low = None
    below_bars = 0

    for i in range(n):
        t = int(ts[i])
        hi = h[i]
        lo = l[i]
        cl = c[i]
        in_zone = (lo <= upper) and (hi >= lower)
        below = lo < lower
        above_close = cl > upper

        if in_zone:
            if not touched:
                touched = True
                touch_ts = t
            touch_low = min(touch_low, lo)
            touch_high = max(touch_high, hi)

        # ZONE_APPROACH: WATCH 且价格进入 approach band (尚未触及)
        if state == "WATCH" and approach_ts is None and (hi <= upper + APPROACH_BAND_ATR * atr_ref) and hi > upper:
            approach_ts = t
            transition_to("ZONE_APPROACH", t, "price_within_approach_band")

        # ZONE_TOUCH
        if touched and state in ("WATCH", "ZONE_APPROACH"):
            transition_to("ZONE_TOUCH", t, "price_entered_zone")

        # ARMED (pre-ARM 状态: WATCH / ZONE_APPROACH / ZONE_TOUCH 均可评估)
        structure_broken = state in ("BREAKDOWN", "INVALIDATED")
        hold_zone_flag = touched and not broke
        recovery_flag = bool(reclaimed) or recovered_once
        if (state in ("WATCH", "ZONE_APPROACH", "ZONE_TOUCH")) and not armed:
            if _arm_condition(setup_type, open_dip_atr, zone_distance_atr, dip_stage, touched,
                             structure_broken, hold_zone_flag, recovery_flag):
                armed = True
                transition_to("ARMED", t, f"setup_prereg_met({setup_type})")

        # ARMED 之后的承接/破位/二次测试 (BREAKDOWN 后停止)
        if armed and state in ("ARMED", "RECLAIM", "SECOND_TEST"):
            if below:
                below_bars += 1
                break_low = min(break_low, lo)
                if not broke:
                    broke = True
                    break_ts = t
            else:
                below_bars = 0
                if broke:
                    # 重新站回 zone lower 以上 -> reclaim
                    if cl >= lower:
                        if not reclaimed:
                            reclaimed = True
                            reclaim_ts = t
                            reclaim_price = float(cl)
                            transition_to("RECLAIM", t, "reclaimed_zone_lower")
                        else:
                            reclaim_failures += 1
                            reclaim_ts = t
                            reclaim_price = float(cl)
                            transition_to("RECLAIM", t, "reclaimed_after_failure")
                        broke = False
                        break_ts = None
                if above_close and touched:
                    recovered_once = True
                # SECOND_TEST: 第一次 recovery 后再次回踩 zone
                if recovered_once and in_zone and not second_tested and reclaimed:
                    second_tested = True
                    second_test_ts = t
                    second_test_low = float(lo)
                    transition_to("SECOND_TEST", t, "pullback_to_zone_after_recovery")
            # BREAKDOWN: 连续低于 lower 且无 reclaim
            if broke and below_bars >= BREAKDOWN_BARS:
                transition_to("BREAKDOWN", t, f"below_lower_{BREAKDOWN_BARS}_bars")

    # 末态归并
    if state not in ("BREAKDOWN", "INVALIDATED"):
        if approach_ts is None and touch_ts is None and abs(zone_distance_atr) > 3.0:
            transition_to("INVALIDATED", int(ts[-1]), "zone_irrelevant")
        elif state in ("ARMED", "RECLAIM", "SECOND_TEST"):
            transition_to("MODEL_SCORE_REQUEST", int(ts[-1]), "armed_at_snapshot")
        else:
            # 仍处于 WATCH/APPROACH/TOUCH 但未 ARM -> 以 snapshot 为边界收尾
            transitions.append({"state": state, "state_enter_ts": int(enter_ts),
                               "state_exit_ts": int(ts[-1]), "trigger_reason": "snapshot_boundary",
                               "zone_id": zone.zone_id, "snapshot_ts": int(snapshot_ts)})
    else:
        transitions.append({"state": state, "state_enter_ts": int(enter_ts),
                           "state_exit_ts": int(ts[-1]), "trigger_reason": "snapshot_boundary",
                           "zone_id": zone.zone_id, "snapshot_ts": int(snapshot_ts)})

    feats = _a4_features(zone, atr_ref, l, h, c, v, last_close, touch_ts, touch_low,
                         touch_high, break_ts, break_low, reclaim_ts, reclaim_price,
                         reclaim_failures, second_test_ts, second_test_low, snapshot_ts)
    feats["final_state"] = state
    feats["state_transitions"] = transitions
    feats["zone_id"] = zone.zone_id
    return feats


def _a4_features(zone, atr_ref, l, h, c, v, last_close, touch_ts, touch_low, touch_high,
                 break_ts, break_low, reclaim_ts, reclaim_price, reclaim_failures,
                 second_test_ts, second_test_low, snapshot_ts):
    lower, upper, mid = zone.zone_lower, zone.zone_upper, zone.zone_mid
    n = len(l)
    day_low = float(np.min(l))

    # recovery_from_zone_low: 仅用 touch 期间形成的 low (snapshot 之前必有, 因果)
    recovery_from_zone_low = (last_close - touch_low) / atr_ref if (touch_ts is not None and not math.isinf(touch_low)) else None
    recovery_atr = (last_close - day_low) / atr_ref if day_low > 0 else None
    dip_recovery_ratio = ((last_close - touch_low) / (touch_high - touch_low)) \
        if (touch_ts is not None and not math.isinf(touch_low) and (touch_high - touch_low) > 0) else None

    zone_break_depth = (lower - break_low) / atr_ref if (break_ts is not None and not math.isinf(break_low)) else 0.0
    zone_break_duration = int(np.sum(l < lower)) if break_ts is not None else 0

    time_since_zone_touch = (_hhmmss_to_min(snapshot_ts) - _hhmmss_to_min(touch_ts)) if touch_ts is not None else None
    time_below_zone = int(np.sum(l < lower))

    reclaim_zone_lower = reclaim_price if (reclaim_ts is not None and reclaim_price is not None) else None
    reclaim_zone_mid = (reclaim_price + (mid - lower)) if reclaim_zone_lower is not None else None
    reclaim_zone_upper = (reclaim_price + (upper - lower)) if reclaim_zone_lower is not None else None

    second_test_flag = 1 if second_test_ts is not None else 0
    if second_test_ts is not None and not math.isinf(touch_low):
        second_test_higher_low = 1 if (second_test_low is not None and second_test_low >= touch_low) else 0
    else:
        second_test_higher_low = None  # 之前必须 NaN

    last_lo = float(l[-1])
    last_hi = float(h[-1])
    last_cl = float(c[-1])
    lower_wick_ratio = ((last_cl - last_lo) / (last_hi - last_lo)) if (last_hi - last_lo) > 0 else None

    up = c[1:] > c[:-1]
    down = c[1:] < c[:-1]
    sell_volume_ratio = float(np.sum(v[1:][down]) / v.sum()) if v.sum() > 0 else None
    rebound_volume_ratio = float(np.sum(v[1:][up]) / v.sum()) if v.sum() > 0 else None

    vwap = float(np.sum(c * v) / v.sum()) if v.sum() > 0 else None
    vwap_reclaim = 1 if (vwap is not None and last_close >= vwap) else (0 if vwap is not None else None)

    return {
        "zone_break_depth_atr": zone_break_depth,
        "zone_break_duration": zone_break_duration,
        "recovery_from_zone_low_atr": recovery_from_zone_low,
        "recovery_atr": recovery_atr,
        "dip_recovery_ratio": dip_recovery_ratio,
        "reclaim_zone_lower": reclaim_zone_lower,
        "reclaim_zone_mid": reclaim_zone_mid,
        "reclaim_zone_upper": reclaim_zone_upper,
        "reclaim_failure_count": reclaim_failures,
        "second_test_flag": second_test_flag,
        "second_test_higher_low": second_test_higher_low,
        "time_since_zone_touch": time_since_zone_touch,
        "time_below_zone": time_below_zone,
        "lower_wick_ratio": lower_wick_ratio,
        "sell_volume_ratio": sell_volume_ratio,
        "rebound_volume_ratio": rebound_volume_ratio,
        "vwap_reclaim": vwap_reclaim,
    }


def volume_baseline(D: int, minute_d: pd.DataFrame, past_minute_dfs: List[pd.DataFrame],
                    snapshot_ts: int) -> Dict[str, Any]:
    """同分钟量能基准: D 日 snapshot 分钟-of-day 的累计量 vs 过去完整交易日同 minute-of-day median.

    比较用 HHMMSS (ts%1e6), 不读 D 日未来分钟, 不与 "今日量/过去20日量" 混淆.
    """
    if minute_d is None or minute_d.empty:
        return {"cum_volume_vs_hist_same_time": None, "cum_turnover_vs_hist_same_time": None}
    md = _mod(snapshot_ts)

    def cum_upto(df):
        sub = df[df["ts"].map(_mod) <= md]
        vol = float(sub["volume"].sum()) if "volume" in sub else 0.0
        amt = float(sub["amount"].sum()) if "amount" in sub else 0.0
        return vol, amt

    d_vol, d_amt = cum_upto(minute_d)
    hist_vol, hist_amt = [], []
    for pdf in past_minute_dfs:
        if pdf is None or pdf.empty:
            continue
        vol, amt = cum_upto(pdf)
        hist_vol.append(vol)
        hist_amt.append(amt)
    base_vol = float(np.median(hist_vol)) if hist_vol else None
    base_amt = float(np.median(hist_amt)) if hist_amt else None
    return {
        "cum_volume_vs_hist_same_time": (d_vol / base_vol) if base_vol else None,
        "cum_turnover_vs_hist_same_time": (d_amt / base_amt) if base_amt else None,
        "_d_cum_vol": d_vol, "_hist_vol_median": base_vol,
    }


if __name__ == "__main__":
    # 轻量自测: 构造一段 D 日分钟, 验证引擎跑通 + 状态单调
    P = 20260105
    n = 40
    d = pd.DataFrame({
        "date": [P - (n - 1 - i) for i in range(n)],
        "open": [100 + i for i in range(n)], "high": [102 + i for i in range(n)],
        "low": [98 + i for i in range(n)], "close": [101 + i for i in range(n)],
        "volume": [1e6] * n, "amount": [1e8] * n,
    })
    zones = build_zones(d, P, atr_ref=2.0)
    print(f"[self] zones={len(zones)} frozen_ok={freeze_check(zones, P)[0]}")
    pz = select_primary_zone(zones, 120.0)
    print(f"[self] primary={pz.zone_id} src={pz.zone_source} mid={pz.zone_mid:.2f}")
    # 分钟: 价格从 105 跌到 zone(101) 触后回收
    rows = []
    base = 20260106
    for i, px in enumerate([105, 104, 103, 102, 101, 100.5, 101.5, 102.5, 103]):
        hh = 9 + (i + 31) // 60
        mm = (i + 31) % 60
        ts = int(f"{base}{hh:02d}{mm:02d}00")
        rows.append({"ts": ts, "open": px, "high": px + 0.5, "low": px - 0.5,
                     "close": px, "volume": 1e5, "amount": 1e7})
    m = pd.DataFrame(rows)
    snap = int(f"{base}150000")
    out = run_state_engine(pz, m, 1.0, "BOTTOM_SETUP", 2.0, snap, 0.0, 0.1)
    print(f"[self] final_state={out['final_state']} transitions={[t['state'] for t in out['state_transitions']]}")
    print(f"[self] recovery_from_zone_low_atr={out['recovery_from_zone_low_atr']}")
