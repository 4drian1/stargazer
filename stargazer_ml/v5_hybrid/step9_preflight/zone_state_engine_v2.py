"""zone_state_engine_v2.py — TIMING-TRIGGER-v2 状态机 (DIP 紧 / BOTTOM 保留稳定化路径).

与冻结版 zone_state_engine.py 的差异 (用户 2026-09-19 裁定):
- FROZEN_TRIGGER_V1 的 ARM Gate 用开盘态代理 (open_dip_atr>=0.05 / zone_distance_atr<=0.20,
  二者 09:30 即真) -> 95.9% 在 09:30 触发. v2 一律要求"真实 D 日观察": 必须 ZONE_TOUCH 之后,
  再出现 EARLY_RECOVERY / RECLAIM / (BOTTOM 专属) STABILIZATION(K=5) 才进入 MODEL_SCORE_REQUEST.
- 显式分离 soft_break (浅/短破位, 仍可 reclaim) 与 structural_breakdown (深/久破位 -> INVALIDATED,
  回收路径关闭). 冻结版没有此分离: structure_broken 仅在 BREAKDOWN 状态(连续>=3根低于lower)后置位,
  导致 >=3 根破位后 reclaim 成死路径.
- running_pullback_low(S) = min(low since zone touch, ts<=S); 不存在 final_touch_low 未来概念.
- 09:30 MODEL_SCORE_REQUEST 硬要求为 0 (触 Zone 后至少 1 根 recovery/reclaim, 或 K=5 根 stab).
- open_dip_atr 语义更正: (D_open_adj - 前15分钟最低low)/atr = 开盘后15分钟下杀, 非隔夜跳空;
  隔夜跳空为独立字段 gap_atr. v2 不重定义 open_dip_atr, 仅保留.

新增 (Step3 Observability Gate 所需): 在 S=score_ts 处用 v2 trace 计算全部 A3/A4 as-of-S 特征,
通过共享 compute_features_asof(S) 产出 —— 绝不调用冻结版 run_state_engine (否则会把 v2 状态映射回旧语义).
A3 价格类复用冻结公式(math, as-of-S 限制 ts<=S); A4 状态类改用 v2 的 running_pullback_low /
touch_high / break_low / reclaim_price / second_test 语义.
"""
from __future__ import annotations
import math
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from zone_state_engine import Zone

# --- 阈值 (可解释默认值, 非收益调参) ---
APPROACH_BAND_ATR = 1.5
BREAKDOWN_BARS = 3                 # 连续 >=3 根收盘低于 lower -> structural breakdown
STRUCT_BREAK_DEPTH_ATR = 1.0       # 穿透 zone_lower 超 1.0 ATR -> structural breakdown
STAB_K_BARS = 5                    # BOTTOM 稳定化所需连续无新低根数
STAB_ZONE_TOL_ATR = 0.10           # 稳定化时允许略破 lower 的容差


def _hhmmss_to_min_local(x: int) -> float:
    x %= 1_000_000
    hh = x // 10000
    mm = (x % 10000) // 100
    ss = x % 100
    return hh * 60 + mm + ss / 60.0


def compute_features_asof(S: int, zone: Zone, m_scaled: pd.DataFrame, atr_ref: float,
                          gap_atr: Optional[float], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """在 S=score_ts 冻结全部特征 (仅 ts<=S), 用 v2 trace 产出 A3/A4.

    ctx 由 run_state_engine_v2 在 score_ts 处提供: pullback_low / touch_high / break_low /
    touch_ts / reclaim_price / reclaim_failures / second_test_flag / second_test_low.
    价格类 (dip_depth/5m/30m/speed/total_dip/open_dip) 复用冻结 as-of-S 公式;
    状态类 (recovery/reclaim/second_test/time/vwap/volume) 用 v2 语义重写.
    """
    lower, upper, mid = zone.zone_lower, zone.zone_upper, zone.zone_mid
    mS = m_scaled[m_scaled["ts"] <= S]
    if mS.empty:
        return {}
    l = mS["low"].to_numpy(float)
    h = mS["high"].to_numpy(float)
    c = mS["close"].to_numpy(float)
    v = mS["volume"].to_numpy(float) if "volume" in mS else np.zeros(len(mS))
    ts = mS["ts"].to_numpy(int)
    open_ts = int(ts[0])
    open_adj = float(mS.iloc[0]["open"])
    close_S = float(c[-1])
    last_lo = float(l[-1]); last_hi = float(h[-1]); last_cl = float(c[-1])
    day_low_S = float(l.min())

    pullback_low = ctx["pullback_low"]
    touch_high = ctx["touch_high"]
    break_low = ctx["break_low"]
    touch_ts = ctx["touch_ts"]
    reclaim_price = ctx["reclaim_price"]
    reclaim_failures = ctx["reclaim_failures"]
    second_test_flag = ctx["second_test_flag"]
    second_test_low = ctx["second_test_low"]

    # ---- A3: 5 个 v2 新特征 (running low 语义) ----
    realized_dip = (open_adj - day_low_S) / atr_ref if atr_ref > 0 else None
    current_price_dip = (open_adj - close_S) / atr_ref if atr_ref > 0 else None
    zone_pen = (upper - pullback_low) / atr_ref if (pullback_low != math.inf and atr_ref > 0) else None
    current_zone_dist = (close_S - mid) / atr_ref if atr_ref > 0 else None
    current_break_depth = max(0.0, (lower - close_S) / atr_ref) if atr_ref > 0 else 0.0

    # ---- A3: 价格类 (复用冻结 as-of-S 公式) ----
    win15 = mS[(mS["ts"] >= open_ts) & (mS["ts"] <= open_ts + 1500)]
    low_15 = float(win15["low"].min()) if not win15.empty else np.nan
    open_dip = (open_adj - low_15) / atr_ref if not np.isnan(low_15) else np.nan
    if gap_atr is not None and not (isinstance(gap_atr, float) and np.isnan(gap_atr)) and atr_ref > 0:
        prev_close = open_adj - float(gap_atr) * atr_ref
        total_dip = (prev_close - day_low_S) / atr_ref
    else:
        total_dip = np.nan

    def _depth(span):
        w = mS[(mS["ts"] > open_ts) & (mS["ts"] <= open_ts + span)]
        return None if w.empty else (open_adj - w["low"].min()) / atr_ref

    d5 = _depth(500)
    d30 = _depth(3000)
    dip_speed_5m = d5 / 5 if d5 is not None else None
    dip_speed_15m = open_dip / 15.0 if not np.isnan(open_dip) else None
    dip_speed_30m = d30 / 30 if d30 is not None else None

    # ---- A4: 状态类 (v2 语义) ----
    recovery_from_zone_low = (close_S - pullback_low) / atr_ref if (pullback_low != math.inf and atr_ref > 0) else None
    recovery_atr = (close_S - day_low_S) / atr_ref if atr_ref > 0 else None
    dip_rec_ratio = ((close_S - pullback_low) / (touch_high - pullback_low)) \
        if (pullback_low != math.inf and touch_high > -math.inf and (touch_high - pullback_low) > 0 and atr_ref > 0) else None
    zone_break_depth = (lower - break_low) / atr_ref if (break_low != math.inf) else 0.0
    zone_break_duration = int(np.sum(l < lower))
    time_since_touch = (_hhmmss_to_min_local(S) - _hhmmss_to_min_local(touch_ts)) if touch_ts is not None else None
    time_below_zone = int(np.sum(l < lower))
    reclaim_zone_lower = reclaim_price if reclaim_price is not None else None
    reclaim_zone_mid = (reclaim_price + (mid - lower)) if reclaim_price is not None else None
    reclaim_zone_upper = (reclaim_price + (upper - lower)) if reclaim_price is not None else None
    second_test_higher_low = (1 if (second_test_low is not None and second_test_low >= pullback_low) else 0) \
        if second_test_flag else None
    lower_wick = ((last_cl - last_lo) / (last_hi - last_lo)) if (last_hi - last_lo) > 0 else None
    if v.sum() > 0:
        up = c[1:] > c[:-1]
        down = c[1:] < c[:-1]
        sell_vol = float(np.sum(v[1:][down]) / v.sum())
        rebound_vol = float(np.sum(v[1:][up]) / v.sum())
        vwap = float(np.sum(c * v) / v.sum())
        vwap_reclaim = 1 if last_cl >= vwap else 0
    else:
        sell_vol = None
        rebound_vol = None
        vwap_reclaim = None

    # window completeness flags (严格完成窗口语义: 正式模型第一轮仅当 flag=1 才用对应字段)
    window_complete_5m = int(S >= open_ts + 500)
    window_complete_15m = int(S >= open_ts + 1500)
    window_complete_30m = int(S >= open_ts + 3000)
    # zone geometry audit (与缩放后分钟同尺度: analytic-at-P)
    zone_width_atr = (upper - lower) / atr_ref if atr_ref > 0 else None
    open_to_zone_upper_atr = (open_adj - upper) / atr_ref if atr_ref > 0 else None
    open_to_zone_mid_atr = (open_adj - mid) / atr_ref if atr_ref > 0 else None
    open_to_zone_lower_atr = (open_adj - lower) / atr_ref if atr_ref > 0 else None

    return {
        "realized_dip_atr": realized_dip,
        "current_price_dip_atr": current_price_dip,
        "zone_penetration_low_atr": zone_pen,
        "current_zone_distance_atr": current_zone_dist,
        "current_break_depth_atr": current_break_depth,
        "open_dip_atr": open_dip,
        "dip_depth_5m_atr": d5,
        "dip_depth_30m_atr": d30,
        "dip_speed_5m": dip_speed_5m,
        "dip_speed_15m": dip_speed_15m,
        "dip_speed_30m": dip_speed_30m,
        "total_dip_atr": total_dip,
        "recovery_from_zone_low_atr": recovery_from_zone_low,
        "recovery_atr": recovery_atr,
        "dip_recovery_ratio": dip_rec_ratio,
        "reclaim_zone_lower": reclaim_zone_lower,
        "reclaim_zone_mid": reclaim_zone_mid,
        "reclaim_zone_upper": reclaim_zone_upper,
        "reclaim_failure_count": reclaim_failures,
        "second_test_flag": second_test_flag,
        "second_test_higher_low": second_test_higher_low,
        "time_since_zone_touch": time_since_touch,
        "time_below_zone": time_below_zone,
        "lower_wick_ratio": lower_wick,
        "sell_volume_ratio": sell_vol,
        "rebound_volume_ratio": rebound_vol,
        "vwap_reclaim": vwap_reclaim,
        "window_complete_5m": window_complete_5m,
        "window_complete_15m": window_complete_15m,
        "window_complete_30m": window_complete_30m,
        "zone_width_atr": zone_width_atr,
        "open_to_zone_upper_atr": open_to_zone_upper_atr,
        "open_to_zone_mid_atr": open_to_zone_mid_atr,
        "open_to_zone_lower_atr": open_to_zone_lower_atr,
    }


V2_FEATURE_REGISTRY = {
    # A3 v2 新 (无窗口, as-of-S 即时值)
    "realized_dip_atr": {"model_eligible": True, "window": None},
    "current_price_dip_atr": {"model_eligible": True, "window": None},
    "zone_penetration_low_atr": {"model_eligible": True, "window": None},
    "current_zone_distance_atr": {"model_eligible": True, "window": None},
    "current_break_depth_atr": {"model_eligible": True, "window": None},
    # A3 价格 (窗口语义: as-of-S partial, 严格正式模型仅用 window_complete 对应 flag=1)
    "open_dip_atr": {"model_eligible": True, "window": "15m", "window_complete_field": "window_complete_15m",
                     "note": "as-of-S partial: open->min_low[open, min(S,open+15m)]; full 仅当 window_complete_15m=1"},
    "dip_depth_5m_atr": {"model_eligible": True, "window": "5m", "window_complete_field": "window_complete_5m",
                         "note": "as-of-S partial; 严格正式模型仅当 window_complete_5m=1"},
    "dip_depth_30m_atr": {"model_eligible": True, "window": "30m", "window_complete_field": "window_complete_30m",
                          "note": "as-of-S partial; 严格正式模型仅当 window_complete_30m=1; 09:35 不可冒充 30min"},
    "dip_speed_5m": {"model_eligible": True, "window": "5m", "window_complete_field": "window_complete_5m"},
    "dip_speed_15m": {"model_eligible": True, "window": "15m", "window_complete_field": "window_complete_15m"},
    "dip_speed_30m": {"model_eligible": True, "window": "30m", "window_complete_field": "window_complete_30m"},
    "total_dip_atr": {"model_eligible": True, "window": None},
    # A4 状态 (v2 语义)
    "recovery_from_zone_low_atr": {"model_eligible": True, "window": None},
    "recovery_atr": {"model_eligible": True, "window": None},
    "dip_recovery_ratio": {"model_eligible": True, "window": None},
    "reclaim_zone_lower": {"model_eligible": True, "window": None,
                           "note": "NaN=合法缺失 (RECOVERY 路径无 reclaim); 不要 NaN->0"},
    "reclaim_zone_mid": {"model_eligible": True, "window": None,
                         "note": "NaN=合法缺失; 不要 NaN->0"},
    "reclaim_zone_upper": {"model_eligible": True, "window": None,
                           "note": "NaN=合法缺失; 不要 NaN->0"},
    "reclaim_failure_count": {"model_eligible": True, "window": None},
    # second_test 在 v2-primary 不可观测 (首触即 score, S 后才有二次测试) -> 保留定义但标记不进 v2 模型
    "second_test_flag": {"model_eligible": False, "reason": "UNOBSERVABLE_BEFORE_FIRST_SCORE",
                         "note": "v3 允许首次 recovery 后继续观察时可重新启用; 不要永久删除"},
    "second_test_higher_low": {"model_eligible": False, "reason": "UNOBSERVABLE_BEFORE_FIRST_SCORE"},
    "time_since_zone_touch": {"model_eligible": True, "window": None},
    "time_below_zone": {"model_eligible": True, "window": None},
    "lower_wick_ratio": {"model_eligible": True, "window": None},
    "sell_volume_ratio": {"model_eligible": True, "window": None},
    "rebound_volume_ratio": {"model_eligible": True, "window": None},
    "vwap_reclaim": {"model_eligible": True, "window": None},
    # window 完整性 flag
    "window_complete_5m": {"model_eligible": True, "window": None, "is_flag": True},
    "window_complete_15m": {"model_eligible": True, "window": None, "is_flag": True},
    "window_complete_30m": {"model_eligible": True, "window": None, "is_flag": True},
    # geometry audit (zone 尺度, 无窗口)
    "zone_width_atr": {"model_eligible": True, "window": None, "is_geometry": True},
    "open_to_zone_upper_atr": {"model_eligible": True, "window": None, "is_geometry": True},
    "open_to_zone_mid_atr": {"model_eligible": True, "window": None, "is_geometry": True},
    "open_to_zone_lower_atr": {"model_eligible": True, "window": None, "is_geometry": True},
}


def run_state_engine_v2(zone: Optional[Zone], minute: pd.DataFrame, price_scale: float,
                        setup_type: str, atr_ref: float, snapshot_ts: int,
                        open_dip_atr: float = 0.0, zone_distance_atr: float = 0.0,
                        dip_stage: Optional[str] = None,
                        stab_k: int = STAB_K_BARS,
                        struct_break_depth_atr: float = STRUCT_BREAK_DEPTH_ATR,
                        gap_atr: Optional[float] = None) -> Dict[str, Any]:
    """扫描 D 日分钟 (ts<=snapshot_ts), 输出 v2 状态机转移 + FIRST MODEL_SCORE_REQUEST + as-of-S 特征.

    price_scale 将 raw 分钟价投影到 zone 尺度 (analytic-at-P).
    兼容冻结签名: open_dip_atr / zone_distance_atr / dip_stage 预留但 v2 Gate 不使用.
    gap_atr 仅用于 A3 total_dip_atr 反推 prev_close (P_CLOSE-safe, 无前视).
    """
    if zone is None or minute is None or minute.empty:
        return {"final_state": "INVALIDATED", "state_transitions": [],
                "trigger_reason": "no_zone_or_minute", "score_path": None,
                "score_ts": None, "rejection": "NO_ZONE", "features_asof_S": {}}
    m = minute[minute["ts"] <= snapshot_ts].copy()
    if m.empty:
        return {"final_state": "INVALIDATED", "state_transitions": [],
                "trigger_reason": "no_minute_before_snapshot", "score_path": None,
                "score_ts": None, "rejection": "NO_MINUTE", "features_asof_S": {}}
    m = m.sort_values("ts").reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        if c in m:
            m[c] = m[c].to_numpy(float) * price_scale

    lower, upper, mid = zone.zone_lower, zone.zone_upper, zone.zone_mid
    ts = m["ts"].to_numpy(int)
    h = m["high"].to_numpy(float)
    l = m["low"].to_numpy(float)
    c = m["close"].to_numpy(float)
    n = len(m)
    open_adj_0 = float(m.iloc[0]["open"])  # 当日复权开盘价 (analytic-at-P)

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
    touch_idx = -1
    touch_ts = None
    touch_low = math.inf
    touch_high = -math.inf
    approach_ts = None
    pullback_observed = False
    pullback_low = math.inf
    bars_since_last_low = 0
    broke = False
    below_bars = 0
    break_low = math.inf
    structural_breakdown = False
    reclaimed = False
    reclaim_ts = None
    reclaim_price = None
    reclaim_failures = 0
    recovered_once = False
    stabilized = False
    scored = False
    score_ts = None
    score_path = None
    # ---- Timing Frontier 臂 (只读观测, 不改任何 Trigger/Selection 语义) ----
    # 预注册定义 (冻结常量, 禁止事后调参):
    #   显著新低 SIGNIFICANT_NEW_LOW(S): lo_S < running_pullback_low_{S-1}   (ε=0, 严格=现有状态机语义)
    #   bars_since_last_low: 每 bar, 若 SIGNIFICANT_NEW_LOW 则归 0, 否则 +1
    #   running_pullback_low(S) = min(low since touch, ts<=S)
    #   touch_high(S)           = max(high since touch, ts<=S)
    #   dip_recovery_ratio(S)   = (close_S - running_pullback_low_S)
    #                             / max(EPS, touch_high(S) - running_pullback_low_S)
    arm_t0 = None   # 第一次真实 ZONE_TOUCH         (signal_ts = touch_ts)
    arm_t1 = None   # TOUCH 后第 1 根完整 bar 无显著新低 (bars_since_last_low==1)
    arm_t2 = None   # 连续 3 根 bar 无显著新低       (bars_since_last_low==3)
    arm_t3 = None   # dip_recovery_ratio(S) >= 0.25  (分母=touch_high-running_pullback_low, 预注册 25%)
    # second test 追踪 (v2 语义: recovery/reclaim 后再次回踩 Zone)
    re_entry_done = False
    re_entry_ts = None
    second_tested = False
    second_test_flag = 0
    second_test_low = None

    def set_score(t, path):
        nonlocal scored, score_ts, score_path
        if not scored:
            scored = True
            score_ts = int(t)
            score_path = path
            transition_to("MODEL_SCORE_REQUEST", t, f"v2_score:{path}")

    for i in range(n):
        t = int(ts[i])
        hi = h[i]; lo = l[i]; cl = c[i]
        in_zone = (lo <= upper) and (hi >= lower)
        below = lo < lower
        above_close = cl >= upper

        if in_zone and not touched:
            touched = True
            touch_idx = i
            touch_ts = t
            arm_t0 = t
        if touched:
            touch_low = min(touch_low, lo)
            touch_high = max(touch_high, hi)
            if lo < pullback_low:
                pullback_low = lo
                bars_since_last_low = 0
            else:
                bars_since_last_low += 1
            # ---- 臂观测 (只读, 与触发逻辑解耦) ----
            if arm_t1 is None and i > touch_idx and bars_since_last_low == 1:
                arm_t1 = t
            if arm_t2 is None and i > touch_idx and bars_since_last_low == 3:
                arm_t2 = t
            # T3_EARLY_RECOVERY_25: dip_recovery_ratio(S) = (close-pullback_low)/max(EPS, touch_high-pullback_low) >= 0.25
            if arm_t3 is None and touch_high > pullback_low:
                _denom = touch_high - pullback_low
                if _denom > 1e-12 and (c[i] - pullback_low) / _denom >= 0.25:
                    arm_t3 = t

        if state == "WATCH" and approach_ts is None and (hi <= upper + APPROACH_BAND_ATR * atr_ref) and hi > upper:
            approach_ts = t
            transition_to("ZONE_APPROACH", t, "price_within_approach_band")

        if touched and state in ("WATCH", "ZONE_APPROACH"):
            transition_to("ZONE_TOUCH", t, "price_entered_zone")
            if not pullback_observed:
                pullback_observed = True
                transition_to("PULLBACK_OBSERVED", t, "pullback_observed")

        if touched:
            if below:
                below_bars += 1
                break_low = min(break_low, lo)
                if not broke:
                    broke = True
                if (below_bars >= BREAKDOWN_BARS) or ((lower - break_low) >= struct_break_depth_atr):
                    if not structural_breakdown:
                        structural_breakdown = True
                        transition_to("INVALIDATED", t, "structural_breakdown")
            else:
                below_bars = 0
                if broke:
                    if cl >= lower and not structural_breakdown:
                        reclaim_price = cl
                        if not reclaimed:
                            reclaimed = True
                            reclaim_ts = t
                            transition_to("RECLAIM", t, "reclaimed_zone_lower")
                        else:
                            reclaim_failures += 1
                            reclaim_ts = t
                            transition_to("RECLAIM", t, "reclaimed_after_failure")
                        broke = False
                        break_low = math.inf

            if (not recovered_once) and touch_idx >= 0 and i > touch_idx and above_close:
                recovered_once = True
                transition_to("EARLY_RECOVERY", t, "recovered_above_zone_upper")

            if (not stabilized) and touch_idx >= 0 and i > touch_idx \
               and (bars_since_last_low >= stab_k) \
               and (cl >= lower - STAB_ZONE_TOL_ATR * atr_ref):
                stabilized = True
                transition_to("STABILIZATION", t, f"no_new_low_{stab_k}_bars")

        # second test: recovery/reclaim 之后再次回踩 Zone (更高/持平低点为 higher_low)
        if (recovered_once or reclaimed) and not re_entry_done:
            re_entry_done = True
            re_entry_ts = t
        if re_entry_done and (not second_tested) and (t > re_entry_ts) and in_zone:
            second_tested = True
            second_test_low = lo
            second_test_flag = 1

        if (not scored) and (not structural_breakdown) and touched:
            if setup_type == "INTRADAY_DIP_SETUP":
                if recovered_once or reclaimed:
                    set_score(t, "DIP_RECOVERY" if recovered_once else "DIP_RECLAIM")
            else:
                if recovered_once or reclaimed or stabilized:
                    if recovered_once:
                        set_score(t, "BOTTOM_RECOVERY")
                    elif reclaimed:
                        set_score(t, "BOTTOM_RECLAIM")
                    else:
                        set_score(t, "BOTTOM_STABILIZATION")

        if scored:
            break

    if not scored:
        if structural_breakdown:
            final_state = "INVALIDATED"
            rejection = "STRUCTURAL_BREAKDOWN"
        elif not touched:
            final_state = state
            rejection = "NEVER_TOUCH"
        else:
            final_state = state
            rejection = "TOUCH_NO_RECOVERY" if setup_type == "INTRADAY_DIP_SETUP" else "TOUCH_NO_STABILIZATION"
        transitions.append({"state": state, "state_enter_ts": int(enter_ts),
                            "state_exit_ts": int(ts[-1]), "trigger_reason": "snapshot_boundary",
                            "zone_id": zone.zone_id, "snapshot_ts": int(snapshot_ts)})
        feats: Dict[str, Any] = {}
    else:
        final_state = "MODEL_SCORE_REQUEST"
        rejection = None
        ctx = {
            "pullback_low": pullback_low,
            "touch_high": touch_high,
            "break_low": break_low,
            "touch_ts": touch_ts,
            "reclaim_price": reclaim_price,
            "reclaim_failures": reclaim_failures,
            "second_test_flag": second_test_flag,
            "second_test_low": second_test_low,
        }
        feats = compute_features_asof(score_ts, zone, m, atr_ref, gap_atr, ctx)

    out = {
        "final_state": final_state,
        "state_transitions": transitions,
        "trigger_reason": rejection,
        "rejection": rejection,
        "score_path": score_path,
        "score_ts": score_ts,
        "features_asof_S": feats,
        "arm_t0": arm_t0, "arm_t1": arm_t1, "arm_t2": arm_t2, "arm_t3": arm_t3,
    }
    out.update(feats)  # 顶层也暴露, 兼容下游
    return out


if __name__ == "__main__":
    # 轻量自测: 合成 BOTTOM 场景 (开盘入 Zone, 横住 6 根不破, 不修复) 应判 BOTTOM_STABILIZATION 且 S>09:30
    P = 20260105
    D = 20260106
    rows = []
    for i in range(7):  # 09:30..09:36 共 7 根
        hh, mm = 9, 30 + i
        t = int(f"{D}{hh:02d}{mm:02d}00")
        px = 100.0 + (0.1 if i % 2 else 0.0)
        rows.append({"ts": t, "open": px, "high": px + 0.2, "low": px - 0.2, "close": px, "volume": 1e5})
    m = pd.DataFrame(rows)
    z = Zone(zone_id="Z1", zone_source="TEST", zone_lower=99.0, zone_upper=101.0, zone_mid=100.0,
             zone_timestamp=P, zone_width_atr=1.0, zone_strength=3, zone_source_count=1,
             zone_distance_atr=0.0)
    out = run_state_engine_v2(z, m, 1.0, "BOTTOM_SETUP", 2.0, int(f"{D}093600"))
    print(f"[self] bottom_stab: scored={out['score_ts'] is not None} path={out['score_path']} S={out['score_ts']} rejection={out['rejection']}")
    out2 = run_state_engine_v2(z, m, 1.0, "INTRADAY_DIP_SETUP", 2.0, int(f"{D}093600"))
    print(f"[self] dip_stab_only: scored={out2['score_ts'] is not None} rejection={out2['rejection']} (expect False / TOUCH_NO_RECOVERY)")
    assert out["score_ts"] is not None and out["score_ts"] != int(f"{D}093000"), "09:30 must not score"
    print("[self] OK: 09:30 never scores")
