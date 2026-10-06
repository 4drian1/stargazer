"""GREEN-CAR-v0.1 新标签函数 (合同 §4, 2026-09-29, 工程范围冻结后实现).

本模块独立于旧 v0.2 引擎的 _compute_minute_window / _compute_daily_window;
**不修改旧函数, 不追溯改写旧 v0.2 报告口径**. 旧 v0.2 函数和结果保持不变.

CAR 主标签与旧 v0.2 的关键区别:
  - 起点 S = 物理事件首个完整信号 bar 结束
  - 主标签 future_reclaim_60m: 只对 S 时 close < zone_upper 的 BREAK/ALREADY_BELOW
    事件定义; S 时已收回 → RECLAIMED_AT_S 分流 (不计入主标签分母)
  - 持续性标签 future_hold_10m: 首次收回后 10 交易分钟 close 均 ≥ zone_upper;
    label_end_ts = 首次收回时刻 + 10 交易分钟 (可能跨日, 用交易分钟日历核验)
  - 未收回 time_to_first_future_reclaim=NULL + 观察终点/截断标志, 不用窗长+1

修正 (2026-09-29 用户反例):
  1. 不完整窗口 → reclaim/hold = None + 失效原因, 不标 0 (不作为有效负样本)
  2. 窗口用完整时间戳 (day+slot), 不只用 HHMMSS → 避免跨日同时刻污染
  3. 跨日观察: 用 cal 跨日补槽 (有下一交易日数据时继续, 不一律截断)
  4. 穿透/收回/持续性分别记录终点; 事件整体闭环取依赖数据的最晚终点
  5. 未知 signal_type → UNKNOWN_SIG_TYPE, 显式拒绝不落入 ELIGIBLE

事件分流 (合同 §4):
  - RECLAIMED_AT_S: S 时 close ≥ zone_upper → 不进主标签分母
  - ALREADY_ABOVE: APPROACH 从未跌破 → 不进主标签分母
  - ELIGIBLE: BREAK/ALREADY_BELOW 且 S 时 close < zone_upper → 进入主标签分母
  - INCOMPLETE_WINDOW: 窗口不完整 → reclaim=NULL, 不作为有效负样本
  - 过滤掉的事件不能算通过

授权边界: 本模块只实现函数 + 合成测试; 真实回放前需另签去重请求清单 + SHA256.
本轮不训练 CAR-G/CAR-LR. 成功标准: 标签和因果链算得对, 可重放.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
try:
    import gcp_round_b_engine as E
except ImportError:
    import sys
    sys.path.insert(0, str(HERE))
    import gcp_round_b_engine as E


# ---- CAR 标签字段名 (与旧 v0.2 字段名不冲突, 加 future_ 前缀) ----
# 分窗口状态/原因 (30m/60m/hold 各自独立, 不互相覆盖)
CAR_LABEL_FIELDS = [
    "future_reclaim_30m", "future_reclaim_60m",
    "future_penetration_30m", "future_penetration_60m",
    "time_to_first_future_reclaim_30m", "time_to_first_future_reclaim_60m",
    "future_hold_10m", "future_hold_label_end_ts",
    "future_window_complete_30m", "future_window_complete_60m",
    "future_window_truncated_30m", "future_window_truncated_60m",
    # 分窗口状态/原因 (各自独立)
    "future_status_30m", "future_status_60m", "future_status_hold",
    "future_reason_30m", "future_reason_60m", "future_reason_hold",
    # 综合状态/原因 (按固定规则从分窗口生成, 不受调用顺序影响)
    "future_reclaim_status", "future_ineligible_reason",
    "future_label_end_ts",
]


def _slot_of(ts: int) -> int:
    """14 位 ts → HHMMSS 槽位."""
    s = str(int(ts))
    return int(s[-6:]) if len(s) >= 6 else int(s)


def _day_of(ts: int) -> int:
    """14 位 ts → YYYYMMDD."""
    return int(str(int(ts))[:8])


def classify_event_at_s(sig: dict) -> str:
    """S 时点分流 (合同 §4).

    返回:
      - "RECLAIMED_AT_S": S 时 close ≥ zone_upper → 不进主标签分母
      - "ALREADY_ABOVE": APPROACH 事件且 S 时 close < zone_upper → 不进主标签分母
      - "ELIGIBLE": BREAK/ALREADY_BELOW 且 S 时 close < zone_upper → 进入主标签分母
      - "UNKNOWN_SIG_TYPE": 未知 signal_type → 显式拒绝, 不落入 ELIGIBLE
    """
    sig_type = sig.get("signal_type", sig.get("sig_type"))
    close_s = float(sig.get("signal_close", sig.get("close_at_signal")))
    zone_upper = float(sig["zone_upper"])

    # 未知 signal_type → 显式拒绝 (不落入 ELIGIBLE)
    if sig_type not in ("APPROACH", "BREAK", "ALREADY_BELOW"):
        return "UNKNOWN_SIG_TYPE"

    # S 时已收回 → 分流
    if close_s >= zone_upper:
        return "RECLAIMED_AT_S"

    # APPROACH 从未跌破 → ALREADY_ABOVE
    if sig_type == "APPROACH":
        return "ALREADY_ABOVE"

    return "ELIGIBLE"


def _build_future_window_slots(
    sig_slot: int, sig_day: int, wlen: int, grid: list, cal: list | None,
) -> list:
    """构建未来 wlen 个交易分钟槽位的完整 (day, slot) 序列 (跨日用 cal).

    返回 [(day, slot), ...] 长度 ≤ wlen.
    - 同日: 从 sig_slot 后第 1 个槽位起, 按 expected_grid 顺序
    - 跨日: 若同日槽位不足 wlen, 用下一交易日 grid 补足 (cal 提供)
    - 无 cal 或 cal 耗尽: 返回实际可构建的部分 (截断)
    """
    exp = E.expected_grid(grid, sig_day)
    try:
        sig_gi = exp.index(sig_slot)
    except ValueError:
        return []
    slots = []
    same_day = exp[sig_gi + 1: sig_gi + 1 + wlen]
    for s in same_day:
        slots.append((sig_day, s))
    if len(slots) >= wlen or cal is None:
        return slots[:wlen]
    try:
        day_idx = cal.index(sig_day)
    except (ValueError, TypeError):
        return slots[:wlen]
    remaining = wlen - len(slots)
    di = day_idx + 1
    while remaining > 0 and di < len(cal):
        next_day = cal[di]
        next_exp = E.expected_grid(grid, next_day)
        for s in next_exp[:remaining]:
            slots.append((next_day, s))
            remaining -= 1
            if remaining <= 0:
                break
        di += 1
    return slots[:wlen]


def compute_future_reclaim_window(
    sig: dict,
    ts: np.ndarray,
    lo: np.ndarray,
    cl: np.ndarray,
    sig_idx: int,
    atr_ref: float,
    grid: list,
    signal_day: int,
    wlen: int,
    wname: str,
    fields: dict,
    cal: list | None = None,
) -> None:
    """计算 CAR 未来收回窗口标签 (30m/60m).

    修正点见模块 docstring. fields 是本模块 init_car_label_fields() 初始化的 dict.
    label_end_ts 取实际依赖数据的最晚终点 (穿透或收回的较晚者).
    """
    zone_upper = float(sig["zone_upper"])
    zone_lower = float(sig["zone_lower"])
    signal_ts = int(sig["signal_ts"])
    sig_slot = _slot_of(signal_ts)

    # ---- 构建未来 wlen 个交易分钟槽位 (跨日用 cal) ----
    future_slots = _build_future_window_slots(sig_slot, signal_day, wlen, grid, cal)
    if not future_slots:
        fields[f"future_window_complete_{wname}"] = False
        fields[f"future_window_truncated_{wname}"] = True
        fields[f"future_reclaim_{wname}"] = None
        fields[f"future_penetration_{wname}"] = None
        fields[f"time_to_first_future_reclaim_{wname}"] = None
        fields["future_ineligible_reason"] = "SIGNAL_SLOT_NOT_IN_GRID"
        fields["future_reclaim_status"] = "INELIGIBLE"
        return

    # 期望的 (day, slot) 集合 (完整时间戳, 不跨日混淆)
    exp_pairs = set(future_slots)
    # 实际 bars: ts > signal_ts, 解析 (day, slot), 必须在 exp_pairs 里
    win_bars = []
    win_bars_by_pair = {}  # (day, slot) → (ts, lo, cl); 用于逐槽核验+冲突检测
    for i in range(sig_idx + 1, len(ts)):
        t = int(ts[i])
        if t <= signal_ts:
            continue
        d = _day_of(t)
        s = _slot_of(t)
        if (d, s) in exp_pairs:
            # 冲突重复: 同 (day, slot) 多条记录 → 显式拒绝
            if (d, s) in win_bars_by_pair:
                fields[f"future_window_complete_{wname}"] = False
                fields[f"future_window_truncated_{wname}"] = True
                fields[f"future_reclaim_{wname}"] = None
                fields[f"future_penetration_{wname}"] = None
                fields[f"time_to_first_future_reclaim_{wname}"] = None
                fields[f"future_status_{wname}"] = "INELIGIBLE"
                fields[f"future_reason_{wname}"] = f"DUPLICATE_SLOT pair={(d, s)}"
                _compute_aggregate_status(fields)
                return
            # 无效价格 (NaN/inf) → 数据有效性失效, 输出 NULL+原因
            lo_v = float(lo[i])
            cl_v = float(cl[i])
            if not (np.isfinite(lo_v) and np.isfinite(cl_v)):
                fields[f"future_window_complete_{wname}"] = False
                fields[f"future_window_truncated_{wname}"] = True
                fields[f"future_reclaim_{wname}"] = None
                fields[f"future_penetration_{wname}"] = None
                fields[f"time_to_first_future_reclaim_{wname}"] = None
                fields[f"future_status_{wname}"] = "INELIGIBLE"
                fields[f"future_reason_{wname}"] = f"INVALID_PRICE pair={(d, s)}"
                _compute_aggregate_status(fields)
                return
            win_bars_by_pair[(d, s)] = (t, lo_v, cl_v)

    # 按预期顺序整理 win_bars; 逐槽核验 (不只看行数)
    win_bars = []
    missing = []
    for pair in future_slots:
        if pair in win_bars_by_pair:
            win_bars.append(win_bars_by_pair[pair])
        else:
            missing.append(pair)

    # 完整性: 逐槽核验 (不只看行数)
    complete = (len(future_slots) == wlen) and not missing
    truncated = (len(future_slots) < wlen) or bool(missing)
    fields[f"future_window_complete_{wname}"] = bool(complete)
    fields[f"future_window_truncated_{wname}"] = bool(truncated)

    # ---- 数据有效性检查: ATR_ref 无效 → 穿透/收回 NULL+原因 ----
    atr_valid = atr_ref is not None and float(atr_ref) > 0 and np.isfinite(float(atr_ref))
    if not atr_valid:
        fields[f"future_status_{wname}"] = "INELIGIBLE"
        fields[f"future_reclaim_{wname}"] = None
        fields[f"future_penetration_{wname}"] = None
        fields[f"time_to_first_future_reclaim_{wname}"] = None
        fields[f"future_reason_{wname}"] = "INVALID_ATR_ref<=0_or_nan"
        if win_bars:
            _update_event_label_end(fields, win_bars[-1][0])
        _compute_aggregate_status(fields)
        return

    # ---- S 时分流 ----
    status = classify_event_at_s(sig)
    if status == "UNKNOWN_SIG_TYPE":
        fields[f"future_status_{wname}"] = "INELIGIBLE"
        fields[f"future_reclaim_{wname}"] = None
        fields[f"time_to_first_future_reclaim_{wname}"] = None
        fields[f"future_reason_{wname}"] = "UNKNOWN_SIG_TYPE"
        if win_bars:
            min_low_val = min(b[1] for b in win_bars)
            fields[f"future_penetration_{wname}"] = max(
                0.0, (zone_lower - min_low_val) / float(atr_ref))
            _update_event_label_end(fields, win_bars[-1][0])
        _compute_aggregate_status(fields)
        return

    if status == "RECLAIMED_AT_S":
        fields[f"future_status_{wname}"] = "RECLAIMED_AT_S"
        fields[f"future_reclaim_{wname}"] = None
        fields[f"time_to_first_future_reclaim_{wname}"] = None
        fields[f"future_reason_{wname}"] = "RECLAIMED_AT_S"
        if win_bars:
            min_low_val = min(b[1] for b in win_bars)
            fields[f"future_penetration_{wname}"] = max(
                0.0, (zone_lower - min_low_val) / float(atr_ref))
            _update_event_label_end(fields, win_bars[-1][0])
        _compute_aggregate_status(fields)
        return

    if status == "ALREADY_ABOVE":
        fields[f"future_status_{wname}"] = "ALREADY_ABOVE"
        fields[f"future_reclaim_{wname}"] = None
        fields[f"time_to_first_future_reclaim_{wname}"] = None
        fields[f"future_reason_{wname}"] = "ALREADY_ABOVE"
        if win_bars:
            min_low_val = min(b[1] for b in win_bars)
            fields[f"future_penetration_{wname}"] = max(
                0.0, (zone_lower - min_low_val) / float(atr_ref))
            _update_event_label_end(fields, win_bars[-1][0])
        _compute_aggregate_status(fields)
        return

    # ---- ELIGIBLE: 主标签定义 ----
    fields[f"future_reason_{wname}"] = None
    if not win_bars:
        fields[f"future_reclaim_{wname}"] = None
        fields[f"future_penetration_{wname}"] = None
        fields[f"time_to_first_future_reclaim_{wname}"] = None
        fields[f"future_status_{wname}"] = "NO_WINDOW_BARS"
        _compute_aggregate_status(fields)
        return

    # 穿透深度 (全窗; 终点 = 窗口末 bar)
    min_low_val = min(b[1] for b in win_bars)
    pen_val = max(0.0, (zone_lower - min_low_val) / float(atr_ref))
    fields[f"future_penetration_{wname}"] = pen_val
    pen_end_ts = win_bars[-1][0]
    _update_event_label_end(fields, pen_end_ts)

    # ---- 不完整窗口 → reclaim=NULL + 失效原因 (不标 0, 不作为有效负样本) ----
    if not complete:
        fields[f"future_reclaim_{wname}"] = None
        fields[f"time_to_first_future_reclaim_{wname}"] = None
        fields[f"future_status_{wname}"] = "INCOMPLETE_WINDOW"
        fields[f"future_reason_{wname}"] = f"WINDOW_INCOMPLETE_missing={len(missing)}"
        _compute_aggregate_status(fields)
        return

    # ---- 完整窗口: 判定收回 ----
    for ts_v, lo_v, cl_v in win_bars:
        if cl_v >= zone_upper:
            fields[f"future_reclaim_{wname}"] = 1
            d_v = _day_of(ts_v)
            s_v = _slot_of(ts_v)
            try:
                tm_idx = future_slots.index((d_v, s_v))
                fields[f"time_to_first_future_reclaim_{wname}"] = int(tm_idx + 1)
            except ValueError:
                fields[f"time_to_first_future_reclaim_{wname}"] = None
            fields[f"future_status_{wname}"] = "RECLAIMED"
            # 事件整体闭环终点 = max(穿透终点, 收回终点) (统一管理)
            _update_event_label_end(fields, ts_v)
            _compute_aggregate_status(fields)
            return

    # 未收回 (完整窗口内)
    fields[f"future_reclaim_{wname}"] = 0
    fields[f"time_to_first_future_reclaim_{wname}"] = None
    fields[f"future_status_{wname}"] = "NOT_RECLAIMED"
    # pen_end_ts 已通过上方 _update_event_label_end 记录
    _compute_aggregate_status(fields)


def compute_future_hold(
    sig: dict,
    ts: np.ndarray,
    lo: np.ndarray,
    cl: np.ndarray,
    sig_idx: int,
    atr_ref: float,
    grid: list,
    signal_day: int,
    reclaim_ts: int | None,
    fields: dict,
    cal: list | None = None,
    signal_cal_idx: int | None = None,
) -> None:
    """持续性辅助标签 P-HOLD (合同 §4).

    首次未来收回后 10 交易分钟 close 均不低于 zone_upper:
      - 无收回 → 只有 60m 有效完整且明确未收回才记 0; 否则 NULL
      - 有收回但后续不足 10m → CENSORED (future_hold_10m=None)
      - 收回后 10 交易分钟 close 均 ≥ zone_upper → 1
      - 收回后 10 交易分钟内出现 close < zone_upper → 0
    label_end_ts = 首次收回时刻 + 10 交易分钟 (跨日用 cal 核验, 不为预算截短定义).

    修正 (2026-09-29 用户反例):
      1. 校验自身价格有效性 (NaN/inf → NULL+INVALID_PRICE)
      2. 继承主标签适用性: ATR 无效/RECLAIMED_AT_S/ALREADY_ABOVE → hold 也 NULL
      3. 只有有效、完整且明确未收回才记 0
      4. 分窗口 status/reason (future_status_hold/future_reason_hold)
    """
    zone_upper = float(sig["zone_upper"])
    signal_ts = int(sig["signal_ts"])

    # ---- 继承主标签适用性 ----
    # ATR 无效 → 持续性也 NULL (不能仅凭槽位完整判失败)
    atr_valid = atr_ref is not None and float(atr_ref) > 0 and np.isfinite(float(atr_ref))
    if not atr_valid:
        fields["future_hold_10m"] = None
        fields["future_hold_label_end_ts"] = None
        fields["future_status_hold"] = "INELIGIBLE"
        fields["future_reason_hold"] = "INVALID_ATR_ref<=0_or_nan"
        _compute_aggregate_status(fields)
        return

    # 主标签为 RECLAIMED_AT_S/ALREADY_ABOVE/INELIGIBLE → hold 也 NULL (不适用)
    s60 = fields.get("future_status_60m")
    if s60 in ("RECLAIMED_AT_S", "ALREADY_ABOVE", "INELIGIBLE"):
        fields["future_hold_10m"] = None
        fields["future_hold_label_end_ts"] = None
        fields["future_status_hold"] = s60
        fields["future_reason_hold"] = f"INHERITED_FROM_60m:{s60}"
        _compute_aggregate_status(fields)
        return

    if reclaim_ts is None:
        # 无收回: 只有 60m 有效完整且明确未收回 (NOT_RECLAIMED) 才记 0
        if s60 == "NOT_RECLAIMED" and fields.get("future_window_complete_60m"):
            fields["future_hold_10m"] = 0
            fields["future_hold_label_end_ts"] = fields.get("future_label_end_ts")
            fields["future_status_hold"] = "NOT_RECLAIMED_HOLD"
            fields["future_reason_hold"] = None
        else:
            fields["future_hold_10m"] = None  # CENSORED (60m 不完整或未判定)
            fields["future_hold_label_end_ts"] = None
            fields["future_status_hold"] = "CENSORED"
            fields["future_reason_hold"] = f"NO_RECLAIM_60m_status={s60}"
        _compute_aggregate_status(fields)
        return

    # 从 reclaim_ts 之后 10 个交易分钟 (跨日用 cal)
    reclaim_slot = _slot_of(reclaim_ts)
    reclaim_day = _day_of(reclaim_ts)
    hold_slots = _build_future_window_slots(reclaim_slot, reclaim_day, 10, grid, cal)
    if len(hold_slots) < 10:
        fields["future_hold_10m"] = None  # CENSORED (跨日数据不足)
        fields["future_hold_label_end_ts"] = None
        fields["future_status_hold"] = "CENSORED"
        fields["future_reason_hold"] = "HOLD_SLOTS_INSUFFICIENT"
        _compute_aggregate_status(fields)
        return

    # 按预期完整时间戳逐槽核验 (不只看行数); 冲突重复显式拒绝
    exp_pairs_list = hold_slots  # 保序
    exp_pairs_set = set(hold_slots)
    hold_bars_by_pair = {}
    for i in range(sig_idx + 1, len(ts)):
        t = int(ts[i])
        if t <= reclaim_ts:
            continue
        d = _day_of(t)
        s = _slot_of(t)
        if (d, s) in exp_pairs_set:
            if (d, s) in hold_bars_by_pair:
                # 冲突重复 → CENSORED
                fields["future_hold_10m"] = None
                fields["future_hold_label_end_ts"] = None
                fields["future_status_hold"] = "INELIGIBLE"
                fields["future_reason_hold"] = f"DUPLICATE_SLOT hold_pair={(d, s)}"
                _compute_aggregate_status(fields)
                return
            lo_v = float(lo[i])
            cl_v = float(cl[i])
            # 无效价格 (NaN/inf) → NULL+原因 (不洗成正常标签)
            if not (np.isfinite(lo_v) and np.isfinite(cl_v)):
                fields["future_hold_10m"] = None
                fields["future_hold_label_end_ts"] = None
                fields["future_status_hold"] = "INELIGIBLE"
                fields["future_reason_hold"] = f"INVALID_PRICE hold_pair={(d, s)}"
                _compute_aggregate_status(fields)
                return
            hold_bars_by_pair[(d, s)] = (t, lo_v, cl_v)

    # 逐槽核验
    ordered_bars = []
    missing_hold = []
    for pair in exp_pairs_list:
        if pair in hold_bars_by_pair:
            ordered_bars.append(hold_bars_by_pair[pair])
        else:
            missing_hold.append(pair)

    if missing_hold:
        fields["future_hold_10m"] = None  # CENSORED (缺槽)
        fields["future_hold_label_end_ts"] = None
        fields["future_status_hold"] = "CENSORED"
        fields["future_reason_hold"] = f"HOLD_INCOMPLETE_missing={len(missing_hold)}"
        _compute_aggregate_status(fields)
        return

    # 检查 10 个 bar 的 close 是否均 ≥ zone_upper (按预期顺序)
    for ts_v, lo_v, cl_v in ordered_bars:
        if cl_v < zone_upper:
            fields["future_hold_10m"] = 0
            fields["future_hold_label_end_ts"] = ts_v
            fields["future_status_hold"] = "HOLD_BROKEN"
            fields["future_reason_hold"] = None
            _update_event_label_end(fields, ts_v)
            _compute_aggregate_status(fields)
            return

    fields["future_hold_10m"] = 1
    hold_end_ts = ordered_bars[-1][0]
    fields["future_hold_label_end_ts"] = hold_end_ts
    fields["future_status_hold"] = "HOLD_OK"
    fields["future_reason_hold"] = None
    _update_event_label_end(fields, hold_end_ts)
    _compute_aggregate_status(fields)


def _update_event_label_end(fields: dict, new_end: int | None) -> None:
    """更新事件整体终点 = max(穿透终点, 收回终点, 持续性终点).

    不受函数调用顺序影响: 始终取 fields 中已记录终点与 new_end 的最大值.
    持续性终点 (future_hold_label_end_ts) 是依赖数据的最晚可能终点.
    """
    cur = fields.get("future_label_end_ts")
    if new_end is None:
        return
    if cur is None or new_end > cur:
        fields["future_label_end_ts"] = new_end


def _compute_aggregate_status(fields: dict) -> None:
    """从分窗口状态按固定规则生成综合状态 (不受调用顺序影响).

    规则 (优先级从高到低):
      1. 任一窗口 INELIGIBLE (数据无效/重复/未知类型) → 综合INELIGIBLE
      2. 任一窗口 RECLAIMED_AT_S → 综合RECLAIMED_AT_S
      3. 任一窗口 ALREADY_ABOVE → 综合ALREADY_ABOVE
      4. 任一窗口 INCOMPLETE_WINDOW/NO_WINDOW_BARS → 综合INCOMPLETE_WINDOW
      5. 主标签 (60m) RECLAIMED → 综合RECLAIMED
      6. 主标签 (60m) NOT_RECLAIMED → 综合NOT_RECLAIMED
      7. 全部 None → 综合None (未计算)
    综合原因 = 第一个非 None 的分窗口原因 (按 30m/60m/hold 顺序).
    """
    s30 = fields.get("future_status_30m")
    s60 = fields.get("future_status_60m")
    sh = fields.get("future_status_hold")
    statuses = [s for s in (s30, s60, sh) if s is not None]

    if not statuses:
        fields["future_reclaim_status"] = None
        fields["future_ineligible_reason"] = None
        return

    # 优先级判定
    if any(s == "INELIGIBLE" for s in statuses):
        agg = "INELIGIBLE"
    elif any(s == "RECLAIMED_AT_S" for s in statuses):
        agg = "RECLAIMED_AT_S"
    elif any(s == "ALREADY_ABOVE" for s in statuses):
        agg = "ALREADY_ABOVE"
    elif any(s in ("INCOMPLETE_WINDOW", "NO_WINDOW_BARS") for s in statuses):
        agg = "INCOMPLETE_WINDOW"
    elif s60 == "RECLAIMED":
        agg = "RECLAIMED"
    elif s60 == "NOT_RECLAIMED":
        agg = "NOT_RECLAIMED"
    else:
        agg = statuses[0]

    fields["future_reclaim_status"] = agg
    # 综合原因 = 第一个非 None 的分窗口原因
    for r in (fields.get("future_reason_30m"),
              fields.get("future_reason_60m"),
              fields.get("future_reason_hold")):
        if r is not None:
            fields["future_ineligible_reason"] = r
            return
    fields["future_ineligible_reason"] = None


def init_car_label_fields() -> dict:
    """初始化 CAR 标签字段."""
    f = {k: None for k in CAR_LABEL_FIELDS}
    f.update({k: False for k in (
        "future_window_complete_30m", "future_window_complete_60m",
        "future_window_truncated_30m", "future_window_truncated_60m",
    )})
    return f


# ---- 事件字典 (合同 §4, 用户要求) ----
EVENT_DICTIONARY = {
    "future_reclaim_status": {
        "ELIGIBLE": "BREAK/ALREADY_BELOW 且 S 时 close < zone_upper; 进入主标签分母",
        "RECLAIMED_AT_S": "S 时 close ≥ zone_upper; 不进主标签分母, 单独报告",
        "ALREADY_ABOVE": "APPROACH 从未跌破; 不进主标签分母",
        "RECLAIMED": "ELIGIBLE 事件在 (S, S+wlen] 内首次 close ≥ zone_upper",
        "NOT_RECLAIMED": "ELIGIBLE 事件完整窗口内未收回; time_to_first=NULL",
        "INCOMPLETE_WINDOW": "窗口不完整 (缺槽/截断); reclaim=NULL, 不作为有效负样本",
        "NO_WINDOW_BARS": "窗口内无 bar; 主标签 NULL",
        "INELIGIBLE": "未知 signal_type 或信号槽不在 grid; 主标签 NULL",
    },
    "future_ineligible_reason": {
        "RECLAIMED_AT_S": "S 时已收回, 不进主标签分母",
        "ALREADY_ABOVE": "APPROACH 从未跌破, 不进主标签分母",
        "UNKNOWN_SIG_TYPE": "未知 signal_type, 显式拒绝",
        "SIGNAL_SLOT_NOT_IN_GRID": "信号槽位不在期望网格",
        "WINDOW_INCOMPLETE_missing=N": "窗口缺 N 个槽位; reclaim=NULL, 不标 0",
        None: "ELIGIBLE, 进入主标签分母",
    },
    "future_reclaim_30m/60m": {
        "1": "RECLAIMED: (S, S+wlen] 内首次 close ≥ zone_upper",
        "0": "NOT_RECLAIMED: 完整窗口未收回",
        "None": "不适用 (RECLAIMED_AT_S / ALREADY_ABOVE / INCOMPLETE_WINDOW / NO_WINDOW_BARS / INELIGIBLE)",
    },
    "time_to_first_future_reclaim_30m/60m": {
        "int": "首次收回的交易分钟序号 (相对 future_slots)",
        "None": "未收回或不适用; 不用窗长+1 求均值",
    },
    "future_penetration_30m/60m": {
        "float": "max(0, (zone_lower - min_low)/atr_ref); 全窗最大穿透",
        "None": "无窗口 bar",
    },
    "future_hold_10m": {
        "1": "首次收回后 10 交易分钟 close 均 ≥ zone_upper",
        "0": "无收回 (60m 完整时) 或收回后 10m 内出现 close < zone_upper",
        "None": "CENSORED: 收回但后续不足 10m, 或 60m 不完整",
    },
    "future_hold_label_end_ts": {
        "int": "持续性标签观察终点 = 首次收回时刻 + 10 交易分钟; 跨日用 cal 核验",
        "None": "无收回或 CENSORED",
    },
    "future_window_complete_30m/60m": {
        "True": "窗口完整 (所有期望槽位都有 bar; 跨日则两日都齐)",
        "False": "窗口截断或缺槽 (源缺/尾部超出 grid/跨日数据不足)",
    },
    "future_label_end_ts": {
        "int": "事件整体闭环终点 = max(穿透终点, 收回终点); 依赖数据的最晚终点",
        "None": "无窗口 bar",
    },
    "5d/10d": {
        "本轮真实回放": "不做; window_complete 一律 False (未抓取)",
        "合成测试": "保留; 真实数据诊断后置",
    },
}
