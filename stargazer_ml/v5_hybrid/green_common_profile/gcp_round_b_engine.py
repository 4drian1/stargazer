"""GREEN-COMMON-PROFILE-v0.1 Round B engine (REV3 口径实现 + 2026-09-29 四项缺陷修复).

实现 gcp_v0_1_round_b_spec.py REV3 的:
  - 区域构造: MA20/BOLL下轨/前日低点 @P, 0.35*ATR_ref 相邻链式合并 (裁决1(a))
  - 目标区域 = 武装区域 (zone_upper <= close_P) 中 zone_upper 最高者 (最近支撑);
    高于前收的区域不参选 —— 消除 '最高区域远高于价格时误判 ALREADY_BELOW' 的歧义
  - 1 分钟 bar 网格校验 (缺失/重复槽, 不插值);
    13:00 槽为源格式差异: <20260624 期望 241 槽(无13:00), >=20260624 期望 242 槽
    (离线探针 2026-09-29 核实: 无同日股间不一致, 切换点 20260622→20260624)
  - 事件状态机: APPROACH/BREAK/ALREADY_BELOW, 每日一信号, AM/PM 会话
  - 统一假设执行标签 (独立于放行决策, 硬性2): 下一合法可执行分钟 open 买入,
    第 3 个【交易所】交易日收盘卖出 (停牌/跌停显式顺延, 上限10交易所日),
    成本 1.0013/0.9982; NO_FILL/SOURCE_MISSING/ADJ_UNKNOWN 分计不填零
  - 实际成交数量: 参与率=执行分钟量(股)×1% 与 5万元上限取小, 整手(100股)向下取整;
    零可成交数量 → NO_FILL_ZERO_QTY, 账户按实际金额记账 (缺陷修复3)
  - 复权链: 请求失败 adj_ok=False → ADJ_UNKNOWN 不生成有效标签 (缺陷修复2);
    确认空链 = 无除权; 因子按日逐段作用于退出价与持有期 low (缺陷修复2b)
  - 条件单元格估计 (硬性1): (setup, zone_sources, session) 匹配, 不足即回退不弃条件;
    历史事件 setup 由【事件发生时】的池资格决定, 非入选时 setup 回填 (缺陷修复1, 驱动层)
  - 决策函数: enter = signal AND mu_hat > 0 AND p_fail_hat <= 0.50 (裁决2, 严格 >)
仅工程验收用途; 合成数据不进任何绩效统计。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ---- 冻结常量 (REV3) ----
MONITOR_START, MONITOR_END = 935, 1455          # 监控窗 [09:35, 14:55]
BUY_MULT, SELL_MULT = 1.0013, 0.9982            # INTRADAY_SD_COST_ASSUMPTIONS_V1
ZONE_MERGE_ATR = 0.35                           # 链式合并半径
APPROACH_ATR = 0.20                             # 接近带宽
RECLAIM_ATR = 0.5                               # 收复/失败判定阈值
LAMBDA, N_MIN = 20, 10                          # 收缩/最小样本 (PROPOSAL_NOT_PROVEN)
TAU_MU, TAU_P = 0.0, 0.50                       # FIXED_RESEARCH_HYPOTHESIS, mu 严格 >
LOOKBACK_DEFAULT = 250                          # 研究回看 (真实抓取另受 60 日封顶)
LIMIT_UP_FACTOR = 1.10                          # 涨跌停近似 (RESEARCH_APPROXIMATION)
LIMIT_DOWN_FACTOR = 0.905                       # 跌停近似 (RESEARCH_APPROXIMATION)
EXIT_HOLD_DAYS = 3                              # H3: 第 3 个交易所交易日收盘卖出
EXIT_DEFER_MAX = 10                             # 停牌/跌停顺延上限 (交易所日)
CAPITAL_CAP = 50000.0                           # 单笔资金上限 (元)
PART_VOL_FRAC = 0.01                            # 参与率: 执行分钟量(股)×1% (研究近似)
GRID_1300_FROM = 20260624                       # 13:00 槽期望起始日 (源格式切换, 非行情缺失)


def hhmm(ts: int) -> int:
    """14 位 ts YYYYMMDDHHMMSS -> HHMM."""
    s = str(int(ts))
    return int(s[8:10]) * 100 + int(s[10:12])


def is_legal_fill(h: int) -> bool:
    return (931 <= h <= 1130) or (1301 <= h <= 1500)   # EXECUTION_CONTRACT_V1


# ================= 网格校验 (13:00 槽 = 源格式差异, 非行情缺失) =================
def expected_grid(grid: list, day: int | None) -> list:
    """按日返回期望网格: day < 20260624 → 241 槽 (无 13:00); 否则并集 242 槽."""
    if day is not None and int(day) < GRID_1300_FROM:
        return [s for s in grid if s != 130000]
    return list(grid)


def derive_grid(minute_df: pd.DataFrame) -> list:
    """从干净参考日枚举预期槽位 (HHMMSS 升序), 验收 A 冻结用."""
    return sorted({int(str(int(t))[-6:]) for t in minute_df["date"].astype(int)})


def validate_minute_day(minute_df: pd.DataFrame, grid: list, day: int | None = None) -> dict:
    """缺失>2 槽或重复/错位 -> invalid; 不插值. day 感知 13:00 源格式."""
    exp = expected_grid(grid, day)
    if minute_df is None or len(minute_df) == 0:
        return {"ok": False, "reason": "SOURCE_MISSING", "missing": list(exp), "dups": []}
    slots = [int(str(int(t))[-6:]) for t in minute_df["date"].astype(int)]
    dups = sorted({s for s in slots if slots.count(s) > 1})
    missing = sorted(set(exp) - set(slots))
    unexpected = sorted(set(slots) - set(exp))
    ok = (len(missing) <= 2) and not dups and not unexpected
    reason = ("OK" if ok else
              f"INVALID missing={len(missing)} dups={len(dups)} unexpected={len(unexpected)}")
    return {"ok": ok, "reason": reason, "missing": missing, "dups": dups,
            "unexpected": unexpected, "expected_slots": len(exp)}


def reclaim30_invalid(signal_ts: int, minute_df: pd.DataFrame, grid: list) -> bool:
    """reclaim30 窗 (信号后 30 交易分钟) 覆盖缺失槽 -> 特征 INVALID (不插值).
    注意: 日本身可 valid (缺失<=2), 但特征窗覆盖缺失槽仍须 INVALID.
    REV2-D 后: 由 window_complete_30m/window_truncated_30m 替代, 本字段保留向后兼容."""
    day = int(str(int(signal_ts))[:8])
    v = validate_minute_day(minute_df, grid, day)
    if not v["missing"]:
        return False
    sig_slot = int(str(int(signal_ts))[-6:])
    exp = expected_grid(grid, day)
    grid_idx = {s: i for i, s in enumerate(exp)}
    if sig_slot not in grid_idx:
        return True
    win = set(exp[grid_idx[sig_slot] + 1: grid_idx[sig_slot] + 31])
    return bool(win & set(v["missing"]))


# ================= 行为画像字段 (REV2-D: 不依赖成交) =================
def _minute_slot(t: int) -> int:
    """14 位 ts YYYYMMDDHHMMSS -> HHMMSS 槽位 (6 位)."""
    s = str(int(t))
    return int(s[-6:]) if len(s) >= 6 else int(s)


def _init_behavior_fields() -> dict:
    """初始化 4 个窗口的行为画像字段 (默认 None / False)."""
    fields = {k: None for k in (
        "min_low_30m", "min_low_60m", "min_low_5d", "min_low_10d",
        "pen_full_window_30m", "pen_full_window_60m",
        "pen_full_window_5d", "pen_full_window_10d",
        "time_to_max_pen_30m", "time_to_max_pen_60m",
        "time_to_max_pen_5d", "time_to_max_pen_10d",
        "reclaim_30m", "reclaim_60m", "reclaim_5d", "reclaim_10d",
        "time_to_first_reclaim_30m", "time_to_first_reclaim_60m",
        "time_to_first_reclaim_5d", "time_to_first_reclaim_10d",
    )}
    fields.update({k: False for k in (
        "window_complete_30m", "window_complete_60m",
        "window_complete_5d", "window_complete_10d",
        "window_truncated_30m", "window_truncated_60m",
        "window_truncated_5d", "window_truncated_10d",
    )})
    return fields


def _compute_minute_window(sig: dict, ts: np.ndarray, lo: np.ndarray, cl: np.ndarray,
                           sig_idx: int, atr_ref: float, grid: list, signal_day: int,
                           wname: str, wlen: int, fields: dict) -> None:
    """计算分钟窗 (30m/60m) 的穿透/收回/完整性字段. 协议 §2-§4.

    窗口 = 信号槽位后第 1..wlen 个交易分钟槽位 (按 expected_grid 感知 13:00).
    穿透 = max(0, (zone_lower - min_low) / atr_ref); time_to_max_pen 用交易分钟数.
    收回 = 先跌破 zone_lower 后 close≥zone_upper 才算 RECLAIMED; 用 close 不用 high.
    """
    zone_upper = float(sig["zone_upper"])
    zone_lower = float(sig["zone_lower"])
    signal_ts = int(sig["signal_ts"])
    sig_slot = _minute_slot(signal_ts)

    exp = expected_grid(grid, signal_day)
    exp_idx = {s: i for i, s in enumerate(exp)}
    if sig_slot not in exp_idx:
        return
    sig_gi = exp_idx[sig_slot]
    end_gi = sig_gi + wlen
    win_slots = set(exp[gi] for gi in range(sig_gi + 1, min(end_gi + 1, len(exp))))

    # 窗口内 bars (df 中 ts > signal_ts 且 槽位在 win_slots)
    win_bars = []
    for i in range(sig_idx + 1, len(ts)):
        slot = _minute_slot(int(ts[i]))
        if slot in win_slots:
            win_bars.append((int(ts[i]), float(lo[i]), float(cl[i])))

    # 完整性 (各窗独立, 60m 不否定 30m)
    win_ts_slots = set(_minute_slot(t) for t, _, _ in win_bars)
    missing_in_win = win_slots - win_ts_slots
    fields[f"window_complete_{wname}"] = bool((end_gi < len(exp))
                                              and not missing_in_win)
    fields[f"window_truncated_{wname}"] = bool((end_gi >= len(exp))
                                                or bool(missing_in_win))
    if not win_bars:
        return

    # 最小 low + 到达时刻 (交易分钟数)
    min_low_val = min(b[1] for b in win_bars)
    min_low_ts = next(b[0] for b in win_bars if b[1] == min_low_val)
    fields[f"min_low_{wname}"] = float(min_low_val)
    fields[f"pen_full_window_{wname}"] = max(0.0, (zone_lower - min_low_val) / atr_ref)
    min_low_slot = _minute_slot(min_low_ts)
    if min_low_slot in exp_idx:
        fields[f"time_to_max_pen_{wname}"] = int(exp_idx[min_low_slot] - sig_gi)

    # 收回分类: 同 bar AMBIGUOUS → RECLAIMED/NOT_RECLAIMED/ALREADY_ABOVE
    signal_bar_low = float(lo[sig_idx])
    signal_bar_close = float(cl[sig_idx])
    if signal_bar_low <= zone_lower and signal_bar_close >= zone_upper:
        fields[f"reclaim_{wname}"] = "AMBIGUOUS"
        return

    sig_type = sig.get("signal_type")
    signal_broke = sig_type in ("BREAK", "ALREADY_BELOW")
    has_broken = signal_broke
    for ts_v, lo_v, cl_v in win_bars:
        if lo_v <= zone_lower:
            has_broken = True
        if (signal_broke or has_broken) and cl_v >= zone_upper:
            slot = _minute_slot(ts_v)
            fields[f"reclaim_{wname}"] = "RECLAIMED"
            if slot in exp_idx:
                fields[f"time_to_first_reclaim_{wname}"] = int(exp_idx[slot] - sig_gi)
            return
    if signal_broke or has_broken:
        fields[f"reclaim_{wname}"] = "NOT_RECLAIMED"
    else:
        fields[f"reclaim_{wname}"] = "ALREADY_ABOVE"


def _compute_daily_window(sig: dict, daily_df: pd.DataFrame, atr_ref: float,
                          cal: list | None, signal_day: int, sig_idx: int,
                          ts: np.ndarray, lo: np.ndarray, cl: np.ndarray,
                          wname: str, wlen: int, fields: dict) -> None:
    """计算日线窗 (5d/10d) 的穿透/收回/完整性字段. 协议 §2-§4.

    窗口 = 信号日 + 信号后第 1..wlen 个交易所日 (cal).
    穿透 = min(信号日剩余分钟 low, 后续交易日 daily low); 收回用 close (信号日剩余分钟 close
    + 后续交易日 daily close); 停牌日 (dl=None) 不参与但计 truncated.
    """
    zone_upper = float(sig["zone_upper"])
    zone_lower = float(sig["zone_lower"])

    if cal is None or daily_df is None:
        return
    ci = {d: i for i, d in enumerate(cal)}
    if signal_day not in ci:
        return
    sig_ci = ci[signal_day]

    ddates = daily_df["date"].astype(int).to_numpy()
    dlow = daily_df["low"].astype(float).to_numpy()
    dclose = daily_df["close"].astype(float).to_numpy()
    dmap = {int(dd): i for i, dd in enumerate(ddates)}

    end_ci = sig_ci + wlen
    win_days = [cal[i] for i in range(sig_ci + 1, min(end_ci + 1, len(cal)))]
    win_daily = []
    for d in win_days:
        if d in dmap:
            di = dmap[d]
            win_daily.append((d, float(dlow[di]), float(dclose[di])))
        else:
            win_daily.append((d, None, None))  # 停牌

    # 信号日剩余分钟 (df 中 signal_day 之后)
    sig_day_after = []
    for i in range(sig_idx + 1, len(ts)):
        if int(str(int(ts[i]))[:8]) == signal_day:
            sig_day_after.append((int(ts[i]), float(lo[i]), float(cl[i])))

    # 完整性
    fields[f"window_complete_{wname}"] = bool((end_ci < len(cal))
                                              and all(dl is not None for _, dl, _ in win_daily))
    fields[f"window_truncated_{wname}"] = bool((end_ci >= len(cal))
                                                or any(dl is None for _, dl, _ in win_daily))

    # 综合 min_low
    lows = [lo_v for _, lo_v, _ in sig_day_after]
    lows.extend(dl for _, dl, _ in win_daily if dl is not None)
    if not lows:
        return
    min_low_val = min(lows)
    fields[f"min_low_{wname}"] = float(min_low_val)
    fields[f"pen_full_window_{wname}"] = max(0.0, (zone_lower - min_low_val) / atr_ref)
    # time_to_max_pen: 0=信号日, 1..N=后续第 k 个交易所日
    if any(lo_v == min_low_val for _, lo_v, _ in sig_day_after):
        fields[f"time_to_max_pen_{wname}"] = 0
    else:
        for k, (d, dl, dc) in enumerate(win_daily):
            if dl is not None and dl == min_low_val:
                fields[f"time_to_max_pen_{wname}"] = k + 1
                break

    # 收回分类: 同 bar AMBIGUOUS → RECLAIMED/NOT_RECLAIMED/ALREADY_ABOVE
    signal_bar_low = float(lo[sig_idx])
    signal_bar_close = float(cl[sig_idx])
    if signal_bar_low <= zone_lower and signal_bar_close >= zone_upper:
        fields[f"reclaim_{wname}"] = "AMBIGUOUS"
        return

    sig_type = sig.get("signal_type")
    signal_broke = sig_type in ("BREAK", "ALREADY_BELOW") or signal_bar_low <= zone_lower
    has_broken = signal_broke
    # 信号日剩余分钟
    for ts_v, lo_v, cl_v in sig_day_after:
        if lo_v <= zone_lower:
            has_broken = True
        if (signal_broke or has_broken) and cl_v >= zone_upper:
            fields[f"reclaim_{wname}"] = "RECLAIMED"
            fields[f"time_to_first_reclaim_{wname}"] = 0
            return
    # 后续交易日 daily close
    for d, dl, dc in win_daily:
        if dl is None:
            continue
        if dl <= zone_lower:
            has_broken = True
        if (signal_broke or has_broken) and dc is not None and dc >= zone_upper:
            fields[f"reclaim_{wname}"] = "RECLAIMED"
            fields[f"time_to_first_reclaim_{wname}"] = ci[d] - sig_ci
            return
    if signal_broke or has_broken:
        fields[f"reclaim_{wname}"] = "NOT_RECLAIMED"
    else:
        fields[f"reclaim_{wname}"] = "ALREADY_ABOVE"


def _compute_behavior_window_fields(sig: dict, minute_df_E: pd.DataFrame,
                                    daily_df: pd.DataFrame, atr_ref: float,
                                    cal: list | None, grid: list | None) -> dict:
    """计算 4 个窗口 (30m/60m/5d/10d) 的行为画像字段, 不依赖成交. 协议 §3-§4.

    入选门槛 (协议 §1.1): SOURCE_MISSING / INVALID_DAY / ADJ_UNKNOWN / HALTED_UP_APPROX 不入选
    (调用方在进入此函数前已过滤); NO_FILL_ZERO_QTY / NO_FILL_UNLABELED / EXIT_* / FILLED 入选.

    事后画像标签, 不进决策函数; 各窗独立完整性, 60m 不否定 30m.
    """
    fields = _init_behavior_fields()
    if minute_df_E is None or len(minute_df_E) == 0:
        return fields
    if not np.isfinite(atr_ref) or atr_ref <= 0:
        return fields

    df = minute_df_E.sort_values("date").reset_index(drop=True)
    ts = df["date"].astype(int).to_numpy()
    lo = df["low"].astype(float).to_numpy()
    cl = df["close"].astype(float).to_numpy()

    signal_ts = int(sig["signal_ts"])
    signal_day = int(str(signal_ts)[:8])
    sig_idx_arr = np.where(ts == signal_ts)[0]
    if len(sig_idx_arr) == 0:
        return fields
    sig_idx = int(sig_idx_arr[0])

    if grid is not None:
        for wname, wlen in (("30m", 30), ("60m", 60)):
            _compute_minute_window(sig, ts, lo, cl, sig_idx, atr_ref, grid,
                                   signal_day, wname, wlen, fields)
    for wname, wlen in (("5d", 5), ("10d", 10)):
        _compute_daily_window(sig, daily_df, atr_ref, cal, signal_day, sig_idx,
                              ts, lo, cl, wname, wlen, fields)
    return fields


# ================= 区域构造 (P 完成值) =================
def build_atr_ref(daily_upto_P: pd.DataFrame) -> float:
    """ATR_ref = median(ATR14[E-5..E-1]) = 最近 5 根 (至 P) ATR14 的中位数."""
    h, l, c = (daily_upto_P["high"].astype(float).to_numpy(),
               daily_upto_P["low"].astype(float).to_numpy(),
               daily_upto_P["close"].astype(float).to_numpy())
    if len(c) < 20:
        return float("nan")
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr14 = pd.Series(tr).rolling(14).mean().to_numpy()
    return float(np.nanmedian(atr14[-5:]))


def build_zones(daily_upto_P: pd.DataFrame, atr_ref: float) -> list:
    """锚点: MA20 / BOLL_L / PREV_LOW(=low(P)), 0.35*ATR_ref 相邻链式合并."""
    c = daily_upto_P["close"].astype(float).to_numpy()
    if len(c) < 20 or not np.isfinite(atr_ref) or atr_ref <= 0:
        return []
    ma20 = float(np.mean(c[-20:]))
    boll_l = ma20 - 2.0 * float(np.std(c[-20:], ddof=1))
    prev_low = float(daily_upto_P["low"].astype(float).iloc[-1])
    anchors = sorted([("MA20", ma20), ("BOLL_L", boll_l), ("PREV_LOW", prev_low)],
                     key=lambda x: x[1])
    radius = ZONE_MERGE_ATR * atr_ref
    clusters, cur = [], [anchors[0]]
    for a in anchors[1:]:
        if a[1] - cur[-1][1] <= radius:
            cur.append(a)
        else:
            clusters.append(cur); cur = [a]
    clusters.append(cur)
    zones = []
    for i, cl in enumerate(clusters):
        prices = [p for _, p in cl]
        zones.append({"zone_id": i, "zone_upper": max(prices), "zone_lower": min(prices),
                      "zone_mid": (max(prices) + min(prices)) / 2.0,
                      "zone_strength": len(cl),
                      "zone_sources": tuple(n for n, _ in sorted(cl, key=lambda x: x[0])),
                      "span_atr": (max(prices) - min(prices)) / atr_ref})
    return zones


def target_zone(zones: list, ref_price: float):
    """武装区域 (zone_upper <= ref_price=close_P) 中 zone_upper 最高者; 无则 None."""
    armed = [z for z in zones if z["zone_upper"] <= ref_price]
    return max(armed, key=lambda z: z["zone_upper"]) if armed else None


# ================= 事件状态机 =================
def detect_signal(minute_df: pd.DataFrame, zones: list, atr_ref: float,
                  ref_price: float) -> dict | None:
    """监控窗 [09:35,14:55] 内, 目标区域 (最近武装支撑) 首个触发 bar; 每日一信号."""
    tz = target_zone(zones, ref_price)
    if minute_df is None or len(minute_df) == 0 or tz is None or not np.isfinite(atr_ref):
        return None
    df = minute_df.sort_values("date")
    ts = df["date"].astype(int).to_numpy()
    o = df["open"].astype(float).to_numpy()
    lo = df["low"].astype(float).to_numpy()
    cl = df["close"].astype(float).to_numpy()
    band = APPROACH_ATR * atr_ref
    first_in_window = True
    for i in range(len(df)):
        h = hhmm(ts[i])
        if not (MONITOR_START <= h <= MONITOR_END):
            continue
        if first_in_window and o[i] <= tz["zone_lower"]:
            sig_type = "ALREADY_BELOW"
        elif lo[i] <= tz["zone_lower"]:
            sig_type = "BREAK"
        elif lo[i] <= tz["zone_upper"] + band and lo[i] > tz["zone_lower"]:
            sig_type = "APPROACH"
        else:
            first_in_window = False
            continue
        return {"signal_ts": int(ts[i]), "signal_type": sig_type,
                "session": "AM" if h <= 1130 else "PM",
                "zone_sources": tz["zone_sources"], "zone_id": tz["zone_id"],
                "zone_upper": tz["zone_upper"], "zone_lower": tz["zone_lower"],
                "signal_close": float(cl[i]),  # CAR 标签 classify_event_at_s 需要 S 时 close
                "d_atr": (cl[i] - tz["zone_upper"]) / atr_ref,
                "pen_atr_bar": max(0.0, (tz["zone_lower"] - lo[i]) / atr_ref)}
    return None


# ================= 统一假设执行标签 (硬性2: 独立于放行决策) =================
def _cum_factor(chain: dict, d0: int, d1: int) -> float:
    """(d0, d1] 区间复权因子连乘; 确认空链 = 1.0."""
    f = 1.0
    for k, v in chain.items():
        if d0 < int(k) <= d1:
            f *= float(v)
    return f


def label_event(sig: dict, minute_df_E: pd.DataFrame, daily_df: pd.DataFrame,
                atr_ref: float, chain: dict | None = None, adj_ok: bool = True,
                cal: list | None = None, grid: list | None = None) -> dict:
    """全部入组事件都生成标签; NO_FILL/SOURCE_MISSING/ADJ_UNKNOWN 分计, 不填零.

    缺陷修复 (2026-09-29):
      2  adj_ok=False (复权请求失败) → ADJ_UNKNOWN, 不生成有效收益标签;
         因子按日逐段作用于退出 close 与持有期 daily low (不再事后整体重标定)
      3  实际成交数量: 量(股)×1% 与 5 万上限取小, 整手向下取整;
         零可成交数量 → NO_FILL_ZERO_QTY (账户不按 5 万满额记账)
      4  退出 = 第 3 个【交易所】交易日收盘 (cal); 个股缺 bar(停牌) 或跌停近似
         → 显式顺延 (上限 EXIT_DEFER_MAX 交易所日), 超限 EXIT_DEFERRED_FAILED

    REV2-D (行为画像字段, 不依赖成交):
      - SOURCE_MISSING / ADJ_UNKNOWN / HALTED_UP_APPROX 不入选 (无 behavior_* 字段)
      - NO_FILL_UNLABELED / NO_FILL_ZERO_QTY / EXIT_* / FILLED 入选 (调用
        _compute_behavior_window_fields 计算 4 窗穿透/收回/完整性字段)
      - 行为有效性 ≠ 执行有效性; 收益标签 y/f 仍只对 FILLED 计算
    """
    out = {"signal_ts": sig["signal_ts"], "signal_type": sig["signal_type"],
           "session": sig["session"], "zone_sources": sig["zone_sources"]}
    if minute_df_E is None or len(minute_df_E) == 0:
        return {**out, "status": "SOURCE_MISSING", "y": None, "f": None,
                "label_end_ts": None}
    if not adj_ok:
        return {**out, "status": "ADJ_UNKNOWN", "y": None, "f": None,
                "label_end_ts": None, "adj_status": "REQUEST_FAILED"}
    chain = dict(chain) if chain else {}
    df = minute_df_E.sort_values("date")
    ts = df["date"].astype(int).to_numpy()
    cand = [i for i in range(len(df))
            if ts[i] > sig["signal_ts"] and is_legal_fill(hhmm(ts[i]))]
    if not cand:
        # NO_FILL_UNLABELED: 行情完整则入选行为画像 (协议 §1.1)
        behavior = _compute_behavior_window_fields(sig, minute_df_E, daily_df,
                                                  atr_ref, cal, grid)
        return {**out, **behavior, "status": "NO_FILL_UNLABELED", "y": None, "f": None,
                "label_end_ts": None}
    fi = cand[0]
    fill_ts, fill_price = int(ts[fi]), float(df["open"].astype(float).iloc[fi])
    if float(df["high"].astype(float).iloc[fi]) >= fill_price * LIMIT_UP_FACTOR * 0.995:
        return {**out, "status": "HALTED_UP_APPROX", "y": None, "f": None,
                "label_end_ts": None, "fill_ts": fill_ts}
    # ---- 行为画像字段 (REV2-D): 行情完整 + adj_ok + 非 HALTED_UP_APPROX 入选 ----
    behavior = _compute_behavior_window_fields(sig, minute_df_E, daily_df,
                                               atr_ref, cal, grid)
    out.update(behavior)
    # ---- 实际成交数量 (缺陷修复3): 量(股)×1% 与 5 万上限, 整手向下取整 ----
    fill_vol_shares = float(df["volume"].astype(float).iloc[fi])   # 分钟量纲=股(对拍核实)
    avail_amount = min(fill_vol_shares * PART_VOL_FRAC * fill_price, CAPITAL_CAP)
    lots = math.floor(avail_amount / (fill_price * BUY_MULT * 100.0))
    fill_shares = lots * 100
    if fill_shares <= 0:
        return {**out, "status": "NO_FILL_ZERO_QTY", "y": None, "f": None,
                "label_end_ts": None, "fill_ts": fill_ts, "fill_price": fill_price,
                "fill_shares": 0, "fill_lots": 0, "fill_amount": 0.0,
                "participation_ratio": 0.0}
    buy_net = fill_price * BUY_MULT
    fill_amount = fill_shares * buy_net
    participation_ratio = fill_amount / CAPITAL_CAP
    fill_date = int(str(fill_ts)[:8])
    # ---- 退出: 第 3 个交易所交易日 (缺陷修复4) ----
    if cal is None:   # 兼容回退 (仅合成路径, 合成日线日期即交易所日历)
        cal = sorted(int(d) for d in daily_df["date"].astype(int).unique())
    ci = {d: i for i, d in enumerate(cal)}
    if fill_date not in ci:
        return {**out, "status": "EXIT_UNAVAILABLE", "y": None, "f": None,
                "label_end_ts": None, "fill_ts": fill_ts, "fill_price": fill_price,
                "fill_shares": fill_shares, "fill_amount": fill_amount,
                "participation_ratio": participation_ratio,
                "exit_reason": "FILL_DATE_NOT_IN_CAL"}
    tgt = ci[fill_date] + EXIT_HOLD_DAYS
    if tgt >= len(cal):
        return {**out, "status": "EXIT_UNAVAILABLE", "y": None, "f": None,
                "label_end_ts": None, "fill_ts": fill_ts, "fill_price": fill_price,
                "fill_shares": fill_shares, "fill_amount": fill_amount,
                "participation_ratio": participation_ratio,
                "exit_reason": "CAL_END"}
    intended_exit_date = cal[tgt]
    ddates = daily_df["date"].astype(int).to_numpy()
    dclose = daily_df["close"].astype(float).to_numpy()
    dlow = daily_df["low"].astype(float).to_numpy()
    dmap = {int(dd): i for i, dd in enumerate(ddates)}
    exit_date, exit_close, exit_deferred, exit_reason = None, None, False, None
    j = tgt
    while j < len(cal) and j <= tgt + EXIT_DEFER_MAX:
        d = cal[j]
        if d not in dmap:                      # 个股缺 bar (停牌)
            exit_deferred, exit_reason = True, "SUSPENDED"
            j += 1
            continue
        di = dmap[d]
        prev = dclose[di - 1] if di > 0 else None
        if prev is not None and dclose[di] <= prev * LIMIT_DOWN_FACTOR:   # 跌停近似
            exit_deferred, exit_reason = True, "HALTED_DOWN_APPROX"
            j += 1
            continue
        exit_date, exit_close = d, float(dclose[di])
        break
    if exit_date is None:
        return {**out, "status": "EXIT_DEFERRED_FAILED", "y": None, "f": None,
                "label_end_ts": None, "fill_ts": fill_ts, "fill_price": fill_price,
                "fill_shares": fill_shares, "fill_amount": fill_amount,
                "participation_ratio": participation_ratio,
                "intended_exit_date": intended_exit_date,
                "exit_reason": (exit_reason or "DEFER_MAX") + f"+{EXIT_DEFER_MAX}"}
    adj_cum = _cum_factor(chain, fill_date, exit_date)
    sell_net = exit_close * adj_cum * SELL_MULT
    y = sell_net / buy_net - 1.0
    # f: E 日 fill 后分钟 low (raw, 同日同尺度) + 持有期各股交易日 low × 各日累计因子
    lows_pm = df["low"].astype(float).to_numpy()[fi:]
    cand_lows = [float(x) for x in lows_pm]
    for d in [c for c in cal[ci[fill_date] + 1: ci[exit_date] + 1] if c in dmap]:
        di = dmap[d]
        cand_lows.append(float(dlow[di]) * _cum_factor(chain, fill_date, d))
    min_low = min(cand_lows) if cand_lows else buy_net
    f = 1 if min_low < buy_net - RECLAIM_ATR * atr_ref else 0
    # min_low 暴露给下游: 供可行性报告计算"完整下杀过程最大穿透"
    # (zone_lower - min_low)/atr_ref; 与 pen_atr_bar(仅信号bar) 区分.
    # 注意 min_low 在 buy_net 同尺度 (fill_date 日复权后), zone_lower 在 P 日尺度;
    # fill 发生在 P 日下一合法分钟, 两者同日同尺度, 可直接比较.
    return {**out, "status": "FILLED", "y": float(y), "f": int(f),
            "label_end_ts": exit_date, "fill_ts": fill_ts, "fill_price": fill_price,
            "fill_shares": fill_shares, "fill_lots": lots,
            "fill_amount": float(fill_amount),
            "participation_ratio": float(participation_ratio),
            "intended_exit_date": intended_exit_date, "exit_date": exit_date,
            "exit_deferred": exit_deferred, "exit_reason": exit_reason,
            "adj_cum": float(adj_cum), "min_low": float(min_low)}


# ================= 条件单元格估计 (硬性1) =================
def estimate(events: list, code: str, setup: str, zone_sources: tuple, session: str,
             E: int, date_idx: dict, lookback: int = LOOKBACK_DEFAULT) -> dict:
    """P0/P1 同单元格 (setup, zone_sources, session); 不足回退, 不弃条件.
    setup=None (事件发生时无池资格) 的事件不进任何单元格."""
    cell = [e for e in events
            if e.get("status") == "FILLED"
            and e.get("setup") is not None
            and e["setup"] == setup and e["zone_sources"] == tuple(zone_sources)
            and e["session"] == session
            and e["date"] < E and e["label_end_ts"] is not None
            and e["label_end_ts"] < E
            and E in date_idx and e["date"] in date_idx
            and date_idx[E] - date_idx[e["date"]] <= lookback]
    stock = [e for e in cell if e["code"] == code]
    n_g, n = len(cell), len(stock)
    res = {"n_g": n_g, "n": n, "cell": (setup, tuple(zone_sources), session),
           "fallback": False, "no_estimate": True, "mu_g": None, "p_g": None,
           "mu_1": None, "p_1": None}
    if n_g < N_MIN:
        return res
    mu_g = float(np.mean([e["y"] for e in cell]))
    p_g = float(np.mean([e["f"] for e in cell]))
    res.update({"no_estimate": False, "mu_g": mu_g, "p_g": p_g})
    if n >= N_MIN:
        w = n / (n + LAMBDA)
        mu_s = float(np.mean([e["y"] for e in stock]))
        p_s = float(np.mean([e["f"] for e in stock]))
        res.update({"mu_stock": mu_s, "p_stock": p_s,
                    "mu_1": w * mu_s + (1 - w) * mu_g,
                    "p_1": w * p_s + (1 - w) * p_g})
    else:  # 回退: 不取消条件凑样本
        res.update({"fallback": True, "mu_1": mu_g, "p_1": p_g})
    return res


def decide(mu, p) -> dict:
    """enter = mu_hat > 0 (严格) AND p_fail_hat <= 0.50; 同函数同门槛 P0/P1."""
    if mu is None or p is None:
        return {"enter": False, "reason": "NO_ESTIMATE"}
    if not mu > TAU_MU:
        return {"enter": False, "reason": "SKIPPED_LOW_MU", "mu": mu, "p": p}
    if not p <= TAU_P:
        return {"enter": False, "reason": "SKIPPED_HIGH_PFAIL", "mu": mu, "p": p}
    return {"enter": True, "reason": "ENTER", "mu": mu, "p": p}


# ================= 单日流水线 =================
def run_day(ctx: dict, E: int, grid: list) -> dict:
    """ctx: {code, setup(事件发生时池资格, 可为 None), daily(date,open,high,low,close),
    minute(E 日 1 分钟 df 或 None), date_idx, cal(交易所日历, 真实回放必传),
    chain(复权链 dict), adj_ok(bool)}. 返回信号+标签全记录 (标签独立于决策)."""
    daily = ctx["daily"]
    upto_P = daily[daily["date"].astype(int) < E].sort_values("date")
    atr = build_atr_ref(upto_P)
    zones = build_zones(upto_P, atr)
    ref_price = float(upto_P["close"].astype(float).iloc[-1]) if len(upto_P) else None
    minute_E = ctx.get("minute")
    v = validate_minute_day(minute_E, grid, E)
    rec = {"code": ctx["code"], "setup": ctx.get("setup"), "date": E,
           "zone_n": len(zones), "atr_ref": atr, "bar_valid": v["ok"],
           "bar_reason": v["reason"], "missing_slots": v["missing"],
           "zone_upper": None, "zone_lower": None, "zone_sources": None,
           "signal_ts": None, "signal_type": None, "session": None,
           "y": None, "f": None, "label_end_ts": None, "status": None}
    if not v["ok"]:
        rec["status"] = "INVALID_DAY"
        return rec
    sig = detect_signal(minute_E, zones, atr, ref_price)
    if sig is None:
        rec["status"] = "NO_SIGNAL"
        return rec
    lab = label_event(sig, minute_E, daily, atr, chain=ctx.get("chain"),
                      adj_ok=ctx.get("adj_ok", True), cal=ctx.get("cal"), grid=grid)
    rec.update(sig)
    rec.update({k: v2 for k, v2 in lab.items() if k not in sig})
    rec["status"] = lab["status"]
    rec["reclaim30_invalid"] = reclaim30_invalid(sig["signal_ts"], minute_E, grid)
    return rec


# ================= 合成数据生成 (仅工程验收, 不进绩效统计) =================
def synth_stock(seed: int, code: str, setup: str, grid: list, n_days: int = 320,
                drift: float = 0.0, scenario_days: dict | None = None) -> dict:
    """确定性合成: 日线随机游走 (可配漂移) + 指定日受控 1 分钟路径.
    scenario_days: {距末日偏移: 'break_am'/'break_pm'/'approach_am'/'approach_pm'/
                    'already_below'/'no_signal'/'missing_slots'}; grid=冻结的真实槽位."""
    rng = np.random.default_rng(seed)
    dates = [int(d.strftime("%Y%m%d")) for d in
             pd.bdate_range("2025-01-02", periods=n_days)]
    ret = rng.normal(drift, 0.012, n_days)
    close = 10.0 * np.cumprod(1 + ret)
    daily = pd.DataFrame({"date": dates,
                          "open": close * (1 + rng.normal(0, 0.004, n_days)),
                          "high": close * (1 + np.abs(rng.normal(0, 0.008, n_days))),
                          "low": close * (1 - np.abs(rng.normal(0, 0.008, n_days))),
                          "close": close})
    minute, scen = {}, scenario_days or {}
    for off, sc in sorted(scen.items()):
        di = n_days - 1 - off
        if di < 25:
            continue
        E = dates[di]
        upto_P = daily.iloc[:di]
        atr = build_atr_ref(upto_P)
        zones = build_zones(upto_P, atr)
        tz = target_zone(zones, float(daily["close"].iloc[di - 1]))
        if tz is None or not np.isfinite(atr):
            continue
        minute[E] = _synth_minute_day(rng, E, float(daily["close"].iloc[di - 1]),
                                      tz, atr, sc, grid)
    return {"code": code, "setup": setup, "daily": daily, "minute": minute,
            "dates": dates}


def _synth_minute_day(rng, date: int, prev_close: float, tz: dict, atr: float,
                      scenario: str, grid: list) -> pd.DataFrame:
    """受控触达目标区域的 1 分钟路径; 槽位 = 该日期的期望网格 (13:00 格式感知)."""
    exp = expected_grid(grid, date)
    n = len(exp)
    ts = [int(f"{date}{s:06d}") if isinstance(s, int) else int(f"{date}{s}") for s in exp]
    target_lo = {"break_am": tz["zone_lower"] - 0.10 * atr,
                 "break_pm": tz["zone_lower"] - 0.15 * atr,
                 "approach_am": tz["zone_upper"] + 0.10 * atr,
                 "approach_pm": tz["zone_upper"] + 0.05 * atr,
                 "already_below": None,
                 "no_signal": prev_close * 0.998,
                 "missing_slots": tz["zone_lower"] - 0.10 * atr}.get(scenario)
    gap = ((tz["zone_lower"] - 0.2 * atr) / prev_close
           if scenario == "already_below" else 1.0)
    hit_i = {"break_am": 30, "break_pm": 135, "approach_am": 30, "approach_pm": 135,
             "already_below": 0, "no_signal": -1, "missing_slots": 30}.get(scenario, 30)
    p = prev_close * gap
    price = np.empty(n)
    pre_floor = tz["zone_upper"] + 0.30 * atr     # 触发前钳制: 高于接近带, 保证首触在 hit_i
    for i in range(n):
        p *= (1 + rng.normal(0, 0.0008))
        if i == hit_i and target_lo is not None:
            p = target_lo
        elif target_lo is not None and hit_i > 0 and i < hit_i:
            p = max(p, pre_floor)
        price[i] = p
    high = price * (1 + np.abs(rng.normal(0, 0.0006, n)))
    low = price * (1 - np.abs(rng.normal(0, 0.0009, n)))
    if hit_i >= 0 and target_lo is not None:
        low[hit_i] = min(low[hit_i], target_lo)
    if hit_i > 0 and target_lo is not None:
        low_floor = tz["zone_upper"] + 0.25 * atr    # 触发前 low 钳制 (带 0.20ATR 之上)
        for i in range(min(hit_i, n)):
            low[i] = max(low[i], low_floor)
    if scenario == "no_signal":
        # 保证任何 low 都高于 zone_upper + 接近带 (0.20ATR) + 余量, 不触发任何信号
        floor = tz["zone_upper"] + 0.30 * atr
        price = np.maximum(price, floor * 1.002)
        low = np.maximum(low, floor * 1.001)
        high = np.maximum(high, price)
    # 量级=股(对拍核实): 30万-80万股 → 1%参与率下 avail 3万-8万, 覆盖部分成交与满额两分支
    vol = rng.integers(3000, 8000, n).astype(float) * 100.0
    df = pd.DataFrame({"date": ts, "open": price, "high": high, "low": low,
                       "close": price, "volume": vol, "amount": vol * price})
    if scenario == "missing_slots":
        df = df.drop(index=[60, 61, 62]).reset_index(drop=True)   # 3 缺失槽 -> invalid
    return df
