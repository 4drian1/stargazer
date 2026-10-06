"""GREEN-CAR-v0.1 标签合成验收 v2 (合同 §9 必测, 2026-09-29 用户反例修正后).

本轮只做标签和因果链验收, 不训练. 成功标准: 算得对、可重放.
本脚本用合成数据测试 CAR 新标签函数, 不动旧 v0.2 函数和结果.

v2 新增/修正测试 (用户反例):
  - 不完整窗口 → reclaim=NULL (不标 0)
  - 窗口用完整时间戳 (day+slot), 追加窗口外另一日数据不改变结果
  - 跨日观察 (有下一交易日数据时继续, 不一律截断)
  - 标签终点 = 依赖数据的最晚终点 (穿透/收回/持续性)
  - 未知 signal_type → 显式拒绝
  - 跨午休 (13:00 槽位)
  - 缺槽、重复时间戳
  - "五类状态正确分流" 真正断言结果, 不再直接 True
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gcp_round_b_engine as E
import gcp_car_v0_1_labels as L

RESULTS = []
FAILS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def make_signal(zone_upper, zone_lower, signal_ts, sig_type, close_s):
    return {
        "zone_upper": zone_upper, "zone_lower": zone_lower,
        "signal_ts": signal_ts, "signal_type": sig_type,
        "signal_close": close_s,
    }


def make_grid():
    """真实交易所分钟槽位 09:30-11:30 + 13:00-15:00."""
    slots = []
    for h in range(9, 12):
        for m in range(0, 60):
            if h == 9 and m < 30:
                continue
            if h == 11 and m > 30:
                break
            slots.append(h * 10000 + m * 100)
    for h in range(13, 15):
        for m in range(0, 60):
            slots.append(h * 10000 + m * 100)
    for m in range(0, 1):
        slots.append(15 * 10000 + m * 100)
    return sorted(slots)


def make_minute_ts(signal_ts, n_future, grid, day=20260109):
    """生成 ts 数组: ts[0]=signal_ts, 之后 n_future 个未来 bar (按 grid 顺序, 同日)."""
    sig_slot = L._slot_of(signal_ts)
    exp = sorted(set(grid))
    idx = exp.index(sig_slot) if sig_slot in exp else 0
    out = [signal_ts]
    for i in range(1, n_future + 1):
        if idx + i >= len(exp):
            break
        slot = exp[idx + i]
        hh = slot // 10000
        mm = (slot // 100) % 100
        out.append(int(f"{day}{hh:02d}{mm:02d}00"))
    return np.array(out, dtype=np.int64)


def make_cal(n_days=5, start=20260109):
    """简单交易日历: start 起 n_days 个连续日 (跨周末跳过)."""
    import datetime as dt
    d0 = dt.date(int(str(start)[:4]), int(str(start)[4:6]), int(str(start)[6:8]))
    cal = []
    seen = 0
    while seen < n_days:
        if d0.weekday() < 5:  # 周一-周五
            cal.append(int(d0.strftime("%Y%m%d")))
            seen += 1
        d0 += dt.timedelta(days=1)
    return cal


# ================= T4: 同 bar 模糊 → RECLAIMED_AT_S 分流 =================
def test_4_reclaimed_at_s():
    print("\n== T4: 同 bar 模糊 → RECLAIMED_AT_S (§9.4) ==")
    sig = make_signal(10.0, 9.0, 20260109093500, "BREAK", 10.5)
    check("T4a BREAK close≥upper→RECLAIMED_AT_S",
          L.classify_event_at_s(sig) == "RECLAIMED_AT_S")
    sig2 = make_signal(10.0, 9.0, 20260109093500, "APPROACH", 10.5)
    check("T4b APPROACH close≥upper→RECLAIMED_AT_S",
          L.classify_event_at_s(sig2) == "RECLAIMED_AT_S")
    sig3 = make_signal(10.0, 9.0, 20260109093500, "BREAK", 9.5)
    check("T4c BREAK close<upper→ELIGIBLE",
          L.classify_event_at_s(sig3) == "ELIGIBLE")


# ================= T5: RECLAIMED_AT_S 不进主标签分母 =================
def test_5_not_in_denominator():
    print("\n== T5: RECLAIMED_AT_S 不进主标签分母 (§9.5) ==")
    grid = make_grid()
    sig = make_signal(10.0, 9.0, 20260109093500, "BREAK", 10.5)
    ts = make_minute_ts(20260109093500, 30, grid)
    lo = np.full(len(ts), 9.5)
    cl = np.full(len(ts), 10.5)
    f = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 30, "30m", f)
    check("T5a reclaim=None", f["future_reclaim_30m"] is None)
    check("T5b status=RECLAIMED_AT_S", f["future_reclaim_status"] == "RECLAIMED_AT_S")
    check("T5c reason=RECLAIMED_AT_S", f["future_ineligible_reason"] == "RECLAIMED_AT_S")


# ================= T6: 未收回 NULL + 观察终点 =================
def test_6_not_reclaimed_null():
    print("\n== T6: 未收回 NULL (§9.6) ==")
    grid = make_grid()
    sig = make_signal(10.0, 9.0, 20260109093500, "BREAK", 9.5)
    ts = make_minute_ts(20260109093500, 30, grid)
    lo = np.full(len(ts), 8.5)
    cl = np.full(len(ts), 9.0)
    f = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 30, "30m", f)
    check("T6a reclaim=0", f["future_reclaim_30m"] == 0)
    check("T6b time_to_first=NULL", f["time_to_first_future_reclaim_30m"] is None)
    check("T6c label_end_ts set", f["future_label_end_ts"] is not None)
    check("T6d status=NOT_RECLAIMED", f["future_reclaim_status"] == "NOT_RECLAIMED")


# ================= T7: 各窗截断独立 (30m/60m) =================
def test_7_window_independence():
    print("\n== T7: 30m/60m 完整性独立 (§9.7) ==")
    grid = make_grid()
    sig = make_signal(10.0, 9.0, 20260109093500, "BREAK", 9.5)
    ts = make_minute_ts(20260109093500, 40, grid)  # 40 未来 bar
    lo = np.full(len(ts), 9.5)
    cl = np.full(len(ts), 9.8)
    f = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 30, "30m", f)
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 60, "60m", f)
    check("T7a 30m 完整 (40≥30)", f["future_window_complete_30m"],
          f"complete={f['future_window_complete_30m']}")
    check("T7b 60m 截断 (40<60)", f["future_window_truncated_60m"])
    check("T7c 60m 截断不否定 30m",
          f["future_window_complete_30m"] and f["future_window_truncated_60m"])


# ================= T8: 内部缺分钟 → INVALID =================
def test_8_internal_gap_invalid():
    print("\n== T8: 窗口缺槽 → INVALID (§9.8) ==")
    grid = make_grid()
    sig = make_signal(10.0, 9.0, 20260109093500, "BREAK", 9.5)
    ts_full = make_minute_ts(20260109093500, 60, grid)
    ts = np.array([t for i, t in enumerate(ts_full) if i not in (10, 11, 12)],
                  dtype=np.int64)
    lo = np.full(len(ts), 9.5)
    cl = np.full(len(ts), 9.8)
    f = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 60, "60m", f)
    check("T8a 缺 3 槽: 60m 不完整", not f["future_window_complete_60m"])
    check("T8b 缺 3 槽: 60m 截断", f["future_window_truncated_60m"])


# ================= T13: 尾部截断显式标记, 不改判"未收回" =================
def test_13_tail_truncation():
    print("\n== T13: E 为窗末日, 60m 尾部截断显式标记 (§9.13) ==")
    grid = make_grid()
    sig = make_signal(10.0, 9.0, 20260109145500, "BREAK", 9.5)
    ts = make_minute_ts(20260109145500, 5, grid)  # 仅 5 未来 bar
    lo = np.full(len(ts), 9.0)
    cl = np.full(len(ts), 9.5)
    f = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 60, "60m", f)
    check("T13a 尾部截断: truncated=True", f["future_window_truncated_60m"])
    check("T13b 尾部截断: complete=False", not f["future_window_complete_60m"])
    # 关键: 不完整窗口 → reclaim=NULL (不硬塞 0, 不硬塞 1)
    check("T13c 尾部截断: reclaim=NULL (不硬塞)",
          f["future_reclaim_60m"] is None,
          f"reclaim={f['future_reclaim_60m']}")
    check("T13d status=INCOMPLETE_WINDOW",
          f["future_reclaim_status"] == "INCOMPLETE_WINDOW",
          f"status={f['future_reclaim_status']}")


# ================= T2: PIT/因果与未来扰动不变性 =================
def test_2_future_perturbation_invariance():
    print("\n== T2: 已闭环事件未来扰动不变 (§9.2) ==")
    grid = make_grid()
    sig = make_signal(10.0, 9.0, 20260109093500, "BREAK", 9.5)
    ts = make_minute_ts(20260109093500, 60, grid)
    lo = np.full(len(ts), 9.0)
    cl = np.full(len(ts), 9.8)
    cl[20] = 10.5  # 第 20 bar 收回
    f1 = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 60, "60m", f1)

    # 扰动: 在 60m 窗之后加 100 个 bar (下周一数据), 乱改值
    cal = make_cal(5)
    # 下一个交易日的 ts
    next_day = cal[1]
    ts_ext = make_minute_ts(int(f"{next_day}093500"), 100, grid, day=next_day)
    ts2 = np.concatenate([ts, ts_ext])
    lo2 = np.concatenate([lo, np.full(len(ts_ext), 7.0)])
    cl2 = np.concatenate([cl, np.full(len(ts_ext), 8.0)])
    f2 = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts2, lo2, cl2, 0, 1.0, grid, 20260109, 60, "60m", f2,
                                     cal=cal)

    check("T2a label_end_ts 不变", f1["future_label_end_ts"] == f2["future_label_end_ts"],
          f"before={f1['future_label_end_ts']} after={f2['future_label_end_ts']}")
    check("T2b reclaim 不变", f1["future_reclaim_60m"] == f2["future_reclaim_60m"])
    check("T2c penetration 不变",
          f1["future_penetration_60m"] == f2["future_penetration_60m"],
          f"before={f1['future_penetration_60m']} after={f2['future_penetration_60m']}")


# ================= T12: 非空转守卫 =================
def test_12_no_empty_pass():
    print("\n== T12: 全缺失不能退化 PASS (§9.12) ==")
    grid = make_grid()
    sig = make_signal(10.0, 9.0, 20260109093500, "BREAK", 9.5)
    ts = np.array([20260109093500], dtype=np.int64)  # 只有信号 bar
    lo = np.array([9.5])
    cl = np.array([9.5])
    f = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 60, "60m", f)
    check("T12a 全缺: reclaim≠1", f["future_reclaim_60m"] != 1)
    check("T12b 全缺: status≠RECLAIMED", f["future_reclaim_status"] != "RECLAIMED")


# ================= 新增: 不完整窗口 → reclaim=NULL (不标 0) =================
def test_incomplete_window_null_not_zero():
    print("\n== 新增: 不完整窗口 → reclaim=NULL (不标 0) ==")
    grid = make_grid()
    sig = make_signal(10.0, 9.0, 20260109093500, "BREAK", 9.5)
    # 30m 窗需要 30 未来 bar, 只给 20
    ts = make_minute_ts(20260109093500, 20, grid)
    lo = np.full(len(ts), 9.0)
    cl = np.full(len(ts), 9.8)  # 全未收回
    f = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 30, "30m", f)
    check("INCa 不完整: reclaim=NULL (不标 0)",
          f["future_reclaim_30m"] is None,
          f"reclaim={f['future_reclaim_30m']}")
    check("INCb 不完整: status=INCOMPLETE_WINDOW",
          f["future_reclaim_status"] == "INCOMPLETE_WINDOW")
    check("INCc 不完整: reason 含 WINDOW_INCOMPLETE",
          f["future_ineligible_reason"] is not None
          and "WINDOW_INCOMPLETE" in f["future_ineligible_reason"])


# ================= 新增: 窗口用完整时间戳, 追加窗口外另一日数据不改变结果 =================
def test_cross_day_no_pollution():
    print("\n== 新增: 追加窗口外另一日数据不改变结果 ==")
    grid = make_grid()
    sig = make_signal(10.0, 9.0, 20260109093500, "BREAK", 9.5)
    ts = make_minute_ts(20260109093500, 30, grid)
    lo = np.full(len(ts), 9.0)
    cl = np.full(len(ts), 9.8)  # 未收回
    f1 = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 30, "30m", f1)

    # 追加下周一同时刻数据 (在 30m 窗外)
    cal = make_cal(5)
    next_day = cal[1]
    ts_ext = make_minute_ts(int(f"{next_day}093500"), 30, grid, day=next_day)
    ts2 = np.concatenate([ts, ts_ext])
    lo2 = np.concatenate([lo, np.full(len(ts_ext), 7.0)])
    cl2 = np.concatenate([cl, np.full(len(ts_ext), 10.5)])  # 下周一收回 (但不在窗口内)
    f2 = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts2, lo2, cl2, 0, 1.0, grid, 20260109, 30, "30m", f2,
                                     cal=cal)
    check("XDAYa 追加窗外数据: reclaim 不变 (仍 0)",
          f2["future_reclaim_30m"] == 0,
          f"reclaim={f2['future_reclaim_30m']}")
    check("XDAYb 追加窗外数据: penetration 不变",
          f1["future_penetration_30m"] == f2["future_penetration_30m"])
    check("XDAYc 追加窗外数据: label_end_ts 不变",
          f1["future_label_end_ts"] == f2["future_label_end_ts"])


# ================= 新增: 跨日观察 (有下一交易日数据时继续) =================
def test_cross_day_continuation():
    print("\n== 新增: 跨日观察 (有下一交易日数据时继续) ==")
    grid = make_grid()
    cal = make_cal(5)
    # 信号在 14:55, 60m 窗跨到下一交易日
    sig = make_signal(10.0, 9.0, 20260109145500, "BREAK", 9.5)
    # 同日剩余 bar (1500-1504 = 5 bar) + 下一交易日 55 bar (够 60)
    ts_day1 = make_minute_ts(20260109145500, 5, grid, day=20260109)
    next_day = cal[1]
    ts_day2 = make_minute_ts(int(f"{next_day}093000"), 55, grid, day=next_day)
    ts = np.concatenate([ts_day1, ts_day2])
    lo = np.full(len(ts), 9.0)
    cl = np.full(len(ts), 9.8)
    cl[30] = 10.5  # 第 30 个未来 bar 收回 (跨日)
    f = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 60, "60m", f,
                                     cal=cal)
    check("CROSSa 跨日: 60m 窗完整 (有下一交易日数据)",
          f["future_window_complete_60m"],
          f"complete={f['future_window_complete_60m']}")
    check("CROSSb 跨日: reclaim=1 (收回)",
          f["future_reclaim_60m"] == 1, f"reclaim={f['future_reclaim_60m']}")


# ================= 新增: 跨午休 (13:00 槽位) =================
def test_cross_lunch():
    print("\n== 新增: 跨午休 (13:00 槽位) ==")
    grid = make_grid()
    sig = make_signal(10.0, 9.0, 20260109112500, "BREAK", 9.5)  # 11:25 信号
    # 30m 窗: 11:26-11:30 (5 bar) + 13:00-13:19 (20 bar) = 25 bar < 30 → 不完整
    ts = make_minute_ts(20260109112500, 25, grid)
    lo = np.full(len(ts), 9.0)
    cl = np.full(len(ts), 9.8)
    f = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 30, "30m", f)
    check("LUNCHa 跨午休 30m: 不完整 (25<30)",
          not f["future_window_complete_30m"])
    check("LUNCHb 跨午休 30m: reclaim=NULL",
          f["future_reclaim_30m"] is None)


# ================= 新增: 未知 signal_type 显式拒绝 =================
def test_unknown_sig_type_rejected():
    print("\n== 新增: 未知 signal_type 显式拒绝 ==")
    sig = make_signal(10.0, 9.0, 20260109093500, "UNKNOWN_TYPE", 9.5)
    check("UNKa 未知类型: classify=UNKNOWN_SIG_TYPE",
          L.classify_event_at_s(sig) == "UNKNOWN_SIG_TYPE")
    grid = make_grid()
    ts = make_minute_ts(20260109093500, 30, grid)
    lo = np.full(len(ts), 9.0)
    cl = np.full(len(ts), 9.8)
    f = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 30, "30m", f)
    check("UNKb 未知类型: status=INELIGIBLE",
          f["future_reclaim_status"] == "INELIGIBLE")
    check("UNKc 未知类型: reclaim=NULL",
          f["future_reclaim_30m"] is None)
    check("UNKd 未知类型: reason=UNKNOWN_SIG_TYPE",
          f["future_ineligible_reason"] == "UNKNOWN_SIG_TYPE")


# ================= 新增: 重复时间戳处理 (收回窗口也逐槽核验+拒绝) =================
def test_duplicate_timestamps():
    print("\n== 新增: 重复时间戳处理 (收回窗口逐槽核验) ==")
    grid = make_grid()
    sig = make_signal(10.0, 9.0, 20260109093500, "BREAK", 9.5)
    ts = make_minute_ts(20260109093500, 30, grid)
    # 插入重复 ts (第 5 个 bar 重复)
    ts_dup = np.insert(ts, 5, ts[5])
    lo = np.full(len(ts_dup), 9.0)
    cl = np.full(len(ts_dup), 9.8)
    cl[10] = 10.5  # 收回 (但重复 ts 会导致拒绝)
    f = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts_dup, lo, cl, 0, 1.0, grid, 20260109, 30, "30m", f)
    # 修正后: 收回窗口也逐槽核验, 重复 ts → INELIGIBLE + DUPLICATE_SLOT
    check("DUPa 重复 ts: status=INELIGIBLE (显式拒绝)",
          f["future_reclaim_status"] == "INELIGIBLE",
          f"status={f['future_reclaim_status']}")
    check("DUPb 重复 ts: reclaim=NULL (不洗成正常标签)",
          f["future_reclaim_30m"] is None,
          f"reclaim={f['future_reclaim_30m']}")
    check("DUPc 重复 ts: reason 含 DUPLICATE_SLOT",
          f["future_ineligible_reason"] is not None
          and "DUPLICATE_SLOT" in f["future_ineligible_reason"],
          f"reason={f['future_ineligible_reason']}")


# ================= 新增: 标签终点 = 依赖数据的最晚终点 =================
def test_label_end_latest():
    print("\n== 新增: 标签终点 = 依赖数据的最晚终点 ==")
    grid = make_grid()
    sig = make_signal(10.0, 9.0, 20260109093500, "BREAK", 9.5)
    ts = make_minute_ts(20260109093500, 60, grid)
    lo = np.full(len(ts), 9.0)
    lo[50] = 5.0  # 第 50 bar 最低 (穿透终点晚于收回终点)
    cl = np.full(len(ts), 9.8)
    cl[20] = 10.5  # 第 20 bar 收回 (收回终点早于穿透终点)
    f = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 60, "60m", f)
    # label_end_ts 应 = max(穿透终点=窗口末bar, 收回终点=第20 bar) = 窗口末 bar
    # 穿透终点 = win_bars[-1] (窗口最后一个 bar, 即第 60 个未来 bar)
    expected_end = ts[-1]  # 窗口末 bar (ts[0]=信号 bar, ts[-1]=第 60 未来 bar)
    check("ENDa label_end_ts = max(穿透, 收回)",
          f["future_label_end_ts"] == expected_end,
          f"end={f['future_label_end_ts']} expected={expected_end}")


# ================= P-HOLD 持续性测试 =================
def test_hold_10m():
    print("\n== P-HOLD: 持续性辅助标签 ==")
    grid = make_grid()
    sig = make_signal(10.0, 9.0, 20260109093500, "BREAK", 9.5)
    ts = make_minute_ts(20260109093500, 60, grid)
    lo = np.full(len(ts), 9.0)
    cl = np.full(len(ts), 9.8)
    cl[10] = 10.5
    cl[11:21] = 10.6  # 保持 10 bar
    f = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 60, "60m", f)
    reclaim_ts = int(ts[10])
    L.compute_future_hold(sig, ts, lo, cl, 0, 1.0, grid, 20260109, reclaim_ts, f)
    check("HOLDa 收回后保持: hold=1", f["future_hold_10m"] == 1)
    check("HOLDb label_end set", f["future_hold_label_end_ts"] is not None)

    cl2 = cl.copy()
    cl2[15] = 9.5  # 第 15 bar 跌破
    f2 = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl2, 0, 1.0, grid, 20260109, 60, "60m", f2)
    L.compute_future_hold(sig, ts, lo, cl2, 0, 1.0, grid, 20260109, reclaim_ts, f2)
    check("HOLDc 收回后跌破: hold=0", f2["future_hold_10m"] == 0)

    cl3 = np.full(len(ts), 9.8)  # 无收回
    f3 = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl3, 0, 1.0, grid, 20260109, 60, "60m", f3)
    L.compute_future_hold(sig, ts, lo, cl3, 0, 1.0, grid, 20260109, None, f3)
    check("HOLDd 无收回 (60m完整): hold=0", f3["future_hold_10m"] == 0)


# ================= 新增: 持续性缺槽被重复时间戳掩盖 → 显式拒绝 =================
def test_hold_duplicate_slot_rejected():
    print("\n== 新增: 持续性缺槽被重复 ts 掩盖 → 显式拒绝 ==")
    grid = make_grid()
    sig = make_signal(10.0, 9.0, 20260109093500, "BREAK", 9.5)
    ts = make_minute_ts(20260109093500, 60, grid)
    lo = np.full(len(ts), 9.0)
    cl = np.full(len(ts), 9.8)
    cl[10] = 10.5  # 第 10 bar 收回
    # hold 窗: 第 11-20 bar; 只给 9 个唯一 + 1 个重复 (第 5 bar 重复)
    # 构造: 收回后只需 10 分钟, 但第 3 个 hold bar 重复
    cl[11:21] = 10.6  # 保持 10 bar
    # 插入重复 ts (第 15 个, 即 hold 窗第 5 个)
    ts_dup = np.insert(ts, 15, ts[15])
    lo_dup = np.insert(lo, 15, lo[15])
    cl_dup = np.insert(cl, 15, cl[15])
    f = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts_dup, lo_dup, cl_dup, 0, 1.0, grid, 20260109, 60,
                                     "60m", f)
    reclaim_ts = int(ts[10])
    L.compute_future_hold(sig, ts_dup, lo_dup, cl_dup, 0, 1.0, grid, 20260109, reclaim_ts, f)
    check("HDUPa 持续性重复 ts: hold=CENSORED (None)",
          f["future_hold_10m"] is None,
          f"hold={f['future_hold_10m']}")
    check("HDUPb 持续性重复 ts: reason 含 DUPLICATE_SLOT",
          f["future_ineligible_reason"] is not None
          and "DUPLICATE_SLOT" in f["future_ineligible_reason"],
          f"reason={f['future_ineligible_reason']}")


# ================= 新增: 事件整体终点纳入持续性 =================
def test_event_end_includes_hold():
    print("\n== 新增: 事件整体终点纳入持续性 ==")
    grid = make_grid()
    sig = make_signal(10.0, 9.0, 20260109093500, "BREAK", 9.5)
    ts = make_minute_ts(20260109093500, 60, grid)
    lo = np.full(len(ts), 9.0)
    cl = np.full(len(ts), 9.8)
    cl[10] = 10.5  # 第 10 bar 收回
    cl[11:21] = 10.6  # 保持 10 bar (hold 终点 = 第 20 bar)
    f = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 60, "60m", f)
    reclaim_ts = int(ts[10])
    # 在收回窗口前, label_end 应 = 穿透终点 (窗口末 bar)
    end_before_hold = f["future_label_end_ts"]
    L.compute_future_hold(sig, ts, lo, cl, 0, 1.0, grid, 20260109, reclaim_ts, f)
    end_after_hold = f["future_label_end_ts"]
    # hold 终点 (第 20 bar) 应纳入事件整体终点
    hold_end = int(ts[20])
    check("ENDHOLDa 事件终点纳入持续性: 终点 ≥ hold 终点",
          end_after_hold >= hold_end,
          f"after={end_after_hold} hold_end={hold_end}")
    check("ENDHOLDb 事件终点纳入持续性: 终点 ≥ 穿透终点",
          end_after_hold >= end_before_hold,
          f"after={end_after_hold} before_hold={end_before_hold}")


# ================= 新增: 无效价格/ATR → NULL+原因 (不洗成正常标签) =================
def test_invalid_price_and_atr_null():
    print("\n== 新增: 无效价格/ATR → NULL+原因 ==")
    grid = make_grid()
    sig = make_signal(10.0, 9.0, 20260109093500, "BREAK", 9.5)
    ts = make_minute_ts(20260109093500, 30, grid)
    # 全 NaN 价格
    lo_nan = np.full(len(ts), np.nan)
    cl_nan = np.full(len(ts), np.nan)
    f = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo_nan, cl_nan, 0, 1.0, grid, 20260109, 30, "30m", f)
    check("NANa 全NaN价格: reclaim=NULL (不标 0)",
          f["future_reclaim_30m"] is None,
          f"reclaim={f['future_reclaim_30m']}")
    check("NANb 全NaN价格: penetration=NULL (不标 0)",
          f["future_penetration_30m"] is None,
          f"pen={f['future_penetration_30m']}")
    check("NANc 全NaN价格: status=INELIGIBLE",
          f["future_reclaim_status"] == "INELIGIBLE")
    check("NANd 全NaN价格: reason 含 INVALID_PRICE",
          f["future_ineligible_reason"] is not None
          and "INVALID_PRICE" in f["future_ineligible_reason"])

    # ATR=0
    lo = np.full(len(ts), 9.0)
    cl = np.full(len(ts), 9.8)
    f2 = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 0.0, grid, 20260109, 30, "30m", f2)
    check("ATR0a ATR=0: reclaim=NULL",
          f2["future_reclaim_30m"] is None,
          f"reclaim={f2['future_reclaim_30m']}")
    check("ATR0b ATR=0: penetration=NULL",
          f2["future_penetration_30m"] is None,
          f"pen={f2['future_penetration_30m']}")
    check("ATR0c ATR=0: reason 含 INVALID_ATR",
          f2["future_ineligible_reason"] is not None
          and "INVALID_ATR" in f2["future_ineligible_reason"])

    # ATR=NaN
    f3 = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, float("nan"), grid, 20260109, 30, "30m", f3)
    check("ATRNc ATR=NaN: reclaim=NULL",
          f3["future_reclaim_30m"] is None)
    check("ATRNd ATR=NaN: reason 含 INVALID_ATR",
          f3["future_ineligible_reason"] is not None
          and "INVALID_ATR" in f3["future_ineligible_reason"])


# ================= 新增: 持续性 hold close 全为 NaN → NULL (不洗成 hold=1) =================
def test_hold_nan_price_rejected():
    print("\n== 新增: 持续性 hold close 全 NaN → NULL ==")
    grid = make_grid()
    sig = make_signal(10.0, 9.0, 20260109093500, "BREAK", 9.5)
    ts = make_minute_ts(20260109093500, 60, grid)
    lo = np.full(len(ts), 9.0)
    cl = np.full(len(ts), 9.8)
    cl[10] = 10.5  # 第 10 bar 收回
    # hold 窗 (第 11-20 bar) close 全 NaN
    cl[11:21] = np.nan
    f = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 60, "60m", f)
    reclaim_ts = int(ts[10])
    L.compute_future_hold(sig, ts, lo, cl, 0, 1.0, grid, 20260109, reclaim_ts, f)
    check("HNANa hold NaN close: hold=NULL (不标 1)",
          f["future_hold_10m"] is None,
          f"hold={f['future_hold_10m']}")
    check("HNANb hold NaN close: status_hold=INELIGIBLE",
          f["future_status_hold"] == "INELIGIBLE",
          f"status_hold={f['future_status_hold']}")
    # hold 继承 60m 的 INELIGIBLE (因 60m 窗内含 NaN → 60m 先拒绝, hold 继承)
    # 或 hold 自身检测到 NaN (若 60m 未先拒) → reason 含 INVALID_PRICE 或 INHERITED
    check("HNANc hold NaN close: reason 含 INVALID_PRICE 或 INHERITED",
          f["future_reason_hold"] is not None
          and ("INVALID_PRICE" in f["future_reason_hold"]
               or "INHERITED" in f["future_reason_hold"]),
          f"reason_hold={f['future_reason_hold']}")


# ================= 新增: ATR=0 主标签已拒绝, hold 也应 NULL =================
def test_hold_atr_zero_null():
    print("\n== 新增: ATR=0 → hold 也 NULL (继承主标签) ==")
    grid = make_grid()
    sig = make_signal(10.0, 9.0, 20260109093500, "BREAK", 9.5)
    ts = make_minute_ts(20260109093500, 60, grid)
    lo = np.full(len(ts), 9.0)
    cl = np.full(len(ts), 9.8)
    cl[10] = 10.5
    f = L.init_car_label_fields()
    # ATR=0: 主标签拒绝
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 0.0, grid, 20260109, 60, "60m", f)
    check("HATR0a ATR=0: 60m reclaim=NULL",
          f["future_reclaim_60m"] is None)
    reclaim_ts = int(ts[10])
    L.compute_future_hold(sig, ts, lo, cl, 0, 0.0, grid, 20260109, reclaim_ts, f)
    check("HATR0b ATR=0: hold=NULL (不标 0)",
          f["future_hold_10m"] is None,
          f"hold={f['future_hold_10m']}")
    check("HATR0c ATR=0: status_hold=INELIGIBLE",
          f["future_status_hold"] == "INELIGIBLE")
    check("HATR0d ATR=0: reason_hold 含 INVALID_ATR",
          f["future_reason_hold"] is not None
          and "INVALID_ATR" in f["future_reason_hold"])


# ================= 新增: 30m/60m 计算顺序不影响综合状态 =================
def test_status_order_invariant():
    print("\n== 新增: 30m/60m 计算顺序不影响综合状态 ==")
    grid = make_grid()
    sig = make_signal(10.0, 9.0, 20260109093500, "BREAK", 9.5)
    ts = make_minute_ts(20260109093500, 40, grid)  # 40 bar: 30m 完整, 60m 截断
    lo = np.full(len(ts), 9.0)
    cl = np.full(len(ts), 9.8)

    # 顺序 A: 30m 先, 60m 后
    fA = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 30, "30m", fA)
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 60, "60m", fA)

    # 顺序 B: 60m 先, 30m 后
    fB = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 60, "60m", fB)
    L.compute_future_reclaim_window(sig, ts, lo, cl, 0, 1.0, grid, 20260109, 30, "30m", fB)

    check("ORDa 综合状态不因调用顺序变化",
          fA["future_reclaim_status"] == fB["future_reclaim_status"],
          f"A={fA['future_reclaim_status']} B={fB['future_reclaim_status']}")
    check("ORDb 分窗口 30m 状态独立",
          fA["future_status_30m"] == fB["future_status_30m"]
          and fA["future_status_30m"] == "NOT_RECLAIMED",
          f"A30={fA['future_status_30m']} B30={fB['future_status_30m']}")
    check("ORDc 分窗口 60m 状态独立",
          fA["future_status_60m"] == fB["future_status_60m"]
          and fA["future_status_60m"] == "INCOMPLETE_WINDOW",
          f"A60={fA['future_status_60m']} B60={fB['future_status_60m']}")
    check("ORDd 综合状态=INCOMPLETE_WINDOW (固定规则)",
          fA["future_reclaim_status"] == "INCOMPLETE_WINDOW",
          f"status={fA['future_reclaim_status']}")


# ================= 分开报告: 5 类事件分流 (真正断言) =================
def test_event_classification_report():
    print("\n== 分开报告: 5 类事件分流 (真正断言) ==")
    grid = make_grid()
    cal = make_cal(5)

    # 1. 未触发: 真正 NO_SIGNAL 应在事件提取/回放层验证, 不混进数据错误桶.
    #    标签函数层: 用未知类型 (非 APPROACH/BREAK/ALREADY_BELOW) 模拟 "事件提取层未产生有效信号",
    #    标签函数应显式拒绝为 INELIGIBLE; 但真正 NO_SIGNAL 的判定不在标签函数职责内.
    sig_none = make_signal(10.0, 9.0, 20260109093500, "NO_SIGNAL_PLACEHOLDER", 10.5)
    ts = make_minute_ts(20260109093500, 30, grid)
    f_none = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig_none, ts, np.full(len(ts), 9.0),
                                    np.full(len(ts), 9.5), 0, 1.0, grid, 20260109, 30, "30m",
                                    f_none)
    check("CLS1 未知类型(非NO_SIGNAL真验证): status=INELIGIBLE",
          f_none["future_reclaim_status"] == "INELIGIBLE",
          f"status={f_none['future_reclaim_status']} (注: 真正NO_SIGNAL在回放层验)")
    check("CLS1b 未知类型: reclaim=NULL",
          f_none["future_reclaim_30m"] is None)

    # 2. 已在 S 收回 → RECLAIMED_AT_S
    sig_s = make_signal(10.0, 9.0, 20260109093500, "BREAK", 10.5)
    f_s = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig_s, ts, np.full(len(ts), 9.0),
                                    np.full(len(ts), 9.5), 0, 1.0, grid, 20260109, 30, "30m", f_s)
    check("CLS2 已在S收回: status=RECLAIMED_AT_S",
          f_s["future_reclaim_status"] == "RECLAIMED_AT_S")
    check("CLS2b 已在S收回: reclaim=NULL (不进分母)",
          f_s["future_reclaim_30m"] is None)

    # 3. 有效未来成功 → RECLAIMED
    sig_e = make_signal(10.0, 9.0, 20260109093500, "BREAK", 9.5)
    cl_succ = np.full(len(ts), 9.8)
    cl_succ[20] = 10.5
    f_e = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig_e, ts, np.full(len(ts), 9.0), cl_succ, 0, 1.0, grid,
                                    20260109, 30, "30m", f_e)
    check("CLS3 有效未来成功: status=RECLAIMED",
          f_e["future_reclaim_status"] == "RECLAIMED")
    check("CLS3b 有效未来成功: reclaim=1", f_e["future_reclaim_30m"] == 1)

    # 4. 有效未来失败 → NOT_RECLAIMED (完整窗口未收回)
    cl_fail = np.full(len(ts), 9.8)
    f_f = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig_e, ts, np.full(len(ts), 9.0), cl_fail, 0, 1.0, grid,
                                    20260109, 30, "30m", f_f)
    check("CLS4 有效未来失败: status=NOT_RECLAIMED",
          f_f["future_reclaim_status"] == "NOT_RECLAIMED")
    check("CLS4b 有效未来失败: reclaim=0", f_f["future_reclaim_30m"] == 0)

    # 5. 数据不完整 → INCOMPLETE_WINDOW (不标 0)
    ts_short = make_minute_ts(20260109093500, 10, grid)  # 仅 10 bar < 30
    f_inc = L.init_car_label_fields()
    L.compute_future_reclaim_window(sig_e, ts_short, np.full(len(ts_short), 9.0),
                                    np.full(len(ts_short), 9.8), 0, 1.0, grid, 20260109, 30,
                                    "30m", f_inc)
    check("CLS5 数据不完整: status=INCOMPLETE_WINDOW",
          f_inc["future_reclaim_status"] == "INCOMPLETE_WINDOW")
    check("CLS5b 数据不完整: reclaim=NULL (不标 0)",
          f_inc["future_reclaim_30m"] is None,
          f"reclaim={f_inc['future_reclaim_30m']}")


def main():
    print("=" * 64)
    print("[GREEN-CAR-v0.1 标签合成验收 v2 — §9 必测 + 用户反例]")
    print("成功标准: 标签和因果链算得对, 可重放")
    print("=" * 64)

    test_4_reclaimed_at_s()
    test_5_not_in_denominator()
    test_6_not_reclaimed_null()
    test_7_window_independence()
    test_8_internal_gap_invalid()
    test_13_tail_truncation()
    test_2_future_perturbation_invariance()
    test_12_no_empty_pass()
    test_incomplete_window_null_not_zero()
    test_cross_day_no_pollution()
    test_cross_day_continuation()
    test_cross_lunch()
    test_unknown_sig_type_rejected()
    test_duplicate_timestamps()
    test_label_end_latest()
    test_hold_10m()
    test_hold_duplicate_slot_rejected()
    test_event_end_includes_hold()
    test_invalid_price_and_atr_null()
    test_hold_nan_price_rejected()
    test_hold_atr_zero_null()
    test_status_order_invariant()
    test_event_classification_report()

    print("\n" + "=" * 64)
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = len(FAILS)
    print(f"总计: {n_pass} PASS / {n_fail} FAIL / {len(RESULTS)} 项")
    if FAILS:
        print(f"失败: {FAILS}")
    else:
        print("全部通过 — 标签边界在合成数据上算稳")
    print("=" * 64)
    print("注: 本轮只做标签和因果链验收, 不训练 CAR-G/CAR-LR.")
    print("    13 类工程门中回放层项目 (资格映射/事件去重/单位/执行分流) 仍待验.")
    print("    真实回放前需另签去重请求清单 + SHA256.")

    # 失败时返回非零退出码, 阻止后续流程 (用户要求)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
