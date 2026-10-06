"""CAR v0.1 合成工程门测试 (2026-09-29, 重构版).

直接调用生产小函数 check_gap_propagation / check_corp_action:
  1. 第59分钟收回, hold 窗内缺槽
  2. E日完整, 下一交易日标签窗内缺一槽 (含241→242网格版本切换)
  3. 复权链失败 → UNKNOWN 并隔离
  4. 窗口内除权 → 隔离
  5. 窗口结束后的除权 → 不误伤当前标签

使用 minute_grid.json 冻结网格, 不手写网格.
独立参考核对期望结果, 不复制生产逻辑.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "step9_preflight"))
sys.path.insert(0, str(HERE))
import gcp_round_b_engine as E  # noqa: E402
from gcp_car_v0_1_engineering_gates import check_gap_propagation, check_corp_action  # noqa: E402

# ---- 加载冻结网格 (不手写) ----
_GRID_JSON = json.loads(
    (HERE / "gcp_round_b_small_sample_v1" / "minute_grid.json").read_text(encoding="utf-8")
)
GRID = _GRID_JSON["grid"]  # 242 槽 (含 13:00, >= 20260624)
GRID_1300_FROM = _GRID_JSON["grid_1300_from"]  # 20260624


def _make_minute_df(day: int, slots: list, base_price: float = 10.0) -> pd.DataFrame:
    """构造分钟 DataFrame, 给定槽位列表."""
    rows = []
    for s in slots:
        ts = int(f"{day}{s:06d}")
        rows.append({
            "date": ts, "open": base_price, "high": base_price + 0.1,
            "low": base_price - 0.1, "close": base_price, "volume": 10000, "amount": 100000,
        })
    return pd.DataFrame(rows)


# ================================================================
# 独立参考: 手工推导期望结果 (不调用生产函数)
# ================================================================

def _independent_hold_window(sig_slot, e_date, ttr, grid, cal):
    """独立推导 hold 窗口的 (day, slot) 列表.

    不使用 _build_future_window_slots, 直接按定义推导:
    reclaim = S 后第 ttr 个交易分钟; hold = reclaim 之后 10 个交易分钟.
    """
    exp = E.expected_grid(grid, e_date)
    if sig_slot not in exp:
        return []
    sig_gi = exp.index(sig_slot)
    # reclaim 位置: sig_gi + ttr (1-indexed: 第1个 = sig_gi+1)
    reclaim_idx = sig_gi + int(ttr)
    # 构建 hold 窗口: reclaim_idx+1 .. reclaim_idx+10 (跨日)
    hold = []
    cur_day = e_date
    cur_exp = exp
    cur_gi = reclaim_idx + 1
    remaining = 10
    while remaining > 0:
        # 当日剩余槽位
        day_slots = cur_exp[cur_gi: cur_gi + remaining]
        for s in day_slots:
            hold.append((cur_day, s))
            remaining -= 1
        if remaining <= 0:
            break
        # 跨到下一交易日
        if cal is None or cur_day not in cal:
            break
        di = cal.index(cur_day) + 1
        if di >= len(cal):
            break
        cur_day = cal[di]
        cur_exp = E.expected_grid(grid, cur_day)
        cur_gi = 0
    return hold


def _independent_window_slots(sig_slot, e_date, wlen, grid, cal):
    """独立推导标签窗口的 (day, slot) 列表."""
    exp = E.expected_grid(grid, e_date)
    if sig_slot not in exp:
        return []
    sig_gi = exp.index(sig_slot)
    win = []
    cur_day = e_date
    cur_exp = exp
    cur_gi = sig_gi + 1
    remaining = wlen
    while remaining > 0:
        day_slots = cur_exp[cur_gi: cur_gi + remaining]
        for s in day_slots:
            win.append((cur_day, s))
            remaining -= 1
        if remaining <= 0:
            break
        if cal is None or cur_day not in cal:
            break
        di = cal.index(cur_day) + 1
        if di >= len(cal):
            break
        cur_day = cal[di]
        cur_exp = E.expected_grid(grid, cur_day)
        cur_gi = 0
    return win


# ================================================================
# 测试用例
# ================================================================

def test_case1_hold_gap():
    """案例1: 第59分钟收回, hold 窗内缺槽."""
    print("\n[案例1] 第59分钟收回, hold 窗内缺槽")
    e_date = 20260115
    cal = [20260115, 20260116]
    sig_slot = 93500  # index 5
    ttr = 59
    sig_ts = int(f"{e_date}{sig_slot:06d}")

    exp = E.expected_grid(GRID, e_date)
    sig_gi = exp.index(sig_slot)

    # 独立推导: hold 窗口 (10 slots after reclaim)
    hold_ref = _independent_hold_window(sig_slot, e_date, ttr, GRID, cal)
    # 独立推导: 30m / 60m 窗口
    win30_ref = set(_independent_window_slots(sig_slot, e_date, 30, GRID, cal))
    win60_ref = set(_independent_window_slots(sig_slot, e_date, 60, GRID, cal))

    # 在 hold 窗内删一个槽 (hold_ref 中间的一个)
    removed = hold_ref[5]  # (e_date, slot)
    removed_slot = removed[1]
    slots = [s for s in exp if s != removed_slot]
    e_df = _make_minute_df(e_date, slots)

    # 调用生产函数
    result = check_gap_propagation(e_df, None, sig_ts, ttr, GRID, cal, e_date)

    # 独立核对
    assert result["hold_10m"] == [removed], \
        f"hold 缺槽应={removed}, 实际={result['hold_10m']}"
    assert result["30m"] == [], f"30m 不应有缺槽, 实际={result['30m']}"
    assert result["60m"] == [], f"60m 不应有缺槽, 实际={result['60m']}"
    assert result["prefix_to_S"] == [], f"prefix 不应有缺槽"
    print(f"  sig_gi={sig_gi}, ttr={ttr}, removed={removed}")
    print(f"  hold_ref={hold_ref}")
    print(f"  result: hold={result['hold_10m']}, 30m={result['30m']}, 60m={result['60m']}")
    print("  PASS: hold 窗内缺槽被正确检测, 30m/60m 无误报")


def test_case2_next_day_gap():
    """案例2: E日完整, 下一交易日标签窗内缺一槽 (含241→242网格切换)."""
    print("\n[案例2] E日完整, 下一交易日标签窗内缺一槽 (241→242网格切换)")
    e_date = 20260623    # < 20260624 → 241 槽 (无 13:00)
    next_date = 20260624  # >= 20260624 → 242 槽 (含 13:00)
    cal = [20260623, 20260624, 20260625]
    sig_slot = 145500  # 接近收盘, 60m 窗口跨日
    sig_ts = int(f"{e_date}{sig_slot:06d}")

    # 独立推导: 60m 窗口
    win60_ref = _independent_window_slots(sig_slot, e_date, 60, GRID, cal)
    win30_ref = set(_independent_window_slots(sig_slot, e_date, 30, GRID, cal))

    # E日完整 (不删槽)
    exp_e = E.expected_grid(GRID, e_date)
    e_df = _make_minute_df(e_date, list(exp_e))

    # 下日: 在 60m 窗口范围但不在 30m 窗口内删一个槽
    next_exp = E.expected_grid(GRID, next_date)
    nd_win60 = [p for p in win60_ref if p[0] == next_date]  # 下日 60m 部分 (55 槽)
    # 30m 在下日只占前 25 槽 (nd_win60[0:25]); 取 nd_win60[30] = 下日 index 30 = 100000
    removed_nd = nd_win60[30]  # 在 60m 内但不在 30m 内
    removed_slot_nd = removed_nd[1]
    slots_nd = [s for s in next_exp if s != removed_slot_nd]
    nd_df = _make_minute_df(next_date, slots_nd)

    # 调用生产函数
    result = check_gap_propagation(e_df, nd_df, sig_ts, None, GRID, cal, e_date)

    # 独立核对
    assert result["60m"] == [removed_nd], \
        f"60m 缺槽应={removed_nd}, 实际={result['60m']}"
    assert result["30m"] == [], f"30m 不应有缺槽, 实际={result['30m']}"
    assert result["hold_10m"] == [], f"hold 不应有缺槽 (ttr=None)"
    nd = result.get("next_day")
    assert nd is not None, "next_day 不应为 None"
    assert nd["60m"] == [removed_nd], \
        f"next_day 60m 应={removed_nd}, 实际={nd['60m']}"
    assert nd["30m"] == [], f"next_day 30m 不应有缺槽"
    # 验证网格版本切换: E日241槽, 下日242槽
    assert len(exp_e) == 241, f"E日应为241槽, 实际={len(exp_e)}"
    assert len(next_exp) == 242, f"下日应为242槽, 实际={len(next_exp)}"
    print(f"  E日槽数={len(exp_e)}, 下日槽数={len(next_exp)} (网格版本切换)")
    print(f"  sig_slot={sig_slot}, removed_nd={removed_nd}")
    print(f"  result: 60m={result['60m']}, next_day.60m={nd['60m']}")
    print("  PASS: 下日标签窗内缺槽被正确检测, 网格版本切换正确")


def test_case3_adj_unknown():
    """案例3: 复权链失败 → UNKNOWN 并隔离."""
    print("\n[案例3] 复权链失败 → UNKNOWN 并隔离")
    adj_entry = {"ok": False, "reason": "ERROR:Timeout"}
    p_date = 20260115
    label_end_ts = int("20260119150000")

    result = check_corp_action(adj_entry, p_date, label_end_ts)

    assert result["status"] == "ADJ_UNKNOWN", f"应为 ADJ_UNKNOWN, 实际={result['status']}"
    assert result["pass"] is False, "链失败应隔离 (pass=False)"
    assert result["events_in_window"] == [], "不应有窗口内事件"
    print(f"  status={result['status']}, pass={result['pass']}")
    print("  PASS: 复权链失败 → ADJ_UNKNOWN + 隔离传播到候选整体")


def test_case4_adj_in_window():
    """案例4: 窗口内除权 → 隔离."""
    print("\n[案例4] 窗口内除权 → 隔离")
    p_date = 20260115
    label_end_ts = int("20260119150000")  # label_end_date = 20260119
    adj_entry = {"ok": True, "chain": {"20260117": 1.1}}  # 在窗口 [P+1, label_end] 内

    result = check_corp_action(adj_entry, p_date, label_end_ts)

    assert result["status"] == "HAS_POST_P_EVENTS", \
        f"应为 HAS_POST_P_EVENTS, 实际={result['status']}"
    assert result["pass"] is False, "窗口内除权应隔离"
    assert len(result["events_in_window"]) == 1, "应有1个窗口内事件"
    assert result["events_in_window"][0]["date"] == "20260117"
    print(f"  events_in={result['events_in_window']}, status={result['status']}, pass={result['pass']}")
    print("  PASS: 窗口内除权 → HAS_POST_P_EVENTS + 隔离")


def test_case5_adj_outside_window():
    """案例5: 窗口结束后的除权 → 不误伤当前标签."""
    print("\n[案例5] 窗口结束后的除权 → 不误伤当前标签")
    p_date = 20260115
    label_end_ts = int("20260119150000")  # label_end_date = 20260119
    adj_entry = {"ok": True, "chain": {"20260601": 1.2}}  # 远在窗口后

    result = check_corp_action(adj_entry, p_date, label_end_ts)

    assert result["status"] == "NO_EVENT_IN_WINDOW", \
        f"应为 NO_EVENT_IN_WINDOW, 实际={result['status']}"
    assert result["pass"] is True, "窗口后除权不应隔离"
    assert result["events_in_window"] == [], "不应有窗口内事件"
    assert len(result["events_outside_window"]) == 1, "应有1个窗口外事件"
    assert result["events_outside_window"][0]["date"] == "20260601"
    print(f"  events_in={result['events_in_window']}, events_out={result['events_outside_window']}")
    print(f"  status={result['status']}, pass={result['pass']}")
    print("  PASS: 窗口后除权 → NO_EVENT_IN_WINDOW + 不误伤")


def main():
    print("=" * 60)
    print("CAR v0.1 合成工程门测试 (重构版, 调用生产函数)")
    print("=" * 60)
    test_case1_hold_gap()
    test_case2_next_day_gap()
    test_case3_adj_unknown()
    test_case4_adj_in_window()
    test_case5_adj_outside_window()
    print("\n" + "=" * 60)
    print("[汇总] 5/5 合成测试全部 PASS")
    print("  1. 第59分钟收回 hold 窗内缺槽 → 正确检测 (调用 check_gap_propagation)")
    print("  2. 下日标签窗内缺槽 → 正确检测 (241→242网格切换)")
    print("  3. 复权链失败 → ADJ_UNKNOWN + 隔离传播 (调用 check_corp_action)")
    print("  4. 窗口内除权 → HAS_POST_P_EVENTS + 隔离")
    print("  5. 窗口后除权 → NO_EVENT_IN_WINDOW + 不误伤")
    print("=" * 60)


if __name__ == "__main__":
    main()
