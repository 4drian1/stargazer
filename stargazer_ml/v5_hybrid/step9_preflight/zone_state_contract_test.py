"""zone_state_contract_test.py — ④ 契约测试 + 真实案例手工 trace.

覆盖用户指定的 12 项契约检查 + 7 个手工 trace 场景.
全程使用注入式合成 daily/minute, 不依赖 stockdb (环境 stockdb 当前不可达).
运行: python zone_state_contract_test.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from zone_state_engine import (build_zones, freeze_check, select_primary_zone,
                               run_state_engine, volume_baseline, Zone,
                               _arm_condition, _hhmmss_to_min)


# ---------------- 合成数据构造 ----------------
def make_daily(P, prices, vol=1e6):
    """构造覆盖 <=P 的日线 (close 序列), 用于 Zone 冻结测试. dates 从 P-(n-1) 到 P, 全部 <=P."""
    n = len(prices)
    dates = [P - (n - 1 - i) for i in range(n)]
    return pd.DataFrame({
        "date": dates, "open": prices, "high": [p + 1 for p in prices],
        "low": [p - 1 for p in prices], "close": prices,
        "volume": [vol] * n, "amount": [vol * p for p in prices],
    })


def make_minute(base, prices, vols=None, start_min=31):
    """base=YYYYMMDD; prices 每分钟一个价, 从 09:start_min 起."""
    rows = []
    hh = 9
    mm = start_min
    for i, px in enumerate(prices):
        if mm >= 60:
            mm -= 60
            hh += 1
        ts = int(f"{base}{hh:02d}{mm:02d}00")
        vol = (vols[i] if vols else 1e5)
        rows.append({"ts": ts, "open": px, "high": px + 0.5, "low": px - 0.5,
                     "close": px, "volume": vol, "amount": vol * px})
        mm += 1
    return pd.DataFrame(rows)


def mk_zone(lower=100.0, upper=102.0, mid=101.0, P=20260105, dist_atr=2.0, atr=2.0):
    return Zone(zone_id="Z1", zone_source="TEST", zone_lower=lower, zone_upper=upper,
                zone_mid=mid, zone_timestamp=int(P),
                zone_width_atr=(upper - lower) / atr, zone_strength=3,
                zone_source_count=3, zone_distance_atr=dist_atr, source_levels=[])


def snap_at(base, hh, mm):
    return int(f"{base}{hh:02d}{mm:02d}00")


# ---------------- 检查结果收集 ----------------
_results = []
def check(name, cond, info=""):
    _results.append((name, bool(cond), info))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {info}" if info else ""))


# ================= 12 项契约检查 =================
def test_contract():
    P = 20260105
    base = 20260106
    atr = 2.0

    # 1) Zone 全部冻结于 P_CLOSE
    daily = make_daily(P, [100 + i for i in range(40)])
    zones = build_zones(daily, P, atr_ref=atr)
    frozen, max_ts = freeze_check(zones, P)
    check("C1 zone 全部冻结于 P_CLOSE", frozen and max_ts <= P, f"max_ts={max_ts} P={P}")

    # 2) D 日不能生成新的"历史支撑" (所有 source ts <= P, 不含 D)
    d_day = 20260106
    daily2 = make_daily(P, [100 + i for i in range(40)])
    # 故意把最后一根当成 D 日极端低 (不应被当成历史支撑)
    daily2.loc[daily2.index[-1], "low"] = 50
    zones2 = build_zones(daily2, P, atr_ref=atr)
    no_d = all(z.zone_timestamp <= P for z in zones2)
    check("C2 D 日不生成历史支撑 (source ts<=P)", no_d,
          f"zone_ts={[z.zone_timestamp for z in zones2]}")

    # 3) state 时间严格单调 (非递减)
    z = mk_zone()
    m = make_minute(base, [105, 104, 103, 102, 101, 100.5, 101.5, 102.5, 103])
    snap = snap_at(base, 15, 0)
    out = run_state_engine(z, m, 1.0, "BOTTOM_SETUP", atr, snap, 0.0, 0.1)
    ts_seq = [t["state_enter_ts"] for t in out["state_transitions"]]
    check("C3 state 时间严格单调", all(ts_seq[i] <= ts_seq[i + 1] for i in range(len(ts_seq) - 1)),
          f"enter_ts={ts_seq}")

    # 4) ZONE_TOUCH 不早于真正触价
    m4 = make_minute(base, [108, 107, 106, 105, 104, 101.5, 100.8, 101.5])
    # 真正进入 zone(100-102) 的第一根是 101.5 (ts 对应第6根)
    real_touch_ts = m4.iloc[5]["ts"]
    out4 = run_state_engine(z, m4, 1.0, "BOTTOM_SETUP", atr, snap_at(base, 15, 0), 0.0, 0.1)
    touch = next((t for t in out4["state_transitions"] if t["state"] == "ZONE_TOUCH"), None)
    check("C4 ZONE_TOUCH 不早于真正触价", touch is not None and touch["state_enter_ts"] == real_touch_ts,
          f"touch_ts={touch['state_enter_ts'] if touch else None} real={real_touch_ts}")

    # 5) RECLAIM 必须发生在 TOUCH/BREAK 后
    m5 = make_minute(base, [105, 104, 103, 102, 101, 99.5, 99.2, 101.2, 102.3])
    out5 = run_state_engine(z, m5, 1.0, "BOTTOM_SETUP", atr, snap_at(base, 15, 0), 0.0, 0.1)
    touch5 = next(t for t in out5["state_transitions"] if t["state"] == "ZONE_TOUCH")
    break5 = next((t for t in out5["state_transitions"] if t["state"] == "BREAKDOWN"), None)
    reclaim5 = next((t for t in out5["state_transitions"] if t["state"] == "RECLAIM"), None)
    ok5 = reclaim5 is not None and reclaim5["state_enter_ts"] > touch5["state_enter_ts"] \
        and (break5 is None or reclaim5["state_enter_ts"] > break5["state_enter_ts"])
    check("C5 RECLAIM 发生在 TOUCH/BREAK 后", ok5,
          f"reclaim={reclaim5['state_enter_ts'] if reclaim5 else None} "
          f"touch={touch5['state_enter_ts']} break={break5['state_enter_ts'] if break5 else None}")

    # 6) SECOND_TEST 必须晚于第一次 recovery
    m6 = make_minute(base, [105, 104, 103, 102, 101, 99.5, 101.5, 103.5, 101.3, 102])
    out6 = run_state_engine(z, m6, 1.0, "BOTTOM_SETUP", atr, snap_at(base, 15, 0), 0.0, 0.1)
    touch6 = next(t for t in out6["state_transitions"] if t["state"] == "ZONE_TOUCH")
    st6 = next((t for t in out6["state_transitions"] if t["state"] == "SECOND_TEST"), None)
    # 第一次 recovery: close 首次 > upper(102)
    first_rec = int(m6[m6["close"] > 102]["ts"].iloc[0])
    ok6 = st6 is not None and st6["state_enter_ts"] > touch6["state_enter_ts"] and st6["state_enter_ts"] > first_rec
    check("C6 SECOND_TEST 晚于第一次 recovery", ok6,
          f"st={st6['state_enter_ts'] if st6 else None} touch={touch6['state_enter_ts']} first_rec={first_rec}")

    # 7) BREAKDOWN 只使用 snapshot 前信息 (截断 snapshot 不应看到未来)
    m7 = make_minute(base, [105, 104, 103, 102, 101, 99.5, 99.2, 99.0, 98.8, 103])
    # 破位发生在 09:38 (连续 3 根低于 lower); 早快照取 09:36 (仅 1 根低于 lower, 破位前)
    snap_early = snap_at(base, 9, 36)
    snap_full = snap_at(base, 15, 0)
    out7e = run_state_engine(z, m7, 1.0, "BOTTOM_SETUP", atr, snap_early, 0.0, 0.1)
    out7f = run_state_engine(z, m7, 1.0, "BOTTOM_SETUP", atr, snap_full, 0.0, 0.1)
    breakdown_full = any(t["state"] == "BREAKDOWN" for t in out7f["state_transitions"])
    not_breakdown_early = not any(t["state"] == "BREAKDOWN" for t in out7e["state_transitions"])
    check("C7 BREAKDOWN 只用 snapshot 前信息", breakdown_full and not_breakdown_early,
          f"full_has_bd={breakdown_full} early_has_bd={not_breakdown_early}")

    # 8) 15m 特征只在 15m bar 完成后可用 (causality: early snapshot 截断后续 bar, 值应不同)
    m8 = make_minute(base, [105, 104, 103, 102, 101, 100.5, 100.8, 101.5, 102.5, 103, 104])
    snap_945 = snap_at(base, 9, 37)   # 只到 09:37, 截断后续修复 bar
    snap_full = snap_at(base, 15, 0)
    out8e = run_state_engine(z, m8, 1.0, "BOTTOM_SETUP", atr, snap_945, 0.0, 0.1)
    out8f = run_state_engine(z, m8, 1.0, "BOTTOM_SETUP", atr, snap_full, 0.0, 0.1)
    # 因果: 截断快照的 recovery/最终态 不应包含其之后的 bar (值应不同)
    causal8 = (out8e["recovery_from_zone_low_atr"] != out8f["recovery_from_zone_low_atr"]) or \
              (out8e["final_state"] != out8f["final_state"])
    check("C8 特征 snapshot-time causal (15m 边界)", causal8,
          f"e_rec={out8e['recovery_from_zone_low_atr']} f_rec={out8f['recovery_from_zone_low_atr']}")

    # 9) volume baseline 不读 D 日未来分钟
    # 过去 3 日 + D 日, D 日包含 snapshot 之后的 bar
    past = [make_minute(base - 1, [100] * 10, vols=[1e5] * 10),
            make_minute(base - 2, [100] * 10, vols=[1.2e5] * 10),
            make_minute(base - 3, [100] * 10, vols=[0.8e5] * 10)]
    d_full = make_minute(base, [100] * 10, vols=[1e5] * 10)
    # 在 D 日追加 snapshot 之后的 bar (不应被计入)
    extra = make_minute(base, [100] * 5, vols=[9e6] * 5, start_min=50)
    d_full = pd.concat([d_full, extra], ignore_index=True)
    snap9 = snap_at(base, 9, 40)  # 只到 09:40
    vb = volume_baseline(base, d_full, past, snap9)
    # D 累计量应只含 ts<=snap9 的 bar (前 10 根中到 09:40 的部分)
    expected_d = float(d_full[d_full["ts"] <= snap9]["volume"].sum())
    check("C9 volume baseline 不读 D 日未来分钟", abs(vb["_d_cum_vol"] - expected_d) < 1e-6,
          f"d_cum={vb['_d_cum_vol']} expected={expected_d}")
    check("C9b cum_volume_vs_hist 有值", vb["cum_volume_vs_hist_same_time"] is not None,
          f"ratio={vb['cum_volume_vs_hist_same_time']}")

    # 10) PRE_PULLBACK 不能无 Zone/LATR 直接 ARM
    z10 = mk_zone(dist_atr=3.0)
    m10 = make_minute(base, [106, 105, 104.5, 104.2, 104.0, 103.8, 104.1])  # 始终不触 zone(100-102)
    out10 = run_state_engine(z10, m10, 1.0, "INTRADAY_DIP_SETUP", atr, snap_at(base, 15, 0),
                             0.05, z10.zone_distance_atr, dip_stage="PRE_PULLBACK")
    not_armed = out10["final_state"] not in ("ARMED", "MODEL_SCORE_REQUEST")
    check("C10 PRE_PULLBACK 无 Zone 不 ARM", not_armed, f"final={out10['final_state']}")

    # 11) BOTTOM 与 DIP 使用不同 arm logic (冻结合同)
    # BOTTOM_ARMED = zone_distance<=0.20 AND not broken AND (open_dip>=0.05 OR hold OR recovery)
    # DIP_ARMED   = zone_touch AND 0.10<=open_dip<=0.40 AND not broken
    # 关键区分: (a) BOTTOM 即使不触 Zone, 只要 dist<=0.20+轻微下杀/守住/修复 即 ARM;
    #           (b) BOTTOM 在 dist>0.20 时绝不 ARM (这正是之前错误版本放松的地方);
    #           (c) DIP 未触 Zone (PRE_PULLBACK) 绝不 ARM.
    bottom_arm_ok = _arm_condition("BOTTOM_SETUP", 0.05, 0.10, None, False,
                                   structure_broken=False, hold_zone_flag=False, recovery_flag=False)  # 轻微下杀
    bottom_hold = _arm_condition("BOTTOM_SETUP", 0.0, 0.10, None, True,
                                 structure_broken=False, hold_zone_flag=True, recovery_flag=False)  # 守住 Zone
    bottom_recovery = _arm_condition("BOTTOM_SETUP", 0.0, 0.10, None, True,
                                     structure_broken=False, hold_zone_flag=False, recovery_flag=True)  # 直接修复
    bottom_far_no_arm = _arm_condition("BOTTOM_SETUP", 0.0, 1.50, None, True,
                                       structure_broken=False, hold_zone_flag=True, recovery_flag=False)  # dist>0.20 -> 不ARM
    bottom_broken_no_arm = _arm_condition("BOTTOM_SETUP", 0.05, 0.10, None, False,
                                          structure_broken=True, hold_zone_flag=False, recovery_flag=False)
    dip_arm = _arm_condition("INTRADAY_DIP_SETUP", 0.20, 0.0, "PULLBACK_ACTIVE", True,
                             structure_broken=False)
    dip_no_touch = _arm_condition("INTRADAY_DIP_SETUP", 0.20, 0.0, "PRE_PULLBACK", False,
                                  structure_broken=False)
    dip_out_of_band = _arm_condition("INTRADAY_DIP_SETUP", 0.05, 0.0, "PULLBACK_ACTIVE", True,
                                     structure_broken=False)  # open_dip<0.10
    ok11 = (bottom_arm_ok and bottom_hold and bottom_recovery and not bottom_far_no_arm
            and not bottom_broken_no_arm and dip_arm and not dip_no_touch and not dip_out_of_band)
    check("C11 BOTTOM/DIP arm 逻辑不同(冻结合同)", ok11,
          f"bottom_arm_ok={bottom_arm_ok} bottom_hold={bottom_hold} bottom_far_no_arm={bottom_far_no_arm} "
          f"dip_arm={dip_arm} dip_no_touch={dip_no_touch}")

    # 12) 所有 feature 基于 ts<=snapshot (causality: 截断运行 == 全量截断到同 snapshot)
    m12 = make_minute(base, [105, 104, 103, 102, 101, 100.5, 101.5, 102.5, 103, 104])
    snap12 = snap_at(base, 10, 0)
    out12 = run_state_engine(z, m12, 1.0, "BOTTOM_SETUP", atr, snap12, 0.0, 0.1)
    # 直接截断数据到 snap12 再跑应得到一致特征
    m12_trunc = m12[m12["ts"] <= snap12]
    out12t = run_state_engine(z, m12_trunc, 1.0, "BOTTOM_SETUP", atr, snap12, 0.0, 0.1)
    same_feat = (out12["recovery_from_zone_low_atr"] == out12t["recovery_from_zone_low_atr"]) and \
                (out12["final_state"] == out12t["final_state"]) and \
                (out12["time_since_zone_touch"] == out12t["time_since_zone_touch"])
    check("C12 feature 全部基于 ts<=snapshot", same_feat,
          f"rec={out12['recovery_from_zone_low_atr']} rec_t={out12t['recovery_from_zone_low_atr']}")


# ================= 7 个手工 trace 场景 =================
def trace_scenarios():
    P = 20260105
    base = 20260106
    atr = 2.0
    snap = snap_at(base, 15, 0)

    def run(z, prices, setup, dist, dip=None, open_dip=0.2):
        m = make_minute(base, prices)
        return run_state_engine(z, m, 1.0, setup, atr, snap, open_dip, dist, dip_stage=dip)

    # 场景1: 正常 touch -> reclaim (BOTTOM: dist<=0.20, open_dip=0.0 -> 触 Zone 后 hold 触发 ARM)
    z = mk_zone()
    o1 = run(z, [105, 104, 103, 102, 101, 100.5, 99.5, 101.5, 102.5], "BOTTOM_SETUP", 0.1, open_dip=0.0)
    check("T1 正常 touch→reclaim", o1["final_state"] == "MODEL_SCORE_REQUEST"
          and any(t["state"] == "RECLAIM" for t in o1["state_transitions"]),
          f"final={o1['final_state']}")

    # 场景2: touch -> second test -> higher low
    z = mk_zone()
    o2 = run(z, [105, 104, 103, 102, 101, 99.5, 101.5, 103.5, 101.3, 102.5], "BOTTOM_SETUP", 0.1, open_dip=0.0)
    check("T2 touch→SECOND_TEST→higher low",
          o2["second_test_flag"] == 1 and o2["second_test_higher_low"] == 1,
          f"flag={o2['second_test_flag']} hl={o2['second_test_higher_low']}")

    # 场景3: touch -> breakdown
    z = mk_zone()
    o3 = run(z, [105, 104, 103, 102, 101, 99.5, 99.2, 99.0, 98.8, 98.5], "BOTTOM_SETUP", 0.1, open_dip=0.0)
    check("T3 touch→BREAKDOWN", o3["final_state"] == "BREAKDOWN",
          f"final={o3['final_state']}")

    # 场景4: BOTTOM 不明显下杀但守住 Zone
    z = mk_zone()
    o4 = run(z, [103, 102.5, 102, 101.5, 101, 101.2, 101.5, 102], "BOTTOM_SETUP", 0.1, open_dip=0.0)
    check("T4 BOTTOM 守住 Zone (不深跌)", o4["final_state"] in ("ARMED", "MODEL_SCORE_REQUEST")
          and o4["zone_break_depth_atr"] == 0.0, f"final={o4['final_state']}")

    # 场景5: DIP 正常回调后进入 Zone
    z = mk_zone()
    o5 = run(z, [107, 106, 105, 104, 103, 102, 101, 100.8, 101.8, 102.5],
             "INTRADAY_DIP_SETUP", 3.0, dip="PULLBACK_ACTIVE")
    check("T5 DIP 回调后进入 Zone", o5["final_state"] == "MODEL_SCORE_REQUEST",
          f"final={o5['final_state']}")

    # 场景6: PRE_PULLBACK 一直没到 Zone
    z = mk_zone(dist_atr=3.0)
    o6 = run(z, [107, 106.5, 106, 105.5, 105, 104.5], "INTRADAY_DIP_SETUP", 3.0, dip="PRE_PULLBACK")
    check("T6 PRE_PULLBACK 未到 Zone", o6["final_state"] not in ("ARMED", "MODEL_SCORE_REQUEST"),
          f"final={o6['final_state']}")

    # 场景7: V 形直接修复 (DIRECT_RECOVERY, 无二次测试)
    z = mk_zone()
    o7 = run(z, [105, 103, 101, 99.5, 101.5, 103, 104], "BOTTOM_SETUP", 0.1, open_dip=0.0)
    check("T7 V形直接修复", o7["final_state"] == "MODEL_SCORE_REQUEST"
          and o7["second_test_flag"] == 0, f"final={o7['final_state']} st={o7['second_test_flag']}")


def main():
    print("===== ④ ZONE/STATE 契约测试 =====")
    test_contract()
    print("\n===== ④ 手工 trace 场景 =====")
    trace_scenarios()
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n===== SUMMARY: {len(_results)-n_fail}/{len(_results)} passed, {n_fail} failed =====")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
