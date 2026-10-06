# -*- coding: utf-8 -*-
"""双路径执行 v0.1.1 缺槽依赖反例回归 (8 组, 修复后门禁).

覆盖任务书反例集:
  ① 退出前缺 1/2 根且缺槽前未闭环 -> NULL (与缺几根无关)
  ② 缺槽后出现明显止损或反弹 -> 仍不能恢复有效
  ③ 缺槽前已全部退出 -> 交易标签保留 (含 14:57-15:00 尾部缺槽)
  ④ 部分卖出后遇缺槽 -> 残仓保留, 净收益 NULL
  ⑤ 入场等待期间缺槽 -> 不能继续成交 (ENTRY_DATA_MISSING, 保持 NULL)
  ⑥ H3 已闭环 / H5 未闭环 -> 分别记录, 互不删除
  ⑦ 跨日 / 午休 / 241-242 网格版本边界
  ⑧ 终点后截断与扰动不改变已闭环字段
另钉: 零量槽不是缺槽 (不默认停牌); 完成 bar 之后的 E 日缺槽不影响入场前缀.

零网络. 产物: gcp_round_b_small_sample_v1/execution_reconcile_v0_1_1/gap_regress/synthetic_gap_regress.json
"""
from __future__ import annotations
import sys, json, os
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dual_path_execution_v0_1_1 as D
from dual_path_synthetic_acceptance import mkdaily

PASS, FAIL = [], []

def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    return cond

V = 1_000_000.0

def mkday(date, base, overrides=None, base_vol=V, drop=()):
    """完整网格合成日; drop=要剔除的槽位 (模拟缺槽)."""
    g = D.day_grid(date)
    rows = []
    for s in g:
        if s in drop:
            continue
        o = h = l = c = base
        vol = base_vol
        if overrides and s in overrides:
            o, h, l, c, vol = overrides[s]
        rows.append({"date": int(f"{date}{s:06d}"), "open": o, "high": h,
                     "low": l, "close": c, "volume": vol, "amount": vol * o})
    return pd.DataFrame(rows)

def mkentry(E, shares=400, price=100.0):
    return {"sim_fill_ts": int(f"{E}100000"), "sim_fill_price": price,
            "shares": shares, "amount": shares * price * D.BUY_MULT}

def run_days(E, days, plan_offset, cal=None, stop=95.0, shares=400, price=100.0):
    """days: list of (date, df); 返回 execute_track 结果."""
    mbd = dict(days)
    cal = cal or [d for d, _ in days]
    pre = {d: 100.0 for d in cal}
    st = {d: False for d in cal}
    return D.execute_track(mkentry(E, shares, price), stop, "SH_MAIN", st, cal,
                           mbd, plan_offset, pre)

CAL6 = [20250720 + i for i in range(6)]

# ================= ① 退出前缺 1/2 根且缺槽前未闭环 -> NULL =================
def g1():
    print("① 退出前缺1/2根未闭环 -> NULL")
    E = 20250720
    # T+1: 09:31 low=94 触发止损(pending), 09:32 缺槽 -> 不得在 09:33 成交
    d1 = mkday(20250721, 100.0, {93100: (100, 100, 94, 100, V)}, drop=(93200,))
    r1 = run_days(E, [(E, mkday(E, 100.0)), (20250721, d1)], 3, cal=CAL6)
    check("1-slot gap before close -> EXIT_DATA_MISSING", r1["status"] == "EXIT_DATA_MISSING", r1["status"])
    check("1-slot: net NULL", r1["net"] is None, str(r1["net"]))
    check("1-slot: no fill past gap (no sells)", len(r1["sells"]) == 0, str(r1["sells"]))
    check("1-slot: position kept (remaining=400)", r1["remaining_shares"] == 400, str(r1["remaining_shares"]))
    check("1-slot: dep recorded as day_gap", any(m.get("type") == "day_gap" for m in r1["missing_dependencies"]),
          str(r1["missing_dependencies"]))
    # T+1: 09:32 与 09:33 同时缺 -> 同等拒绝
    d1b = mkday(20250721, 100.0, {93100: (100, 100, 94, 100, V)}, drop=(93200, 93300))
    r1b = run_days(E, [(E, mkday(E, 100.0)), (20250721, d1b)], 3, cal=CAL6)
    check("2-slot gap equally rejected -> EXIT_DATA_MISSING", r1b["status"] == "EXIT_DATA_MISSING", r1b["status"])
    check("2-slot: net NULL / no sells", r1b["net"] is None and len(r1b["sells"]) == 0, str(r1b["sells"]))
    # 边界-: 完成 bar 早于当日首个缺槽 -> 不否定 (09:30 跳空触发, 09:31 成交; 11:30 缺槽)
    d1c = mkday(20250721, 100.0, {93000: (90, 90, 90, 90, V), 93100: (90, 90, 90, 90, V)}, drop=(113000,))
    r1c = run_days(E, [(E, mkday(E, 100.0)), (20250721, d1c)], 3, cal=CAL6)
    check("close before same-day gap -> SIM_COMPLETE kept", r1c["status"] == "SIM_COMPLETE", r1c["status"])
    check("close before gap: net not NULL", r1c["net"] is not None, str(r1c["net"]))

# ============ ② 缺槽后出现明显止损或反弹 -> 仍不能恢复有效 ============
def g2():
    print("② 缺槽后明显止损/反弹仍无效")
    E = 20250720
    # T+1 缺 10:00; T+2 深破止损 (low=80)
    d1 = mkday(20250721, 100.0, drop=(100000,))
    d2 = mkday(20250722, 100.0, {93100: (100, 100, 80, 100, V), 93200: (85, 85, 85, 85, V)})
    r = run_days(E, [(E, mkday(E, 100.0)), (20250721, d1), (20250722, d2)], 3, cal=CAL6)
    check("later stop breach does not recover track", r["status"] == "EXIT_DATA_MISSING" and r["net"] is None, r["status"])
    check("later stop breach not filled (no sells)", len(r["sells"]) == 0, str(r["sells"]))
    check("remaining kept after later breach", r["remaining_shares"] == 400, str(r["remaining_shares"]))
    # T+1 缺 10:00; T+2 强反弹 (high=120) -> 同样无效
    d2b = mkday(20250722, 100.0, {93100: (100, 120, 100, 118, V)})
    rb = run_days(E, [(E, mkday(E, 100.0)), (20250721, d1), (20250722, d2b)], 3, cal=CAL6)
    check("later rebound does not recover track", rb["status"] == "EXIT_DATA_MISSING" and rb["net"] is None, rb["status"])

# ================= ③ 缺槽前已全部退出 -> 标签保留 =================
def g3():
    print("③ 缺槽前已闭环 -> 标签保留")
    E = 20250720
    # T+1 09:31 触发, 09:32 成交清仓; T+2 整日缺失 -> 不影响
    d1 = mkday(20250721, 100.0, {93100: (100, 100, 94, 100, V), 93200: (96, 96, 96, 96, V)})
    r = run_days(E, [(E, mkday(E, 100.0)), (20250721, d1)], 3, cal=CAL6)  # T+2 不在 mbd
    check("closed before later missing day -> SIM_COMPLETE", r["status"] == "SIM_COMPLETE", r["status"])
    check("closed: net kept", r["net"] is not None and r["fully_exited"], str(r["net"]))
    check("closed: label_end_ts set", r["label_end_ts"] is not None, str(r["label_end_ts"]))
    # 计划退出日仅缺 14:57-15:00 尾部 (集合竞价/盘后, 不在本轮成交模型) -> 不否定
    d3 = mkday(20250723, 100.0, {145600: (105, 105, 105, 105, V)}, drop=(145700, 145800, 145900, 150000))
    r2 = run_days(E, [(E, mkday(E, 100.0)), (20250721, mkday(20250721, 100.0)),
                      (20250722, mkday(20250722, 100.0)), (20250723, d3)], 3, cal=CAL6)
    check("tail-only gap (14:57-15:00) does not negate closed trade",
          r2["status"] == "SIM_COMPLETE" and r2["net"] is not None, r2["status"])

# ============ ④ 部分卖出后遇缺槽 -> 残仓保留, 净收益 NULL ============
def g4():
    print("④ 部分卖出后遇缺槽 -> 残仓/净收益NULL")
    E = 20250720
    # 计划日 T+1 14:56 低量 (vol=20000 -> q_cap=200) 部分卖 200/400; T+2 首 bar(09:31)缺槽
    d1 = mkday(20250721, 100.0, {145600: (105, 105, 105, 105, 20000)})
    d2 = mkday(20250722, 100.0, drop=(93100,))
    r = run_days(E, [(E, mkday(E, 100.0)), (20250721, d1), (20250722, d2)], 1, cal=CAL6)
    check("partial sell then gap -> EXIT_DATA_MISSING", r["status"] == "EXIT_DATA_MISSING", r["status"])
    check("partial: remaining_shares kept (200, shares not cash)", r["remaining_shares"] == 200, str(r["remaining_shares"]))
    check("partial: realized C_sell recorded", r["C_sell"] > 0, str(r["C_sell"]))
    check("partial: net NULL", r["net"] is None and r["pnl"] is None, str(r["net"]))
    check("partial: label_end_ts NULL", r["label_end_ts"] is None, str(r["label_end_ts"]))
    check("partial: last_fill_ts recorded", r["last_fill_ts"] is not None, str(r["last_fill_ts"]))

# ================= ⑤ 入场等待期间缺槽 -> 不能继续成交 =================
def g5():
    print("⑤ 入场等待期间缺槽 -> 入场NULL")
    E = 20250710
    cal = [E + i for i in range(6)]
    daily = mkdaily("600001", E, 100.0)
    def ev(drop):
        mE = mkday(E, 100.0, drop=drop)
        mbd = {d: mkday(d, 100.0) for d in cal}
        mbd[E] = mE
        return D.execute_event("600001", E, "SH_MAIN", None, "A_SDP",
                               signal_ts=20250710095600, minute_by_day=mbd,
                               daily_df=daily, cal=cal)
    # 信号 09:56; 09:57 缺槽; 09:58 可成交 -> 不得跳过后成交
    r = ev((95700,))
    check("entry-window gap -> ENTRY_DATA_MISSING", r.get("status") == "ENTRY_DATA_MISSING", str(r.get("status")))
    check("entry kept NULL (no SIM_FILLED)", r.get("entry", {}).get("status") == "ENTRY_DATA_UNKNOWN" or
          r.get("entry", {}).get("status") == "ENTRY_DATA_MISSING", str(r.get("entry")))
    check("no H3/H5 execution after blocked entry", "H3" not in r or r["H3"].get("sim_fill") is None,
          str(r.get("H3", {}).get("sim_fill")))
    # 对照: 完整日 -> 正常成交在 09:57
    r2 = ev(())
    check("no-gap control fills at 09:57", r2.get("status") == "EXECUTED"
          and r2["entry"]["sim_fill_ts"] % 1000000 == 95700, str(r2.get("status")))
    check("control entry price base", abs(r2["entry"]["sim_fill_price"] - 100.0) < 1e-9,
          str(r2["entry"]["sim_fill_price"]))
    # 钉: 零量槽不是缺槽 (不默认停牌) -> 跳过零量槽在 09:58 成交
    mE3 = mkday(E, 100.0, {95700: (100, 100, 100, 100, 0.0)})
    mbd3 = {d: mkday(d, 100.0) for d in cal}; mbd3[E] = mE3
    r3 = D.execute_event("600001", E, "SH_MAIN", None, "A_SDP",
                         signal_ts=20250710095600, minute_by_day=mbd3,
                         daily_df=daily, cal=cal)
    check("zero-vol slot is not a gap (skip, fill later)", r3.get("status") == "EXECUTED"
          and r3["entry"]["sim_fill_ts"] % 1000000 == 95800, str(r3.get("entry", {}).get("sim_fill_ts")))
    # 钉: 完成 bar 之后的 E 日缺槽不破坏入场前缀 (14:00 缺槽, 09:57 成交)
    mE4 = mkday(E, 100.0, drop=(140000,))
    mbd4 = {d: mkday(d, 100.0) for d in cal}; mbd4[E] = mE4
    r4 = D.execute_event("600001", E, "SH_MAIN", None, "A_SDP",
                         signal_ts=20250710095600, minute_by_day=mbd4,
                         daily_df=daily, cal=cal)
    check("post-fill E-day gap does not break entry prefix", r4.get("status") == "EXECUTED"
          and r4["entry"]["sim_fill_ts"] % 1000000 == 95700, str(r4.get("status")))

# ================= ⑥ H3 闭环 / H5 未闭环 -> 分别记录 =================
def g6():
    print("⑥ H3闭环/H5未闭环 独立记录")
    E = 20250720
    cal = [20250720 + i for i in range(8)]
    daily = mkdaily("600002", E, 100.0)
    # 日线须覆盖扫描窗内全部交易日 (is_st/pre_close 按日派生)
    rows = [{"date": d, "code": "600002", "open": 100.0, "high": 102.0, "low": 98.0,
             "close": 100.0, "volume": 1e6, "pre_close": 100.0, "amount": 1e8, "is_st": 0}
            for d in cal]
    daily = pd.concat([daily, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("date")
    mbd = {d: mkday(d, 100.0) for d in cal}
    mbd[20250725] = mkday(20250725, 100.0, drop=(100000,))  # E+5 缺槽 -> H5 NULL
    r = D.execute_event("600002", E, "SH_MAIN", None, "B_LVB",
                        signal_ts=20250720095600, minute_by_day=mbd, daily_df=daily, cal=cal)
    h3, h5 = r.get("H3") or {}, r.get("H5") or {}
    check("H3 complete kept", h3.get("status") == "SIM_COMPLETE" and h3.get("net") is not None, h3.get("status"))
    check("H5 nulled independently", h5.get("status") == "EXIT_DATA_MISSING" and h5.get("net") is None, h5.get("status"))
    check("H3 not deleted by H5 failure", h3.get("fully_exited") is True and h3.get("label_end_ts") is not None,
          str(h3.get("label_end_ts")))
    check("H5 remaining kept", h5.get("remaining_shares") == 400, str(h5.get("remaining_shares")))

# ================= ⑦ 跨日 / 午休 / 241-242 网格版本边界 =================
def g7():
    print("⑦ 跨日/午休/网格版本边界")
    E = 20250720
    # 7a 跨日: T+1 跳空触发但整日跌停锁死卖不出; T+2 首 bar(09:31)缺槽 -> NULL, 缺失日记 T+2
    lower = D._round2(100.0 * 0.90)
    d1 = mkday(20250721, 100.0, {**{s: (lower, lower, lower, lower, V) for s in D.day_grid(20250721)},
                                 93000: (90, 90, 90, 90, V), 93100: (90, 90, 90, 90, V)})  # 09:30/09:31 跳空触发
    d2 = mkday(20250722, 100.0, drop=(93100,))
    r = run_days(E, [(E, mkday(E, 100.0)), (20250721, d1), (20250722, d2)], 3, cal=CAL6)
    dep_day = [m.get("date") for m in r["missing_dependencies"] if m.get("type") == "day_gap"]
    check("cross-day gap -> NULL with dep on gap day", r["status"] == "EXIT_DATA_MISSING" and dep_day == [20250722],
          f"{r['status']} deps={r['missing_dependencies']}")
    # 7b 午休: 241 日完整 (无 13:00 期望) -> 正常; 242 日含 13:00 完整 -> 正常
    E241, E242 = 20260623, 20260701
    check("241 grid expects no 13:00", 130000 not in D.day_grid(E241) and 130000 in D.day_grid(E242),
          f"241={len(D.day_grid(E241))} 242={len(D.day_grid(E242))}")
    calB = [E241, E241 + 1, E241 + 2]
    rB = run_days(E241, [(E241, mkday(E241, 100.0)), (E241 + 1, mkday(E241 + 1, 100.0)),
                         (E241 + 2, mkday(E241 + 2, 100.0, {145600: (105, 105, 105, 105, V)}))], 2, cal=calB)
    check("full 241-day (lunch adjacency 11:30->13:01) no false gap",
          rB["status"] == "SIM_COMPLETE", rB["status"])
    calC = [E242, E242 + 1, E242 + 2]
    rC = run_days(E242, [(E242, mkday(E242, 100.0)), (E242 + 1, mkday(E242 + 1, 100.0)),
                         (E242 + 2, mkday(E242 + 2, 100.0, {145600: (105, 105, 105, 105, V)}))], 2, cal=calC)
    check("full 242-day (with 13:00) no false gap", rC["status"] == "SIM_COMPLETE", rC["status"])
    # 242 日仅缺 13:00 -> 必要网格内缺 1 槽 -> NULL
    d242g = mkday(E242 + 1, 100.0, drop=(130000,))
    rC2 = run_days(E242, [(E242, mkday(E242, 100.0)), (E242 + 1, d242g),
                          (E242 + 2, mkday(E242 + 2, 100.0, {145600: (105, 105, 105, 105, V)}))], 2, cal=calC)
    check("242-day missing only 13:00 -> NULL (1-slot equal treatment)",
          rC2["status"] == "EXIT_DATA_MISSING", rC2["status"])
    # 241 日缺 11:30 -> NULL (两版本同一规则)
    d241g = mkday(E241 + 1, 100.0, drop=(113000,))
    rB2 = run_days(E241, [(E241, mkday(E241, 100.0)), (E241 + 1, d241g),
                          (E241 + 2, mkday(E241 + 2, 100.0, {145600: (105, 105, 105, 105, V)}))], 2, cal=calB)
    check("241-day missing 11:30 -> NULL", rB2["status"] == "EXIT_DATA_MISSING", rB2["status"])

# ================= ⑧ 终点后截断与扰动不改变已闭环字段 =================
def g8():
    print("⑧ 终点后截断/扰动不改变已闭环字段")
    E = 20250720
    cal = [20250720 + i for i in range(8)]
    mbd = {d: mkday(d, 100.0) for d in cal}
    mbd[20250723] = mkday(20250723, 100.0, {145600: (105, 105, 105, 105, V)})  # plan E+3 清仓
    pre = {d: 100.0 for d in cal}; st = {d: False for d in cal}
    entry = mkentry(E)
    base = D.execute_track(entry, 95.0, "SH_MAIN", st, cal, mbd, 3, pre)
    snap = {k: base.get(k) for k in ("status", "net", "C_sell", "C_buy", "label_end_ts", "last_fill_ts",
                                     "fully_exited", "remaining_shares")}
    # a) 截断: 删除完成日之后的所有日
    mbd_trunc = {d: df for d, df in mbd.items() if d <= 20250723}
    ra = D.execute_track(entry, 95.0, "SH_MAIN", st, cal, mbd_trunc, 3, pre)
    check("truncation after endpoint: fields unchanged",
          all(ra.get(k) == v for k, v in snap.items()), f"{ra['status']} vs {snap['status']}")
    # b) 扰动: 完成日之后 (及完成日 14:56 之后) 价格量极端化
    mbd_pert = dict(mbd)
    for d in cal:
        if d > 20250723:
            mbd_pert[d] = mkday(d, 7.77, base_vol=3.0)
    mbd_pert[20250723] = mkday(20250723, 100.0, {145600: (105, 105, 105, 105, V),
                                                 145700: (1.0, 999.0, 0.5, 888.0, 7.0)})
    rb = D.execute_track(entry, 95.0, "SH_MAIN", st, cal, mbd_pert, 3, pre)
    check("perturbation after endpoint: fields unchanged",
          all(rb.get(k) == v for k, v in snap.items()), f"{rb['status']} net={rb['net']} vs {snap['net']}")
    # c) 完成日成交 bar 之前缺槽仍拒 (扰动不能反向救回): 同日 14:00 缺槽 -> NULL
    mbd_gap = dict(mbd)
    mbd_gap[20250723] = mkday(20250723, 100.0, {145600: (105, 105, 105, 105, V)}, drop=(140000,))
    rc = D.execute_track(entry, 95.0, "SH_MAIN", st, cal, mbd_gap, 3, pre)
    check("gap before completion bar on final day -> NULL", rc["status"] == "EXIT_DATA_MISSING"
          and rc["net"] is None, rc["status"])

def main():
    g1(); g2(); g3(); g4(); g5(); g6(); g7(); g8()
    report = {"gap_regress_v0_1_1": {"pass": PASS, "fail": FAIL, "n_pass": len(PASS),
                                    "n_fail": len(FAIL), "ok": len(FAIL) == 0,
                                    "network_requests": 0}}
    out_dir = HERE / "gcp_round_b_small_sample_v1" / "execution_reconcile_v0_1_1" / "gap_regress"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "synthetic_gap_regress.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n缺槽反例回归: {len(PASS)} PASS / {len(FAIL)} FAIL -> {'ALL PASS' if not FAIL else 'HAS FAIL'}")
    return 1 if FAIL else 0

if __name__ == "__main__":
    raise SystemExit(main())
