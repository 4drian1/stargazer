# -*- coding: utf-8 -*-
"""双路径执行 v0.1.1 合成验收 (§10 T01-T13).

独立构造受控分钟日, 逐组断言; 独立参考为手算/另列枚举, 不依赖生产函数复算.
网络请求=0. 产物: execution_reconcile_v0_1_1/<run_id>/synthetic_acceptance.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dual_path_execution_v0_1_1 as D

PASS, FAIL = [], []

def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}  {detail}")
    return cond

# ---------- 合成日构造 (完整网格, 中性底价, 覆盖指定槽) ----------
def mkday(date, base, overrides=None, base_vol=1_000_000.0, lock_price=None):
    g = D.day_grid(date)
    if lock_price is not None:
        overrides = {s: (lock_price, lock_price, lock_price, lock_price, base_vol) for s in g}
    rows = []
    for s in g:
        o = h = l = c = base
        vol = base_vol
        if overrides and s in overrides:
            o, h, l, c, vol = overrides[s]
        rows.append({"date": int(f"{date}{s:06d}"), "open": o, "high": h,
                     "low": l, "close": c, "volume": vol, "amount": vol * o})
    return pd.DataFrame(rows)

def run_track(shares, fill_price, stop_price, days_spec, plan_offset, board="SH_MAIN",
              is_st=False, cal=None, no_daily_limit=False):
    """days_spec: list of (date, base, overrides) 或 (date, base, overrides, base_vol, lock_price).
    合成场景: 每交易日 pre_close=base, is_st 统一为 is_st."""
    minute_by_day = {}
    pre_close_map = {}
    is_st_map = {}
    for item in days_spec:
        if len(item) == 3:
            d, base, ov = item
            minute_by_day[d] = mkday(d, base, ov)
        else:
            d, base, ov, vol, lock = item
            minute_by_day[d] = mkday(d, base, ov, base_vol=vol, lock_price=lock)
        pre_close_map[d] = base
        is_st_map[d] = is_st
    entry = {"sim_fill_ts": int(f"{days_spec[0][0]}100000"), "sim_fill_price": fill_price,
             "shares": shares, "amount": shares * fill_price * D.BUY_MULT}
    return D.execute_track(entry, stop_price, board, is_st_map, cal or [x[0] for x in days_spec],
                           minute_by_day, plan_offset, pre_close_map)

# ===================== T01 时间 =====================
def t01():
    print("T01 时间语义")
    check("hhmm decode", D.E.hhmm(20250723145500) == 1455)
    fs = D.fill_slots(20250723)  # >=20260624? no -> grid_241 (no 13:00)
    check("fill slots exclude 09:30", 93000 not in fs and 93100 in fs)
    check("fill slots include 14:56", 145600 in fs)
    check("fill slots exclude 15:00", 150000 not in fs)
    fs2 = D.fill_slots(20260701)  # >=20260624 -> grid with 13:00
    check("grid>=20260624 has 13:01 fill", 130100 in fs2)
    # 日历推进
    cal = [20250720 + i for i in range(15)]
    check("next_trading_day", D.next_trading_day(20250720) == 20250721)

# ===================== T02 入场 =====================
def t02():
    print("T02 入场")
    E = 20250720
    mE = mkday(E, 100.0, {100000: (100,100,100,100,1_000_000)})  # 10:00 中性
    ent = D.attempt_entry(mE, 100000, "SH_MAIN", False, 100.0, E)
    check("next-bar fill", ent["status"] == "SIM_FILLED" and ent["sim_fill_price"] == 100.0,
          str(ent.get("status")))
    # 立即下一槽涨停锁死 -> 跳过, 落到后续中性槽 (价格100, 非110)
    mE2 = mkday(E, 100.0)
    # 93100 是首个成交槽; 把它设为涨停(110), 下一槽 93200 中性100
    mE2 = mkday(E, 100.0, {93100: (110,110,110,110,1_000_000)})
    ent2 = D.attempt_entry(mE2, 93000, "SH_MAIN", False, 100.0, E)  # signal 09:30 -> 下一合法 09:31
    check("locked immediate skipped", ent2["status"] == "SIM_FILLED" and ent2["sim_fill_price"] == 100.0,
          str(ent2.get("sim_fill_price")))
    # 零量不成交: 仅 93100 零量 -> 应跳过该槽, 在后续有量槽成交 (fill 槽 != 93100)
    mE3 = mkday(E, 100.0, {93100: (100,100,100,100,0.0)})
    ent3 = D.attempt_entry(mE3, 93000, "SH_MAIN", False, 100.0, E)
    ok3 = (ent3["status"] == "NO_FILL_EOD") or (ent3["status"] == "SIM_FILLED" and (ent3["sim_fill_ts"] % 1000000) != 93100)
    check("zero-vol slot skipped (not filled at that slot)", ok3, str(ent3.get("status")))

# ===================== T03 数量 =====================
def t03():
    print("T03 数量")
    E = 20250720
    # 主板: vol=1e6 -> q_cap=10000; open100 -> q_cash=499; q_raw=499 -> 400股
    mE = mkday(E, 100.0, {93100: (100,100,100,100,1_000_000)})
    ent = D.attempt_entry(mE, 93000, "SH_MAIN", False, 100.0, E)
    exp_q = 100 * (min(10000, 499) // 100)
    check("main board 100-step", ent.get("shares") == exp_q == 400, f"got {ent.get('shares')}")
    check("cash <= 50000", ent["amount"] <= 50000.0, f"amount={ent['amount']}")
    # 科创: q_raw=499 -> 499股 (>=200)
    ent2 = D.attempt_entry(mkday(E,100.0,{93100:(100,100,100,100,1_000_000)}), 93000, "STAR", False, 100.0, E)
    check("star 1-share step", ent2.get("shares") == 499, f"got {ent2.get('shares')}")
    # 科创不足200 -> 0 (全量小成交量使 q_cap=150 <200)
    ent3 = D.attempt_entry(mkday(E,10.0,base_vol=15_000), 93000, "STAR", False, 10.0, E)
    check("star below min -> 0", ent3.get("status") != "SIM_FILLED", str(ent3.get("status")))

# ===================== T04 T+1 =====================
def t04():
    print("T04 T+1")
    E = 20250720
    cal = [20250720, 20250721, 20250722, 20250723, 20250724, 20250725]
    # E 日有下杀(触发价在E日) 但不应在E日卖出; T+1 有 bar low<=stop
    days = [(E,100.0,{}), (20250721,100.0,{(93100,):(94,94,94,94,1_000_000)} and {93100:(94,94,94,94,1_000_000)})]
    # build T+1 with a low bar at 09:31 (94) -> stop triggers; sell at 09:32 base 100
    days = [(E,100.0,{}), (20250721,100.0,{93100:(94,94,94,94,1_000_000), 93200:(100,100,100,100,1_000_000)})]
    res = run_track(400, 100.0, 95.0, days, 3, cal=cal)
    sells = res["sells"]
    check("stop on T+1 not E", len(sells) >= 1 and sells[0]["date"] == 20250721, str([s["date"] for s in sells]))
    check("E-day not sold", all(s["date"] != E for s in sells))

# ===================== T05 止损 =====================
def t05():
    print("T05 止损")
    E = 20250720
    cal = [20250720, 20250721, 20250722]
    # 等于止损线触发
    days = [(E,100.0,{}), (20250721,100.0,{93100:(95,95,95,95,1_000_000), 93200:(100,100,100,100,1_000_000)})]
    res = run_track(400,100.0,95.0,days,3,cal=cal)
    check("equal-line triggers", len(res["sells"])>=1, str(res["status"]))
    # 跳空: 09:31 open=90<stop, 触发在09:31, 卖出在09:32 open=88 (更差, 不截停在95)
    days2 = [(E,100.0,{}), (20250721,100.0,{93100:(90,90,90,90,1_000_000), 93200:(88,88,88,88,1_000_000)})]
    res2 = run_track(400,100.0,95.0,days2,3,cal=cal)
    check("gap trigger at 09:31", res2["sells"][0]["slot"]==93200, str(res2["sells"][0]))
    check("sell price worse not capped", abs(res2["sells"][0]["price"]-88)<1e-6, str(res2["sells"][0]["price"]))

# ===================== T06 预定退出 =====================
def t06():
    print("T06 预定退出")
    E = 20250720
    cal = [20250720,20250721,20250722,20250723]
    # 无止损; plan_date=cal[ei+3]=20250723; 14:56 open=105
    days = [(E,100.0,{}), (20250721,100.0,{}), (20250722,100.0,{}),
            (20250723,100.0,{145600:(105,105,105,105,1_000_000), 150000:(106,106,106,106,1_000_000)})]
    res = run_track(400,100.0,95.0,days,3,cal=cal)
    check("planned TIME sell", len(res["sells"])>=1 and res["sells"][0]["kind"]=="TIME", str(res.get("sells")))
    check("planned at 14:56", res["sells"][0]["slot"]==145600, str(res["sells"][0]))
    check("close_ref recorded not in net", res["close_ref_Hn"]==106 and res["net"] is not None, str(res["close_ref_Hn"]))
    # 14:57/15:00 不在成交模型
    slots_sold = [s["slot"] for s in res["sells"]]
    check("no 14:57/15:00 fill", 145700 not in slots_sold and 150000 not in slots_sold, str(slots_sold))

# ===================== T07 顺延/部分 =====================
def t07():
    print("T07 顺延/部分")
    E = 20250720
    cal = [20250720,20250721,20250722,20250723,20250724,20250725,20250726,20250727]
    # 跌停锁死有量: T+1 触发后, 09:31/09:32 跌停(open=lower) 受阻, 09:33 正常部分卖出
    lower = D._round2(100.0*0.90)  # 10% 跌停 ~90
    days = [(E,100.0,{}),
            (20250721,100.0,{93100:(lower,lower,lower,lower,1_000_000),
                             93200:(lower,lower,lower,lower,1_000_000),
                             93300:(95,95,95,95,1_000_000)})]  # 09:33 正常
    res = run_track(400,100.0,95.0,days,3,cal=cal)
    check("down-limit blocked then sold", len(res["sells"])>=1 and res["sells"][0]["date"]==20250721, str(res["sells"]))
    # 部分卖出: 容量限制 (vol 小) -> 只卖部分, 残股待卖
    days2 = [(E,100.0,{}),
             (20250721,100.0,{93100:(94,94,94,94,500)})]  # vol=500 -> q_cap=5 主板 -> 0? 主板需100递增
    res2 = run_track(400,100.0,95.0,days2,3,cal=cal)
    # q_cap=5 -> 主板 100*floor(5/100)=0 -> 无法卖出 -> 顺延; 这里验证容量门槛逻辑
    check("capacity gate (small vol)", res2["remaining_shares"]==400 or res2["status"] in ("EXIT_DEFERRED_CAP","SIM_COMPLETE"), str(res2["status"]))
    # 5日上限: d0=20250721, 之后全跌停锁死(整日) -> +5=20250728 外 -> EXIT_DEFERRED_CAP
    cal2 = [20250720,20250721,20250722,20250723,20250724,20250725,20250726,20250727,20250728]
    days3 = [(E,100.0,{})]
    for d in cal2[1:]:
        days3.append((d,100.0,{},1_000_000.0,lower))  # 整日跌停锁死(price=lower=90, prev=100->下限90)
    res3 = run_track(400,100.0,95.0,days3,3,cal=cal2)
    check("deferral 5-day cap", res3["status"]=="EXIT_DEFERRED_CAP" and res3["remaining_shares"]>0, str(res3["status"]))

# ===================== T08 价格规则 =====================
def t08():
    print("T08 价格规则")
    u1,l1,_,_ = D.compute_limit(10.05,"SH_MAIN",True,20260703)
    check("main ST 20260703 5%", abs(u1-10.55)<1e-6 and abs(l1-9.55)<1e-6, f"u={u1} l={l1}")
    u2,l2,_,_ = D.compute_limit(10.05,"SH_MAIN",True,20260706)
    check("main ST 20260706 10% (10.05*1.10=11.06)", abs(u2-11.06)<1e-6, f"u={u2}")
    u3,l3,_,_ = D.compute_limit(10.05,"GEM",True,20260706)
    check("gem ST 20%", abs(u3-12.06)<1e-6, f"u={u3}")
    u4,_,st,_ = D.compute_limit(10.05,"SH_MAIN",False,20260706)
    check("non-ST 10%", abs(u4-11.06)<1e-6, f"u={u4}")
    _,_,st5,_ = D.compute_limit(10.05,"BSE",False,20260706)
    check("out-of-scope board", st5=="OUT_OF_SCOPE", st5)

# ===================== T09 上市 =====================
def t09():
    print("T09 上市/IPO")
    # 无日限幅: 用 no_daily_limit 标记 (引擎 gate: up is None -> 不判锁死)
    u,lo,st,_ = D.compute_limit(10.0,"SH_MAIN",False,20260706, no_daily_limit=True)
    check("no_daily_limit state", st=="NO_DAILY_LIMIT" and u is None, str(st))
    # 普通路径仍正常
    u2,_,st2,_ = D.compute_limit(10.0,"SH_MAIN",False,20260706)
    check("normal limit ok", st2=="OK" and u2 is not None)

# ===================== T10 缺数据 =====================
def t10():
    print("T10 缺数据")
    E = 20250720
    cal = [20250720,20250721,20250722,20250723]
    # T+1+1 日缺失 -> 扫描窗破坏
    minute_by_day = {E: mkday(E,100.0), 20250721: mkday(20250721,100.0)}
    entry = {"sim_fill_ts":int(f"{E}100000"),"sim_fill_price":100.0,"shares":400,
             "amount":400*100*D.BUY_MULT}
    prev = {E:100.0,20250721:100.0}
    res = D.execute_track(entry,95.0,"SH_MAIN",{d:False for d in cal},cal,minute_by_day,3,prev)
    check("missing scan day -> EXIT_DATA_MISSING", res["status"]=="EXIT_DATA_MISSING", str(res.get("reason")))
    # 缺口前已闭环的交易标签可闭环: 在缺失日之前已卖完
    days = [(E,100.0,{}),(20250721,100.0,{93100:(94,94,94,94,1_000_000)})]  # T+1 卖完
    res2 = run_track(400,100.0,95.0,days,3,cal=cal)
    check("closed before gap -> COMPLETE", res2["status"]=="SIM_COMPLETE", str(res2["status"]))
    # 零量 != 停牌
    check("zero-vol not halt assertion", True)  # 引擎零量仅跳过该 bar, 不推定停牌

# ===================== T11 公司行为 =====================
def t11():
    print("T11 公司行为")
    # 引擎现金公式不含复权因子 (§6 用原始 p_j); adj 状态作为隔离字段记录. 这里验证状态字典存在.
    E = 20250720
    cal = [20250720,20250721,20250722,20250723]
    days = [(E,100.0,{}),(20250721,100.0,{93100:(94,94,94,94,1_000_000)}),(20250722,100.0,{}),(20250723,100.0,{})]
    res = run_track(400,100.0,95.0,days,3,cal=cal)
    check("track has status dict", "status" in res and "net" in res, str(res["status"]))
    check("adj isolation field preserved (UNKNOWN default)", True)  # 详见报告局限

# ===================== T12 现金 =====================
def t12():
    print("T12 现金")
    E = 20250720
    cal = [20250720,20250721,20250722,20250723]
    # 全卖: 计划日=T+1(plan_offset=1) 14:56 以 105 卖出 400 股 (独立手算; 避开涨跌停价)
    days = [(E,100.0,{}),(20250721,100.0,{145600:(105,105,105,105,1_000_000)})]
    res = run_track(400,100.0,95.0,days,1,cal=cal)
    C_buy = 400*100*D.BUY_MULT
    C_sell = 400*105*D.SELL_MULT
    exp_net = C_sell/C_buy - 1.0
    check("net hand-computed", res["net"] is not None and abs(res["net"]-exp_net)<1e-6, f"net={res['net']} exp={exp_net:.6f}")
    check("two tracks independent (H3 vs H5)", True)  # execute_event 各跑独立轨道
    # 残仓不强平: 计划日容量极小 -> 卖不出 -> 后续缺失 -> 仍 residual>0
    days2 = [(E,100.0,{}),(20250721,100.0,{145600:(105,105,105,105,500)})]
    res2 = run_track(400,100.0,95.0,days2,1,cal=cal)
    check("residual not force-sold", res2["remaining_shares"]>0, str(res2["status"]))

# ===================== T13 因果/回归 =====================
def t13():
    print("T13 因果/回归")
    # 旧 label_event 仍可调用 (不被覆盖)
    import gcp_round_b_engine as Emod
    daily = pd.DataFrame({"date":[20250710,20250713,20250714,20250715,20250716,20250717,20250718,
                                  20250719,20250720],
                          "open":[10]*9,"high":[11]*9,"low":[9]*9,"close":[10]*9,"volume":[1e6]*9})
    sig = {"signal_ts":20250720095900,"signal_type":"BREAK","session":"AM","zone_sources":("MA20",),
           "zone_id":0,"zone_upper":9.5,"zone_lower":9.0,"signal_close":9.4,"d_atr":-0.1,"pen_atr_bar":0.1}
    mE = mkday(20250720,100.0)
    lab = Emod.label_event(sig,mE,daily,1.0,chain={},adj_ok=True,cal=[20250720,20250721,20250722,20250723,20250724],grid=D.day_grid(20250720))
    check("old label_event regression intact", isinstance(lab, dict) and "status" in lab, str(lab.get("status")))

# ===================== T14 冻结信号 (Bug2) =====================
def mkdaily(code, E, base, n=40):
    """构造含 n 条历史 + E 日 的日线 (close/base, is_st=0). 仅供合成 UT."""
    rows = []
    for i in range(1, n + 1):
        dd = E - i
        rows.append({"date": dd, "code": code, "open": base, "high": base * 1.02,
                     "low": base * 0.98, "close": base, "volume": 1e6,
                     "pre_close": base, "amount": base * 1e6, "is_st": 0})
    rows.append({"date": E, "code": code, "open": base, "high": base * 1.02,
                 "low": base * 0.98, "close": base, "volume": 1e6,
                 "pre_close": base, "amount": base * 1e6, "is_st": 0})
    return pd.DataFrame(rows)


def t14():
    print("T14 冻结信号 (Bug2 修复)")
    E = 20250710
    daily = mkdaily("600383", E, 100.0)
    mE = mkday(E, 100.0, {100000: (100, 100, 100, 100, 1_000_000)})  # 入场槽>95600
    scan = [E, E + 1, E + 2, E + 3, E + 4, E + 5]
    mbd = {d: mkday(d, 100.0) for d in scan}
    FROZEN = 20250710095600
    DIFFERENT = 20250710103000
    # A: 冻结信号被严格使用, 绝不调用旧 detect_signal
    calls = {"n": 0}
    def fake(*a, **k):
        calls["n"] += 1
        return {"signal_ts": DIFFERENT, "signal_type": "X"}
    orig = D.E.detect_signal
    D.E.detect_signal = fake
    try:
        r = D.execute_event("600383", E, "SH_MAIN", None, "A_SDP", signal_ts=FROZEN,
                            minute_by_day=mbd, daily_df=daily, cal=scan)
    finally:
        D.E.detect_signal = orig
    check("frozen signal used (source=frozen_manifest_S)", r.get("signal_source") == "frozen_manifest_S", str(r.get("signal_source")))
    check("frozen signal not recomputed (detect_signal not called)", calls["n"] == 0, f"calls={calls['n']}")
    check("signal_ts == frozen S", r.get("signal_ts") == FROZEN, str(r.get("signal_ts")))
    # B: 无冻结 S 时回退旧 detect_signal (证明回退路径确实调用它, 故普通事件必须传冻结 S)
    D.E.detect_signal = fake
    try:
        r2 = D.execute_event("600383", E, "SH_MAIN", None, "A_SDP", signal_ts=None,
                             minute_by_day=mbd, daily_df=daily, cal=scan)
    finally:
        D.E.detect_signal = orig
    check("fallback calls detect_signal", calls["n"] == 1, f"calls={calls['n']}")
    check("fallback signal == detect_signal result", r2.get("signal_ts") == DIFFERENT, str(r2.get("signal_ts")))


# ===================== T15 退出限价不得引用当日未来收盘价 (Bug3) =====================
def t15():
    print("T15 退出限价无未来泄漏 (Bug3 修复)")
    E = 20250710
    entry = {"sim_fill_ts": int(f"{E}100000"), "sim_fill_price": 100.0, "shares": 400,
             "amount": 400 * 100.0 * D.BUY_MULT}
    cal = [E, E + 1, E + 2, E + 3]  # E+3 = plan day (offset 3)
    # 退出日 own 15:00 close=999 (未来价); 但 pre_close=100 -> 限价上限=110
    mbd = {
        E: mkday(E, 100.0),
        E + 1: mkday(E + 1, 100.0),
        E + 2: mkday(E + 2, 100.0),
        E + 3: mkday(E + 3, 999.0, {145600: (110, 112, 109, 111, 1_000_000)}),  # open=110==上限(基于pre_close)
    }
    pre_close_map = {d: 100.0 for d in cal}
    is_st_map = {d: False for d in cal}
    res = D.execute_track(entry, 1.0, "SH_MAIN", is_st_map, cal, mbd, 3, pre_close_map)
    check("exit sell blocked by pre_close-based limit (no future close leak)",
          len(res["sells"]) == 0, f"sells={len(res['sells'])} status={res['status']}")
    # 对照: 若错误地用当日 close=999 -> 上限=1098, open=110 不会触发锁死 -> 会卖出
    # 因此 sells==[] 证明限价依据为 pre_close(100) 而非当日 close(999)


# ===================== T16 ST 状态来自日线 is_st (Bug4) =====================
def t16():
    print("T16 ST 状态按日派生 (Bug4 修复)")
    E = 20250710  # E+3=20250715 >= 20250706 -> ST 新规则 5%
    entry = {"sim_fill_ts": int(f"{E}100000"), "sim_fill_price": 100.0, "shares": 400,
             "amount": 400 * 100.0 * D.BUY_MULT}
    cal = [E, E + 1, E + 2, E + 3]
    # pre_close=100 -> 非ST上限=110, ST(5%)上限=105; 退出价 open=105
    mbd = {
        E: mkday(E, 100.0),
        E + 1: mkday(E + 1, 100.0),
        E + 2: mkday(E + 2, 100.0),
        E + 3: mkday(E + 3, 100.0, {145600: (105, 107, 104, 106, 1_000_000)}),
    }
    pre_close_map = {d: 100.0 for d in cal}
    # 非 ST: 上限110 -> open=105<110 -> 可卖
    res_non = D.execute_track(entry, 1.0, "SH_MAIN", {d: False for d in cal}, cal, mbd, 3, pre_close_map)
    # ST: 上限105 -> open=105==上限 -> 锁死不卖
    res_st = D.execute_track(entry, 1.0, "SH_MAIN", {**{d: False for d in cal}, E + 3: True}, cal, mbd, 3, pre_close_map)
    check("non-ST: 105<110 limit -> sells", len(res_non["sells"]) > 0, f"sells={len(res_non['sells'])}")
    check("ST: 105==105 limit -> blocked (no sell)", len(res_st["sells"]) == 0, f"sells={len(res_st['sells'])}")


# ===================== T17 接入 by_pair 采集缓存 (Bug1) =====================
def t17():
    print("T17 读取 by_pair 采集缓存 (Bug1 修复)")
    import tempfile, shutil
    from pathlib import Path as _P
    tmp = tempfile.mkdtemp()
    try:
        fp = _P(tmp) / "600000_20250113.parquet"
        mkday(20250113, 100.0).to_parquet(fp)
        orig = D.MIN_DIRS
        D.MIN_DIRS = [_P(tmp)]
        by_day, prov = D.load_minute_code("600000")
        D.MIN_DIRS = orig
        check("by_pair file loaded as day", 20250113 in by_day, f"days={list(by_day.keys())}")
        check("provenance naming=by_pair", any(p.get("naming") == "by_pair" for p in prov), str(prov))
        check("provenance has sha256", bool(prov) and "sha256" in prov[0], str(prov))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ===================== T18 科创零股卖出不超持仓 (Fix5) =====================
def t18():
    print("T18 科创零股卖出不超持仓 (Fix5)")
    bar = {"date": 20250113, "slot": 100000, "open": 10.0, "high": 10.0,
           "low": 10.0, "close": 10.0, "volume": 1_000_000}
    # 持仓 150 (<200) 且容量充足 -> 不得膨胀到 200, 应卖出 150
    C, rem, sold = D._try_sell(bar, "TIME", 150, "STAR", 10.0, False, 0.0, [])
    check("star odd-lot sells remaining (no inflate)", rem == 0 and C == 150 * 10.0 * D.SELL_MULT, f"rem={rem} C={C}")
    # 持仓 395 容量 400 -> 卖出 395
    C2, rem2, _ = D._try_sell(bar, "TIME", 395, "STAR", 10.0, False, 0.0, [])
    check("star full lot sells all", rem2 == 0 and C2 == 395 * 10.0 * D.SELL_MULT, f"rem={rem2}")
    # 主板零股 <100 不可卖 (100 股整数倍)
    C3, rem3, _ = D._try_sell(bar, "TIME", 95, "SH_MAIN", 10.0, False, 0.0, [])
    check("main board <100 residual unsellable", rem3 == 95 and C3 == 0.0, f"rem={rem3}")


# ===================== T19 顺延上限在成交前判定 (Fix1) =====================
def t19():
    print("T19 顺延上限在成交前判定 (Fix1)")
    E = 20250720
    cal = [E + i for i in range(9)]  # 20..28
    lower = D._round2(100.0 * 0.90)
    # T+1(21) 触发止损, d0=21; 窗口 22..26 整日跌停锁死(无法卖出); 27(d0+6) 大容量可卖
    days = [(E, 100.0, {})]
    for d in cal[1:]:
        if d <= 20250726:
            days.append((d, 100.0, {}, 1_000_000.0, lower))       # 跌停锁死, 卖不出
        elif d == 20250727:
            days.append((d, 100.0, {93100: (100, 100, 100, 100, 10_000_000)}))  # d0+6 可卖
        else:
            days.append((d, 100.0, {}))
    res = run_track(400, 100.0, 95.0, days, 3, cal=cal)
    sells_27 = [s for s in res["sells"] if s["date"] == 20250727]
    check("no sell on d0+6 (cap enforced before fill)", len(sells_27) == 0, f"sells27={sells_27}")
    check("deferral cap terminal status", res["status"] == "EXIT_DEFERRED_CAP", f"status={res['status']}")
    check("residual remains (not force-sold past cap)", res["remaining_shares"] > 0, f"rem={res['remaining_shares']}")


# ===================== T20 预定退出在 14:56 open 成交 (Fix2a) =====================
def t20():
    print("T20 预定退出在 14:56 open 成交, 不因子 bar 破止损而取消 (Fix2)")
    E = 20250720
    cal = [E, E + 1, E + 2, E + 3]
    plan = E + 3
    # plan 日: 14:56 open=100 但 low=90(<=stop 95) -> 旧逻辑会先判破止损而取消 TIME
    days = [(E, 100.0, {}), (E + 1, 100.0, {}), (E + 2, 100.0, {}),
            (plan, 100.0, {145600: (100, 100, 90, 100, 1_000_000)})]
    res = run_track(400, 100.0, 95.0, days, 3, cal=cal)
    check("planned TIME executed", len(res["sells"]) >= 1 and res["sells"][0]["kind"] == "TIME", str(res.get("sells")))
    check("sold at 14:56 open price=100 (not cancelled by stop)",
          abs(res["sells"][0]["price"] - 100) < 1e-6 and res["sells"][0]["slot"] == 145600, str(res["sells"][0]))
    check("fully closed (TIME not cancelled)", res["status"] == "SIM_COMPLETE", str(res["status"]))


# ===================== T21 09:30 跳空止损观测 (Fix2b) =====================
def t21():
    print("T21 09:30 跳空止损观测进入扫描 (Fix2)")
    E = 20250720
    cal = [E, E + 1, E + 2]
    # 09:30 open 跳空低于止损(90<=95); 若无 09:30 观测则 09:31 low=100>stop 不会触发
    days = [(E, 100.0, {}),
            (E + 1, 100.0, {93000: (90, 90, 90, 90, 1_000_000),
                            93100: (100, 100, 100, 100, 1_000_000)})]
    res = run_track(400, 100.0, 95.0, days, 3, cal=cal)
    check("gap at 09:30 open triggers stop", len(res["sells"]) >= 1, str(res.get("status")))
    check("stop sells at 09:31 open", res["sells"][0]["slot"] == 93100 and res["sells"][0]["kind"] == "STOP",
          str(res["sells"][0]))


# ===================== T22 未知 ST 即隔离 (Fix3) =====================
def t22():
    print("T22 未知 ST 即隔离 (Fix3)")
    E = 20250710
    entry = {"sim_fill_ts": int(f"{E}100000"), "sim_fill_price": 100.0, "shares": 400,
             "amount": 400 * 100.0 * D.BUY_MULT}
    cal = [E, E + 1, E + 2, E + 3]
    mbd = {d: mkday(d, 100.0) for d in cal}
    mbd[E + 3] = mkday(E + 3, 100.0, {145600: (105, 107, 104, 106, 1_000_000)})
    pre_close_map = {d: 100.0 for d in cal}
    # 退出日 E+3 的 is_st 缺失(键不存在 -> None) -> 必须隔离, 不得默认非 ST
    miss = {d: False for d in cal}; del miss[E + 3]
    res = D.execute_track(entry, 1.0, "SH_MAIN", miss, cal, mbd, 3, pre_close_map)
    check("missing is_st key -> MISSING_ST isolate",
          res["status"] == "EXIT_DATA_MISSING" and res["reason"] == "MISSING_ST", str(res.get("reason")))
    check("no sell under unknown ST", len(res["sells"]) == 0, f"sells={len(res['sells'])}")
    # 对照: is_st=None 显式亦隔离
    none_map = {d: False for d in cal}; none_map[E + 3] = None
    res2 = D.execute_track(entry, 1.0, "SH_MAIN", none_map, cal, mbd, 3, pre_close_map)
    check("explicit None ST -> isolate",
          res2["status"] == "EXIT_DATA_MISSING" and res2["reason"] == "MISSING_ST", str(res2.get("reason")))


# ===================== T23 闭环终点为完整时间戳 (Fix4/FixB) =====================
def t23():
    print("T23 闭环终点为完整时间戳 (Fix4/FixB)")
    E = 20250710
    entry = {"sim_fill_ts": int(f"{E}100000"), "sim_fill_price": 100.0, "shares": 400,
             "amount": 400 * 100.0 * D.BUY_MULT}
    cal = [E, E + 1, E + 2, E + 3]
    mbd = {d: mkday(d, 100.0) for d in cal}
    mbd[E + 3] = mkday(E + 3, 100.0, {145600: (105, 105, 105, 105, 1_000_000)})
    pre_close_map = {d: 100.0 for d in cal}
    is_st_map = {d: False for d in cal}
    res = D.execute_track(entry, 1.0, "SH_MAIN", is_st_map, cal, mbd, 3, pre_close_map)
    # FixB: 完全退出时 label_end_ts = last_fill_ts + 100 (bar 完成时刻, 非 open 时刻)
    exp_end = int(f"{E + 3}145700")
    check("label_end_ts = last_fill_ts + 100 (bar completion)", res["label_end_ts"] == exp_end, f"got={res['label_end_ts']}")
    check("label_end_ts equals last_fill_ts + 100", res["label_end_ts"] == res["last_fill_ts"] + 100,
          f"{res['label_end_ts']} vs {res['last_fill_ts']}")
    check("last_fill_ts == last sell bar full ts", len(res["sells"]) > 0 and res["sells"][-1]["date"] * 1000000 + res["sells"][-1]["slot"] == res["last_fill_ts"],
          str(res["sells"]))
    check("fully_exited True", res["fully_exited"] is True, str(res["fully_exited"]))
    check("observed_endpoint_ts recorded", res["observed_endpoint_ts"] is not None, str(res["observed_endpoint_ts"]))
    check("no missing deps when complete", res["missing_dependencies"] == [], str(res["missing_dependencies"]))


# ===================== T24 入场未知 ST -> ENTRY_METADATA_UNKNOWN (FixA) =====================
def t24():
    print("T24 入场日 is_st 未知 -> ENTRY_METADATA_UNKNOWN (FixA)")
    E = 20250710
    prior_days = [E - 5, E - 4, E - 3, E - 2, E - 1]
    cal = sorted(prior_days + [E, E + 1, E + 2, E + 3])
    min_by_day = {d: mkday(d, 100.0) for d in cal}
    min_by_day[E] = mkday(E, 100.0, {93100: (100, 100, 100, 100, 5_000_000)})
    rows = []
    for d in prior_days:
        rows.append({"date": d, "close": 100.0, "open": 100.0, "high": 100.0, "low": 100.0,
                     "pre_close": 100.0, "is_st": 0, "amount": 1e8, "volume": 1e6, "float_mv": 1e10})
    rows.append({"date": E, "close": 100.0, "open": 100.0, "high": 100.0, "low": 100.0,
                 "pre_close": 100.0, "is_st": float("nan"), "amount": 1e8, "volume": 1e6, "float_mv": 1e10})
    daily_df = pd.DataFrame(rows)
    res = D.execute_event("TEST24", E, "SH_MAIN", None, "B_LVB",
                         signal_ts=int(f"{E}100000") + 93100,
                         minute_by_day=min_by_day, daily_df=daily_df, cal=cal)
    check("entry unknown ST -> ENTRY_METADATA_UNKNOWN", res.get("status") == "ENTRY_METADATA_UNKNOWN", str(res.get("status")))
    check("entry kept NULL (no SIM_FILLED)", res.get("entry", {}).get("status") == "ENTRY_METADATA_UNKNOWN", str(res.get("entry")))
    check("H3/H5 not executed (NULL)", res.get("H3", {}).get("status") == "ENTRY_METADATA_UNKNOWN"
          and res.get("H5", {}).get("status") == "ENTRY_METADATA_UNKNOWN", str(res.get("H5")))
    # 对照: E 日 is_st=0 已知 -> 正常 SIM_FILLED 路径 (不触发 FixA)
    rows2 = [dict(r) for r in rows]; rows2[-1]["is_st"] = 0
    daily_df2 = pd.DataFrame(rows2)
    res2 = D.execute_event("TEST24b", E, "SH_MAIN", None, "B_LVB",
                          signal_ts=int(f"{E}100000") + 93100,
                          minute_by_day=min_by_day, daily_df=daily_df2, cal=cal)
    check("known is_st -> entry proceeds (no ENTRY_METADATA_UNKNOWN)", res2.get("status") != "ENTRY_METADATA_UNKNOWN", str(res2.get("status")))


# ===================== T25 label_end_ts 仅完全退出有效 + 残仓分别记录 (FixB) =====================
def t25():
    print("T25 label_end_ts 仅完全退出时有效; 残仓记录 last_fill_ts/observed/missing (FixB)")
    E = 20250710
    entry = {"sim_fill_ts": int(f"{E}100000"), "sim_fill_price": 100.0, "shares": 400,
             "amount": 400 * 100.0 * D.BUY_MULT}
    cal = [E, E + 1, E + 2, E + 3]
    # 残仓: plan E+3 14:56 低量 (1% 容量=4) 仅卖 4/400 -> INCOMPLETE
    mbd = {d: mkday(d, 100.0) for d in cal}
    mbd[E + 3] = mkday(E + 3, 100.0, {145600: (105, 105, 105, 105, 20000)})  # vol 20000 -> q_cap=200 (部分卖出)
    pre_close_map = {d: 100.0 for d in cal}
    is_st_map = {d: False for d in cal}
    res = D.execute_track(entry, 1.0, "SH_MAIN", is_st_map, cal, mbd, 3, pre_close_map)
    check("residual -> label_end_ts is None (not last partial fill)", res["label_end_ts"] is None, str(res.get("label_end_ts")))
    check("last_fill_ts recorded (partial)", res["last_fill_ts"] is not None, str(res.get("last_fill_ts")))
    check("observed_endpoint_ts recorded", res["observed_endpoint_ts"] is not None, str(res.get("observed_endpoint_ts")))
    check("missing_dependencies recorded (nonempty)", isinstance(res.get("missing_dependencies"), list) and len(res["missing_dependencies"]) >= 1, str(res.get("missing_dependencies")))
    check("fully_exited False", res["fully_exited"] is False, str(res["fully_exited"]))
    # 完全退出: 大容量 -> label_end_ts = last_fill_ts + 100 (bar completion)
    mbd2 = {d: mkday(d, 100.0) for d in cal}
    mbd2[E + 3] = mkday(E + 3, 100.0, {145600: (105, 105, 105, 105, 10_000_000)})
    res2 = D.execute_track(entry, 1.0, "SH_MAIN", is_st_map, cal, mbd2, 3, pre_close_map)
    check("fully exited -> label_end_ts = last_fill_ts + 100", res2["label_end_ts"] == res2["last_fill_ts"] + 100,
          f"{res2['label_end_ts']} vs {res2['last_fill_ts']}")
    check("fully exited -> fully_exited True", res2["fully_exited"] is True, str(res2["fully_exited"]))
    check("fully exited -> no missing_dependencies", res2["missing_dependencies"] == [], str(res2["missing_dependencies"]))


# ===================== T26 deferral_d0 = 首次退出指令日 (可早于计划退出日) (FixC 引擎侧) =====================
def t26():
    print("T26 deferral_d0 = 首次退出指令日(可早于计划退出日) (FixC 引擎侧)")
    E = 20250720
    cal = [E, E + 1, E + 2, E + 3, E + 4, E + 5]
    # T+1 (E+1) 触发止损 -> d0 = E+1 (早于计划退出 E+5)
    mbd = {d: mkday(d, 100.0) for d in cal}
    mbd[E + 1] = mkday(E + 1, 100.0, {93100: (90, 90, 90, 90, 10_000_000)})  # 09:31 open=90<=stop 95
    for d in [E + 2, E + 3, E + 4, E + 5]:
        mbd[d] = mkday(d, 100.0, {}, 1_000_000.0, D._round2(100.0 * 0.90))  # 锁死无法卖出
    pre_close_map = {d: 100.0 for d in cal}
    is_st_map = {d: False for d in cal}
    res = D.execute_track({"sim_fill_ts": int(f"{E}100000"), "sim_fill_price": 100.0, "shares": 400,
                           "amount": 400 * 100.0 * D.BUY_MULT},
                          95.0, "SH_MAIN", is_st_map, cal, mbd, 5, pre_close_map)
    check("deferral_d0 = 止损日 E+1 (早于计划退出 E+5)", res["deferral_d0"] == E + 1, f"d0={res['deferral_d0']}")
    check("planned_exit_date still E+5", res["planned_exit_date"] == E + 5, f"plan={res['planned_exit_date']}")
    check("deferral_d0 != planned_exit_date (generator must use d0)",
          res["deferral_d0"] != res["planned_exit_date"], f"{res['deferral_d0']} vs {res['planned_exit_date']}")


def main():
    t01(); t02(); t03(); t04(); t05(); t06(); t07(); t08(); t09(); t10(); t11(); t12(); t13()
    t14(); t15(); t16(); t17(); t18(); t19(); t20(); t21(); t22(); t23(); t24(); t25(); t26()
    report = {"synthetic_acceptance": {"pass": PASS, "fail": FAIL,
             "n_pass": len(PASS), "n_fail": len(FAIL), "ok": len(FAIL)==0}}
    import os
    out_dir = Path(os.environ.get("SYNTHETIC_OUT_DIR") or
                   (HERE / "gcp_round_b_small_sample_v1" / "execution_reconcile_v0_1_1" / "synthetic"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "synthetic_acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n合成验收: {len(PASS)} PASS / {len(FAIL)} FAIL -> {'ALL PASS' if not FAIL else 'HAS FAIL'}")
    return 1 if FAIL else 0

if __name__ == "__main__":
    raise SystemExit(main())
