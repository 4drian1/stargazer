"""阶段2.1 合成测试 + 因果截断测试。

覆盖：
  - 同bar创新高又跌破（新代码用旧高水位，退出价更保守）
  - 同bar止损与止盈（止损悲观优先）
  - T+1 触发但不可卖（入场日不卖出）
  - 涨停无法买入 -> 顺延入场
  - 跌停锁死 -> 延迟到下一可卖交易日出场
  - 因果截断：t 之后注入极端价格 / 删除 t 之后数据，t 之前状态与输出不变
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd

from build_exit_labels import run_exit, compute_atr_ref

ATR = 0.333  # 使 1.5*ATR/entry=0.04995 -> 止损 9.5005
STOP = 10 * (1 - min(max(1.5 * ATR / 10, 0.03), 0.06))   # = 9.5005
T1D = max(0.01, 0.45 * ATR / 10)                          # 0.014985


def mk_px(rows):
    px = pd.DataFrame(rows, columns=["date", "raw_open", "raw_high", "raw_low", "raw_close"])
    px["date"] = px["date"].astype(int)
    prev = {int(px.iloc[k]["date"]): float(px.iloc[k - 1]["raw_close"]) if k > 0
            else float(px.iloc[k]["raw_close"]) for k in range(len(px))}
    return px, prev


def run(px, prev, sig_idx, variant="X1", atr=ATR):
    return run_exit(px, sig_idx, atr, variant, None, prev, None, "TEST", 20240101, 0)


PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name + ("" if cond else f"  <<< {detail}"))


# ---------- 场景 A：同bar创新高又跌破（移动止盈保守性，仅T1激活） ----------
# entry=10(T+1)。T+1 冲高10.3(峰值3%>=T1阈值2.67%, <T2阈值5%)。
# T+2 低10.1 <= 旧hw10.3*(1-T1D) -> TRAIL1@10.3*(1-T1D)
pxA, prevA = mk_px([
    (20240101, 10, 10, 10, 10),
    (20240102, 10, 10.3, 9.9, 10.3),    # T+1 entry, 高=10.3 -> 激活T1(次bar生效)
    (20240103, 10.3, 10.4, 10.1, 10.2), # T+2 触发TRAIL1, 旧hw=10.3 -> 线=10.3*(1-T1D)
    (20240104, 10.2, 10.3, 10.1, 10.2),
    (20240105, 10.2, 10.3, 10.1, 10.2),
    (20240108, 10.2, 10.3, 10.1, 10.2),
])
rA = run(pxA, prevA, 0, "X1")
# T+1 冲高激活T1，但"新激活次bar生效" -> 移动止盈自 T+3 起生效；
# T+3 低开10.2 已低于该bar止盈线(10.4*(1-T1D)=10.244) -> 以开盘价10.2 出场
expA = 10.2
check("A: TRAIL1 触发(次bar生效)", rA is not None and rA["exit_reason"] == "TRAIL1", rA)
check("A: 退出价=次bar开盘价10.2", rA is not None and abs(rA["exit_price"] - expA) < 1e-6,
      None if rA is None else rA["exit_price"])
check("A: 出场于激活次日 T+3(20240104)", rA is not None and rA["exit_time"] == 20240104,
      None if rA is None else rA["exit_time"])

# ---------- 场景 B：同bar止损与止盈，止损悲观优先 ----------
pxB, prevB = mk_px([
    (20240101, 10, 10, 10, 10),
    (20240102, 10, 10, 9.9, 10),       # T+1 entry
    (20240103, 10, 11.5, 9.4, 9.5),    # T+2 low=9.4 同时击穿止损(9.5005)与止盈线(9.85)
    (20240104, 9.8, 9.9, 9.5, 9.6),
    (20240105, 9.6, 9.7, 9.4, 9.5),
    (20240108, 9.6, 9.7, 9.4, 9.5),
])
rB = run(pxB, prevB, 0, "X1")
check("B: 同bar止损优先(STOP)", rB is not None and rB["exit_reason"] == "STOP", rB)
check("B: 止损价=9.5005", rB is not None and abs(rB["exit_price"] - STOP) < 1e-6, None if rB is None else rB["exit_price"])

# ---------- 场景 C：T+1 触发但不可卖 ----------
# T+1 跌破止损但 can_sell=False；T+2..T+5 开盘/最低均高于止损 -> TIME_EXIT@T+5
pxC, prevC = mk_px([
    (20240101, 10, 10, 10, 10),
    (20240102, 10, 10, 9.0, 9.2),      # T+1 entry 日即跌破止损，但不可卖
    (20240103, 9.6, 9.7, 9.55, 9.7),   # T+2 已回到止损之上(开9.6>9.5005)
    (20240104, 9.6, 9.7, 9.55, 9.7),
    (20240105, 9.6, 9.7, 9.55, 9.7),
    (20240108, 9.6, 9.7, 9.55, 9.7),   # T+5
])
rC = run(pxC, prevC, 0, "X1")
check("C: T+1不卖出 -> 非入场日出场", rC is not None and rC["exit_time"] != 20240102, rC)
check("C: TIME_EXIT@T+5(20240108)", rC is not None and rC["exit_reason"] == "TIME_EXIT" and rC["exit_time"] == 20240108,
      None if rC is None else (rC["exit_reason"], rC["exit_time"]))

# ---------- 场景 D：涨停无法买入 -> 顺延入场 ----------
pxD, prevD = mk_px([
    (20240101, 10, 10, 10, 10),
    (20240102, 11.0, 11.0, 11.0, 11.0),# T+1 涨停开盘(>=prev10*1.098) -> 跳过
    (20240103, 10, 10, 9.9, 10),       # T+2 正常 -> 入场
    (20240104, 10, 10, 9.9, 10),
    (20240105, 10, 10, 9.9, 10),
    (20240108, 10, 10, 9.9, 10),
])
rD = run(pxD, prevD, 0, "X0_CONTROL")
check("D: 顺延入场到T+2", rD is not None and rD["entry_date"] == 20240103, None if rD is None else rD["entry_date"])

# ---------- 场景 E：跌停锁死 -> 延迟出场 ----------
# entry=10(T+1)。T+2 触发止损但当日跌停锁死 -> 延迟T+3开盘出场
pxE, prevE = mk_px([
    (20240101, 10, 10, 10, 10),
    (20240102, 10, 10, 9.9, 10),       # T+1 entry
    (20240103, 9.5, 9.6, 9.0, 9.0),    # T+2 low=9.0<=stop 触发；close=low=9.0<=prev10*0.902 跌停锁死
    (20240104, 8.5, 8.6, 8.4, 8.5),    # T+3 开盘8.5 -> 实际出场价
    (20240105, 8.5, 8.6, 8.4, 8.5),
    (20240108, 8.5, 8.6, 8.4, 8.5),
])
rE = run(pxE, prevE, 0, "X0_CONTROL")
check("E: 跌停延迟出场(blocked=1)", rE is not None and rE["exit_blocked"] == 1, rE)
check("E: 出场日=T+3(20240104)", rE is not None and rE["exit_time"] == 20240104, None if rE is None else rE["exit_time"])
check("E: 出场价=次日开盘8.5", rE is not None and abs(rE["exit_price"] - 8.5) < 1e-6, None if rE is None else rE["exit_price"])
check("E: 原因为STOP", rE is not None and rE["exit_reason"] == "STOP", None if rE is None else rE["exit_reason"])

# ---------- 场景 F：因果截断（无未来信息） ----------
pxF, prevF = mk_px([
    (20240101, 10, 10, 10, 10),
    (20240102, 10, 11, 9.9, 11),       # T+1 冲高激活T1
    (20240103, 11, 11.5, 10.2, 10.3),  # T+2 TRAIL1@10.835
    (20240104, 10.3, 10.4, 10.2, 10.3),
    (20240105, 10.3, 10.4, 10.2, 10.3),
    (20240108, 10.3, 10.4, 10.2, 10.3),
])
rF = run(pxF, prevF, 0, "X1")
check("F: 基准退出存在", rF is not None, None)
if rF is not None:
    exit_t = rF["exit_time"]
    cut = pxF[pxF["date"] <= exit_t].copy()
    cut_prev = {int(cut.iloc[k]["date"]): float(cut.iloc[k - 1]["raw_close"]) if k > 0
                else float(cut.iloc[k]["raw_close"]) for k in range(len(cut))}
    rF_cut = run(cut, cut_prev, 0, "X1")
    same = (rF_cut is not None and rF_cut["exit_price"] == rF["exit_price"]
            and rF_cut["exit_reason"] == rF["exit_reason"] and rF_cut["exit_time"] == rF["exit_time"])
    check("F: 删除t之后数据输出不变", same, f"full={rF['exit_price']} cut={None if rF_cut is None else rF_cut['exit_price']}")
    px_ext = pxF.copy()
    mask = px_ext["date"] > exit_t
    if mask.any():
        px_ext.loc[mask, "raw_high"] = 99999.0
        px_ext.loc[mask, "raw_low"] = 0.0001
        px_ext.loc[mask, "raw_close"] = 99999.0
        rF_ext = run(px_ext, prevF, 0, "X1")
        same2 = (rF_ext is not None and rF_ext["exit_price"] == rF["exit_price"]
                 and rF_ext["exit_reason"] == rF["exit_reason"] and rF_ext["exit_time"] == rF["exit_time"])
        check("F: t后注入极端价格输出不变", same2, None if rF_ext is None else rF_ext["exit_price"])
    else:
        check("F: t后注入极端价格输出不变", True)

print(f"\n通过 {len(PASS)} / 失败 {len(FAIL)}")
for f in FAIL:
    print("  FAIL:", f)
print("全部通过" if not FAIL else "存在失败")
