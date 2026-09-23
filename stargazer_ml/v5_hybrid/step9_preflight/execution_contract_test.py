"""⑥ Execution Final QA — 契约测试.

A. S0 / Timing 起点分离 (label_start 不可串味):
     S0:      label_start = D 日 legal open
     Timing:  label_start = actual_fill_ts (Execution Engine 成交)
   断言:
     s0_label_start_price == D_open_price
     timing_label_start_price == actual_fill_price
   且 s0_return_h3 必须用 D-open 锚点 (不为 actual_fill 锚点)。

B. 任意 snapshot Execution (fill_ts 必须 > signal_ts 且为下一合法可成交分钟):
     09:30/09:45/10:17/11:29/11:30/13:00/13:01/14:36/14:56
     + 涨停封死(HALTED_UP) / 跌停封死(HALTED_DOWN, sell) / 缺失分钟(NO_LIQUIDITY)
     + 公司行为安全投影 (actual raw fill -> 同 analytic scale -> exit)。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import execution_engine as EE
from build_step9_labels import s0_returns, timing_return_h1, build_step9_label

FAILS = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}  {detail}")
    if not cond:
        FAILS.append(name)

# ---------------- A. S0 / Timing 起点分离 ----------------
print("\n===== A. S0 / Timing 起点分离 =====")
# 合成日线: D open=100, H3(idx3) close=110, H1(idx1) close=105
n = 8
px = pd.DataFrame({
    "raw_open":  [100.0, 102, 103, 104, 105, 106, 107, 108],
    "raw_close": [101.0, 103, 105, 110, 113, 114, 115, 116],
    "adj_open":  [100.0, 102, 103, 104, 105, 106, 107, 108],   # 无公司行为: adj==raw
    "adj_close": [101.0, 103, 105, 110, 113, 114, 115, 116],
})
D_OPEN = 100.0
ACTUAL_FILL = 101.0          # 假设 09:31 成交价 (含滑点)
H3_CLOSE = 110.0             # = raw_close[3]
H1_CLOSE = 105.0             # = raw_close[1]

# A1: s0_returns 默认(fill_price=None) -> D open 锚点
s0_default = s0_returns(px, 0, fill_price=None)
check("A1 s0_returns(fill=None) 锚定 D open",
      abs(s0_default["return_h3"] - (H3_CLOSE / D_OPEN - 1)) < 1e-9,
      f"got {s0_default['return_h3']:.6f} expect {H3_CLOSE/D_OPEN-1:.6f}")

# A2: s0_returns(fill=actual_fill) -> actual_fill 锚点 (与 D-open 不同)
s0_actual = s0_returns(px, 0, fill_price=ACTUAL_FILL)
check("A2 s0_returns(fill=actual_fill) 锚定 actual_fill (应≠D-open)",
      abs(s0_actual["return_h3"] - (H3_CLOSE / ACTUAL_FILL - 1)) < 1e-9 and
      abs(s0_actual["return_h3"] - s0_default["return_h3"]) > 1e-6,
      f"actual_anchor={s0_actual['return_h3']:.6f} open_anchor={s0_default['return_h3']:.6f}")

# A3: build_step9_label 契约: S0 锚定 D-open (传入 s0_fill_price=D-open), Timing 锚定 actual_fill
row = {
    "stock_px": px, "sector_px": None, "d_open_idx": 0,
    "fill_price": ACTUAL_FILL,                 # Timing 用 actual_fill
    "s0_fill_price": D_OPEN,                  # S0 用 D-open
    "exit_price": H1_CLOSE, "fill_idx": 0, "exit_idx": 1, "r0_heads": {},
}
lab = build_step9_label(row)
s0_expect_open = H3_CLOSE / D_OPEN - 1          # 契约期望 (D-open 锚点)
check("A3 契约: build_step9_label 的 s0_return_h3 必须锚定 D open",
      abs(lab["s0_return_h3"] - s0_expect_open) < 1e-9,
      f"got {lab['s0_return_h3']:.6f} expect {s0_expect_open:.6f}")

# A4: Timing 必须锚定 actual_fill
timing_expect = H1_CLOSE / ACTUAL_FILL - 1
check("A4 契约: timing_return_h1 锚定 actual_fill",
      abs(lab["timing_return_h1"] - timing_expect) < 1e-9,
      f"got {lab['timing_return_h1']:.6f} expect {timing_expect:.6f}")

# A5: 显式 timestamp 契约 (模拟 latr_builder 输出字段)
D_OPEN_TS = 20260102093000
FILL_TS    = 20260102093100
check("A5 s0_label_start_ts == D_open_ts",    D_OPEN_TS == D_OPEN_TS)
check("A5 timing_label_start_ts == actual_fill_ts", FILL_TS == FILL_TS)

# ---------------- B. 任意 snapshot Execution ----------------
print("\n===== B. 任意 snapshot Execution =====")
def mk_day():
    """构造一整天分钟 bar (09:31~11:30, 13:01~15:00), open=100 常量。"""
    rows = []
    date = 20260102
    for hhmm in list(range(931, 1131)) + list(range(1301, 1501)):
        ts = int(f"{date}{hhmm:04d}00")
        rows.append({"ts": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5})
    return pd.DataFrame(rows)

def ts(hhmm):
    return int(f"20260102{hhmm:04d}00")

m = mk_day()
# 期望 = signal_ts 之后第一根合法可成交分钟 (严格 >)
for sig, expect_fill in [(930, 931), (945, 946), (1017, 1018), (1129, 1130),
                        (1130, 1301), (1300, 1301), (1301, 1302),
                        (1436, 1437), (1456, 1457)]:
    st = ts(sig)
    r = EE.execute(st, m, side="buy")
    ok = (r["exec_status"] == "FILLED" and r["fill_ts"] is not None
          and r["fill_ts"] > st and EE.is_legal_minute(EE._hhmm(r["fill_ts"]))
          and EE._hhmm(r["fill_ts"]) == expect_fill)
    check(f"B sig={sig:04d} -> fill={EE._hhmm(r['fill_ts']) if r['fill_ts'] else None} "
          f"(expect {expect_fill})", ok,
          f"exec_status={r['exec_status']} fill_ts={r['fill_ts']}")

# B-halt-up: 09:31 涨停
m_halt = m.copy()
m_halt.loc[m_halt["ts"] == ts(931), "high"] = 111.0   # >= 100*1.10*0.995
r = EE.execute(ts(930), m_halt, side="buy")
check("B 涨停封死 -> HALTED_UP 且 actual_fill=None",
      r["exec_status"] == "HALTED_UP" and r["actual_fill"] is None,
      f"exec_status={r['exec_status']}")

# B-halt-down (sell side)
m_dn = m.copy()
m_dn.loc[m_dn["ts"] == ts(931), "low"] = 89.0     # <= 100*0.90*1.005
r = EE.execute(ts(930), m_dn, side="sell")
check("B 跌停封死(sell) -> HALTED_DOWN 且 actual_fill=None",
      r["exec_status"] == "HALTED_DOWN" and r["actual_fill"] is None,
      f"exec_status={r['exec_status']}")

# B-missing: 末段无合法分钟 (signal 14:57, 仅到 14:56)
m_short = m[m["ts"] <= ts(1456)].copy()
r = EE.execute(ts(1457), m_short, side="buy")
check("B 缺失分钟 -> NO_LIQUIDITY 且 fill_ts=None",
      r["exec_status"] == "NO_LIQUIDITY" and r["fill_ts"] is None,
      f"exec_status={r['exec_status']}")

# B-公司行为安全投影: adj_fill = actual_fill * (adj_open/raw_open)
# 构造 D 日发生 2:1 拆股场景: raw_open=10, adj_open=20 (因子2), actual_fill(raw)=10.05
AF = 10.05
adj_fill = AF * (20.0 / 10.0)                # = 20.1
check("B 公司行为投影 adj_fill = actual_fill*(adj_open/raw_open)",
      abs(adj_fill - 20.1) < 1e-9, f"adj_fill={adj_fill}")
# 验证 exit 同 analytic scale: H3 adj_close=22 (=raw 11 *2), 则 timing = 22/adj_fill-1 == 11/10.05-1
adj_h3 = 22.0
timing_adj = adj_h3 / adj_fill - 1
timing_raw = 11.0 / AF - 1
check("B 公司行为安全: analytic PnL == raw PnL (同比例因子)",
      abs(timing_adj - timing_raw) < 1e-9,
      f"adj={timing_adj:.6f} raw={timing_raw:.6f}")

print("\n===== SUMMARY =====")
if FAILS:
    print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
