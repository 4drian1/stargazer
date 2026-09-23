"""execution_engine — ⑥ Real Execution / Actual Fill.

把 D 日 replay 的 entry 从 `open[D]` 占位替换为真实成交:

    snapshot(D日Open) -> signal_time
        -> next legal executable minute
        -> raw fill price
        -> slippage / fee
        -> actual_fill (raw)

纪律 (latr_builder.py 头部裁决, 用户 ⑥):
  - analytic adjusted price 只用于特征/结构 (Zone / Minute State / ATR)
  - raw price 只用于成交 / PnL
  - 涨停(买不进) / 跌停(卖不出) / 停牌 -> 不可成交 -> exec_status != FILLED,
    交给上游排除 (不伪造 alpha=0; ② 阶段明确排除相应 secondary diagnostic)

已知近似偏差 (记入 DEVIATIONS):
  - 涨跌停用开盘价 * 1.10 / * 0.90 近似 (主板 ±10%); 创业板/科创板 ±20% 未细分。
  - slippage / fee 用固定 bps 近似 (主动单吃对手盘的流动性溢价 + 印花佣金均摊),
    未做按档位/按成交量的真实冲击模型。
  - 分红/拆股在 horizon 内会造成 raw 价跳变; 此处按用户裁决用 raw 价计算 PnL,
    不排除未来在 ② 阶段引入拆股 ratio 调整。

永久合同 (EXECUTION_CONTRACT_V1, v1, 2026-09-23 冻结):
  - price_anchor_bar_ts = 09:30 (开盘 bar; **仅作定价锚, 不可作为 signal_ts 的成交分钟**)
  - legal_fill_minutes  = [09:31, 11:30] ∪ [13:01, 15:00]  (与下方 _AM / _PM 同义)
  - 永远: signal_ts ∈ legal_fill_minutes 且 signal_ts < actual_fill_ts (严格 <; 触发同分钟不可成交)
  - 详见 opportunity_v2/contracts/execution_contract_v1.md
"""
from __future__ import annotations

import pandas as pd

# 连续竞价时段 (HHMM, 含边界); 排除 11:30-13:00 午休与集合竞价伪分钟
_AM = (931, 1130)
_PM = (1301, 1500)

DEFAULT_SLIPPAGE_BPS = 2.0   # 主动单吃对手盘的流动性溢价 (保守估计)
DEFAULT_FEE_BPS = 2.1        # 印花(卖出)+佣金的双向均摊近似

# 涨跌停近似 (主板 ±10%)
LIMIT_UP_FACTOR = 1.10
LIMIT_DN_FACTOR = 0.90

# === EXECUTION_CONTRACT_V1 冻结常量 (do not change without v2) ===
EXECUTION_CONTRACT_VERSION = "v1"
PRICE_ANCHOR_BAR_HHMM = 930              # 09:30 开盘 bar; 仅作定价锚, 不可成交
LEGAL_FILL_AM = _AM                     # [09:31, 11:30] 连续竞价 (与 _AM 同义, 公开名)
LEGAL_FILL_PM = _PM                     # [13:01, 15:00] 连续竞价 (与 _PM 同义, 公开名)


def _hhmm(ts14: int) -> int:
    return ts14 % 1000000 // 100


def is_legal_minute(hhmm: int) -> bool:
    return (_AM[0] <= hhmm <= _AM[1]) or (_PM[0] <= hhmm <= _PM[1])


def find_next_executable_minute(minute_df: pd.DataFrame, signal_ts: int):
    """signal_ts(14位) 之后第一根处于连续竞价时段的分钟 bar。

    09:30 开盘价本身不是可成交分钟 (集合竞价), 故首根可执行分钟为 09:31,
    与 'next legal executable minute' 语义一致。返回 row(Series) 或 None。
    """
    if minute_df is None or minute_df.empty:
        return None
    # 严格下一合法可成交分钟: fill_ts 必须 > signal_ts (用户契约: 不能在触发同分钟成交)
    df = minute_df[minute_df["ts"] > int(signal_ts)]
    legal = df[df["ts"].map(lambda t: is_legal_minute(_hhmm(int(t))))]
    if legal.empty:
        return None
    return legal.iloc[0]


def _halt_state(row, side: str = "buy") -> str:
    """粗略涨跌停判定。买方向遇涨停买不进, 卖方向遇跌停卖不出 -> 不可成交。"""
    o = row.get("open")
    if o is None or pd.isna(o) or o <= 0:
        return "NO_QUOTE"
    if side == "buy":
        if float(row.get("high", 0)) >= o * LIMIT_UP_FACTOR * 0.995:
            return "HALTED_UP"
    else:
        if float(row.get("low", 1e9)) <= o * LIMIT_DN_FACTOR * 1.005:
            return "HALTED_DOWN"
    return "OK"


def execute(signal_ts: int, minute_df: pd.DataFrame, side: str = "buy",
            slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
            fee_bps: float = DEFAULT_FEE_BPS) -> dict:
    """执行一次 entry, 返回成交结果 dict。

    exec_status : "FILLED" | "NO_LIQUIDITY" | "HALTED_UP" | "HALTED_DOWN" | "NO_QUOTE"
    actual_fill : raw 成交价 (含 slippage+fee) 或 None
    raw_fill_price : 该分钟原始 open
    fill_ts : 14位成交时间戳 或 None
    alpha_label_available : 1 if FILLED else 0 (供 ② 排除, 不伪造)
    slippage / fee : 绝对值 (基于 raw_fill_price)
    """
    row = find_next_executable_minute(minute_df, signal_ts)
    if row is None:
        return {"exec_status": "NO_LIQUIDITY", "actual_fill": None,
                "raw_fill_price": None, "fill_ts": None,
                "alpha_label_available": 0, "slippage": None, "fee": None}
    halt = _halt_state(row, side)
    if halt != "OK":
        return {"exec_status": halt, "actual_fill": None,
                "raw_fill_price": float(row["close"]), "fill_ts": int(row["ts"]),
                "alpha_label_available": 0, "slippage": None, "fee": None}
    raw_price = float(row["open"])
    slip = raw_price * (slippage_bps / 1e4)
    fee = raw_price * (fee_bps / 1e4)
    actual_fill = raw_price + slip + fee if side == "buy" else raw_price - slip - fee
    return {"exec_status": "FILLED", "actual_fill": actual_fill,
            "raw_fill_price": raw_price, "fill_ts": int(row["ts"]),
            "alpha_label_available": 1, "slippage": slip, "fee": fee}
