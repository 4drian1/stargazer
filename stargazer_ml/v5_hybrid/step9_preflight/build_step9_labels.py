"""build_step9_labels — Gate 0 真实标签生成 (Step 9 Label Contract v1).

与 EXIT-DYN 标签引擎 (build_exit_labels.py) 解耦：本模块只生成
预注册的固定 Horizon 标签与 sector-neutral alpha，不直接构造退出收益。

固定 Horizon 标签（Primary / Secondary / Robustness）：
  S0-v1  : s0_return_h3 (Primary, H3)  + s0_alpha_h3 (Secondary)
  Timing : timing_return_h1 (Primary, H1) + timing_alpha_h1 (Secondary)
  R0-v1  : p_loss5 / mae_q80 / mae_q90 / failed_rebound  (沿用 Label Builder)
  Portfolio Gate: net_return_x0 (X0 唯一 Primary Exit)
  Exit Sensitivity: net_return_x1 / x1_rf / x2_staged

所有样本同时保留 return_h1/h3/h5 与 net_return_x0/x1/x1_rf/x2_staged。
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ---------- 基础工具 (与 build_exit_labels 同口径, 本地实现避免导入副作用) ----------
ATR_LOOK = 14
ATR_REF_DAYS = 5


def true_range(hi, lo, cl):
    prev = np.roll(cl, 1)
    prev[0] = cl[0]
    return np.maximum.reduce([hi - lo, np.abs(hi - prev), np.abs(lo - prev)])


def compute_atr_ref(px: pd.DataFrame, entry_idx: int) -> float | None:
    hi = px["raw_high"].to_numpy(dtype=float)
    lo = px["raw_low"].to_numpy(dtype=float)
    cl = px["raw_close"].to_numpy(dtype=float)
    if entry_idx < ATR_LOOK:
        return None
    tr = true_range(hi, lo, cl)
    atr = pd.Series(tr).rolling(ATR_LOOK, min_periods=ATR_LOOK).mean().to_numpy()
    need = np.arange(entry_idx - ATR_REF_DAYS, entry_idx)
    valid = need[need >= ATR_LOOK - 1]
    if valid.size == 0:
        return None
    return float(np.median(atr[valid]))


# ---------- S0-v1 固定 Horizon 标签 ----------
def s0_returns(stock_px: pd.DataFrame, d_open_idx: int, horizons=(1, 3, 5),
              fill_price: float | None = None) -> dict:
    """stock_px: 按 date 升序, 含 raw_open/raw_close。
    d_open_idx: D日Open所在行索引 (P日候选 -> D日观察)。
    fill_price: ⑥ Real Execution 的实际成交价 (raw); 若提供则以其为基准,
                否则回退到 raw_open[d_open_idx] (旧 open[D] 口径, 仅兼容)。
    返回 return_h1/h3/h5 (真实 PnL, 基于 actual_fill)。
    """
    open_px = float(fill_price) if fill_price is not None else stock_px["raw_open"].iloc[d_open_idx]
    out = {}
    for h in horizons:
        j = d_open_idx + h
        if j >= len(stock_px):
            out[f"return_h{h}"] = np.nan
        else:
            out[f"return_h{h}"] = stock_px["raw_close"].iloc[j] / open_px - 1.0
    return out


def s0_alpha(return_h: float, sector_px: pd.DataFrame, d_open_idx: int, h: int) -> float:
    """sector_return_h 与 stock_return_h 必须完全同起止时刻 (D日Open -> H日Close)。"""
    if sector_px is None or pd.isna(return_h):
        return np.nan
    open_px = sector_px["raw_open"].iloc[d_open_idx]
    j = d_open_idx + h
    if j >= len(sector_px):
        return np.nan
    sector_ret = sector_px["raw_close"].iloc[j] / open_px - 1.0
    return return_h - sector_ret


# ---------- Timing-v1 固定 Horizon 标签 ----------
def timing_return_h1(fill_price: float, exit_price: float) -> float:
    """actual_fill -> 第一个合法卖出交易日 Close (T+1 约束)。"""
    if fill_price is None or exit_price is None or fill_price <= 0:
        return np.nan
    return exit_price / fill_price - 1.0


def timing_alpha(timing_ret: float, sector_px: pd.DataFrame,
                 fill_idx: int, exit_idx: int) -> float:
    """Sector 起点用 actual_fill_time 对齐, 终点用 Exit_H1 对齐。"""
    if sector_px is None or pd.isna(timing_ret):
        return np.nan
    sec_fill = sector_px["raw_open"].iloc[fill_idx] if "raw_open" in sector_px else sector_px["raw_close"].iloc[fill_idx]
    sec_exit = sector_px["raw_close"].iloc[exit_idx]
    if sec_fill <= 0:
        return np.nan
    sector_ret = sec_exit / sec_fill - 1.0
    return timing_ret - sector_ret


# ---------- R0-v1: 沿用 Label Builder 四头 ----------
R0_HEADS = ["p_loss5", "mae_q80", "mae_q90", "failed_rebound"]
DEPRECATED_HEADS = ["p_hard_stop"]


def validate_r0_heads(head_list: list) -> None:
    forbidden = set(head_list) & set(DEPRECATED_HEADS)
    if forbidden:
        raise ValueError(f"[Gate0] R0 头含作废项 {forbidden}; p_hard_stop 不因换特征恢复。")


# ---------- 组合: 生成单样本 Label Contract ----------
def build_step9_label(row: dict) -> dict:
    """row 需包含: stock_px, sector_px(可None), d_open_idx, fill_price(=actual_fill),
    s0_fill_price(=D日Open, 可选, 缺省回退 fill_price), exit_price, fill_idx, exit_idx,
    r0_heads(dict)。返回 Label Contract 全部字段。

    ⑥ 契约: S0 起点 = D 日 legal open (snapshot), Timing 起点 = actual_fill (Execution Engine)。
    两者锚点严格分离, 不得共用 fill_price 让 S0 偷偷变成 Timing fill -> H3。
    S0 优先用 analytic(adj) 口径以保证公司行为安全; 否则回退 raw。
    """
    stock_px = row["stock_px"]
    d_open_idx = row["d_open_idx"]
    s0_fill = row.get("s0_fill_price", row.get("fill_price"))
    if "adj_open" in stock_px.columns and "adj_close" in stock_px.columns:
        open_px = float(stock_px["adj_open"].iloc[d_open_idx])
        s0 = {}
        for h in (1, 3, 5):
            j = d_open_idx + h
            s0[f"return_h{h}"] = (stock_px["adj_close"].iloc[j] / open_px - 1.0) if j < len(stock_px) else np.nan
    else:
        s0 = s0_returns(stock_px, d_open_idx, fill_price=s0_fill)
    lab = dict(s0)
    lab["s0_return_h3"] = s0.get("return_h3")
    lab["s0_alpha_h3"] = s0_alpha(s0.get("return_h3"), row.get("sector_px"), d_open_idx, 3)
    # Timing 锚定 actual_fill (不受 S0 锚点影响)
    lab["timing_return_h1"] = timing_return_h1(row.get("fill_price"), row.get("exit_price"))
    lab["timing_alpha_h1"] = timing_alpha(lab["timing_return_h1"], row.get("sector_px"),
                                          row.get("fill_idx"), row.get("exit_idx"))
    validate_r0_heads(list(row.get("r0_heads", {}).keys()))
    for k, v in row.get("r0_heads", {}).items():
        lab[k] = v
    return lab
