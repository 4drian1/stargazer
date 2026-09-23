"""a0a4_exit_sim — X0-style 退出模拟 (冻结 X0_CONTROL 参数移植到 LATR fill 锚点).

规则 (预注册, 参数来自冻结 green_exit_v1/build_exit_labels.py X0_CONTROL):
  entry       = actual_fill (raw, 已含买侧 slippage 2.0bps + fee 2.1bps)
  stop_dist   = clip(1.5 * atr_ref / fill, 3%, 6%)
  stop_px     = fill * (1 - stop_dist)
  可卖起始    = D+1 (T+1); tmax = D+4 收盘时间退出
  跌停锁死    = close==low 且跌幅 <= -(limit-0.5pp) -> 顺延, 最多 3 日, 之后强制开盘价出
  日线近似    : open<=stop -> 开盘出; low<=stop -> stop 价出; D+4 -> 收盘出
  卖侧成本    = 0.0020 (slippage 2bps + 费用 18bps)
  net         = exit_px * (1-0.0020) / fill - 1

数据: data_daily_parts parquet (本地, 全量一次载入)。
产出: datasets/{a0,a14}_x0_exits.parquet
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import a0a4_config as C  # noqa

sys.path.insert(0, str(C.PREFLIGHT))


def _limit_rate(code: str) -> float:
    return 0.20 if code[:2] in ("30", "68") else 0.10


def simulate_exits(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """daily: 全市场日线 (code,date,open,high,low,close,pre_close)。"""
    need = ["sample_id", "symbol", "d_date", "actual_fill", "atr_ref"]
    sub = df[[c for c in need if c in df] + ["timing_eligible"]].copy()
    sub = sub[sub["actual_fill"].notna()]
    by_code = {c: g.set_index("date") for c, g in daily.groupby("code")}

    recs = []
    for r in sub.itertuples(index=False):
        code, d, fill, atr = r.symbol, int(r.d_date), float(r.actual_fill), float(r.atr_ref)
        dd = by_code.get(code)
        if dd is None or d not in dd.index:
            continue
        idx = dd.index.get_indexer([d])
        if idx[0] < 0:
            continue
        i = int(idx[0])
        n = len(dd)
        stop_dist = min(max(1.5 * atr / fill, 0.03), 0.06)
        stop_px = fill * (1.0 - stop_dist)
        lim = _limit_rate(code)
        exit_px = exit_date = exit_reason = None
        blocked = 0
        j = i + 1
        while j < n:
            o, h, l, cl, pc = (float(dd[c].iloc[j]) for c in
                               ("open", "high", "low", "close", "pre_close"))
            dt = int(dd.index[j])
            # 跌停锁死判定 (近似: 收在最低 & 跌幅贴近跌停)
            if cl == l and pc > 0 and (cl / pc - 1) <= -(lim - 0.005):
                blocked += 1
                if blocked > 3:
                    exit_px, exit_date, exit_reason = o, dt, "limit_down_forced"
                    break
                j += 1
                continue
            if o <= stop_px:
                exit_px, exit_date, exit_reason = o, dt, "stop_gap_open"
                break
            if l <= stop_px:
                exit_px, exit_date, exit_reason = stop_px, dt, "stop"
                break
            if j - i >= 4:
                exit_px, exit_date, exit_reason = cl, dt, "time_exit"
                break
            j += 1
        if exit_px is None:
            if j >= n:
                last = n - 1
                exit_px = float(dd["close"].iloc[last])
                exit_date = int(dd.index[last])
                exit_reason = "data_end_forced"
            else:
                continue
        net = exit_px * (1.0 - C.X0_STYLE["exit_cost"]) / fill - 1.0
        recs.append({"sample_id": r.sample_id, "x0_exit_date": exit_date,
                     "x0_exit_px": round(exit_px, 4),
                     "x0_exit_reason": exit_reason,
                     "x0_net_return": round(net, 6),
                     "x0_holding_days": j - i,
                     "x0_stop_dist": round(stop_dist, 5)})
    out = pd.DataFrame(recs)
    return out


def main():
    from data_source_provenance import get_daily_parquet
    t0 = time.time()
    daily = get_daily_parquet()
    daily = daily[["code", "date", "open", "high", "low", "close", "pre_close"]]
    daily["date"] = daily["date"].astype(int)
    print(f"[exit_sim] daily rows={len(daily)} ({time.time()-t0:.0f}s)")
    for pool in ("a14", "a0"):
        df = pd.read_parquet(C.DATASETS / f"{pool}_universe.parquet")
        out = simulate_exits(df, daily)
        f = C.DATASETS / f"{pool}_x0_exits.parquet"
        out.to_parquet(f, index=False)
        print(f"[exit_sim] {pool}: {len(out)} exits -> {f} "
              f"({time.time()-t0:.0f}s)")
        if len(out):
            print(f"[exit_sim] {pool} reasons: "
                  f"{out['x0_exit_reason'].value_counts().to_dict()}")
            print(f"[exit_sim] {pool} net mean={out['x0_net_return'].mean():.5f} "
                  f"median={out['x0_net_return'].median():.5f} "
                  f"win={(out['x0_net_return']>0).mean():.3f}")


if __name__ == "__main__":
    main()
