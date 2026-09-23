"""P_CLOSE 修正: 重新计算 BOTTOM-WASHOUT 链中所有"zone/volume"特征 at plan_date 15:00.

Gate 1 修复 (2026-09-22): a14_universe 里的 zone_distance_atr / penetration_atr /
lower_wick_ratio / zone_break_* / sell_volume_ratio / rebound_volume_ratio 全部由
latr_builder 在 effective_trade_date(=D日) 15:00 计算 (feature_asof_ts = d_date 150000),
对 S0 (P_CLOSE Selection) 目标构成未来信息泄漏. 本脚本在 plan_date(P日) 15:00 用当日分钟
重算, 与 W0(A1_CORE 日线) 口径一致, 得到可合法用于 P_CLOSE Selection 的特征.

计算全部为 stateless (不依赖 D 日 state machine, 因为 BOTTOM 候选的 zone touch 多发生在 D 日,
P 日未必触发, 故不跑 run_state_engine, 直接用 P 日分钟 + a14 的 zone 几何).

输出: bottom_washout_pclose_w4.parquet (sample_id + 7 个 pc_* 特征), 覆盖 BOTTOM universe.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import a0a4_config as C  # noqa
sys.path.insert(0, str(_HERE.parent / "step9_preflight"))
from data_source_provenance import get_minute_http  # noqa

OUT = C.DATASETS / "bottom_washout_pclose_w4.parquet"


def _pc(code, plan, atr, zlow, zup, zmid):
    """返回 7 个 P_CLOSE 特征; 任一不可算返回 NaN 元组."""
    if pd.isna(atr) or atr <= 0 or pd.isna(zlow) or pd.isna(zup) or pd.isna(zmid):
        return (np.nan,) * 7
    try:
        m = get_minute_http(str(code).zfill(6), str(int(plan)))
    except Exception:
        return (np.nan,) * 7
    if m is None or len(m) == 0:
        return (np.nan,) * 7
    if "date" in m.columns and "ts" not in m.columns:
        m = m.rename(columns={"date": "ts"})
    m = m[m["ts"] <= int(f"{int(plan)}150000")].copy()
    if len(m) == 0:
        return (np.nan,) * 7
    lo = m["low"].to_numpy(float)
    hi = m["high"].to_numpy(float)
    cl = m["close"].to_numpy(float)
    vo = m["volume"].to_numpy(float)
    d_lo, d_hi, d_cl = lo.min(), hi.max(), cl[-1]
    # 1) zone_distance_atr = (close_P - zone_mid)/atr  (带符号)
    zdist = (d_cl - zmid) / atr
    # 2) penetration_atr = (zone_upper - dayP_low)/atr  (穿透 zone 下沿的程度)
    pen = (zup - d_lo) / atr
    # 3) lower_wick_ratio = (close - low)/(high - low)  (P 日整体下影)
    lwr = (d_cl - d_lo) / (d_hi - d_lo) if (d_hi - d_lo) > 0 else np.nan
    # 4) zone_break_depth_atr = max(0,(zone_lower - dayP_low)/atr)
    zbd = max(0.0, (zlow - d_lo) / atr)
    # 5) zone_break_duration = P 日分钟中 low<zone_lower 的 bar 数
    zbdur = int(np.sum(lo < zlow))
    # 6/7) sell/rebound volume ratio = down/up bar 成交量占比
    if vo.sum() > 0:
        down = vo[1:][cl[1:] < cl[:-1]]
        up = vo[1:][cl[1:] > cl[:-1]]
        svr = float(down.sum() / vo.sum())
        rvr = float(up.sum() / vo.sum())
    else:
        svr = rvr = np.nan
    return zdist, pen, lwr, zbd, zbdur, svr, rvr


def main():
    t0 = time.time()
    a = pd.read_parquet(C.DATASETS / "a14_universe.parquet")
    sub = a[a["is_bottom"].astype(bool)].copy()
    sub["code"] = sub["sample_id"].str.split("|").str[0]
    cols = ["sample_id", "code", "plan_date", "atr_ref",
            "zone_lower", "zone_upper", "zone_mid"]
    z = sub[[c for c in cols if c in sub.columns]].copy()
    print(f"[pclose-w4] BOTTOM universe n={len(z)}")
    rows = []
    for i, (_, r) in enumerate(z.iterrows()):
        vals = _pc(r["code"], r["plan_date"], r["atr_ref"],
                  r.get("zone_lower"), r.get("zone_upper"), r.get("zone_mid"))
        rows.append((r["sample_id"],) + vals)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(z)} ({time.time() - t0:.0f}s)")
    out = pd.DataFrame(rows, columns=[
        "sample_id", "pc_zone_distance_atr", "pc_penetration_atr", "pc_lower_wick_ratio",
        "pc_zone_break_depth_atr", "pc_zone_break_duration",
        "pc_sell_volume_ratio", "pc_rebound_volume_ratio"])
    out.to_parquet(OUT, index=False)
    print(f"[pclose-w4] wrote {OUT}")
    for c in out.columns[1:]:
        print(f"    {c}: nonNaN={out[c].notna().mean():.1%} "
              f"median={out[c].median():.3f}" if out[c].notna().any() else f"    {c}: ALL NaN")


if __name__ == "__main__":
    main()
