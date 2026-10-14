"""Demo latest-day equivalent of frozen pools; full history retained.

Research implementation remains unchanged. Regression tests compare all columns
against its latest-day members. Only MAD evaluation and output scope differ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gcp_v0_1_spec import SPEC

A = SPEC["pool_a_strong_first_divergence"]
B = SPEC["pool_b_lowvol_bottom"]["gates"]
OUT = SPEC["outer_universe"]["rules"]


def _streak(flag: pd.Series) -> pd.Series:
    """连续 True 计数 (任一 False 重置)."""
    flag = flag.fillna(False).astype(bool)
    return flag.groupby((~flag).cumsum()).cumsum()


def process_latest_code(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """g: 单码日线 (date 升序). 返回 (A池成员行, B池成员行); 无成员返回空 DataFrame."""
    n = len(g)
    if n < 70:
        return pd.DataFrame(), pd.DataFrame()
    c = g["close"].astype(float)
    h = g["high"].astype(float)
    l = g["low"].astype(float)
    v = g["volume"].astype(float)
    o = g["open"].astype(float)
    pc = g["pre_close"].astype(float) if "pre_close" in g else c.shift(1)

    ma20 = c.rolling(20).mean()
    boll_std = c.rolling(20).std()
    boll_mid, boll_up, boll_lo = ma20, ma20 + 2 * boll_std, ma20 - 2 * boll_std
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    atr = atr14.rolling(5).median()                       # ATR_ref (冻结口径)
    delta = c.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    rsi = 100 * up.ewm(alpha=1 / 14, adjust=False).mean() / \
        (up.ewm(alpha=1 / 14, adjust=False).mean() + dn.ewm(alpha=1 / 14, adjust=False).mean())

    down = c < pc                                        # 相对前收 (实体红绿不混用)
    body_red = c < o                                     # 阴线实体, 仅记录不参与定义
    down_streak = _streak(down)
    pct = c / pc - 1.0
    flat_day = down | (pct.abs() <= B["flat_tolerance_abs_pct"]["value"])
    down_flat_streak = _streak(flat_day)

    ret10_prev = c.pct_change(10).shift(1)               # t 日看: 截至前一日的10日收益
    vol_med20_prev = v.rolling(20).median().shift(1)
    above_ma20_prev = (c > ma20).shift(1).fillna(False)
    vol_med3 = v.rolling(3).median()
    vol_med60_prior = v.shift(3).rolling(60).median()
    vol_level_ratio = vol_med3 / vol_med60_prior

    # 稳量 MAD (实现修复 2026-09-28, 用户裁决#1): 同一个三日窗口内统一减该窗口中位数再取中位,
    # 与规格 volume_stability=MAD(近3日)/中位(近3日) 一致. 旧实现(各日减自己的滚动中位再滚动中位)
    # 属算法 bug, 已修复; 阈值 0.30 不变, 属算法修复非策略定义变更.
    windows = np.lib.stride_tricks.sliding_window_view(v.to_numpy(), 3)
    medians = np.median(windows, axis=1)
    stability = np.median(np.abs(windows - medians[:, None]), axis=1) / (medians + 1e-12)
    stability[~np.isfinite(windows).all(axis=1)] = np.nan
    vol_stability = pd.Series(np.r_[np.nan, np.nan, stability], index=g.index)
    vol_trend = np.log(v.where(v > 0)).diff(2) / 2.0
    ret20 = c.pct_change(20)
    ret60 = c.pct_change(60)
    drawdown60 = c / c.rolling(60).max() - 1.0
    low60_prior = l.rolling(60).min().shift(1)
    boll_lower_dist_atr = (c - boll_lo) / atr
    boll_mid_dist_atr = (c - boll_mid) / atr
    boll_width_atr = (boll_up - boll_lo) / atr
    new_low_gap_atr = (c - low60_prior) / atr

    # 外层合格宇宙 (两池共用; NaN 特征自然不通过)
    warm_ok = pd.Series(np.arange(n) >= OUT["warmup_rows_ge"]["value"], index=g.index)
    outer = warm_ok & (g["is_st"].astype(float) == 0) & (g["float_mv"].astype(float) > 0) \
        & (g["amount"].astype(float) >= OUT["amount_ge"]["value"]) & (v > 0)

    idx = g.index.to_numpy()
    c_np, h_np, v_np = c.to_numpy(), h.to_numpy(), v.to_numpy()
    atr_np, mid_np = atr.to_numpy(), boll_mid.to_numpy()
    down_streak_np = down_streak.to_numpy()
    date_np = g["date"].to_numpy()

    # ---------------- A 池: 强势首次分歧状态机 ----------------
    strength_prev = (ret10_prev >= A["state_machine"]["strength_at_div_prev_day"]["ret10_ge"]["value"]) \
        & above_ma20_prev
    shape_div = ((h - c) / atr >= A["state_machine"]["first_divergence_day_t"]["shape_high_close_atr_ge"]["value"]) | down
    static_cand = (strength_prev & shape_div).fillna(False).to_numpy() \
        & (v_np >= A["state_machine"]["first_divergence_day_t"]["div_volume_ratio_ge"]["value"]
           * vol_med20_prev.to_numpy()) \
        & (c_np >= mid_np) & np.isfinite(atr_np)

    reset = A["state_machine"]["first_divergence_day_t"]["first_reset_window_trading_days_gt"]["value"]
    pb = A["state_machine"]["pullback_t1_to_D"]
    Lmin, Lmax = pb["length_trading_days"]["min"], pb["length_trading_days"]["max"]
    vratio_cap = pb["pullback_volume_ratio_le"]["value"]
    band = pb["near_boll_mid_band_atr"]["value"]

    last_div = -10 ** 9
    divs = []                                            # (t, div 特征快照)
    for t in np.flatnonzero(static_cand):
        if t - last_div <= reset:
            continue
        last_div = t
        divs.append((int(t), {
            "prior_ret10": float(ret10_prev.iloc[t]),
            "prior_above_ma20_atr": float((c_np[t - 1] - mid_np[t - 1]) / atr_np[t - 1]),
            "div_volume_ratio": float(v_np[t] / vol_med20_prev.iloc[t]),
            "div_close_ret": float(pct.iloc[t]),
            "div_high_close_atr": float((h_np[t] - c_np[t]) / atr_np[t]),
            "div_body_red": bool(body_red.iloc[t]),
            "div_volume": float(v_np[t]),
        }))

    a_rows, a_taken = [], {}
    for t, snap in reversed(divs):                       # 最近分歧优先
        for L in range(Lmin, Lmax + 1):
            D = t + L
            if D != n - 1 or date_np[D] in a_taken:
                continue
            if not (down_streak_np[D] >= L):             # t+1..D 全部下跌
                continue
            seg = v_np[t + 1:D + 1]
            if not np.all(np.isfinite(seg)):
                continue
            pvr = float(np.median(seg) / snap["div_volume"])
            if pvr > vratio_cap:
                continue
            if abs(c_np[D] - mid_np[D]) > band * atr_np[D]:
                continue
            if not outer.iloc[D]:
                continue
            logs = np.log(seg)
            xs = np.arange(1, L + 1, dtype=float)
            slope = float(np.polyfit(xs, logs, 1)[0]) if L >= 2 else np.nan
            a_rows.append({
                "date": int(date_np[D]), "code": str(g["code"].iloc[0]),
                "setup_type": "A_SDP", "div_date": int(date_np[t]), "days_since_first_div": L,
                "down_streak": int(down_streak_np[D]),
                "pullback_volume_ratio": pvr, "pullback_volume_slope": slope,
                "boll_mid_dist_atr": float(boll_mid_dist_atr.iloc[D]),
                "boll_mid_slope_atr": float((mid_np[D] - mid_np[D - 5]) / atr_np[D]),
                "boll_width_atr": float(boll_width_atr.iloc[D]),
                "rsi14": float(rsi.iloc[D]), "ret10_prev": None, **snap,
            })
            a_taken[date_np[D]] = True

    # ---------------- B 池: 低位稳量底部 (全向量化门槛) ----------------
    b_mask = ((ret60 <= B["ret60_le"]["value"]) & (ret20 <= B["ret20_le"]["value"])
              & (down_flat_streak >= B["down_flat_streak_ge"]["value"])
              & (vol_level_ratio <= B["volume_level_ratio_le"]["value"])
              & (vol_stability <= B["volume_stability_le"]["value"])
              & (vol_trend.abs() <= B["volume_trend_abs_le"]["value"])
              & (rsi < B["rsi_lt"]["value"])
              & (boll_lower_dist_atr <= B["boll_lower_dist_atr_le"]["value"])
              & outer).fillna(False)

    b_rows = []
    v_np_full = v.to_numpy()
    for D in ([n - 1] if b_mask.iloc[-1] else []):
        # volume_level_percentile: 近3日量中位在此前60日逐日量(不含近3日)中的分位
        prior60 = v_np_full[D - 62:D - 2]
        prior60 = prior60[np.isfinite(prior60)]
        pctile = float((prior60 < vol_med3.iloc[D]).mean()) if len(prior60) >= 30 else np.nan
        b_rows.append({
            "date": int(date_np[D]), "code": str(g["code"].iloc[0]), "setup_type": "B_LVB",
            "ret20": float(ret20.iloc[D]), "ret60": float(ret60.iloc[D]),
            "drawdown60": float(drawdown60.iloc[D]),
            "down_flat_streak": int(down_flat_streak.iloc[D]),
            "volume_level_ratio": float(vol_level_ratio.iloc[D]),
            "volume_level_percentile": pctile,
            "volume_stability": float(vol_stability.iloc[D]),
            "volume_trend": float(vol_trend.iloc[D]),
            "rsi14": float(rsi.iloc[D]),
            "boll_lower_dist_atr": float(boll_lower_dist_atr.iloc[D]),
            "boll_width_atr": float(boll_width_atr.iloc[D]),
            "new_low_gap_atr": float(new_low_gap_atr.iloc[D]),
        })

    return (pd.DataFrame(a_rows), pd.DataFrame(b_rows))
