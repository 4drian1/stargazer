# -*- coding: utf-8 -*-
"""T6 鲁棒动态价格带特征（2026-09-09 冻结方案第三节）。

三条参考轴：20 根 15 分钟 K 短周期鲁棒中轴（IRLS-Huber）、48 根 15 分钟 K
慢周期鲁棒中轴、当日累计 VWAP。通道半宽 = 2×1.4826×MAD(残差)，
上下界 = 中轴±半宽。全部距离/深度/斜率用 dp 日 pre_close×atr_pct
（冻结 ATR_ref 的价格尺度）归一为 ATR 单位。

特征（全部只用 ≤DP 已完成的 15 分钟槽，无前视）：
  轴距离：  t6_dist_mid_s_atr / t6_dist_mid_l_atr / t6_dist_vwap_atr
  通道：    t6_below_lower_depth_atr / t6_above_upper_depth_atr /
            t6_space_to_upper_atr / t6_width_atr / t6_width_chg_2h_atr /
            t6_mid_slope_atr_h
  穿越行为：t6_below_now_min（当前连续在中轴下方分钟数）/
            t6_since_first_below_min（当前下方段起点至今，上方=NaN）/
            t6_since_reclaim_min / t6_hold_after_reclaim_min /
            t6_last_cross_dir(+1=下→上收复 / -1=上→下跌破 / 0=无穿越) /
            t6_low2_minus_low1_atr（收复点之后回撤低点 − 之前下杀低点）
  上界行为：t6_upper_touch(0/1) / t6_upper_mode
            (0未触/1站稳/2假突破/3冲高衰竭)
  量能修正：t6_volsur_break（当前下方段槽的均值）/ t6_volsur_reclaim
            （收复后2槽均值）；surprise=槽量/过去20日同时段槽量中位数

实现：分钟线映射到 16 格 15 分钟槽（上午8+下午8，交易分钟轴）；
尾窗 = 前3完整日(48槽)+当日已完成槽，右对齐 64 宽缓冲；
16 个槽位各做一次短轴(20)/慢轴(48) IRLS-Huber，向量化到全体交易日。
幂等；--samples/--out 跑 2025 与 2026；审计见 t6_audit.py。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minute_label_engine import load_code  # noqa: E402

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
SAMPLES = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
OUT = os.path.join(OUT_DIR, "t6_features_2025.parquet")
DPS = (945, 1000, 1030, 1330, 1400, 1430, 1450)
# dp → 最后一个已完成15分钟槽（slot_of_t(dp*100)-1）。
# 决策点dp的约定（与引擎一致）：可知数据=分钟t<=dp*100；dp所在槽未走完，
# 用其前一槽（已完成bar）。945→0(9:30-9:45) 1000→1 1030→3 1330→9
# 1400→11 1430→13 1450→14。修正前直接用dp所在槽=含未来最多14分钟（§18.30）。
DP_SLOT = {945: 0, 1000: 1, 1030: 3, 1330: 9, 1400: 11, 1430: 13, 1450: 14}
NSLOT = 16
X = np.arange(64.0)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def slot_of_t(t):
    """HHMMSS → 15分钟槽 0..15（交易分钟轴，午休不计）。

    上午 9:30(槽0)..11:30(槽7，含113000边界分钟：msm=120 须归上午，
    否则11:30 bar会错落到槽2造成污染——§18.30无前视审计抓出）；
    下午 13:00(槽8)..14:59(槽15)；15:00(槽16)由 reindex(16) 丢弃。"""
    msm = (t // 10000) * 60 + (t // 100) % 100 - 570
    am = msm < 121
    tm = np.where(am, np.minimum(msm, 119), msm - 90)
    return tm // 15


def day_slots(cm, day_i):
    m = cm["day_idx"] == day_i
    t = cm["t"][m].astype(int)
    df = pd.DataFrame({
        "sl": slot_of_t(t),
        "o": cm["o"][m].astype(float), "h": cm["h"][m].astype(float),
        "l": cm["l"][m].astype(float), "c": cm["c"][m].astype(float),
        "v": cm["v"][m].astype(float), "a": cm["a"][m].astype(float)})
    g = df.groupby("sl")
    out = pd.DataFrame({
        "o": g["o"].first(), "h": g["h"].max(), "l": g["l"].min(),
        "c": g["c"].last(), "v": g["v"].sum(), "a": g["a"].sum()})
    out = out.reindex(range(NSLOT))
    return {k: out[k].values for k in ("o", "h", "l", "c", "v", "a")}


def huber_irls(y, mask, n_iter=8):
    """向量化 IRLS-Huber。y (n,64)（掩码处置0）、mask (n,64)。
    返回末端(x=63)中轴、斜率(px/槽)、MAD 尺度。"""
    import warnings
    ym = np.where(mask, y, 0.0)
    W = mask.astype(float)
    b = np.zeros(len(y))
    a = np.zeros(len(y))
    med = np.zeros(len(y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_iter):
            sw = W.sum(1) + 1e-12
            xbar = (W * X).sum(1) / sw
            ybar = (W * ym).sum(1) / sw
            xc = X - xbar[:, None]
            yc = ym - ybar[:, None]
            b = (W * xc * yc).sum(1) / ((W * xc * xc).sum(1) + 1e-12)
            a = ybar - b * xbar
            r = np.where(mask, ym - (a[:, None] + b[:, None] * X), np.nan)
            med = np.nanmedian(r, 1)
            scale = 1.4826 * np.nanmedian(np.abs(r - med[:, None]), 1) + 1e-12
            u = np.abs(r) / (1.345 * scale[:, None])
            w_new = np.where(mask, np.minimum(1.0, 1.0 / u), 0.0)
            W = np.where(np.isfinite(w_new), w_new, 0.0)
        mad = 1.4826 * np.nanmedian(np.abs(r - med[:, None]), 1)
    return a + b * 63.0, b, mad


def compute_code(cm, want_days):
    days = cm["days"]
    nd = len(days)
    # 全量槽数组 (nd,16)：一次 groupby
    t = cm["t"].astype(int)
    df = pd.DataFrame({
        "d": cm["day_idx"], "sl": slot_of_t(t),
        "o": cm["o"].astype(float), "h": cm["h"].astype(float),
        "l": cm["l"].astype(float), "c": cm["c"].astype(float),
        "v": cm["v"].astype(float), "a": cm["a"].astype(float)})
    g = df.groupby(["d", "sl"])
    agg = pd.DataFrame({
        "o": g["o"].first(), "h": g["h"].max(), "l": g["l"].min(),
        "c": g["c"].last(), "v": g["v"].sum(), "a": g["a"].sum()})
    agg = agg.reindex(pd.MultiIndex.from_product(
        [range(nd), range(NSLOT)], names=["d", "sl"]))
    C = agg["c"].values.reshape(nd, NSLOT)
    H = agg["h"].values.reshape(nd, NSLOT)
    L = agg["l"].values.reshape(nd, NSLOT)
    V = np.nan_to_num(agg["v"].values.reshape(nd, NSLOT))
    A = np.nan_to_num(agg["a"].values.reshape(nd, NSLOT))
    pc = np.full(nd, np.nan)
    pc[1:] = cm["day_close"][:-1]
    cv = np.cumsum(V, 1)
    with np.errstate(all="ignore"):
        vwap = np.cumsum(A, 1) / np.where(cv > 0, cv, np.nan)

    # 尾窗缓冲：槽0..47 = d-3,d-2,d-1；槽48..63 = 当日。越界置 NaN。
    bufc = np.full((nd, 64), np.nan)
    for off in range(3):
        src = np.roll(C, 3 - off, axis=0)
        src[:3 - off] = np.nan
        bufc[:, 16 * off:16 * (off + 1)] = src
    bufc[:, 48:64] = C
    didx = np.arange(nd)[:, None]

    # 只在事件日行拟合（尾窗历史在 64 宽缓冲内，不依赖其他行）
    rel = [d for d in sorted(want_days) if 0 <= d < nd]
    ridx = np.array(rel)
    bufc_r = bufc[ridx]
    didx_r = didx[ridx]

    mid_s = np.full((nd, NSLOT), np.nan)
    half_s = np.full((nd, NSLOT), np.nan)
    slope_s = np.full((nd, NSLOT), np.nan)
    mid_l = np.full((nd, NSLOT), np.nan)
    for k in range(NSLOT):
        Lwin = 48 + (k + 1)
        hist_ok = (X[None, :] < 48) & (didx_r >= 3)
        today_ok = (X[None, :] >= 48) & (X[None, :] < 48 + k + 1)
        base = np.isfinite(bufc_r) & (hist_ok | today_ok)
        have_s = base.sum(1) >= 20
        m_s = base & (X[None, :] >= 64 - 20)
        mid, b, mad = huber_irls(np.nan_to_num(bufc_r),
                                 m_s & have_s[:, None])
        mid_s[ridx, k] = np.where(have_s, mid, np.nan)
        half_s[ridx, k] = np.where(have_s, 2.0 * 1.4826 * mad, np.nan)
        slope_s[ridx, k] = np.where(have_s, b, np.nan)
        have_l = base.sum(1) >= 48
        m_l = base & (X[None, :] >= 64 - 48)
        midl, _, _ = huber_irls(np.nan_to_num(bufc_r),
                                m_l & have_l[:, None])
        mid_l[ridx, k] = np.where(have_l, midl, np.nan)

    Vmed = np.full((nd, NSLOT), np.nan)
    for s in range(NSLOT):
        Vmed[:, s] = pd.Series(V[:, s]).rolling(20, min_periods=10) \
            .median().shift(1).values
    with np.errstate(all="ignore"):
        vsurp = V / Vmed

    rows = []
    for d in sorted(want_days):
        if d >= nd or not np.isfinite(C[d]).any():
            continue
        for dp, s in DP_SLOT.items():
            if not np.isfinite(C[d, s]):
                continue
            px = C[d, s]
            ms, hf = mid_s[d, s], half_s[d, s]
            if not np.isfinite(ms):
                continue
            upper_k = mid_s[d, :s + 1] + half_s[d, :s + 1]
            upper, lower = ms + hf, ms - hf
            tc, th, tl = C[d, :s + 1], H[d, :s + 1], L[d, :s + 1]
            mids = mid_s[d, :s + 1]
            below = np.isfinite(tc) & np.isfinite(mids) & (tc < mids)
            run = 0
            for j in range(s, -1, -1):
                if below[j]:
                    run += 1
                else:
                    break
            trans, cross_dir = -1, 0
            for j in range(s, 0, -1):
                if below[j - 1] != below[j]:
                    trans = j
                    cross_dir = 1 if not below[j] else -1   # 下→上=+1
                    break
            since_reclaim = hold_after = low2m1 = np.nan
            if cross_dir == 1 and trans >= 0:
                since_reclaim = (s - trans) * 15
                hold = 0
                for j in range(trans, s + 1):
                    if not below[j]:
                        hold += 1
                hold_after = (hold - 1) * 15 if hold > 0 else 0
                seg1 = L[d, max(0, trans - 16):trans]
                if np.isfinite(seg1).any():
                    low2m1 = np.nanmin(L[d, trans:s + 1]) - np.nanmin(seg1)
            touch = np.isfinite(th) & np.isfinite(upper_k) & (th >= upper_k)
            upper_touch = float(touch.any())
            upper_mode = 0
            if touch.any():
                kt = int(np.argmax(touch))
                ke = min(kt + 3, s)
                cw = C[d, kt:ke + 1]
                uw = upper_k[kt:ke + 1]
                mw = mids[kt:ke + 1]
                fin = np.isfinite(cw) & np.isfinite(uw)
                if fin.sum() >= 2 and (cw[fin] > uw[fin]).sum() >= 2:
                    upper_mode = 1                       # 站稳
                elif fin.sum() >= 1 and cw[fin][-1] < mw[fin][-1]:
                    upper_mode = 2                       # 假突破（收回中轴下）
                else:
                    upper_mode = 3                       # 冲高衰竭
            vs = vsurp[d, :s + 1]
            vs_break = float(np.nanmean(vs[below])) \
                if run > 0 and np.isfinite(vs[below]).any() else np.nan
            vs_reclaim = np.nan
            if cross_dir == 1 and trans >= 0:
                seg_r = vs[trans:min(trans + 2, s + 1)]
                if np.isfinite(seg_r).any():
                    vs_reclaim = float(np.nanmean(seg_r))
            half_prev = half_s[d, s - 8] if s >= 8 else np.nan
            rows.append({
                "day": int(days[d]), "dp": dp, "t6_px": px,
                "t6_mid_s": ms, "t6_mid_l": mid_l[d, s], "t6_vwap": vwap[d, s],
                "t6_upper": upper, "t6_lower": lower, "t6_half": hf,
                "t6_half_prev": half_prev,
                "t6_slope_px": slope_s[d, s],
                "t6_run": run, "t6_cross_dir": cross_dir,
                "t6_since_reclaim": since_reclaim, "t6_hold_after": hold_after,
                "t6_low2m1": low2m1,
                "t6_upper_touch": upper_touch, "t6_upper_mode": upper_mode,
                "t6_vs_break": vs_break, "t6_vs_reclaim": vs_reclaim,
                "t6_pc": pc[d]})
    return pd.DataFrame(rows)


def normalize(raw, atr_map):
    """原始 compute_code 输出 + {dp_date: atr_pct} → 归一特征 DataFrame
    （索引 day,dp；不含 code/dp_date 重命名，由调用方处理）。"""
    raw = raw.set_index(["day", "dp"]) if "day" in raw.columns else raw
    raw["atr_pct"] = [atr_map.get(int(d), np.nan) for d in
                      raw.index.get_level_values(0)]
    ap = raw["t6_pc"] * raw["atr_pct"]
    ok = np.isfinite(ap) & (ap > 0)
    out = pd.DataFrame(index=raw.index)
    out["t6_dist_mid_s_atr"] = (raw["t6_px"] - raw["t6_mid_s"]) / ap
    out["t6_dist_mid_l_atr"] = (raw["t6_px"] - raw["t6_mid_l"]) / ap
    out["t6_dist_vwap_atr"] = (raw["t6_px"] - raw["t6_vwap"]) / ap
    out["t6_below_lower_depth_atr"] = np.maximum(
        0.0, (raw["t6_lower"] - raw["t6_px"]) / ap)
    out["t6_above_upper_depth_atr"] = np.maximum(
        0.0, (raw["t6_px"] - raw["t6_upper"]) / ap)
    out["t6_space_to_upper_atr"] = (raw["t6_upper"] - raw["t6_px"]) / ap
    out["t6_width_atr"] = 2.0 * raw["t6_half"] / ap
    out["t6_width_chg_2h_atr"] = \
        2.0 * (raw["t6_half"] - raw["t6_half_prev"]) / ap
    out["t6_mid_slope_atr_h"] = raw["t6_slope_px"] * 4.0 / ap
    out["t6_below_now_min"] = raw["t6_run"] * 15
    out["t6_since_first_below_min"] = \
        np.where(raw["t6_run"] > 0, raw["t6_run"] * 15, np.nan)
    out["t6_since_reclaim_min"] = raw["t6_since_reclaim"]
    out["t6_hold_after_reclaim_min"] = raw["t6_hold_after"]
    out["t6_last_cross_dir"] = raw["t6_cross_dir"]
    out["t6_low2_minus_low1_atr"] = raw["t6_low2m1"] / ap
    out["t6_upper_touch"] = raw["t6_upper_touch"]
    out["t6_upper_mode"] = raw["t6_upper_mode"]
    out["t6_volsur_break"] = raw["t6_vs_break"]
    out["t6_volsur_reclaim"] = raw["t6_vs_reclaim"]
    return out[ok.values].reset_index()


def main():
    global SAMPLES, OUT
    args = sys.argv[1:]
    if "--samples" in args:
        SAMPLES = args[args.index("--samples") + 1]
    if "--out" in args:
        OUT = args[args.index("--out") + 1]
    t0 = time.time()
    df = pd.read_parquet(SAMPLES)
    df = df[(df["dp"].isin(DPS)) & (df["entry_type"] == "vwap")].copy()
    df["code"] = df["code"].astype(str)
    g = (df.groupby(["code", "dp_date", "dp"])
           .agg(atr_pct=("atr_pct", "first")).reset_index())
    log(f"事件 (code,dp_date,dp): {len(g):,}")
    frames = []
    for code, sub in g.groupby("code"):
        cm = load_code(code)
        if cm is None:
            continue
        days = cm["days"]
        hit_map = {int(d): i for i, d in enumerate(days)}
        want = {hit_map[int(dd)] for dd in sub["dp_date"].unique()
                if int(dd) in hit_map}
        if not want:
            continue
        feats = compute_code(cm, want)
        if not len(feats):
            continue
        atr_map = {int(k): float(v)
                   for k, v in sub.groupby("dp_date")["atr_pct"].first().items()}
        out = normalize(feats, atr_map).rename(columns={"day": "dp_date"})
        out["code"] = code
        frames.append(out)
    res = pd.concat(frames, ignore_index=True)
    res.to_parquet(OUT, index=False)
    log(f"写入 {OUT}: {res.shape}；用时 {time.time()-t0:.0f}s")
    t6c = [c for c in res if c.startswith("t6_")]
    for c in t6c:
        log(f"  {c}: NaN {res[c].isna().mean():.1%}")


if __name__ == "__main__":
    main()
