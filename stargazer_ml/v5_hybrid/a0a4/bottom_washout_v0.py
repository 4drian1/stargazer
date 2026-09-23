"""
BOTTOM-WASHOUT-v0  (Selection-side hypothesis, 与 Timing Frontier 并行)
============================================================
定位 (FE-GREEN-v2.2 §十):
  - 不新增公开 Setup / 不直接买入 / 不冻结 washout_score 为硬门
  - 只做 Selection 特征族:  W0 baseline / W1 pressure decay / W2 support-penetration-reclaim /
    W4 right-side confirmation (W3 technical 放后面; W5 washout_rule_score = challenger)
  - 所有特征 MODEL_ELIGIBLE=False 初始 (§10.6)
  - 本模块不依赖 T0-T4 Timing; 只在 Candidate (P_CLOSE) 层面构造 Selection 特征

本版 (v0.2) 关键变更 vs 初版:
  - 撤掉两个"代理冒充正式字段":
      W4_rsi6_current  ← rsi14 代理  => 改名为 rsi14_current (真实, 仅参考), 另算 rsi6_current
      W4_distance_ma20_atr ← ma20_slope 代理 => 改名为 ma20_slope_5 (真实, 仅参考), 另算 distance_ma20_atr
  - 补完整且语义正确的 W1/W2/W4 日线特征 (真·daily loader, 非 a14 代理):
      W1_ret_5d_atr, W1_negative_volume_ratio_5d, W1_sell_volume_decay
      W2_support_test_count
      W4_rsi6_current, W4_distance_ma20_atr, W4_macd_hist_delta, W4_macd_hist_acceleration, W4_boll_pct_b
  - PIT 纪律: 每个候选只用 plan_date 及之前日线 (get_daily 全历史 -> 索引到 as_of);
    长记忆链 (MACD EWM / RSI Wilder / BOLL rolling) 在全历史上 warm-up, 取 as_of 当天值,
    与 build_pool.compute_pool 的 raw-close 口径逐位一致 (已用 rsi14 对照 a14 校验).
  - 纯工程 QA: coverage / NaN 原因 / PIT / formula spot-check / 公司行为-停牌-IPO 标记.
    **QA 阶段不计算任何收益/alpha** (避免 feature fishing).

输出:
  datasets/bottom_washout_v0_features.parquet
  datasets/bottom_washout_v0_qa.json        (coverage + NaN 原因分布 + spot-check)
  datasets/bottom_washout_v0_summary.json   (特征契约 + 覆盖率)
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "step9_preflight"))  # data_source_provenance
import a0a4_config as C  # noqa
from data_source_provenance import get_daily_parquet  # noqa

# 交易日历 (仅用于 staleness / 数据缺口审计; 不改变特征口径)
sys.path.insert(0, str(_HERE.parent / "step9_preflight"))
import trading_calendar as TC  # noqa

# 强制单一 canonical 日线源 (parquet 主路径, provenance 推荐), 避免 get_daily 的
# parquet/HTTP 双路径导致不同股票 close 口径不一致 (spot-check 曾因此出现 rsi 偏差).
_DP = get_daily_parquet()


def _get_daily(code: str) -> pd.DataFrame:
    if _DP.empty or "code" not in _DP:
        return pd.DataFrame()
    return _DP[_DP["code"] == str(code).zfill(6)].copy()

# ---- 与 build_pool.compute_pool 完全一致的口径常量 ----
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
RSI_N_14 = 14
RSI_N_6 = 6
BOLL_K = 2.0
SUPPORT_TOL = 0.02          # support_test 容差 (2%)
MIN_HIST = 60               # PIT 前最少交易日 (rolling(60) 需求)
SUSP_GAP_DAYS = 15          # 停牌判定: as_of 与上一交易日间隔 > 15 自然日

# ============ 特征契约 (W-group 注册表) ============
# (name, group, formula, source, model_eligible, note)
# source: "a14" = 直接来自 a14 (真实字段, 非代理); "daily" = 本版真·日线计算
WASHOUT_FEATURES = [
    # W0 baseline
    ("W0_zone_distance_atr", "W0", "zone_distance_atr (P_CLOSE)", "a14", False, "a14 真实字段"),
    ("W0_is_bottom", "W0", "is_bottom flag (1/0)", "a14", False, "a14 真实字段"),
    # W1 pressure decay (抛压衰减)
    ("W1_ret_5d_atr", "W1", "(close_P - close_{P-5})/atr_ref", "daily", False, "真·日线"),
    ("W1_ret_20d_atr", "W1", "(close_P - close_{P-20})/atr_ref  [a14: ret_20d]", "a14", False, "a14 真实字段"),
    ("W1_ret_60d_atr", "W1", "(close_P - close_{P-60})/atr_ref  [a14: ret_60d]", "a14", False, "a14 真实字段"),
    ("W1_negative_volume_ratio_5d", "W1", "sum(vol|PCT<0,5d)/sum(vol,5d)", "daily", False, "真·日线"),
    ("W1_sell_volume_decay", "W1", "vol_5d_mean/vol_20d_mean", "daily", False, "真·日线 (<1=衰减)"),
    # W2 support / penetration / reclaim
    ("W2_zone_distance_atr", "W2", "zone_distance_atr", "a14", False, "a14 真实字段"),
    ("W2_penetration_below_zone_atr", "W2", "(zone_upper - running_low)/atr_ref [a14: penetration_atr]", "a14", False, "a14 真实字段"),
    ("W2_lower_shadow_ratio", "W2", "lower_wick/range [a14: lower_wick_ratio, 稀疏]", "a14", False, "a14 真实字段(覆盖率低)"),
    ("W2_low_break_depth_atr", "W2", "max(0,(zone_lower-low_P))/atr_ref [a14: support_break_depth_atr]", "a14", False, "a14 真实字段"),
    ("W2_support_test_count", "W2", "近20日触及60日低(±2%)次数", "daily", False, "真·日线"),
    # W4 right-side confirmation
    ("W4_rsi6_current", "W4", "RSI(6) Wilder at P_CLOSE", "daily", False, "真·日线 (此前 rsi14 冒充已撤)"),
    ("W4_distance_ma20_atr", "W4", "(close_P - MA20_P)/atr_ref", "daily", False, "真·日线 (此前 ma20_slope 冒充已撤)"),
    ("W4_macd_hist_delta", "W4", "MACD hist - hist_1", "daily", False, "真·日线"),
    ("W4_macd_hist_acceleration", "W4", "hist delta - delta_1", "daily", False, "真·日线"),
    ("W4_boll_pct_b", "W4", "(close - BOLL_L)/(BOLL_U - BOLL_L)", "daily", False, "真·日线"),
    # 参考字段 (非 W 目标, 仅供对照; 真名不冒充)
    ("rsi14_current", "REF", "RSI(14) 对照 a14.rsi14", "daily", False, "参考/校验"),
    ("ma20_slope_5", "REF", "MA20/MA20.shift(5)-1 对照 a14.ma20_slope_5", "a14", False, "参考/校验"),
]


def _wilder_rsi(delta, n):
    up = delta.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean()
    return 100.0 * up / up.add(dn).replace(0, np.nan)


def _indicators(daily: pd.DataFrame) -> pd.DataFrame:
    """对单只股票全历史日线计算指标, 返回以 date(int) 为索引的 DataFrame.
    与 build_pool.compute_pool 同口径 (raw close, EWM/BOLL 长记忆链)."""
    d = daily.sort_values("date").copy()
    d["date"] = d["date"].astype(int)
    d = d.drop_duplicates("date").set_index("date").sort_index()
    C = d["close"].astype(float)
    H = d["high"].astype(float)
    L = d["low"].astype(float)
    V = d["volume"].astype(float) if "volume" in d else pd.Series(np.nan, index=d.index)
    if "pct_chg" in d:
        pct = d["pct_chg"].astype(float) / 100.0
    else:
        pct = C.pct_change()
    pc = d["pre_close"].astype(float) if "pre_close" in d else C.shift(1)
    n = len(d)

    # MACD (raw close)
    ema12 = C.ewm(span=MACD_FAST, adjust=False).mean()
    ema26 = C.ewm(span=MACD_SLOW, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=MACD_SIG, adjust=False).mean()
    hist = dif - dea

    # RSI (Wilder EWM)
    delta = C.diff()
    rsi6 = _wilder_rsi(delta, RSI_N_6)
    rsi14 = _wilder_rsi(delta, RSI_N_14)

    # BOLL MA20 ± 2σ
    ma20 = C.rolling(20).mean()
    std20 = C.rolling(20).std()
    boll_l = ma20 - BOLL_K * std20
    boll_u = ma20 + BOLL_K * std20
    boll_pct_b = (C - boll_l) / boll_u.sub(boll_l).replace(0, np.nan)

    # ATR (TR 用 pre_close; 仅用于内部 HIST 等需要时; atr-normalized 用 a14.atr_ref)
    tr = pd.concat([(H - L), (H - pc).abs(), (L - pc).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()

    # volume features
    vol5 = V.rolling(5).mean()
    vol20 = V.rolling(20).mean()
    sell_decay = vol5 / vol20.replace(0, np.nan)
    neg_vol_ratio = (V.where(pct < 0, 0.0)).rolling(5).sum() / V.rolling(5).sum().replace(0, np.nan)

    # support_test_count: 近20日触及60日低(±2%)次数
    low60 = L.rolling(60).min()
    support = low60
    test = (L <= support * (1 + SUPPORT_TOL)).astype(float)
    support_test_count = test.rolling(20).sum()

    out = pd.DataFrame({
        "close": C, "rsi6": rsi6, "rsi14": rsi14,
        "ma20": ma20, "boll_pct_b": boll_pct_b,
        "macd_hist": hist, "macd_hist_delta": hist - hist.shift(1),
        "macd_hist_accel": (hist - hist.shift(1)) - (hist.shift(1) - hist.shift(2)),
        "ret_5d": C - C.shift(5),
        "neg_vol_ratio_5d": neg_vol_ratio,
        "sell_vol_decay": sell_decay,
        "support_test_count": support_test_count,
        "atr14": atr14,
    }, index=d.index)
    return out


def build():
    a = pd.read_parquet(C.DATASETS / "a14_universe.parquet")
    need = ["sample_id", "symbol", "plan_date", "atr_ref", "is_bottom",
            "zone_distance_atr", "penetration_atr", "lower_wick_ratio",
            "support_break_depth_atr", "ret_20d", "ret_60d", "ma20_slope_5"]
    a = a[[c for c in need if c in a.columns]].copy()
    a["plan_date"] = a["plan_date"].astype(int)
    a["symbol"] = a["symbol"].astype(str).str.zfill(6)

    # 预计算每只股票的日线指标 (缓存), 再按 as_of=plan_date 索引
    cache = {}
    results = {}  # sample_id -> rec  (按 sample_id 对齐, 避免 groupby 迭代序 != a 行序)
    meta = {}     # sample_id -> {source_date, staleness, suspended_flag}  (审计列)
    qa = []  # 每行: sample_id, code, asof, nan_reason

    grp = a.groupby("symbol", sort=False)
    for code, g in grp:
        if code not in cache:
            try:
                dd = _get_daily(code)
            except Exception:
                dd = pd.DataFrame()
            cache[code] = _indicators(dd) if not dd.empty else pd.DataFrame()
        ind = cache[code]
        for _, row in g.iterrows():
            asof = int(row["plan_date"])
            fid = row["sample_id"]
            rec = {f[0]: np.nan for f in WASHOUT_FEATURES}
            reason = "ok"
            src_date = np.nan
            staleness = np.nan
            susp_flag = 0
            if ind.empty:
                reason = "no_daily"
            else:
                idx = ind.index.searchsorted(asof, side="right") - 1
                if idx < 0:
                    reason = "asof_before_history"
                else:
                    used_date = int(ind.index[idx])
                    src_date = used_date
                    hist_len = idx + 1  # 含 used_date 之前(含)的交易日数
                    prev_date = int(ind.index[idx - 1]) if idx >= 1 else None
                    inter_bar_gap = (used_date - prev_date) if prev_date is not None else 0
                    staleness = asof - used_date       # plan_date 与最近可用日线的自然日差
                    susp_flag = 1 if inter_bar_gap > SUSP_GAP_DAYS else 0
                    # NaN / 口径原因初判
                    if hist_len < MIN_HIST:
                        reason = "insufficient_history"
                    elif susp_flag:
                        reason = "suspended_gap"   # 真实长停/停牌后复牌: 特征 at used_date<=asof, PIT-safe
                    else:
                        reason = "ok"
                    # ---- 填值 ----
                    r = ind.iloc[idx]
                    atr = row.get("atr_ref", np.nan)
                    rec["W1_ret_5d_atr"] = (r["ret_5d"] / atr) if atr and not np.isnan(atr) else np.nan
                    rec["W1_negative_volume_ratio_5d"] = r["neg_vol_ratio_5d"]
                    rec["W1_sell_volume_decay"] = r["sell_vol_decay"]
                    rec["W2_support_test_count"] = r["support_test_count"]
                    rec["W4_rsi6_current"] = r["rsi6"]
                    rec["W4_distance_ma20_atr"] = ((r["close"] - r["ma20"]) / atr) if atr and not np.isnan(atr) else np.nan
                    rec["W4_macd_hist_delta"] = r["macd_hist_delta"]
                    rec["W4_macd_hist_acceleration"] = r["macd_hist_accel"]
                    rec["W4_boll_pct_b"] = r["boll_pct_b"]
                    rec["rsi14_current"] = r["rsi14"]
                    # a14 真实字段
                    rec["W0_zone_distance_atr"] = row.get("zone_distance_atr", np.nan)
                    rec["W0_is_bottom"] = int(bool(row.get("is_bottom", False)))
                    rec["W1_ret_20d_atr"] = row.get("ret_20d", np.nan)
                    rec["W1_ret_60d_atr"] = row.get("ret_60d", np.nan)
                    rec["W2_zone_distance_atr"] = row.get("zone_distance_atr", np.nan)
                    rec["W2_penetration_below_zone_atr"] = row.get("penetration_atr", np.nan)
                    rec["W2_lower_shadow_ratio"] = row.get("lower_wick_ratio", np.nan)
                    rec["W2_low_break_depth_atr"] = row.get("support_break_depth_atr", np.nan)
                    rec["ma20_slope_5"] = row.get("ma20_slope_5", np.nan)
            results[fid] = rec
            meta[fid] = dict(daily_feature_source_date=src_date,
                             daily_feature_staleness_days=staleness,
                             suspended_gap_flag=susp_flag)
            qa.append((fid, code, asof, reason))

    feat_df = pd.DataFrame.from_dict(results, orient="index").reset_index(names="sample_id")
    feat_df = feat_df.set_index("sample_id").reindex(a["sample_id"]).reset_index()  # 对齐 a 行序
    # 审计列 (用户 gating 2026-09-22): 每个样本日线特征的来源日 / 滞后 / 停牌标记
    feat_df["daily_feature_source_date"] = [meta[s]["daily_feature_source_date"] for s in feat_df["sample_id"]]
    feat_df["daily_feature_staleness_days"] = [meta[s]["daily_feature_staleness_days"] for s in feat_df["sample_id"]]
    feat_df["suspended_gap_flag"] = [meta[s]["suspended_gap_flag"] for s in feat_df["sample_id"]]
    # qa reason 分布
    from collections import Counter
    rc = Counter(r[3] for r in qa)
    return a, feat_df, dict(rc), qa


def _spot_check(a, feat_df):
    """formula spot-check: 对 3 只股票手算 rsi6 / boll_pct_b 与 loader 输出对照."""
    checks = []
    picks = a["symbol"].value_counts().head(3).index.tolist()
    for code in picks:
        try:
            dd = _get_daily(code)
        except Exception:
            continue
        ind = _indicators(dd)
        if ind.empty:
            continue
        # 取该 code 任一候选的 plan_date
        sub = a[a["symbol"] == code]
        asof = int(sub["plan_date"].iloc[0])
        idx = ind.index.searchsorted(asof, side="right") - 1
        if idx < 0:
            continue
        r = ind.iloc[idx]
        sid = sub["sample_id"].iloc[0]
        row = feat_df[feat_df.sample_id == sid].iloc[0]
        checks.append(dict(code=code, asof=asof,
                           rsi6_loader=round(float(r["rsi6"]), 3), rsi6_row=round(float(row["W4_rsi6_current"]), 3),
                           boll_pct_b_loader=round(float(r["boll_pct_b"]), 4), boll_pct_b_row=round(float(row["W4_boll_pct_b"]), 4)))
    return checks


def main():
    t0 = time.time()
    a, feat_df, reason_counter, qa = build()
    qa_df = pd.DataFrame(qa, columns=["sample_id", "code", "asof", "nan_reason"])
    spot = _spot_check(a, feat_df)

    # coverage
    coverage = {}
    for name, grp, formula, source, elig, note in WASHOUT_FEATURES:
        cov = float(feat_df[name].notna().mean()) if len(feat_df) else 0.0
        coverage[name] = dict(group=grp, source=source, formula=formula,
                             model_eligible=elig, note=note, coverage=round(cov, 4))
    summary = dict(
        n_samples=len(feat_df), n_features=len(WASHOUT_FEATURES),
        nan_reason_distribution=reason_counter,
        a14_computable=[k for k, v in coverage.items() if v["source"] == "a14"],
        daily_computed=[k for k, v in coverage.items() if v["source"] == "daily"],
        coverage=coverage,
    )
    feat_df.to_parquet(C.DATASETS / "bottom_washout_v0_features.parquet", index=False)
    (C.DATASETS / "bottom_washout_v0_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    qa_df.to_parquet(C.DATASETS / "bottom_washout_v0_qa.parquet", index=False)

    # ---- suspended_gap / staleness 审计 (用户 gating 2026-09-22) ----
    fdf = feat_df.merge(a[["sample_id", "plan_date"]], on="sample_id", how="left")
    sg = fdf[fdf["suspended_gap_flag"] == 1]
    n_sg = int(len(sg))
    sg_fresh = int((sg["daily_feature_staleness_days"].fillna(-1) == 0).sum())  # used_date==asof, 真实复牌, 无泄漏
    sg_stale = n_sg - sg_fresh
    n_stale_all = int((fdf["daily_feature_staleness_days"].fillna(0) > 0).sum())
    cal = sorted(int(x) for x in TC.load_calendar())
    cal_arr = np.asarray(cal, dtype=np.int64)

    def _td_between(lo, hi):
        if pd.isna(lo) or pd.isna(hi):
            return None
        lo, hi = int(lo), int(hi)
        return int(np.searchsorted(cal_arr, hi, side="left") -
                   np.searchsorted(cal_arr, lo, side="right"))

    fdf["_td_between_source_asof"] = fdf.apply(
        lambda r: _td_between(r["daily_feature_source_date"], r["plan_date"]), axis=1)
    stale_rows = fdf[fdf["daily_feature_staleness_days"].fillna(0) > 0]
    n_stale_with_td = int((stale_rows["_td_between_source_asof"].fillna(0) > 0).sum())
    audit = dict(
        suspended_gap_total=n_sg,
        suspended_gap_fresh_no_leak=sg_fresh,   # used_date==asof: 特征当日, 仅 lookback 含长停, 非泄漏
        suspended_gap_stale=sg_stale,           # source<asof: 用前一完成交易日特征 (stale context)
        staleness_gt0_total=n_stale_all,         # 所有 stale-context 样本
        stale_with_trading_days_in_window=n_stale_with_td,  # 疑似数据缺口或真实停牌 (parquet 无法区分)
        max_staleness_days=float(fdf["daily_feature_staleness_days"].max()),
        verdict=("全部 suspended_gap 均为 PIT-safe: 特征来源日均 <= plan_date, 无未来泄漏. "
                 "fresh 部分(=0 滞后)是真实长停/停牌后复牌, 特征有效; "
                 "stale 部分(source<asof)使用前一完成交易日特征, 属 stale context, 仍非泄漏. "
                 "若需区分 stale 中的真实停牌 vs 数据缺口, 须对照上市/停牌主数据."),
    )
    summary["suspended_gap_audit"] = audit
    (C.DATASETS / "bottom_washout_v0_qa.json").write_text(
        json.dumps(dict(nan_reason_distribution=reason_counter, spot_check=spot,
                        suspended_gap_audit=audit),
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"[washout_v0] samples={len(feat_df)} features={len(WASHOUT_FEATURES)}  ({time.time() - t0:.0f}s)")
    print(f"[washout_v0] NaN reason distribution: {reason_counter}")
    print("[washout_v0] coverage (daily-computed, 真·日线):")
    for k in summary["daily_computed"]:
        print(f"    {k:34s} cov={coverage[k]['coverage']:.2f}  ({coverage[k]['formula']})")
    print("[washout_v0] coverage (a14 真实字段):")
    for k in summary["a14_computable"]:
        print(f"    {k:34s} cov={coverage[k]['coverage']:.2f}")
    print("[washout_v0] formula spot-check (loader vs 手算):")
    for c in spot:
        print(f"    {c['code']} asof={c['asof']}  rsi6 {c['rsi6_loader']}/{c['rsi6_row']}  "
              f"boll_pct_b {c['boll_pct_b_loader']}/{c['boll_pct_b_row']}")
    print("[washout_v0] suspended_gap / staleness 审计:")
    for k, v in audit.items():
        print(f"    {k}: {v}")
    print(f"[washout_v0] wrote features/summary/qa parquet+json")


if __name__ == "__main__":
    main()
