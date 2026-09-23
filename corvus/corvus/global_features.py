"""外围行情特征构建（Global Context Model 第一轮，C2 纯外围特征集）。

防前视铁律执行（F-v2.0 修复 1）：
- 对每个 A 股交易日 t，decision_time = t 08:30（北京时间）。
- 每序列取「available_at ≤ decision_time」的最新已完成 bar——是否完成/可见完全由
  available_at 与 visibility.py 的 session 规则决定，不再额外要求 event_date < t
  （旧条件会把「最新一晚美股收盘」错误排除：美股 t-1(美东) 收盘对应北京时间 t 凌晨，
  其 available_at ≤ t 08:30，应当可用）。
- 选中 bar 抽查复核 visibility.is_available（违反直接抛错，实验作废保护）；
  隔夜美股 bar 必须进入 t 的 8:30 特征（断言回归测试，见 _assert_overnight_us）。
- 缺失绝不填 0：HSTECH 2020-08 前等真实缺失一律 NaN + missing 标志列。
- 特征只用各序列自身历史 bar 计算回看量，禁止加入 A 股/候选池/HMM 等信息。
"""
import numpy as np
import pandas as pd

from .db import get_conn
from .market_collect import load_registry
from .visibility import is_available

# 已入库的 13 个序列（KS11/USDCNH 等未入库序列不使用，绝不拿其他序列冒充）
INSTRUMENTS = ["SPX", "NDX", "DJI", "VIX", "HSI", "HSTECH", "N225",
               "US10Y", "US2Y", "USDCNY_BOC", "WTI", "GOLD", "HSCE"]

# 需要保留最新可见 close 水平的序列（VIX 水平、美债 2s10s 计算用）
LEVEL_INSTRUMENTS = ["VIX", "US10Y", "US2Y"]

US3 = ["SPX", "NDX", "DJI"]                      # 美股三指数
SYNC6 = ["SPX", "NDX", "DJI", "HSI", "HSTECH", "N225"]  # 跨市场同步大跌计数域


def load_bars(conn, instrument_id):
    """单序列已完成 bar（EXCLUDED 除外），按 event_time 升序。"""
    df = pd.read_sql_query(
        """select event_time, available_at, open, high, low, close
           from market_observation
           where instrument_id = ? and quality_status <> 'EXCLUDED'
           order by event_time""",
        conn, params=(instrument_id,))
    df["event_date"] = df["event_time"].str[:10]
    # 统一转 UTC 纳秒整数，便于与决策时刻做向量化比较。
    # 注意 pandas 3.0 会推断非纳秒分辨率（astype("int64") 得到微秒），
    # 必须用 Timestamp.value（恒为纳秒）逐元素转换。
    utc = pd.to_datetime(df["available_at"], utc=True)
    df["avail_ns"] = utc.map(lambda x: x.value)
    return df


def _series_features(bars):
    """在序列自身 bar 序列上预计算每个 bar 位置的回看特征（与交易日无关）。"""
    c = bars["close"].to_numpy(dtype=float)
    o = bars["open"].to_numpy(dtype=float)
    h = bars["high"].to_numpy(dtype=float)
    l = bars["low"].to_numpy(dtype=float)
    n = len(bars)
    r = c[1:] / c[:-1] - 1.0  # r[m-1] = bar m 相对前一根的收益

    def ratio(k):
        out = np.full(n, np.nan)
        if n > k:
            out[k:] = c[k:] / c[:-k] - 1.0
        return out

    sr = pd.Series(r)
    vol = np.concatenate([[np.nan], sr.rolling(20, min_periods=15).std().to_numpy()])
    # 最大回撤：close 相对历史高点的最小偏离（20 bar 窗口）
    runmax = np.fmax.accumulate(c)
    dd = c / runmax - 1.0
    mdd = pd.Series(dd).rolling(20, min_periods=20).min().to_numpy()
    gap = np.full(n, np.nan)
    if n > 1:
        gap[1:] = o[1:] / c[:-1] - 1.0  # 隔夜缺口 open/prev_close
    rng = np.full(n, np.nan)
    ok = ~(np.isnan(h) | np.isnan(l))
    rng[ok] = (h[ok] - l[ok]) / c[ok]   # 日内振幅 (high-low)/close
    w5 = np.concatenate([[np.nan], sr.rolling(5, min_periods=5).min().to_numpy()])

    return pd.DataFrame({
        "ret_1": np.concatenate([[np.nan], r]),
        "ret_3": ratio(3),
        "ret_5": ratio(5),
        "ret_20": ratio(20),
        "vol_20": vol,
        "mdd_20": mdd,
        "gap": gap,
        "range": rng,
        "worst_5": w5,                       # 近 5 日最差单日收益
    })


def _latest_visible_map(bars, trade_dates, dec_ns):
    """每交易日 → 最新可见 bar 位置（-1 = 无）与数据陈旧 bar 数。

    F-v2.0 修复 1：只按 available_at ≤ decision_time 选最新已完成 bar，不再额外
    要求 event_date < t——旧条件会把最新一晚美股收盘（北京时间 t 凌晨才完成）
    错误排除。是否完成/可见完全由 available_at 与 session 规则决定。
    陈旧数 = event_date < t 但 08:30 尚不可见的 bar 数（FRED 发布延迟等）。
    """
    ed = bars["event_date"].to_numpy()
    avail = bars["avail_ns"].to_numpy()   # 随 event_time 单调不减
    idx = np.full(len(trade_dates), -1, dtype=np.int64)
    stale = np.full(len(trade_dates), np.nan)
    for i, t in enumerate(trade_dates):
        j = int(np.searchsorted(avail, dec_ns[t], side="right")) - 1  # 最新可见 bar
        if j < 0:
            continue
        idx[i] = j
        pos_lt = int(np.searchsorted(ed, t, side="left"))  # event_date < t 的 bar 数
        stale[i] = max(pos_lt - 1 - j, 0)
    return idx, stale


def _assert_overnight_us(bars, trade_dates, idx, inst, tail=60):
    """回归断言（F-v2.0 修复 1）：最近 tail 个交易日中，绝大多数日子选中的必须是
    event_date == t 的 bar，即最新一晚美股收盘必须进入 t 的 8:30 特征。"""
    ed = bars["event_date"].to_numpy()
    recent = [i for i in range(max(0, len(trade_dates) - tail), len(trade_dates))
              if idx[i] >= 0]
    if not recent:
        return
    same_day = sum(1 for i in recent if ed[int(idx[i])] == trade_dates[i])
    assert same_day / len(recent) >= 0.9, \
        f"{inst} 隔夜 bar 未进入 8:30 特征：仅 {same_day}/{len(recent)} 天选中当日 bar"


def _verify_visibility_sample(bars, reg_row, trade_dates, idx, k=80, seed=42):
    """抽查复核：选中 bar 必须过 is_available，违反说明可见性推导有漏洞，直接抛错。"""
    cand = [i for i in range(len(trade_dates)) if idx[i] >= 0]
    if not cand:
        return
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(cand), size=min(k, len(cand)), replace=False)
    inst = reg_row["instrument_id"]
    for p in pick:
        i = cand[int(p)]
        row = bars.iloc[int(idx[i])]
        decision = f"{trade_dates[i]}T08:30:00+08:00"
        assert is_available(dict(row), decision, reg_row), \
            f"{inst} {trade_dates[i]} 可见性复核失败：选中的 bar 当时不可见"


def _cross_features(f):
    """跨市场聚合特征（全部由单序列回看特征推导，无新增信息源）。"""
    us = f[[f"{i}_ret_1" for i in US3]]
    f["us_three_down_cnt"] = (us < -0.015).sum(axis=1)   # 三指数隔夜齐跌>1.5% 的个数
    f["us_down_mean"] = us.where(us < 0).mean(axis=1)    # 下跌幅度横截面均值
    f["us_down_max"] = us.min(axis=1)                    # 最大跌幅
    f["ust_2s10s"] = f["US10Y_lvl"] - f["US2Y_lvl"]      # 美债 10Y-2Y 利差
    f["us10y_chg1_bp"] = f["US10Y_ret_1"] * f["US10Y_lvl"] * 100.0  # 10Y 单日变化(bp)
    f["us2y_chg1_bp"] = f["US2Y_ret_1"] * f["US2Y_lvl"] * 100.0
    sync = f[[f"{i}_ret_1" for i in SYNC6]]
    f["sync_crash_cnt"] = (sync <= -0.02).sum(axis=1)    # 单日跌≥2% 的市场数
    f["sync_markets_n"] = sync.notna().sum(axis=1)       # 当日有数据的市场数
    miss_cols = [f"{i}_missing" for i in INSTRUMENTS]
    f["n_missing"] = f[miss_cols].sum(axis=1)
    f["any_missing"] = (f["n_missing"] > 0).astype(float)
    stale_cols = [f"{i}_stale" for i in INSTRUMENTS]
    f["max_stale"] = f[stale_cols].max(axis=1)   # 各序列陈旧 bar 数最大值（NaN 感知）
    return f


def build_features(trade_dates, conn=None):
    """对交易日列表构建全量外围特征宽表（一行一个 trade_date）。"""
    own = conn is None
    conn = conn or get_conn()
    try:
        reg = load_registry()
        dec_ns = {t: pd.Timestamp(t, tz="Asia/Shanghai").replace(hour=8, minute=30)
                      .tz_convert("UTC").value for t in trade_dates}
        frames = []
        for inst in INSTRUMENTS:
            bars = load_bars(conn, inst)
            sfeat = _series_features(bars)
            idx, stale = _latest_visible_map(bars, trade_dates, dec_ns)
            row = dict(reg[inst], instrument_id=inst)
            _verify_visibility_sample(bars, row, trade_dates, idx)
            if inst in US3:
                _assert_overnight_us(bars, trade_dates, idx, inst)
            pos = np.maximum(idx, 0)  # idx=-1 处的取值随后被掩掉
            dfi = pd.DataFrame(
                {f"{inst}_{c}": np.where(idx >= 0, sfeat[c].to_numpy()[pos], np.nan)
                 for c in sfeat.columns},
                index=trade_dates)
            dfi[f"{inst}_missing"] = (idx < 0).astype(float)
            dfi[f"{inst}_stale"] = np.where(idx >= 0, stale, np.nan)
            if inst in LEVEL_INSTRUMENTS:
                dfi[f"{inst}_lvl"] = np.where(
                    idx >= 0, bars["close"].to_numpy(dtype=float)[pos], np.nan)
            frames.append(dfi)
        f = pd.concat(frames, axis=1)
        f = _cross_features(f)
        f.index.name = "trade_date"
        return f.reset_index()
    finally:
        if own:
            conn.close()


def feature_columns(feat):
    """C2 入模特征列（排除 trade_date，全部为纯外围特征）。"""
    return [c for c in feat.columns if c != "trade_date"]
