"""build_pool.py — GREEN-WATCH-v0 盘后池构建（历史回填 + 每日盘后增量）。

规格来源：
  C:/Users/22107/Desktop/交易/选股标准.txt（GREEN-WATCH-v1 全文，下称"选股标准"）
  v5_hybrid/latr_v0/latr_candidate_engine_spec.md（字段定义精神）

口径纪律：
  - 全部条件只用 T 日及以前数据（MA20Slope5 用 T-5..T，ATR_ref 用 T-4..T 等）。
  - 自由裁量处（MACD/RSI/BOLL 参数、swing 近似、板块代理）均用最常规口径并在
    DEVIATIONS 中逐条记录。
  - 评分/gate/tier/KEEP 逻辑为终裁冻结件：回填与增量共用同一段 compute_pool，
    任何修改必须两侧同时生效（增量路径只是输入窗口与产出不同，指标代码逐位一致）。

输出：pool_daily.parquet（一行一日一票，含两 Setup 分数/门槛/分层/全部规格字段
  以及 pre_close/close 原始价）；增量模式另输出 pool_latest.json（同目录）。

运行：
  d:/stargazer/.venv/Scripts/python.exe build_pool.py [--force]          # 历史回填
  d:/stargazer/.venv/Scripts/python.exe build_pool.py --date 20260916    # 每日增量
回填幂等：已存在且非 --force 则直接复用；增量幂等：旧文件中目标日期行先剔除再 append。
"""
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# ---- ③.1 统一交易日历 / 快照指纹 (与 LATR loader 同源) ----
# 约束1: effective_trade_date 只能来自 canonical 交易日历。
# canonical 不可用时 HARD ERROR, 绝不静默回退本地日历 (错位比报错危险得多)。
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "step9_preflight"))
    from trading_calendar import (load_calendar as _tc_load_calendar,  # noqa
                                 next_trading_day as _tc_next,
                                 data_epoch_fingerprint as _tc_snapshot)
    _HAVE_CANON_CAL = True
except Exception as _cal_err:  # noqa
    raise RuntimeError(
        "canonical trading_calendar 不可用; 拒绝静默回退本地日历 "
        "(③约束1: effective_trade_date 必须来自 canonical 日历, 错一天会错位 Candidate/Minute Replay)。"
        " 若需离线运行, 请提供经 hash 校验一致的 local mirror 再覆盖。"
    ) from _cal_err

BASE = Path(__file__).resolve().parent
_REPO_ROOT = BASE.parents[2]  # green_watch_v0 → v5_hybrid → stargazer_ml → 仓库根


def _bars_glob() -> str:
    """日线目录：env STARGAZER_DAILY_PARTS（mquote 同名先例）→ 仓库根 data_daily_parts
    （监控机布局）→ 本机 D:/a股数据 缺省。返回目录（调用方拼 part_*.parquet）。"""
    import os
    v = os.environ.get("STARGAZER_DAILY_PARTS")
    if v:
        return str(Path(v) / "part_*.parquet")
    for cand in (_REPO_ROOT / "data_daily_parts", Path("D:/a股数据/data_daily_parts")):
        if cand.exists():
            return str(cand / "part_*.parquet")
    return "D:/a股数据/data_daily_parts/part_*.parquet"  # 都不存在：保持缺省让报错信息可识别


BARS_GLOB = _bars_glob()
OUT = BASE / "pool_daily.parquet"
LATEST_JSON = BASE / "pool_latest.json"

POOL_START = 20210101          # 标签自 2021-01-04 起，池子从 2021 年开始构建
MV_LO, MV_HI = 2.5e9, 1.0e10   # 总市值 25–100 亿（元）
DAILY_WINDOW_DAYS = 120        # 增量模式向前加载的自然日数（≈82 个交易日，覆盖 rolling(60)；
                               # EWM/BOLL 长记忆链另用全历史 close 矩阵，见 load_close_hist）

# ---- 自由裁量口径（最常规参数，记入报告偏离清单）----
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9   # EMA12/26/DEA9，hist = DIF - DEA
RSI_N = 14                                    # Wilder 平滑
BOLL_K = 2.0                                  # MA20 ± 2σ
DIP_BAND = 0.15                               # MA20 容差 / 低点结构容差（ATR）
EVIDENCE_MIN = 1                              # BOTTOM_SETUP 止跌证据至少 1 条
NEAR_MISS_LO = 45.0                           # NEAR_MISS 下沿
BREAK_VOL_MULT = 2.0                          # "明显放量"= volume > 2×前5日均量
SUPPORT_TOL = 0.5                             # 支撑距离计分容差（ATR）

DEVIATIONS = [
    "MACD：EMA12/EMA26/DEA9，hist=DIF-DEA（选股标准未给公式，用最常见口径）",
    "RSI14：Wilder 指数平滑；BOLL：MA20±2σ(20)",
    "swing high/low 无逐笔拐点算法：用 High60/Low60 及滚动窗口近似，"
    "ImpulseReturn=High60/Low60-1、PullbackATR=(High60-Close)/ATR_ref（近似）",
    "BOTTOM_SETUP 硬门槛按任务书口径实现：DistLow60<=6% 且 9 条止跌证据>=1；"
    "选股标准第八节 PULLBACK_BOTTOM（Impulse>=10% 分支）不作独立硬门槛，"
    "其特征仅作为评分字段保存（偏离说明见报告）",
    "Anchored VWAP / POC / VAL / 成交密集区：日线数据不可得，用 MA20/Low20/"
    "BOLL下轨/Low60 四类支撑近似；INTRADAY 支撑模块满分由 13 分按比例 rescale 到 20",
    "板块/龙头：全市场无 code→行业映射的完整覆盖，用当日全池等权指数作板块代理"
    "（个股 vs 池指数 1d/3d/5d 相对强弱），龙头健康用池指数 5 日跌幅判定",
    "总量市值口径：total_mv 字段（元），25–100亿 = 2.5e9–1.0e10",
    "v1.1 DIP 分解的聚合 flag 数值阈值（PriorStrengthPass/StructureIntact/ClearDistribution/"
    "StructuralBreakdown/LiquidityPass）为第一版近似: 文档只给分解结构、未给具体 cutoff, "
    "故用底层连续量做合理阈值; 待 v1→v1.1 Diff (step2) 与后续 Gate 数据证伪后再校准, "
    "不在此硬编码为市场真理。当前采用 B_med 校准 "
    "(ret_60d>=-0.02 / dist_from_60h<=0.45 / up_vol_exp>=0.8 / recent_lower_low<=5 / "
    "support_break_depth>=1.0&vol>=1.5 / dist_score>=3.5); IMPULSE>=0.10 硬门槛已移除(§6 明示仅为研究阈值)。",
]


# ---------------------------------------------------------------- 数据装载
def load_bars(start=None, with_name=False):
    """加载日线。start=None 全量（回填）；否则只加载 date>=start 的窗口（增量）。

    增量窗口用 pyarrow dataset 过滤器下推，只读需要的行/列。
    """
    import glob
    cols = ["date", "code", "open", "high", "low", "close", "pre_close",
            "volume", "amount", "turnover", "pct_chg", "vol_ratio",
            "total_mv", "float_mv", "is_st"]
    if with_name:
        cols = cols + ["name"]
    files = sorted(glob.glob(BARS_GLOB))
    if start is None:
        parts = [pd.read_parquet(f, columns=cols) for f in files]
        df = pd.concat(parts, ignore_index=True)
        del parts
    else:
        import pyarrow.dataset as ds
        dset = ds.dataset(files, format="parquet")
        df = dset.to_table(columns=cols, filter=ds.field("date") >= start) \
                .to_pandas()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    return df


def pivot(df, col):
    m = df.pivot(index="date", columns="code", values=col)
    if m.dtypes.iloc[0] != bool:
        m = m.astype(np.float32)
    return m


def load_close_hist(start=20180101):
    """全历史收盘价矩阵（date×code，float32），仅含 close 列，供增量模式的
    EWM/BOLL 长记忆链使用；构造口径（过滤起点/排序/pivot/astype）与回填的
    C 矩阵完全一致，保证逐位相同。"""
    import glob
    import pyarrow.dataset as ds
    dset = ds.dataset(sorted(glob.glob(BARS_GLOB)), format="parquet")
    df = dset.to_table(columns=["date", "code", "close"],
                       filter=ds.field("date") >= start).to_pandas()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    return pivot(df, "close")


# -------------------------------------------------- 统一指标/硬条件/评分（冻结件）
def compute_pool(df, pool_start, c_hist=None):
    """[2/5]-[4/5] 计算段：透视 → 指标 → 硬条件 → 评分 → 组装为池表。

    回填与增量共用此函数；调用方只需保证 df 内含 pool_start 当日往前足够的
    预热窗口（回填 2018 年起全量，增量为目标日前 DAILY_WINDOW_DAYS 个自然日，
    需覆盖 rolling(60)），且 df 已按 [code, date] 排序。c_hist 为全历史收盘价
    矩阵（增量模式喂入，保证 EWM/BOLL 长记忆链逐位一致；回填模式为 None）。
    返回的表含全部规格字段 + pre_close/close，rule_version 由调用方设置。
    只保留 date >= pool_start 的行。
    """
    print("[2/5] 透视成 date×code 矩阵并计算统一指标 …")
    O, H, L, C = (pivot(df, c) for c in ("open", "high", "low", "close"))
    PC = pivot(df, "pre_close")
    VOL, AMT = pivot(df, "volume"), pivot(df, "amount")
    TO, VR = pivot(df, "turnover"), pivot(df, "vol_ratio")
    TMV, FMV = pivot(df, "total_mv"), pivot(df, "float_mv")
    ST = pivot(df, "is_st").fillna(False).astype(bool)
    PCT = pivot(df, "pct_chg") / 100.0
    del df

    MA20 = C.rolling(20).mean()
    MA20S5 = MA20 / MA20.shift(5) - 1.0                       # 选股标准 §4.2
    TR = pd.concat([(H - L), (H - PC).abs(), (L - PC).abs()]).groupby(level=0).max()
    ATR14 = TR.rolling(14).mean()                             # exit spec §1：均值口径
    ATR_REF = ATR14.rolling(5).median()                       # 选股标准 §4.1

    # 长记忆链（MACD 的 EWM 递推、RSI 的 Wilder 递推、BOLL 的 rolling std）：
    # ewm 误差按 (1-α)^N 衰减需 ~250 个交易日才能沉到 float32 精度以下，而 pandas
    # 滑动窗口 std 的实现误差随遍历位置累积、加长预热也不收敛。二者要逐位一致
    # 只能喂"与回填同起点"的全历史收盘价矩阵 CH。回填模式 c_hist=None 时 CH 即
    # C 本身（同一对象，数值逐位不变）；增量模式 CH 列为窗口内全部 code、行覆盖
    # 2018 起全历史，算完后把结果 reindex 回窗口日期。除变量名 C→CH 与末尾
    # reindex 外，以下各行公式与回填逐字符一致。
    CH = C if c_hist is None else c_hist.reindex(columns=C.columns)
    EMA12 = CH.ewm(span=MACD_FAST, adjust=False).mean()
    EMA26 = CH.ewm(span=MACD_SLOW, adjust=False).mean()
    DIF = EMA12 - EMA26
    DEA = DIF.ewm(span=MACD_SIG, adjust=False).mean()
    HIST = DIF - DEA

    DELTA = CH.diff()
    UP = DELTA.clip(lower=0).ewm(alpha=1/RSI_N, adjust=False).mean()
    DN = (-DELTA.clip(upper=0)).ewm(alpha=1/RSI_N, adjust=False).mean()
    RSI = 100.0 * UP / (UP + DN).replace(0, np.nan)

    BOLL_STD = CH.rolling(20).std()
    if c_hist is not None:
        DIF = DIF.reindex(C.index)
        DEA = DEA.reindex(C.index)
        HIST = HIST.reindex(C.index)
        RSI = RSI.reindex(C.index)
        BOLL_STD = BOLL_STD.reindex(C.index)
    BOLL_L = MA20 - BOLL_K * BOLL_STD
    BOLL_U = MA20 + BOLL_K * BOLL_STD

    LOW20, LOW60 = L.rolling(20).min(), L.rolling(60).min()
    HIGH20, HIGH60 = H.rolling(20).max(), H.rolling(60).max()
    DIST_LOW60 = C / LOW60 - 1.0
    DIST_LOW20 = C / LOW20 - 1.0
    RPOS20 = (C - LOW20) / (HIGH20 - LOW20).replace(0, np.nan)
    RPOS60 = (C - LOW60) / (HIGH60 - LOW60).replace(0, np.nan)

    # 低点结构（容差 0.15ATR）：最近5日低点 vs 之前5日低点
    REC_LOW = L.rolling(5).min()
    PRI_LOW = REC_LOW.shift(5)
    LOW_RAISE = REC_LOW >= PRI_LOW - DIP_BAND * ATR_REF

    # 量能
    VOL_MA5 = VOL.rolling(5).mean()
    VOL_MED20 = VOL.rolling(20).median()
    AMT_MED20 = AMT.rolling(20).median()
    UP_VMED = VOL.where(PCT > 0).rolling(20, min_periods=5).median()
    DN_VMED = VOL.where(PCT < 0).rolling(20, min_periods=5).median()

    RNG = (H - L).replace(0, np.nan)
    LWICK = ((np.minimum(O, C) - L) / RNG).clip(0, 1)         # 长下影
    CPOS = ((C - L) / RNG).clip(0, 1)                         # 收盘位置/低位承接

    DS3 = PCT.rolling(3).sum()
    DS5 = PCT.rolling(5).sum()
    NEW_LOW = L <= LOW60.shift(1) * (1 + 1e-9)                # 缩量创新低等用
    NEW_LOW_CNT5 = NEW_LOW.rolling(5).sum()

    print("[3/5] Setup 硬条件 …")
    # ---------- BOTTOM_SETUP 止跌证据（9 条，选股标准 §9）----------
    E1 = NEW_LOW & (VOL < VOL_MA5.shift(1))                    # 缩量创新低
    E2 = (HIST < 0) & (HIST > HIST.shift(1))                   # MACD绿柱缩短
    E3 = (RSI.shift(1) < 30) & (RSI >= 30)                     # RSI站回30
    E4 = (C.shift(1) < BOLL_L.shift(1)) & (C >= BOLL_L)        # 收回BOLL下轨
    E5 = (L < LOW20.shift(1)) & (C > LOW20.shift(1))           # 跌破前低快速收回
    E6 = LWICK >= 0.5                                          # 长下影
    E7 = REC_LOW > PRI_LOW                                     # 更高次级低点
    E8 = (VOL_MA5 < VOL_MA5.shift(5)) & (C < MA20)             # 量衰减（下跌中）
    E9 = (CPOS >= 0.6) & (LWICK >= 0.3)                        # 低位承接
    EVID = sum(E.fillna(False).astype(int) for E in (E1, E2, E3, E4, E5, E6, E7, E8, E9))
    BOTTOM_GATE = (DIST_LOW60 <= 0.06) & (EVID >= EVIDENCE_MIN)

    # ---------- INTRADAY_DIP_SETUP（选股标准 §12, v1.0 旧 Hard Gate）----------
    # v1.0 的 MA20S5>0 短斜率硬门槛已在 v1.1 取消; 资格见下方 v1.1 分解 (DIP_ELIGIBLE_v11)。
    MACD_A = (DIF > 0) & (DEA > 0)
    MACD_B = ((DIF / C).abs() <= 0.005) & (HIST >= HIST.shift(1))  # 近零轴且未继续恶化
    BREAK_DOWN = (VOL > BREAK_VOL_MULT * VOL_MA5.shift(1)) & \
                 (C < LOW20.shift(1)) & (C < O)                    # 明显放量破位收不回

    print("[4/5] 板块代理（全池等权指数）与评分 …")
    # 当日全池等权指数（仅资格内票）
    ELIG_BASE = (~ST) & (VOL > 0) & (C > 1) & (C < 1000) & \
                (TMV >= MV_LO) & (TMV <= MV_HI) & LOW60.notna()
    MKT1 = PCT.where(ELIG_BASE).mean(axis=1)                   # 池指数 1d
    MKT3 = MKT1.rolling(3).sum()
    MKT5 = MKT1.rolling(5).sum()
    MKT1X = MKT1.reindex(C.index).ffill()
    MKT3X = MKT3.reindex(C.index).ffill()
    MKT5X = MKT5.reindex(C.index).ffill()

    def bcol(ser):
        """把按日的序列广播成 date×code 矩阵（列对齐）。"""
        return pd.DataFrame(np.repeat(ser.to_numpy()[:, None], C.shape[1], axis=1),
                            index=C.index, columns=C.columns)
    STOCK3 = PCT.rolling(3).sum()

    HIGH20_PREV = HIGH20.shift(5)                              # 最近高点抬升/持平
    HIGH_RAISE = HIGH20 > HIGH20_PREV + DIP_BAND * ATR_REF
    HIGH_FLAT = (HIGH20 >= HIGH20_PREV - DIP_BAND * ATR_REF) & ~HIGH_RAISE

    IMPULSE = HIGH60 / LOW60 - 1.0                             # 近似 ImpulseReturn
    PULLBACK_ATR = (HIGH60 - C) / ATR_REF
    PULLBACK_PCT = C / HIGH60 - 1.0
    PB_DAYS = (C < MA20).rolling(30).sum().clip(0, 30)

    # ========= v1.1 DIP GATE 分解：底层连续特征 + 聚合 flag + stage/双评分 =========
    # 文档: 选股标准 §5–§11, FE §7–§11。短周期斜率(ma5/10/ma20_*/ret_3/5d)降级为连续特征,
    # 不再作 Hard Gate; 资格改为 PriorStrength / StructureIntact / Liquidity 分解。
    to_pct = TO.rank(pct=True, axis=1)                       # 当日资格内分位 (上移供 Gate 复用)
    MA5 = C.rolling(5).mean()
    MA10 = C.rolling(10).mean()
    MA60 = C.rolling(60).mean()
    ma5_slope = MA5 / MA5.shift(5) - 1.0
    ma10_slope = MA10 / MA10.shift(5) - 1.0
    ma20_slope_3 = MA20 / MA20.shift(3) - 1.0
    ma20_slope_10 = MA20 / MA20.shift(10) - 1.0
    ret_3d = DS3
    ret_5d = DS5
    ret_10d = PCT.rolling(10).sum()
    ret_20d = PCT.rolling(20).sum()
    ret_60d = PCT.rolling(60).sum()
    dist_from_20h = C / HIGH20 - 1.0
    dist_from_60h = C / HIGH60 - 1.0
    # F3B Prior Strength 底层连续量
    prior_impulse_return = IMPULSE
    prior_impulse_days = (30 - PB_DAYS).clip(0, 30)          # 近似: 之前处于上升状态的天数
    breakout_strength_atr = (HIGH20 - MA20) / ATR_REF.replace(0, np.nan)
    recent_breakout_flag = (HIGH20 >= HIGH20.rolling(20).max()) & (C > MA20)
    up_vol_expansion = UP_VMED / VOL_MED20.replace(0, np.nan)
    dn_vol_contraction = DN_VMED / VOL_MED20.replace(0, np.nan)
    prior_range_expansion = (HIGH20 - LOW20) / ATR_REF.replace(0, np.nan)
    prior_turnover_strength = AMT / AMT_MED20.replace(0, np.nan)
    # F3C Structure Intact 底层连续量
    dist_confirmed_swing_low_atr = (C - LOW20) / ATR_REF.replace(0, np.nan)
    dist_breakout_platform_atr = (C - LOW60) / ATR_REF.replace(0, np.nan)
    recent_lower_low_count = (L <= LOW20.shift(1)).rolling(5).sum()
    recent_lower_high_count = (H <= HIGH20.shift(1)).rolling(5).sum()
    support_break_depth_atr = ((MA20 - C) / ATR_REF).clip(lower=0)
    support_break_volume_ratio = (VOL / VOL_MA5.shift(1)).replace(0, np.nan)
    distribution_score_raw = (
        (PCT < 0).astype(float) * 1.0 +
        (RSI < 40).astype(float) * 1.0 +
        (C < MA20).astype(float) * 1.0 +
        ((HIGH20 <= HIGH20.shift(5) - DIP_BAND * ATR_REF)).astype(float) * 1.0 +
        NEW_LOW.astype(float) * 1.0
    )
    # 冻结 Zone 距离 (仅用 registry ENABLED 的支撑近似: MA20/LOW20/BOLL_L/LOW60)
    _support_min = pd.concat([MA20, LOW20, BOLL_L, LOW60]).groupby(level=0).min()
    zone_distance_atr = ((C - _support_min) / ATR_REF).clip(lower=0)

    # ---- 聚合 RULE_FLAG (每个都有底层连续量支持, 非人工布尔) ----
    # 选股标准 §6 明示 prior_impulse_return>=10% 仅"预注册研究阈值, 不视为永久市场真理",
    # 故 PriorStrengthPass 不硬 gate 在 IMPULSE>=0.10, 改用底层连续量做宽松先验强度判定。
    # 阈值(B_med)为第一版近似, 待 v1→v1.1 Diff 与后续 Gate 数据证伪后再校准。
    LiquidityPass = (VR >= 0.5) & (VR <= 3.0) & (AMT >= 5e7)
    PriorStrengthPass = (ret_60d >= -0.02) & (dist_from_60h <= 0.45) & \
                        (up_vol_expansion >= 0.8)
    StructuralBreakdown = BREAK_DOWN | (recent_lower_low_count >= 5) | \
        ((support_break_depth_atr >= 1.0) & (support_break_volume_ratio >= 1.5))
    ClearDistribution = (distribution_score_raw >= 3.5) & (C < MA20)
    StructureIntact = (~StructuralBreakdown) & (support_break_depth_atr < 1.5) & \
                      (recent_lower_low_count <= 5)
    DIP_ELIGIBLE_v11 = PriorStrengthPass & StructureIntact & LiquidityPass & \
                       ~ClearDistribution & ~StructuralBreakdown

    # ---- dip_quality_score (值得等?) / dip_readiness_score (离低吸窗口多近?) ----
    q_prior = (IMPULSE.clip(0, 0.6) / 0.6 * 30 +
               ret_60d.clip(0, 0.6) / 0.6 * 20 +
               up_vol_expansion.clip(0, 3) / 3 * 15 +
               recent_breakout_flag.astype(float) * 10 +
               prior_turnover_strength.clip(0, 3) / 3 * 10)
    q_struct = (~ClearDistribution).astype(float) * 15 + (~StructuralBreakdown).astype(float) * 10
    dip_quality_score = (q_prior + q_struct).clip(0, 100)
    r_pb = ((PULLBACK_ATR - 0.3) / 2.7).clip(0, 1) * 30
    r_slope = (1 - (MA20S5.clip(-0.1, 0.1) + 0.1) / 0.2).clip(0, 1) * 20
    r_zone = (1 - zone_distance_atr.clip(0, 1)) * 25
    r_vol = (1 - (VR.clip(0, 3) / 3)) * 15
    r_exh = (NEW_LOW_CNT5 <= 1).astype(float) * 10
    dip_readiness_score = (r_pb + r_slope + r_zone + r_vol + r_exh).clip(0, 100)

    # ---- dip_stage (仅 DIP 候选赋予; 其它为 NaN) ----
    pullback_active = PriorStrengthPass & StructureIntact & (PULLBACK_ATR >= 0.3) & (C < MA20)
    near_zone = PriorStrengthPass & StructureIntact & (zone_distance_atr <= 0.20) & (PULLBACK_ATR >= 0.3)
    _stage = np.select([near_zone.fillna(False), pullback_active.fillna(False)],
                       ["NEAR_ZONE", "PULLBACK_ACTIVE"], default="PRE_PULLBACK")
    dip_stage = pd.DataFrame(_stage, index=C.index, columns=C.columns) \
        .where(DIP_ELIGIBLE_v11.fillna(False))

    def near(x, ref, tol_atr):
        return (x - ref).abs() <= tol_atr * ATR_REF

    # ================= BOTTOM_SETUP 评分（选股标准 §10，100 分）================
    # 10.1 前期走势/背景 15
    s_bg = (IMPULSE >= 0.10) * 6 + (UP_VMED > DN_VMED) * 4 + \
           ((PCT > 0.05).rolling(60).sum() < 3) * 3 + (AMT < 3 * AMT_MED20) * 2
    # 10.2 回调充分 20（回调过浅低分、适中高分、过深且结构破坏降分）
    s_pb = 20 * ((PULLBACK_ATR - 0.5) / 2.5).clip(0, 1)
    s_pb = s_pb.where(PULLBACK_ATR <= 8, s_pb * 0.5)
    # 10.3 支撑与流动性区域 25
    s_sup = near(C, MA20, SUPPORT_TOL) * 6 + \
            ((C - LOW20) <= SUPPORT_TOL * ATR_REF) * 6 + \
            near(C, BOLL_L, SUPPORT_TOL) * 5 + \
            ((C - LOW60) <= SUPPORT_TOL * ATR_REF) * 8
    # 10.4 调整衰竭 20
    s_exh = (DS5 > DS3) * 4 + (DN_VMED <= DN_VMED.shift(5)) * 4 + \
            (NEW_LOW_CNT5 <= 1) * 3 + E7.astype(int) * 3 + \
            (HIST > HIST.shift(1)) * 2 + (RSI >= RSI.shift(1)) * 2 + \
            ((E4 | E5).fillna(False).astype(int)) * 2
    # 10.5 量价修复 10
    s_rep = (LWICK >= 0.4) * 4 + ((C > MA20) | E4 | E5) * 3 + (C >= O) * 3
    # 10.6 板块环境 10（代理：个股不显著跑输池指数 + 池指数未崩）
    s_sec = (STOCK3 >= (MKT3X - 0.05).to_numpy()[:, None]) * 5 + \
            bcol(MKT5X > -0.03) * 5
    BOTTOM_SCORE = (s_bg + s_pb + s_sup + s_exh + s_rep + s_sec).clip(0, 100)

    # ============== INTRADAY_DIP_SETUP 评分（选股标准 §13–20，100 分）==========
    # §14 趋势结构 20（各项 capped at 20）
    s_tr = (MA20S5 > 0) * 6 + (C >= MA20) * 4 + \
           ((C >= MA20 - DIP_BAND * ATR_REF) & (C < MA20)) * 2 + \
           MACD_A.astype(int) * 4 + \
           (((DIF / C).abs() <= 0.01) & (HIST >= HIST.shift(1))) * 2 + \
           E7.astype(int) * 4 + (REC_LOW >= PRI_LOW) * 2 + \
           HIGH_RAISE.astype(int) * 2 + HIGH_FLAT.astype(int) * 1
    s_tr = s_tr.clip(upper=20)
    # §15 支撑与位置 20（可得支撑满分 13，rescale 到 20）
    s_sup2_raw = (near(C, MA20, 0.25)) * 3 + \
                 ((C - LOW20) <= 0.35 * ATR_REF) * 4 + \
                 near(C, BOLL_L, 0.25) * 2 + \
                 ((near(C, MA20, SUPPORT_TOL).astype(int) +
                   ((C - LOW20) <= SUPPORT_TOL * ATR_REF).astype(int) +
                   near(C, BOLL_L, SUPPORT_TOL).astype(int)) >= 2) * 4
    s_sup2 = s_sup2_raw * 20.0 / 13.0
    # §16 量价质量 15
    s_vq = (UP_VMED > DN_VMED) * 5 + (VOL_MA5 < VOL_MED20) * 4 + \
           (~BREAK_DOWN) * 3 + ~((AMT > 3 * AMT_MED20) & (C < O) & (PCT.abs() < 0.02)) * 3
    # §17 调整质量 15（RSI 分档 + MACD/回调/BOLL 位置）
    rsi_s = np.select([(RSI >= 45) & (RSI <= 65), (RSI >= 35) & (RSI < 45), RSI > 65],
                      [6, 3, 1], default=0)
    s_adj = pd.DataFrame(rsi_s, index=C.index, columns=C.columns) + \
        (HIST >= HIST.shift(1)) * 3 + \
        ((PULLBACK_ATR >= 0.3) & (PULLBACK_ATR <= 3.0)) * 3 + \
        (((C - BOLL_L) / (BOLL_U - BOLL_L).replace(0, np.nan)) >= 0.2) * 3
    # §18 板块/龙头 15（代理口径）
    s_sec2 = bcol(MKT1X > 0) * 5 + bcol(MKT3X > 0) * 4 + \
             (STOCK3 >= (MKT3X - 0.03).to_numpy()[:, None]) * 4 + \
             (PCT >= (MKT1X - 0.03).to_numpy()[:, None]) * 2
    # §19 流动性/换手 10
    # to_pct 已上移至 v1.1 特征块 (供 DIP Gate 复用)
    s_liq = ((to_pct >= 0.2) & (to_pct <= 0.9)) * 5 + \
            ((VR >= 0.5) & (VR <= 3.0)) * 3 + (AMT >= 5e7) * 2
    # §20 LATR 潜力 5
    n_sup = (near(C, MA20, 0.35).astype(int) +
             ((C - LOW20) <= 0.35 * ATR_REF).astype(int) +
             near(C, BOLL_L, 0.35).astype(int) +
             ((C - LOW60) <= 0.35 * ATR_REF).astype(int))
    s_latr = ((C - pd.concat([MA20, LOW20, BOLL_L, LOW60]).groupby(level=0).min()
               ) <= 0.5 * ATR_REF) * 3 + (n_sup >= 2) * 2
    DIP_SCORE = (s_tr + s_sup2 + s_vq + s_adj + s_sec2 + s_liq + s_latr).clip(0, 100)

    print("[5/5] 组装输出 …")
    ELIG = ELIG_BASE & MA20S5.notna() & ATR_REF.notna() & DIF.notna() & RSI.notna() & \
        (ATR_REF > 0) & (C > 0)
    # v1.1: DIP 资格 = PriorStrength/Structure 分解 (DIP_ELIGIBLE_v11);
    # DIP_SCORE 仅作分层用, 不再作为准入门槛 (选股标准 §5.2)
    KEEP = ELIG & ((BOTTOM_SCORE >= 40) | BOTTOM_GATE.fillna(False) |
                   DIP_ELIGIBLE_v11.fillna(False))
    KEEP = KEEP & (KEEP.index >= pool_start)[:, None]

    def melt(m, name):
        s = m.where(KEEP).stack().dropna()
        s.index.names = ["date", "code"]
        return s.rename(name)

    frames = [
        melt(BOTTOM_SCORE, "bottom_score"),
        # v1.1: DIP 分数只在通过分解资格 DIP_ELIGIBLE_v11 时计入, 否则归零 ->
        # primary_setup==DIP 严格等价于通过 DIP 分解 gate (避免 BOTTOM 入选股被按分数误标为 DIP)
        melt(DIP_SCORE.where(DIP_ELIGIBLE_v11.fillna(False), 0.0), "intraday_dip_score"),
        melt(BOTTOM_GATE.fillna(False), "bottom_gate").astype(bool),
        melt(DIP_ELIGIBLE_v11.fillna(False), "dip_eligible_v11").astype(bool),
        melt(EVID, "bottom_evidence_count"),
        melt(E1, "ev_shrink_newlow").astype("int8"), melt(E2, "ev_macd_recover").astype("int8"),
        melt(E3, "ev_rsi_reclaim30").astype("int8"), melt(E4, "ev_reclaim_boll").astype("int8"),
        melt(E5, "ev_reclaim_pivotlow").astype("int8"), melt(E6, "ev_long_lowwick").astype("int8"),
        melt(E7, "ev_higher_low").astype("int8"), melt(E8, "ev_volume_decay").astype("int8"),
        melt(E9, "ev_low_absorb").astype("int8"),
        melt(ATR_REF, "atr_ref"), melt(ATR_REF / C, "atr_pct"),
        melt(MA20S5, "ma20_slope5"), melt(C / MA20 - 1, "dist_ma20"),
        melt(DIST_LOW60, "dist_low60"), melt(DIST_LOW20, "dist_low20"),
        melt(RPOS20, "range_pos20"), melt(RPOS60, "range_pos60"),
        melt(DIF, "macd_dif"), melt(DEA, "macd_dea"), melt(HIST, "macd_hist"),
        melt(RSI, "rsi14"), melt(C - BOLL_L, "dist_boll_lower"),
        melt(IMPULSE, "impulse_return"), melt(PULLBACK_ATR, "pullback_atr"),
        melt(PULLBACK_PCT, "pullback_pct"), melt(PB_DAYS, "pullback_days"),
        melt(LWICK, "lower_wick_ratio"), melt(CPOS, "close_position"),
        melt(DS3, "drop_speed_3d"), melt(DS5, "drop_speed_5d"),
        melt(NEW_LOW.astype("int8"), "new_low_flag"),
        melt(UP_VMED, "up_volume_median"), melt(DN_VMED, "down_volume_median"),
        melt(VOL / VOL_MED20, "volume_ratio"),
        melt(TO, "turnover_rate"), melt(AMT, "amount"),
        melt(TMV, "market_cap"), melt(FMV, "float_market_cap"),
        melt(PCT * 100, "stock_return_1d"), melt(STOCK3 * 100, "stock_return_3d"),
        melt(bcol(MKT1X * 100), "sector_proxy_ret1"),
        melt(bcol(MKT3X * 100), "sector_proxy_ret3"),
        melt(n_sup, "zone_strength"),
        melt(VOL, "volume"),
        melt(PC, "pre_close"), melt(C, "close"),
        # ---- v1.1 DIP 分解底层连续特征 + 聚合 flag + stage/双评分 ----
        melt(ma5_slope, "ma5_slope"), melt(ma10_slope, "ma10_slope"),
        melt(ma20_slope_3, "ma20_slope_3"), melt(MA20S5, "ma20_slope_5"),
        melt(ma20_slope_10, "ma20_slope_10"),
        melt(ret_3d, "ret_3d"), melt(ret_5d, "ret_5d"), melt(ret_10d, "ret_10d"),
        melt(ret_20d, "ret_20d"), melt(ret_60d, "ret_60d"),
        melt(dist_from_20h, "dist_from_20d_high"), melt(dist_from_60h, "dist_from_60d_high"),
        melt(prior_impulse_return, "prior_impulse_return"), melt(prior_impulse_days, "prior_impulse_days"),
        melt(breakout_strength_atr, "breakout_strength_atr"),
        melt(recent_breakout_flag, "recent_breakout_flag").astype(bool),
        melt(up_vol_expansion, "up_volume_expansion_ratio"),
        melt(dn_vol_contraction, "down_volume_contraction_ratio"),
        melt(prior_range_expansion, "prior_range_expansion"),
        melt(prior_turnover_strength, "prior_turnover_strength"),
        melt(dist_confirmed_swing_low_atr, "dist_confirmed_swing_low_atr"),
        melt(dist_breakout_platform_atr, "dist_breakout_platform_atr"),
        melt(recent_lower_low_count, "recent_lower_low_count"),
        melt(recent_lower_high_count, "recent_lower_high_count"),
        melt(support_break_depth_atr, "support_break_depth_atr"),
        melt(support_break_volume_ratio, "support_break_volume_ratio"),
        melt(distribution_score_raw, "distribution_score_raw"),
        melt(zone_distance_atr, "zone_distance_atr"),
        melt(PriorStrengthPass, "prior_strength_pass").astype(bool),
        melt(StructureIntact, "structure_intact_flag").astype(bool),
        melt(StructuralBreakdown, "structural_breakdown_flag").astype(bool),
        melt(ClearDistribution, "clear_distribution_flag").astype(bool),
        melt(LiquidityPass, "liquidity_pass").astype(bool),
        melt(dip_quality_score, "dip_quality_score"),
        melt(dip_readiness_score, "dip_readiness_score"),
        melt(dip_stage, "dip_stage"),
    ]
    pool = pd.concat(frames, axis=1).reset_index()
    pool["date"] = pool["date"].astype(int)
    pool["code"] = pool["code"].astype(str).str.zfill(6)

    # primary_setup / 分层 / tier
    pool["primary_setup"] = np.where(pool["bottom_score"] >= pool["intraday_dip_score"],
                                     "BOTTOM_SETUP", "INTRADAY_DIP_SETUP")
    pool["primary_score"] = pool[["bottom_score", "intraday_dip_score"]].max(axis=1)

    def tier(s):
        return pd.cut(s, [-np.inf, NEAR_MISS_LO, 55, 65, 75, np.inf],
                      labels=["DROP", "NEAR_MISS", "RESEARCH", "WATCH", "HIGH_PRIORITY_WATCH"])
    pool["bottom_tier"] = tier(pool["bottom_score"]).astype(str)
    pool["dip_tier"] = tier(pool["intraday_dip_score"]).astype(str)
    pool["primary_tier"] = tier(pool["primary_score"]).astype(str)
    return pool


# ---------------------------------------------------------------- ③.1 候选审计字段
def _build_calendar():
    if _HAVE_CANON_CAL:
        return _tc_load_calendar()
    import pyarrow.dataset as _ds
    import glob as _gl
    return sorted(set(int(x) for x in _ds.dataset(sorted(_gl.glob(BARS_GLOB)),
                     format="parquet").to_table(columns=["date"]).column("date").to_pylist()))


def _effective_trade_date(d):
    if _HAVE_CANON_CAL:
        nxt = _tc_next(d)
        return int(nxt) if nxt else nxt
    return int(_next_trading_date(_build_calendar(), int(d)))


def _snapshot_id():
    if _HAVE_CANON_CAL:
        return _tc_snapshot()
    return "local-unknown"


def _inject_candidate_fields(pool):
    """注入 ③.1 审计字段: plan_date / effective_trade_date / 契约版本 / asof / 源快照指纹。"""
    pool["plan_date"] = pool["date"]
    pool["effective_trade_date"] = pool["plan_date"].map(_effective_trade_date)
    pool["candidate_contract_version"] = "GREEN-WATCH-v1.1-DIP-GATE-DECOMPOSITION"
    # 收紧为真实时间语义: 候选在 P 日收盘 (15:00) 即可得, 故 asof = P_CLOSE 时间戳
    pool["candidate_generated_asof"] = (pool["plan_date"].astype(str) + "1500").astype(int)
    pool["source_data_snapshot_id"] = _snapshot_id()
    return pool


# ---------------------------------------------------------------- 历史回填
def main(force=False):
    if OUT.exists() and not force:
        print(f"[skip] {OUT} 已存在（--force 可重建）")
        return
    t0 = time.time()
    print("[1/5] 加载日线 …")
    df = load_bars()
    df = df[df["date"] >= 20180101].reset_index(drop=True)
    print(f"  {len(df)} 行，{df['code'].nunique()} 票，{df['date'].min()}..{df['date'].max()}")

    pool = compute_pool(df, POOL_START)
    pool["rule_version"] = "GREEN-WATCH-v0-backfill"
    pool = _inject_candidate_fields(pool)
    pool = pool.sort_values(["date", "primary_score"], ascending=[True, False])
    pool.to_parquet(OUT, index=False)
    print(f"  池行数 {len(pool)}，日期 {pool['date'].min()}..{pool['date'].max()}，"
          f"票 {pool['code'].nunique()} 只")
    print(f"  用时 {time.time()-t0:.0f}s → {OUT}")
    print("\n分层分布（primary）：")
    print(pool["primary_tier"].value_counts().to_string())


# ---------------------------------------------------------------- 每日增量
def _selection_reasons(r, setup):
    tier_col = "bottom_tier" if setup == "BOTTOM_SETUP" else "dip_tier"
    parts = [f"tier={r[tier_col]}"]
    parts.append(f"止跌证据 {int(r['bottom_evidence_count'])}/9")
    parts.append(f"距60日低点 {r['dist_low60']:.1%}")
    parts.append(f"回撤 {r['pullback_atr']:.1f}ATR")
    parts.append(f"MA20斜率5日 {r['ma20_slope5']:+.2%}")
    if setup == "INTRADAY_DIP_SETUP":
        parts.append(f"距MA20 {r['dist_ma20']:+.1%}")
        parts.append(f"区间位置20日 {r['range_pos20']:.2f}")
        if "dip_stage" in r and pd.notna(r.get("dip_stage")):
            parts.append(f"stage={r['dip_stage']}")
    return "，".join(parts)


def _risk_reasons(r):
    risks = []
    if r["ma20_slope5"] < 0:
        risks.append("MA20 五日斜率为负，中期趋势尚未扭转")
    if r["atr_pct"] > 0.05:
        risks.append(f"波动率偏高（ATR≈{r['atr_pct']:.1%}）")
    if r["volume_ratio"] < 0.6:
        risks.append("量能萎缩，反弹确认不足")
    if r["dist_ma20"] < -0.05:
        risks.append(f"收盘低于 MA20 {abs(r['dist_ma20']):.0%}，支撑有效性格外依赖次日承接")
    if not risks:
        risks.append("支撑失守需严格止损，注意大盘环境变化")
    return "；".join(risks[:3])


def _next_trading_date(cal, pool_date):
    later = [d for d in cal if d > pool_date]
    if later:
        return str(later[0])
    d = pd.Timestamp(str(pool_date)) + pd.Timedelta(days=1)
    while d.weekday() >= 5:                                   # 5=Sat, 6=Sun
        d += pd.Timedelta(days=1)
    return d.strftime("%Y%m%d")


def write_latest_json(pool_today, names, pool_date, cal):
    candidates = []

    def add_group(sub, setup, score_col, tier_col):
        g = sub[sub[tier_col].isin(["WATCH", "HIGH_PRIORITY_WATCH"])] \
              .sort_values(score_col, ascending=False).head(10)
        for rank, (_, r) in enumerate(g.iterrows(), 1):
            candidates.append({
                "rank": rank,
                "code": r["code"],
                "name": names.get(r["code"], ""),
                "setup": setup,
                "score": round(float(r[score_col]), 1),
                "tier": r[tier_col],
                "bottom_score": round(float(r["bottom_score"]), 1),
                "intraday_dip_score": round(float(r["intraday_dip_score"]), 1),
                "atr_ref": round(float(r["atr_ref"]), 3),
                "atr_pct": round(float(r["atr_pct"]), 3),
                "pre_close": round(float(r["pre_close"]), 2),
                "close": round(float(r["close"]), 2),
                "trigger_price": round(float(r["pre_close"]) - 0.25 * float(r["atr_ref"]), 2),
                "selection_reasons": _selection_reasons(r, setup),
                "risk_reasons": _risk_reasons(r),
            })

    add_group(pool_today, "BOTTOM_SETUP", "bottom_score", "bottom_tier")
    add_group(pool_today, "INTRADAY_DIP_SETUP", "intraday_dip_score", "dip_tier")

    payload = {
        "pool_date": str(pool_date),
        "next_trading_date": _next_trading_date(cal, pool_date),
        "rule_version": "GREEN-WATCH-v0-daily",
        "candidates": candidates,
    }
    LATEST_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print(f"  → {LATEST_JSON}（{len(candidates)} 个候选）")


def run_daily(target):
    t0 = time.time()
    target = int(target)
    start = int((pd.Timestamp(str(target)) - pd.Timedelta(days=DAILY_WINDOW_DAYS))
                .strftime("%Y%m%d"))
    print(f"[daily] 目标日 {target}，加载窗口 {start}..{target}（{DAILY_WINDOW_DAYS} 自然日）…")
    df = load_bars(start=start, with_name=True)
    df = df[df["date"] <= target].reset_index(drop=True)
    if not (df["date"] == target).any():
        raise SystemExit(f"[abort] 日线数据中不存在交易日 {target}")
    names = (df[df["date"] == target]
             .drop_duplicates("code").set_index("code")["name"].to_dict())
    print(f"  {len(df)} 行，{df['code'].nunique()} 票，{df['date'].min()}..{df['date'].max()}")
    print("  加载全历史收盘价矩阵（EWM/BOLL 长记忆链逐位一致所需）…")
    c_hist = load_close_hist()

    pool = compute_pool(df, pool_start=target, c_hist=c_hist)
    pool["rule_version"] = "GREEN-WATCH-v0-daily"
    pool = _inject_candidate_fields(pool)
    pool = pool.sort_values(["date", "primary_score"], ascending=[True, False])
    print(f"  目标日池行数 {len(pool)}")

    if OUT.exists():
        old = pd.read_parquet(OUT)
        n_drop = int((old["date"] == target).sum())
        old = old[old["date"] != target]
        pool = pd.concat([old, pool], ignore_index=True, sort=False)
        if n_drop:
            print(f"  剔除旧文件中 {target} 既有行 {n_drop} 条（幂等重写）")
    pool.to_parquet(OUT, index=False)
    print(f"  池总行数 {len(pool)}，重写 → {OUT}")

    import pyarrow.dataset as ds
    import glob as _glob
    cal = sorted(ds.dataset(sorted(_glob.glob(BARS_GLOB)), format="parquet")
                 .to_table(columns=["date"]).column("date").to_pylist())
    write_latest_json(pool[pool["date"] == target], names, target, cal)
    print(f"  用时 {time.time()-t0:.0f}s")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GREEN-WATCH-v0 池构建（回填 / 每日增量）")
    p.add_argument("--date", default=None,
                   help="增量模式：目标交易日 YYYYMMDD；不带则走历史回填")
    p.add_argument("--force", action="store_true", help="回填模式：已存在则重建")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.date:
        run_daily(args.date)
    else:
        main(force=args.force)
