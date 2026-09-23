"""a0a4_config — A0-A4 Harness 预注册配置 (唯一事实源, 全部脚本 import 此处).

协议 (用户 2026-09-19 授权开工, 固定不变):
  A0  = V0 (pool_daily.parquet, 旧门控) 负基线重放, 当前冻结 Label/Execution;
        不训练模型, 只出 universe 经济学 + 旧系统自身排序 (intraday_dip_score) 的
        top-bucket 经济学。A0 -> A1 = Candidate System Increment (不是特征消融)。
  A1  = GREEN-WATCH-v1.1 Candidate (pool_daily_v11.parquet) + 盘后 Core 特征
  A2  = A1 + ATR_ref / Zone / Structural Liquidity
  A3  = A2 + Dip depth / speed (D_INTRADAY)
  A4  = A3 + Recovery / Reclaim / VWAP / Wick / Volume / SecondTest / MinuteState
  A1-A4: 同一 Candidate Universe / 同一 Sample IDs / 同一 Label / 同一 Split /
        同一 Execution / 同一 Adjustment / 同一 Cost / 同一 LightGBM params / 同一 seeds;
        唯一变化 = Feature Group。

纪律:
  - SAMPLE_IDENTITY_PASS: set(A1)==set(A2)==set(A3)==set(A4) 不等则停
  - FEATURE_DELTA_PASS:   A2-A1 只能是 Zone 组; A3-A2 只能是 Dip/Speed 组;
                          A4-A3 只能是 Recovery/Reclaim 组
  - 3 seeds {42, 2026, 917}, LightGBM 参数固定不调
  - FLOAT_MV pre/post 20260701 为强制报告切片 (非附录)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_V5 = Path(__file__).resolve().parent.parent          # v5_hybrid/
PREFLIGHT = _V5 / "step9_preflight"
GW = _V5 / "green_watch_v0"
RUN = _V5 / "a0a4"                                     # 运行产物根目录
DATASETS = RUN / "datasets"
TIER_DIRS = {t: RUN / t for t in ("a0", "a1", "a2", "a3", "a4")}

sys.path.insert(0, str(PREFLIGHT))

# ---------------- 冻结输入 ----------------
POOL_V1 = GW / "pool_daily.parquet"        # A0 宇宙 (V0 旧门控)
POOL_V11 = GW / "pool_daily_v11.parquet"   # A1-A4 宇宙 (v1.1 门控)
FROZEN_FILES = {
    "label_contract": PREFLIGHT / "label_contract_v1.yaml",
    "feature_registry": PREFLIGHT / "feature_registry.yaml",
    "execution_engine": PREFLIGHT / "execution_engine.py",
    "latr_builder": PREFLIGHT / "latr_builder.py",
    "zone_state_engine": PREFLIGHT / "zone_state_engine.py",
    "market_proxy_v1": PREFLIGHT / "market_proxy_v1.py",
    "adjustment_chain": PREFLIGHT / "adjustment_chain.py",
    "green_watch_build_pool": GW / "build_pool.py",
    "green_watch_candidate_loader": PREFLIGHT / "green_watch_candidate_loader.py",
    "gate3_manifest": PREFLIGHT / "GATE3_MANIFEST.yaml",
}

# ---------------- 抽样 (预注册, 与 coverage_02 同 procedure) ----------------
SAMPLE_SPEC = {
    "min_plan": 20250102,
    "max_plan": 20260917,
    "minute_start": 20250102,          # effective_trade_date 下限 (分钟数据起点)
    "cap_per_day": 40,
    "seed": 20260918,
    "procedure": ("per plan_date: df.sample(n=min(cap,len), "
                  "random_state=int(rng.integers(0,1_000_000))), "
                  "rng=np.random.default_rng(seed) 按 plan_date 升序消费"),
}

# ---------------- Walk-forward 切分 (按 plan_date; purge=8 交易日 > H5=5) ----------------
FOLDS = [
    {"fold": 1, "test_lo": 20250901, "test_hi": 20251130},
    {"fold": 2, "test_lo": 20251201, "test_hi": 20260228},
    {"fold": 3, "test_lo": 20260301, "test_hi": 20260531},
    {"fold": 4, "test_lo": 20260601, "test_hi": 20260910},
]
PURGE_TRADING_DAYS = 8
SPLIT_HASH_NOTE = "train: plan_date <= prev_trading_day(test_lo, purge); test: test_lo<=plan_date<=test_hi"

# ---------------- 标签轨道 ----------------
# 主轨 = Timing-v1 primary (timing_return_h1, 锚 actual_fill; 与 X0 账本/Q4 对齐)
# 副轨 = S0-v1 primary (s0_return_h3, 锚 D 日 open)  — 同协议独立评估
LABEL_TRACKS = {
    "timing": {"label": "timing_return_h1", "eligibility": "timing_eligible",
               "alpha": "timing_market_alpha_h1"},
    "s0": {"label": "s0_return_h3", "eligibility": "s0_eligible",
           "alpha": "s0_market_alpha_h3"},
}

# ---------------- LightGBM 固定参数 (第一轮不调参) ----------------
LGBM_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "n_estimators": 400,
    "verbosity": -1,
}
SEEDS = [42, 2026, 917]

# ---------------- 特征分组 (FEATURE_DELTA_PASS 的预注册期望) ----------------
# origin: POOL_INHERITED(v11 P_CLOSE) / LATR_FROZEN / ZONE_FROZEN / HARNESS_DERIVED
A1_CORE = [
    ("setup_type", "POOL_INHERITED"),          # = primary_setup (categorical)
    ("dip_stage", "POOL_INHERITED"),           # categorical, NaN->"NA"
    ("prior_strength_pass", "POOL_INHERITED"),
    ("structure_intact_flag", "POOL_INHERITED"),
    ("prior_impulse_return", "POOL_INHERITED"),
    ("ma20_slope_5", "POOL_INHERITED"),
    ("ret_60d", "POOL_INHERITED"),
    ("ret_20d", "POOL_INHERITED"),
    ("dist_from_60d_high", "POOL_INHERITED"),
    ("range_pos60", "POOL_INHERITED"),
    ("range_pos20", "POOL_INHERITED"),
    ("rsi14", "POOL_INHERITED"),
    ("dist_low60", "POOL_INHERITED"),
    ("distribution_score_raw", "POOL_INHERITED"),
    ("support_break_depth_atr", "POOL_INHERITED"),
]
A2_ZONE = [
    ("atr_ref", "LATR_FROZEN"),
    ("zone_strength", "LATR_FROZEN"),
    ("zone_distance_atr", "LATR_FROZEN"),
    ("penetration_atr", "LATR_FROZEN"),
    ("primary_zone_distance_atr", "ZONE_FROZEN"),
    ("zone_band_atr", "HARNESS_DERIVED"),      # (zone_upper-zone_lower)/atr_ref
    ("n_zones", "HARNESS_DERIVED"),            # len(json(zones_json))
    ("zone_n_sources", "HARNESS_DERIVED"),     # primary_zone_source 的 | 分隔计数
]
A3_DIP = [
    ("open_dip_atr", "LATR_FROZEN"),
    ("gap_atr", "LATR_FROZEN"),
    ("total_dip_atr", "LATR_FROZEN"),
    ("dip_depth_5m_atr", "HARNESS_DERIVED"),
    ("dip_speed_5m", "HARNESS_DERIVED"),       # dip_depth_5m_atr / 5
    ("dip_speed_15m", "HARNESS_DERIVED"),      # open_dip_atr / 15
    ("dip_depth_30m_atr", "HARNESS_DERIVED"),
    ("dip_speed_30m", "HARNESS_DERIVED"),      # dip_depth_30m_atr / 30
    ("new_low_flag", "LATR_FROZEN"),
]
A4_RECOVERY = [
    ("recovery_atr", "LATR_FROZEN"),
    ("recovery_from_zone_low_atr", "ZONE_FROZEN"),
    ("dip_recovery_ratio", "ZONE_FROZEN"),
    ("close_position_15m", "LATR_FROZEN"),
    ("reclaim_zone_lower", "ZONE_FROZEN"),
    ("reclaim_zone_mid", "ZONE_FROZEN"),
    ("reclaim_zone_upper", "ZONE_FROZEN"),
    ("reclaim_failure_count", "ZONE_FROZEN"),
    ("second_test_flag", "ZONE_FROZEN"),
    ("second_test_higher_low", "ZONE_FROZEN"),
    ("time_since_zone_touch", "ZONE_FROZEN"),
    ("time_below_zone", "ZONE_FROZEN"),
    ("lower_wick_ratio", "ZONE_FROZEN"),
    ("sell_volume_ratio", "ZONE_FROZEN"),
    ("rebound_volume_ratio", "ZONE_FROZEN"),
    ("vwap_reclaim", "ZONE_FROZEN"),
    ("zone_break_depth_atr", "ZONE_FROZEN"),
    ("zone_break_duration", "ZONE_FROZEN"),
    ("cum_volume_vs_hist_same_time", "ZONE_FROZEN"),
    ("cum_turnover_vs_hist_same_time", "ZONE_FROZEN"),
    ("final_state", "ZONE_FROZEN"),            # categorical (6 态)
]
TIER_FEATURES = {
    "a1": A1_CORE,
    "a2": A1_CORE + A2_ZONE,
    "a3": A1_CORE + A2_ZONE + A3_DIP,
    "a4": A1_CORE + A2_ZONE + A3_DIP + A4_RECOVERY,
}
EXPECTED_DELTA = {
    ("a2", "a1"): [f for f, _ in A2_ZONE],
    ("a3", "a2"): [f for f, _ in A3_DIP],
    ("a4", "a3"): [f for f, _ in A4_RECOVERY],
}
CATEGORICAL_FEATURES = {"setup_type", "dip_stage", "final_state"}

# HARNESS_DERIVED 公式 (预注册, 供 feature_manifest 引用)
DERIVED_FORMULAS = {
    "zone_band_atr": "(zone_upper-zone_lower)/atr_ref",
    "n_zones": "len(json.loads(zones_json))",
    "zone_n_sources": "primary_zone_source 按 '|' 分隔后的来源数 (None->0)",
    "dip_depth_5m_atr": "(D日首分钟open - min(low[开盘后5分钟]))/atr_ref",
    "dip_speed_5m": "dip_depth_5m_atr/5",
    "dip_speed_15m": "open_dip_atr/15",
    "dip_depth_30m_atr": "(D日首分钟open - min(low[开盘后30分钟]))/atr_ref",
    "dip_speed_30m": "dip_depth_30m_atr/30",
    "mae_30m": "min(low[fill_ts..fill+30m])/actual_fill - 1 (raw)",
    "mfe_30m": "max(high[fill_ts..fill+30m])/actual_fill - 1 (raw)",
    "mae_day": "min(low[fill_ts..D日收盘])/actual_fill - 1 (raw)",
    "mfe_day": "max(high[fill_ts..D日收盘])/actual_fill - 1 (raw)",
    "mae_h1": "min(D日post-fill low, H1日 daily low)/actual_fill - 1 (raw)",
    "mfe_h1": "max(D日post-fill high, H1日 daily high)/actual_fill - 1 (raw)",
    "failed_rebound": "int(mfe_30m < 0.01) — 买入后30分钟内未出现 >=1% 反弹",
}

# ---------------- 报告切片 (强制) ----------------
SLICES = {
    "ALL": None,
    "BOTTOM_ONLY": (lambda d: d["is_bottom"].astype(bool) & ~d["is_dip"].astype(bool)),
    "DIP_ONLY": (lambda d: d["is_dip"].astype(bool) & ~d["is_bottom"].astype(bool)),
    "DUAL_HIT": (lambda d: d["is_bottom"].astype(bool) & d["is_dip"].astype(bool)),
    "DUAL_HIT_SCORE40": (lambda d: d["dual_hit_score40"].astype(bool)),
    "DIP_PRE_PULLBACK": (lambda d: d["dip_stage"] == "PRE_PULLBACK"),
    "DIP_PULLBACK_ACTIVE": (lambda d: d["dip_stage"] == "PULLBACK_ACTIVE"),
    "DIP_NEAR_ZONE": (lambda d: d["dip_stage"] == "NEAR_ZONE"),
    "FLOAT_MV_PRE_20260701": (lambda d: d["plan_date"] < 20260701),
    "FLOAT_MV_POST_20260701": (lambda d: d["plan_date"] >= 20260701),
    "V1_REJECTED_V11_ACCEPTED": (lambda d: d["v1_rejected_v11_accepted"] == 1),
}

# ---------------- X0-style 账本 (预注册) ----------------
X0_STYLE = {
    "slots": 20,
    "stop_dist": "clip(1.5*atr_ref/fill, 3%, 6%)  [冻结 build_exit_labels 参数]",
    "stop_mult": 1.5, "stop_min": 0.03, "stop_max": 0.06,
    "tmax_days": 4,                    # entry 日 + 4 交易日 (冻结 X0_CONTROL tmax=4)
    "entry": "actual_fill (raw, 已含买侧 slippage 2.0bps + fee 2.1bps)",
    "exit_cost": 0.0020,               # 卖侧 slippage 2bps + 费用 18bps
    "sellable_from": "entry 次一交易日起 (T+1)",
    "limit_down_blocked": "exit 日跌停锁死 -> 顺延最多 3 交易日的开盘价出场",
    "approximation": "日线近似 (DAILY_APPROX): 止损用日线 open/low 悲观序, 时间退出用 exit 日 close",
    "round_trip_note": "entry 已含买侧成本; 卖侧 0.0020; 与冻结 0.0013/0.0018 口径差异已注明",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_hashes() -> dict:
    return {k: sha256_file(v) for k, v in FROZEN_FILES.items()}


def config_fingerprint() -> str:
    """本配置文件的 sha256 — 保证 A1-A4 共用同一协议版本。"""
    return sha256_file(Path(__file__).resolve())


def lgbm_params_with_seed(seed: int) -> dict:
    p = dict(LGBM_PARAMS)
    p["seed"] = int(seed)
    p["feature_fraction_seed"] = int(seed)
    p["bagging_seed"] = int(seed)
    p["data_random_seed"] = int(seed)
    return p


# ---------------- Track-aware 特征角色 (红队审计修复, 2026-09-19) ----------------
# S0 轨仅允许 P_CLOSE 可用特征 (A1_CORE + A2_ZONE); A3/A4 的 D 日盘中特征不属于 S0。
# Timing 轨允许全部 tier 特征, 但特征 as-of 必须 <= signal_ts (事件驱动快照)。
S0_FEATURES_BY_TIER = {
    "a1": [f for f, _ in A1_CORE],
    "a2": [f for f, _ in A1_CORE] + [f for f, _ in A2_ZONE],
    # a3 / a4: S0 不适用 -> N/A (track_features 返回 None, 不训练)
}


def track_features(tier: str, track: str):
    """返回 (tier, track) 实际应喂给模型的特征列。

    S0 轨: 仅 a1/a2 定义; a3/a4 返回 None (N/A, 不训练)。
    Timing 轨: 返回该 tier 全部特征 (TIER_FEATURES[tier])。
    """
    if track == "s0":
        return S0_FEATURES_BY_TIER.get(tier)
    return [f for f, _ in TIER_FEATURES[tier]]


# 两层 lineage registry: 每个特征的 role_allowed / availability_rule / source / origin
LINEAGE_REGISTRY = {}
for _f, _o in A1_CORE + A2_ZONE:
    LINEAGE_REGISTRY[_f] = {"group": "CORE_ZONE", "role_allowed": ["S0", "TIMING"],
                            "availability_rule": "P_CLOSE", "source": _o, "origin": _o}
for _f, _o in A3_DIP:
    LINEAGE_REGISTRY[_f] = {"group": "DIP", "role_allowed": ["TIMING"],
                            "availability_rule": "SNAPSHOT", "source": _o, "origin": _o}
for _f, _o in A4_RECOVERY:
    LINEAGE_REGISTRY[_f] = {"group": "RECOVERY", "role_allowed": ["TIMING"],
                            "availability_rule": "EVENT_TIME", "source": _o, "origin": _o}


def role_allowed(track: str):
    return [f for f, reg in LINEAGE_REGISTRY.items() if track.upper() in reg["role_allowed"]]
