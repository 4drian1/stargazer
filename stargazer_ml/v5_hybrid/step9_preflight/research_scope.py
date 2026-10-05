# -*- coding: utf-8 -*-
"""RESEARCH_DATA_SCOPE_V1 — 研究域 / 推理域 / 基准域 三域显式合同 (2026-09-24 冻结)

为什么要拆两个合同
------------------
2026-09-24 stockdb 服务不可用, 本地分钟镜像 (3959 票) 成为唯一可用分钟源。
镜像的域缺口是 **BSE + ETF/基金 + 少量未采集票**, 而不是"小盘"(实测: 被排除的
候选票 float_market_cap 中位数 40.9 亿, 反而略高于域内的 37.4 亿)。

因此 **绝不能**把"当前能研究什么"写成"策略永久允许什么":

    STRATEGY_ELIGIBLE_SCOPE          策略理论上允许研究的资产范围 (宽, 不随数据变动)
    RESEARCH_DATA_SCOPE_V1           当前能用一致分钟+一致基准做因果实验的范围 (窄, 随数据变动)
    INFERENCE_SCOPE                  当前模型推理域 (必须 == research scope)
    BENCHMARK_SCOPE                  基准域 (必须 == research scope, 否则 alpha 无意义)

数据引擎修好后, BSE/ETF 是否进入 Stargazer 由 STRATEGY_ELIGIBLE_SCOPE 决定,
而不是被一句"TRAIN_SCOPE 已冻结"误杀。本文件不主张 GREEN 永远只做四板。

三域一致性铁律
--------------
    candidate / feature / market proxy / matched control / label
    必须全部落在同一个 research_data_scope 内。

最危险的组合是:
    Candidate  = 含 BSE/ETF
    Benchmark  = 不含 BSE/ETF
    然后 BSE/ETF 因为没有 benchmark 被 dropna() 静默消失。
本模块用 annotate_scope() 给母表加**显式 validity 列**, 禁止 drop:
    research_scope_valid / research_scope_reason
    benchmark_valid      / benchmark_invalid_reason
    minute_data_valid    / minute_data_invalid_reason
(与既有纪律 candidate_status != sample_data_validity 同源。)

实测域覆盖 (2026-09-24, pool_daily_v11 epoch >= 20250102)
---------------------------------------------------------
    核心四板 (SH_MAIN/SZ_MAIN/GEM/STAR) 真实 (code,date) 覆盖 = 97.58%
    日覆盖 min 0.0 / p05 0.9852 / median 0.9931   (min=0 仅因镜像截止 20260908 后 7 日)
    SH 99.76% / SZ 99.72% / GEM 98.84% / STAR 97.41%
    BSE 0% / ETF-FUND 0%
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

HERE = Path(__file__).resolve().parent
V5 = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import instrument_master as IM  # noqa: E402

SCOPE_CONTRACT_VERSION = "RESEARCH_DATA_SCOPE_V1"

# ------------------------------------------------------------------ #
# ① 策略域 (宽, 不随数据可用性变动)
# ------------------------------------------------------------------ #
STRATEGY_ELIGIBLE_SCOPE = "A_SHARE_STOCK"
STRATEGY_ELIGIBLE_INSTRUMENT_TYPES = (IM.T_STOCK,)

# ------------------------------------------------------------------ #
# ② 研究域 / 推理域 (窄, 当前数据能力)
# ------------------------------------------------------------------ #
RESEARCH_DATA_SCOPE_ID = "CORE_A_SHARE_RESEARCH_SCOPE_V1"
RESEARCH_DATA_SCOPE_BOARDS = (IM.B_SH, IM.B_SZ, IM.B_GEM, IM.B_STAR)
INFERENCE_SCOPE_ID = RESEARCH_DATA_SCOPE_ID      # 当前推理域 == 研究域 (强制)
DATA_SCOPE_ID_MINUTE = "CORE_A_SHARE_MINUTE_V1"

# ------------------------------------------------------------------ #
# ③ 基准域 (必须与研究域一致)
#    C2 产出的 market_return_same_time 在当前 scope 下**不是** FULL_A_SHARE_MARKET,
#    故基准名与 alpha 名必须显式:
# ------------------------------------------------------------------ #
# 2026-09-24 更名: 原 CORE_A_SHARE_MARKET_PROXY_V1 暗示"完整 CORE A 股 market proxy",
# 但实测 float_mv coverage 仅 ~39.5% 且明确偏中小盘/GEM/STAR。名字必须显式 MIRROR 来源。
# 即使 B (FULL vs MIRROR sector 审计) 表现好, 它仍不是完整 CORE market proxy。
BENCHMARK_SCOPE_ID = "MIRROR_CORE_A_SHARE_PROXY_V1"
# 旧 benchmark identity, 2026-09-24 被 MIRROR_CORE_A_SHARE_PROXY_V1 supersede (float_mv cov 审计 39.5%)
BENCHMARK_SCOPE_SUPERSEDED_ID = "CORE_A_SHARE_MARKET_PROXY_V1"
# 2026-09-27 新增 FULL 源基准域 (stockdb 规范全源, 非镜像): 全量重建 C2/C3/C4/C5 使用。
# MIRROR 域仅作回归/降级参考, 不能承担 sector diffusion
# (见 gate1c/GATE1C_C3_SECTOR_DIFFUSION_MIRROR_NOT_USABLE.md)。
BENCHMARK_SCOPE_ID_FULL = "FULL_A_SHARE_MARKET_V1"
FULL_SCOPE_FREEZE_DATE = "2026-09-27"
FULL_MARKET_NOTE = "full_stockdb_source_no_proxy_substitution"


def benchmark_scope_id(is_proxy: bool) -> str:
    """基准域 id 随分钟源切换: 全源=完整 CORE A 股市场; 镜像=代理 proxy。"""
    return BENCHMARK_SCOPE_ID_FULL if not is_proxy else BENCHMARK_SCOPE_ID
BENCHMARK_SCOPE_BOARDS = RESEARCH_DATA_SCOPE_BOARDS
BENCHMARK_FIELD_NAME = "core_market_return_same_time"
ALPHA_FIELD_NAME = "alpha_vs_core_market_proxy"

# --- 镜像代理代表性证据 (2026-09-24, FULL vs MIRROR 审计) ---
# 分母 = CORE = SH_MAIN/SZ_MAIN/GEM/STAR 的总流通市值 (P 日冻结):
#   proxy_total_weight_coverage_overall = Σ mirror 观察市值 / Σ CORE 总市值 ≈ 0.3951。
#   注意: 这不是 member coverage (镜像域内 ~95%), 而是"市场权重代表度"。
#   偏置方向明确: SH 主板最缺 (25%), GEM/STAR 覆盖最高 (56-68%)。
#   因此 MIRROR_CORE_A_SHARE_PROXY_V1 可承担"粗粒度 market regime / shock detector",
#   但不能承担"精确 full-market 相对行业 context"。
PROXY_TOTAL_WEIGHT_COVERAGE_OVERALL = 0.3951
PROXY_BOARD_WEIGHT_COVERAGE = {
    "SH_MAIN": 0.2536,
    "SZ_MAIN": 0.4734,
    "GEM": 0.6834,
    "STAR": 0.5615,
}
PROXY_BIAS_PROFILE = "mid_small_gem_star_tilt"

# --- 预检 Gate (2026-09-24 用户裁决, 看结果前冻结) ---
# 在 C3 / Intraday Sector Diffusion 之前, 判断 MIRROR 代理是否有资格继续承担
# sector-relative context。三级裁决:
#   PASS_WITH_SCOPE_LIMITATION / DEGRADED_BUT_RESEARCHABLE / FAIL
GATE1C_C3_PRECHECK_FULL_VS_MIRROR_SECTOR_V1 = \
    "GATE1C_C3_PRECHECK_FULL_VS_MIRROR_SECTOR_V1"
# 未来出现正 edge、准备上线扩大推理域时的硬 Gate (需 full source 重跑对账)
FULL_SOURCE_RECONCILIATION_GATE = "FULL_SOURCE_RECONCILIATION_GATE"

# ------------------------------------------------------------------ #
# 排除 reason code (显式状态, 禁止裸 drop)
# ------------------------------------------------------------------ #
RS_IN_SCOPE = "IN_RESEARCH_SCOPE_V1"
RS_BSE_UNSUPPORTED = "CURRENT_RESEARCH_UNSUPPORTED"       # BSE: 股票但当前不支持
RS_ASSET_OUT_OF_MODEL = "ASSET_TYPE_OUT_OF_CURRENT_STOCK_MODEL"
RS_MINUTE_MISSING = "MINUTE_SOURCE_MISSING"
RS_MINUTE_OUT_OF_RANGE = "OUT_OF_MINUTE_SOURCE_TIME_RANGE"
RS_TYPE_UNRESOLVED = "INSTRUMENT_TYPE_UNRESOLVED"
RS_MINUTE_SOURCE_UNKNOWN = "MINUTE_SOURCE_UNKNOWN"        # 源域不可判定 -> 不得判 valid

SUB_BSE_MINUTE = "MINUTE_DATA_UNAVAILABLE_CURRENT_SCOPE"
SUB_BSE_MICROSTRUCTURE = "DISTINCT_MICROSTRUCTURE"        # 30% 涨跌幅 + 流动性结构
SUB_ASSET_TYPE = "ASSET_TYPE_OUT_OF_CURRENT_STOCK_MODEL"

# benchmark 侧
BK_IN_SCOPE = "IN_BENCHMARK_SCOPE_V1"
BK_OUT_OF_SCOPE = "BENCHMARK_SCOPE_EXCLUDES_INSTRUMENT"
BK_MINUTE_UNAVAILABLE = "BENCHMARK_MEMBER_MINUTE_UNAVAILABLE"

# ------------------------------------------------------------------ #
# 分钟源声明范围 (冻结常量, 与 gate1c_minute_io.MINUTE_SOURCE_EPOCH_START 同惯例)
#   镜像实测 (2026-09-24): 3959 票 / 409 交易日 / 20250102..20260908
#   数据刷新后必须重跑 verify_minute_range_scan() 并更新此处
# ------------------------------------------------------------------ #
MINUTE_SOURCE_RANGE_DECLARED = {
    "mirror": (20250102, 20260908),
}


def declared_minute_range(source: str = "mirror") -> tuple:
    lo, hi = MINUTE_SOURCE_RANGE_DECLARED.get(source, (None, None))
    env_hi = os.environ.get("GATE1C_MINUTE_MAX_DATE")
    if env_hi:
        hi = int(env_hi)
    return lo, hi


# ------------------------------------------------------------------ #
# 分钟源域 (code 集合)
# ------------------------------------------------------------------ #
def minute_source_codes(strict: bool = False) -> Optional[set]:
    """当前分钟源实际覆盖的 code 集合。

    strict=True 时不吞异常 (数据源缺失应当暴露, 而不是退化为"全部有效")。
    返回 None = 无法确定源域 (调用方必须视为 UNKNOWN, 不得视为 valid)。
    """
    try:
        g1c = V5 / "opportunity_v2" / "gate1c"
        if str(g1c) not in sys.path:
            sys.path.insert(0, str(g1c))
        import gate1c_minute_io as IO
        if IO.resolve_source() == "mirror":
            return set(IO.mirror_codes())
        return None          # stockdb 源为全市场, 域由 universe 决定, 不在此收窄
    except Exception:                                          # noqa: BLE001
        if strict:
            raise
        return None


def scope_manifest(is_proxy: bool = True) -> dict:
    """artifact 强制头 (每个实验/panel 都要落)。is_proxy 决定基准域身份与覆盖统计。"""
    lo, hi = declared_minute_range()
    bscope = benchmark_scope_id(is_proxy)
    is_full = not is_proxy
    return {
        "scope_contract_version": SCOPE_CONTRACT_VERSION,
        "instrument_master_version": IM.INSTRUMENT_MASTER_VERSION,
        "strategy_scope": STRATEGY_ELIGIBLE_SCOPE,
        "research_data_scope": RESEARCH_DATA_SCOPE_ID,
        "inference_scope": INFERENCE_SCOPE_ID,
        "benchmark_scope": bscope,
        "data_scope_id": DATA_SCOPE_ID_MINUTE,
        "instrument_scope": STRATEGY_ELIGIBLE_SCOPE,
        "alpha_field_name": ALPHA_FIELD_NAME,
        "benchmark_field_name": BENCHMARK_FIELD_NAME,
        "research_data_scope_boards": list(RESEARCH_DATA_SCOPE_BOARDS),
        "minute_source_range": [lo, hi],
        # 源覆盖统计: 全源=完整市场(≈1.0); 镜像=代理(冻结审计值, 见 PROXY_* 常量)
        "benchmark_is_full_market": is_full,
        "benchmark_proxy_total_weight_coverage_overall": (
            PROXY_TOTAL_WEIGHT_COVERAGE_OVERALL if not is_full else 1.0),
        "benchmark_proxy_board_weight_coverage": (
            dict(PROXY_BOARD_WEIGHT_COVERAGE) if not is_full else {}),
        "benchmark_proxy_bias_profile": (
            PROXY_BIAS_PROFILE if not is_full else "n/a_full_source"),
    }


# ------------------------------------------------------------------ #
# 单样本三域判定
# ------------------------------------------------------------------ #
def scope_flags(code: str, date: Optional[int] = None,
                minute_codes: Optional[set] = None,
                minute_range: Optional[tuple] = None,
                minute_source_is_full: bool = False) -> dict:
    """单 (code[, date]) -> 显式三域状态。**永不返回"静默 valid"**。

    依赖不可判定时 (例如分钟源域未知) 一律判 invalid 并给出 reason,
    绝不放行为 valid。

    minute_source_is_full=True 表示当前分钟源覆盖整个股票 universe (例如
    stockdb HTTP), 此时 minute_data_valid 只受时间范围约束, 不做 code 级收窄。
    默认 False —— 需要调用方显式声明, 不得默认假设源是全市场。
    """
    info = IM.instrument_info(code)
    c = info["code"]
    itype, board = info["instrument_type"], info["exchange_board"]

    if not minute_source_is_full and minute_codes is None:
        minute_codes = minute_source_codes()
    if minute_range is None:
        minute_range = declared_minute_range()
    lo, hi = minute_range

    # ---- research scope ----
    sub: list = []
    if itype == IM.T_UNKNOWN:
        rs_valid, rs_reason = 0, RS_TYPE_UNRESOLVED
    elif itype != IM.T_STOCK:
        rs_valid, rs_reason = 0, RS_ASSET_OUT_OF_MODEL
        sub.append(SUB_ASSET_TYPE)
    elif board == IM.B_BSE:
        # BSE 仍是股票 -> CURRENT_RESEARCH_UNSUPPORTED, 不是 PERMANENTLY_INVALID
        rs_valid, rs_reason = 0, RS_BSE_UNSUPPORTED
        sub += [SUB_BSE_MINUTE, SUB_BSE_MICROSTRUCTURE]
    elif board not in RESEARCH_DATA_SCOPE_BOARDS:
        rs_valid, rs_reason = 0, RS_ASSET_OUT_OF_MODEL
        sub.append(SUB_ASSET_TYPE)
    else:
        rs_valid, rs_reason = 1, RS_IN_SCOPE

    # ---- minute data validity ----
    if minute_codes is None and not minute_source_is_full:
        md_valid, md_reason = 0, RS_MINUTE_SOURCE_UNKNOWN   # 源域未知 -> 不得判 valid
    elif (not minute_source_is_full) and c not in minute_codes:
        md_valid, md_reason = 0, RS_MINUTE_MISSING
    elif date is not None and lo is not None and hi is not None and not (lo <= int(date) <= hi):
        md_valid, md_reason = 0, RS_MINUTE_OUT_OF_RANGE
    else:
        md_valid, md_reason = 1, "IN_MINUTE_SOURCE_SCOPE"

    # ---- benchmark validity ----
    if not rs_valid:
        bk_valid, bk_reason = 0, BK_OUT_OF_SCOPE
    elif not md_valid:
        bk_valid, bk_reason = 0, BK_MINUTE_UNAVAILABLE
    else:
        bk_valid, bk_reason = 1, BK_IN_SCOPE

    return {
        "instrument_type": itype,
        "exchange_board": board,
        "instrument_type_conflict": info["instrument_type_conflict"],
        "research_scope_valid": rs_valid,
        "research_scope_reason": rs_reason,
        "research_scope_subreasons": "|".join(sub),
        "minute_data_valid": md_valid,
        "minute_data_invalid_reason": "" if md_valid else md_reason,
        "benchmark_valid": bk_valid,
        "benchmark_invalid_reason": "" if bk_valid else bk_reason,
        "research_data_scope": RESEARCH_DATA_SCOPE_ID,
        "benchmark_scope": BENCHMARK_SCOPE_ID,
        "scope_contract_version": SCOPE_CONTRACT_VERSION,
    }


# ------------------------------------------------------------------ #
# 母表批注 (不删行)
# ------------------------------------------------------------------ #
_SCOPE_COLS = [
    "instrument_type", "exchange_board", "instrument_type_conflict",
    "research_scope_valid", "research_scope_reason", "research_scope_subreasons",
    "minute_data_valid", "minute_data_invalid_reason",
    "benchmark_valid", "benchmark_invalid_reason",
    "research_data_scope", "benchmark_scope", "scope_contract_version",
]


def annotate_scope(df: pd.DataFrame, code_col: str = "code",
                   date_col: str = "date",
                   minute_codes: Optional[set] = None,
                   minute_source_is_full: bool = False) -> pd.DataFrame:
    """给任意母表加显式三域列。**绝不 drop 行**。

    全表落盘时, BSE/ETF/未采集票仍在表里, 只是 research_scope_valid=0 + reason,
    半年后仍能查出它为什么没进训练。

    minute_source_is_full=True: 源覆盖整个股票 universe (stockdb HTTP), 此时
    minute_data_valid 只受时间范围约束。默认 False。
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    codes = out[code_col].astype(str).str.zfill(6)
    uniq = sorted(codes.unique().tolist())
    master = IM.build_instrument_master(uniq).set_index("code")

    if not minute_source_is_full and minute_codes is None:
        minute_codes = minute_source_codes()
    lo, hi = declared_minute_range()

    out["instrument_type"] = codes.map(master["instrument_type"])
    out["exchange_board"] = codes.map(master["exchange_board"])
    out["instrument_type_conflict"] = codes.map(
        master["instrument_type_conflict"]).fillna(0).astype(int)

    itype = out["instrument_type"]
    board = out["exchange_board"]
    is_stock = itype.eq(IM.T_STOCK)
    is_bse = board.eq(IM.B_BSE)
    in_boards = board.isin(list(RESEARCH_DATA_SCOPE_BOARDS))
    unknown = itype.eq(IM.T_UNKNOWN)

    rs_valid = (is_stock & in_boards).astype(int)
    rs_reason = pd.Series(RS_IN_SCOPE, index=out.index)
    rs_reason = rs_reason.mask(unknown, RS_TYPE_UNRESOLVED)
    rs_reason = rs_reason.mask(~is_stock & ~unknown, RS_ASSET_OUT_OF_MODEL)
    rs_reason = rs_reason.mask(is_stock & is_bse, RS_BSE_UNSUPPORTED)
    rs_reason = rs_reason.mask(is_stock & ~is_bse & ~in_boards, RS_ASSET_OUT_OF_MODEL)

    sub = pd.Series("", index=out.index)
    sub = sub.mask(is_stock & is_bse, f"{SUB_BSE_MINUTE}|{SUB_BSE_MICROSTRUCTURE}")
    sub = sub.mask(~is_stock & ~unknown, SUB_ASSET_TYPE)

    if minute_codes is None and not minute_source_is_full:
        md_valid = pd.Series(0, index=out.index)
        md_reason = pd.Series(RS_MINUTE_SOURCE_UNKNOWN, index=out.index)
    else:
        if minute_source_is_full:
            md_valid = pd.Series(1, index=out.index)
            md_reason = pd.Series("", index=out.index)
        else:
            in_min = codes.isin(minute_codes)
            md_valid = in_min.astype(int)
            md_reason = pd.Series("", index=out.index)
            md_reason = md_reason.mask(~in_min, RS_MINUTE_MISSING)
        if date_col in out.columns and lo is not None and hi is not None:
            d = pd.to_numeric(out[date_col], errors="coerce")
            out_of_range = d.notna() & ((d < lo) | (d > hi))
            md_valid = md_valid.mask(out_of_range, 0)
            md_reason = md_reason.mask(out_of_range, RS_MINUTE_OUT_OF_RANGE)

    out["research_scope_valid"] = rs_valid
    out["research_scope_reason"] = rs_reason
    out["research_scope_subreasons"] = sub
    out["minute_data_valid"] = md_valid
    out["minute_data_invalid_reason"] = md_reason
    bk_valid = (rs_valid.eq(1) & md_valid.eq(1)).astype(int)
    bk_reason = pd.Series("", index=out.index)
    bk_reason = bk_reason.mask(rs_valid.eq(0), BK_OUT_OF_SCOPE)
    bk_reason = bk_reason.mask(rs_valid.eq(1) & md_valid.eq(0), BK_MINUTE_UNAVAILABLE)
    out["benchmark_valid"] = bk_valid
    out["benchmark_invalid_reason"] = bk_reason
    out["research_data_scope"] = RESEARCH_DATA_SCOPE_ID
    out["benchmark_scope"] = benchmark_scope_id(minute_source_is_full)
    out["scope_contract_version"] = SCOPE_CONTRACT_VERSION
    return out


def coverage_report(df: pd.DataFrame, date_col: str = "date",
                    setup_col: Optional[str] = "primary_setup",
                    sector_col: Optional[str] = None,
                    min_date: Optional[int] = None) -> dict:
    """coverage_by_trade_date / by_symbol / by_setup / by_sector。

    只统计 research scope 内的样本 (分母 = scope 内应覆盖的样本)。
    """
    if df is None or df.empty:
        return {}
    d = df if min_date is None else df[pd.to_numeric(df[date_col], errors="coerce") >= min_date]
    core = d[d["research_scope_valid"] == 1]
    rep: dict = {
        "n_rows_all": int(len(d)),
        "n_rows_in_scope": int(len(core)),
        "scope_share": round(float(len(core) / max(len(d), 1)), 4),
        "weighted_minute_coverage_in_scope": round(float(core["minute_data_valid"].mean()), 4)
        if len(core) else 0.0,
        "excluded_by_reason": d.loc[d["research_scope_valid"] == 0,
                                    "research_scope_reason"].value_counts().to_dict(),
    }
    if len(core):
        g = core.groupby(date_col)["minute_data_valid"].agg(["mean", "size"])
        rep["coverage_by_trade_date"] = {
            "n_dates": int(len(g)),
            "min": round(float(g["mean"].min()), 4),
            "p05": round(float(g["mean"].quantile(0.05)), 4),
            "median": round(float(g["mean"].median()), 4),
            "max": round(float(g["mean"].max()), 4),
            "worst_dates": g.sort_values("mean").head(10)["mean"].round(4).to_dict(),
        }
        rep["coverage_by_board"] = (core.groupby("exchange_board")["minute_data_valid"]
                                    .agg(["mean", "size"]).round(4)
                                    .to_dict(orient="index"))
        if setup_col and setup_col in core.columns:
            rep["coverage_by_setup"] = (core.groupby(setup_col)["minute_data_valid"]
                                        .agg(["mean", "size"]).round(4)
                                        .to_dict(orient="index"))
        if sector_col and sector_col in core.columns:
            rep["coverage_by_sector"] = (core.groupby(sector_col)["minute_data_valid"]
                                         .agg(["mean", "size"]).round(4)
                                         .to_dict(orient="index"))
        cc = core.groupby("code")["minute_data_valid"].mean()
        rep["coverage_by_symbol"] = {
            "n_symbols": int(cc.size),
            "n_zero_coverage_symbols": int((cc == 0).sum()),
            "n_full_coverage_symbols": int((cc == 1).sum()),
        }
    return rep


def verify_minute_range_scan(mirror_root: Optional[str] = None,
                             max_files: Optional[int] = None) -> dict:
    """重扫分钟镜像, 复derive (min_date, max_date, n_dates, n_codes)。

    数据刷新后必须跑一次, 并与 MINUTE_SOURCE_RANGE_DECLARED 对账。
    """
    import glob
    from collections import Counter
    import numpy as np
    import pyarrow.parquet as pq

    root = mirror_root or os.environ.get(
        "GATE1C_MIRROR_ROOT",
        r"D:\stargazer\stargazer_ml\pullback_train\minute_cache_clean")
    files = sorted(glob.glob(os.path.join(root, "*.parquet")))
    if max_files:
        files = files[:max_files]
    per: Counter = Counter()
    codes = set()
    for f in files:
        a = pq.read_table(f, columns=["date"]).column("date").to_numpy()
        codes.add(os.path.basename(f)[:6])
        for d in np.unique(a // 1000000).tolist():
            per[d] += 1
    return {
        "mirror_root": root,
        "n_files": len(files),
        "n_codes": len(codes),
        "n_dates": len(per),
        "min_date": min(per) if per else None,
        "max_date": max(per) if per else None,
        "declared": list(declared_minute_range("mirror")),
        "matches_declared": (bool(per)
                             and (min(per), max(per)) == declared_minute_range("mirror")),
    }


def _selftest() -> None:
    m = scope_manifest()
    assert m["research_data_scope"] == RESEARCH_DATA_SCOPE_ID
    assert m["benchmark_scope"] == BENCHMARK_SCOPE_ID
    # 000002: 核心域 + 镜像内 -> 三域全 valid
    f = scope_flags("000002", 20250303)
    assert f["research_scope_valid"] == 1, f
    assert f["minute_data_valid"] == 1, f
    assert f["benchmark_valid"] == 1, f
    # 600519: 核心域股票但**不在当前分钟镜像** -> scope valid / minute invalid。
    # 这是"域一致但数据缺"的正确表达, 必须与 BSE/ETF 的"域外"区分开。
    f = scope_flags("600519", 20250303)
    assert f["research_scope_valid"] == 1, f
    assert f["minute_data_valid"] == 0, f
    assert f["minute_data_invalid_reason"] == RS_MINUTE_MISSING, f
    assert f["benchmark_valid"] == 0, f
    assert f["benchmark_invalid_reason"] == BK_MINUTE_UNAVAILABLE, f
    f = scope_flags("920819", 20250303)
    assert f["research_scope_valid"] == 0 and f["research_scope_reason"] == RS_BSE_UNSUPPORTED
    assert f["benchmark_invalid_reason"] == BK_OUT_OF_SCOPE
    f = scope_flags("510300", 20250303)
    assert f["research_scope_reason"] == RS_ASSET_OUT_OF_MODEL
    # 20261101: 在镜像内但日期超出分钟源声明范围 (镜像止于 20260908)
    f = scope_flags("000002", 20261101)
    assert f["minute_data_valid"] == 0, f
    assert f["minute_data_invalid_reason"] == RS_MINUTE_OUT_OF_RANGE, f
    print("research_scope selftest PASS")
    print(scope_manifest())


if __name__ == "__main__":
    _selftest()
