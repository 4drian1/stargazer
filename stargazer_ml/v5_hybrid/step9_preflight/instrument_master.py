# -*- coding: utf-8 -*-
"""INSTRUMENT_MASTER_V1 — instrument_type / exchange_board 权威解析 (2026-09-24 冻结)

存在原因
--------
`data_source_provenance.get_universe()` = 日线 parquet 全部 code, **不含资产类型**。
后果 (2026-09-24 实测): pool_daily_v11 epoch 候选里有 9,633 行落在非股票代码段
(ETF/LOF/REIT), 它们不是"数据缺失"问题, 而是**资产类型本就不属于股票机会模型**。
禁止再用"代码前缀临时过滤"——必须冻结成显式 instrument_type 字段。

冻结的多源解析顺序 (authority order)
------------------------------------
A. SW_PIT_MEMBERSHIP      : opportunity_v2/o5_sector_lag/pit_membership.parquet
                            申万 L1 行业 PIT 区间表。实测 5,929 symbol, **0 个 ETF**
                            (北交所股票在内, 378)。→ 只含股票, 强权威。
B. STOCKDB_CONCEPT_BOARD  : opportunity_v2/o5_sector_lag/_upstream/
                            stockdb_boards_snapshot.parquet
                            同花顺概念 + 申万行业指数成员并集。实测 5,647 code,
                            ETF-like 5 个, 北交所 2,774。→ 只含股票, 次级权威。
C. CODE_NAMESPACE_RULE    : A 股代码段分配规则 (交易所编号事实, 非启发式)
                            —— 权威决定 exchange_board; 资产类型仅在 A/B 均缺席时兜底。

冲突纪律 (禁止静默择一)
----------------------
namespace 判 ETF/FUND/REIT 而 A/B 判 STOCK (或反之) 时:
    instrument_type          = 权威源值 (A > B)
    instrument_type_conflict = 1
    namespace_instrument_type / resolution_method 同时落盘, 供审计。
B 的表里有 5 个 ETF-like 成员引用 —— 正是此类冲突, 不得当作噪声抹掉。

冻结规则
--------
本文件新增/修改 = INSTRUMENT_MASTER_VERSION 升级。禁止原地改语义:
instrument_type 只能加值不能改值含义; exchange_board 同理。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

HERE = Path(__file__).resolve().parent
V5 = HERE.parent

INSTRUMENT_MASTER_VERSION = "INSTRUMENT_MASTER_V1"
NAMESPACE_RULE_VERSION = "CN_A_SHARE_CODE_NAMESPACE_2026"

# ------------------------------------------------------------------ #
# 枚举 (冻结, 只加不改)
# ------------------------------------------------------------------ #
T_STOCK = "STOCK"
T_ETF = "ETF"
T_REIT = "REIT"
T_FUND = "FUND"
T_BOND = "BOND"
T_INDEX = "INDEX"
T_UNKNOWN = "UNKNOWN"

INSTRUMENT_TYPES = (T_STOCK, T_ETF, T_REIT, T_FUND, T_BOND, T_INDEX, T_UNKNOWN)

B_SH = "SH_MAIN"
B_SZ = "SZ_MAIN"
B_GEM = "GEM"
B_STAR = "STAR"
B_BSE = "BSE"
B_FUND_LIST = "FUND_LISTING"
B_BOND_LIST = "BOND_LISTING"
B_UNKNOWN = "UNKNOWN"

EXCHANGE_BOARDS = (B_SH, B_SZ, B_GEM, B_STAR, B_BSE,
                   B_FUND_LIST, B_BOND_LIST, B_UNKNOWN)

# 解析方法 (落盘审计用)
M_SW = "SW_PIT_MEMBERSHIP"
M_CONCEPT = "STOCKDB_CONCEPT_BOARD"
M_NAMESPACE = "CODE_NAMESPACE_RULE"
M_UNRESOLVED = "UNRESOLVED"

# ------------------------------------------------------------------ #
# 代码段命名空间规则 (交易所编号事实)
#   先查 3 位前缀, 再查 2 位前缀 (顺序不可颠倒: '159' 会被 '15' 误捕)
# ------------------------------------------------------------------ #
_NS_P3: dict[str, tuple[str, str]] = {
    # 沪市主板
    "600": (B_SH, T_STOCK), "601": (B_SH, T_STOCK),
    "603": (B_SH, T_STOCK), "605": (B_SH, T_STOCK),
    # 科创板
    "688": (B_STAR, T_STOCK), "689": (B_STAR, T_STOCK),
    # 深市主板 (含原中小板)
    "000": (B_SZ, T_STOCK), "001": (B_SZ, T_STOCK),
    "002": (B_SZ, T_STOCK), "003": (B_SZ, T_STOCK),
    # 创业板
    "300": (B_GEM, T_STOCK), "301": (B_GEM, T_STOCK), "302": (B_GEM, T_STOCK),
    # 沪市 LOF / 封闭基金
    "501": (B_FUND_LIST, T_FUND), "502": (B_FUND_LIST, T_FUND),
    "504": (B_FUND_LIST, T_FUND), "506": (B_FUND_LIST, T_FUND),
    # 沪市基础设施 REIT
    "508": (B_FUND_LIST, T_REIT),
    # 沪市 ETF
    "510": (B_FUND_LIST, T_ETF), "511": (B_FUND_LIST, T_ETF),
    "512": (B_FUND_LIST, T_ETF), "513": (B_FUND_LIST, T_ETF),
    "515": (B_FUND_LIST, T_ETF), "516": (B_FUND_LIST, T_ETF),
    "517": (B_FUND_LIST, T_ETF), "518": (B_FUND_LIST, T_ETF),
    "520": (B_FUND_LIST, T_ETF), "560": (B_FUND_LIST, T_ETF),
    "561": (B_FUND_LIST, T_ETF), "562": (B_FUND_LIST, T_ETF),
    "563": (B_FUND_LIST, T_ETF), "588": (B_FUND_LIST, T_ETF),
    "589": (B_FUND_LIST, T_ETF),
    # 深市 ETF
    "159": (B_FUND_LIST, T_ETF),
}

_NS_P2: dict[str, tuple[str, str]] = {
    # 北交所 (含原新三板精选层代码段)
    "43": (B_BSE, T_STOCK), "83": (B_BSE, T_STOCK),
    "87": (B_BSE, T_STOCK), "92": (B_BSE, T_STOCK),
    # 深市基金 (LOF/封闭)
    "15": (B_FUND_LIST, T_FUND), "16": (B_FUND_LIST, T_FUND),
    "18": (B_FUND_LIST, T_FUND),
    # 债券 (沪深)
    "10": (B_BOND_LIST, T_BOND), "11": (B_BOND_LIST, T_BOND),
    "12": (B_BOND_LIST, T_BOND), "13": (B_BOND_LIST, T_BOND),
}


def namespace_classify(code: str) -> tuple[str, str]:
    """代码段 -> (exchange_board, instrument_type)。未识别 -> (UNKNOWN, UNKNOWN)。

    仅表达"编号段分配事实", 不表达"该代码当日是否在交易"。
    """
    c = str(code).zfill(6)
    p3, p2 = c[:3], c[:2]
    if p3 in _NS_P3:
        return _NS_P3[p3]
    if p2 in _NS_P2:
        return _NS_P2[p2]
    return B_UNKNOWN, T_UNKNOWN


# ------------------------------------------------------------------ #
# 权威源 A / B
# ------------------------------------------------------------------ #
SW_MEMBERSHIP_PATH = Path(os.environ.get(
    "STARGAZER_PIT_MEMBERSHIP",
    str(V5 / "opportunity_v2" / "o5_sector_lag" / "pit_membership.parquet")))
CONCEPT_BOARDS_PATH = Path(os.environ.get(
    "STARGAZER_CONCEPT_BOARDS",
    str(V5 / "opportunity_v2" / "o5_sector_lag" / "_upstream"
        / "stockdb_boards_snapshot.parquet")))


@lru_cache(maxsize=1)
def sw_member_codes() -> frozenset:
    """权威 A: 申万 L1 PIT membership 出现过的 symbol (只含股票)。"""
    if not SW_MEMBERSHIP_PATH.exists():
        return frozenset()
    df = pd.read_parquet(SW_MEMBERSHIP_PATH, columns=["symbol"])
    return frozenset(df["symbol"].astype(str).str.zfill(6).unique().tolist())


@lru_cache(maxsize=1)
def concept_member_codes() -> frozenset:
    """权威 B: stockdb 概念/行业指数成员并集 (只含股票)。"""
    if not CONCEPT_BOARDS_PATH.exists():
        return frozenset()
    df = pd.read_parquet(CONCEPT_BOARDS_PATH, columns=["symbols"])
    out: set = set()
    for lst in df["symbols"].tolist():
        if lst is None:
            continue
        for x in list(lst):
            out.add(str(x).zfill(6))
    return frozenset(out)


def authority_status() -> dict:
    return {
        "instrument_master_version": INSTRUMENT_MASTER_VERSION,
        "namespace_rule_version": NAMESPACE_RULE_VERSION,
        "sw_membership_path": str(SW_MEMBERSHIP_PATH),
        "sw_membership_available": SW_MEMBERSHIP_PATH.exists(),
        "sw_membership_codes": len(sw_member_codes()),
        "concept_boards_path": str(CONCEPT_BOARDS_PATH),
        "concept_boards_available": CONCEPT_BOARDS_PATH.exists(),
        "concept_board_codes": len(concept_member_codes()),
    }


# ------------------------------------------------------------------ #
# 单码解析
# ------------------------------------------------------------------ #
def instrument_info(code: str) -> dict:
    """单码 -> 显式 instrument 描述 (权威解析 + 冲突标记, 绝不静默择一)。"""
    c = str(code).zfill(6)
    board, ns_type = namespace_classify(c)
    in_sw = c in sw_member_codes()
    in_concept = c in concept_member_codes()

    if in_sw:
        resolved, method = T_STOCK, M_SW
    elif in_concept:
        resolved, method = T_STOCK, M_CONCEPT
    else:
        resolved, method = ns_type, (M_NAMESPACE if ns_type != T_UNKNOWN
                                    else M_UNRESOLVED)

    # 冲突: 两源对资产类型判定不一致 (禁止静默取一个)
    conflict = int(resolved != ns_type and T_UNKNOWN not in (resolved, ns_type))

    return {
        "code": c,
        "instrument_type": resolved,
        "exchange_board": board,
        "namespace_instrument_type": ns_type,
        "in_sw_membership": int(in_sw),
        "in_concept_board": int(in_concept),
        "resolution_method": method,
        "instrument_type_conflict": conflict,
        "instrument_master_version": INSTRUMENT_MASTER_VERSION,
        "namespace_rule_version": NAMESPACE_RULE_VERSION,
    }


def build_instrument_master(codes: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """全量 instrument master 表 (可落盘, 供候选母表 left-join)。

    codes=None 时取 canonical universe ∪ 两个权威源, 保证不漏。
    """
    if codes is None:
        base: set = set()
        try:
            import data_source_provenance as DSP
            base |= {str(x).zfill(6) for x in DSP.get_universe()}
        except Exception:                                      # noqa: BLE001
            pass
        base |= set(sw_member_codes())
        base |= set(concept_member_codes())
        codes = sorted(base)
    return pd.DataFrame([instrument_info(c) for c in codes])


def instrument_master_index(codes: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """以 code 为索引的 master (annotate 用)。"""
    return build_instrument_master(codes).set_index("code")


def assert_no_unresolved(df: pd.DataFrame, raise_on_unknown: bool = False) -> dict:
    """审计: master 里是否残留 UNKNOWN / 冲突。返回统计, 不静默。"""
    if df is None or df.empty:
        return {"n_rows": 0}
    unk = int((df["instrument_type"] == T_UNKNOWN).sum())
    cfl = int(df["instrument_type_conflict"].sum())
    if raise_on_unknown and (unk or cfl):
        raise ValueError(f"instrument master 未解析: UNKNOWN={unk} CONFLICT={cfl}")
    return {
        "n_rows": int(len(df)),
        "n_unknown": unk,
        "n_conflict": cfl,
        "by_type": df["instrument_type"].value_counts().to_dict(),
        "by_board": df["exchange_board"].value_counts().to_dict(),
        "by_method": df["resolution_method"].value_counts().to_dict(),
    }


# ------------------------------------------------------------------ #
# selftest
# ------------------------------------------------------------------ #
def _selftest() -> None:
    assert namespace_classify("600519") == (B_SH, T_STOCK)
    assert namespace_classify("688981") == (B_STAR, T_STOCK)
    assert namespace_classify("300750") == (B_GEM, T_STOCK)
    assert namespace_classify("000002") == (B_SZ, T_STOCK)
    assert namespace_classify("920819") == (B_BSE, T_STOCK)
    assert namespace_classify("159915") == (B_FUND_LIST, T_ETF)   # 前缀顺序回归
    assert namespace_classify("510300") == (B_FUND_LIST, T_ETF)
    assert namespace_classify("508000") == (B_FUND_LIST, T_REIT)
    assert namespace_classify("113050") == (B_BOND_LIST, T_BOND)
    # 权威覆盖决定性: 股票段但不在任何权威源 -> 仍按命名空间
    info = instrument_info("600519")
    assert info["instrument_type"] == T_STOCK
    assert info["exchange_board"] == B_SH
    assert info["instrument_type_conflict"] == 0
    st = authority_status()
    assert st["sw_membership_available"], "SW PIT membership 缺失, 解析退化为单源"
    print("instrument_master selftest PASS")
    print({k: st[k] for k in ("sw_membership_codes", "concept_board_codes",
                              "instrument_master_version")})


if __name__ == "__main__":
    _selftest()
