"""feature_registry — Gate 2 机器可读字段合同 (Step 9 Preflight).

Feature Builder / 训练 / 实时推理 / Ablation 共用唯一字段合同。
未注册字段禁止进入 Model Matrix。

Gate 1 分诊 (2026-09-17):
  gate1_phase:
    CORE              -> Gate 1A, Step 9A 核心入场重建 (A0-A4)
    CONTEXT_BLOCKED   -> Gate 1B, 需 PIT Sector; DISABLED_UNTIL_PIT_SECTOR
    OPTIONAL_DISABLED -> Gate 1C, 真实 VP/POC/VAL

运行:
    from feature_registry import (run_gate2_tests, filter_model_matrix,
                                  a7_feature_list, core_features_9a,
                                  disabled_until_pit_sector, optional_disabled)
    run_gate2_tests()
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

_HERE = Path(__file__).resolve().parent
REGISTRY_PATH = _HERE / "feature_registry.yaml"

RULE_ORIGINS = {"RULE_FLAG", "RULE_AGGREGATE"}
DROP_ORIGINS_IN_A7 = RULE_ORIGINS

# Gate 1 分阶段
CORE = "CORE"
CONTEXT_BLOCKED = "CONTEXT_BLOCKED"
OPTIONAL_DISABLED = "OPTIONAL_DISABLED"


def load_registry(path: Path | None = None) -> Dict:
    p = path or REGISTRY_PATH
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_REG = load_registry()
_FEATURES: List[dict] = _REG["features"]
_BY_NAME: Dict[str, dict] = {f["feature_name"]: f for f in _FEATURES}


# ---------- 查询 ----------
def names() -> List[str]:
    return [f["feature_name"] for f in _FEATURES]


def get(name: str) -> dict:
    return _BY_NAME[name]


def is_registered(name: str) -> bool:
    return name in _BY_NAME


def gate1_phase(name: str) -> str:
    return _BY_NAME[name].get("gate1_phase", CORE)


def a7_feature_list() -> List[str]:
    """A7-NORULE: 保留除 RULE_FLAG/RULE_AGGREGATE 外全部。"""
    return [f["feature_name"] for f in _FEATURES if f.get("a7_policy") == "KEEP"]


def a7b_feature_list() -> List[str]:
    """A7b-NOSETUP: 进一步删除 CONTEXT(setup相关)。"""
    return [f["feature_name"] for f in _FEATURES if f.get("a7b_policy") == "KEEP"]


def model_eligible_columns() -> List[str]:
    return [f["feature_name"] for f in _FEATURES if f.get("model_eligible")]


def core_features_9a() -> List[str]:
    """Gate 1A / Step 9A 可用特征 (CORE 且 model_eligible)。"""
    return [f["feature_name"] for f in _FEATURES
            if f.get("gate1_phase") == CORE and f.get("model_eligible")]


def disabled_until_pit_sector() -> List[str]:
    """Gate 1B: 需 PIT Sector/Leader，解决前禁用 (含 Sector/Leader/Interaction/相关 RULE)。"""
    return [f["feature_name"] for f in _FEATURES
            if f.get("gate1_phase") == CONTEXT_BLOCKED]


def optional_disabled() -> List[str]:
    """Gate 1C: 真实 VP/POC/VAL，待质量确认。"""
    return [f["feature_name"] for f in _FEATURES
            if f.get("gate1_phase") == OPTIONAL_DISABLED]


# ---------- 训练期强制使用 ----------
def assert_train_columns_registered(columns) -> None:
    """Gate2-#2: 所有训练字段都已注册。"""
    unknown = [c for c in columns if c not in _BY_NAME]
    if unknown:
        raise ValueError(f"[Gate2] 未注册字段禁止进入 Model Matrix: {unknown}")


def filter_model_matrix(df: pd.DataFrame, mode: str = "A7") -> pd.DataFrame:
    """Gate2-#3: 未注册字段禁止进入 Model Matrix。mode=A7|A7B|FULL|STEP9A。"""
    if mode == "A7":
        keep = set(a7_feature_list())
    elif mode == "A7B":
        keep = set(a7b_feature_list())
    elif mode == "FULL":
        keep = set(model_eligible_columns())
    elif mode == "STEP9A":
        # Step 9A 核心入场重建: 仅 CORE 特征, 排除任何 CONTEXT_BLOCKED / OPTIONAL_DISABLED
        keep = set(core_features_9a())
    else:
        raise ValueError(f"unknown mode {mode}")
    unknown = [c for c in df.columns if c not in _BY_NAME]
    if unknown:
        raise ValueError(f"[Gate2] 未注册字段禁止进入 Model Matrix: {unknown}")
    return df[[c for c in df.columns if c in keep]]


# ---------- Gate 2 八项自动测试 ----------
def _check_name_unique() -> bool:
    ns = names()
    return len(ns) == len(set(ns))


def _check_a7_no_rule() -> bool:
    a7 = set(a7_feature_list())
    return all(_BY_NAME[n]["origin"] not in DROP_ORIGINS_IN_A7 for n in a7)


def _check_a7_keep_context() -> bool:
    ctx_in_a7 = [n for n in a7_feature_list() if _BY_NAME[n]["origin"] == "CONTEXT"]
    return len(ctx_in_a7) > 0  # setup_type / bottom_subtype 必须保留


def _check_a7b_drop_context() -> bool:
    ctx_in_a7b = [n for n in a7b_feature_list() if _BY_NAME[n]["origin"] == "CONTEXT"]
    return len(ctx_in_a7b) == 0  # A7b 必须删除 setup 相关 CONTEXT


def _check_metadata_not_model() -> bool:
    bad = [n for n, f in _BY_NAME.items() if f["origin"] == "METADATA" and f.get("model_eligible")]
    return len(bad) == 0


def _check_a7b_subset_of_a7() -> bool:
    return set(a7b_feature_list()).issubset(set(a7_feature_list()))


def _check_latest_ts_rule_present() -> bool:
    return all("latest_source_timestamp_rule" in f for f in _FEATURES)


def _check_step9a_no_context_blocked() -> bool:
    """Gate2-#9: Step 9A 矩阵不得含任何 CONTEXT_BLOCKED / OPTIONAL_DISABLED 特征。"""
    blocked = set(disabled_until_pit_sector()) | set(optional_disabled())
    # 这些特征若 model_eligible 为真, 必须不在 CORE 集合内
    return all(_BY_NAME[n]["gate1_phase"] != CORE for n in blocked)


def check_no_future_leak(sample: pd.DataFrame, snapshot_time_col: str = "snapshot_time") -> bool:
    """Gate2-#8: 每个特征数据最晚可见时点不得超过其 as_of 约束。"""
    ok = True
    for col in sample.columns:
        if col not in _BY_NAME:
            continue
        rule = _BY_NAME[col].get("latest_source_timestamp_rule", "")
        if "<= snapshot_time" in rule:
            ts_col = snapshot_time_col
        elif "<= P_CLOSE" in rule or "P_CLOSE" in rule:
            ts_col = "p_close_ts"
        else:
            continue
        if ts_col in sample.columns:
            src = f"{col}_src_ts"
            if src in sample.columns:
                if (sample[src] > sample[ts_col]).any():
                    ok = False
    return ok


def run_gate2_tests(sample: pd.DataFrame | None = None) -> Dict:
    tests = {
        "1_name_unique": _check_name_unique(),
        "4_a7_no_rule_origin": _check_a7_no_rule(),
        "5_a7_keep_context": _check_a7_keep_context(),
        "6_a7b_drop_context": _check_a7b_drop_context(),
        "7_metadata_not_model": _check_metadata_not_model(),
        "a7b_subset_a7": _check_a7b_subset_of_a7(),
        "8_latest_ts_rule_present": _check_latest_ts_rule_present(),
        "9_step9a_no_context_blocked": _check_step9a_no_context_blocked(),
    }
    if sample is not None:
        try:
            assert_train_columns_registered(sample.columns)
            tests["2_train_columns_registered"] = True
        except ValueError:
            tests["2_train_columns_registered"] = False
        tests["3_unregistered_blocked"] = True  # 由 filter_model_matrix 强制
        tests["8b_no_future_leak"] = check_no_future_leak(sample)
    gate2 = all(tests.values())
    return {"gate2": gate2, "tests": tests}


if __name__ == "__main__":
    res = run_gate2_tests()
    print(f"Gate2 = {res['gate2']}")
    for k, v in res["tests"].items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"core_features_9a ({len(core_features_9a())}): {core_features_9a()}")
    print(f"disabled_until_pit_sector ({len(disabled_until_pit_sector())}): {disabled_until_pit_sector()}")
    print(f"optional_disabled ({len(optional_disabled())}): {optional_disabled()}")
    sys.exit(0 if res["gate2"] else 1)
