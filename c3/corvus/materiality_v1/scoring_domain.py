"""scoring_domain.py — C4_SHADOW_SCORING_DOMAIN_V1 契约与门禁。

关键边界：C4 实时评分域 **不要求、也不使用** 任何标签侧信息：
  label_valid / corp_action_within_horizon / later_realized_label / training_eligible
这些只在事后标签成熟与评估阶段使用。线上若误用 corp_action_within_horizon
（要看未来 T..T+5）即未来信息泄漏，因此本模块刻意不检查它们。
"""
from __future__ import annotations

import pandas as pd

from .materiality_features_v1 import MODEL_FEATURES

CONTRACT = {
    "name": "C4_SHADOW_SCORING_DOMAIN_V1",
    "requires": [
        "event_tier == P0",
        "cluster-level",
        "non-HEAVY (attribution_quality != HEAVY)",
        "category != DISTRIBUTION",
        "event/entity mapping valid (code 非 NaN)",
        "available_at 已真实可见 (announce_date <= as_of_date)",
        "feature completeness 满足 (MODEL_FEATURES 缺失比例 <= 阈值)",
    ],
    "forbidden_at_scoring_time": [
        "label_valid",
        "corp_action_within_horizon",
        "later_realized_label",
        "training_eligible",
    ],
    "note": "标签侧字段只用于事后标签成熟与 shadow evaluation，绝不进入实时打分。",
}


def is_scoring_eligible(record, feat_row=None, as_of_date=None,
                        completeness_max_nan_frac: float = 0.5):
    """返回 (eligible: bool, problems: list[str])。

    record: dict/Series，含 tier / attribution_quality / category / code / announce_date。
    feat_row: 该簇已提取的特征行（Series）；用于 completeness 检查（可选）。
    本函数刻意不检查任何标签侧字段。
    """
    problems = []
    if as_of_date is None:
        problems.append("scoring_asof_required")
    else:
        try:
            feature_day = int(record["label_start"])
            if feature_day > int(as_of_date):
                problems.append("feature_asof_in_future")
        except (KeyError, ValueError, TypeError, OverflowError):
            problems.append("feature_asof_unknown")
    if str(record.get("tier")) != "P0":
        problems.append("tier!=P0")
    if str(record.get("attribution_quality")) == "HEAVY":
        problems.append("attribution_quality==HEAVY")
    if str(record.get("category")) == "DISTRIBUTION":
        problems.append("category==DISTRIBUTION")
    if pd.isna(record.get("code")):
        problems.append("entity_unmapped")
    if as_of_date is not None and not pd.isna(record.get("announce_date")):
        try:
            if int(record["announce_date"]) > int(as_of_date):
                problems.append("available_at_in_future")
        except (ValueError, TypeError):
            problems.append("available_at_unparseable")
    if feat_row is not None:
        nan_frac = pd.Series(feat_row).reindex(MODEL_FEATURES).isna().mean()
        if nan_frac > completeness_max_nan_frac:
            problems.append("feature_incomplete")
    return (len(problems) == 0, problems)
