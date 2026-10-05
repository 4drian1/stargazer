"""production_chain.py — C4 拥有的最薄生产链编排（不含上游实时 ingest）。

上游（部署侧负责）：raw document persist → clustering → entity/taxonomy → available_at
本模块只覆盖从「available_at 已确认」之后的 C4 段：

    C4 scoring domain → materiality_features_v1 → model v1.0.0
    → CorvusInformationSnapshot → shadow_store (append-only)

返回 dict 供部署层计算四个健康指标：
    ingest_to_prediction_latency / duplicate_retry_rate /
    not_scored_rate + reason / feature_missing_drift_rate
不加载/不修改模型；模型侧已冻结。
"""
from __future__ import annotations

import time

from c4_shadow import score_event
from corvus.materiality_v1.shadow_store import ShadowStore


def run_shadow_once(cluster_record, store=None, as_of_date=99991231,
                    daily_provider=None, model=None):
    if store is None:
        store = ShadowStore()
    t0 = time.perf_counter()
    snap, diag = score_event(cluster_record, as_of_date=as_of_date,
                             daily_provider=daily_provider, model=model)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    if diag["eligible"]:
        store.upsert_prediction(snap)
        action = "SCORED"
    else:
        action = "NOT_SCORED"
    reason = ";".join(diag.get("problems", [])) or snap.calibration_status
    return {
        "event_cluster_id": snap.event_cluster_id,
        "action": action,
        "eligible": diag["eligible"],
        "reason": reason,
        "p_materiality": snap.p_materiality,
        "model_version": snap.materiality_model_version,
        "feature_asof": snap.feature_asof,
        "missing_warnings": snap.missing_warnings,
        "latency_ms": latency_ms,
        "record": snap.to_record(),
    }
