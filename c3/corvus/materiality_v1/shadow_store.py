"""shadow_store.py — C4 Shadow 持久化参考实现（双表不可变结构）。

生产 PostgreSQL 层应**物理拆成两张表**，由数据库结构（而非仅靠代码纪律）保证不可变：

  c4_shadow_prediction(
      event_cluster_id, materiality_model_version, prediction_ts, p_materiality,
      feature_snapshot_id, ...,
      PRIMARY KEY(event_cluster_id, materiality_model_version))
      → 永不 UPDATE；仅允许 INSERT（同键重复 INSERT 视为幂等去重 / 拒绝）

  c4_shadow_realization(
      event_cluster_id, materiality_model_version, label_horizon, realized_label,
      label_ts, label_version, ...,
      PRIMARY KEY(event_cluster_id, materiality_model_version, label_horizon))
      → 标签成熟后独立 append；绝不回写 prediction 表

内存参考实现用两个 dict 对应这两张表：
- _predictions:  key = (event_cluster_id, model_version)
- _realizations: key = (event_cluster_id, model_version, label_horizon)
"""
from __future__ import annotations

import datetime as dt
from typing import Optional


class ShadowStore:
    def __init__(self):
        self._predictions: dict = {}
        self._realizations: dict = {}

    # ---- prediction 表（不可变） ----
    def upsert_prediction(self, snap) -> str:
        key = (snap.event_cluster_id, snap.materiality_model_version)
        existing = self._predictions.get(key)
        if existing is not None and existing.get("p_materiality") is not None:
            if existing["p_materiality"] != snap.p_materiality:
                raise RuntimeError(
                    f"IDEMPOTENCY VIOLATION: 同一 (event_cluster_id, version)={key} "
                    "产生了不同 prediction；应拒绝而不是重复写。")
        rec = dict(existing or {})
        rec.update(snap.to_record())
        rec["prediction_ts"] = snap.created_ts
        self._predictions[key] = rec
        return "INSERTED" if existing is None else "UNCHANGED"

    # ---- realization 表（独立 append，绝不回写 prediction） ----
    def attach_realized_label(self, event_cluster_id, materiality_model_version,
                              label_horizon, realized_label,
                              label_ts: Optional[str] = None,
                              label_version: Optional[str] = None) -> None:
        if (event_cluster_id, materiality_model_version) not in self._predictions:
            raise KeyError(
                f"未找到 prediction 行 ({event_cluster_id}, {materiality_model_version})；"
                "不能先于打分附加 realized label（APPEND-ONLY 顺序约束）。")
        rkey = (event_cluster_id, materiality_model_version, label_horizon)
        self._realizations[rkey] = {
            "event_cluster_id": event_cluster_id,
            "materiality_model_version": materiality_model_version,
            "label_horizon": label_horizon,
            "realized_label": bool(realized_label),
            "label_ts": label_ts or dt.datetime.now().isoformat(timespec="seconds"),
            "label_version": label_version or "v1",
        }

    def get(self, event_cluster_id, materiality_model_version):
        return self._predictions.get((event_cluster_id, materiality_model_version))

    def get_realization(self, event_cluster_id, materiality_model_version, label_horizon):
        return self._realizations.get(
            (event_cluster_id, materiality_model_version, label_horizon))

    def __len__(self):
        return len(self._predictions)

    def __bool__(self):
        # 实例恒为真；空与否请用 len()。避免 `store or ShadowStore()` 把空 store 当 falsy 误建新对象。
        return True

    def n_realizations(self):
        return len(self._realizations)
