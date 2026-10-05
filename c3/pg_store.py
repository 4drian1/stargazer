"""pg_store.py — C4 Shadow 生产持久化（PostgreSQL 双表不可变，C4_SHADOW.md DDL 落地）。

语义与 shadow_store.ShadowStore 对齐，由数据库结构保证：
- c4_shadow_prediction：PRIMARY KEY(event_cluster_id, materiality_model_version)，
  仅 INSERT（ON CONFLICT DO NOTHING）；同键已有不同 p → RuntimeError（IDEMPOTENCY）。
  本模块不提供任何 UPDATE 路径（APPEND-ONLY）。
- c4_shadow_realization：PRIMARY KEY(event_cluster_id, materiality_model_version, label_horizon)，
  独立 append；必须先有 prediction 行（顺序约束）。

连接：localhost:5432 dbname=corvus（与 corvus 五表、stargazer_plan 同实例，独立 schema）。
"""
from __future__ import annotations

import json
import os
from typing import Optional

import psycopg

SCHEMA = "corvus_c4"
DSN_DEFAULT = "host=localhost port=5432 dbname=corvus user=corvus password=123456"

DDL = f"""
CREATE SCHEMA IF NOT EXISTS {SCHEMA};
CREATE TABLE IF NOT EXISTS {SCHEMA}.c4_shadow_prediction (
    event_cluster_id            TEXT    NOT NULL,
    materiality_model_version   TEXT    NOT NULL,
    prediction_ts               TIMESTAMP NOT NULL,
    p_materiality              DOUBLE PRECISION,
    code                        TEXT,
    event_family                TEXT,
    label_start                 INTEGER,
    available_at                INTEGER,
    event_tier                  TEXT,
    calibration_status          TEXT,
    missing_warnings            TEXT,
    feature_snapshot_id         TEXT,
    feature_asof                INTEGER,
    evidence_ids                TEXT,
    created_ts                  TIMESTAMP,
    PRIMARY KEY (event_cluster_id, materiality_model_version)
);
CREATE TABLE IF NOT EXISTS {SCHEMA}.c4_shadow_realization (
    event_cluster_id            TEXT    NOT NULL,
    materiality_model_version   TEXT    NOT NULL,
    label_horizon               TEXT    NOT NULL,
    realized_label              BOOLEAN NOT NULL,
    label_ts                    TIMESTAMP,
    label_version               TEXT,
    label_status                TEXT,
    created_ts                  TIMESTAMP DEFAULT now(),
    PRIMARY KEY (event_cluster_id, materiality_model_version, label_horizon)
);
"""


def connect(dsn: Optional[str] = None):
    return psycopg.connect(dsn or os.environ.get("CORVUS_PG_DSN", DSN_DEFAULT))


def ensure_schema(conn) -> None:
    conn.execute(DDL)
    conn.commit()


def upsert_prediction(conn, rec: dict) -> str:
    """prediction 幂等落库。同键已存在且 p 不同 → RuntimeError（拒绝，不改写）。"""
    key = (rec["event_cluster_id"], rec["materiality_model_version"])
    cur = conn.execute(
        f"SELECT p_materiality FROM {SCHEMA}.c4_shadow_prediction "
        "WHERE event_cluster_id=%s AND materiality_model_version=%s", key)
    row = cur.fetchone()
    if row is not None:
        old = row[0]
        new = rec.get("p_materiality")
        if old is not None and new is not None and abs(float(old) - float(new)) > 1e-12:
            raise RuntimeError(
                f"IDEMPOTENCY VIOLATION: {key} 已有 p={old}，新 p={new}；拒绝写入（prediction 永不 UPDATE）")
        return "UNCHANGED"
    conn.execute(
        f"INSERT INTO {SCHEMA}.c4_shadow_prediction "
        "(event_cluster_id, materiality_model_version, prediction_ts, p_materiality, "
        " code, event_family, label_start, available_at, event_tier, calibration_status, "
        " missing_warnings, feature_snapshot_id, feature_asof, evidence_ids, created_ts) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (rec["event_cluster_id"], rec["materiality_model_version"], rec["prediction_ts"],
         rec.get("p_materiality"), rec.get("code"), rec.get("event_family"),
         rec.get("label_start"), rec.get("available_at"), rec.get("event_tier"),
         rec.get("calibration_status"),
         json.dumps(rec.get("missing_warnings") or [], ensure_ascii=False),
         rec.get("feature_snapshot_id"), rec.get("feature_asof"),
         json.dumps(rec.get("evidence_ids") or [], ensure_ascii=False),
         rec.get("created_ts")))
    conn.commit()
    return "INSERTED"


def attach_realization(conn, event_cluster_id: str, materiality_model_version: str,
                       label_horizon: str, realized_label: bool,
                       label_version: str, label_status: str = "OK") -> str:
    """标签成熟后独立 append；无 prediction 行 → KeyError（顺序约束）。幂等。"""
    cur = conn.execute(
        f"SELECT 1 FROM {SCHEMA}.c4_shadow_prediction "
        "WHERE event_cluster_id=%s AND materiality_model_version=%s",
        (event_cluster_id, materiality_model_version))
    if cur.fetchone() is None:
        raise KeyError(f"无 prediction 行 ({event_cluster_id}, {materiality_model_version})，"
                       "不能先于打分附加 realized label（APPEND-ONLY 顺序约束）")
    cur = conn.execute(
        f"SELECT 1 FROM {SCHEMA}.c4_shadow_realization "
        "WHERE event_cluster_id=%s AND materiality_model_version=%s AND label_horizon=%s",
        (event_cluster_id, materiality_model_version, label_horizon))
    if cur.fetchone() is not None:
        return "UNCHANGED"
    conn.execute(
        f"INSERT INTO {SCHEMA}.c4_shadow_realization "
        "(event_cluster_id, materiality_model_version, label_horizon, realized_label, "
        " label_version, label_status) VALUES (%s,%s,%s,%s,%s,%s)",
        (event_cluster_id, materiality_model_version, label_horizon,
         bool(realized_label), label_version, label_status))
    conn.commit()
    return "INSERTED"


def stats(conn) -> dict:
    cur = conn.execute(f"SELECT count(*), count(DISTINCT event_cluster_id) "
                       f"FROM {SCHEMA}.c4_shadow_prediction")
    n_pred, n_ev = cur.fetchone()
    cur = conn.execute(f"SELECT count(*) FROM {SCHEMA}.c4_shadow_realization")
    n_real = cur.fetchone()[0]
    return {"predictions": n_pred, "distinct_events": n_ev, "realizations": n_real}
