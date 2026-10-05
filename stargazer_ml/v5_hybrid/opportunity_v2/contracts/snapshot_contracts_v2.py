# -*- coding: utf-8 -*-
"""Versioned snapshot boundary for a future shadow runtime.

This module describes inputs. It does not select trades or run a workflow.
The existing snapshot_contracts.py v1 remains unchanged.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any


KINDS = frozenset({"opportunity", "entry_state", "risk", "context", "position_state"})
QUALITY_STATES = frozenset({"VALID", "DEGRADED", "UNKNOWN", "INVALID"})


def _instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("v2 timestamps require an explicit UTC offset")
    return parsed


@dataclass(frozen=True)
class SnapshotHeaderV2:
    snapshot_id: str
    schema_version: str
    frozen_artifact_ref: str
    symbol: str
    asof_ts: str
    available_at: str
    created_ts: str
    expires_at: str
    quality_state: str
    data_scope: str
    missing_reason: str | None = None
    benchmark_scope: str | None = None
    evidence_ids: tuple[str, ...] = ()
    input_refs: tuple[str, ...] = ()
    replay_mode: str = "LIVE"

    def __post_init__(self) -> None:
        if not self.snapshot_id or not self.symbol or not self.data_scope:
            raise ValueError("snapshot_id, symbol and data_scope are required")
        if not self.schema_version.endswith(".v2"):
            raise ValueError("v2 snapshot requires a .v2 schema version")
        ref_parts = self.frozen_artifact_ref.split(":")
        if len(ref_parts) != 3 or not all(part.strip() for part in ref_parts):
            raise ValueError("frozen_artifact_ref requires type:version:hash")
        for name in ("evidence_ids", "input_refs"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"{name} requires a list or tuple of nonempty strings")
            object.__setattr__(self, name, tuple(values))
        if self.quality_state not in QUALITY_STATES:
            raise ValueError("unknown quality_state")
        if self.quality_state != "VALID" and not self.missing_reason:
            raise ValueError("non-valid snapshots require missing_reason")
        if self.replay_mode not in {"LIVE", "REPLAY"}:
            raise ValueError("unknown replay_mode")
        if _instant(self.asof_ts) > _instant(self.available_at):
            raise ValueError("available_at precedes asof_ts")
        if _instant(self.expires_at) <= _instant(self.available_at):
            raise ValueError("expires_at must follow available_at")
        _instant(self.created_ts)


@dataclass(frozen=True)
class PredictionMetadataV2:
    target_definition: str
    prediction_horizon: str
    label_version: str
    exit_contract_version: str
    cost_contract_version: str
    execution_contract_version: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip() for value in asdict(self).values()):
            raise ValueError("prediction metadata fields must be nonempty")


@dataclass(frozen=True)
class SnapshotV2:
    kind: str
    header: SnapshotHeaderV2
    payload: dict[str, Any] = field(default_factory=dict)
    prediction: PredictionMetadataV2 | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError("unknown snapshot kind")
        if self.header.schema_version != f"{self.kind}.v2":
            raise ValueError("kind and schema_version disagree")
        json.dumps(asdict(self), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class SnapshotRequirementV2:
    """A caller's frozen input requirements; not inferred from a received snapshot.

    benchmark_scope=None requires an unbenchmarked input. prediction=None adds
    no prediction comparison, allowing context/state inputs without labels.
    """
    kind: str
    symbol: str
    data_scope: str
    benchmark_scope: str | None = None
    prediction: PredictionMetadataV2 | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS or not self.symbol or not self.data_scope:
            raise ValueError("requirement needs a known kind, symbol and data_scope")


def unavailable_snapshot(
    kind: str, header: SnapshotHeaderV2, reason: str = "MODEL_NOT_READY",
) -> SnapshotV2:
    """Return UNKNOWN without fabricated scores. Header ref identifies the adapter."""
    return SnapshotV2(
        kind=kind,
        header=replace(header, quality_state="UNKNOWN", missing_reason=reason),
    )


def assess_for_shadow(
    snapshot: SnapshotV2, decision_ts: str, *,
    requirement: SnapshotRequirementV2 | None = None,
) -> tuple[bool, str]:
    """Check one input. READY is input readiness, not permission to trade.

    Omit requirement only for the legacy single-snapshot readiness check.
    Model adapters should pass their explicit input requirement.
    """
    decision_at = _instant(decision_ts)
    header = snapshot.header
    if header.replay_mode != "LIVE":
        return False, "REPLAY_ONLY"
    if decision_at < _instant(header.available_at):
        return False, "NOT_YET_AVAILABLE"
    if decision_at < _instant(header.created_ts):
        return False, "OUTPUT_NOT_READY"
    if decision_at >= _instant(header.expires_at):
        return False, "EXPIRED"
    if header.quality_state != "VALID":
        return False, f"QUALITY_{header.quality_state}"
    if requirement is not None:
        for actual, expected, reason in (
            (snapshot.kind, requirement.kind, "KIND_MISMATCH"),
            (header.symbol, requirement.symbol, "SYMBOL_MISMATCH"),
            (header.data_scope, requirement.data_scope, "DATA_SCOPE_MISMATCH"),
            (header.benchmark_scope, requirement.benchmark_scope, "BENCHMARK_SCOPE_MISMATCH"),
        ):
            if actual != expected:
                return False, reason
        if requirement.prediction is not None and snapshot.prediction != requirement.prediction:
            return False, "PREDICTION_CONTRACT_MISMATCH"
    return True, "READY"


def snapshot_to_json(snapshot: SnapshotV2) -> str:
    return json.dumps(asdict(snapshot), ensure_ascii=False, sort_keys=True, allow_nan=False)


def snapshot_from_json(raw: str) -> SnapshotV2:
    data = json.loads(raw)
    data["header"] = SnapshotHeaderV2(**data["header"])
    if data.get("prediction") is not None:
        data["prediction"] = PredictionMetadataV2(**data["prediction"])
    return SnapshotV2(**data)
