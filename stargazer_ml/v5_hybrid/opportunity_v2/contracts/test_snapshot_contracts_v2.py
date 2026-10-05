import unittest
from dataclasses import replace

from snapshot_contracts_v2 import (
    PredictionMetadataV2, SnapshotHeaderV2, SnapshotRequirementV2, SnapshotV2,
    assess_for_shadow, snapshot_from_json, snapshot_to_json, unavailable_snapshot,
)


def sample_snapshot() -> SnapshotV2:
    header = SnapshotHeaderV2(
        snapshot_id="ctx-1", schema_version="context.v2",
        frozen_artifact_ref="corvus:events_v1:abc123", symbol="600000",
        asof_ts="2026-09-26T09:30:00+08:00",
        available_at="2026-09-26T09:31:00+08:00",
        created_ts="2026-09-26T09:32:00+08:00",
        expires_at="2026-09-26T09:40:00+08:00",
        quality_state="VALID", data_scope="A_SHARE_RESEARCH_V1",
    )
    return SnapshotV2(kind="context", header=header, payload={"event_count": 1})


class SnapshotContractV2Tests(unittest.TestCase):
    def test_shadow_time_boundaries(self):
        snap = sample_snapshot()
        for moment, reason in (
            ("09:30:59", "NOT_YET_AVAILABLE"),
            ("09:31:30", "OUTPUT_NOT_READY"),
            ("09:32:00", "READY"),
            ("09:40:00", "EXPIRED"),
        ):
            usable, actual = assess_for_shadow(snap, f"2026-09-26T{moment}+08:00")
            self.assertEqual((usable, actual), (reason == "READY", reason))

    def test_replay_and_missing_data_cannot_be_consumed_live(self):
        snap = sample_snapshot()
        moment = "2026-09-26T09:35:00+08:00"
        replay = replace(snap, header=replace(snap.header, replay_mode="REPLAY"))
        self.assertEqual(assess_for_shadow(replay, moment), (False, "REPLAY_ONLY"))
        missing = replace(snap, header=replace(
            snap.header, quality_state="UNKNOWN", missing_reason="SOURCE_GAP"))
        self.assertEqual(assess_for_shadow(missing, moment), (False, "QUALITY_UNKNOWN"))

    def test_invalid_time_or_quality_contract_is_rejected(self):
        snap = sample_snapshot()
        with self.assertRaises(ValueError):
            replace(snap.header, available_at="2026-09-26T09:29:00+08:00")
        with self.assertRaises(ValueError):
            replace(snap.header, quality_state="UNKNOWN")
        with self.assertRaises(ValueError):
            replace(snap.header, asof_ts="2026-09-26T09:30:00")

    def test_roundtrip_preserves_readiness(self):
        snap = sample_snapshot()
        snap = replace(snap, header=replace(
            snap.header, evidence_ids=("news:42",), input_refs=("minute:v1:abc",)))
        restored = snapshot_from_json(snapshot_to_json(snap))
        self.assertEqual(restored, snap)
        self.assertEqual(snapshot_to_json(restored), snapshot_to_json(snap))
        self.assertEqual(assess_for_shadow(restored, "2026-09-26T09:35:00+08:00"),
                         (True, "READY"))

    def test_empty_artifact_segments_and_bad_references_are_rejected(self):
        for ref in ("::", "model::abc", "model:v1:  ", "model:v1:abc:extra"):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                replace(sample_snapshot().header, frozen_artifact_ref=ref)
        for refs in ("news:42", [None], [""]):
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                replace(sample_snapshot().header, evidence_ids=refs)

    def test_untrained_or_failed_provider_returns_unknown_without_scores(self):
        for reason in ("MODEL_NOT_READY", "SOURCE_TIMEOUT", "ADAPTER_ERROR"):
            snap = unavailable_snapshot("context", sample_snapshot().header, reason)
            self.assertEqual(snap.payload, {})
            self.assertIsNone(snap.prediction)
            self.assertEqual(snap.header.missing_reason, reason)
            self.assertEqual(assess_for_shadow(snap, "2026-09-26T09:35:00+08:00"),
                             (False, "QUALITY_UNKNOWN"))

    def test_input_requirements_reject_wrong_stock_scope_and_kind(self):
        snap = sample_snapshot()
        expected = SnapshotRequirementV2("context", "600000", "A_SHARE_RESEARCH_V1")
        moment = "2026-09-26T09:35:00+08:00"
        self.assertEqual(assess_for_shadow(snap, moment, requirement=expected), (True, "READY"))
        for changed, reason in (
            ({"kind": "risk"}, "KIND_MISMATCH"),
            ({"symbol": "600001"}, "SYMBOL_MISMATCH"),
            ({"data_scope": "FULL_MARKET"}, "DATA_SCOPE_MISMATCH"),
            ({"benchmark_scope": "CSI300"}, "BENCHMARK_SCOPE_MISMATCH"),
        ):
            with self.subTest(reason=reason):
                self.assertEqual(assess_for_shadow(
                    snap, moment, requirement=replace(expected, **changed)), (False, reason))

    def test_prediction_versions_and_horizons_are_not_interchangeable(self):
        meta = PredictionMetadataV2("loss_gt_5pct", "5_trading_days", "labels_v1",
                                    "exit_v1", "cost_v1", "execution_v1")
        snap = replace(sample_snapshot(), kind="risk", prediction=meta,
                       header=replace(sample_snapshot().header, schema_version="risk.v2"))
        expected = SnapshotRequirementV2("risk", "600000", "A_SHARE_RESEARCH_V1", prediction=meta)
        moment = "2026-09-26T09:35:00+08:00"
        self.assertEqual(assess_for_shadow(snap, moment, requirement=expected), (True, "READY"))
        self.assertEqual(snapshot_from_json(snapshot_to_json(snap)), snap)
        for field in meta.__dataclass_fields__:
            with self.subTest(field=field):
                different = replace(expected, prediction=replace(meta, **{field: "other"}))
                self.assertEqual(assess_for_shadow(snap, moment, requirement=different),
                                 (False, "PREDICTION_CONTRACT_MISMATCH"))
        self.assertEqual(assess_for_shadow(replace(snap, prediction=None), moment,
                                          requirement=expected),
                         (False, "PREDICTION_CONTRACT_MISMATCH"))


if __name__ == "__main__":
    unittest.main()
