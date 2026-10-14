"""Offline engineering tests only; synthetic rows never become research samples."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import forward_training_adapter as adapter
import forward_training_cli as cli


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.day = 20261012
        self.grid = [93000, 93100, 93200, 93300, 93400, 93500, 93600]
        self.daily = pd.DataFrame([{"date": 20261009, "close": 11.0}])
        self.minute = pd.DataFrame([
            dict(date=self.day * 1000000 + slot, open=10.2, high=10.4,
                 low=10.15, close=10.3, volume=100, amount=1020)
            for slot in self.grid])
        self.dip = dict(signal_ts=20261012093500, signal_type="BREAK",
                        zone_lower=10.0, zone_upper=10.5, zone_sources=["MA20"])
        self.patches = [patch.object(adapter.engine, "build_atr_ref", return_value=1.0),
                        patch.object(adapter.engine, "build_zones", return_value=[]),
                        patch.object(adapter.probe, "evaluate_calibers", return_value={"main": {"valid": False}}),
                        patch.object(adapter, "detect_signal_v02", return_value=self.dip)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def run_extract(self, minute=None, observed=20261012093600):
        return adapter.extract(self.day, self.daily, self.minute if minute is None else minute,
                               observed, self.grid)

    def test_signed_penetration_and_completed_bar(self):
        result = self.run_extract()
        self.assertEqual(result["decision_available_ts"], 20261012093600)
        self.assertAlmostEqual(result["pen_atr_bar"], -0.15)
        self.assertEqual(result["signal_close"], 10.3)

    def test_future_bar_does_not_change_fields(self):
        changed = self.minute.copy()
        changed.loc[changed.date == 20261012093600, ["open", "high", "low", "close"]] *= 2
        self.assertEqual(self.run_extract(), self.run_extract(changed))

    def test_prefix_gap_blocks_first_branch(self):
        result = self.run_extract(self.minute[self.minute.date != 20261012093200])
        self.assertEqual(result["reason"], "BLOCKS_FIRST_BRANCH")

    def test_same_bar_prefers_s1_and_captures_box(self):
        with patch.object(adapter.probe, "evaluate_calibers", return_value={
                "main": {"valid": True, "lower": 10.1, "upper": 11.0}}):
            result = self.run_extract()
        self.assertEqual(result["branch"], "BASE_HOLD")
        self.assertEqual(result["box_lower"], 10.1)

    def test_reclaim_is_not_dip_action(self):
        with patch.object(adapter, "detect_signal_v02", return_value={**self.dip, "signal_type": "RECLAIMED_AT_S"}):
            self.assertEqual(self.run_extract()["status"], "OBSERVING")

    def test_duplicate_rejected(self):
        result = self.run_extract(pd.concat([self.minute, self.minute.iloc[:1]]))
        self.assertEqual(result["reason"], "DUPLICATE_SLOT")

    def test_daily_future_rejected(self):
        self.daily.loc[0, "date"] = self.day
        self.assertEqual(self.run_extract()["reason"], "DAILY_PIT_VIOLATION")


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.row = dict(code="000001", E=20261012, event_id="one", status="ACTION_CERTIFIED",
                        branch="DIP_REPAIR", setup="A_SDP", contraction=.8, vol_cv10=.3,
                        pen_atr_bar=.1, d_atr=-.2, width_atr=.5,
                        signal_bar_ts=20261012093500, decision_available_ts=20261012093600,
                        input_manifest_sha256="inputs", trigger_content_hash="trigger")
        self.label = dict(event_id="one", code="000001", E=20261012, track="H3",
                          status="SIM_COMPLETE", remaining_shares=0, missing_dependencies=[],
                          dependency_check_passed=True, corporate_action_check_passed=True,
                          scale_check_passed=True, executor_sha256="executor",
                          input_manifest_sha256="inputs", trigger_content_hash="trigger",
                          C_buy=100.0, C_sell=101.0, net=.01,
                          last_fill_ts=20261015145500, label_end_ts=20261015145600)
        gate = dict(events=2, positive=1, nonpositive=1, dates=2, C_det=1, F_int=1, L3_avl=1)
        self.config = dict(inputs={}, executor_sha256="executor", branch_scope=["DIP_REPAIR", "BASE_HOLD"],
                           t0_ts=20261012000000, gates={"train": gate, "val": gate},
                           branch_gates={s: {b: {k: gate[k] for k in ("events", "positive", "nonpositive", "dates")}
                                             for b in ("DIP_REPAIR", "BASE_HOLD")} for s in ("train", "val")},
                           windows={"train": {"start": 20261012000000, "end": 20261016235959},
                                    "val": {"start": 20261019000000, "end": 20261023235959}})

    def prepare(self):
        for role, records in (("dataset", [self.row]), ("labels", [self.label]), ("population", [self.row])):
            path = self.path / (role + ".json")
            path.write_text(json.dumps({"records": records}), encoding="utf-8")
            self.config["inputs"][role] = {"path": str(path)}
        # Structural tests isolate prepare; production verify_config is never bypassed by CLI.
        with patch.object(cli, "verify_config"):
            return cli.prepare(self.config)

    def test_valid_row_does_not_pass_sample_gates(self):
        rows, report = self.prepare()
        self.assertEqual(len(rows), 1)
        self.assertFalse(report["training_ready"])

    def test_unclosed_not_negative(self):
        self.label["remaining_shares"] = 100
        self.assertEqual(self.prepare()[0], [])

    def test_no_fill_never_training_negative(self):
        self.row["status"] = "NO_FILL_CONFIRMED"
        self.assertEqual(self.prepare()[0], [])

    def test_cash_ledger_mismatch_rejected(self):
        self.label["net"] = .5
        self.assertEqual(self.prepare()[0], [])

    def test_cross_window_rejected(self):
        self.label["label_end_ts"] = 20261019093100
        rows, report = self.prepare()
        self.assertEqual(rows, [])
        self.assertEqual(report["excluded"][0]["reason"], "CROSS_WINDOW_LABEL")

    def test_decision_not_fill_or_future_bar(self):
        self.row["decision_available_ts"] = 20261012093700
        self.assertEqual(self.prepare()[0], [])

    def test_trigger_reference_mismatch_rejected(self):
        self.label["trigger_content_hash"] = "another"
        self.assertEqual(self.prepare()[0], [])

    def test_missing_execution_evidence_rejected(self):
        self.label["dependency_check_passed"] = False
        self.assertEqual(self.prepare()[0], [])

    def test_draft_config_cannot_train(self):
        with self.assertRaisesRegex(ValueError, "AUTHORIZATION_REQUIRED"):
            cli.verify_config({"status": "DRAFT", "fit_authorized": False})

    def test_one_branch_cannot_stand_in_for_other(self):
        _, report = self.prepare()
        self.assertIn("train:BASE_HOLD:events", report["gate_failures"])


if __name__ == "__main__":
    unittest.main()
