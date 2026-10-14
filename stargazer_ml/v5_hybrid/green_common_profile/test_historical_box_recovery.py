"""Synthetic provenance counterexamples, never training samples."""
import unittest
from unittest.mock import patch
import pandas as pd
import audit_historical_box_recovery as audit


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(code="000001", E=20261012, segment="train", atr_ref=1.0,
                        signal_bar_ts=20261012093600, decision_available_ts=20261012093700,
                        signal_close=10.2)
        self.daily = pd.DataFrame([dict(date=20261009, close=10.5)])
        self.minute = pd.DataFrame([
            dict(date=20261012093500, open=10.5, high=10.7, low=10.4, close=10.5, volume=100, amount=1050),
            dict(date=20261012093600, open=10.1, high=10.3, low=10.0, close=10.2, volume=100, amount=1020),
            dict(date=20261012093700, open=10.4, high=10.7, low=10.3, close=10.5, volume=100, amount=1050)])
        for p in (patch.object(audit.engine, "build_atr_ref", return_value=1.0),
                  patch.object(audit.probe, "evaluate_calibers", return_value={
                      "main": {"valid": True, "lower": 10.0, "upper": 11.0}})):
            p.start()
            self.addCleanup(p.stop)

    def check(self):
        return audit.reconstruct(self.daily, self.minute, self.row, 20261009, [93500, 93600, 93700])

    def test_reconstructed_is_not_training_release(self):
        result = self.check()
        self.assertEqual(result["status"], "RECONSTRUCTION_CANDIDATE")
        self.assertFalse(result["training_eligible"])
        self.assertEqual(result["old_region_comparison"], "NO_BASELINE")

    def test_earlier_s1_prevents_wrong_binding(self):
        self.minute.loc[0, "low"] = 10.0
        self.assertEqual(self.check()["reason"], "FIRST_S1_FROZEN_TIMESTAMP_MISMATCH")

    def test_missing_pre_trigger_slot_is_not_restored(self):
        self.minute = self.minute.iloc[1:]
        self.assertEqual(self.check()["reason"], "FIRST_DECISION_PREFIX_GAP")

    def test_wrong_close_not_accepted_by_timestamp_match(self):
        self.row["signal_close"] = 10.3
        self.assertEqual(self.check()["reason"], "SIGNAL_CLOSE_FROZEN_VALUE_MISMATCH")

    def test_fill_timestamp_not_decision(self):
        self.row["decision_available_ts"] = 20261012093800
        self.assertEqual(self.check()["reason"], "DECISION_TIMESTAMP_MISMATCH")


if __name__ == "__main__":
    unittest.main()
