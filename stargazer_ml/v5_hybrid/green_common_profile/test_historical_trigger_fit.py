"""Offline validation tests. No sklearn fit and no real-data writes."""
import copy
import unittest
from unittest.mock import patch
import numpy as np

import historical_trigger_extension as build
import historical_trigger_fit as cli


class HistoricalTests(unittest.TestCase):
    def test_minute_carry(self):
        self.assertEqual(build.completed(20260330095900), 20260330100000)
        self.assertEqual(build.completed(20260331145900), 20260331150000)

    def test_duplicate_population_rejected(self):
        with self.assertRaisesRegex(ValueError, "DUPLICATE_EVENT"):
            build.index([dict(code="000001", E=20250101)] * 2)

    def test_unaccepted_reconstruction_rejected(self):
        r = dict(code="000001", E=20250101, branch="BASE_HOLD")
        with self.assertRaisesRegex(ValueError, "RECONSTRUCTION_NOT_ACCEPTABLE"):
            build.merge_fields(r, r, None)

    def test_incomplete_population_rejected(self):
        with self.assertRaisesRegex(ValueError, "POPULATION_CHANGED"):
            cli.validate_rows([])

    def test_cluster_event_weighting(self):
        result = cli.clustered([1., 1., -1.], [1, 1, 2])
        self.assertAlmostEqual(result["mean"], 1/3)
        self.assertEqual(result["date_clusters"], 2)
        self.assertEqual(result, cli.clustered([1., 1., -1.], [1, 1, 2]))

    def test_empty_selected_not_zero_profit(self):
        self.assertIsNone(cli.clustered([], [])["mean"])

    def test_authorize_requires_both_approvals_without_writing(self):
        with patch.object(cli, "check", return_value=([], {})), patch.object(cli, "save") as save:
            with self.assertRaisesRegex(ValueError, "EXPLICIT_USER_SIGNATURE"):
                cli.authorize(dict(status="DRAFT"), "署名", "")
            save.assert_not_called()

    def test_changed_fixed_parameters_rejected(self):
        with self.assertRaisesRegex(ValueError, "FIXED_CONFIGURATION_CHANGED"):
            cli.check(dict(fixed={**cli.FIXED, "C": 1}, scope="HISTORICAL_EXPLORATORY_124"))

    def test_cash_ledger_not_net_alias(self):
        key = dict(code="000001", E=20250101)
        buy, sell = 100 * 10 * 1.0013, 100 * 9 * .9982
        net = round(sell/buy-1, 6)
        row = dict(**key, C_buy=buy, C_sell=sell, pnl=sell-buy, net=net,
                   le=20250106093600, original_label_end_ts=20250106093600)
        t = dict(status="SIM_COMPLETE", fully_exited=True, remaining_shares=0,
                 missing_dependencies=[], sells=[dict(price=9, shares=100)], net=net)
        led = dict(**key, entry=dict(sim_fill_price=10, shares=100), H3=t, label_end_ts=row["le"])
        adm = dict(**key, track="H3", admitted=True, net=net, le=row["le"],
                   assert_fails=[], verdict="PASS_NO_EVENT")
        cli.verify_ledgers([row], [led], [adm])
        wrong = copy.deepcopy(row); wrong["pnl"] = wrong["net"] * 50000
        with self.assertRaisesRegex(ValueError, "FULL_CASH_LEDGER_MISMATCH"):
            cli.verify_ledgers([wrong], [led], [adm])
        wrong = copy.deepcopy(led); wrong["H3"]["remaining_shares"] = 1
        with self.assertRaisesRegex(ValueError, "NOT_ADMITTED"):
            cli.verify_ledgers([row], [wrong], [adm])


if __name__ == "__main__":
    unittest.main()
