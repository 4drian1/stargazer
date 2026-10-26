import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import build_common_candidate_sidecar_v1 as build
import common_candidate_aggregation_v1 as aggregation
import continuous_motion_common_v1 as common
from continuous_motion_matrix_v1 import path_arrays
from intraday_relative_stall_v1 import TIME, slots
import research_budget_common_v3 as budget


class SidecarTests(unittest.TestCase):
    def setup_case(self, frequency):
        width = 240 // frequency
        r = np.vstack([np.linspace(.1, .04, width), np.linspace(.01, .05, width),
                       np.linspace(-.02, -.01, width)])
        r[2, 5] = np.nan
        clock = slots(20250103, frequency)
        markets = {'sse': np.linspace(100, 101, width), 'cyb': np.linspace(200, 199, width)}
        pop = pd.DataFrame({'code': ['A', 'B'], TIME: [clock[6], clock[7]], 'segment': ['train']*2})
        return pop, ['A', 'B', 'C'], [[0, 1, 2], [0, 1]], r, markets, clock

    def test_shared_and_own_equal_scalar_reference(self):
        for frequency in (5, 15):
            pop, codes, memberships, r, markets, clock = self.setup_case(frequency)
            actual, audit, checked = build.month_features(
                pop, codes, memberships, r, markets, clock, frequency, check_real_prefix=True)
            reference = common.context_common(r, memberships, markets, frequency)
            self.assertEqual(len(actual), len(pop))
            self.assertEqual(len(checked), 1)
            self.assertEqual(audit['frequency'].tolist(), [frequency]*2)
            stocks = path_arrays(1+r, np.isfinite(r), frequency)
            for i, row in pop.iterrows():
                at = clock.index(int(row[TIME]))
                for scope in ('all', 'positive'):
                    for metric in common.METRICS_COMMON:
                        np.testing.assert_allclose(actual[f'ccm_v1_{frequency}_{scope}_{metric}'].iloc[i],
                            reference[f'cm_common{frequency}_{scope}_{metric}_mean'][at],
                            atol=1e-12, rtol=0, equal_nan=True)
                own = common.context_common(r, memberships, markets, frequency,
                                            exclude=codes.index(row.code), stocks=stocks)
                for metric in aggregation.OWN_METRICS:
                    prefix = '' if metric.startswith(('current_', 'captured_')) else 'all_'
                    np.testing.assert_allclose(actual[f'ccm_v1_{frequency}_own_{metric}'].iloc[i],
                        own[f'cm_common{frequency}_{prefix}{metric}_mean'][at],
                        atol=1e-12, rtol=0, equal_nan=True)

    def test_full_shared_and_own_future_perturbation_and_truncation(self):
        for frequency in (5, 15):
            pop, codes, members, r, markets, clock = self.setup_case(frequency)
            baseline = build.month_features(pop, codes, members, r, markets, clock, frequency)[0]
            for future in (.5, np.nan):
                changed = r.copy(); changed[:, 8:] = future
                changed_market = {k: v.copy() for k, v in markets.items()}
                for values in changed_market.values():
                    values[8:] = 1000 if np.isfinite(future) else np.nan
                actual = build.month_features(pop, codes, members, changed, changed_market, clock, frequency)[0]
                pd.testing.assert_frame_equal(baseline, actual)
            self.assertTrue(baseline.filter(like='own_deceleration').notna().any().any())

    def test_clock_and_membership_missing_fail(self):
        pop, codes, members, r, markets, clock = self.setup_case(5)
        bad = pop.copy(); bad.loc[0, TIME] = 20250103093100
        with self.assertRaisesRegex(ValueError, 'CLOCK_MISS'):
            build.month_features(bad, codes, members, r, markets, clock, 5)
        with self.assertRaisesRegex(ValueError, 'NO_FROZEN_BOARD'):
            build.month_features(pop, codes, [[1, 2]], r, markets, clock, 5)

    def test_bounded_storage_and_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(budget, 'ROOTS', [root]), patch.object(budget, 'bytes_used', return_value=0):
                out = root / 'run'
                writer = budget.EvidenceWriter(out, cap=1024, reserve=128)
                writer.write('old.bin', b'old')
                with self.assertRaisesRegex(ValueError, 'BUDGET_STOP'):
                    writer.write('too_big.bin', b'x'*1024)
                self.assertFalse((out / 'too_big.bin').exists())
                with self.assertRaises(FileExistsError):
                    budget.EvidenceWriter(out, cap=1024, reserve=128)
                self.assertEqual((out / 'old.bin').read_bytes(), b'old')
                with self.assertRaisesRegex(ValueError, 'OUTSIDE_REGISTERED'):
                    budget.EvidenceWriter(root.parent / 'outside', cap=1024, reserve=128)


if __name__ == '__main__':
    unittest.main()
