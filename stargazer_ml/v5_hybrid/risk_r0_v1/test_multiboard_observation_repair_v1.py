import io
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from newhigh_common_motion_v1 import combine, sector_weights, footer_bounds
from multiboard_risk_features_v1 import matrix_features, summarize, day_high, rate


class RepairTests(unittest.TestCase):
    def test_new_high_true_reuses_parent_false_is_zero(self):
        r, coverage = combine([.3, .8], [1., .5], [.5, 1.], [1., 0.])
        np.testing.assert_allclose(r, [.3, 0.])
        np.testing.assert_allclose(coverage, [1., .5])

    def test_empty_unknown_motion_and_unknown_market_stay_different(self):
        r, coverage = combine([np.nan, np.nan, .2], [np.nan, 0., .5], [0., .2, .3], [1., 0., np.nan])
        self.assertTrue(np.isnan(r).all())
        self.assertTrue(np.isnan(coverage[0]))
        np.testing.assert_array_equal(coverage[1:], [0., 0.])

    def test_invalid_parent_and_market_states_refused(self):
        for args in [([np.inf], [1.], [.5], [1.]), ([.2], [0.], [.5], [1.]),
                     ([.2], [1.], [0.], [1.]), ([.2], [1.], [.5], [2.])]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                combine(*args)

    def test_algebra_matches_direct_board_flags_both_periods(self):
        for frequency in [5, 15]:
            n = 240 // frequency
            x = np.arange(n)
            returns = np.stack([.03 + .01*np.sin(x), .02 + .01*np.cos(x), -.01 + .015*np.sin(x*.7)])
            returns[1, 4] = np.nan
            market = 100 + .2*x + np.sin(x)
            for exclude in [None, 0]:
                board = matrix_features(returns, [[0, 1], [0, 2]], frequency, exclude)
                old = summarize(board, {'sse': market}, frequency, 'parent_', own=exclude is not None)
                high = day_high(market, np.isfinite(market), frequency)
                for flag, condition in [('nonpositive', board['motion'] <= 0), ('falling', board['motion'] < 0)]:
                    direct = np.where(np.isfinite(board['motion']) & np.isfinite(high), (high == 1) & condition, np.nan)
                    expected = rate(direct, np.isfinite(board['mean']) & (board['mean'] > 0))
                    actual = combine(old[f'parent_sse_positive_{flag}_rate'], old[f'parent_sse_positive_{flag}_coverage'],
                                     old['parent_positive_share'], high)
                    for a, b in zip(actual, expected):
                        np.testing.assert_allclose(a, b, equal_nan=True, atol=0, rtol=0)

    def test_candidate_own_price_not_reintroduced(self):
        frequency, n = 5, 48
        returns = np.stack([np.linspace(.1, .2, n), .02 + .005*np.sin(np.arange(n)),
                            .03 + .005*np.cos(np.arange(n))])
        market = 100. + np.arange(n)
        def result(x):
            b = matrix_features(x, [[0, 1, 2]], frequency, exclude_index=0)
            parent = summarize(b, {'sse': market}, frequency, 'own_', own=True)
            return combine(parent['own_sse_positive_nonpositive_rate'], parent['own_sse_positive_nonpositive_coverage'],
                           parent['own_positive_share'], day_high(market, np.isfinite(market), frequency))[0]
        old = result(returns)
        changed = returns.copy(); changed[0] *= -100
        np.testing.assert_allclose(old, result(changed), equal_nan=True, atol=0, rtol=0)

    def test_prefix_and_future_perturbation_both_periods(self):
        for frequency in [5, 15]:
            n = 240 // frequency
            x = np.arange(n, dtype=float) + 100
            high = day_high(x, np.ones(n, bool), frequency)
            point = n // 2
            for value in [np.nan, -1.e5]:
                revised = x.copy(); revised[point+1:] = value
                got = day_high(revised, np.isfinite(revised), frequency)
                self.assertTrue(np.isfinite(high[:point+1]).any())
                np.testing.assert_allclose(high[:point+1], got[:point+1], equal_nan=True)
            self.assertTrue(np.isnan(high[0]))

    def test_coverage_weight_exact_reconstruction_and_partition(self):
        members = [['A', 'B'], ['B', 'C', 'D']]
        weights = sector_weights(list('ABCD'), members)
        valid = np.array([[1, 0], [1, 1], [0, 1], [0, 1]], bool)
        direct = (valid[:2].mean(0) + valid[1:].mean(0)) / 2
        np.testing.assert_allclose(np.einsum('i,ij->j', weights, valid), direct, atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.einsum('i,ij->j', weights, valid) +
                                   np.einsum('i,ij->j', weights, ~valid), 1., atol=1e-12, rtol=0)

    def test_empty_and_duplicate_catalog_members_refused(self):
        for memberships in [[], [[]], [['A', 'A']]]:
            with self.assertRaises(ValueError):
                sector_weights(['A'], memberships)

    def test_footer_all_groups_and_missing_statistics(self):
        for statistics in [True, False]:
            sink = io.BytesIO()
            pq.write_table(pa.Table.from_pandas(pd.DataFrame({'date': [3, 4, 1, 2]}), preserve_index=False),
                           sink, row_group_size=2, write_statistics=statistics)
            result = footer_bounds(pq.ParquetFile(pa.BufferReader(sink.getvalue())))
            self.assertEqual(result, (1, 4) if statistics else (None, None))


if __name__ == '__main__':
    unittest.main()
