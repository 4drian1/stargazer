import unittest

import numpy as np

from multiboard_risk_features_v1 import board_arrays, matrix_features, rate, summarize


class ThreeGroupTests(unittest.TestCase):
    def test_denominator_known_only_and_coverage_separate(self):
        flag = np.array([[1., np.nan], [np.nan, np.nan], [0., 1.]])
        selected = np.array([[True, False], [True, False], [True, False]])
        value, coverage = rate(flag, selected)
        self.assertEqual(value[0], .5)
        self.assertAlmostEqual(coverage[0], 2 / 3)
        self.assertTrue(np.isnan(value[1]) and np.isnan(coverage[1]))

    def test_unknown_not_false_all_missing(self):
        value, coverage = rate(np.full((2, 2), np.nan), np.ones((2, 2), bool))
        self.assertTrue(np.isnan(value).all())
        np.testing.assert_equal(coverage, 0)

    def test_exclude_self_from_basket_and_not_promote_runner_up(self):
        values = np.full((2, 48), np.nan)
        values[:, :3] = [[.1, .05, .01], [.04, .08, .02]]
        result = board_arrays(values, [0, 1], 5, exclude_index=0)
        np.testing.assert_allclose(result['mean'][:3], [.04, .08, .02])
        self.assertTrue(np.isnan(result['current_change'][0]))
        self.assertEqual(result['current_count'][0], 0)
        self.assertAlmostEqual(result['captured_change'][2], -.06)

    def test_self_only_reference_empty_not_global_fallback(self):
        result = board_arrays(np.full((1, 48), .1), [0], 5, exclude_index=0)
        for name in ('mean', 'breadth', 'motion', 'coverage', 'current_change', 'captured_change', 'high'):
            self.assertTrue(np.isnan(result[name]).all(), name)

    def test_same_count_different_members_not_common_motion(self):
        values = np.full((2, 48), np.nan)
        values[0, 0], values[1, 1] = .1, .2
        result = board_arrays(values, [0, 1], 5)
        self.assertEqual(result['common_count'][1], 0)
        self.assertTrue(np.isnan(result['motion'][1]))
        self.assertTrue(np.isnan(result['high']).all())

    def test_both_periods_prefix_and_future_changes(self):
        for frequency in (5, 15):
            width = 240 // frequency
            values = np.vstack([np.linspace(.01, .1, width), np.linspace(.02, -.03, width)])
            market = np.linspace(100, 110, width)
            complete = summarize(matrix_features(values, [[0], [0, 1]], frequency),
                                 {'sse': market}, frequency, 'x_')
            changed = values.copy()
            changed[:, 5:] *= -10
            altered_market = market.copy(); altered_market[5:] *= 2
            altered = summarize(matrix_features(changed, [[0], [0, 1]], frequency),
                                {'sse': altered_market}, frequency, 'x_')
            truncated = values.copy(); truncated[:, 5:] = np.nan
            prefix = summarize(matrix_features(truncated, [[0], [0, 1]], frequency),
                               {'sse': market}, frequency, 'x_')
            for name in complete:
                np.testing.assert_allclose(complete[name][:5], altered[name][:5], rtol=0, atol=0, equal_nan=True)
                np.testing.assert_allclose(complete[name][:5], prefix[name][:5], rtol=0, atol=0, equal_nan=True)

    def test_positive_group_is_current_not_future_close(self):
        values = np.vstack([np.full(48, .1), np.full(48, -.1)])
        first = summarize(matrix_features(values, [[0], [1]], 5), {'sse': np.arange(48) + 100.}, 5, 'x_')
        values[1, -1] = .2
        second = summarize(matrix_features(values, [[0], [1]], 5), {'sse': np.arange(48) + 100.}, 5, 'x_')
        self.assertEqual(first['x_positive_share'][0], .5)
        self.assertEqual(second['x_positive_share'][0], .5)
        self.assertEqual(second['x_positive_share'][-1], 1.)

    def test_no_positive_boards_rate_null_not_zero(self):
        result = summarize(matrix_features(np.full((2, 48), -.1), [[0], [1]], 5),
                           {'sse': np.arange(48) + 100.}, 5, 'x_')
        self.assertTrue(np.isnan(result['x_sse_positive_nonpositive_rate']).all())
        self.assertTrue(np.isnan(result['x_sse_positive_nonpositive_coverage']).all())


if __name__ == '__main__':
    unittest.main()
