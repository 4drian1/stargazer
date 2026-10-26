import unittest
import numpy as np
import pandas as pd
from repair_postclose_daily_context_v1 import closing_panel, daily_returns, board_panel, excluded_board, unique_leaders
from prepare_postclose_research_v2 import complete_daily_close


class DailyContextTests(unittest.TestCase):
    def test_interior_gap_does_not_destroy_daily_close(self):
        frame = pd.DataFrame(dict(code=['000001'] * 31,
            cutoff_index=[i for i in range(32) if i != 7], stock_close=[10.] * 31))
        self.assertTrue(np.isnan(complete_daily_close(frame, 2)['000001'][0]))
        np.testing.assert_equal(closing_panel(frame, 2)['000001'], [10., 10.])

    def test_missing_close_not_forward_filled(self):
        np.testing.assert_equal(daily_returns(np.array([[10., np.nan, 12., 13.]])),
                                [[np.nan, np.nan, np.nan, 13/12-1]])

    def test_nonclosing_rows_never_substitute(self):
        frame = pd.DataFrame(dict(code=['000001'], cutoff_index=[14], stock_close=[10.]))
        self.assertNotIn('000001', closing_panel(frame, 1))

    def test_invalid_price_and_duplicate(self):
        f = pd.DataFrame(dict(code=['000001'], cutoff_index=[15], stock_close=[np.inf]))
        self.assertTrue(np.isnan(closing_panel(f, 1)['000001'][0]))
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            closing_panel(pd.concat([f, f]), 1)

    def test_exclusion_single_member_stays_null(self):
        means, counts, ups = board_panel(np.array([[.1]]), [np.array([0])])
        values, breadth, n = excluded_board(means, counts, ups, np.array([[.1]]))
        self.assertTrue(np.isnan(values).all() and np.isnan(breadth).all())
        self.assertEqual(n[0, 0], 0)

    def test_missing_members_not_zero_contribution(self):
        means, counts, ups = board_panel(np.array([[.1], [.2], [np.nan]]), [np.array([0,1,2])])
        values, breadth, n = excluded_board(means, counts, ups, np.array([[.1]]))
        self.assertAlmostEqual(values[0,0], .2)
        self.assertEqual(breadth[0,0], 1.)
        self.assertEqual(n[0,0], 1)

    def test_leaders_use_return_not_volume_and_ties_unknown(self):
        returns = np.array([[.1, .1], [.2, .1]])
        b, s = unique_leaders(returns, np.array([[.15, .1]]), [np.array([0,1])], ['000001','000002'])
        np.testing.assert_equal(b, [0,0]); np.testing.assert_equal(s, [1,-1])

    def test_prefix_and_future_perturbation(self):
        closes = np.array([[10.,11.,12.,13.], [20.,18.,19.,20.]])
        full = daily_returns(closes)
        prefix = daily_returns(closes[:, :3])
        self.assertTrue(np.isfinite(prefix).any())
        np.testing.assert_equal(full[:, :3], prefix)
        changed = closes.copy(); changed[:, 3] *= 100
        np.testing.assert_equal(daily_returns(changed)[:, :3], prefix)


if __name__ == '__main__':
    unittest.main()
