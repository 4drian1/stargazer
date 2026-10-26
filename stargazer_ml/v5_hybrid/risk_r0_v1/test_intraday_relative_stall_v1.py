"""Offline examples and adversarial prefix tests; no HTTP or model fitting."""
import unittest

import pandas as pd

from intraday_relative_stall_v1 import TIME, board_evidence, close_path, leader_watch, slots


def market(frequency=15, values=(100, 101, 102, 103)):
    return pd.DataFrame({TIME: slots(20250103, frequency)[:len(values)],
                         'level': [float(x) for x in values], 'day_return': [x / 100 - 1 for x in values],
                         'path_ok': True})


def boards(frequency=15):
    ts = slots(20250103, frequency)[:4]
    return pd.DataFrame([dict(board_code=code, **{TIME: t}, day_return=r,
                              common_change=c, breadth=b, full_members=True)
                         for code in ('A', 'B')
                         for t, r, c, b in zip(ts, [.05, .06, .055, .055],
                                               [None, .01, -.005, 0], [.8, .9, .7, .7])])


class RelativeStallTests(unittest.TestCase):
    def test_both_frequencies_and_grid(self):
        for frequency, count in ((5, 48), (15, 16)):
            self.assertEqual(len(slots(20250103, frequency)), count)
            got = board_evidence(market(frequency), boards(frequency), frequency)
            self.assertEqual(set(got.frequency_minutes), {frequency})
            a = got[got.board_code == 'A'].reset_index(drop=True)
            self.assertTrue(a.loc[2, 'market_new_high_sector_not_new_high'])
            self.assertTrue(a.loc[2, 'market_rising_sector_falling'])
            self.assertTrue(a.loc[3, 'market_rising_sector_nonpositive'])
            self.assertFalse(a.loc[3, 'market_rising_sector_falling'])

    def test_positive_day_gain_can_stall(self):
        got = board_evidence(market(), boards(), 15)
        row = got[(got.board_code == 'A')].iloc[2]
        self.assertGreater(row.day_return, 0)
        self.assertTrue(row.market_rising_sector_falling)

    def test_ties_and_all_boards_preserved(self):
        got = board_evidence(market(), boards(), 15)
        self.assertEqual(len(got), 8)
        self.assertEqual(set(got.observed_day_gain_rank), {1.})

    def test_partial_not_new_high_but_common_motion_known(self):
        b = boards()
        b['full_members'] = False
        got = board_evidence(market(), b, 15)
        self.assertTrue(got.sector_close_new_day_high.isna().all())
        self.assertTrue(got.market_new_high_sector_not_new_high.isna().all())
        self.assertTrue(got.sector_breadth_change_full.isna().all())
        self.assertEqual(int(got.market_rising_sector_falling.sum()), 2)

    def test_missing_sector_slot_not_cross_filled(self):
        b = boards().drop(index=1)
        got = board_evidence(market(), b, 15)
        a = got[got.board_code == 'A']
        self.assertTrue(pd.isna(a.iloc[1].sector_day_return_change_common))
        self.assertTrue(pd.isna(a.iloc[-1].sector_close_new_day_high))
        self.assertTrue(got[got.board_code == 'B'].iloc[2].market_rising_sector_falling)

    def test_missing_market_cutoff_not_asof_filled(self):
        got = board_evidence(market().drop(index=2), boards(), 15)
        rows = got[got[TIME] == slots(20250103, 15)[2]]
        self.assertTrue(rows.market_rising_sector_falling.isna().all())

    def test_missing_and_invalid_path_do_not_heal_high_prefix(self):
        for bad in (None, float('inf'), -1):
            m = market()
            m.loc[1, 'level'] = bad
            result = close_path(m, 15)
            self.assertTrue(pd.isna(result.iloc[2].interval_return))
            self.assertTrue(pd.isna(result.iloc[3].close_new_day_high))

    def test_lunch_is_trading_adjacency(self):
        ts = slots(20250103, 15)
        m = pd.DataFrame({TIME: ts, 'level': range(100, 116), 'path_ok': True})
        got = close_path(m, 15)
        self.assertAlmostEqual(got.iloc[8].interval_return, 108 / 107 - 1)
        self.assertEqual(ts[7] % 1000000, 113000)
        self.assertEqual(ts[8] % 1000000, 131500)

    def test_next_day_reset_not_overnight_interval(self):
        m = pd.concat([market(), market().assign(**{TIME: lambda f: f[TIME] + 3000000})])
        got = close_path(m, 15)
        self.assertTrue(pd.isna(got.iloc[4].interval_return))
        self.assertTrue(pd.isna(got.iloc[4].close_new_day_high))

    def test_duplicates_and_offgrid_rejected(self):
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_CUTOFF'):
            close_path(pd.concat([market(), market().iloc[:1]]), 15)
        m = market()
        m.loc[0, TIME] += 100
        with self.assertRaisesRegex(ValueError, 'OFF_GRID'):
            close_path(m, 15)
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_BOARD'):
            board_evidence(market(), pd.concat([boards(), boards().iloc[:1]]), 15)

    def test_no_future_prefix_and_perturbation(self):
        for frequency in (5, 15):
            full = board_evidence(market(frequency), boards(frequency), frequency)
            cutoff = slots(20250103, frequency)[2]
            expected = full[full[TIME] <= cutoff].reset_index(drop=True)
            m, b = market(frequency), boards(frequency)
            prefix = board_evidence(m[m[TIME] <= cutoff], b[b[TIME] <= cutoff], frequency)
            pd.testing.assert_frame_equal(expected, prefix.reset_index(drop=True))
            m.loc[m[TIME] > cutoff, 'level'] *= 10
            b.loc[b[TIME] > cutoff, ['day_return', 'common_change']] = 100
            changed = board_evidence(m, b, frequency)
            pd.testing.assert_frame_equal(expected, changed[changed[TIME] <= cutoff].reset_index(drop=True))
            self.assertGreater(int(expected.market_rising_sector_falling.sum()), 0)

    def test_leaders_multi_board_capture_ties_missing(self):
        ts = slots(20250103, 15)[:3]
        stocks = pd.DataFrame([{'code': c, TIME: t, 'day_return': r}
                              for c, values in [('X', [.10, .04, None]), ('Y', [.05, .06, .06])]
                              for t, r in zip(ts, values)])
        b = boards()
        b = b[b[TIME].isin(ts)]
        got = leader_watch(b, {'A': ['X', 'Y'], 'B': ['Y']}, stocks)
        last = got[(got.board_code == 'A') & (got[TIME] == ts[-1])].set_index('code')
        self.assertEqual(set(last.index), {'X', 'Y'})
        self.assertTrue(pd.isna(last.loc['X', 'day_return']))
        self.assertFalse(last.loc['X', 'current_observed_leader'])
        self.assertFalse(last.loc['Y', 'strict_member_rank'])
        self.assertEqual(last.loc['X', 'capture_ts_within_day'], ts[0])
        self.assertTrue((got[got.board_code == 'B'].code == 'Y').all())
        tied = stocks.copy()
        tied.loc[tied[TIME] == ts[0], 'day_return'] = .1
        tie = leader_watch(b.iloc[:1], {'A': ['X', 'Y']}, tied)
        self.assertEqual(set(tie.code), {'X', 'Y'})
        self.assertEqual(set(tie.current_tie_count), {2})

    def test_leader_prefix_and_duplicates(self):
        ts = slots(20250103, 15)[:4]
        stocks = pd.DataFrame({'code': 'X', TIME: ts, 'day_return': [.1, .05, .03, .01]})
        full = leader_watch(boards(), {'A': ['X'], 'B': ['X']}, stocks)
        prefix = leader_watch(boards()[boards()[TIME] <= ts[1]],
                              {'A': ['X'], 'B': ['X']}, stocks[stocks[TIME] <= ts[1]])
        pd.testing.assert_frame_equal(full[full[TIME] <= ts[1]].reset_index(drop=True), prefix)
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_STOCK'):
            leader_watch(boards(), {}, pd.concat([stocks, stocks.iloc[:1]]))


if __name__ == '__main__':
    unittest.main()
