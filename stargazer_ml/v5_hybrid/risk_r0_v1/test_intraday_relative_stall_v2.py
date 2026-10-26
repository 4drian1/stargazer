import unittest

import numpy as np
import pandas as pd

from intraday_relative_stall_v1 import TIME, board_evidence as old_evidence, close_path, slots
from intraday_relative_stall_v2 import basket, board_evidence, minute_returns, paths
from test_intraday_relative_stall_v1 import boards, market


def bars(code='X', day=20250103):
    return pd.DataFrame([dict(code=code, date=t, open=100., high=102., low=99., close=101., volume=1., amount=100.)
                         for t in [day * 1000000 + m // 60 * 10000 + m % 60 * 100
                                   for m in list(range(571, 691)) + list(range(781, 901))]])


class BatchObservationTests(unittest.TestCase):
    def test_v1_v2_exact_values_both_periods(self):
        for step in (5, 15):
            for partial, gap in ((False, False), (True, False), (False, True)):
                m, b = market(step), boards(step)
                b['full_members'] = not partial
                if gap:
                    m, b = m.drop(index=1), b.drop(index=2)
                expected = old_evidence(m, b, step).sort_index(axis=1)
                actual = board_evidence(m, b, step).sort_index(axis=1)
                # Only NULL representation may differ (object None / float NaN).
                self.assertEqual(list(expected.columns), list(actual.columns))
                for column in expected:
                    left, right = expected[column].reset_index(drop=True), actual[column].reset_index(drop=True)
                    pd.testing.assert_series_equal(left.isna(), right.isna())
                    pd.testing.assert_series_equal(left[left.notna()].reset_index(drop=True),
                        right[right.notna()].reset_index(drop=True), check_dtype=False, check_exact=True)

    def test_paths_multiple_entities_days_lunch_and_missing(self):
        ts = slots(20250103, 5)
        data = market(5, values=tuple(range(100, 148)))
        second = data.copy().assign(code='B')
        data['code'] = 'A'
        data = data.drop(index=[4])
        data.loc[data[TIME] == ts[6], 'level'] = np.nan
        f = pd.concat([data, second, second.assign(**{TIME: second[TIME] + 3000000})], ignore_index=True)
        got = paths(f, 5)
        for code, group in f.groupby('code'):
            expected = close_path(group, 5)
            actual = got[got.code == code].drop(columns='code').reset_index(drop=True)
            pd.testing.assert_frame_equal(expected, actual, check_dtype=False)

    def test_prefix_and_future_perturbation_nonempty(self):
        for step in (5, 15):
            m, b = market(step), boards(step)
            full = board_evidence(m, b, step)
            cutoff = slots(20250103, step)[2]
            expected = full[full[TIME] <= cutoff].reset_index(drop=True)
            got = board_evidence(m[m[TIME] <= cutoff], b[b[TIME] <= cutoff], step)
            pd.testing.assert_frame_equal(expected, got.reset_index(drop=True))
            m.loc[m[TIME] > cutoff, 'level'] *= 5
            b.loc[b[TIME] > cutoff, 'day_return'] = 10
            got = board_evidence(m, b, step)
            pd.testing.assert_frame_equal(expected, got[got[TIME] <= cutoff].reset_index(drop=True))
            self.assertGreater(int(expected.market_rising_sector_falling.sum()), 0)

    def test_duplicate_and_bad_clock_refused(self):
        f = market().assign(code='X')
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            paths(pd.concat([f, f.iloc[:1]]), 15)
        f.loc[0, TIME] += 100
        with self.assertRaisesRegex(ValueError, 'OFF_GRID'):
            paths(f, 15)

    def test_one_minute_to_five_fifteen_independent(self):
        f = bars()
        result, _ = minute_returns(f, 'X', [20250103], {20250103: 100.})
        np.testing.assert_allclose(result[5], .01)
        np.testing.assert_allclose(result[15], .01)
        missing = f.drop(index=1)
        result, _ = minute_returns(missing, 'X', [20250103], {20250103: 100.})
        self.assertTrue(np.isnan(result[5][0]))
        self.assertAlmostEqual(result[5][2], .01)
        self.assertTrue(np.isnan(result[15][0]))
        self.assertEqual(len(result[5]), 48)

    def test_conflicts_invalid_source_and_quarantine(self):
        f = bars()
        conflict = f.iloc[:1].copy()
        conflict['close'] = 100.
        result, quality = minute_returns(pd.concat([f, conflict]), 'X', [20250103], {20250103: 100.})
        self.assertEqual(quality['conflicting_timestamps'], 1)
        self.assertTrue(np.isnan(result[5][0]))
        result, _ = minute_returns(f, 'X', [20250103], {}, excluded_days=[20250103])
        self.assertTrue(np.isnan(result[5]).all())
        with self.assertRaisesRegex(ValueError, 'SOURCE_CODE'):
            minute_returns(f, 'Y', [20250103], {20250103: 100.})
        f.loc[0, 'close'] = 200.
        result, _ = minute_returns(f, 'X', [20250103], {20250103: 100.})
        self.assertTrue(np.isnan(result[5][0]))

    def test_missing_reference_not_filled_or_pre_close_used(self):
        f = bars().assign(pre_close=100.)
        result, _ = minute_returns(f, 'X', [20250103], {20250103: None})
        self.assertTrue(np.isnan(result[5]).all())

    def test_minute_future_change_does_not_change_past(self):
        f = bars()
        cutoff = f.iloc[14].date
        expected, _ = minute_returns(f[f.date <= cutoff], 'X', [20250103], {20250103: 100.})
        f.loc[f.date > cutoff, ['open', 'high', 'low', 'close']] *= 2
        got, _ = minute_returns(f, 'X', [20250103], {20250103: 100.})
        np.testing.assert_equal(expected[5][:3], got[5][:3])
        np.testing.assert_equal(expected[15][:1], got[15][:1])

    def test_fixed_membership_common_change_no_cross_stock_fill(self):
        values = np.array([[.1, .08, np.nan], [np.nan, .5, .49]])
        got = basket(values, [0, 1], 5)
        self.assertAlmostEqual(got['common_change'][1], -.02)
        self.assertAlmostEqual(got['common_change'][2], -.01)
        self.assertAlmostEqual(got['day_return'][1], .29)
        self.assertEqual(list(got['full_members']), [False, True, False])
        self.assertEqual(list(got['present']), [1, 2, 1])


if __name__ == '__main__':
    unittest.main()
