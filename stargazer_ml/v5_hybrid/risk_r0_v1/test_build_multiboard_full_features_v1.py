import unittest

import numpy as np
import pandas as pd

from build_multiboard_full_features_v1 import TIME, daily_references, golden_fields, market_arrays
from intraday_relative_stall_v1 import slots


class FullBuildHelpersTests(unittest.TestCase):
    def test_daily_reference_keeps_keys_and_rejects_conflict(self):
        frame = pd.DataFrame(dict(code=['X', 'X'], cutoff_index=[10, 11],
            stock_reference_close_raw=[100., 100.], stock_day_return_raw=[.01, .02]))
        reference = np.full((1, 1), np.nan)
        old = dict(values=np.full((1, 2), np.nan), _map={10: 0, 11: 1})
        daily_references(frame, {'X': 0}, {10: 0, 11: 0}, reference, old)
        np.testing.assert_equal(reference, [[100.]])
        np.testing.assert_equal(old['values'], [[.01, .02]])
        frame.loc[1, 'stock_reference_close_raw'] = 99.
        with self.assertRaisesRegex(ValueError, 'REFERENCE_CONFLICT'):
            daily_references(frame, {'X': 0}, {10: 0, 11: 0}, reference, old)

    def test_current_daily_golden_never_used(self):
        states = [dict(asof_day=20250102, low=90., high=110.),
                  dict(asof_day=20250103, low=10., high=200.)]
        result = golden_fields([20250103], np.full(16, 100.), states)
        np.testing.assert_equal(np.asarray(result['golden_asof_day']), 20250102)
        np.testing.assert_equal(np.asarray(result['new_sse_golden_distance_0_5']), 0.)
        np.testing.assert_equal(np.asarray(result['new_sse_golden_position']), .5)
        self.assertEqual(result, golden_fields([20250103], np.full(16, 100.), states[:1]))

    def test_market_exact_grid_missing_not_filled(self):
        times = slots(20250103, 5)
        frame = pd.DataFrame([dict(symbol=s, frequency_minutes=5, **{TIME: t}, close=100., source_complete=True)
                              for s in ('000001.SH', '399006.SZ') for t in times])
        frame = frame.drop(index=1)
        grid, result = market_arrays(frame, [20250103], 5)
        self.assertEqual(grid, times)
        self.assertTrue(np.isnan(result['sse'][1]))
        self.assertEqual(result['cyb'][1], 100.)

    def test_market_duplicate_rejected(self):
        row = dict(symbol='000001.SH', frequency_minutes=15, **{TIME: 20250103094500}, close=100., source_complete=True)
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            market_arrays(pd.DataFrame([row, row]), [20250103], 15)


if __name__ == '__main__':
    unittest.main()
