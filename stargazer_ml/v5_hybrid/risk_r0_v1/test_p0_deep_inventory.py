import unittest
import numpy as np
import pandas as pd

from p0_deep_inventory import normalize, forward_coverage, keys, board


class P0Tests(unittest.TestCase):
    def test_invalid_date_and_code_not_silently_fixed(self):
        for code, date in [('1', 20260101), ('000001', 20260230), ('000001', 20260101.5)]:
            with self.assertRaises(ValueError):
                normalize(pd.DataFrame({'code': [code], 'date': [date]}))

    def test_missing_stock_day_does_not_shift_horizon(self):
        calendar = np.array([20260105, 20260106, 20260107, 20260108, 20260109, 20260112, 20260113])
        candidate = pd.DataFrame({'code': [1], 'date': [20260105]})
        daily = pd.DataFrame({'code': [1]*6, 'date': calendar[[0, 1, 3, 4, 5, 6]]})
        stat = forward_coverage(candidate, np.sort(keys(daily)), calendar)
        self.assertEqual(stat['mature_but_missing_stock_rows'], 1)
        self.assertEqual(stat['all_next5_rows_present'], 0)

    def test_tail_and_off_calendar_are_not_missing_stock_day(self):
        calendar = np.array([20260105, 20260106, 20260107, 20260108, 20260109, 20260112])
        candidate = pd.DataFrame({'code': [1, 1], 'date': [20260106, 20260110]})
        daily = pd.DataFrame({'code': [1]*6, 'date': calendar})
        stat = forward_coverage(candidate, np.sort(keys(daily)), calendar)
        self.assertEqual(stat['calendar_tail'], 1)
        self.assertEqual(stat['anchor_date_not_observed'], 1)
        self.assertEqual(stat['mature_but_missing_stock_rows'], 0)

    def test_board_inference_does_not_call_etf_stock(self):
        self.assertEqual(board('588200'), 'OTHER_OR_NON_STOCK')
        self.assertEqual(board('688001'), 'STAR_PREFIX')


if __name__ == '__main__':
    unittest.main()
