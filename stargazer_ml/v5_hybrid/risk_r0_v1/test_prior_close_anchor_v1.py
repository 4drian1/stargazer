import unittest

import numpy as np
import pandas as pd

from verify_prior_close_anchor_v1 import exact_previous


class AnchorTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(code=['A']*5, cutoff_index=[0, 15, 16, 32, 48],
                                 stock_close=[90., 100., 102., 103., 104.]))

    def test_exact_adjacent_not_last_seen(self):
        x = exact_previous(self.frame())
        np.testing.assert_allclose(x, [np.nan, np.nan, 100., np.nan, np.nan], equal_nan=True)

    def test_other_stock_does_not_fill(self):
        f = self.frame()
        f.loc[1, 'code'] = 'B'
        self.assertTrue(np.isnan(exact_previous(f)).all())

    def test_future_perturbation_and_prefix(self):
        f = self.frame()
        result = exact_previous(f)
        future = pd.concat([f, pd.DataFrame(dict(code=['A'], cutoff_index=[63], stock_close=[999.]))])
        np.testing.assert_allclose(exact_previous(future)[:len(f)], result, equal_nan=True)
        np.testing.assert_allclose(exact_previous(f.iloc[:3]), result[:3], equal_nan=True)
        self.assertTrue(np.isfinite(result).any())

    def test_invalid_close(self):
        for v in [np.nan, np.inf, -1., 0.]:
            with self.subTest(value=v):
                f = self.frame(); f.loc[1, 'stock_close'] = v
                self.assertTrue(np.isnan(exact_previous(f)).all())

    def test_duplicate_refused(self):
        f = self.frame()
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            exact_previous(pd.concat([f, f.iloc[[1]]]))

    def test_negative_cutoff(self):
        f = self.frame(); f.loc[0, 'cutoff_index'] = -1
        with self.assertRaisesRegex(ValueError, 'INVALID_CUTOFF'):
            exact_previous(f)

    def test_missing_final_bar_not_replaced_by_earlier_bar(self):
        f = self.frame(); f.loc[1, 'cutoff_index'] = 14
        self.assertTrue(np.isnan(exact_previous(f)).all())


if __name__ == '__main__':
    unittest.main()
