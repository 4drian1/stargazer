import unittest

import numpy as np
import pandas as pd

from conditional_board_patch_v1 import patch_missing, with_new_high, differences
from multiboard_risk_features_v1 import matrix_features, summarize


class PatchTests(unittest.TestCase):
    def test_only_admitted_missing_cells_change(self):
        old = np.array([.1, np.nan, np.nan])
        new, count = patch_missing(old, np.array([.9,.2,.3]), [False,True,False])
        np.testing.assert_allclose(new, [.1,.2,np.nan], equal_nan=True)
        np.testing.assert_allclose(old, [.1,np.nan,np.nan], equal_nan=True)
        self.assertEqual(count, 1)

    def test_known_overwrite_and_infinity_refused(self):
        with self.assertRaisesRegex(ValueError, 'REPLACE_KNOWN'):
            patch_missing([.1], [.2], [True])
        with self.assertRaisesRegex(ValueError, 'INFINITE'):
            patch_missing([np.nan], [np.inf], [True])

    def test_missing_fresh_never_becomes_zero(self):
        new, count = patch_missing([np.nan], [np.nan], [True])
        self.assertTrue(np.isnan(new[0])); self.assertEqual(count, 0)

    def test_sparse_diff_preserves_null_transitions(self):
        keys = pd.DataFrame({'t': [1,2,3]})
        result = differences(keys, {'x':[np.nan,0.,1.]}, {'x':[0.,np.nan,1.]})
        self.assertEqual(result.t.tolist(), [1,2])
        self.assertTrue(np.isnan(result.old_value.iloc[0]))
        self.assertTrue(np.isnan(result.new_value.iloc[1]))

    def test_both_frequencies_prefix_and_future_perturbation(self):
        for frequency in (5,15):
            n = 240//frequency
            data = np.array([np.linspace(.01,.04,n),np.linspace(.03,-.01,n)])
            markets = {'sse': np.linspace(100,105,n), 'cyb':np.linspace(100,102,n)}
            def evaluate(x):
                board = matrix_features(x, [[0,1]], frequency)
                return with_new_high(summarize(board,markets,frequency,'mb_'),markets,frequency,'mb_')
            original = evaluate(data)
            for value in [np.nan, 9.]:
                changed = data.copy(); changed[:,n//2+1:] = value
                actual = evaluate(changed)
                for name in original:
                    np.testing.assert_allclose(actual[name][:n//2+1],original[name][:n//2+1],equal_nan=True)
            self.assertTrue(any(np.isfinite(a).any() for a in original.values()))

    def test_candidate_exclusion_and_retained_leader(self):
        for frequency in (5,15):
            n = 240//frequency
            x = np.full((2,n), .01); x[0,:2] = [.2,.3]; x[0,2:] = -.1
            b = matrix_features(x, [[0,1]], frequency, exclude_index=0)
            np.testing.assert_allclose(b['mean'], .01)
            self.assertEqual(b['captured_count'][0,2], 0.)
            full = matrix_features(x, [[0,1]], frequency)
            self.assertGreater(full['captured_count'][0,2], 0.)


if __name__ == '__main__':
    unittest.main()
