import unittest

import numpy as np
import pandas as pd
from pathlib import Path

from multiboard_context_counts_v1 import integer_counts, reconstruct


def example():
    f = pd.DataFrame({'x_observed_share': [1., 1., 0.], 'x_positive_share': [.75, 0., np.nan]})
    names = [f'{b}_positive_{s}' for b in ('sse', 'cyb') for s in ('nonpositive', 'falling', 'not_new_high')]
    names += ['current_change_negative', 'captured_change_negative']
    for name in names:
        f['x_' + name + '_rate'] = [.5, np.nan, np.nan]
        f['x_' + name + '_coverage'] = [2 / 3, np.nan, np.nan]
    return f


class DenominatorTests(unittest.TestCase):
    def test_true_known_unknown_counts_do_not_fill_rates(self):
        f = example(); before = f.copy()
        counts = reconstruct(f, 'x_', 4)
        self.assertEqual(list(counts.x_positive_count), [3, 0, 0])
        self.assertEqual(list(counts.x_sse_positive_nonpositive_known_count), [2, 0, 0])
        self.assertEqual(list(counts.x_sse_positive_nonpositive_true_known_count), [1, 0, 0])
        self.assertEqual(list(counts.x_sse_positive_nonpositive_unknown_count), [1, 0, 0])
        pd.testing.assert_frame_equal(f, before)

    def test_noninteger_or_negative_counts_refused(self):
        for values in ([.3], [-1]):
            with self.assertRaises(ValueError):
                integer_counts(values)

    def test_unknown_coverage_not_assumed_safe(self):
        f = example()
        f.loc[0, 'x_sse_positive_falling_coverage'] = np.nan
        result = reconstruct(f, 'x_', 4)
        self.assertTrue(pd.isna(result.loc[0, 'x_sse_positive_falling_known_count']))
        self.assertTrue(pd.isna(result.loc[0, 'x_sse_positive_falling_unknown_count']))

    def test_outside_ratio_refused(self):
        f = example(); f.loc[0, 'x_positive_share'] = 2.
        with self.assertRaisesRegex(ValueError, 'OUTSIDE'):
            reconstruct(f, 'x_', 4)

    @unittest.skipUnless((Path(__file__).parent / 'multiboard_full_features_20261018_v1/report.json').exists(), 'Full feature build pending')
    def test_real_full_cohort_counts_and_rate_identity(self):
        from multiboard_context_counts_v1 import read_counts
        from read_multiboard_full_features_v1 import read_segment
        rows = 0
        for segment in ('train', 'select', 'calibration', 'test'):
            counts, original = read_counts(segment), read_segment(segment)
            self.assertEqual(len(counts), len(original))
            rows += len(counts)
            for frequency in (5, 15):
                for group in ('mb', 'own'):
                    prefix = f'{group}{frequency}_'
                    positive = counts[prefix + 'positive_count']
                    known = counts[prefix + 'sse_positive_nonpositive_known_count']
                    true = counts[prefix + 'sse_positive_nonpositive_true_known_count']
                    self.assertTrue((known.dropna() >= 0).all())
                    self.assertTrue((known <= positive).fillna(True).all())
                    mask = known > 0
                    np.testing.assert_allclose((true[mask] / known[mask]).to_numpy(dtype=float),
                        original.loc[mask, prefix + 'sse_positive_nonpositive_rate'], atol=1e-12, rtol=0)
                    mask = positive > 0
                    np.testing.assert_allclose((known[mask] / positive[mask]).to_numpy(dtype=float),
                        original.loc[mask, prefix + 'sse_positive_nonpositive_coverage'], atol=1e-12, rtol=0, equal_nan=True)
        self.assertEqual(rows, 100704)


if __name__ == '__main__':
    unittest.main()
