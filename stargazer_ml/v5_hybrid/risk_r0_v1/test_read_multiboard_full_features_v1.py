import unittest

import numpy as np
import pandas as pd

from build_multiboard_full_features_v1 import OUT, SEGMENTS
from intraday_relative_stall_v1 import TIME
from read_multiboard_full_features_v1 import attach, metadata, read_segment


class FeatureReaderTests(unittest.TestCase):
    def inputs(self):
        original = pd.DataFrame({'code': ['X', 'Y'], TIME: [20250103101500] * 2, 'y_tail_1200m': [0, 1]})
        own = pd.DataFrame({'code': ['Y', 'X'], TIME: [20250103101500] * 2, 'own5_mean': [.02, .01]})
        shared = pd.DataFrame({TIME: [20250103101500], 'mb5_mean': [.03]})
        return original, own, shared

    def test_exact_join_preserves_labels_and_order(self):
        original, own, shared = self.inputs()
        result = attach(original, own, shared, ['own5_mean', 'mb5_mean'])
        pd.testing.assert_frame_equal(result[list(original)], original)
        np.testing.assert_allclose(result.own5_mean, [.01, .02])

    def test_missing_extra_duplicate_and_collision_rejected(self):
        original, own, shared = self.inputs()
        with self.assertRaisesRegex(ValueError, 'KEYS_NOT_IDENTICAL'):
            attach(original, own.iloc[:1], shared, ['own5_mean', 'mb5_mean'])
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            attach(original, pd.concat([own, own.iloc[:1]]), shared, ['own5_mean', 'mb5_mean'])
        with self.assertRaisesRegex(ValueError, 'NO_ASOF'):
            attach(original, own, shared.assign(**{TIME: 20250103100000}), ['own5_mean', 'mb5_mean'])
        with self.assertRaisesRegex(ValueError, 'COLLISION'):
            attach(original, own, shared, ['y_tail_1200m'])

    def test_unknown_segment_refused(self):
        with self.assertRaisesRegex(ValueError, 'UNKNOWN_SEGMENT'):
            read_segment('all')

    @unittest.skipUnless((OUT / 'report.json').exists(), 'Full period feature build pending')
    def test_real_all_segments_no_population_or_label_change(self):
        from join_momentum_review_v1 import read_segment as original_reader
        report, spec = metadata()
        count = 0
        self.assertEqual(len(spec['shared']) + len(spec['own']), 106)
        for segment in SEGMENTS:
            actual, expected = read_segment(segment), original_reader(segment)
            pd.testing.assert_frame_equal(actual[list(expected)], expected)
            self.assertTrue((actual.golden_asof_day < actual[TIME] // 1000000).all())
            self.assertFalse(np.isinf(actual[spec['shared'] + spec['own']].to_numpy(dtype=float)).any())
            count += len(actual)
        self.assertEqual(count, 100704)
        self.assertFalse(report['fit'])
        self.assertFalse(report['labels_changed'])


if __name__ == '__main__':
    unittest.main()
