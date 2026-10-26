import unittest
import numpy as np
import pandas as pd

from explain_multiboard_unknown_rates_v1 import reasons


class UnknownReasonTests(unittest.TestCase):
    def test_empty_group_and_unobservable_nonempty_group_are_different(self):
        frame = pd.DataFrame({'own5_positive_share': [0., .5, .5, .5, np.nan],
            'own5_sse_positive_not_new_high_rate': [np.nan, np.nan, 0., .4, np.nan],
            'own5_sse_positive_not_new_high_coverage': [np.nan, 0., .5, .5, np.nan]})
        counts = reasons(frame, 'own5_sse_positive_not_new_high_rate')
        self.assertEqual(counts, dict(known=2, no_current_positive_group=1,
            nonempty_group_no_evaluable_signal=1, unresolved_other=1))
        self.assertEqual(sum(counts.values()), 5)

    def test_original_unknown_signal_not_filled(self):
        frame = pd.DataFrame({'mb15_positive_share': [0.],
            'mb15_sse_positive_falling_rate': [np.nan], 'mb15_sse_positive_falling_coverage': [np.nan]})
        original = frame.copy()
        reasons(frame, 'mb15_sse_positive_falling_rate')
        pd.testing.assert_frame_equal(frame, original)


if __name__ == '__main__':
    unittest.main()
