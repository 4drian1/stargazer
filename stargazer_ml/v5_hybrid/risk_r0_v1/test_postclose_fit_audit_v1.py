import unittest
import numpy as np
import pandas as pd
from audit_postclose_first_fit_v1 import loss, verify_rows, day_block_interval


class AuditTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(code=['000001', '000002'], day=[20260101, 20260101],
            label_end_time=[20260102150000] * 2,
            next_session_adverse_excursion=[.01, .02], segment=['test'] * 2))

    def test_pinball_direction(self):
        np.testing.assert_allclose(loss([.1, .1], [.05, .15]), [.045, .005])

    def test_exact_keys_and_targets(self):
        frame = self.frame()
        verify_rows(frame.iloc[::-1], frame)
        changed = frame.copy()
        changed.loc[0, 'next_session_adverse_excursion'] = .03
        with self.assertRaises(AssertionError):
            verify_rows(changed, frame)

    def test_duplicate_rejected(self):
        frame = self.frame()
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            verify_rows(pd.concat([frame, frame]), frame)

    def test_date_clusters_not_individual_rows(self):
        frame = pd.DataFrame(dict(day=np.repeat(np.arange(20), 4), loss_difference=.001))
        result = day_block_interval(frame, draws=50)
        self.assertEqual(result['blocks'], 2)
        np.testing.assert_allclose(result['interval_95'], [.001, .001])


if __name__ == '__main__':
    unittest.main()
