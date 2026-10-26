import unittest
import numpy as np

from analyze_volume_exhaustion_wait_v1 import conditional_arms, paired_block_interval


class ConditionalWaitTests(unittest.TestCase):
    def test_continue_shrink_is_separate_from_delay(self):
        x = np.tile([100., 102., 98., 100., 100.], (20, 1))
        x[5, 4], x[6, 4] = 200, 120
        x[7, 4], x[8, 4], x[9, 4] = 110, 115, 90
        arms = conditional_arms(x, 5, 6)
        self.assertEqual(arms['shrink_wait_1']['entry_index'], 8)
        self.assertIsNone(arms['shrink_wait_2'])
        self.assertIsNone(arms['shrink_wait_3'])

    def test_distant_future_does_not_change_decision(self):
        x = np.tile([100., 102., 98., 100., 100.], (20, 1))
        x[5:10, 4] = [200, 120, 110, 100, 90]
        before = conditional_arms(x, 5, 6)
        self.assertEqual(before['shrink_wait_3']['entry_index'], 10)
        x[15:] *= 2
        self.assertEqual(before, conditional_arms(x, 5, 6))

    def test_same_date_cross_section_preserved(self):
        events = [dict(signal_date=5), dict(signal_date=5)]
        interval = paired_block_interval(events, [1., -1.], list(range(30)), 30)
        self.assertEqual(interval, [0., 0.])


if __name__ == '__main__':
    unittest.main()
