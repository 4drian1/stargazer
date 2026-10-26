import unittest
import numpy as np
import pandas as pd

from volume_exhaustion_exploration_v1 import (signals, compare_arms, anchor_context,
                                            aggregate_prices, valid, summarize)


class ExplorationTests(unittest.TestCase):
    def bars(self):
        return np.tile([100., 102., 98., 100., 100.], (100, 1))

    def event(self):
        x = self.bars()
        x[70] = [100., 101., 94., 95., 200.]
        x[71] = [95., 96., 92., 94., 120.]
        return x

    def test_signal_known_only_at_first_shrink(self):
        x = self.event()
        self.assertIn((70, 71), signals(x))
        self.assertEqual(signals(x[:72]), [(70, 71)])
        x[72:, :4] *= 2
        self.assertIn((70, 71), signals(x))

    def test_no_same_close_entry(self):
        x = self.event()
        arms = compare_arms(x, 71)
        self.assertEqual([arms[f'wait_{k}']['entry_index'] for k in range(4)], [72, 73, 74, 75])

    def test_confirmation_next_open_and_previous_mean(self):
        x = self.event()
        x[73] = [99, 102, 98, 101, 180]
        self.assertEqual(compare_arms(x, 71)['reversal_within_3']['entry_index'], 74)

    def test_no_confirmation_stays_in_denominator(self):
        arms = compare_arms(self.event(), 71)
        self.assertIsNone(arms['reversal_within_3'])
        result = summarize([dict(arms=arms)])['reversal_within_3']
        self.assertEqual((result['signals'], result['reference_entries'], result['no_confirmation']), (1, 0, 1))
        self.assertEqual(result['opportunity_price_return_mean'], 0)

    def test_invalid_or_missing_future_rejects_all_arms(self):
        x = self.event()
        x[79, 2] = np.nan
        self.assertIsNone(compare_arms(x, 71))
        self.assertIsNone(compare_arms(self.event()[:79], 71))

    def test_ohlc_and_volume(self):
        x = self.bars()
        x[0, 3] = 200
        x[1, 4] = 0
        self.assertFalse(valid(x)[:2].any())

    def test_no_future_anchor(self):
        x = self.event()
        before = anchor_context(x, 70, 71)
        x[72:] *= 10
        self.assertEqual(before, anchor_context(x, 70, 71))

    def test_duplicate_conflict_is_not_silent(self):
        a = dict(code='000001', date=20260101, open=10, high=11, low=9,
                 close=10, volume=100, source='a')
        frame, conflicts = aggregate_prices([pd.DataFrame([a, dict(a, source='b', close=10.5)])])
        self.assertTrue(frame.empty)
        self.assertEqual(len(conflicts), 1)
        frame, conflicts = aggregate_prices([pd.DataFrame([a, dict(a, source='b')])])
        self.assertEqual((len(frame), len(conflicts)), (1, 0))


if __name__ == '__main__':
    unittest.main()
