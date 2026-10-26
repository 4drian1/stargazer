from copy import deepcopy
import unittest

from golden_anchor_rules_v2 import initialize, close_update, replay


def bar(day, close, opening=None, high=None, low=None):
    opening = close if opening is None else opening
    return dict(day=day, open=opening, close=close, high=max(close, opening) if high is None else high,
                low=min(close, opening) if low is None else low, completed=True)


class AsymmetricTests(unittest.TestCase):
    def setUp(self):
        self.seed = [bar(d, 100, opening=99, high=120, low=80)
                     for d in (20250102, 20250103, 20250106, 20250107, 20250108, 20250109, 20250110)]

    def walk(self, closes, opens=None):
        days = (20250113, 20250114, 20250115, 20250116, 20250117, 20250120, 20250121, 20250122)
        bars = self.seed + [bar(d, c, opening=(opens[i] if opens else c)) for i, (d, c) in enumerate(zip(days, closes))]
        return replay(bars)

    def test_three_small_declines_do_not_shrink(self):
        result = self.walk([99.9, 99.8, 99.7])[-1]
        self.assertEqual(result['high'], 120)
        self.assertFalse(result['run_has_one_percent_drop'])

    def test_one_percent_equal_boundary(self):
        result = self.walk([99, 98.9, 98.8])[-1]
        self.assertEqual(result['high'], 100)
        self.assertIn('SHRINK_HIGH', result['events'])

    def test_two_days_not_enough(self):
        self.assertEqual(self.walk([98, 97])[-1]['high'], 120)

    def test_red_candle_inside_decline_allowed(self):
        result = self.walk([99, 98.9, 98.8], [98, 98, 98])[-1]
        self.assertIn('SHRINK_HIGH', result['events'])

    def test_magnitude_can_confirm_after_third_day(self):
        result = self.walk([99.9, 99.8, 99.7, 98])[-1]
        self.assertEqual(result['high'], 100)
        self.assertEqual(result['run_count'], 4)

    def test_five_watch_seven_confirm_no_up_size_gate(self):
        result = self.walk([100 + i / 100 for i in range(1, 8)])
        self.assertTrue(result[5]['watch_up'])
        self.assertEqual(result[5]['low'], 80)
        self.assertEqual(result[-1]['low'], 100)
        self.assertEqual(result[-1]['low_day'], 20250110)

    def test_down_break_updates_immediately_despite_small_size(self):
        state = initialize(self.seed)
        state['low'] = 99.99
        result = close_update(state, bar(20250113, 99.98))
        self.assertEqual(result['low'], 99.98)
        self.assertIn('EXPAND_LOW', result['events'])

    def test_up_break_updates_immediately(self):
        state = initialize(self.seed)
        result = close_update(state, bar(20250113, 121, high=125))
        self.assertEqual(result['high'], 121)
        self.assertEqual(result['low'], 80)

    def test_equal_close_resets_run(self):
        result = self.walk([99, 98.9, 98.9])[-1]
        self.assertEqual(result['high'], 120)
        self.assertEqual(result['run_count'], 0)

    def test_direction_change_resets_origin(self):
        result = self.walk([99, 98, 99])[-1]
        self.assertEqual(result['run_count'], 1)
        self.assertEqual(result['run_origin']['close'], 98)

    def test_previous_state_unchanged_and_no_repeat(self):
        state = self.walk([99, 98.9, 98.8])[-1]
        frozen = deepcopy(state)
        result = close_update(state, bar(20250116, 98.7))
        self.assertEqual(state, frozen)
        self.assertNotIn('SHRINK_HIGH', result['events'])

    def test_confirm_and_expansion_both_recorded(self):
        state = self.walk([101, 102, 103, 104, 105, 106])[-1]
        result = close_update(state, bar(20250121, 121))
        self.assertEqual(result['events'], ['EXPAND_HIGH', 'SHRINK_LOW'])
        self.assertEqual((result['high'], result['low']), (121, 100))
        self.assertEqual(result['version'], state['version'] + 1)

    def test_incomplete_day_rejected(self):
        b = bar(20250113, 99)
        b['completed'] = False
        with self.assertRaises(ValueError):
            close_update(initialize(self.seed), b)


if __name__ == '__main__':
    unittest.main()
