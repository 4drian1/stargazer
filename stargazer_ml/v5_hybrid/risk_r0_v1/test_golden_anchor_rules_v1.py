from copy import deepcopy
import unittest

from golden_anchor_rules_v1 import initialize, close_update, observe, between_levels, time_nodes


def bar(day, high=110., low=90., close=100., completed=True):
    return dict(day=day, high=high, low=low, close=close, completed=completed)


class AnchorTests(unittest.TestCase):
    def setUp(self):
        self.days = [bar(d) for d in (20250102, 20250103, 20250106, 20250107, 20250108, 20250109, 20250110)]
        self.state = initialize(self.days)

    def test_seven_required(self):
        with self.assertRaises(ValueError):
            initialize(self.days[:6])

    def test_duplicate_day_rejected(self):
        with self.assertRaises(ValueError):
            initialize(self.days[:-1] + [self.days[0]])

    def test_incomplete_rejected(self):
        with self.assertRaises(ValueError):
            close_update(self.state, bar(20250113, completed=False))

    def test_hold_when_old_anchor_outside_seven(self):
        old = deepcopy(self.state)
        result = close_update(self.state, bar(20250210, high=105, low=95))
        self.assertEqual((result['high'], result['low'], result['version']), (110, 90, 1))
        self.assertEqual(old, self.state)

    def test_high_update_is_close_not_high(self):
        new = close_update(self.state, bar(20250113, high=120, low=95, close=115))
        self.assertEqual((new['high'], new['low']), (115, 90))

    def test_low_update_is_close_not_low(self):
        new = close_update(self.state, bar(20250113, high=100, low=80, close=85))
        self.assertEqual((new['high'], new['low']), (110, 85))

    def test_intraday_two_sides_do_not_move_line(self):
        original = deepcopy(self.state)
        self.assertEqual(observe(self.state, 120)['relation'], 'ABOVE')
        self.assertEqual(observe(self.state, 80)['relation'], 'BELOW')
        self.assertEqual(original, self.state)
        new = close_update(self.state, bar(20250113, high=120, low=80))
        self.assertEqual(new['event'], 'HOLD')

    def test_equal_is_explicit_engineering_hold(self):
        self.assertEqual(close_update(self.state, bar(20250113, close=110))['event'], 'HOLD')

    def test_reverse_day_rejected(self):
        with self.assertRaises(ValueError):
            close_update(self.state, self.days[-1])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            close_update(self.state, bar(20250113, close=float('nan')))

    def test_original_example_levels(self):
        result = between_levels(3674.82, 3822.11, ['0.236', '0.5'])
        self.assertAlmostEqual(result['0.236'], 3709.58044)
        self.assertAlmostEqual(result['0.5'], 3748.465)

    def test_sessions_separate(self):
        nodes = time_nodes(20261016, ['0.382'])
        self.assertTrue(nodes[0]['node'].endswith('10:15:50.400000'))
        self.assertTrue(nodes[1]['node'].endswith('13:45:50.400000'))
        self.assertTrue(nodes[0]['latest_bar_end_under_end_assumption']['5'].endswith('10:15:00'))

    def test_no_complete_bar_at_open(self):
        self.assertIsNone(time_nodes(20261016, ['0'])[0]['latest_bar_end_under_end_assumption']['5'])

    def test_ratio_outside_session_rejected(self):
        with self.assertRaises(ValueError):
            time_nodes(20261016, ['1.272'])


if __name__ == '__main__':
    unittest.main()
