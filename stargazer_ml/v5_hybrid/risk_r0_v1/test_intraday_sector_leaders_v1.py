import unittest
from copy import deepcopy

from intraday_sector_leaders_v1 import advance


class LeaderTests(unittest.TestCase):
    def setUp(self):
        self.t = 20250102100000
        self.universe = dict(id='fixed-concept-v1', taxonomy='EM_CONCEPT', boards=['A', 'B'],
                             known_at=20250102090000, reference_rule='verified_previous_close')
        self.boards = {key: self.quote(ret) for key, ret in [('A', .03), ('B', .02)]}
        self.members = {'A': dict(codes=['x', 'y'], known_at=20250102090000),
                        'B': dict(codes=['z'], known_at=20250102090000)}
        self.stocks = {key: self.quote(ret) for key, ret in [('x', .07), ('y', .05), ('z', .04)]}

    def quote(self, ret):
        return dict(bar_end=self.t, known_at=self.t, day_return=ret)

    def run_step(self, old=None):
        return advance(old, self.t, self.t, self.universe, self.boards, self.members, self.stocks)

    def test_select(self):
        result = self.run_step()
        self.assertEqual((result['current_board'], result['current_stock']), ('A', 'x'))
        self.assertEqual(result['action'], 'NONE')

    def test_fall_retained_and_previous_immutable(self):
        first = self.run_step()
        frozen = deepcopy(first)
        self.t = 20250102100500
        self.boards = {key: self.quote(ret) for key, ret in [('A', -.01), ('B', .04)]}
        self.stocks = {key: self.quote(ret) for key, ret in [('x', -.03), ('y', .01), ('z', .06)]}
        second = self.run_step(first)
        self.assertEqual(second['current_stock'], 'z')
        self.assertIn('A|x', second['captured_stocks'])
        self.assertIn('B|z', second['captured_stocks'])
        self.assertEqual(first, frozen)

    def test_missing_board_not_safe(self):
        self.boards.pop('B')
        self.assertEqual(self.run_step()['status'], 'UNKNOWN_BOARD_RANK')

    def test_late_board(self):
        self.boards['B']['known_at'] = 20250102100500
        self.assertEqual(self.run_step()['status'], 'UNKNOWN_BOARD_RANK')

    def test_different_cutoff(self):
        self.boards['A']['bar_end'] = 20250102095500
        self.assertEqual(self.run_step()['status'], 'UNKNOWN_BOARD_RANK')

    def test_tie_not_code_order_winner(self):
        self.boards['B']['day_return'] = .03
        result = self.run_step()
        self.assertEqual(result['status'], 'BOARD_TIE_PENDING')
        self.assertIsNone(result['current_board'])

    def test_stock_tie(self):
        self.stocks['y']['day_return'] = .07
        self.assertEqual(self.run_step()['status'], 'STOCK_TIE_PENDING')

    def test_missing_member_not_resized(self):
        self.stocks.pop('y')
        self.assertEqual(self.run_step()['status'], 'UNKNOWN_STOCK_RANK')

    def test_late_membership(self):
        self.members['A']['known_at'] = 20250102100500
        self.assertEqual(self.run_step()['status'], 'UNKNOWN_MEMBERSHIP')

    def test_future_prices_cannot_change_past(self):
        baseline = self.run_step()
        self.stocks['future'] = dict(day_return=999, bar_end=20250102150000, known_at=20250102150000)
        self.assertEqual(baseline, self.run_step())

    def test_no_day_reset_without_policy(self):
        first = self.run_step()
        self.t = 20250103100000
        result = self.run_step(first)
        self.assertEqual(result['status'], 'DAY_BOUNDARY_POLICY_PENDING')
        self.assertEqual(result['captured_stocks'], first['captured_stocks'])

    def test_drift_and_time_reversal(self):
        first = self.run_step()
        with self.assertRaises(ValueError):
            self.run_step(first)
        self.universe['id'] = 'changed'
        with self.assertRaises(ValueError):
            self.run_step(first)

    def test_nan_not_rankable(self):
        self.stocks['x']['day_return'] = float('nan')
        self.assertEqual(self.run_step()['status'], 'UNKNOWN_STOCK_RANK')


if __name__ == '__main__':
    unittest.main()
