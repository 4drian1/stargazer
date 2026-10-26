from copy import deepcopy
import unittest

from intraday_sector_leaders_v2 import advance


class DynamicCatalogTests(unittest.TestCase):
    def setUp(self):
        self.time = 20250102100000
        self.universe = dict(id='v1', taxonomy='THS_CONCEPT', boards=['A'],
                             known_at=20250102090000, reference_rule='verified_previous_close')
        self.members = {b: dict(codes=[b + '_stock'], known_at=20250102090000) for b in ('A', 'B')}

    def quote(self, ret):
        return dict(day_return=ret, bar_end=self.time, known_at=self.time)

    def step(self, previous=None, missing_new=False):
        boards = {b: self.quote(ret) for b, ret in [('A', .01), ('B', .05)]}
        if missing_new:
            boards.pop('B')
        stocks = {'A_stock': self.quote(.02), 'B_stock': self.quote(.08)}
        return advance(previous, self.time, self.time, self.universe, boards, self.members, stocks)

    def new_version(self):
        self.time = 20250102100500
        self.universe.update(id='v2', parent_id='v1', boards=['A', 'B'], known_at=self.time)

    def test_new_board_can_lead_and_old_leader_remains(self):
        first = self.step()
        unchanged = deepcopy(first)
        self.new_version()
        result = self.step(first)
        self.assertEqual(result['current_board'], 'B')
        self.assertIn('A|A_stock', result['captured_stocks'])
        self.assertIn('B|B_stock', result['captured_stocks'])
        self.assertEqual(len(result['catalog_history']), 2)
        self.assertEqual(first, unchanged)

    def test_missing_new_quote_does_not_shrink_scope(self):
        first = self.step()
        self.new_version()
        result = self.step(first, missing_new=True)
        self.assertEqual(result['status'], 'UNKNOWN_BOARD_RANK')
        self.assertIn('A|A_stock', result['captured_stocks'])

    def test_future_catalog_not_usable_now(self):
        first = self.step()
        self.new_version()
        self.universe['known_at'] = 20250102101000
        with self.assertRaisesRegex(ValueError, 'UNIVERSE_NOT_AVAILABLE'):
            self.step(first)

    def test_parent_required(self):
        first = self.step()
        self.new_version()
        self.universe['parent_id'] = 'unrelated'
        with self.assertRaisesRegex(ValueError, 'CATALOG_PARENT'):
            self.step(first)

    def test_same_id_cannot_change_catalog(self):
        first = self.step()
        self.time = 20250102100500
        self.universe['boards'].append('B')
        with self.assertRaisesRegex(ValueError, 'CATALOG_CHANGED'):
            self.step(first)

    def test_em_not_silently_used_as_ths(self):
        self.universe['taxonomy'] = 'EM_CONCEPT'
        with self.assertRaisesRegex(ValueError, 'THS_CONCEPT'):
            self.step()


if __name__ == '__main__':
    unittest.main()
