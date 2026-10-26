from copy import deepcopy
import unittest

from check_ths_cache_slots_v1 import check_day, grid, window_digest


class SlotTests(unittest.TestCase):
    def setUp(self):
        self.day, self.code = 20250102, '000001'
        self.rows = [dict(code=self.code, date=t, open=10, high=11, low=9, close=10, volume=1, amount=10)
                     for t in sorted(grid(self.day, 'END'))]

    def check(self, rows=None):
        return check_day(self.rows if rows is None else rows, self.code, self.day)

    def test_full_end_not_verified_clock(self):
        r=self.check()
        self.assertTrue(r['scenarios']['END']['structural_complete'])
        self.assertFalse(r['clock_verified'])
        self.assertFalse(r['admitted'])

    def test_missing_slot_cannot_be_filled_by_other_stock(self):
        self.rows[0]['code']='000002'
        self.assertFalse(self.check()['scenarios']['END']['structural_complete'])

    def test_duplicate_does_not_replace_missing_slot(self):
        self.rows[0]=deepcopy(self.rows[1])
        r=self.check()
        self.assertEqual(r['scenarios']['END']['missing'],1)

    def test_conflicting_duplicate_blocks(self):
        self.rows.append(dict(self.rows[0],close=9.9))
        self.assertEqual(self.check()['conflicting_minutes'],1)
        self.assertFalse(self.check()['scenarios']['END']['structural_complete'])

    def test_bad_price_and_nan_not_accepted(self):
        for bad in [0,float('nan'),float('inf')]:
            self.rows[0]['close']=bad
            self.assertFalse(self.check()['scenarios']['END']['structural_complete'])
            self.assertIsInstance(window_digest(self.rows),str)

    def test_lunch_extra_not_needed_slot(self):
        self.rows.append(dict(self.rows[0],date=20250102120000))
        r=self.check()
        self.assertTrue(r['scenarios']['END']['structural_complete'])
        self.assertEqual(r['scenarios']['END']['extraneous'],1)

    def test_start_end_not_interchangeable(self):
        self.assertFalse(self.check()['scenarios']['START']['structural_complete'])
        self.assertEqual(len(grid(self.day,'START')),240)

    def test_hash_changes_when_price_changes(self):
        before=window_digest(self.rows)
        self.rows[0]['close']=9.9
        self.assertNotEqual(before,window_digest(self.rows))


if __name__=='__main__':
    unittest.main()
