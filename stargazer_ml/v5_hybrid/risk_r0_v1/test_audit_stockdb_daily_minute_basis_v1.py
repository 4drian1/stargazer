import unittest
from audit_stockdb_daily_minute_basis_v1 import compare_close


class BasisTests(unittest.TestCase):
    def setUp(self):
        self.code, self.day = '000001', 20250102
        self.daily = [dict(code=self.code, date=self.day, close=10)]
        self.minute = [dict(code=self.code, date=20250102150000, close=10)]

    def compare(self):
        return compare_close(self.daily, self.minute, self.code, self.day)['status']

    def test_match(self):
        self.assertEqual(self.compare(), 'SAME_CLOSE_WITHIN_001')

    def test_missing_daily_not_replaced_by_minute(self):
        self.daily = []
        self.assertEqual(self.compare(), 'MISSING_DAILY_OR_EXACT_1500')

    def test_missing_exact_close_does_not_fallback(self):
        self.minute[0]['date'] = 20250102145900
        self.assertEqual(self.compare(), 'MISSING_DAILY_OR_EXACT_1500')

    def test_close_mismatch(self):
        self.minute[0]['close'] = 9.8
        self.assertEqual(self.compare(), 'CLOSE_MISMATCH')

    def test_duplicate_conflict_not_overwritten(self):
        self.daily.append(dict(self.daily[0], close=11))
        self.assertEqual(self.compare(), 'DUPLICATE_CLOSE_CONFLICT')

    def test_invalid_close_and_wrong_identity(self):
        for bad in [float('nan'), float('inf'), 0, -1, True]:
            self.minute[0]['close'] = bad
            self.assertEqual(self.compare(), 'INVALID_CLOSE')
        self.minute[0].update(close=10, code='600000')
        self.assertEqual(self.compare(), 'CODE_IDENTITY_CONFLICT')


if __name__ == '__main__':
    unittest.main()
