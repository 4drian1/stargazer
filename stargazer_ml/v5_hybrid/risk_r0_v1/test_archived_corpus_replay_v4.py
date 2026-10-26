import unittest

from archived_corpus_replay_v4 import normalize_external, parse_daily
from historical_risk_core_v1 import label_at


class ReplayTests(unittest.TestCase):
    def test_cash_decimal_conversion(self):
        for cash, expected in [('2.49', .249), ('6.8', .68), ('39.74', 3.974)]:
            parsed = dict(bonus=[{'进度': '实施', '派息(税前)(元)': cash}], allotment=[])
            self.assertEqual(normalize_external('000001', parsed)[0]['per_share_cash'], expected)

    def test_null_stays_missing(self):
        bars, missing = parse_daily('000008', [('日k:000008:20000926', None)])
        self.assertEqual(bars, {})
        self.assertEqual(missing, [dict(date=20000926, reason='SOURCE_NULL_RECORD')])

    def test_null_window_not_labelled(self):
        calendar = [20260105, 20260106, 20260107, 20260108, 20260109, 20260112]
        records = [(f'日k:000008:{d}', dict(code='000008', date=d, open=100,
                    high=101, low=99, close=100)) for d in calendar]
        records[2] = (records[2][0], None)
        bars, _ = parse_daily('000008', records)
        result = label_at(bars, calendar, calendar[0], {d: 1.0 for d in calendar},
                          scale_verified=True, coverage_verified=True)
        self.assertIsNone(result['y_tail_5d'])

    def test_identity_rejected(self):
        for code, date in [('000001', 20260105), ('000008', 20260106)]:
            with self.assertRaises(ValueError):
                parse_daily('000008', [('日k:000008:20260105', dict(code=code, date=date))])

    def test_conflict_rejected(self):
        row = dict(code='000008', date=20260105, open=100, high=101, low=99, close=100)
        for other in (None, dict(row, close=100.5)):
            with self.assertRaisesRegex(ValueError, 'DUPLICATE_PRICE_CONFLICT'):
                parse_daily('000008', [('日k:000008:20260105', row), ('日k:000008:20260105', other)])

    def test_identical_duplicate_allowed(self):
        row = dict(code='000008', date=20260105, open=100, high=101, low=99, close=100)
        bars, _ = parse_daily('000008', [('日k:000008:20260105', row)] * 2)
        self.assertEqual(len(bars), 1)


if __name__ == '__main__':
    unittest.main()
