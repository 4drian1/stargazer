from copy import deepcopy
from datetime import datetime, timedelta
import unittest

from minute_risk_features_v1 import aggregate, encode, features, legacy_panel_clock, parse_ts


class MinuteRiskFeatureTests(unittest.TestCase):
    def rows(self, date=20250106):
        out = []
        d = datetime.strptime(str(date), '%Y%m%d')
        for t in (d.replace(hour=9, minute=30), d.replace(hour=13)):
            for i in range(1, 121):
                out.append(dict(code='000002', date=encode(t + timedelta(minutes=i)),
                                open=10., high=11., low=9., close=10., volume=100.))
        return out

    def test_exact_grid_and_no_lunch_stitch(self):
        bars, _ = aggregate(self.rows(), '000002', [20250106], 'END')
        self.assertEqual(len(bars), 16)
        self.assertTrue(all(b['status'].startswith('COMPLETE') for b in bars))
        self.assertEqual((bars[7]['bar_end'], bars[8]['bar_start']), (20250106113000, 20250106130000))
        self.assertEqual(bars[0]['volume'], 1500.)
        self.assertIsNone(features(bars)[8]['volume_change_previous_adjacent_bar'])

    def test_missing_minute_not_filled(self):
        rows = self.rows(); rows.pop(3)
        bars, _ = aggregate(rows, '000002', [20250106], 'END')
        self.assertEqual(bars[0]['status'], 'INCOMPLETE')
        self.assertIsNone(bars[0]['close'])
        self.assertTrue(all(f['day_return'] is None for f in features(bars)))

    def test_duplicates_identical_vs_conflicting(self):
        rows = self.rows(); rows.append(deepcopy(rows[0]))
        bars, issues = aggregate(rows, '000002', [20250106], 'END')
        self.assertTrue(bars[0]['status'].startswith('COMPLETE'))
        self.assertEqual(issues[0]['reason'], 'IDENTICAL_DUPLICATE')
        rows[-1]['close'] = 10.5
        bars, issues = aggregate(rows, '000002', [20250106], 'END')
        self.assertEqual(bars[0]['status'], 'INCOMPLETE')
        self.assertEqual(issues[0]['reason'], 'CONFLICTING_DUPLICATE')

    def test_start_end_are_not_silently_equivalent(self):
        rows = self.rows()
        a, _ = aggregate(rows, '000002', [20250106], 'START')
        b, _ = aggregate(rows, '000002', [20250106], 'END')
        self.assertEqual(a[0]['missing_minutes'], [20250106093000])
        self.assertNotEqual(a[0]['status'], b[0]['status'])
        with self.assertRaises(ValueError): aggregate(rows, '000002', [20250106], 'UNVERIFIED')

    def test_invalid_price_and_zero_volume(self):
        rows = self.rows(); rows[0]['volume'] = 0
        self.assertTrue(aggregate(rows, '000002', [20250106], 'END')[0][0]['status'].startswith('COMPLETE'))
        rows[0]['close'] = float('inf')
        self.assertEqual(aggregate(rows, '000002', [20250106], 'END')[0][0]['status'], 'INCOMPLETE')

    def test_real_dates_and_code_identity(self):
        with self.assertRaises(ValueError): parse_ts(202503123000)
        rows = self.rows(); rows[0]['code'] = '000001'
        with self.assertRaisesRegex(ValueError, 'CODE_MISMATCH'): aggregate(rows, '000002', [20250106], 'END')

    def test_legacy_clock_explicit_conversion(self):
        self.assertEqual(legacy_panel_clock(20250303, 930, 202503123000), 20250303093000)
        for h, t in [(1260, 202503156000), (930, 202503123001)]:
            with self.assertRaises(ValueError): legacy_panel_clock(20250303, h, t)

    def test_rsi_warmup_flat_and_missing_reset(self):
        bars, _ = aggregate(self.rows(), '000002', [20250106], 'END')
        f = features(bars)
        self.assertTrue(all(r['rsi14'] is None for r in f[:14]))
        self.assertEqual(f[14]['rsi14'], 50.)
        bars[4]['status'] = 'INCOMPLETE'
        self.assertTrue(all(r['rsi14'] is None for r in features(bars)))

    def test_future_price_perturbation_invariant(self):
        rows = self.rows(); changed = deepcopy(rows)
        for r in changed:
            if r['date'] > 20250106101500:
                for k in ('open', 'high', 'low', 'close', 'volume'): r[k] *= 10
        a = features(aggregate(rows, '000002', [20250106], 'END')[0])
        b = features(aggregate(changed, '000002', [20250106], 'END')[0])
        self.assertEqual(a[:3], b[:3])


if __name__ == '__main__':
    unittest.main()
