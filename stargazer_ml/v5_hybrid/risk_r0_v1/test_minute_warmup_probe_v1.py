from datetime import datetime, timedelta
import unittest

from minute_risk_features_v1 import encode
from minute_warmup_probe_v1 import compare_sources, rebuild, timestamp_profile


def fixture(day):
    base = datetime.strptime(str(day), '%Y%m%d')
    rows = []
    for hour, minute in ((9, 30), (13, 0)):
        start = base.replace(hour=hour, minute=minute)
        for i in range(1, 121):
            rows.append(dict(code='A', date=encode(start + timedelta(minutes=i)),
                             open=10., high=10., low=10., close=10., volume=1.))
    return rows


class WarmupTests(unittest.TestCase):
    def test_prior_session_fills_rsi_without_source_stitching(self):
        out = rebuild(fixture(20250103) + fixture(20250106), 'A', 20250103, 20250106, 'END')
        self.assertEqual(out['e_rsi_nonnull'], 16)
        self.assertEqual(out['e_same_slot_volume_nonnull'], 16)
        self.assertTrue(all(v['fields']['rsi14'] == 50 for v in out['e_day']))

    def test_missing_previous_keeps_null_baseline(self):
        out = rebuild(fixture(20250106), 'A', 20250103, 20250106, 'END')
        self.assertEqual(out['e_rsi_nonnull'], 2)
        self.assertEqual(out['e_same_slot_volume_nonnull'], 0)

    def test_comparison_preserves_conflicts_and_volume_difference(self):
        a = fixture(20250106)[:1]
        b = [dict(a[0], volume=2.), dict(a[0], volume=3.)]
        out = compare_sources(a, b)
        self.assertEqual(out['price_different'], 0)
        self.assertEqual(out['volume_different'], 1)
        self.assertEqual(out['conflicting_right_keys'], 1)

    def test_empty_profile_not_full_coverage(self):
        out = timestamp_profile([], 20250103)
        self.assertEqual(out['rows'], 0)
        self.assertEqual(len(out['missing_boundary_hhmm']), 4)


if __name__ == '__main__':
    unittest.main()
