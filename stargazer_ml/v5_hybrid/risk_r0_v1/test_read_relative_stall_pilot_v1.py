import unittest

import pandas as pd

from read_relative_stall_pilot_v1 import OUT, golden_context, read


class ContextTests(unittest.TestCase):
    def test_prior_daily_only(self):
        states = [{'asof_day': 20250102, 'low': 90., 'high': 110.},
                  {'asof_day': 20250103, 'low': 20., 'high': 150.}]
        result = golden_context({'close': 100.}, states, 20250103)
        self.assertEqual(result['sse_golden_v2_high'], 110.)
        self.assertEqual(result['sse_golden_v2_asof_day'], 20250102)
        self.assertEqual(result['sse_golden_v2_distance_0_5'], 0.)
        self.assertFalse(result['golden_proximity_rule_frozen'])
        changed = golden_context({'close': 100.}, states[:1], 20250103)
        self.assertEqual(result, changed)

    def test_no_daily_state_remains_unknown(self):
        result = golden_context({'close': 100.}, [], 20250103)
        self.assertTrue(all(value is None for key, value in result.items()
                            if key != 'golden_proximity_rule_frozen'))

    def test_five_minute_never_borrows_fifteen(self):
        with self.assertRaisesRegex(ValueError, 'NO_15M_FALLBACK'):
            read(20250103094000, frequency=5)

    @unittest.skipUnless((OUT / 'report.json').exists(), 'Real pilot not yet materialized')
    def test_actual_view_both_benchmarks_and_unknowns(self):
        for benchmark in ('000001.SH', '399006.SZ'):
            record, boards, leaders = read(20250103101500, benchmark)
            self.assertEqual(len(boards), 978)
            self.assertEqual(record['benchmark'], benchmark)
            self.assertLess(record['sse_golden_v2_asof_day'], 20250103)
            self.assertEqual(set(boards.benchmark_symbol), {benchmark})
            self.assertIsNone(record['actual_historical_arrival'])
            self.assertIsNone(record['selection_topk'])
            self.assertTrue((leaders.capture_ts_within_day <= 20250103101500).all())
            self.assertFalse(boards.empty)
            self.assertFalse(leaders.empty)
            self.assertTrue(pd.isna(boards.loc[~boards.full_members, 'sector_close_new_day_high']).all())

    @unittest.skipUnless((OUT / 'report.json').exists(), 'Real pilot not yet materialized')
    def test_actual_no_unlisted_day(self):
        with self.assertRaisesRegex(ValueError, 'NO_FALLBACK'):
            read(20261016101500)


if __name__ == '__main__':
    unittest.main()
