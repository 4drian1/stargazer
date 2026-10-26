import unittest

import numpy as np
import pandas as pd

from intraday_relative_stall_v1 import TIME
from materialize_relative_stall_5m_v1 import OUT
from read_relative_stall_v2 import read, research_segment


class ReadTests(unittest.TestCase):
    def test_incomplete_period_and_unknown_benchmark_refused(self):
        with self.assertRaisesRegex(ValueError, 'COMPLETED_FREQUENCY'):
            read(20250103101000, frequency=15)
        with self.assertRaisesRegex(ValueError, 'UNSUPPORTED_BENCHMARK'):
            read(20250103101000, benchmark='STOCK')

    @unittest.skipUnless((OUT / 'report.json').exists(), 'Native 5m pilot not yet built')
    def test_actual_five_minute_never_borrows_later_fifteen(self):
        info, boards, leaders = read(20250103101000, frequency=5)
        self.assertEqual(info['frequency_minutes'], 5)
        self.assertEqual(info['source_frequency'], 'NATIVE_1M_TO_5M')
        self.assertEqual(set(boards[TIME]), {20250103101000})
        self.assertLess(info['sse_golden_v2_asof_day'], 20250103)
        self.assertIn('sse_macd_dif_over_close', info)
        self.assertIsNone(info['selection_topk'])
        self.assertFalse(leaders.empty)
        self.assertTrue((leaders.capture_ts_within_day <= 20250103101000).all())

    @unittest.skipUnless((OUT / 'report.json').exists(), 'Native 5m pilot not yet built')
    def test_both_benchmarks_periods_same_cutoff(self):
        for frequency in (5, 15):
            for symbol in ('000001.SH', '399006.SZ'):
                info, boards, _ = read(20250103101500, benchmark=symbol, frequency=frequency)
                self.assertEqual(len(boards), 978)
                self.assertEqual(info['benchmark'], symbol)
                self.assertEqual(set(boards.frequency_minutes), {frequency})
                self.assertTrue(boards.loc[~boards.full_members, 'sector_close_new_day_high'].isna().all())

    @unittest.skipUnless((OUT / 'report.json').exists(), 'Native 5m pilot not yet built')
    def test_original_cohort_labels_unmodified_context_no_labels(self):
        from join_momentum_review_v1 import read_segment
        base, context = research_segment('train')
        original = read_segment('train')
        expected = original[original[TIME].floordiv(1000000).isin([20250102, 20250103, 20250106])]
        pd.testing.assert_frame_equal(base, expected.reset_index(drop=True))
        self.assertGreater(len(base), 0)
        for frame in context.values():
            self.assertFalse(set(frame) & {'y_tail_1200m', 'mae_1200m', 'label_end_time'})
            self.assertEqual(set(frame[TIME]), set(base[TIME]))
            self.assertFalse(np.isinf(frame.select_dtypes(include='number').to_numpy(dtype=float)).any())

    @unittest.skipUnless((OUT / 'report.json').exists(), 'Native 5m pilot not yet built')
    def test_unlisted_day_never_falls_back(self):
        with self.assertRaisesRegex(ValueError, 'NO_FALLBACK'):
            read(20261016101000)


if __name__ == '__main__':
    unittest.main()
