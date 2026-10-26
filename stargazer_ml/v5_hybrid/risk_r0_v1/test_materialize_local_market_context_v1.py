from copy import deepcopy
from datetime import datetime, timedelta
import unittest

from materialize_local_market_context_v1 import build, bars_and_days, rsi_series
from tdx_index_acceptance_v1 import grid


class MarketContextTests(unittest.TestCase):
    def setUp(self):
        self.days = []
        date = datetime(2025, 1, 2)
        while len(self.days) < 24:
            if date.weekday() < 5:
                self.days.append(int(date.strftime('%Y%m%d')))
            date += timedelta(days=1)
        self.rows = []
        for i, day in enumerate(self.days):
            for j, ts in enumerate(grid(day, 5)):
                c = 100+i+j*.01
                self.rows.append(dict(date=ts, open=c, high=c+.1, low=c-.1, close=c,
                                      volume=100, amount=10000))
        self.states = [dict(asof_day=self.days[6], low=90, high=120, version=1),
                       dict(asof_day=self.days[7], low=91, high=121, version=2)]

    def test_materialize_both_frequencies(self):
        for freq, count in [(5, 48), (15, 16)]:
            rows = build(self.rows, '000001.SH', self.states, freq)
            self.assertEqual(len(rows), len(self.days)*count)
            self.assertTrue(all(r['source_complete'] for r in rows))

    def test_daily_anchors_strictly_prior_day(self):
        rows = build(self.rows, '000001.SH', self.states, 5)
        current = next(r for r in rows if r['day'] == self.days[7])
        self.assertEqual(current['anchor_asof_day'], self.days[6])
        self.assertEqual(current['anchor_low'], 90)
        self.assertTrue(all(r['anchor_asof_day'] < r['day'] for r in rows if r['anchor_asof_day']))

    def test_current_day_final_price_cannot_change_past_features(self):
        cutoff = grid(self.days[10], 5)[4]
        full = build(self.rows, '000001.SH', self.states, 5)
        expected = [r for r in full if r['decision_ts_assumed_end'] <= cutoff]
        changed = deepcopy(self.rows)
        for row in changed:
            if row['date'] > cutoff:
                for k in ['open', 'high', 'low', 'close']:
                    row[k] *= 2
        actual = build(changed, '000001.SH', self.states, 5)
        self.assertEqual(expected, [r for r in actual if r['decision_ts_assumed_end'] <= cutoff])

    def test_prefix_rebuild_and_rsi_warmup(self):
        cutoff = grid(self.days[10], 15)[3]
        full = build(self.rows, '000001.SH', self.states, 15)
        prefix = build([r for r in self.rows if r['date'] <= cutoff], '000001.SH', self.states, 15)
        self.assertEqual([r for r in full if r['decision_ts_assumed_end'] <= cutoff],
                         [r for r in prefix if r['decision_ts_assumed_end'] <= cutoff])
        self.assertIsNone(full[13]['rsi14'])
        self.assertIsNotNone(full[14]['rsi14'])

    def test_prior_close_not_current_open_reference(self):
        rows = build(self.rows, '000001.SH', self.states, 5)
        first = next(r for r in rows if r['day'] == self.days[1])
        self.assertEqual(first['reference_close'], self.rows[47]['close'])
        self.assertAlmostEqual(first['day_return_prev_close'], 101/self.rows[47]['close']-1)

    def test_daily_background_excludes_current_day(self):
        rows = build(self.rows, '000001.SH', self.states, 5)
        r = next(r for r in rows if r['day'] == self.days[21])
        self.assertAlmostEqual(r['daily_return20_asof_prior_day'], (120+.47)/(100+.47)-1)

    def test_lunch_break_resets_adjacent_momentum_not_session_fraction(self):
        rows = build(self.rows, '000001.SH', self.states, 5)
        pm = next(r for r in rows if str(r['decision_ts_assumed_end']).endswith('130500'))
        self.assertIsNone(pm['return_adjacent_bar'])
        self.assertEqual(pm['session_fraction'], 5/120)

    def test_unknown_volume_arrival_and_divergence_stay_null(self):
        rows = build(self.rows, '000001.SH', self.states, 15)
        self.assertTrue(all(r['volume_feature'] is None and r['actual_available_at'] is None
                            and r['divergence_confirmed'] is None for r in rows))

    def test_no_cyb_or_intraday_golden_redrawing(self):
        rows = build(self.rows, '399006.SZ', self.states, 15)
        self.assertTrue(all(r['anchor_asof_day'] is None for r in rows))
        sse = build(self.rows, '000001.SH', self.states, 15)
        low = {r['anchor_low'] for r in sse if r['day'] == self.days[8]}
        self.assertEqual(low, {91})

    def test_duplicate_and_invalid_prices_fail(self):
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            bars_and_days(self.rows+[self.rows[0]], 5)
        bad = deepcopy(self.rows)
        bad[0]['close'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'PRICE'):
            bars_and_days(bad, 5)

    def test_missing_5m_source_not_interpolated(self):
        bars, _ = bars_and_days(self.rows[1:], 15)
        self.assertFalse(bars[0]['complete'])
        self.assertIsNone(bars[0]['close'])

    def test_rsi_missing_resets_and_flat_is_50(self):
        self.assertEqual(rsi_series([10]*15)[-1], 50.)
        self.assertIsNone(rsi_series([10]*15+[None, 10, 10])[-1])


if __name__ == '__main__':
    unittest.main()
