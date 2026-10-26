import unittest
import math
import numpy as np
import pandas as pd
from multitimeframe_momentum_v1 import MomentumState, SeededEMA, confirmed_divergence
from materialize_multitimeframe_momentum_v1 import build, PARAMETERS
from materialize_local_market_context_v1 import rsi_series
from tdx_index_acceptance_v1 import grid


class MomentumTests(unittest.TestCase):
    def state(self):
        return MomentumState(**PARAMETERS)

    def test_rsi_matches_existing_reference(self):
        values = [100 + math.sin(i / 2) * 2 + i / 10 for i in range(80)]
        state = self.state()
        actual = [state.update(v)['rsi14'] for v in values]
        np.testing.assert_allclose(np.array(actual, dtype=float), np.array(rsi_series(values), dtype=float), equal_nan=True, atol=1e-12)

    def test_ema_seed_and_recurrence(self):
        ema = SeededEMA(3)
        self.assertEqual([ema.update(v) for v in [2, 4, 6, 8]], [None, None, 4., 6.])

    def test_macd_warmup_and_formula(self):
        state = self.state()
        rows = [state.update(float(v)) for v in range(1, 50)]
        self.assertTrue(all(r['macd_dif'] is None for r in rows[:25]))
        self.assertAlmostEqual(rows[25]['macd_dif'], 7.)
        self.assertTrue(all(r['macd_hist'] is None for r in rows[:33]))
        self.assertAlmostEqual(rows[33]['macd_dea'], 7.)
        self.assertAlmostEqual(rows[33]['macd_hist'], 0.)

    def test_bbi_formula_and_warmup(self):
        state = self.state()
        rows = [state.update(float(v)) for v in range(1, 26)]
        self.assertIsNone(rows[22]['bbi'])
        expected = sum(sum(range(25-p, 25))/p for p in [3, 6, 12, 24])/4
        self.assertAlmostEqual(rows[23]['bbi'], expected)
        self.assertAlmostEqual(rows[24]['bbi_slope'], (expected+1)/expected-1)

    def test_flat_and_missing_reset(self):
        state = self.state()
        rows = [state.update(10.) for _ in range(40)]
        self.assertEqual(rows[-1]['rsi14'], 50.)
        for value in [None, float('nan'), float('inf'), 0, -1, True]:
            self.assertTrue(all(v is None for v in state.update(value).values()))
            self.assertIsNone(state.update(10.)['rsi14'])

    def test_provisional_branch_does_not_update_daily_history(self):
        state = self.state()
        for close in range(100, 140):
            state.update(float(close))
        expected = state.branch(138.)
        for close in range(140, 188):
            state.branch(float(close))
        self.assertEqual(state.branch(138.), expected)
        self.assertEqual(state.update(138.), expected)

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            MomentumState(14, (26, 12, 9), (3, 6, 12, 24))
        with self.assertRaises(ValueError):
            SeededEMA(0)

    def divergence(self, prices, oscillator, **extra):
        return confirmed_divergence(prices, oscillator, left=1, right=1, min_distance=2, max_distance=8, **extra)

    def test_bottom_confirmation_not_backdated(self):
        rows = self.divergence([5., 3., 5., 2., 5.], [40., 20., 40., 30., 40.])
        self.assertIsNone(rows[3]['bottom'])
        self.assertEqual(rows[4]['bottom'], 1)
        self.assertEqual(rows[4]['pivot_index'], 3)
        self.assertEqual(rows[4]['previous_pivot_index'], 1)

    def test_top_and_non_divergent_pair(self):
        rows = self.divergence([2., 4., 2., 5., 2.], [40., 80., 40., 70., 40.])
        self.assertEqual(rows[-1]['top'], 1)
        rows = self.divergence([5., 3., 5., 2., 5.], [40., 20., 40., 10., 40.])
        self.assertEqual(rows[-1]['bottom'], 0)

    def test_equal_pivot_not_confirmed(self):
        rows = self.divergence([5., 3., 3., 5.], [40., 20., 30., 40.])
        self.assertTrue(all(r['pivot_index'] is None for r in rows))

    def test_gap_or_missing_oscillator_never_becomes_zero(self):
        rows = self.divergence([5., 3., 5., None, 5., 2., 5.], [40., 20., 40., 40., 40., 30., 40.])
        self.assertIsNone(rows[-1]['bottom'])
        rows = self.divergence([5., 3., 5., 2., 5.], [40., None, 40., 30., 40.])
        self.assertIsNone(rows[-1]['bottom'])

    def test_divergence_prefix_future_perturbation(self):
        p = [5., 3., 5., 2., 5., 1., 6.]
        r = [40., 20., 40., 30., 40., 35., 40.]
        full = self.divergence(p, r)
        for n in range(1, len(p)+1):
            self.assertEqual(self.divergence(p[:n], r[:n]), full[:n])
        self.assertEqual(self.divergence(p[:5]+[100., 100.], r[:5]+[0., 0.])[:5], full[:5])

    def frame(self):
        rows = []
        for day in [20260105, 20260106]:
            for step in [5, 15]:
                for i, ts in enumerate(grid(day, step)):
                    rows.append(dict(symbol='TEST', frequency_minutes=step, day=day,
                                     decision_ts_assumed_end=ts, source_complete=True, close=100.+i/10))
        return pd.DataFrame(rows)

    def test_market_prefix_and_two_frequencies(self):
        frame = self.frame()
        full = build(frame)
        cutoff = 20260106101500
        prefix = build(frame[frame.decision_ts_assumed_end <= cutoff])
        pd.testing.assert_frame_equal(prefix, full[full.decision_ts_assumed_end <= cutoff].reset_index(drop=True))
        self.assertEqual(set(full.frequency_minutes), {5, 15})
        self.assertFalse(full.training_admitted.any())
        self.assertTrue(full.confirmed_divergence.isna().all())

    def test_incomplete_15m_does_not_erase_complete_5m(self):
        frame = self.frame()
        ts = 20260106101500
        frame.loc[(frame.frequency_minutes == 15) & (frame.decision_ts_assumed_end == ts), 'source_complete'] = False
        # Different completion status must not overwrite the other timeframe's branch.
        full = build(frame)
        current = full[full.decision_ts_assumed_end == ts].set_index('frequency_minutes')
        self.assertTrue(pd.isna(current.loc[15, 'price_return_from_prior_daily_close']))
        self.assertFalse(pd.isna(current.loc[5, 'price_return_from_prior_daily_close']))

    def test_all_null_warmup_prefix_keeps_numeric_schema(self):
        template = self.frame()
        template = template[template.day == 20260105]
        parts = []
        for day in pd.bdate_range('2026-01-05', periods=30):
            number = int(day.strftime('%Y%m%d'))
            part = template.copy()
            part['day'] = number
            part['decision_ts_assumed_end'] = number * 1000000 + part.decision_ts_assumed_end % 1000000
            parts.append(part)
        frame = pd.concat(parts, ignore_index=True)
        full = build(frame)
        cutoff = 20260105150000
        early = build(frame[frame.decision_ts_assumed_end <= cutoff])
        pd.testing.assert_frame_equal(early, full[full.decision_ts_assumed_end <= cutoff].reset_index(drop=True))

    def test_missing_grid_resets_intraday_and_incomplete_daily(self):
        frame = self.frame()
        frame = frame[~((frame.frequency_minutes == 5) & (frame.decision_ts_assumed_end == 20260105100500))]
        data = build(frame)
        row = data[(data.frequency_minutes == 5) & (data.decision_ts_assumed_end == 20260105101000)].iloc[0]
        self.assertTrue(pd.isna(row.intraday_rsi14))
        self.assertTrue(data[data.decision_ts_assumed_end // 1000000 == 20260106].price_return_from_prior_daily_close.isna().all())

    def test_duplicate_or_off_grid_rejected(self):
        frame = self.frame()
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            build(pd.concat([frame, frame.iloc[:1]]))
        frame.loc[0, 'decision_ts_assumed_end'] += 100
        with self.assertRaisesRegex(ValueError, 'OFF_GRID'):
            build(frame)


if __name__ == '__main__':
    unittest.main()
