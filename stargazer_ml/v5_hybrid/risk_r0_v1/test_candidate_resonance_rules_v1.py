import unittest
import numpy as np

from candidate_resonance_rules_v1 import (assess, known_wave, market_regime,
                                         market_location, first_confirmation, exit_reference)


class CandidateRuleTests(unittest.TestCase):
    def bars(self):
        return np.tile([100., 102., 98., 100., 100.], (100, 1))

    def day(self, date, **kwargs):
        return dict(date=date, open=100., high=102., low=98., close=100., can_sell=True, **kwargs)

    def test_environment_boundaries_and_gaps(self):
        for r in (-.05, 0, .08):
            self.assertEqual(market_regime(r), 'ALLOW')
        for r in (-.08, -.06, .09, .10, .11):
            self.assertEqual(market_regime(r), 'WATCH_ONLY')
        self.assertEqual(market_regime(-.081), 'MAIN_DECLINE_BLOCK')
        with self.assertRaises(ValueError): market_regime(float('nan'))

    def test_all_five_layers_positive_then_no_repeat(self):
        x, m = self.bars(), self.bars()
        x[20] = [70, 71, 60, 70, 100]; x[55, 1] = 110
        x[70] = [95, 96, 84, 85, 200]
        x[71] = [85, 86, 80, 82, 120]
        x[72] = [83, 84, 81, 82, 110]
        x[73] = [82, 85, 81, 84, 180]
        m[40] = [90, 91, 80, 90, 100]; m[55, 1] = 120
        r = assess(x, m, 73)
        self.assertTrue(r['candidate'])
        self.assertTrue(all(r['components'].values()))
        self.assertEqual(assess(x, m, 74)['state'], 'ALREADY_CONFIRMED')

    def test_anchor_strictly_before_decision(self):
        x = self.bars(); x[40] = [90, 91, 80, 90, 100]; x[55, 1] = 120
        w = known_wave(x, 70)
        self.assertEqual((w['a'], w['b']), (40, 55))
        x[70:] *= 20
        self.assertEqual(w, known_wave(x, 70))

    def test_midpoint_and_extension_arithmetic(self):
        x = self.bars(); x[40] = [90, 91, 80, 90, 100]; x[55, 1] = 120
        self.assertEqual(market_location(x, 70)['ratio'], .5)
        x[70] = [69.12, 70, 68, 69.12, 100]
        self.assertEqual(market_location(x, 70)['ratio'], 1.272)

    def test_first_confirmation_and_expiry(self):
        x = self.bars(); x[72] = [99, 102, 98, 101, 150]; x[74] = [99, 102, 98, 101, 200]
        self.assertEqual(first_confirmation(x, 70, 75), 72)
        self.assertIsNone(first_confirmation(x, 72, 73))
        self.assertIsNone(first_confirmation(x, 80, 90))

    def test_prefix_and_future_perturbation(self):
        x, m = self.bars(), self.bars()
        x[70] = [100, 101, 94, 95, 200]; x[71] = [95, 96, 92, 94, 120]
        x[73] = [94, 97, 93, 96, 180]
        expected = assess(x, m, 73)
        self.assertEqual(expected, assess(x[:74], m[:74], 73))
        x[74:] *= 100; m[74:] *= .01
        self.assertEqual(expected, assess(x, m, 73))

    def test_unknown_is_not_no_pattern(self):
        x = self.bars(); x[50, 2] = np.nan
        self.assertIsNone(assess(x, self.bars(), 73)['candidate'])
        self.assertFalse(assess(self.bars(), self.bars(), 73)['candidate'])

    def test_t1_no_entry_day_sale(self):
        a = self.day(20260105); a.update(high=110, low=90)
        r = exit_reference(100, 80, 20260102, [a, self.day(20260106)])
        self.assertEqual(r['state'], 'CENSORED_TAIL')
        self.assertIsNone(r['net'])

    def test_dual_touch_is_unknown(self):
        b = self.day(20260106); b.update(high=110, low=90)
        r = exit_reference(100, 80, 20260102, [self.day(20260105), b])
        self.assertEqual(r['state'], 'AMBIGUOUS_INTRADAY_ORDER')
        self.assertIsNone(r['reference_gross_return'])

    def test_gap_stop_not_fake_five_percent(self):
        b = self.day(20260106); b.update(open=90, high=92, low=88, close=90)
        r = exit_reference(100, 80, 20260102, [self.day(20260105), b])
        self.assertEqual(r['state'], 'GAP_STOP_REFERENCE')
        self.assertAlmostEqual(r['reference_gross_return'], -.1)

    def test_close_break_waits_until_next_open(self):
        b = self.day(20260106); b.update(close=99)
        r = exit_reference(100, 99.5, 20260102, [self.day(20260105), b, self.day(20260107)])
        self.assertEqual((r['state'], r['exit_date']), ('CLOSE_BREAK_NEXT_OPEN', 20260107))

    def test_unknown_metadata_not_zero_loss(self):
        b = self.day(20260106); b.pop('can_sell')
        r = exit_reference(100, 80, 20260102, [self.day(20260105), b])
        self.assertEqual(r['state'], 'METADATA_UNKNOWN')
        self.assertIsNone(r['reference_gross_return'])

    def test_unclosed_and_time_exit(self):
        dates = [20260105, 20260106, 20260107, 20260108, 20260109,
                 20260112, 20260113, 20260114, 20260115, 20260116]
        days = [self.day(d) for d in dates]
        self.assertEqual(exit_reference(100, 80, 20260102, days)['state'], 'TIME_EXIT_REFERENCE')
        for d in days[1:]: d['can_sell'] = False
        r = exit_reference(100, 80, 20260102, days)
        self.assertEqual(r['state'], 'POSITION_UNCLOSED')
        self.assertIsNone(r['reference_gross_return'])


if __name__ == '__main__':
    unittest.main()
