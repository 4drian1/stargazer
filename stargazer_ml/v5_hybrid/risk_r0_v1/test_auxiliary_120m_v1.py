"""Synthetic strict-future, clock, normalization and missing-window checks."""
import unittest

import numpy as np

from build_auxiliary_120m_v1 import auxiliary_values,minute_clock


class AuxiliaryTests(unittest.TestCase):
    def grid(self,n_days=2):
        return np.full((n_days*240,4),100.)

    def test_exact_120_and_excludes_current(self):
        g=self.grid();g[14,2]=1.;g[134,2]=95.;g[135,2]=10.
        signed,drop,end,valid=auxiliary_values(g,[14],[1.,1.])
        self.assertAlmostEqual(signed[0],-.05)
        self.assertAlmostEqual(drop[0],.05)
        self.assertEqual(end[0],134);self.assertTrue(valid[0])

    def test_first_future_included(self):
        g=self.grid();g[15,2]=90.
        self.assertAlmostEqual(auxiliary_values(g,[14],[1.,1.])[1][0],.1)

    def test_last_future_included(self):
        g=self.grid();g[134,2]=80.
        self.assertAlmostEqual(auxiliary_values(g,[14],[1.,1.])[1][0],.2)

    def test_all_future_above_reference_zero_downside(self):
        g=self.grid();g[15:135,2]=101.
        signed,drop,_,valid=auxiliary_values(g,[14],[1.,1.])
        self.assertAlmostEqual(signed[0],.01)
        self.assertEqual(drop[0],0);self.assertTrue(valid[0])

    def test_missing_middle_not_ignored(self):
        g=self.grid();g[50,2]=np.nan
        signed,drop,_,valid=auxiliary_values(g,[14],[1.,1.])
        self.assertTrue(np.isnan(signed[0]));self.assertTrue(np.isnan(drop[0]));self.assertFalse(valid[0])

    def test_calendar_tail_not_truncated(self):
        signed,drop,_,valid=auxiliary_values(self.grid(),[470],[1.,1.])
        self.assertFalse(valid[0]);self.assertTrue(np.isnan(drop[0]));self.assertTrue(np.isnan(signed[0]))

    def test_factor_direction_cross_day(self):
        g=self.grid();g[240:]=50.
        signed,drop,_,valid=auxiliary_values(g,[239],[1.,2.])
        self.assertTrue(valid[0]);self.assertAlmostEqual(signed[0],0);self.assertAlmostEqual(drop[0],0)

    def test_real_downside_survives_adjustment(self):
        g=self.grid();g[240:]=45.
        self.assertAlmostEqual(auxiliary_values(g,[239],[1.,2.])[1][0],.1)

    def test_invalid_factors(self):
        for bad in [0,-1,np.nan,np.inf]:
            with self.assertRaisesRegex(ValueError,'INVALID_FACTOR'):
                auxiliary_values(self.grid(),[14],[1.,bad])

    def test_lunch_skipped(self):
        clock=minute_clock([20260105,20260106])
        self.assertEqual(clock[119],20260105113000)
        self.assertEqual(clock[120],20260105130100)
        self.assertEqual(clock[119+120],20260105150000)

    def test_late_day_continues_next_session(self):
        clock=minute_clock([20260105,20260106])
        self.assertEqual(clock[239],20260105150000)
        self.assertEqual(clock[239+120],20260106113000)

    def test_future_perturbation_after_window_no_change(self):
        g=self.grid();a=auxiliary_values(g,[14],[1.,1.])[1]
        g[135:,2]=1.
        np.testing.assert_array_equal(a,auxiliary_values(g,[14],[1.,1.])[1])

    def test_invalid_position(self):
        with self.assertRaisesRegex(ValueError,'REFERENCE_OUTSIDE'):
            auxiliary_values(self.grid(),[-1],[1.,1.])

    def test_invalid_grid(self):
        with self.assertRaisesRegex(ValueError,'INVALID_AUXILIARY_GRID'):
            auxiliary_values(self.grid(),[14],[1.])
        with self.assertRaisesRegex(ValueError,'INVALID_AUXILIARY_GRID'):
            auxiliary_values(self.grid(),[14],[1.,1.],horizon=True)


if __name__=='__main__':unittest.main()
