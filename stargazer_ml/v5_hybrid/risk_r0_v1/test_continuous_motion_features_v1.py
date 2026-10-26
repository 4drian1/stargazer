import unittest

import numpy as np
import pandas as pd

from continuous_motion_features_v1 import entity_motion, relative_motion, summarize_motion
from intraday_relative_stall_v1 import TIME, slots


def path(returns=(.003, .002, .001), frequency=5, code='A'):
    n = len(returns)
    return pd.DataFrame(dict(code=[code]*n, **{TIME: slots(20250102, frequency)[:n]},
        interval_return=returns, close_drawdown=np.zeros(n),
        interval_known=[True]*n, peak_known=[True]*n))


class MotionTests(unittest.TestCase):
    def test_still_positive_but_slowing(self):
        f = entity_motion(path(), ['code'], 5)
        self.assertGreater(f.motion_interval_return.iloc[-1], 0)
        self.assertAlmostEqual(f.motion_deceleration.iloc[-1], .001)

    def test_normal_strengthening_not_warning(self):
        f = entity_motion(path((.001,.002,.003)), ['code'], 5)
        self.assertLess(f.motion_deceleration.iloc[-1], 0)

    def test_negative_flat_and_small_positive_not_collapsed(self):
        f = entity_motion(path((-.001,0,.000001)), ['code'], 5)
        np.testing.assert_equal(f.motion_interval_return.to_numpy(), [-.001,0,.000001])

    def test_giveback_is_already_observed_peak_not_future(self):
        p = path(); p['close_drawdown'] = [0,-.01,-.02]
        f = entity_motion(p, ['code'], 5)
        np.testing.assert_equal(f.motion_giveback.to_numpy(), [0,.01,.02])

    def test_unknown_not_zero(self):
        p = path(); p.loc[1,'interval_known'] = False; p.loc[2,'peak_known'] = False
        f = entity_motion(p, ['code'], 5)
        self.assertTrue(np.isnan(f.motion_interval_return.iloc[1]))
        self.assertTrue(f.motion_deceleration.isna().all())
        self.assertTrue(np.isnan(f.motion_giveback.iloc[2]))

    def test_no_cross_entity_acceleration(self):
        p = pd.concat([path(code='A'),path((.01,.02,.03),code='B')])
        f = entity_motion(p, ['code'], 5)
        self.assertAlmostEqual(f[f.code.eq('A')].motion_deceleration.iloc[-1], .001)
        self.assertAlmostEqual(f[f.code.eq('B')].motion_deceleration.iloc[-1], -.01)

    def test_leader_switch_not_mean_difference(self):
        # A leads first, B leads last: compute A/B paths, then select winners.
        a = path((.004,.003,.002),code='A'); b = path((.001,.006,.005),code='B')
        f = entity_motion(pd.concat([a,b]), ['code'], 5)
        last_b = f[f.code.eq('B')].iloc[-1]
        self.assertAlmostEqual(last_b.motion_deceleration, .001)
        self.assertNotAlmostEqual(last_b.motion_deceleration, .003-.005)

    def test_lunch_boundary_and_gap(self):
        p = path(); p[TIME] = [20250102113000,20250102130500,20250102131000]
        f = entity_motion(p, ['code'], 5)
        self.assertTrue(np.isnan(f.motion_deceleration.iloc[1]))
        self.assertAlmostEqual(f.motion_deceleration.iloc[2], .001)
        p = path(); p[TIME] = [20250102093500,20250102094500,20250102095000]
        self.assertTrue(np.isnan(entity_motion(p,['code'],5).motion_deceleration.iloc[1]))

    def test_day_reset(self):
        p = path(); p[TIME] = [20250102145500,20250102150000,20250103093500]
        self.assertTrue(np.isnan(entity_motion(p,['code'],5).motion_deceleration.iloc[-1]))

    def test_both_frequencies_prefix_and_future_perturbation(self):
        for freq in (5,15):
            p = path((.004,.003,.002,.001),freq)
            original = entity_motion(p,['code'],freq)
            short = entity_motion(p.iloc[:3],['code'],freq)
            changed = p.copy(); changed.loc[3,['interval_return','close_drawdown']] = [.5,-.5]
            pd.testing.assert_frame_equal(original.iloc[:3],short)
            pd.testing.assert_frame_equal(original.iloc[:3],entity_motion(changed,['code'],freq).iloc[:3])

    def test_market_difference_near_zero_and_relative_slowing(self):
        sector = entity_motion(path(),['code'],5)
        market = entity_motion(path((.001,.001,0)),['code'],5)
        f = relative_motion(sector,market,['code'])
        self.assertAlmostEqual(f.motion_relative_return.iloc[-1],.001)
        self.assertAlmostEqual(f.motion_relative_deceleration.iloc[-1],0)
        self.assertTrue(np.isfinite(f.motion_relative_return).all())

    def test_weak_board_can_still_rise(self):
        a=entity_motion(path((.001,.001,.001)),['code'],5)
        b=entity_motion(path((.003,.003,.003)),['code'],5)
        self.assertAlmostEqual(relative_motion(a,b,['code']).motion_relative_return.iloc[-1],-.002)

    def test_missing_market_no_asof_fill(self):
        a=entity_motion(path(),['code'],5)
        f=relative_motion(a,a.iloc[:2],['code'])
        self.assertTrue(np.isnan(f.motion_relative_return.iloc[-1]))

    def test_summary_empty_unknown_denominators(self):
        f=relative_motion(entity_motion(path(),['code'],5),entity_motion(path(),['code'],5),['code'])
        f.loc[1,'motion_relative_return']=np.nan
        s=summarize_motion(f,np.ones(3,bool))
        self.assertEqual(s['relative_return']['unknown'],1)
        self.assertEqual(s['relative_return']['coverage'],2/3)
        self.assertIsNone(summarize_motion(f,np.zeros(3,bool))['relative_return']['mean'])

    def test_invalid_values_keys_and_grid(self):
        for bad in ('inf','bad_drawdown','duplicate','offgrid','flag'):
            p=path()
            if bad=='inf':p.loc[1,'interval_return']=np.inf
            elif bad=='bad_drawdown':p.loc[1,'close_drawdown']=.01
            elif bad=='duplicate':p.loc[1,TIME]=p.loc[0,TIME]
            elif bad=='offgrid':p.loc[1,TIME]=20250102093600
            else:p['interval_known']=1
            with self.assertRaises(ValueError,msg=bad):entity_motion(p,['code'],5)


if __name__ == '__main__':
    unittest.main()
