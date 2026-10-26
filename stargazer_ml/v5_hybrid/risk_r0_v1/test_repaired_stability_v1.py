import unittest
import numpy as np
import pandas as pd
from diagnose_repaired_stability_v1 import TARGET,summarize,rolling,coverage_bins,shift_profile


class StabilityTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(code=['a','b','c'],day=[1,1,2],q90_adverse=[.02,.02,.04],
            q90_raw=[.01,.01,.03],**{TARGET:[.01,.03,.03]}))

    def test_date_weights_not_stock_count_weights(self):
        r=summarize(self.frame(),.025)
        self.assertAlmostEqual(r['coverage'],2/3)
        self.assertAlmostEqual(r['date_weighted_coverage'],.75)

    def test_residual_is_diagnostic_no_mutation(self):
        f=self.frame(); original=f.copy(deep=True)
        r=summarize(f,.025)
        self.assertAlmostEqual(r['residual_q90_diagnostic_only'],.016)
        pd.testing.assert_frame_equal(f,original)

    def test_short_window_not_padded(self):
        self.assertEqual(rolling(self.frame(),np.array([1,2]),.025),[])

    def test_same_date_window_for_sparse_population(self):
        windows=rolling(self.frame().iloc[:2],np.array([1,2,3]),.025,window=2)
        self.assertEqual([(r['start'],r['end'],r['rows']) for r in windows],[(1,2,2),(2,3,0)])

    def test_unknown_and_tied_coverage_bins(self):
        np.testing.assert_equal(coverage_bins(np.array([.1,.5,.8,np.nan]),[.5,.5]),[0,0,2,-1])

    def test_shift_preserves_missingness(self):
        a=pd.DataFrame(dict(x=[1.,2.,3.,np.nan])); b=pd.DataFrame(dict(x=[4.,np.nan]))
        r=shift_profile(a,b,['x'])['x']
        self.assertEqual(r['train_missing'],.25)
        self.assertEqual(r['month_missing'],.5)
        self.assertEqual(r['outside_train_range'],1.)

    def test_empty_group_null_not_zero(self):
        r=summarize(self.frame().iloc[:0],.025)
        self.assertEqual(r['rows'],0); self.assertIsNone(r['coverage'])

    def test_rolling_does_not_use_after_end_outcomes(self):
        f=self.frame(); a=rolling(f,np.array([1,2]),.025,window=1)
        f.loc[f.day.eq(2),TARGET]=.99
        self.assertEqual(a[0],rolling(f,np.array([1,2]),.025,window=1)[0])


if __name__=='__main__':unittest.main()
