import unittest
import numpy as np
import pandas as pd
from finalize_postclose_cohort_v3 import corrected_masks
from postclose_training_v1 import preprocessing,transform,pinball,train


class TrainingBoundaryTests(unittest.TestCase):
    def test_no_fit_without_explicit_authorization(self):
        with self.assertRaisesRegex(ValueError,'AUTHORIZATION_REQUIRED'):train()

    def test_training_statistics_not_changed_by_future_transform(self):
        frame=pd.DataFrame(dict(a=[1.,3.,np.nan],b=[2.,4.,6.]))
        state=preprocessing(frame);before=state.copy()
        out=transform(pd.DataFrame(dict(a=[1000.,np.nan],b=[2.,3.])),state,True)
        self.assertEqual(state,before);self.assertTrue(np.isfinite(out).all())
        self.assertEqual(out[1,2],1.)

    def test_native_nan_preserved_and_feature_order_checked(self):
        frame=pd.DataFrame(dict(a=[1.,2.],b=[2.,3.]));state=preprocessing(frame)
        x=transform(pd.DataFrame(dict(a=[np.nan],b=[2.])),state,False)
        self.assertTrue(np.isnan(x[0,0]));self.assertEqual(x[0,2],1.)
        with self.assertRaises(ValueError):transform(frame[['b','a']],state,True)

    def test_all_null_or_inf_rejected(self):
        with self.assertRaises(ValueError):preprocessing(pd.DataFrame(dict(a=[np.nan])))
        with self.assertRaises(ValueError):preprocessing(pd.DataFrame(dict(a=[np.inf])))

    def test_pinball_and_non_probability_target(self):
        self.assertAlmostEqual(pinball([.1,.2],[.2,.1]),.05)

    def test_daily_return_leader_evidence_does_not_require_rsi(self):
        f=pd.DataFrame(dict(stock_daily_rsi14=[20.],sse_daily_rsi14=[30.],cyb_daily_rsi14=[40.],
            context_sector_mean_return_loo_raw=[.01],context_current_leader_daily_rsi14_raw=[np.nan],
            context_captured_leader_daily_rsi14_raw=[np.nan],context_current_leader_return_loo_raw=[.03],
            context_captured_leader_mean_return_loo_raw=[np.nan],golden_position=[.3]))
        self.assertTrue(corrected_masks(f)['leader_daily_observation'].iloc[0])
        f['context_current_leader_return_loo_raw']=np.nan
        self.assertFalse(corrected_masks(f)['leader_daily_observation'].iloc[0])


if __name__=='__main__':unittest.main()
