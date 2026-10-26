import unittest
import numpy as np
import pandas as pd

from joint_fit_preparation_v1 import (whitelist,derive,group_observation_masks,fit_lr_preprocessing,
                                      transform_lr,native_missing_view,cutoff_support)


class PreparationTests(unittest.TestCase):
    def test_future_label_and_code_forbidden(self):
        for name in ['code','stock_close','y_tail_1200m','diagnostic_tail_raw_1200m','label_state','sse_15m_golden_level_0_5']:
            with self.assertRaisesRegex(ValueError,'FORBIDDEN'):whitelist({'feature_groups':{'a':[name]}})

    def test_duplicate_field_rejected(self):
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):whitelist({'feature_groups':{'a':['x'],'b':['x']}})

    def test_only_train_can_fit(self):
        for phase in ['test','select','calibration']:
            with self.assertRaisesRegex(ValueError,'TRAIN_ONLY'):fit_lr_preprocessing(pd.DataFrame({'a':[1]}),phase)

    def test_future_extremes_do_not_change_training_statistics(self):
        train=pd.DataFrame({'a':[1.,3.,5.],'b':[np.nan,5.,7.]})
        state=fit_lr_preprocessing(train);median=state['median'].copy()
        test=pd.DataFrame({'a':[1e9],'b':[np.nan]});out=transform_lr(test,state)
        np.testing.assert_array_equal(state['median'],median)
        self.assertEqual(out[0,-1],1);self.assertEqual(out.shape,(1,4))

    def test_all_null_train_column_stops_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError,'ALL_NULL'):fit_lr_preprocessing(pd.DataFrame({'a':[np.nan]}))

    def test_constant_train_safe_scale(self):
        state=fit_lr_preprocessing(pd.DataFrame({'a':[2.,2.]}))
        self.assertEqual(state['scale'][0],1.)

    def test_inf_rejected_not_filled(self):
        with self.assertRaisesRegex(ValueError,'INFINITE'):native_missing_view(pd.DataFrame({'a':[np.inf]}))

    def test_native_nan_kept_and_indicated(self):
        x=native_missing_view(pd.DataFrame({'a':[np.nan]}))
        self.assertTrue(np.isnan(x[0,0]));self.assertEqual(x[0,1],1.)

    def test_feature_reordering_stops(self):
        frame=pd.DataFrame({'a':[1.],'b':[2.]});state=fit_lr_preprocessing(frame)
        with self.assertRaisesRegex(ValueError,'ORDER'):transform_lr(frame[['b','a']],state)

    def test_whole_group_missing_is_not_complete(self):
        frame=pd.DataFrame({'a':[np.nan,1.],'b':[np.nan,np.nan]})
        self.assertEqual(group_observation_masks(frame,{'g':['a','b']})['g'].tolist(),[False,True])

    def test_tiny_cutoffs_do_not_silently_pass_ranking(self):
        r=cutoff_support([1]*4+[2]*5)
        self.assertEqual(r['comparable_rows'],5);self.assertEqual(r['tiny_cutoffs'],1)

    def test_time_and_relative_features_use_current_inputs_only(self):
        frame=pd.DataFrame(dict(decision_ts_assumed_end=[20250102131500],stock_day_return_raw=[.01],
            context_sector_mean_return_loo_raw=[.02],context_current_leader_return_loo_raw=[.03],
            sse_15m_day_return_prev_close=[.005]))
        for ratio in ['0_236','0_382','0_5','0_618','0_786']:frame['sse_15m_time_fraction_distance_'+ratio]=np.nan
        out=derive(frame)
        self.assertTrue(pd.isna(out.time_node_nearest_abs_distance.iloc[0]))
        self.assertEqual(out.session_is_pm.iloc[0],1)
        self.assertAlmostEqual(out.context_stock_minus_leader_return_raw.iloc[0],-.02)


if __name__=='__main__':unittest.main()
