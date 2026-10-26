import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from frozen_observation_sensitivity_v1 import mapped_inputs, compare
from multiboard_three_arm_fit_v1 import replay
from prepare_multiboard_comparison_v1 import TIME


class SensitivityTests(unittest.TestCase):
    def test_only_frozen_fields_mapped_and_input_unchanged(self):
        f=pd.DataFrame(dict(a=[.1,np.nan],ar_a=[.2,.3],nh_unused=[999.,999.],y_tail_1200m=[0,1]))
        before=f.copy(deep=True)
        old,new,mapping,changed=mapped_inputs(f,['a'])
        self.assertEqual(mapping,dict(a='ar_a'))
        np.testing.assert_allclose(new.a,[.2,.3])
        self.assertEqual(int(changed.sum()),2)
        pd.testing.assert_frame_equal(f,before)
        self.assertTrue(np.isnan(old.a.iloc[1]))

    def test_new_warning_or_labels_not_silently_whitelisted(self):
        for name in ['nh_warning','ar_a','y_tail_1200m','label_end_time']:
            with self.assertRaisesRegex(ValueError,'WHITELIST'):
                mapped_inputs(pd.DataFrame({name:[1]}),[name])

    def test_known_input_infinity_refused_and_no_alias_unchanged(self):
        with self.assertRaisesRegex(ValueError,'INFINITE'):
            mapped_inputs(pd.DataFrame({'a':[np.inf]}),['a'])
        old,new,mapping,changed=mapped_inputs(pd.DataFrame({'a':[np.nan,1.]}),['a'])
        pd.testing.assert_frame_equal(old,new)
        self.assertFalse(changed.any());self.assertEqual(mapping,{})

    def test_replay_uses_saved_statistics_without_fit_and_retains_negative_calibration(self):
        f=pd.DataFrame({'a':[0.,1.,np.nan]})
        state=dict(columns=['a'],median=[0.],mean=[0.],scale=[1.])
        model=dict(coef=[1.,0.],intercept=0.,classes=[0,1])
        calibration=dict(coef=[-1.],intercept=0.,classes=[0,1])
        with patch('sklearn.linear_model.LogisticRegression.fit',side_effect=AssertionError('FIT_FORBIDDEN')):
            result=replay({'test':f},['a'],'lr',state,model,calibration,1e-12)['test']
        self.assertGreater(result['raw'][1],result['raw'][0])
        self.assertLess(result['calibrated'][1],result['calibrated'][0])

    def test_saved_column_order_enforced(self):
        f=pd.DataFrame({'a':[1.]})
        with self.assertRaisesRegex(ValueError,'FEATURE_ORDER'):
            replay({'test':f},['a'],'lr',dict(columns=['b'],median=[0.],mean=[0.],scale=[1.]),
                   dict(coef=[1.,0.],intercept=0.),dict(coef=[1.],intercept=0.),1e-12)

    def test_label_change_never_changes_predictions(self):
        state=dict(columns=['a'],median=[0.],mean=[0.],scale=[1.])
        model=dict(coef=[1.,0.],intercept=0.)
        calibration=dict(coef=[1.],intercept=0.)
        f=pd.DataFrame(dict(a=[0.,1.],y_tail_1200m=[0,1]))
        a=replay({'test':f},['a'],'lr',state,model,calibration,1e-12)
        f.y_tail_1200m=1-f.y_tail_1200m
        b=replay({'test':f},['a'],'lr',state,model,calibration,1e-12)
        np.testing.assert_array_equal(a['test']['raw'],b['test']['raw'])

    def test_identical_scores_have_zero_paired_difference_and_same_filter(self):
        frame=pd.DataFrame({'code':['A','B']*12,TIME:np.repeat(np.arange(20250102,20250114)*1000000+100000,2),
                            'y_tail_1200m':[0,1]*12})
        p=np.tile([.2,.7],12)
        policy=dict(rejection_fraction=.2,minimum_cross_section=5,
                    uncertainty=dict(block_dates=10,reps=20,seed=42))
        result=compare(frame,p,p.copy(),policy)
        self.assertEqual(result['changed_probability_rows'],0)
        self.assertEqual(result['paired_loss']['brier']['left_minus_right'],0.)
        self.assertEqual(result['same_cutoff_filter']['comparable_rows'],0)


if __name__=='__main__':
    unittest.main()
