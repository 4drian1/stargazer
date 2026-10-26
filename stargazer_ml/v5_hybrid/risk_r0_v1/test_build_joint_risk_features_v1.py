import unittest
import numpy as np
import pandas as pd

from build_joint_risk_features_v1 import stock_features, wilder_rsi, basket_features, observed_leaders
from check_ths_cache_slots_v1 import grid


class JointFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.days=np.array([20250102,20250103,20250106])
        rows=[]
        for i,day in enumerate(self.days):
            for j,ts in enumerate(sorted(grid(int(day),'END'))):
                c=10+i+j/10000
                rows.append(dict(code='000001',date=ts,open=c,high=c+.01,low=c-.01,
                                 close=c,volume=100.,amount=1000.))
        self.frame=pd.DataFrame(rows)

    def test_stock_same_cutoff_two_frequencies(self):
        frame,returns,_=stock_features(self.frame,'000001',self.days,set())
        self.assertEqual(len(frame),48)
        self.assertTrue(np.isnan(returns[:16]).all())
        self.assertFalse(np.isnan(returns[16:]).any())
        self.assertIn('stock_rsi14_5m_raw',frame)
        self.assertIn('stock_rsi14_15m_raw',frame)
        self.assertFalse(frame['stock_corporate_action_scale_verified'].any())

    def test_raw_prior_close_and_no_current_day_final_close(self):
        frame,_,_=stock_features(self.frame,'000001',self.days,set())
        r=frame[frame.cutoff_index==16].iloc[0]
        self.assertAlmostEqual(r.stock_reference_close_raw,10.0239)
        self.assertAlmostEqual(r.stock_day_return_raw,r.stock_close/10.0239-1)

    def test_future_perturbation(self):
        frame,_,_=stock_features(self.frame,'000001',self.days,set())
        changed=self.frame.copy()
        chosen=changed.date>20250103100000
        for col in ['open','high','low','close']:
            changed.loc[chosen,col]*=2
        other,_,_=stock_features(changed,'000001',self.days,set())
        pd.testing.assert_frame_equal(frame[frame.cutoff_index<=17].reset_index(drop=True),
                                      other[other.cutoff_index<=17].reset_index(drop=True))

    def test_missing_and_wrong_stock_not_filled(self):
        missing=self.frame.drop(index=0)
        frame,_,_=stock_features(missing,'000001',self.days,set())
        self.assertNotIn(0,frame.cutoff_index.tolist())
        wrong=self.frame.copy();wrong.loc[0,'code']='000002'
        with self.assertRaisesRegex(ValueError,'IDENTITY|CODE'):
            stock_features(wrong,'000001',self.days,set())

    def test_conflicts_and_quarantine_excluded(self):
        extra=self.frame.iloc[[0]].copy();extra['close']+=.001
        conflicted=pd.concat([self.frame,extra],ignore_index=True)
        frame,_,quality=stock_features(conflicted,'000001',self.days,{20250103})
        self.assertNotIn(0,frame.cutoff_index.tolist())
        self.assertFalse(((frame.cutoff_index>=16)&(frame.cutoff_index<32)).any())
        self.assertEqual(quality['conflicting_timestamps'],1)

    def test_illegal_clock_is_not_shifted(self):
        frame=self.frame.copy();frame.loc[0,'date']=20250102099900
        result,_,_=stock_features(frame,'000001',self.days,set())
        self.assertNotIn(0,result.cutoff_index.tolist())

    def test_two_pass_values_identical_and_positions_not_dataframe_index(self):
        frame,returns,_=stock_features(self.frame,'000001',self.days,set())
        _,only,_=stock_features(self.frame,'000001',self.days,set(),returns_only=True)
        np.testing.assert_allclose(returns,only,equal_nan=True)
        sparse,_,_=stock_features(self.frame.iloc[240:],'000001',self.days,set())
        self.assertEqual(sparse.index.tolist(),list(range(len(sparse))))
        self.assertEqual(sparse.cutoff_index.iloc[0],16)

    def test_empty_source_has_no_synthetic_rows(self):
        frame,returns,quality=stock_features(self.frame.iloc[:0],'000001',self.days,set())
        self.assertEqual(len(frame),0)
        self.assertTrue(np.isnan(returns).all())
        self.assertEqual(quality['feature_rows'],0)

    def test_rsi_flat_and_gap_reset(self):
        self.assertEqual(wilder_rsi([10.]*15)[-1],50.)
        result=wilder_rsi([10.]*15+[np.nan,11.,12.])
        self.assertTrue(np.isnan(result[-1]))
        self.assertEqual(wilder_rsi(np.arange(20.))[-1],100.)

    def test_partial_basket_no_zero_fill_and_common_members(self):
        returns=np.array([[.01,.02],[np.nan,.5]])
        fields=basket_features(returns,[0,1])
        self.assertEqual(fields['observed_return_raw'][0],.01)
        self.assertEqual(fields['coverage'][0],.5)
        self.assertAlmostEqual(fields['common_member_return_change'][1],.01)

    def test_observed_rank_is_not_complete_rank(self):
        stocks=np.array([[.1,.05],[.0,.2]])
        boards=np.array([[.1,.05],[.0,.2]])
        counts=np.array([[1,1],[1,1]])
        result=observed_leaders(boards,counts,np.array([2,2]),[[0],[1]],stocks,['A','B'],['000001','000002'])
        self.assertEqual(result.leader_observed_stock.tolist(),['000001','000002'])
        self.assertTrue(result.leader_strict_board.isna().all())
        self.assertEqual(result.captured_stock_count_today.tolist(),[1,2])
        self.assertAlmostEqual(result.iloc[1].captured_stock_mean_return_raw,.125)

    def test_tie_not_broken_arbitrarily(self):
        returns=np.array([[.1],[.1]])
        result=observed_leaders(returns,np.ones((2,1)),np.ones(2),[[0],[1]],returns,['A','B'],['000001','000002'])
        self.assertIsNone(result.iloc[0].leader_observed_board)


if __name__=='__main__':
    unittest.main()
