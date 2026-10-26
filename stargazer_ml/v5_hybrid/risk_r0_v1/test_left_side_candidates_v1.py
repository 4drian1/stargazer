import unittest
import numpy as np
import pandas as pd
from left_side_candidates_v1 import candidates
from materialize_left_side_candidates_v1 import build, TIME, KEYS
from tdx_index_acceptance_v1 import grid


class LeftSideTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(close=[10.,9.,8.,7.], rsi14=[20.,25.,30.,35.],
            macd_dif=[-2.,-1.8,-1.5,-1.], macd_hist=[-1.,-.8,-.5,-.2], bbi=[11.,10.,9.,8.]))

    def test_current_left_signal_without_right_bar(self):
        result=candidates(self.frame().iloc[:2])
        self.assertEqual(result.iloc[-1].lag1_rsi_bottom_candidate,1.)
        self.assertEqual(result.iloc[-1].lag1_macd_bottom_candidate,1.)
        self.assertEqual(result.iloc[-1].lag1_bottom_confluence_candidate,1.)

    def test_all_lags_separate_and_warmup_unknown(self):
        result=candidates(self.frame())
        self.assertTrue(result.iloc[0].isna().all())
        self.assertTrue(pd.isna(result.iloc[1].lag3_rsi_bottom_candidate))
        self.assertEqual(result.iloc[-1].lag3_rsi_bottom_candidate,1.)
        self.assertEqual(len(result.columns),33)

    def test_top_symmetry(self):
        frame=self.frame()
        frame['close']=[7.,8.,9.,10.]
        frame['rsi14']=[80.,75.,70.,65.]
        frame['macd_dif']=[2.,1.8,1.5,1.]
        result=candidates(frame)
        self.assertEqual(result.iloc[-1].lag3_top_confluence_candidate,1.)
        self.assertEqual(result.iloc[-1].lag3_bottom_confluence_candidate,0.)

    def test_equal_not_candidate_and_bbi_not_vote(self):
        frame=self.frame()
        frame['close']=10.
        result=candidates(frame)
        self.assertEqual(result.iloc[-1].lag1_rsi_bottom_candidate,0.)
        self.assertFalse(any('bbi_bottom' in c for c in result))

    def test_missing_endpoints_and_intermediate_block_not_zero(self):
        frame=self.frame()
        frame.loc[1,'close']=np.nan
        result=candidates(frame)
        self.assertTrue(pd.isna(result.iloc[3].lag3_rsi_bottom_candidate))
        frame=self.frame();frame.loc[3,'rsi14']=np.nan
        result=candidates(frame)
        self.assertTrue(pd.isna(result.iloc[-1].lag1_bottom_confluence_candidate))
        self.assertEqual(result.iloc[-1].lag1_macd_bottom_candidate,1.)

    def test_no_three_indicator_gate(self):
        frame=self.frame();frame['bbi']=np.nan
        result=candidates(frame)
        self.assertEqual(result.iloc[-1].lag1_rsi_bottom_candidate,1.)
        self.assertEqual(result.iloc[-1].lag1_bottom_confluence_candidate,1.)

    def test_invalid_parameters_and_values(self):
        for lag in [(0,), (True,), (1,1)]:
            with self.assertRaises(ValueError):candidates(self.frame(),lag)
        for column,value in [('rsi14',101.),('bbi',-1.),('macd_dif',np.inf)]:
            frame=self.frame();frame.loc[3,column]=value
            with self.assertRaises(ValueError):candidates(frame)

    def test_full_prefix_and_future(self):
        frame=self.frame();full=candidates(frame)
        for n in range(1,len(frame)+1):
            pd.testing.assert_frame_equal(candidates(frame.iloc[:n]),full.iloc[:n])
        changed=frame.copy();changed.loc[3,'close']=100.
        pd.testing.assert_frame_equal(candidates(changed).iloc[:3],full.iloc[:3])

    def market(self):
        prices=[];momentum=[]
        for day in [20260105,20260106,20260107,20260108]:
            for step in [5,15]:
                for i,ts in enumerate(grid(day,step)):
                    row=dict(symbol='TEST',frequency_minutes=step,decision_ts_assumed_end=ts)
                    prices.append(dict(row,day=day,close=10.+i/100,source_complete=True))
                    point=row.copy()
                    for prefix in ['intraday','provisional_daily']:
                        for name,value in [('rsi14',30.+i/10),('macd_dif',-1.+i/100),('macd_hist',-.5+i/100),('bbi',10.)]:
                            point[prefix+'_'+name]=value
                    momentum.append(point)
        return pd.DataFrame(prices),pd.DataFrame(momentum)

    def test_daily_only_emitted_after_close(self):
        p,m=self.market();cutoff=20260106101500
        intra,daily=build(p[p[TIME]<=cutoff],m[m[TIME]<=cutoff])
        self.assertEqual(daily.day.tolist(),[20260105])
        self.assertEqual(set(intra.frequency_minutes),{5,15})
        self.assertFalse(intra.training_admitted.any())
        self.assertTrue((intra.right_confirmation_bars==0).all())

    def test_first_session_prefix_emits_no_postclose_row(self):
        p,m=self.market();cutoff=20260105101500
        full=build(p,m)
        before=build(p[p[TIME]<=cutoff],m[m[TIME]<=cutoff])
        self.assertTrue(before[1].empty)
        for a,b in zip(full,before):
            pd.testing.assert_frame_equal(a[a[TIME]<=cutoff].reset_index(drop=True),b)

    def test_missing_slot_not_compressed(self):
        p,m=self.market();mask=~((p.frequency_minutes==5)&(p[TIME]==20260105094500))
        p=p[mask];m=m[mask]
        intra,daily=build(p,m)
        row=intra[(intra.frequency_minutes==5)&(intra[TIME]==20260105095000)].iloc[0]
        self.assertTrue(pd.isna(row.lag1_rsi_bottom_candidate))
        self.assertTrue(daily[daily.day==20260105].filter(regex='^lag').isna().all().all())

    def test_market_prefix_and_no_backfill(self):
        p,m=self.market();full=build(p,m);cutoff=20260106101500
        prefix=build(p[p[TIME]<=cutoff],m[m[TIME]<=cutoff])
        for a,b in zip(full,prefix):
            pd.testing.assert_frame_equal(a[a[TIME]<=cutoff].reset_index(drop=True),b)

    def test_mismatched_indicator_cutoff_rejected(self):
        p,m=self.market();m.loc[0,TIME]+=100
        with self.assertRaisesRegex(ValueError,'CUTOFF'):build(p,m)


if __name__=='__main__':unittest.main()
