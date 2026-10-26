import unittest
import numpy as np
import pandas as pd
from postclose_next_session_core_v1 import stock_days


class PostcloseCoreTests(unittest.TestCase):
    def setUp(self):
        self.days=np.array([int(d.strftime('%Y%m%d')) for d in pd.bdate_range('2025-01-02',periods=45)])
        self.records=[]
        for i,d in enumerate(self.days):
            c=10.+i/100
            self.records.append((f'日k:TEST:{d}',dict(code='TEST',date=int(d),open=c,high=c+.1,
                low=c-.1,close=c,volume=100.+i,amount=1000.+i)))
        self.audit=dict(events=[],issues=[])

    def test_strict_next_session_not_current_low(self):
        records=[(k,dict(v)) for k,v in self.records]
        records[0][1]['low']=1.
        result=stock_days('TEST',records,self.days,self.audit)
        self.assertAlmostEqual(result.next_session_adverse_excursion.iloc[0],1-9.91/10)
        self.assertTrue(pd.isna(result.next_session_adverse_excursion.iloc[-1]))
        self.assertFalse(result.training_eligible.any())

    def test_above_reference_low_has_zero_not_negative_risk(self):
        records=[(k,dict(v)) for k,v in self.records]
        records[1][1].update(open=11.,low=10.5,high=11.5,close=11.)
        result=stock_days('TEST',records,self.days,self.audit)
        self.assertEqual(result.next_session_adverse_excursion.iloc[0],0.)
        self.assertAlmostEqual(result.next_session_low_return.iloc[0],.05)

    def test_action_normalization_and_unresolved_isolation(self):
        records=[(k,dict(v)) for k,v in self.records]
        records[1][1].update(open=5.,low=4.9,high=5.1,close=5.)
        audit=dict(events=[dict(date=int(self.days[1]),multiplier=2.,status='ORDINARY_PARAMETERS_CORROBORATED')],issues=[])
        result=stock_days('TEST',records,self.days,audit)
        self.assertAlmostEqual(result.next_session_adverse_excursion.iloc[0],.02)
        audit['issues']=[dict(date=int(self.days[1]),reason='SPECIAL')]
        self.assertTrue(pd.isna(stock_days('TEST',records,self.days,audit).next_session_adverse_excursion.iloc[0]))

    def test_missing_next_day_not_skipped_and_not_zero(self):
        result=stock_days('TEST',self.records[:1]+self.records[2:],self.days,self.audit)
        self.assertTrue(pd.isna(result.next_session_adverse_excursion.iloc[0]))
        self.assertTrue(pd.isna(result.stock_return1d.iloc[2]))

    def test_invalid_ohlc_and_duplicate_conflict(self):
        records=[(k,dict(v)) for k,v in self.records]
        records[1][1]['close']=100.
        self.assertTrue(pd.isna(stock_days('TEST',records,self.days,self.audit).next_session_adverse_excursion.iloc[0]))
        conflicting=(self.records[1][0],dict(self.records[1][1],close=10.02))
        self.assertTrue(pd.isna(stock_days('TEST',self.records+[conflicting],self.days,self.audit).next_session_adverse_excursion.iloc[0]))

    def test_undated_issue_is_not_ignored(self):
        result=stock_days('TEST',self.records,self.days,dict(events=[],issues=[dict(date=None)]))
        self.assertTrue(result.next_session_adverse_excursion.isna().all())
        self.assertTrue(result.stock_daily_rsi14.isna().all())

    def test_features_prefix_invariant_targets_separate(self):
        full=stock_days('TEST',self.records,self.days,self.audit)
        early=stock_days('TEST',self.records[:25],self.days[:25],self.audit)
        features=[c for c in full if c.startswith('stock_')]
        pd.testing.assert_frame_equal(full.iloc[:25][features],early[features])
        self.assertTrue(pd.isna(early.next_session_adverse_excursion.iloc[-1]))
        self.assertFalse(pd.isna(full.next_session_adverse_excursion.iloc[24]))

    def test_invalid_factors_or_identity_rejected(self):
        with self.assertRaises(ValueError):stock_days('WRONG',self.records,self.days,self.audit)
        with self.assertRaises(ValueError):stock_days('TEST',self.records,self.days[::-1],self.audit)
        audit=dict(events=[dict(date=int(self.days[1]),multiplier=-1.,status='ORDINARY_PARAMETERS_CORROBORATED')],issues=[])
        with self.assertRaises(ValueError):stock_days('TEST',self.records,self.days,audit)


if __name__=='__main__':unittest.main()
