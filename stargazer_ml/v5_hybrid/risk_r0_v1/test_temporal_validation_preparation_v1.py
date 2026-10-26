import unittest
import numpy as np
import pandas as pd
from prepare_postclose_temporal_validation_v1 import assign_segments


class TemporalPreparationTests(unittest.TestCase):
    def data(self):
        return pd.DataFrame(dict(code=['a']*4,day=[20250130,20250131,20250203,20250204],
            decision_ts_assumed_end=[20250130150000,20250131150000,20250203150000,20250204150000],
            label_end_time=[20250131150000,20250203150000,20250204150000,20250205150000],
            proposed_purged=[True,False,False,False]))

    def bounds(self):
        return dict(train=[20250101,20250131],select=[20250201,20250228],calibration=[20250301,20250331],test=[20250401,20250430])

    def test_full_timestamp_purge_and_no_move(self):
        d=self.data();a=assign_segments(d,self.bounds(),pd.Series(True,index=d.index))
        self.assertTrue(a.loc[1,'purged']);self.assertEqual(a.loc[1,'segment'],'train')
        self.assertEqual(a.loc[1,'reason'],'CROSS_SEGMENT_LABEL')

    def test_original_fold_purge_not_inherited(self):
        d=self.data();a=assign_segments(d,self.bounds(),pd.Series(True,index=d.index))
        self.assertTrue(a.loc[0,'eligible'])

    def test_unknown_end_not_zero_or_eligible(self):
        d=self.data();d.loc[0,'label_end_time']=np.nan
        a=assign_segments(d,self.bounds(),pd.Series(True,index=d.index))
        self.assertFalse(a.loc[0,'eligible']);self.assertEqual(a.loc[0,'reason'],'INVALID_OR_UNKNOWN_LABEL_END')

    def test_missing_family_still_blocks(self):
        d=self.data();e=pd.Series(True,index=d.index);e.iloc[0]=False
        self.assertFalse(assign_segments(d,self.bounds(),e).loc[0,'eligible'])

    def test_overlapping_dates_refused(self):
        b=self.bounds();b['select'][0]=20250131;d=self.data()
        with self.assertRaisesRegex(ValueError,'OVERLAPPING'):assign_segments(d,b,pd.Series(True,index=d.index))

    def test_duplicate_population_refused(self):
        d=self.data();d=pd.concat([d,d])
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):assign_segments(d,self.bounds(),pd.Series(True,index=d.index))

    def test_label_not_future_refused_from_eligibility(self):
        d=self.data();d.loc[0,'label_end_time']=d.loc[0,'decision_ts_assumed_end']
        self.assertFalse(assign_segments(d,self.bounds(),pd.Series(True,index=d.index)).loc[0,'eligible'])

    def test_outside_date_preserved(self):
        d=self.data();d.loc[0,'day']=20241231
        a=assign_segments(d,self.bounds(),pd.Series(True,index=d.index))
        self.assertEqual(len(a),4);self.assertIsNone(a.loc[0,'segment']);self.assertFalse(a.loc[0,'eligible'])


if __name__=='__main__':unittest.main()
