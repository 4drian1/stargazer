import unittest
import numpy as np
import pandas as pd
from replay_causal_calibration_v1 import replay,TARGET,run


class CausalCalibrationTests(unittest.TestCase):
    def data(self):
        h=pd.DataFrame(dict(code=['a']*4,day=[20260101,20260102,20260103,20260104],
            segment=['calibration','calibration','test','test'],
            label_end_time=[20260102150000,20260103150000,20260104150000,20260105150000],
            q90_raw=[.02]*4,**{TARGET:[.03,.04,.10,.15]}))
        d=pd.DataFrame(dict(code=['a'],day=[20260104],decision_ts_assumed_end=[20260104150000],q90_raw=[.02]))
        return h,d

    def test_strict_maturity_and_quantile(self):
        h,d=self.data();p,l=replay(h,d,.005,window=2)
        self.assertEqual(l[0]['used_dates'],[20260101,20260102])
        self.assertLess(l[0]['max_label_end_used'],d.decision_ts_assumed_end.iloc[0])
        self.assertAlmostEqual(p.rolling_q90_adverse.iloc[0],.039)

    def test_cold_start_not_future_padding(self):
        h,d=self.data();p,l=replay(h,d,.005,window=60)
        self.assertEqual(l[0]['mode'],'FROZEN_BASE_COLD_START')
        self.assertAlmostEqual(p.rolling_q90_adverse.iloc[0],.025)

    def test_unmatured_base_refused(self):
        h,d=self.data();h.loc[1,'label_end_time']=20260105150000
        with self.assertRaisesRegex(ValueError,'BASE_CALIBRATION_NOT_MATURE'):replay(h,d,.005)

    def test_future_and_same_time_labels_do_not_change_prediction(self):
        h,d=self.data();p,_=replay(h,d,.005,window=2)
        h.loc[h.day.ge(20260103),TARGET]=.9
        pd.testing.assert_frame_equal(p,replay(h,d,.005,window=2)[0])

    def test_train_select_excluded_even_with_poison(self):
        h,d=self.data();p,_=replay(h,d,.005,window=2)
        poison=h.iloc[:1].copy();poison['code']='b';poison['segment']='train';poison[TARGET]=.99
        pd.testing.assert_frame_equal(p,replay(pd.concat([h,poison],ignore_index=True),d,.005,window=2)[0])

    def test_latest_qualified_dates_only(self):
        h,d=self.data();d['day']=20260106;d['decision_ts_assumed_end']=20260106150000
        _,l=replay(h,d,.005,window=2)
        self.assertEqual(l[0]['used_dates'],[20260103,20260104])

    def test_unknown_label_excluded(self):
        h,d=self.data();h.loc[2,TARGET]=np.nan;d['day']=20260106;d['decision_ts_assumed_end']=20260106150000
        _,l=replay(h,d,.005,window=2)
        self.assertEqual(l[0]['used_dates'],[20260102,20260104])

    def test_duplicates_refused(self):
        h,d=self.data()
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):replay(pd.concat([h,h]),d,.005)

    def test_inputs_unchanged(self):
        h,d=self.data();a=h.copy(deep=True);b=d.copy(deep=True)
        replay(h,d,.005,window=2);pd.testing.assert_frame_equal(a,h);pd.testing.assert_frame_equal(b,d)

    def test_requires_authorization(self):
        with self.assertRaisesRegex(ValueError,'AUTHORIZATION_REQUIRED'):run()

    def test_nonfinite_raw_refused(self):
        h,d=self.data();d['q90_raw']=np.inf
        with self.assertRaisesRegex(ValueError,'NONFINITE'):replay(h,d,.005)

    def test_common_unclipped_offset_keeps_order(self):
        x=np.array([.02,.03,.04]);np.testing.assert_equal(np.argsort(x+.01),np.argsort(x-.01))


if __name__=='__main__':unittest.main()
