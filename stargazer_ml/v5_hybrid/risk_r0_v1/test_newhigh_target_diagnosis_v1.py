"""Synthetic state/denominator tests; no real-data acquisition or fitting."""
import unittest

import numpy as np
import pandas as pd

from diagnose_newhigh_target_v1 import states,paired_days,daily_counts,group_stats,high_for,TIME


class DiagnosisTests(unittest.TestCase):
    def test_state_separation(self):
        self.assertEqual(states([0,0,.5,np.nan],[0,1,1,1]).tolist(),
            ['index_not_new_high','new_high_no_stall','new_high_stall','unknown'])

    def test_unknown_index_not_clear(self):
        self.assertEqual(states([0],[np.nan]).tolist(),['unknown'])

    def test_contradictory_conjunction_rejected(self):
        with self.assertRaisesRegex(ValueError,'WITHOUT_NEW_HIGH'):
            states([.1],[0])

    def test_invalid_ratio_rejected(self):
        for bad in [-.1,1.1,np.inf]:
            with self.assertRaises(ValueError):
                states([bad],[1])

    def test_invalid_high_rejected(self):
        with self.assertRaises(ValueError):
            states([0],[2])

    def test_empty(self):
        self.assertEqual(states([],[]).tolist(),[])

    def test_field_market_mapping(self):
        self.assertEqual(high_for('nh_own15_cyb_positive_common_falling_rate'),'nh_cyb15_close_new_day_high')
        self.assertEqual(high_for('nh_mb5_sse_positive_common_nonpositive_rate'),'nh_sse5_close_new_day_high')

    def daily(self,n=12):
        return pd.DataFrame([dict(day=20260101+i,state=s,rows=10,positives=v,mae_sum=-1.)
            for i in range(n) for s,v in [('new_high_no_stall',1),('new_high_stall',3)]])

    def test_equal_day_paired_difference(self):
        r=paired_days(self.daily())
        self.assertAlmostEqual(r['equal_date_tail_rate_difference'],.2)
        np.testing.assert_allclose(r['ci95'],[.2,.2])
        self.assertEqual(r['paired_dates'],12)

    def test_one_sided_dates_excluded(self):
        daily=self.daily();daily=daily[~((daily.day==20260101)&daily.state.eq('new_high_no_stall'))]
        self.assertEqual(paired_days(daily)['paired_dates'],11)

    def test_too_few_dates(self):
        self.assertEqual(paired_days(self.daily(3))['status'],'NOT_TESTED_TOO_FEW_DATES')

    def test_no_paired_dates(self):
        d=self.daily();d=d[d.state.eq('new_high_stall')]
        self.assertEqual(paired_days(d)['status'],'NOT_TESTED_NO_PAIRED_DATES')

    def test_equal_date_not_row_weighted(self):
        d=self.daily(2)
        d.loc[d.day.eq(20260101),['rows','positives']]*=100
        r=paired_days(d)
        self.assertAlmostEqual(r['equal_date_tail_rate_difference'],.2)

    def test_duplicate_market_rows_not_independent_cutoffs(self):
        f=pd.DataFrame({'code':['A','B'],TIME:[20260101100000]*2,
            'y_tail_1200m':[0,1],'mae_1200m':[-.01,-.1]})
        r=group_stats(f,np.ones(2,bool),[])
        self.assertEqual((r['rows'],r['dates'],r['cutoffs']),(2,1,1))
        d=daily_counts(f,np.array(['new_high_stall']*2))
        self.assertEqual(d.rows.iloc[0],2)
        self.assertEqual(d.positives.iloc[0],1)


if __name__=='__main__':
    unittest.main()
