"""Offline regressions for intensity boundaries and exact-time comparisons."""
import unittest

import numpy as np
import pandas as pd

from diagnose_newhigh_intensity_v1 import bands,contrasts,cutoff_ranking,daily_interval,summarize,TIME


class IntensityTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'code':['A','B'],TIME:[20260101100000]*2,
            'ratio':[.1,.8],'y_tail_1200m':[0,1],'mae_1200m':[-.01,-.1]})

    def test_boundaries(self):
        self.assertEqual(bands([0,.25,.5,.75,1,np.nan]).tolist(),
            ['zero','(0,.25]','(.25,.50]','(.50,.75]','(.75,1]','unknown'])

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):bands([-.1])

    def test_infinite_rejected(self):
        with self.assertRaises(ValueError):bands([np.inf])

    def test_gt_one_rejected(self):
        with self.assertRaises(ValueError):bands([1.1])

    def test_exact_cutoff_difference(self):
        r,days=contrasts(self.frame(),'ratio',False)
        self.assertEqual(r['equal_date_mean'],1)
        self.assertEqual(r['supported_cutoffs'],1)
        self.assertEqual(days.supported_cutoffs.iloc[0],1)

    def test_same_day_is_not_same_cutoff(self):
        f=self.frame();f.loc[1,TIME]=20260101101500
        r,_=contrasts(f,'ratio',False)
        self.assertEqual(r['status'],'NOT_TESTED_NO_SUPPORTED_DATES')
        r,_=contrasts(f,'ratio',True)
        self.assertEqual(r['equal_date_mean'],1)
        self.assertEqual(r['matching'],'SAME_DAY_DIFFERENT_CUTOFFS')

    def test_middle_excluded(self):
        f=self.frame();f.loc[0,'ratio']=.5
        r,_=contrasts(f,'ratio',False)
        self.assertEqual(r['status'],'NOT_TESTED_NO_SUPPORTED_DATES')

    def test_empty_support(self):
        f=self.frame().iloc[:0]
        self.assertEqual(contrasts(f,'ratio',False)[0]['status'],'NOT_TESTED_NO_SUPPORTED_DATES')
        self.assertEqual(cutoff_ranking(f,'ratio',False)[0]['mixed_label_cutoffs'],0)

    def test_rank_perfect(self):
        r,_=cutoff_ranking(self.frame(),'ratio',False)
        self.assertEqual(r['equal_date_mean'],1)
        self.assertEqual(r['positive_negative_pairs'],1)

    def test_rank_ties_half_credit(self):
        f=self.frame();f.ratio=.3
        r,_=cutoff_ranking(f,'ratio',False)
        self.assertEqual(r['equal_date_mean'],.5)
        self.assertEqual(r['all_score_tied_mixed_cutoffs'],1)

    def test_rank_no_mixed_labels(self):
        f=self.frame();f.y_tail_1200m=0
        self.assertEqual(cutoff_ranking(f,'ratio',False)[0]['status'],'NOT_TESTED_NO_SUPPORTED_DATES')

    def test_shared_not_stock_rank(self):
        self.assertEqual(cutoff_ranking(self.frame(),'ratio',True)[0]['status'],
            'NOT_IDENTIFIABLE_SHARED_ACROSS_STOCKS')

    def test_cutoff_equal_not_pair_count_weighted(self):
        first=pd.DataFrame({'code':list('ABCDEF'),TIME:[20260101100000]*6,
            'ratio':[.9,.9,.9,.1,.1,.1],'y_tail_1200m':[1,1,1,0,0,0]})
        second=self.frame();second[TIME]=20260101101500;second['ratio']=[.9,.1]
        r,_=cutoff_ranking(pd.concat([first,second],ignore_index=True),'ratio',False)
        self.assertEqual(r['equal_date_mean'],.5)
        self.assertAlmostEqual(r['pooled_pair_weighted_auc'],.9)

    def test_short_date_support_not_passed(self):
        r=daily_interval(pd.DataFrame({'day':[20260101],'effect':[.2]}),'effect')
        self.assertEqual(r['status'],'NOT_TESTED_TOO_FEW_DATES')

    def test_constant_date_interval(self):
        r=daily_interval(pd.DataFrame({'day':range(20260101,20260113),'effect':[.2]*12}),'effect')
        np.testing.assert_allclose(r['ci95'],[.2,.2])

    def test_summary_separates_rows_and_times(self):
        r=summarize(self.frame())
        self.assertEqual((r['rows'],r['cutoffs'],r['dates']),(2,1,1))


if __name__=='__main__':unittest.main()
