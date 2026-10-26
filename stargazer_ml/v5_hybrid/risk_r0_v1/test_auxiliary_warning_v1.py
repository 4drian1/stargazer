"""Continuous-target, ties and matching regressions; no real fitting."""
import unittest

import numpy as np
import pandas as pd

from diagnose_auxiliary_120m_v1 import continuous_concordance,contrast,stats,TARGET,TIME


class WarningTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'code':['A','B'],TIME:[20260101100000]*2,'ratio':[.1,.8],TARGET:[.01,.03]})

    def test_concordance_positive(self):
        r,_=continuous_concordance(self.frame(),'ratio',False)
        self.assertEqual(r['equal_date_mean'],1)

    def test_concordance_negative(self):
        f=self.frame();f['ratio']=[.8,.1]
        self.assertEqual(continuous_concordance(f,'ratio',False)[0]['equal_date_mean'],0)

    def test_feature_tie_credit(self):
        f=self.frame();f['ratio']=.2
        r,_=continuous_concordance(f,'ratio',False)
        self.assertEqual(r['equal_date_mean'],.5)
        self.assertEqual(r['feature_tied_pairs'],1)

    def test_target_ties_excluded(self):
        f=self.frame();f[TARGET]=0
        r,_=continuous_concordance(f,'ratio',False)
        self.assertEqual(r['status'],'NOT_TESTED_NO_SUPPORTED_DATES')
        self.assertEqual(r['target_unequal_pairs'],0)

    def test_continuous_not_binary_outcome(self):
        r,_=continuous_concordance(self.frame(),'ratio',False)
        self.assertEqual(r['target_unequal_pairs'],1)
        self.assertFalse(r['is_classification_accuracy'])

    def test_shared_no_stock_identification(self):
        self.assertEqual(continuous_concordance(self.frame(),'ratio',True)[0]['status'],
                         'NOT_IDENTIFIABLE_SHARED_ACROSS_STOCKS')

    def test_contrast_continuous_units(self):
        r,_=contrast(self.frame(),'ratio',False)
        self.assertAlmostEqual(r['equal_date_mean'],.02)

    def test_same_day_not_same_cutoff(self):
        f=self.frame();f.loc[1,TIME]=20260101101500
        self.assertEqual(contrast(f,'ratio',False)[0]['status'],'NOT_TESTED_NO_SUPPORTED_DATES')
        self.assertAlmostEqual(contrast(f,'ratio',True)[0]['equal_date_mean'],.02)

    def test_stats_empty(self):
        self.assertIsNone(stats(self.frame().iloc[:0])['mean_drop'])

    def test_future_target_not_interpreted_as_probability(self):
        f=self.frame();f[TARGET]=[.02,.04]
        self.assertAlmostEqual(stats(f)['mean_drop'],.03)


if __name__=='__main__':unittest.main()
