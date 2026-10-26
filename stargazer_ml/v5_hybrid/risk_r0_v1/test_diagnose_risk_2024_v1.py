from copy import deepcopy
import unittest

import numpy as np

from diagnose_risk_2024_v1 import calibration_delta, mapping_check, overlap_clusters, swap_groups, swap_summary
from test_risk_failure_diagnostics_v1 import row
from test_risk_crossyear_v1 import cohort,EVAL


class DiagnosticTests(unittest.TestCase):
    def test_identity_mapping_same_masks_zero_brier_delta(self):
        rows,p=cohort()
        result=mapping_check(rows,p['lr'],p['lr'],.2)
        self.assertTrue(result['primary_daily_masks_identical'])
        self.assertTrue(result['monotone_nondecreasing'])
        self.assertEqual(calibration_delta(rows,p['lr'],p['lr'],EVAL),dict(mean=0.,ci95=[0.,0.]))

    def test_order_preserving_transform_cannot_improve_ranking(self):
        rows,p=cohort()
        result=mapping_check(rows,p['lr'],p['lr']*.5+.1,.2)
        self.assertTrue(result['primary_daily_masks_identical'])
        self.assertTrue(result['monotone_nondecreasing'])
        self.assertFalse(mapping_check(rows,p['lr'],1-p['lr'],.2)['monotone_nondecreasing'])

    def test_brier_positive_delta_means_worse(self):
        rows,p=cohort()
        y=np.array([r['y_tail_5d'] for r in rows])
        result=calibration_delta(rows,y,1-y,EVAL)
        self.assertEqual(result,dict(mean=1.,ci95=[1.,1.]))

    def test_nonfinite_rejected(self):
        rows,p=cohort()
        corrupted=p['lr'].copy()
        corrupted[0]=np.nan
        with self.assertRaisesRegex(ValueError,'PROBABILITY_DOMAIN'):
            mapping_check(rows,p['lr'],corrupted,.2)

    def test_disjoint_partition_and_paired_tail_identity(self):
        rows,_=cohort()
        for r in rows:
            i=int(r['code'])
            r['lr_retained_primary']=i<8
            r['lightgbm_retained_primary']=i>=2
        before=deepcopy(rows)
        subsets=swap_groups(rows)
        self.assertEqual(sum(map(len,subsets.values())),len(rows))
        self.assertEqual(len({id(r) for values in subsets.values() for r in values}),len(rows))
        s=swap_summary(rows)
        delta=np.mean([r['y_tail_5d'] for r in rows if r['lightgbm_retained_primary']])-np.mean(
            [r['y_tail_5d'] for r in rows if r['lr_retained_primary']])
        self.assertAlmostEqual(s['paired_rate_delta'],delta)
        self.assertEqual(rows,before)

    def test_equal_total_not_same_day_paired(self):
        a,b=row(p=20240102),row(p=20240103)
        a.update(lightgbm_retained_primary=True,lr_retained_primary=False)
        b.update(lightgbm_retained_primary=False,lr_retained_primary=True)
        s=swap_summary([a,b])
        self.assertTrue(s['equal_total_retained'])
        self.assertFalse(s['daily_retained_counts_paired'])
        self.assertIsNone(s['paired_rate_delta'])

    def test_overlap_clusters_same_code_only_and_no_population_change(self):
        rows=[row(p=20240102,end=20240109),row(p=20240105,end=20240112),
              row(p=20240115,end=20240122),row(code='600001',p=20240105,end=20240112)]
        before=deepcopy(rows)
        clusters=overlap_clusters(rows)
        self.assertEqual(len(clusters),3)
        self.assertEqual(clusters[0]['rows'],2)
        self.assertEqual(clusters[0]['end'],20240112)
        self.assertEqual(rows,before)


if __name__=='__main__':
    unittest.main()
