from copy import deepcopy
import unittest

import numpy as np

from audit_risk_crossyear_result_v1 import weighted_tail, audit_evaluation
from risk_crossyear_eval_v1 import evaluate_test
from risk_crossyear_v1 import prediction_records
from test_risk_crossyear_v1 import cohort, EVAL


class AuditTests(unittest.TestCase):
    def test_weighted_quantile_matches_repeated_observations(self):
        rng=np.random.default_rng(20261015)
        for _ in range(100):
            values=np.sort(rng.normal(size=30))
            weights=rng.integers(0,6,size=30)
            repeated=np.repeat(values,weights)
            expected=repeated[repeated<=np.quantile(repeated,.05)].mean()
            self.assertAlmostEqual(weighted_tail(values,weights),expected,places=12)

    def test_ties_and_zero_weight_values(self):
        self.assertEqual(weighted_tail(np.array([-.9,-.4,-.4,-.01]),np.array([0,3,2,95])),-.4)

    def test_repeated_cutoff_not_rounded_below_exact_tie(self):
        rng=np.random.default_rng(17)
        for _ in range(500):
            values=np.sort(np.round(rng.normal(size=30),2))
            weights=rng.integers(0,8,size=30)
            repeated=np.repeat(values,weights)
            expected=repeated[repeated<=np.quantile(repeated,.05)].mean()
            self.assertAlmostEqual(weighted_tail(values,weights),expected,places=12)

    def test_empty_weighted_tail_refused(self):
        with self.assertRaisesRegex(ValueError,'EMPTY_WEIGHTED_TAIL'):
            weighted_tail(np.array([-.2]),np.array([0]))

    def test_independent_evaluation_replay_and_corruption_detection(self):
        rows,p=cohort()
        constant=.5
        records=prediction_records(rows,p,constant,dict(evaluation=EVAL),'F1')
        report=evaluate_test(rows,p,constant,EVAL)
        self.assertTrue(audit_evaluation(rows,records,report,EVAL)['deep_mae_cis_independently_weighted_recomputed'])
        broken=deepcopy(report)
        broken['paired_additions']['worst5pct_mae_differences']['lightgbm_minus_lr']['ci95'][0]-=.2
        with self.assertRaises(AssertionError):
            audit_evaluation(rows,records,broken,EVAL)


if __name__=='__main__':
    unittest.main()
