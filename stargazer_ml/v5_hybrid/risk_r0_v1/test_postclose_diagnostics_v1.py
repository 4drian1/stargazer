import unittest
import numpy as np
import pandas as pd
from diagnose_postclose_predictions_v1 import TARGET, filter_by_day, filter_summary, strata


class DiagnosticTests(unittest.TestCase):
    def frame(self, n=5):
        return pd.DataFrame(dict(code=[str(i) for i in range(n)], day=20260101,
            q90_adverse=np.arange(n) / 10, q90_linear=np.arange(n) / 10,
            stock_return1d=-np.arange(n) / 10, **{TARGET: np.arange(n) / 10}))

    def test_top_risk_excluded_equal_counts(self):
        daily, coverage = filter_by_day(self.frame(), .25)
        self.assertEqual(daily[0]['kept'], 4)
        self.assertAlmostEqual(daily[0]['model']['adverse_sum'], .6)
        self.assertAlmostEqual(daily[0]['simple_weakness']['adverse_sum'], .6)
        self.assertEqual(coverage['comparable_rows'], 5)

    def test_random_exact_expectation(self):
        daily, _ = filter_by_day(self.frame(), .25)
        self.assertAlmostEqual(daily[0]['random_expectation']['adverse_sum'], .8)
        self.assertAlmostEqual(daily[0]['random_expectation']['tail_sum'], 1.6)

    def test_four_stocks_not_evaluable(self):
        daily, coverage = filter_by_day(self.frame(4), .25)
        self.assertEqual(coverage['comparable_rows'], 0)
        self.assertEqual(filter_summary(daily)['status'], 'NOT_EVALUABLE')

    def test_missing_score_shared_population(self):
        frame = self.frame()
        frame.loc[0, 'stock_return1d'] = np.nan
        daily, coverage = filter_by_day(frame, .25)
        self.assertEqual(daily, [])
        self.assertEqual(coverage['finite_intersection_rows'], 4)

    def test_dates_never_pooled_to_meet_minimum(self):
        a = self.frame(3)
        b = self.frame(3).assign(day=20260102)
        self.assertEqual(filter_by_day(pd.concat([a, b]), .25)[0], [])

    def test_ties_do_not_use_outcomes(self):
        frame = self.frame().assign(q90_adverse=.1)
        daily, _ = filter_by_day(frame, .25)
        changed = frame.copy()
        changed[TARGET] = changed[TARGET] + 1
        second, _ = filter_by_day(changed.iloc[::-1], .25)
        self.assertAlmostEqual(second[0]['model']['adverse_sum'] - daily[0]['model']['adverse_sum'], 4)

    def test_empty_quintiles_remain(self):
        bins = strata(self.frame().assign(q90_adverse=.1), [.1] * 4, .25)
        self.assertEqual([r['rows'] for r in bins], [5, 0, 0, 0, 0])

    def test_floor_rounding_and_interval(self):
        daily, _ = filter_by_day(self.frame(6), .25)
        self.assertEqual(daily[0]['kept'], 5)
        output = filter_summary(daily)
        interval = output['differences']['random_expectation']['adverse_sum']
        self.assertAlmostEqual(interval['difference'], -.05)
        np.testing.assert_allclose(interval['interval_95'], [-.05, -.05])


if __name__ == '__main__':
    unittest.main()
