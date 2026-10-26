"""Synthetic future dates only: no real independent-validation outcomes."""
import unittest

import numpy as np
import pandas as pd

from postclose_next_period_evaluation_v1 import (TARGET, validate_population,
                                               descriptive_evaluation, preparation_check)


def sample():
    rows = []
    for day, next_day in [(20261019, 20261020), (20261020, 20261021)]:
        for i in range(5):
            rows.append(dict(code=f'{i+1:06d}', day=day,
                feature_max_available_at=day*1000000+160000,
                scored_at=day*1000000+170000, next_open_ts=next_day*1000000+93000,
                label_end_time=next_day*1000000+150000, data_complete_time=next_day*1000000+180000,
                label_state='COMPLETE', next_session_adverse_excursion=.01*(i+1),
                q90_adverse=.02*(i+1), q90_linear=.015*(i+1), stock_return1d=-.01*i,
                score_state='SCORED'))
    return pd.DataFrame(rows)


class EvaluationTests(unittest.TestCase):
    def check_frame(self, frame):
        validate_population(frame, {f'{i+1:06d}' for i in range(5)}, 20261016,
                            20261022180000, 20261017180000,
                            {20261019:20261020, 20261020:20261021})

    def test_01_valid_declared_times(self):
        self.check_frame(sample())

    def test_02_developed_dates_not_independent(self):
        f = sample()
        f.loc[0, 'day'] = 20261016
        with self.assertRaisesRegex(ValueError, 'DEVELOPED'):
            self.check_frame(f)

    def test_03_duplicate_population_rejected(self):
        f = sample()
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            self.check_frame(pd.concat([f, f.iloc[:1]], ignore_index=True))

    def test_04_late_or_lookahead_predictions_rejected(self):
        for column, value in [('scored_at',20261020093000),
                              ('feature_max_available_at',20261019180000),
                              ('feature_max_available_at',20261019145900)]:
            f = sample()
            f.loc[0, column] = value
            with self.assertRaisesRegex(ValueError, 'LOOKAHEAD_OR_LATE'):
                self.check_frame(f)

    def test_05_invalid_ts_precision_and_calendar(self):
        for value in (20261019,20261301160000,0):
            f = sample()
            f.loc[0,'scored_at'] = value
            with self.assertRaises(ValueError):
                self.check_frame(f)

    def test_06_future_label_not_yet_mature(self):
        f = sample()
        f.loc[0,'data_complete_time'] = 20261023160000
        with self.assertRaisesRegex(ValueError, 'NOT_MATURE'):
            self.check_frame(f)

    def test_07_pending_label_no_zero_fill(self):
        f = sample()
        f.loc[0,'label_state'] = 'PENDING'
        with self.assertRaisesRegex(ValueError, 'MUST_BE_NULL'):
            self.check_frame(f)
        f.loc[0,TARGET] = np.nan
        f.loc[0,'data_complete_time'] = np.nan
        self.check_frame(f)

    def test_08_unscorable_not_low_risk(self):
        f = sample()
        f.loc[0,'score_state'] = 'UNSCORABLE'
        with self.assertRaisesRegex(ValueError, 'MUST_NOT_HAVE_SCORE'):
            self.check_frame(f)
        f.loc[0,['q90_adverse','q90_linear']] = np.nan
        self.check_frame(f)

    def test_09_no_scope_expansion(self):
        f = sample()
        f.loc[0,'code'] = '999999'
        with self.assertRaisesRegex(ValueError, 'SCOPE'):
            self.check_frame(f)

    def test_10_calendar_mismatch_rejected(self):
        f = sample()
        f.loc[0,'next_open_ts'] = 20261022093000
        f.loc[0,'label_end_time'] = 20261022150000
        with self.assertRaisesRegex(ValueError, 'CALENDAR_MISMATCH'):
            self.check_frame(f)

    def test_11_target_and_prediction_nonfinite(self):
        for field in [TARGET,'q90_adverse','q90_linear']:
            for value in [np.nan,np.inf,-.01,1.1]:
                f = sample()
                f.loc[0,field] = value
                with self.assertRaises(ValueError):
                    self.check_frame(f)

    def test_12_population_denominator_retained(self):
        f = sample()
        f.loc[0,'label_state'] = 'PENDING'
        f.loc[0,TARGET] = np.nan
        f.loc[5,'score_state'] = 'UNSCORABLE'
        f.loc[5,['q90_adverse','q90_linear']] = np.nan
        r = descriptive_evaluation(f,.04,[.02,.04,.06,.08])
        self.assertEqual(r['population']['rows'],10)
        self.assertEqual(r['metrics']['rows'],8)
        self.assertEqual(r['evaluated_fraction'],.8)
        # Only four complete common rows per day remain: neither qualifies.
        self.assertEqual(r['same_day_fraction_of_population'],0)
        self.assertFalse(r['independent_validation_passed'])

    def test_13_same_day_equal_retained_counts(self):
        r = descriptive_evaluation(sample(),.04,[.02,.04,.06,.08])
        methods = r['same_day_comparison']['methods']
        for method in ['model','linear','simple_weakness','random_expectation']:
            self.assertEqual(methods[method]['denominator'],8)
        self.assertEqual(methods['unfiltered']['denominator'],10)

    def test_14_all_missing_metrics_null_not_zero(self):
        f = sample()
        f['label_state'] = 'DATA_INSUFFICIENT'
        f[TARGET] = np.nan
        r = descriptive_evaluation(f,.04,[.02,.04,.06,.08])
        self.assertEqual(r['metrics']['rows'],0)
        self.assertIsNone(r['metrics']['pinball'])

    def test_15_order_independence(self):
        f = sample()
        a = descriptive_evaluation(f,.04,[.02,.04,.06,.08])
        b = descriptive_evaluation(f.iloc[::-1],.04,[.02,.04,.06,.08])
        self.assertEqual(a,b)

    def test_16_preparation_does_not_claim_pass(self):
        r = preparation_check()
        self.assertFalse(r['evaluation_ready'])
        self.assertIn('UNBOUND:population_manifest',r['blockers'])
        self.assertIsNone(r['next_period_metrics'])


if __name__ == '__main__':
    unittest.main()
