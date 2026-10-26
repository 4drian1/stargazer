"""Synthetic boundary cases; no raw data retrieval and no estimator fit."""
import ast
import inspect
import unittest

import numpy as np
import pandas as pd
from scipy.special import expit

import diagnose_multiboard_transfer_v1 as diag


class TransferTests(unittest.TestCase):
    def test_diagnostic_module_has_no_model_fit_entry(self):
        tree = ast.parse(inspect.getsource(diag))
        called = [n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
                  for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, (ast.Attribute, ast.Name))]
        self.assertFalse(set(called) & {'fit', 'fit_algorithm', 'fit_lr_preprocessing', 'train'})

    def test_same_cutoff_pair_auc_ignores_between_day_separation(self):
        frame = pd.DataFrame({'code': list('ABAB'), diag.prep.TIME: [20250101094500]*2+[20250102100000]*2,
                              'y_tail_1200m': [0, 0, 1, 1]})
        result = diag.rank_summary(frame, [.1, .2, .8, .9])
        self.assertEqual(result['pooled_auc'], 1.)
        self.assertIsNone(result['same_cutoff_pair_weighted_auc'])
        self.assertEqual(result['mixed_label_cutoffs'], 0)

    def test_same_cutoff_ties_and_reversed_ranking(self):
        frame = pd.DataFrame({'code': list('ABCD'), diag.prep.TIME: [20250101094500]*4,
                              'y_tail_1200m': [0, 1, 0, 1]})
        self.assertEqual(diag.rank_summary(frame, [.5]*4)['same_cutoff_pair_weighted_auc'], .5)
        self.assertEqual(diag.rank_summary(frame, [.9, .1, .9, .1])['same_cutoff_pair_weighted_auc'], 0.)
        self.assertEqual(diag.rank_summary(frame, [.1, .9, .1, .9])['same_cutoff_pair_weighted_auc'], 1.)

    def test_empty_rank_population_is_not_pass(self):
        empty = pd.DataFrame({'code': [], diag.prep.TIME: [], 'y_tail_1200m': []})
        self.assertEqual(diag.rank_summary(empty, [])['status'], 'NOT_TESTED_EMPTY_POPULATION')

    def test_nonfinite_rank_rejected(self):
        frame = pd.DataFrame({'code': ['A'], diag.prep.TIME: [20250101094500], 'y_tail_1200m': [1]})
        with self.assertRaisesRegex(ValueError, 'INVALID_RANK'):
            diag.rank_summary(frame, [np.nan])

    def test_shared_signal_not_multiplied_by_stock_count(self):
        frame = pd.DataFrame({'code': list('ABCDEF'), diag.prep.TIME: [1, 1, 2, 2, 3, 3],
            'y_tail_1200m': [1, 0, 0, 1, 0, 0], 'mb5_x_rate': [np.nan, np.nan, 0., 0., .2, .2]})
        result = diag.signal_summary(frame, 'mb5_x_rate')
        self.assertEqual(result['observations'], 3)
        self.assertEqual(sum(v['observations'] for v in result['states'].values()), 3)
        self.assertEqual(sum(v['candidate_rows'] for v in result['states'].values()), 6)
        self.assertEqual(result['states']['active']['candidate_rows'], 2)
        self.assertEqual(result['states']['unknown']['observations'], 1)

    def test_own_signal_retains_candidate_unit(self):
        frame = pd.DataFrame({'code': list('AB'), diag.prep.TIME: [20250101094500]*2,
            'y_tail_1200m': [1, 0], 'own5_x_rate': [np.nan, 0.]})
        result = diag.signal_summary(frame, 'own5_x_rate')
        self.assertEqual(result['observations'], 2)
        self.assertIsNone(result['states']['active']['tail_rate'])

    def test_conflicting_shared_value_rejected(self):
        frame = pd.DataFrame({diag.prep.TIME: [1, 1], 'mb5_x_rate': [0., .1]})
        with self.assertRaisesRegex(ValueError, 'SHARED_SIGNAL_CONFLICT'):
            diag.signal_unit(frame, 'mb5_x_rate')

    def test_signal_future_label_does_not_change_observed_count(self):
        frame = pd.DataFrame({'code': list('AB'), diag.prep.TIME: [20250101094500]*2,
            'y_tail_1200m': [1, 0], 'own5_x_rate': [np.nan, .1]})
        first = diag.signal_summary(frame, 'own5_x_rate')
        frame['y_tail_1200m'] = 1 - frame.y_tail_1200m
        second = diag.signal_summary(frame, 'own5_x_rate')
        self.assertEqual([v['observations'] for v in first['states'].values()],
                         [v['observations'] for v in second['states'].values()])

    def test_shift_constant_and_unknown_not_divide_by_zero(self):
        train = pd.DataFrame({'v': [1., 1.], 'empty': [np.nan, np.nan]})
        other = pd.DataFrame({'v': [2., 2.], 'empty': [np.nan, np.nan]})
        result = diag.shift(train, other, list(train))
        self.assertIsNone(result['v']['mean_shift_train_sd'])
        self.assertEqual(result['v']['outside_train_range'], 1.)
        self.assertIsNone(result['empty']['outside_train_range'])

    def test_shift_infinite_is_not_missing(self):
        with self.assertRaisesRegex(ValueError, 'INFINITE_SHIFT'):
            diag.shift(pd.DataFrame({'v': [1.]}), pd.DataFrame({'v': [np.inf]}), ['v'])

    def test_linear_value_and_missing_contributions_exact(self):
        state = {'columns': ['v'], 'median': [2.], 'mean': [1.], 'scale': [2.]}
        linear = {'coef': [3., -4.], 'intercept': .5}
        frame = pd.DataFrame({'v': [3., np.nan]})
        logits, terms = diag.logit_terms(frame, state, linear)
        np.testing.assert_allclose(logits, [3.5, -2.])
        expected_mean = .5 + terms['v']['numeric_mean_logit'] + terms['v']['missing_mean_logit']
        self.assertAlmostEqual(expected_mean, logits.mean())
        np.testing.assert_allclose(expit(logits), [1/(1+np.exp(-3.5)), 1/(1+np.exp(2))])

    def test_first_touch_strictly_excludes_current_low(self):
        first, minimum = diag.positive_offsets(np.array([1., 95., 94., 92., 93.]),
            np.array([100.]*5), [0], [-.08], horizon=4)
        self.assertEqual(first.tolist(), [3])
        self.assertEqual(minimum.tolist(), [3])

    def test_last_future_slot_is_included(self):
        first, _ = diag.positive_offsets(np.array([100., 99., 98., 97., 92.]), np.array([100.]*5), [0], [-.08], horizon=4)
        self.assertEqual(first.tolist(), [4])

    def test_company_action_normalization_has_no_fake_touch(self):
        raw = np.array([100., 50., 49., 48., 46.])
        normal = np.array([1., 2., 2., 2., 2.])
        first, _ = diag.positive_offsets(raw * normal, np.array([100.]*5), [0], [-.08], horizon=4)
        self.assertEqual(first.tolist(), [4])

    def test_missing_slot_and_unreproduced_mae_rejected(self):
        for lows, expected, reason in [([100., np.nan, 93., 92.], -.08, 'MISSING'),
                                        ([100., 95., 93., 92.], -.09, 'MAE_NOT'),
                                        ([100., 99., 98., 97.], -.03, 'POSITIVE_NOT')]:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                diag.positive_offsets(np.array(lows), np.array([100.]*4), [0], [expected], horizon=3)

    def test_window_truncation_not_shortened(self):
        with self.assertRaisesRegex(ValueError, 'TRUNCATED'):
            diag.positive_offsets(np.array([100., 92.]), np.array([100.]*2), [0], [-.08], horizon=2)


if __name__ == '__main__':
    unittest.main()
