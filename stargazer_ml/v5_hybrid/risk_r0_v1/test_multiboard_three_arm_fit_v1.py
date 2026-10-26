"""Small synthetic fits, budget/finality/auth negatives; never fit real populations."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import multiboard_three_arm_fit_v1 as fit
from multiboard_fit_evaluation_v1 import probabilities, paired_loss, filtering


def synthetic():
    rng = np.random.default_rng(42)
    parts = {}
    for j, segment in enumerate(fit.prep.SEGMENTS):
        values = rng.normal(size=(120, 2))
        y = (values[:, 0] + rng.normal(size=120) > 0).astype(int)
        values[::7, 1] = np.nan
        parts[segment] = pd.DataFrame(dict(a=values[:, 0], b=values[:, 1], y_tail_1200m=y))
    config = json.loads((fit.prep.OUT / 'config.json').read_bytes())
    return parts, config


class FitTests(unittest.TestCase):
    def test_synthetic_lr_fit_and_json_replay(self):
        parts, config = synthetic()
        result = fit.fit_algorithm(parts, ['a', 'b'], config, 'lr')
        restored = fit.replay(parts, ['a', 'b'], 'lr', fit.serializable(result['preprocessing']),
                             result['model'], result['calibration'], 1e-12)
        for segment in parts:
            for state in ['raw', 'calibrated']:
                np.testing.assert_allclose(restored[segment][state], result[state][segment], atol=1e-12, rtol=0)

    def test_synthetic_lightgbm_fit_and_text_replay(self):
        parts, config = synthetic()
        result = fit.fit_algorithm(parts, ['a', 'b'], config, 'lightgbm')
        restored = fit.replay(parts, ['a', 'b'], 'lightgbm', fit.serializable(result['preprocessing']),
                             result['model'], result['calibration'], 1e-12)
        self.assertLessEqual(result['best_iteration'], 800)
        for segment in parts:
            np.testing.assert_allclose(restored[segment]['raw'], result['raw'][segment], atol=1e-12, rtol=0)

    def test_test_labels_cannot_change_fit_or_calibration(self):
        parts, config = synthetic()
        first = fit.fit_algorithm(parts, ['a', 'b'], config, 'lr')
        parts['test']['y_tail_1200m'] = 1 - parts['test'].y_tail_1200m
        second = fit.fit_algorithm(parts, ['a', 'b'], config, 'lr')
        self.assertEqual(first['model'], second['model'])
        self.assertEqual(first['calibration'], second['calibration'])
        for s in parts:
            np.testing.assert_array_equal(first['calibrated'][s], second['calibrated'][s])

    def test_calibration_labels_do_not_change_base_model(self):
        parts, config = synthetic()
        first = fit.fit_algorithm(parts, ['a', 'b'], config, 'lr')
        parts['calibration']['y_tail_1200m'] = 1 - parts['calibration'].y_tail_1200m
        second = fit.fit_algorithm(parts, ['a', 'b'], config, 'lr')
        self.assertEqual(first['model'], second['model'])
        self.assertNotEqual(first['calibration'], second['calibration'])

    def test_unknown_algorithm_refused(self):
        parts, config = synthetic()
        with self.assertRaisesRegex(ValueError, 'UNKNOWN_ALGORITHM'):
            fit.fit_algorithm(parts, ['a', 'b'], config, 'other')

    def test_authentication_binds_exact_configuration_and_entry(self):
        valid = dict(status='CHAT_APPROVED_ONE_FIXED_RESEARCH_FIT', scope='MULTIBOARD_THREE_ARM_RESEARCH_V1',
            config_sha256='config', entry_acceptance_sha256='entry', user_quote='好，请继续',
            approval_context='continue the preceding fixed research experiment handoff',
            new_http_authorized=False, deployment_authorized=False, parameter_search_authorized=False)
        fit.check_authorization(valid, 'config', 'entry')
        for key, value in [('config_sha256', 'changed'), ('entry_acceptance_sha256', 'changed'),
                           ('deployment_authorized', True), ('user_quote', ''), ('scope', 'OTHER')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                fit.check_authorization(dict(valid, **{key: value}), 'config', 'entry')

    def test_train_without_explicit_flag_never_enters_preflight(self):
        with patch.object(fit, 'preflight', side_effect=AssertionError('must not run')):
            with self.assertRaisesRegex(ValueError, 'EXPLICIT_FIXED'):
                fit.train()

    def test_budget_stop_before_directory_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            writer = fit.EvidenceWriter(root, cap=20, reserve=5)
            with self.assertRaisesRegex(ValueError, 'BUDGET_STOP'):
                writer.write('big.bin', b'x' * 16)
            self.assertFalse(root.exists())

    def test_terminal_reserve_and_restart_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            writer = fit.EvidenceWriter(root, cap=100, reserve=20)
            writer.write('data.bin', b'x' * 80)
            writer.write('stop_record.json', b'{"status":"FAILED"}', terminal=True)
            with self.assertRaisesRegex(ValueError, 'OUTPUT_EXISTS'):
                fit.EvidenceWriter(root)
            with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
                writer.write('data.bin', b'other')

    def test_global_budget_checked_before_first_write(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(fit, 'bytes_used', return_value=fit.TOTAL_CAP):
            root = Path(temporary) / 'run'
            with self.assertRaisesRegex(ValueError, 'RESERVATION_FAILED'):
                fit.EvidenceWriter(root)
            self.assertFalse(root.exists())

    def test_writer_refuses_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = fit.EvidenceWriter(Path(temporary) / 'run')
            with self.assertRaisesRegex(ValueError, 'EVIDENCE_NAME'):
                writer.write('../bad', b'bad')

    def test_probability_metrics_boundaries_and_no_false_accuracy(self):
        result = probabilities([0, 1], [0., 1.])
        self.assertEqual(result['brier'], 0.)
        self.assertEqual(sum(b['rows'] for b in result['reliability_bins']), 2)
        self.assertNotIn('accuracy', result)
        self.assertIsNone(probabilities([0, 0], [.1, .2])['roc_auc'])
        for bad in [[np.nan, .5], [-.1, .5], [1.1, .5]]:
            with self.assertRaisesRegex(ValueError, 'INVALID_PROBABILITY'):
                probabilities([0, 1], bad)

    def test_paired_date_bootstrap_identity_and_direction(self):
        dates = np.repeat(np.arange(20), 3)
        y = np.resize([0, 1], 60)
        policy = dict(block_dates=10, reps=100, seed=42)
        same = paired_loss(y, np.full(60, .5), np.full(60, .5), dates, policy)
        self.assertEqual(same['brier']['ci95'], [0., 0.])
        better = paired_loss(y, y*.8+.1, np.full(60, .5), dates, policy)
        self.assertLess(better['brier']['ci95'][1], 0.)
        self.assertEqual(paired_loss([0], [.1], [.2], [1], policy)['status'], 'NOT_TESTED_TOO_FEW_DATES')

    def test_equal_retention_random_and_missing_rule_subset(self):
        frame = pd.DataFrame(dict(code=list('ABCDE')*12,
            decision_ts_assumed_end=np.repeat(np.arange(12)*1000000+20250101094500, 5),
            y_tail_1200m=np.tile([0, 0, 0, 0, 1], 12)))
        scores = dict(good=frame.y_tail_1200m.to_numpy(dtype=float), equal=np.zeros(60))
        policy = dict(rejection_fraction=.2, minimum_cross_section=5,
                      uncertainty=dict(block_dates=10, reps=100, seed=42))
        result = filtering(frame, scores, policy)
        self.assertEqual(result['arms']['good']['retained_rows'], 48)
        self.assertEqual(result['arms']['good']['tail_rate'], 0.)
        self.assertAlmostEqual(result['random_expected_tail_rate'], .2)
        pair = result['paired_tail_rate_differences']['good_minus_random_expected']
        self.assertAlmostEqual(pair['left_minus_right'], -.2)
        self.assertLess(pair['ci95'][1], 0.)
        small = filtering(frame.iloc[:3], {k: v[:3] for k, v in scores.items()}, policy)
        self.assertEqual(small['comparable_rows'], 0)
        self.assertIsNone(small['random_expected_tail_rate'])


if __name__ == '__main__':
    unittest.main()
