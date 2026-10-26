from copy import deepcopy
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import warnings

import numpy as np

from risk_fit_v1 import (CONFIG, SCOPE, authorize, arrays, fit_models, logits, main,
                         train, validate_rows, probabilities)
from risk_fit_eval_v1 import keep_mask, block_indices, evaluate_segment, probability_metrics
from price_verification_guarded_run import sha
from historical_risk_core_v1 import STOCK_FIELDS, MARKET_FIELDS

FIELDS = list(STOCK_FIELDS + MARKET_FIELDS)
CONFIG_DATA = json.loads(CONFIG.read_bytes())


class SyntheticFixedModel:
    def __init__(self, logit_input=False):
        self.logit_input = logit_input

    def predict_proba(self, x):
        p = 1 / (1 + np.exp(-x[:, 0])) if self.logit_input else x[:, 0]
        return np.column_stack([1 - p, p])


def row(code='000001', p=20200102, y=0):
    return dict(code=code, P=p, label_end_date=20200109, segment='train',
                training_eligible=True, label_status='LABEL_COMPLETE', y_tail_5d=y,
                mae_5d=-.1 if y else -.02, admission_scope=SCOPE, coverage_verified=False,
                features={f: .02 for f in FIELDS})


class AdmissionTests(unittest.TestCase):
    def test_null_isolated_not_negative(self):
        r = row()
        r.update(training_eligible=False, y_tail_5d=None, mae_5d=None)
        self.assertEqual(validate_rows([r], FIELDS)['train'], [])

    def test_duplicate_blocks(self):
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            validate_rows([row(), row()], FIELDS)

    def test_future_label_crossing_blocks(self):
        r = row(p=20231229)
        r['label_end_date'] = 20240108
        with self.assertRaisesRegex(ValueError, 'PURGE'):
            validate_rows([r], FIELDS)

    def test_features_strict_whitelist(self):
        with self.assertRaisesRegex(ValueError, 'WHITELIST'):
            validate_rows([row()], FIELDS + ['code'])
        r = row()
        r['features']['ret_1d'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'FEATURE_INVALID'):
            validate_rows([r], FIELDS)

    def test_no_scope_upgrade(self):
        r = row()
        r['coverage_verified'] = True
        with self.assertRaisesRegex(ValueError, 'SCOPE'):
            validate_rows([r], FIELDS)

    def test_no_code_or_date_in_arrays(self):
        x, y = arrays([row()], FIELDS)
        self.assertEqual(x.shape, (1, 16))
        self.assertEqual(y.tolist(), [0])


class AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.check = dict(manifest_sha256='a'*64, config_sha256='b'*64, contract_sha256='c'*64)
        self.record = dict(self.check, status='AUTHORIZED', scope='FIT_RESTRICTED_R0_V1',
                           signer='SYNTHETIC_TEST_ONLY', approval_reference='not a real approval',
                           authorized_at=datetime.now(timezone.utc).isoformat())

    def authorize_record(self, record):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'auth.json'
            path.write_bytes(json.dumps(record).encode())
            return authorize(path, self.check)

    def test_missing_draft_and_exact_scope(self):
        self.assertFalse(authorize(None, self.check)[0])
        self.assertFalse(self.authorize_record(dict(self.record, status='DRAFT'))[0])
        allowed, error = self.authorize_record(dict(self.record, scope='NOT_FIT_RESTRICTED_R0_V1'))
        self.assertFalse(allowed)
        self.assertEqual(error, 'FIT_AUTHORIZATION_REQUIRED')

    def test_authorization_hash_binding(self):
        self.assertTrue(self.authorize_record(self.record)[0])
        for key in self.check:
            self.assertFalse(self.authorize_record(dict(self.record, **{key: 'd'*64}))[0])

    def test_signer_and_time_required(self):
        for values in (dict(signer=''), dict(authorized_at='2026-10-15'), dict(approval_reference=None),
                       dict(authorized_at='2099-01-01T00:00:00+00:00')):
            self.assertFalse(self.authorize_record(dict(self.record, **values))[0])

    def test_cli_without_authorization_never_calls_fit(self):
        with patch('risk_fit_v1.preflight', return_value=(CONFIG_DATA, {}, self.check)), patch(
                'risk_fit_v1.train') as fit, redirect_stdout(io.StringIO()):
            self.assertEqual(main(['check']), 2)
            self.assertEqual(main(['train']), 2)
            fit.assert_not_called()

    def test_direct_train_guard_before_output(self):
        with patch('risk_fit_v1.fit_models') as fit:
            with self.assertRaisesRegex(ValueError, 'FIT_AUTHORIZATION_REQUIRED'):
                train(CONFIG_DATA, {}, self.check, None)
            fit.assert_not_called()


class EvaluationTests(unittest.TestCase):
    def population(self):
        rows = []
        for d in range(20):
            for j in range(10):
                r = row(code=f'{j:06}', p=20200101+d, y=int(j >= 8))
                r['features']['atr14_pct'] = j / 100
                rows.append(r)
        return rows

    def test_daily_counts_and_tie_break(self):
        rows = self.population()
        keep = keep_mask(rows, np.ones(len(rows)), .2)
        for d in sorted({r['P'] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r['P'] == d]
            self.assertEqual(int(keep[idx].sum()), 8)
            self.assertFalse(keep[idx[0]])
            self.assertFalse(keep[idx[1]])

    def test_no_forced_rejection_on_tiny_day(self):
        self.assertTrue(keep_mask([row()], np.array([.9]), .2)[0])

    def test_block_sampling_and_pairing(self):
        indices = block_indices(25, 20, 10, 42)
        np.testing.assert_equal(indices, block_indices(25, 20, 10, 42))
        self.assertEqual(indices.shape, (20, 25))
        np.testing.assert_equal(np.diff(indices[:, :10], axis=1), np.ones((20, 9)))
        with self.assertRaises(ValueError):
            block_indices(3, 20, 10, 42)

    def test_pair_direction_and_random_expectation(self):
        rows = self.population()
        scores = np.array([.9 if r['y_tail_5d'] else .1 for r in rows])
        cfg = dict(CONFIG_DATA['evaluation'], bootstrap_reps=20)
        result = evaluate_segment(rows, dict(lr=scores, lightgbm=scores), .2, cfg)
        primary = result['filtering'][0]
        self.assertAlmostEqual(primary['random_expected_tail_rate'], .2)
        self.assertEqual(primary['arms']['lightgbm']['tail_rate'], 0)
        self.assertAlmostEqual(primary['paired_tail_rate_differences']['lightgbm_minus_random_expected']['mean'], -.2)
        self.assertEqual(primary['paired_tail_rate_differences']['lightgbm_minus_atr_rule']['mean'], 0)
        self.assertAlmostEqual(result['probability']['constant']['brier'], .16)

    def test_nonfinite_probability_rejected(self):
        with self.assertRaises(ValueError):
            probability_metrics([0, 1], [np.nan, .5])
        with self.assertRaises(ValueError):
            logits([1.1])
        self.assertTrue(np.isfinite(logits([0, 1])).all())


class SyntheticFitTests(unittest.TestCase):
    def test_nonpositive_sigmoid_stops_and_data_routes_are_distinct(self):
        from sklearn.linear_model import LogisticRegression
        base = LogisticRegression(**CONFIG_DATA['lr'])
        negative = Mock()
        negative.coef_ = np.array([[-1.]])
        negative.intercept_ = np.array([0.])
        gb = Mock(best_iteration_=1)
        groups = {}
        for k, name in enumerate(('train', 'select', 'calibration')):
            groups[name] = [dict(features={f: (i + 1) / 100 + k for f in FIELDS}, y_tail_5d=i % 2)
                            for i in range(40)]
        with (patch('sklearn.linear_model.LogisticRegression', side_effect=[base, negative]),
              patch('lightgbm.LGBMClassifier', return_value=gb), warnings.catch_warnings()):
            warnings.simplefilter('ignore', FutureWarning)
            with self.assertRaisesRegex(ValueError, 'SIGMOID_NONPOSITIVE'):
                fit_models(groups, CONFIG_DATA)
        negative.fit.assert_called_once()
        np.testing.assert_equal(negative.fit.call_args.args[1], arrays(groups['calibration'], FIELDS)[1])
        np.testing.assert_equal(gb.fit.call_args.kwargs['eval_set'][0][0], arrays(groups['select'], FIELDS)[0])
        np.testing.assert_allclose(base.coef_.shape, (1, 16))

    def test_actual_libraries_only_synthetic_train_select_calibration(self):
        rng = np.random.default_rng(42)
        groups = {}
        for name, n in (('train', 2000), ('select', 500), ('calibration', 500)):
            x = rng.normal(size=(n, 16))
            y = (x[:, 0] + .4 * x[:, 1] + rng.normal(scale=.5, size=n) > .5).astype(int)
            groups[name] = [dict(features=dict(zip(FIELDS, values)), y_tail_5d=int(label))
                            for values, label in zip(x, y)]
        # No real stock data, test partition, predictions or model files are used or written.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            warnings.simplefilter('ignore', UserWarning)
            models = fit_models(groups, CONFIG_DATA)
            mean = np.mean(arrays(groups['train'], FIELDS)[0], axis=0)
            np.testing.assert_allclose(models['lr'].named_steps['standardscaler'].mean_, mean)
            self.assertGreater(models['best_iteration'], 0)
            self.assertLessEqual(models['best_iteration'], 800)
            for name in ('lr', 'lightgbm'):
                self.assertGreater(models['calibrators'][name].coef_[0, 0], 0)
                self.assertEqual(len(probabilities(models[name], arrays(groups['select'], FIELDS)[0])), 500)


class RunLifecycleTests(unittest.TestCase):
    def test_successful_serialization_and_evaluation_with_fake_models_only(self):
        # End-to-end artifact/report checks with synthetic rows and deterministic NON-FITTING models.
        groups = {}
        for name in ('train', 'select', 'calibration', 'test', 'description'):
            groups[name] = []
            for d in range(20):
                for j in range(10):
                    r = row(code=f'{j:06}', p=20200101+d, y=int(j >= 8))
                    r['features'].update(ret_1d=.9 if j >= 8 else .1, atr14_pct=j/100)
                    groups[name].append(r)
        models = dict(lr=SyntheticFixedModel(), lightgbm=SyntheticFixedModel(),
                      calibrators={k: SyntheticFixedModel(True) for k in ('lr', 'lightgbm')},
                      constant=.2, best_iteration=1)
        with tempfile.TemporaryDirectory() as tmp:
            dummy = Path(tmp) / 'input.json'
            dummy.write_bytes(b'{}')
            manifest = Path(tmp) / 'manifest.json'
            manifest.write_bytes(json.dumps(dict(inputs=[dict(path=str(dummy), sha256=sha(dummy))])).encode())
            check = dict(output=str(Path(tmp) / 'run'), manifest_sha256=sha(manifest),
                         config_sha256='b'*64, contract_sha256='c'*64)
            auth = Path(tmp) / 'auth.json'
            auth.write_bytes(json.dumps(dict(check, status='AUTHORIZED', scope='FIT_RESTRICTED_R0_V1',
                signer='SYNTHETIC_TEST_ONLY', approval_reference='synthetic end-to-end mock test',
                authorized_at=datetime.now(timezone.utc).isoformat())).encode())
            cfg = dict(CONFIG_DATA, manifest=str(manifest),
                       evaluation=dict(CONFIG_DATA['evaluation'], bootstrap_reps=20))
            with patch('risk_fit_v1.fit_models', return_value=models), redirect_stdout(io.StringIO()):
                self.assertEqual(train(cfg, groups, check, auth), 0)
            import pyarrow.parquet as pq
            output = Path(check['output'])
            self.assertEqual(pq.read_table(output / 'predictions.parquet').num_rows, 1000)
            report = json.loads((output / 'report.json').read_bytes())
            self.assertFalse(report['production_gate_passed'])
            self.assertEqual(report['segments']['calibration']['role'], 'CALIBRATOR_FIT_IN_SAMPLE')
            stop = json.loads((output / 'stop_record.json').read_bytes())
            self.assertEqual(stop['status'], 'completed')
            self.assertTrue(all(sha(output / r['name']) == r['sha256'] for r in stop['files']))

    def test_failure_terminal_no_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            check = dict(output=str(Path(tmp) / 'run'), manifest_sha256='a'*64,
                         config_sha256='b'*64, contract_sha256='c'*64)
            auth = Path(tmp) / 'auth.json'
            auth.write_bytes(json.dumps(dict(check, status='AUTHORIZED', scope='FIT_RESTRICTED_R0_V1',
                signer='SYNTHETIC_TEST_ONLY', approval_reference='synthetic mock failure test',
                authorized_at=datetime.now(timezone.utc).isoformat())).encode())
            with (patch('risk_fit_v1.fit_models', side_effect=RuntimeError('synthetic failure')) as fit,
                  redirect_stdout(io.StringIO())):
                self.assertEqual(train(CONFIG_DATA, {}, check, auth), 1)
                stop = json.loads((Path(check['output']) / 'stop_record.json').read_bytes())
                self.assertEqual(stop['status'], 'failed')
                self.assertTrue(stop['model_fit_attempted'])
                self.assertFalse(stop['real_model_fit_completed'])
                with self.assertRaises(FileExistsError):
                    train(CONFIG_DATA, {}, check, auth)
                self.assertEqual(fit.call_count, 1)

    def test_budget_preflight_does_not_write_oversized_input_or_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            check = dict(output=str(Path(tmp) / 'run'), manifest_sha256='a'*64,
                         config_sha256='b'*64, contract_sha256='c'*64)
            auth = Path(tmp) / 'auth.json'
            auth.write_bytes(json.dumps(dict(check, status='AUTHORIZED', scope='FIT_RESTRICTED_R0_V1',
                signer='SYNTHETIC_TEST_ONLY', approval_reference='synthetic budget test',
                authorized_at=datetime.now(timezone.utc).isoformat())).encode())
            with patch('risk_fit_v1.fit_models') as fit, redirect_stdout(io.StringIO()):
                self.assertEqual(train(dict(CONFIG_DATA, new_output_cap=2*1024**2 + 1), {}, check, auth), 1)
                fit.assert_not_called()
                stop = json.loads((Path(check['output']) / 'stop_record.json').read_bytes())
                self.assertFalse(stop['model_fit_attempted'])
                self.assertFalse((Path(check['output']) / 'input_and_authorization.json').exists())


if __name__ == '__main__':
    unittest.main()
