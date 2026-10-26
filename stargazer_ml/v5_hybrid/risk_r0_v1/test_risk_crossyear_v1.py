from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import risk_crossyear_v1 as entry
from price_verification_guarded_run import encoded
from risk_crossyear_eval_v1 import deep_tail, evaluate_test, paired_additions, consistency_flags
from test_risk_failure_diagnostics_v1 import row, F2


EVAL = dict(primary_rejection_fraction=.2, descriptive_rejection_fractions=[.1,.3],
            bootstrap_reps=20, block_dates=2, seed=20261015, probability_bins=10)


def cohort():
    rows = [row(code=f'{i:06}', p=20200102+d, y=int(i>=3), end=20200120)
            for d in range(6) for i in range(10)]
    for r in rows:
        i = int(r['code'])
        r['features']['atr14_pct'] = i/100
        r['mae_5d'] = -(.02+i/100)
    p = np.array([int(r['code'])/12 for r in rows])
    return rows, dict(lr_raw=p, lr=p, lightgbm_raw=p, lightgbm=p)


def approval(check):
    return dict(status='AUTHORIZED', scope='FIT_R0_CROSSYEAR_V1', signer='test_fixture_only',
        authorized_at=datetime.now(timezone.utc).isoformat(), approval_reference='MOCK_NOT_REAL_FIT',
        new_fit_folds=['F1','F2'], reuse_only_folds=['F3'], **check)


class EvaluationTests(unittest.TestCase):
    def test_identical_predictions_paired_zero(self):
        rows, p = cohort()
        r = paired_additions(rows,p,EVAL)
        self.assertEqual(r['lightgbm_minus_lr_tail_rate'], dict(mean=0.,ci95=[0.,0.]))
        self.assertEqual(r['worst5pct_mae_differences']['lightgbm_minus_lr'], dict(mean=0.,ci95=[0.,0.]))

    def test_deep_tail_not_plain_mean_and_ties_included(self):
        values = [-.4,-.4]+[-.01]*18
        self.assertEqual(deep_tail(values), -.4)
        self.assertNotAlmostEqual(deep_tail(values), np.mean(values))

    def test_recompute_tail_after_sampling_reference(self):
        rows, p = cohort()
        for i,r in enumerate(rows):
            r['mae_5d'] -= (i//10)*.03
        p['lightgbm'] = 1-p['lr']
        from risk_fit_eval_v1 import keep_mask, block_indices
        masks = {m:keep_mask(rows, p[m], .2) for m in ('lr','lightgbm')}
        blocks = block_indices(6,20,2,20261015)
        mae=np.array([r['mae_5d'] for r in rows])
        deltas=[]
        for block in blocks:
            ix=np.concatenate([np.arange(d*10,(d+1)*10) for d in block])
            deltas.append(deep_tail(mae[ix[masks['lightgbm'][ix]]])-deep_tail(mae[ix[masks['lr'][ix]]]))
        got=paired_additions(rows,p,EVAL)['worst5pct_mae_differences']['lightgbm_minus_lr']
        np.testing.assert_allclose(got['ci95'],np.quantile(deltas,[.025,.975]))

    def test_worse_negative_mae_is_red_flag_not_improvement(self):
        rows,p=cohort()
        p['lightgbm']=1-p['lr']
        for r in rows:
            if int(r['code'])==9:
                r['mae_5d']=-.8
        result=evaluate_test(rows,p,.5,EVAL)
        self.assertLess(result['paired_additions']['worst5pct_mae_differences']['lightgbm_minus_lr']['ci95'][1],0)
        self.assertTrue(result['paired_additions']['deep_tail_red_flag'])

    def test_all_years_required_and_red_flag_blocks(self):
        rows,p=cohort()
        result=evaluate_test(rows,p,.5,EVAL)
        with self.assertRaisesRegex(ValueError,'ALL_THREE'):
            consistency_flags([result]*2)
        r=deepcopy(result)
        for pair in r['filtering'][0]['paired_tail_rate_differences'].values():
            pair.update(mean=-.01,ci95=[-.02,-.001])
        r['probability']['lightgbm']['brier_minus_constant']['ci95']=[-.1,-.01]
        r['probability']['lightgbm']['log_loss']=0
        self.assertTrue(consistency_flags([r]*3)['research_consistency'])
        bad=deepcopy(r)
        bad['paired_additions']['deep_tail_red_flag']=True
        flags=consistency_flags([r,r,bad])
        self.assertFalse(flags['research_consistency'])
        self.assertEqual(flags['deep_tail_red_flag_folds'],[3])
        self.assertFalse(flags['production_gate_passed'])


class DataTests(unittest.TestCase):
    def test_fold_reassignment_preserves_original_flags(self):
        r=row(p=20240628,end=20240705)
        r.update(training_eligible=False,segment='select')
        before=deepcopy(r)
        groups,report=entry.assign_fold([r],F2)
        self.assertEqual(groups['test'],[r])
        self.assertEqual(r,before)
        self.assertEqual(report['segments']['test']['old_admitted_false_but_this_fold_eligible'],1)

    def test_duplicate_or_non_temporal_revival_rejected(self):
        r=row()
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):
            entry.assign_fold([r,r],F2)
        r['training_eligible']=False
        with self.assertRaisesRegex(ValueError,'NON_TEMPORAL'):
            entry.assign_fold([r],F2)

    def test_population_hash_captures_features_and_key(self):
        groups=dict(train=[row()])
        first=entry.population_fingerprint(groups)
        groups['train'][0]['features']['ret_1d']=.2
        second=entry.population_fingerprint(groups)
        self.assertEqual(first['train']['keys_sha256'],second['train']['keys_sha256'])
        self.assertNotEqual(first['train']['rows_sha256'],second['train']['rows_sha256'])

    def test_hash_drift_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'input'
            p.write_bytes(b'old')
            refs=[dict(path=str(p),sha256=entry.sha(p))]
            p.write_bytes(b'new')
            with self.assertRaisesRegex(ValueError,'FROZEN_INPUT_DRIFT'):
                entry.verify_refs(refs)


class AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'auth.json'
        self.check=dict(manifest_sha256='m',config_sha256='c',contract_sha256='t')

    def accepted(self, record):
        self.path.write_bytes(encoded(record))
        return entry.authorize(self.path,self.check)

    def test_absent_draft_old_scope_and_fold_mismatch(self):
        self.assertFalse(entry.authorize(None,self.check)[0])
        for update in (dict(status='DRAFT'),dict(scope='FIT_RESTRICTED_R0_V1'),
                       dict(new_fit_folds=['F3']),dict(reuse_only_folds=[])):
            record=approval(self.check)
            record.update(update)
            self.assertFalse(self.accepted(record)[0])

    def test_evidence_time_and_hash_binding(self):
        self.assertTrue(self.accepted(approval(self.check))[0])
        for update in (dict(signer=''),dict(manifest_sha256='changed'),dict(authorized_at='2026-01-01'),
                       dict(authorized_at=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat())):
            record=approval(self.check)
            record.update(update)
            self.assertFalse(self.accepted(record)[0])

    def test_train_without_approval_never_calls_fit_or_creates_output(self):
        output=Path(self.tmp.name)/'run'
        with patch.object(entry,'OUTPUT',output),patch.object(entry,'fit_models') as fit:
            with self.assertRaisesRegex(ValueError,'SEPARATE_CROSSYEAR'):
                entry.train({}, {}, {}, [], self.check, None)
            fit.assert_not_called()
            self.assertFalse(output.exists())

    def test_cli_check_and_train_block_before_fit(self):
        for command in ('check','train'):
            with patch.object(entry,'preflight',return_value=({}, {}, {}, [], self.check.copy())), \
                 patch.object(entry,'train') as train, redirect_stdout(io.StringIO()):
                self.assertEqual(entry.main([command]),2)
                train.assert_not_called()

    def test_fit_failure_finalizes_and_does_not_start_second_fold(self):
        # All model calls mocked; this is not a synthetic or real fit.
        output=Path(self.tmp.name)/'run'
        self.assertTrue(self.accepted(approval(self.check))[0])
        manifest=Path(self.tmp.name)/'manifest'
        manifest.write_bytes(encoded(dict(inputs=[])))
        self.check['manifest_sha256']=entry.sha(manifest)
        self.accepted(approval(self.check))
        with patch.object(entry,'OUTPUT',output),patch.object(entry,'MANIFEST',manifest), \
             patch.object(entry,'fit_models',side_effect=ValueError('MOCK_FIT_FAILURE')) as fit, \
             patch.object(entry,'bytes_used',return_value=0),redirect_stdout(io.StringIO()):
            rc=entry.train(dict(new_output_cap=64*1024**2),{},dict(F1=dict(test=[])),[],self.check,self.path)
        self.assertEqual(rc,1)
        fit.assert_called_once()
        stop=json.loads((output/'stop_record.json').read_bytes())
        self.assertEqual(stop['status'],'failed')
        self.assertEqual(stop['fit_attempted_folds'],['F1'])
        self.assertEqual(stop['fit_completed_folds'],[])

    def test_full_control_flow_fits_only_f1_f2_and_reuses_f3(self):
        # Model functions are mocked; real evaluation is tested separately above.
        output=Path(self.tmp.name)/'run'
        manifest=Path(self.tmp.name)/'manifest'
        manifest.write_bytes(encoded(dict(inputs=[])))
        self.check['manifest_sha256']=entry.sha(manifest)
        self.accepted(approval(self.check))
        rows,p=cohort()
        base=dict(fields=list(rows[0]['features']),evaluation=EVAL)
        constant=float(np.mean([r['y_tail_5d'] for r in rows]))
        reused=entry.prediction_records(rows,p,constant,base,'F3')
        for r in reused:
            r.pop('fold')
        groups={f:dict(train=rows,test=rows) for f in ('F1','F2','F3')}
        evaluated=evaluate_test(rows,p,constant,EVAL)
        with patch.object(entry,'OUTPUT',output),patch.object(entry,'MANIFEST',manifest), \
             patch.object(entry,'fit_models',return_value=dict(constant=constant,best_iteration=22)) as fit, \
             patch.object(entry,'predictions_for',return_value=p), \
             patch.object(entry,'evaluate_test',return_value=deepcopy(evaluated)), \
             patch.object(entry,'bytes_used',return_value=0),redirect_stdout(io.StringIO()):
            rc=entry.train(dict(new_output_cap=64*1024**2),base,groups,reused,self.check,self.path)
        self.assertEqual(rc,0)
        self.assertEqual(fit.call_count,2)
        self.assertFalse((output/'F3_models.joblib').exists())
        stop=json.loads((output/'stop_record.json').read_bytes())
        self.assertEqual(stop['fit_completed_folds'],['F1','F2'])
        self.assertEqual(stop['reused_folds'],['F3'])

    def test_budget_stops_before_any_fit_or_body_write(self):
        output=Path(self.tmp.name)/'run'
        manifest=Path(self.tmp.name)/'manifest'
        manifest.write_bytes(encoded(dict(inputs=[])))
        self.check['manifest_sha256']=entry.sha(manifest)
        self.accepted(approval(self.check))
        with patch.object(entry,'OUTPUT',output),patch.object(entry,'MANIFEST',manifest), \
             patch.object(entry,'fit_models') as fit,patch.object(entry,'bytes_used',return_value=0), \
             redirect_stdout(io.StringIO()):
            rc=entry.train(dict(new_output_cap=2*1024**2),{}, {},[],self.check,self.path)
        self.assertEqual(rc,1)
        fit.assert_not_called()
        self.assertEqual([p.name for p in output.iterdir()],['stop_record.json'])
        self.assertIn('SERIALIZED_BUDGET_STOP',json.loads((output/'stop_record.json').read_bytes())['error'])


if __name__ == '__main__':
    unittest.main()
