"""Synthetic-only fits and fail-closed authorization/evaluation regressions."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import auxiliary_regression_fit_v1 as fit
from auxiliary_regression_eval_v1 import errors,paired_errors,retention


class AuxiliaryFitTests(unittest.TestCase):
    def authorization(self):
        return dict(status='CHAT_APPROVED_ONE_FIXED_RESEARCH_FIT',scope='FULL_FRAMEWORK_AUXILIARY_120M_FIXED_RESEARCH_V1',
            config_sha256='config',entry_acceptance_sha256='entry',user_quote='approved fixed research',approval_context='test',
            http_authorized=False,deployment_authorized=False,parameter_search_authorized=False)

    def test_no_auth_never_loads(self):
        with patch.object(fit,'preflight',side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError,'SEPARATE_AUXILIARY'):
                fit.train()

    def test_auth_binding(self):
        fit.authorization(self.authorization(),'config','entry')
        with self.assertRaisesRegex(ValueError,'HASH_BINDING'):
            fit.authorization(self.authorization(),'other','entry')

    def test_wrong_scope(self):
        a=self.authorization();a['scope']='old binary fit'
        with self.assertRaises(ValueError):fit.authorization(a,'config','entry')

    def test_scope_expansion(self):
        a=self.authorization();a['http_authorized']=True
        with self.assertRaisesRegex(ValueError,'SCOPE_EXPANDED'):fit.authorization(a,'config','entry')

    def parts(self):
        rng=np.random.default_rng(42);result={}
        for s,n in [('train',80),('select',30),('calibration',20),('test',20)]:
            x=rng.normal(size=n);z=rng.normal(size=n)
            f=pd.DataFrame({'x':x,'z':z,fit.prep.TARGET:np.clip(.02+.004*x+.002*z,0,.2)})
            f.loc[0,'x']=np.nan;result[s]=f
        return result

    def config(self):
        return dict(models=dict(ridge=dict(alpha=1.,fit_intercept=True,solver='cholesky'),
            lightgbm=dict(objective='regression_l2',n_estimators=20,num_leaves=3,min_child_samples=5,
                learning_rate=.03,random_state=42,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)),
            selection=dict(metric='l2',early_stopping_rounds=3))

    def test_ridge_saved_model_replay(self):
        parts=self.parts();r=fit.fit_model(parts,['x','z'],self.config(),'ridge')
        restored=fit.replay(parts,['x','z'],'ridge',r['preprocessing'],r['model'])
        for s in parts:np.testing.assert_allclose(restored[s],r['raw'][s],atol=1e-12)

    def test_holdout_labels_do_not_fit_ridge(self):
        parts=self.parts();changed=deepcopy(parts);changed['test'][fit.prep.TARGET]=.5;changed['calibration'][fit.prep.TARGET]=.6
        a=fit.fit_model(parts,['x','z'],self.config(),'ridge');b=fit.fit_model(changed,['x','z'],self.config(),'ridge')
        self.assertEqual(a['model'],b['model'])
        np.testing.assert_array_equal(a['preprocessing']['median'],b['preprocessing']['median'])

    def test_lightgbm_saved_model_replay_and_holdout_isolation(self):
        parts=self.parts();changed=deepcopy(parts);changed['test'][fit.prep.TARGET]=.4;changed['calibration'][fit.prep.TARGET]=.3
        a=fit.fit_model(parts,['x','z'],self.config(),'lightgbm');b=fit.fit_model(changed,['x','z'],self.config(),'lightgbm')
        self.assertEqual(a['model'],b['model'])
        restored=fit.replay(parts,['x','z'],'lightgbm',a['preprocessing'],a['model'])
        np.testing.assert_allclose(restored['test'],a['raw']['test'],atol=1e-12)

    def test_all_null_train_rejected(self):
        parts=self.parts();parts['train']['x']=np.nan
        with self.assertRaisesRegex(ValueError,'ALL_NULL_TRAIN'):
            fit.fit_model(parts,['x','z'],self.config(),'ridge')

    def test_invalid_continuous_label(self):
        with self.assertRaises(ValueError):errors([np.nan],[.1])
        with self.assertRaises(ValueError):errors([-.1],[.1])

    def test_physical_clipping_is_not_probability_calibration(self):
        y=np.array([0.,.2]);raw=np.array([-.1,1.1]);physical=np.clip(raw,0,1)
        self.assertEqual(errors(y,raw)['below_zero'],1)
        self.assertEqual(errors(y,raw)['above_one'],1)
        self.assertEqual(errors(y,physical)['below_zero'],0)

    def test_paired_identity(self):
        policy=dict(block_dates=10,reps=1000,seed=42);x=np.repeat(.1,12)
        r=paired_errors(x,x,x,np.arange(12),policy)
        self.assertEqual(r['mse']['ci95'],[0,0])

    def test_exact_whitelist_and_target_exclusion(self):
        good=[f'f{i}' for i in range(232)]
        fit.prep.validate_fields(good,good)
        bad=good.copy();bad[0]=fit.prep.TARGET
        with self.assertRaisesRegex(ValueError,'LEAKAGE'):fit.prep.validate_fields(bad,bad)
        with self.assertRaisesRegex(ValueError,'WHITELIST'):fit.prep.validate_fields(good[::-1],good)

    def test_retention_no_support(self):
        f=pd.DataFrame({'code':['A'],fit.prep.TIME:[20260101100000],fit.prep.TARGET:[.1]})
        policy=dict(rejection_fraction=.2,minimum_cross_section=5,uncertainty=dict(block_dates=10,reps=1000,seed=42))
        r=retention(f,dict(ridge=np.array([.1]),lightgbm=np.array([.2])),policy)
        self.assertEqual(r['status'],'NOT_TESTED_NO_SUPPORTED_CUTOFFS')


if __name__=='__main__':unittest.main()
