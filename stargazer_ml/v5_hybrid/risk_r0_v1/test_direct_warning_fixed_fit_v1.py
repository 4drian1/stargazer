"""Offline entry and paired evaluation regressions; no real-data fit."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import direct_warning_fixed_fit_v1 as fit
from direct_warning_fit_evaluation_v1 import paired_filter,cutoff_ranking


class EntryTests(unittest.TestCase):
    def record(self):
        return dict(scope='DIRECT_WARNING_C0_C1_FIXED_RESEARCH_V1',
            status='CHAT_APPROVED_ONE_FIXED_RESEARCH_FIT',config_sha256=fit.CONFIG_SHA,
            entry_acceptance_sha256='entry',user_quote='好，请继续',approval_context='#18.99',
            http_authorized=False,deployment_authorized=False,parameter_search_authorized=False)

    def test_auth_exact(self):
        fit.authorize(self.record(),'entry')

    def test_wrong_hash(self):
        with self.assertRaisesRegex(ValueError,'HASH_BINDING'):
            fit.authorize(self.record(),'other')

    def test_scope_expansion(self):
        r=self.record();r['http_authorized']=True
        with self.assertRaisesRegex(ValueError,'SCOPE_EXPANDED'):
            fit.authorize(r,'entry')

    def test_missing_auth(self):
        with patch.object(fit,'preflight',side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError,'EXPLICIT_FIXED'):
                fit.train()

    def test_c0_never_main_fit_and_test_labels_do_not_calibrate(self):
        config=json.loads((fit.prep.OUT/'config.json').read_bytes())
        parts={s:pd.DataFrame({'y_tail_1200m':[0,1,0,1]}) for s in fit.SEGMENTS}
        raw=np.array([.1,.8,.2,.9])
        replay={s:dict(raw=raw.copy(),calibrated=np.zeros(4)) for s in fit.SEGMENTS}
        with patch.object(fit.base,'replay',return_value=replay),patch.object(fit.base,'fit_algorithm',side_effect=AssertionError('base refit')):
            a=fit.c0_predict(parts,config,'lr')
            parts['test']['y_tail_1200m']=[1,0,1,0]
            b=fit.c0_predict(parts,config,'lr')
        self.assertEqual(a['calibration'],b['calibration'])
        self.assertTrue(np.all(a['calibrated']['test']>0))

    def test_writer_budget_before_disk(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'out'
            writer=fit.base.EvidenceWriter(target,cap=100,reserve=20)
            with self.assertRaisesRegex(ValueError,'BUDGET_STOP'):
                writer.write('large.bin',b'x'*81)
            self.assertFalse(target.exists())

    def test_writer_reject_restart(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'OUTPUT_EXISTS'):
                fit.base.EvidenceWriter(d,cap=100,reserve=20)

    def frame(self):
        return pd.DataFrame({'code':['A','B','C','D','E']*12,
            fit.TIME:np.repeat([20260101094500+i*1000000 for i in range(12)],5),
            'y_tail_1200m':[0,0,0,1,1]*12})

    def policy(self):
        return dict(rejection_fraction=.2,minimum_cross_section=5,
                    uncertainty=dict(block_dates=10,reps=1000,seed=42))

    def test_pair_zero(self):
        frame=self.frame();p=frame.y_tail_1200m.to_numpy()
        answer=paired_filter(frame,p,p,self.policy())
        self.assertEqual(answer['left_minus_right'],0)
        self.assertEqual(answer['ci95'],[0,0])
        self.assertEqual(answer['support_dates'],12)

    def test_pair_direction(self):
        frame=self.frame();p=frame.y_tail_1200m.to_numpy()
        answer=paired_filter(frame,p,-p,self.policy())
        self.assertEqual(answer['left_minus_right'],-.25)

    def test_no_support(self):
        f=self.frame().iloc[:3]
        self.assertEqual(paired_filter(f,np.zeros(3),np.zeros(3),self.policy())['status'],
                         'NOT_TESTED_NO_SUPPORTED_CUTOFFS')

    def test_rank_within_cutoff(self):
        f=self.frame();p=f.y_tail_1200m.to_numpy()
        a=cutoff_ranking(f,p)
        self.assertEqual(a['pair_weighted_auc'],1)
        self.assertEqual(a['mixed_label_cutoffs'],12)

    def test_rank_one_class(self):
        f=self.frame();f.y_tail_1200m=0
        self.assertIsNone(cutoff_ranking(f,np.ones(len(f)))['pair_weighted_auc'])


if __name__=='__main__':
    unittest.main()
