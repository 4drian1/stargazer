"""Zero-real-fit entry replay using dummy estimators, never historical estimator fitting."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import sklearn.linear_model as linear_module
import postclose_training_v1 as entry


class DummyLinear:
    def __init__(self,**kwargs):pass
    def fit(self,x,y):self.coef_=np.zeros(x.shape[1]);self.intercept_=float(np.median(y));return self
    def predict(self,x):return np.full(len(x),self.intercept_)


class DummyGBM:
    def __init__(self,**kwargs):
        self.best_iteration_=1
        self.booster_=SimpleNamespace(model_to_string=lambda:'DUMMY_NO_LEARNED_MODEL')
    def fit(self,x,y,eval_set,callbacks):
        if len(eval_set)!=1:raise AssertionError('SELECT_ONLY')
        return self
    def predict(self,x):return np.full(len(x),.03)


class FitEntryTests(unittest.TestCase):
    def invoke(self,root,fail=False):
        root=Path(root);meta=root/'cohort';meta.mkdir()
        population=root/'dummy_population';population.write_bytes(b'dummy')
        (meta/'manifest.json').write_text('{}',encoding='utf-8')
        (meta/'eligibility.parquet').write_bytes(b'dummy')
        (root/'postclose_research_policy_v1.json').write_text('{}',encoding='utf-8')
        frame=pd.DataFrame(dict(code=['A','B'],day=[20250102,20250103],label_end_time=[20250103150000,20250106150000],
            next_session_adverse_excursion=[.01,.04],x=[1.,np.nan]))
        manifest=dict(fields=['x'],population=dict(path=str(population),sha256=entry.digest(population.read_bytes())),
            outputs={'eligibility.parquet':dict(sha256=entry.digest((meta/'eligibility.parquet').read_bytes()))})
        policy=dict(primary_model_proposed=dict(name='dummy'),sample_gates={})
        dummy=SimpleNamespace(LGBMRegressor=DummyGBM,early_stopping=lambda *a,**k:None)
        if fail:
            class Broken(DummyGBM):
                def predict(self,x):raise RuntimeError('EXPECTED_FAILURE')
            dummy.LGBMRegressor=Broken
        with patch.object(entry,'HERE',root),patch.object(entry,'OUT',meta),patch.object(entry,'bytes_used',return_value=0),\
             patch.object(entry,'check',return_value=(frame,manifest,policy,{s:frame for s in ['train','select','calibration','test']})),\
             patch.dict('sys.modules',{'lightgbm':dummy}),patch.object(linear_module,'QuantileRegressor',DummyLinear):
            if fail:
                with self.assertRaisesRegex(RuntimeError,'EXPECTED_FAILURE'):entry.train(authorize=True)
            else:
                entry.train(authorize=True)
                with self.assertRaisesRegex(ValueError,'NO_OVERWRITE'):entry.train(authorize=True)
        folder=root/'postclose_first_research_fit_20261017_v1'
        return json.loads((folder/'stop_record.json').read_bytes())

    def test_dummy_flow_and_terminal_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as root:
            stop=self.invoke(root)
            self.assertEqual(stop['status'],'COMPLETE_RESEARCH_FIT_NOT_DEPLOYMENT_READY')

    def test_dummy_failure_is_terminal_not_completed(self):
        with tempfile.TemporaryDirectory() as root:
            stop=self.invoke(root,True)
            self.assertEqual(stop['status'],'FAILED')


if __name__=='__main__':unittest.main()
