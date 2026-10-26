"""Same market evidence may survive a plan-map update; changed inputs may not."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import paths
import plan_board_monitor as boards
import plan_evidence
import postclose_risk_runtime as runtime

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'stargazer_desktop/dist/data'


class RiskBindingRepairTests(unittest.TestCase):
    def setUp(self):
        self.original=json.loads((DATA/'postclose_risk/run_20261023_164935_014936/input_snapshot.json').read_bytes())
        self.current=json.loads((DATA/'pentacles_demo/snapshot.json').read_bytes())

    def test_real_same_inputs_accept_map_change_and_subset(self):
        self.assertTrue(runtime.risk_inputs_equivalent(self.original,self.current))
        subset=copy.deepcopy(self.current);subset['rows']=subset['rows'][:1]
        subset.update(assistant_plan_id='confirmed',boards=[{'name':'军工'}])
        self.assertTrue(runtime.risk_inputs_equivalent(self.original,subset))

    def test_changed_price_feature_score_model_or_source_rejected(self):
        for key,value in [('reference_price',7.23),('selection_score',999),('feature_values',{'rsi14':999}),('selection_model',{'sha256':'changed'})]:
            bad=copy.deepcopy(self.current);bad['rows'][0][key]=value
            self.assertFalse(runtime.risk_inputs_equivalent(self.original,bad),key)
        for key,value in [('data_date',20261022),('monitor_date',20261027),('inputs',[{'path':'changed','sha256':'changed'}])]:
            bad={**self.current,key:value}
            self.assertFalse(runtime.risk_inputs_equivalent(self.original,bad),key)

    def test_new_stock_duplicate_or_changed_frozen_kernel_rejected(self):
        bad=copy.deepcopy(self.current);bad['rows'][0]['code']='600000'
        self.assertFalse(runtime.risk_inputs_equivalent(self.original,bad))
        bad=copy.deepcopy(self.current);bad['rows'].append(bad['rows'][0])
        self.assertFalse(runtime.risk_inputs_equivalent(self.original,bad))
        bad=copy.deepcopy(self.current)
        entry=next(v for v in bad['rule_inputs'] if Path(v['path']).name=='rebuild_features_v1.py')
        entry['sha256']='changed'
        self.assertFalse(runtime.risk_inputs_equivalent(self.original,bad))

    def test_original_snapshot_hash_still_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/'input_snapshot.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'SNAPSHOT_HASH_MISMATCH'):
                runtime.snapshot_bound(dict(run_dir=str(folder),snapshot_sha256='changed'),self.current)

    def test_frozen_module_location_does_not_change_project_binding(self):
        with patch.object(paths,'APP_DIR',str(ROOT/'stargazer_desktop/dist')),patch.object(runtime,'__file__','D:/temporary/_MEI/postclose_risk_runtime.py'):
            self.assertTrue(runtime.risk_inputs_equivalent(self.original,self.current))

    def test_confirmed_view_supplements_only_matching_plan_and_never_writes(self):
        original_plan=json.loads((DATA/'trade_plan.json').read_bytes())
        confirmed=json.loads((DATA/'pentacles_demo/assistant_plan.json').read_bytes())
        fresh={r['code']:{'risk':dict(status='AVAILABLE',asof_ts='20261023150000',context={})} for r in confirmed['rows']}
        plan=copy.deepcopy(original_plan)
        for c in fresh:plan['model_evidence'][c]['risk']={'status':'MISSING','reason':'before'}
        saved=copy.deepcopy(plan)
        with patch.object(paths,'APP_DIR',str(DATA.parent)),patch('trade_plan_v1.load_plan',return_value=plan),patch.object(plan_evidence,'for_snapshot',return_value=fresh),patch('risk_outlook_v1.latest',return_value={}):
            view=plan_evidence.model_view('risk')
            self.assertTrue(all(r['facts']['status']=='AVAILABLE' for r in view['items']))
            self.assertIn('确认时历史讨论',view['combined'])
            self.assertEqual(plan,saved)
            plan['assistant_plan_id']='other-plan'
            self.assertTrue(all(r['facts']['status']=='MISSING' for r in plan_evidence.model_view('risk')['items']))

    def test_military_alias_requires_canonical_catalog_and_not_child_sector(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'data').mkdir();catalog=root/'data/board_list.json'
            with patch.object(paths,'APP_DIR',str(root)):
                catalog.write_text(json.dumps({'boards':{'国防军工':'BK1204','军工电子':'BK1233'}}))
                self.assertEqual(boards.resolve(['军工'])[0]['bk'],'BK1204')
                catalog.write_text(json.dumps({'boards':{'军工电子':'BK1233','军工装备':'BK9999'}}))
                self.assertIsNone(boards.resolve(['军工'])[0]['bk'])
                catalog.write_text(json.dumps({'boards':{'军工电子':'BK1233'}}))
                self.assertIsNone(boards.resolve(['军工'])[0]['bk'])


if __name__=='__main__':unittest.main()
