from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

import plan_evidence
import postclose_risk_runtime as runtime
from oracles_v1 import EvidencePack
from test_shadow_ui import pack_dict
import golden_rules


class PostcloseRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.root.joinpath('run').mkdir()
        self.patch=patch.object(runtime,'directory',return_value=self.root)
        self.patch.start()
        runtime._cache.clear()
        self.snapshot={'data_date':20261021,'monitor_date':20261022,
                       'rows':[{'code':'000001','name':'合成验收','reference_price':10.}]}
        raw=json.dumps(self.snapshot).encode()
        self.root.joinpath('run/input_snapshot.json').write_bytes(raw)
        pack=pack_dict()
        pack.update(market_date='20261021',asof_ts='20261021150000',query_ts='20261021180000')
        pack['candidates'][0]['code']='000001'
        pack['risk_context']['asof_ts']=pack['asof_ts']
        body=(json.dumps(pack)+'\n').encode()
        self.root.joinpath('run/packs.jsonl').write_bytes(body)
        self.state={'status':'COMPLETE','acceptance_status':'PASS','data_date':20261021,
            'run_dir':str(self.root/'run'),'snapshot_sha256':hashlib.sha256(raw).hexdigest(),
            'packs_path':str(self.root/'run/packs.jsonl'),'packs_sha256':hashlib.sha256(body).hexdigest()}
        self.write_state()

    def write_state(self):
        self.root.joinpath('state.json').write_text(json.dumps(self.state))

    def tearDown(self):
        self.patch.stop()
        runtime._cache.clear()
        self.temp.cleanup()

    def test_real_plan_prompt_consumes_same_snapshot_pack(self):
        with patch.object(plan_evidence.shadow_ui,'find_pack',side_effect=AssertionError('NO_HISTORY')):
            result=plan_evidence.for_snapshot(self.snapshot)
        self.assertEqual(result['000001']['risk']['status'],'AVAILABLE')
        self.assertEqual(result['000001']['risk']['asof_ts'],'20261021150000')

    def test_changed_candidate_population_rejected(self):
        changed={**self.snapshot,'rows':[{'code':'000002'}]}
        pack,reason=runtime.find_pack('000001',changed)
        self.assertIsNone(pack)
        self.assertIn('SNAPSHOT_CHANGED',reason)

    def test_tampered_pack_rejected(self):
        self.root.joinpath('run/packs.jsonl').write_bytes(b'{}\n')
        self.assertIn('HASH_MISMATCH',runtime.find_pack('000001',self.snapshot)[1])

    def test_running_does_not_borrow_historical_score(self):
        self.state.update(status='RUNNING',stage='CAPTURING_MINUTES',done=20,total=100)
        self.write_state()
        with patch.object(plan_evidence.shadow_ui,'find_pack',side_effect=AssertionError('NO_HISTORY')):
            self.assertEqual(plan_evidence.for_snapshot(self.snapshot)['000001']['risk']['status'],'MISSING')
            self.assertIn('后台重建 20/100',plan_evidence.status(self.snapshot)['risk'])

    def test_cutoff_mismatch_rejected(self):
        file=self.root/'run/packs.jsonl'
        p=json.loads(file.read_bytes());p['asof_ts']='20261021110000'
        body=json.dumps(p).encode();file.write_bytes(body)
        self.state['packs_sha256']=hashlib.sha256(body).hexdigest();self.write_state()
        self.assertIn('CUTOFF_MISMATCH',runtime.find_pack('000001',self.snapshot)[1])

    def test_other_day_not_applicable(self):
        self.assertEqual(runtime.find_pack('000001',{**self.snapshot,'data_date':20261020}),(None,None))

    def publish_then_rebuild(self):
        folder=self.root/'run_accepted';folder.mkdir()
        parent={**self.snapshot,'created_at':'2026-10-21T18:00:00','inputs':[]}
        raw=json.dumps(parent).encode()
        (folder/'input_snapshot.json').write_bytes(raw)
        body=(self.root/'run/packs.jsonl').read_bytes()
        (folder/'accepted_packs.jsonl').write_bytes(body)
        record={**self.state,'run_dir':str(folder),'snapshot_sha256':hashlib.sha256(raw).hexdigest(),
                'packs_path':str(folder/'accepted_packs.jsonl')}
        (folder/'independent_acceptance.json').write_text(json.dumps(record))
        self.state={'status':'RUNNING','stage':'CAPTURING_MINUTES','data_date':20261021}
        self.write_state()
        return parent,record

    def test_running_rebuild_does_not_hide_matching_published_evidence(self):
        parent,record=self.publish_then_rebuild()
        repeated={**parent,'created_at':'2026-10-21T21:00:00'}
        pack,reason=runtime.find_pack('000001',repeated)
        self.assertIsNotNone(pack);self.assertIsNone(reason)
        self.assertFalse(runtime.needs_rebuild(repeated))
        self.state['status']='FAILED';self.write_state()
        self.assertIsNotNone(runtime.find_pack('000001',repeated)[0])
        self.assertFalse(runtime.needs_rebuild(repeated))

    def test_regenerated_golden_parent_timestamp_is_not_new_market_data(self):
        parent,record=self.publish_then_rebuild()
        repeated={**parent,'created_at':'2026-10-21T21:12:00'}
        rule={'eligible':True,'f618':9.,'atr_ref':1.,'cycle_distance_atr':.1}
        child=golden_rules.project(repeated,{'000001':rule})
        self.assertFalse(golden_rules.is_projection_of(child,parent))  # exact original bug
        self.assertTrue(runtime.snapshot_bound(record,child))
        self.assertIsNotNone(runtime.find_pack('000001',child)[0])
        self.assertFalse(runtime.needs_rebuild(child))

    def test_changed_real_inputs_cannot_use_last_good_snapshot(self):
        parent,_=self.publish_then_rebuild()
        for field,value in [('data_date',20261020),('inputs',[{'path':'changed','sha256':'new'}])]:
            changed={**parent,field:value}
            self.assertIsNone(runtime.find_pack('000001',changed)[0])
        changed={**parent,'rows':[dict(parent['rows'][0],reference_price=10.01)]}
        self.assertIsNone(runtime.find_pack('000001',changed)[0])

    def test_corrupt_last_good_pack_still_rejected_while_rebuilding(self):
        parent,record=self.publish_then_rebuild()
        Path(record['packs_path']).write_bytes(b'{}')
        pack,reason=runtime.find_pack('000001',parent)
        self.assertIsNone(pack);self.assertIn('HASH_MISMATCH',reason)

    def test_launcher_repeat_selection_makes_zero_worker_calls(self):
        parent,_=self.publish_then_rebuild()
        repeated={**parent,'created_at':'2026-10-21T21:00:00'}
        source=self.root/'snapshot.json';source.write_text(json.dumps(repeated))
        import pentacles_demo,index_minutes_v1
        with patch.object(runtime,'_thread',None),patch.object(runtime.threading,'Thread') as thread:
            runtime.start()
            target=thread.call_args.kwargs['target']
        with patch.object(pentacles_demo,'snapshot_path',return_value=source),patch.object(
                index_minutes_v1,'get_index_freshness',return_value={'last_date':20261021,'stale':False}),patch.object(
                runtime,'datetime') as clock,patch.object(runtime.subprocess,'run',side_effect=AssertionError('NO_REBUILD')) as run,patch(
                'time.sleep',side_effect=KeyboardInterrupt):
            clock.now.return_value=datetime(2026,10,21,16,0)
            with self.assertRaises(KeyboardInterrupt):
                target()
            run.assert_not_called()

    def test_prompt_keeps_amplitude_and_missingness_without_audit_bloat(self):
        evidence=plan_evidence.for_snapshot(self.snapshot)['000001']['risk']
        evidence['context']['feature_missing']={'232':['field_'+str(i) for i in range(100)]}
        compact=plan_evidence.prompt_risk(evidence)
        self.assertEqual(compact['context']['amp_232'],evidence['context']['amp_232'])
        self.assertEqual(compact['context']['feature_missing_counts']['232'],100)
        self.assertNotIn('feature_missing',compact['context'])
        self.assertEqual(len(evidence['context']['feature_missing']['232']),100)


if __name__=='__main__':
    unittest.main()
