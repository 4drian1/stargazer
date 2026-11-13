import copy,json,sys,tempfile,unittest,subprocess
from pathlib import Path
from datetime import datetime
from unittest.mock import patch,Mock
H=Path(__file__).resolve().parent;A=H.parent;sys.path.insert(0,str(A))
import postclose_risk_runtime as r
import plan_evidence as e

class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.p=patch.object(r,'directory',return_value=self.root);self.p.start()
        self.snapshot=dict(data_date=20261028,monitor_date=20261029,rows=[dict(code='002933',entry_rule=dict(policy='GOLDEN_RESONANCE_CROSSDAY_V4'))])
    def tearDown(self):self.p.stop();self.tmp.cleanup()
    def test_closed_old_observation_never_starts_even_with_ready_data(self):
        old=dict(self.snapshot,data_date=20261027,monitor_date=20261028)
        with patch.object(r.postclose_index_source,'inspect',side_effect=AssertionError('NO_SOURCE_READ')),patch.object(r.subprocess,'run') as call:
            self.assertEqual(r._decide_and_launch(old,datetime(2026,10,28,19)),'gate_closed');call.assert_not_called()
    def test_live_worker_is_never_replaced_or_restarted(self):
        activity=dict(status='RUNNING',data_date=20261027)
        with patch.object(r.postclose_index_source,'inspect',return_value=dict(status='READY')),patch.object(r,'worker_active',return_value=True),patch.object(r.subprocess,'run') as call:
            self.assertEqual(r._decide_and_launch(self.snapshot,datetime(2026,10,28,19),_state=lambda:activity),'no_action');call.assert_not_called()
    def test_stale_running_does_not_block_new_day(self):
        previous=dict(status='RUNNING',data_date=20261027,run_dir=str(self.root/'old'))
        (self.root/'old').mkdir();raw=json.dumps(dict(self.snapshot,data_date=20261027)).encode();(self.root/'old/input_snapshot.json').write_bytes(raw)
        import hashlib
        previous['snapshot_sha256']=hashlib.sha256(raw).hexdigest()
        with patch.object(r.postclose_index_source,'inspect',return_value=dict(status='READY')),patch.object(r.postclose_index_source,'prepare',return_value=(self.root/'source','sha')),patch.object(r,'worker_active',return_value=False):
            calls=[]
            def execute(command,**kw):calls.append(command);return Mock(returncode=1)
            result=r._decide_and_launch(self.snapshot,datetime(2026,10,28,19),_state=lambda:previous,_subprocess_run=execute,_worker_paths=lambda:(Path(sys.executable),self.root/'worker.py'))
        self.assertEqual(result,'worker_failed');self.assertEqual(len(calls),1)
        self.assertEqual(json.loads((self.root/'state.json').read_bytes())['status'],'INTERRUPTED')
        self.assertEqual(len(list(self.root.glob('interrupted_*.json'))),1)
    def test_real_lock_probe_sees_other_process_and_unlock(self):
        file=self.root/'worker.lock';file.write_bytes(b'0')
        cmd=[sys.executable,'-B','-c',"import msvcrt,sys;f=open(sys.argv[1],'r+b');msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);print('ready',flush=True);sys.stdin.readline()",str(file)]
        p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        try:
            self.assertEqual(p.stdout.readline().strip(),'ready');self.assertTrue(r.worker_active())
            p.communicate('\n',timeout=10);self.assertFalse(r.worker_active())
        finally:
            if p.poll() is None:p.kill();p.wait()
    def test_dynamic_progress_does_not_mutate_frozen_draft(self):
        draft=dict(snapshot=copy.deepcopy(self.snapshot),selected_codes=['002933'],model_evidence={});before=copy.deepcopy(draft)
        with patch.object(r,'state',return_value=dict(status='RUNNING',data_date=20261027,done=1,total=10)):
            a=e.presentation(draft)
        with patch.object(r,'state',return_value=dict(status='RUNNING',data_date=20261027,done=2,total=10)):
            b=e.presentation(draft)
        self.assertNotEqual(a['risk_task'],b['risk_task']);self.assertEqual(draft,before)
        self.assertIn('三项至少两项',a['entry_condition']);self.assertNotIn('F618',a['entry_condition'])
    def test_wait_reason_stable_across_progress_ticks(self):
        with patch.object(r,'state',return_value=dict(status='RUNNING',data_date=20261027,done=1,total=10)):
            a=r.wait_reason(self.snapshot)
        with patch.object(r,'state',return_value=dict(status='RUNNING',data_date=20261027,done=2,total=10)):
            self.assertEqual(a,r.wait_reason(self.snapshot));self.assertIn('排队中',a)
    def test_preopen_catchup_remains_allowed(self):
        old=dict(self.snapshot,data_date=20261027,monitor_date=20261028)
        with patch.object(r.postclose_index_source,'inspect',return_value=dict(status='READY')),patch.object(r,'needs_rebuild',return_value=False),patch.object(r,'state',return_value=dict(status='NOT_STARTED')):
            self.assertEqual(r._decide_and_launch(old,datetime(2026,10,28,8)),'no_action')
if __name__=='__main__':unittest.main()
