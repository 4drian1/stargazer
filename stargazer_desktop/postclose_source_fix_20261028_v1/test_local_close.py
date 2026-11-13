import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import index_minutes_v1 as index
import postclose_index_source as local
import postclose_risk_runtime as runtime
import plan_evidence


def binary(day, count=48, close=3601.):
    updater = index._load_updater()
    packed = (day//10000-2004)*2048+(day//100%100)*100+day%100
    return b''.join(struct.pack('<HHfffffiI', packed, t//10000%100*60+t//100%100,
        3600., 3602., 3599., close, 10000., 100, 0) for t in updater.grid(day)[:count])


class LocalCloseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.files = {s:self.root/name for s,name in zip(local.SYMBOLS, ['sh999999.lc5', 'sz399006.lc5'])}
        for file in self.files.values():
            file.write_bytes(binary(20261026)+binary(20261027))
        self.source_patch = patch.object(index, 'SOURCE_FILES', self.files)
        self.source_patch.start()
        self.runtime_patch = patch.object(runtime, 'directory', return_value=self.root/'runtime')
        self.runtime_patch.start()
        self.snapshot = dict(data_date=20261027, monitor_date=20261028, rows=[dict(code='002780')])

    def tearDown(self):
        self.source_patch.stop()
        self.runtime_patch.stop()
        self.temp.cleanup()

    def test_complete_native_ignores_stale_conflicting_live_mirror(self):
        with patch.object(index, 'get_index_freshness', side_effect=AssertionError('DO_NOT_USE_LIVE_MIRROR')):
            result = local.inspect(20261027, 20261028085000)
            path, sha = local.prepare(20261027, 20261028085000, self.root/'runtime')
        self.assertEqual(result['status'], 'READY')
        frames, receipts = local.load(path, sha, 20261027)
        for symbol, frame in frames.items():
            self.assertEqual(len(frame), 96)
            self.assertEqual(frame.date.max(), 20261027150000)
            self.assertTrue(frame.volume.isna().all())
            self.assertTrue(frame.amount.isna().all())
            self.assertFalse(receipts[symbol]['volume_verified'])
            R = Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(R/'stargazer_ml/v5_hybrid/risk_r0_v1'))
            from joint_shadow_context_v1 import _prefix
            accepted = _prefix(frame, 20261027150000, receipts[symbol], observed_at=20261028085000)
            self.assertEqual(accepted.date.max(), 20261027150000)
            self.assertEqual(receipts[symbol]['time_semantics'], 'END_5M_ASSUMED_NOT_CERTIFIED')
        before = {p:p.read_bytes() for p in path.parent.iterdir()}
        self.assertEqual(local.prepare(20261027, 20261028085100, self.root/'runtime'), (path, sha))
        self.assertTrue(all(p.read_bytes() == raw for p,raw in before.items()))

    def test_missing_one_index_cannot_launch(self):
        self.files['399006.SZ'].write_bytes(binary(20261026)+binary(20261027, 47))
        result = local.inspect(20261027, 20261028085000)
        self.assertEqual(result['status'], 'WAIT_LOCAL_INDEX')
        self.assertIn('399006.SZ:20261027:bars=47', result['reason'])
        self.assertIn('LOCAL_INDEX_DAY_INCOMPLETE', runtime.wait_reason(self.snapshot))
        run = unittest.mock.Mock()
        with patch.object(index, 'get_index_freshness', return_value={'stale':True}):
            result = runtime._decide_and_launch(self.snapshot, datetime(2026,10,28,8,50), _subprocess_run=run)
        self.assertEqual(result, 'gate_closed')
        run.assert_not_called()

    def test_no_future_or_in_progress_day_admitted(self):
        self.assertEqual(local.inspect(20261027, 20261027145959)['status'], 'WAIT_LOCAL_INDEX')
        self.assertFalse(local.job_due(self.snapshot, datetime(2026,10,27,14,59)))
        self.assertTrue(local.job_due(self.snapshot, datetime(2026,10,27,15)))
        self.assertTrue(local.job_due(self.snapshot, datetime(2026,10,28,8,50)))
        self.assertFalse(local.job_due(self.snapshot, datetime(2026,10,29,8,50)))
        for file in self.files.values():
            file.write_bytes(file.read_bytes()+binary(20261028))
        path, sha = local.prepare(20261027, 20261028085000, self.root/'runtime')
        frames, _ = local.load(path, sha, 20261027)
        self.assertTrue(all(f.date.max() == 20261027150000 for f in frames.values()))

    def test_bad_prices_order_and_gaps_fail(self):
        file = self.files['000001.SH']; valid = file.read_bytes()
        for broken in [binary(20261026)+binary(20261027,close=9999.), valid+valid[-32:],
                       valid[:-64]+valid[-32:], binary(20261027)[32:]]:
            with self.subTest(bytes=len(broken)):
                file.write_bytes(broken)
                self.assertEqual(local.inspect(20261027, 20261028085000)['status'], 'WAIT_LOCAL_INDEX')
        file.write_bytes(valid)

    def test_source_hash_date_and_identity_enforced(self):
        path, sha = local.prepare(20261027, 20261028085000, self.root/'runtime')
        with self.assertRaises(ValueError):local.load(path, 'bad', 20261027)
        with self.assertRaises(ValueError):local.load(path, sha, 20261026)
        manifest = json.loads(path.read_bytes())
        raw = path.parent/manifest['sources']['000001.SH']['captured_file']
        raw.write_bytes(raw.read_bytes()[:-32])
        with self.assertRaises(ValueError):local.load(path, sha, 20261027)

    def test_capture_read_hook_records_all_bound_inputs(self):
        path, sha = local.prepare(20261027, 20261028085000, self.root/'runtime')
        seen = []
        def read(file, digest):
            seen.append((str(file), digest));return file.read_bytes()
        local.load(path, sha, 20261027, read=read)
        self.assertEqual(len(seen), 3)

    def test_actual_disk_state_and_scheduler_catchup_no_real_batch(self):
        root = runtime.directory();root.mkdir()
        (root/'state.json').write_text(json.dumps(dict(status='COMPLETE', data_date=20261023, acceptance_status='PASS')))
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            if 'live_postclose_job_v1.py' in command[4]:
                self.assertIn('--index-source', command)
                path = Path(command[command.index('--index-source')+1])
                digest = command[command.index('--index-source-sha')+1]
                local.load(path, digest, 20261027)
                (root/'state.json').write_text(json.dumps(dict(status='COMPLETE', data_date=20261027)))
            return subprocess.CompletedProcess(command, 0)
        with patch.object(runtime, 'needs_rebuild', return_value=True), \
             patch.object(index, 'get_index_freshness', side_effect=AssertionError('NOT_LIVE')), \
             patch.object(runtime, '_worker_paths_real', return_value=(Path('python.exe'), Path('worker.py'))):
            result = runtime._decide_and_launch(self.snapshot, datetime(2026,10,28,8,50),
                snapshot_file=self.root/'snapshot.json', _subprocess_run=run)
        self.assertEqual(result, 'launched_worker')
        self.assertEqual(len(calls), 2)
        self.assertIn('accept_live_postclose_v1.py', calls[1][4])

    def test_failed_worker_does_not_check_old_complete_state(self):
        with patch.object(runtime, 'state', return_value=dict(status='COMPLETE', data_date=20261023)), \
             patch.object(runtime, 'needs_rebuild', return_value=True), \
             patch.object(runtime, '_worker_paths_real', return_value=(Path('python.exe'), Path('worker.py'))):
            run = unittest.mock.Mock(return_value=subprocess.CompletedProcess([], 2))
            result = runtime._decide_and_launch(self.snapshot, datetime(2026,10,28,8,50), _subprocess_run=run)
            repeated = runtime._decide_and_launch(self.snapshot, datetime(2026,10,28,8,51), _subprocess_run=run)
        self.assertEqual(result, 'worker_failed')
        self.assertEqual(repeated, 'no_action')
        self.assertEqual(run.call_count, 1)

    def test_terminated_same_snapshot_never_restarts(self):
        root = runtime.directory();run_dir = root/'run_failed';run_dir.mkdir(parents=True)
        raw = json.dumps(self.snapshot).encode();(run_dir/'input_snapshot.json').write_bytes(raw)
        (root/'state.json').write_text(json.dumps(dict(status='FAILED', data_date=20261027,
            run_dir=str(run_dir), snapshot_sha256=hashlib.sha256(raw).hexdigest())))
        run = unittest.mock.Mock()
        result = runtime._decide_and_launch(self.snapshot, datetime(2026,10,28,8,50), _subprocess_run=run)
        self.assertEqual(result, 'no_action');run.assert_not_called()

    def test_actual_missing_plan_facts_explain_source_without_changing_plan(self):
        before = copy.deepcopy(self.snapshot)
        with patch.object(runtime, 'find_pack', return_value=(None, None)), \
             patch.object(plan_evidence.shadow_ui, 'find_pack', return_value=(None, None)):
            result = plan_evidence.for_snapshot(self.snapshot)
        self.assertEqual(result['002780']['risk']['status'], 'MISSING')
        self.assertIn('尚未生成并验收', result['002780']['risk']['reason'])
        self.assertEqual(before, self.snapshot)

    def test_worker_preopen_consumes_explicit_binding_and_real_calendar(self):
        R = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(R/'stargazer_ml/v5_hybrid/risk_r0_v1'))
        import live_postclose_job_v1 as worker
        path, sha = local.prepare(20261027, 20261028085000, self.root/'runtime')
        calendar = self.root/'calendar.json'
        calendar.write_text(json.dumps([20261026,20261027,20261028]))
        value = {**self.snapshot, 'inputs':[dict(path=str(calendar), sha256=hashlib.sha256(calendar.read_bytes()).hexdigest())]}
        snap = self.root/'snapshot.json';snap.write_text(json.dumps(value))
        seen = []
        def stop_before_stock_reads(*args, **kwargs):
            seen.append(kwargs['index_source'])
            raise ValueError('TEST_STOP_BEFORE_ANY_BATCH_READ')
        with patch.object(worker, 'now_stamp', return_value=20261028085000):
            result = worker.run(snap, self.root/'job_runtime', producer=stop_before_stock_reads, index_source=(path,sha))
        self.assertEqual(result, 2)
        self.assertEqual(seen, [(path,sha)])
        state = json.loads((self.root/'job_runtime/state.json').read_bytes())
        self.assertIn('TEST_STOP_BEFORE_ANY_BATCH_READ', state['reason'])

    def test_worker_keeps_historical_default_gate_without_binding(self):
        R = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(R/'stargazer_ml/v5_hybrid/risk_r0_v1'))
        import live_postclose_job_v1 as worker
        snap = self.root/'snapshot.json';snap.write_text(json.dumps(self.snapshot))
        with patch.object(worker, 'now_stamp', return_value=20261028085000), self.assertRaisesRegex(ValueError, 'POSTCLOSE_CURRENT_DAY_REQUIRED'):
            worker.run(snap, self.root/'job_runtime')


if __name__ == '__main__':
    unittest.main()
