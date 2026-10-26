"""Bridge/HTTP isolation checks. No production threads, real worker or network."""
import ast
from datetime import datetime
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from flask import Flask, jsonify
import joint_risk_bridge as bridge


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='risk_bridge_')
        self.addCleanup(self.tmp.cleanup)
        self.patches = [patch('paths.APP_DIR', self.tmp.name),
                       patch.object(bridge, '_worker_paths', return_value=(Path(__file__), Path(__file__)))]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        bridge._running, bridge._current_key, bridge._last = False, None, None
        self.cfg = dict(joint_shadow=dict(enabled=True))
        self.plan = dict(plan_ready=True, data_date=20261015, monitor_date=20261016,
                         rows=[dict(code='000001'), dict(code='000002')])
        self.now = datetime(2026,10,16,11,10,tzinfo=bridge.SHANGHAI)

    def finish(self):
        end = time.monotonic()+3
        while bridge.state(self.cfg)['running'] and time.monotonic()<end:
            time.sleep(.01)
        self.assertFalse(bridge.state(self.cfg)['running'])
        return bridge.state(self.cfg)['last']

    def success(self):
        rows = [dict(code=c, P=20261015, E=20261016, feature_cutoff=20261016110000, risk_horizon=120)
                for c in ('000001','000002')]
        return SimpleNamespace(returncode=0, stdout=json.dumps(dict(ok=True, status='SHADOW_OBSERVATION',
            schema='JOINT_SHADOW_PACKET_V1', risk_records=rows, decision_authority=False,
            llm_called=False)).encode(), stderr=b'')

    def test_wrong_horizon_child_result_not_admitted(self):
        proc = self.success()
        result = json.loads(proc.stdout)
        result['risk_records'][0]['risk_horizon'] = 1200
        proc.stdout = json.dumps(result).encode()
        with patch.object(bridge.subprocess, 'run', return_value=proc):
            bridge.run(self.cfg, self.plan, now=self.now)
            self.assertEqual(self.finish()['result']['status'], 'WORKER_ERROR')

    def test_wrong_candidate_child_result_not_admitted(self):
        proc = self.success()
        result = json.loads(proc.stdout)
        result['risk_records'][0]['code'] = '999999'
        proc.stdout = json.dumps(result).encode()
        with patch.object(bridge.subprocess, 'run', return_value=proc):
            bridge.run(self.cfg, self.plan, now=self.now)
            self.assertEqual(self.finish()['result']['status'], 'WORKER_ERROR')

    def test_disabled_never_starts_process(self):
        with patch.object(bridge.subprocess, 'run', side_effect=AssertionError('no spawn')):
            self.assertEqual(bridge.run({}, self.plan, now=self.now)['status'], 'DISABLED')

    def test_closed_bar_clock_including_lunch(self):
        self.assertIsNone(bridge.completed_cutoff(self.now.replace(hour=9,minute=40)))
        self.assertEqual(bridge.completed_cutoff(self.now), 20261016110000)
        self.assertEqual(bridge.completed_cutoff(self.now.replace(hour=12)), 20261016113000)
        self.assertEqual(bridge.completed_cutoff(self.now.replace(hour=13,minute=10)), 20261016113000)
        self.assertIsNone(bridge.completed_cutoff(self.now.replace(day=17)))

    def test_plan_expiry_not_silently_used(self):
        self.plan['monitor_date'] = 20261015
        self.assertEqual(bridge.run(self.cfg, self.plan, now=self.now)['status'], 'PLAN_DAY_MISMATCH')

    def test_async_worker_and_restart_dedup(self):
        with patch.object(bridge.subprocess, 'run', return_value=self.success()) as mock:
            reply = bridge.run(self.cfg, self.plan, now=self.now)
            self.assertEqual(reply['status'], 'QUEUED')
            event = self.finish()
        args = mock.call_args
        self.assertIn('-B', args.args[0])
        self.assertEqual(args.kwargs['timeout'], 300)
        self.assertNotIn('shell', args.kwargs)
        sent = json.loads(args.kwargs['input'])
        self.assertEqual(sent['candidates'][0]['cutoff_ts'], 20261016110000)
        self.assertFalse(event['valid_for_automatic_decision'])
        file = bridge.directory()/'events.jsonl'
        old = file.read_bytes()
        bridge._last = None
        with patch.object(bridge.subprocess, 'run', side_effect=AssertionError('no repeat')):
            self.assertEqual(bridge.run(self.cfg, self.plan, now=self.now)['status'], 'ALREADY_RECORDED')
        self.assertEqual(file.read_bytes(), old)

    def test_timeout_isolated_and_recorded(self):
        with patch.object(bridge.subprocess, 'run', side_effect=subprocess.TimeoutExpired('worker',300)):
            bridge.run(self.cfg, self.plan, now=self.now)
            event = self.finish()
        self.assertEqual(event['result']['status'], 'TIMEOUT')

    def test_non_json_child_output_isolated(self):
        proc = SimpleNamespace(returncode=1, stdout=b'Traceback', stderr=b'private detail')
        with patch.object(bridge.subprocess, 'run', return_value=proc):
            bridge.run(self.cfg, self.plan, now=self.now)
            event = self.finish()
        self.assertEqual(event['result']['status'], 'WORKER_ERROR')
        self.assertNotIn('private detail', json.dumps(event))

    def test_output_budget_refusal(self):
        proc = SimpleNamespace(returncode=0, stdout=b'x'*(bridge.MAX_RESULT+1), stderr=b'')
        with patch.object(bridge.subprocess, 'run', return_value=proc):
            bridge.run(self.cfg, self.plan, now=self.now)
            self.assertEqual(self.finish()['result']['status'], 'WORKER_ERROR')

    def test_write_failure_does_not_claim_persisted_and_retry_works(self):
        with patch.object(bridge.subprocess, 'run', return_value=self.success()):
            with patch.object(bridge, '_persist', side_effect=OSError('disk')):
                bridge.run(self.cfg, self.plan, now=self.now)
                self.assertEqual(self.finish()['persistence_error'], 'OSError')
            self.assertEqual(bridge.run(self.cfg, self.plan, now=self.now)['status'], 'QUEUED')
            self.assertNotIn('persistence_error', self.finish())
        self.assertTrue((bridge.directory()/'events.jsonl').exists())

    def test_one_running_job_only(self):
        entered, release = threading.Event(), threading.Event()
        def slow(*args, **kwargs):
            entered.set(); release.wait(2)
            return self.success()
        with patch.object(bridge.subprocess, 'run', side_effect=slow):
            bridge.run(self.cfg, self.plan, now=self.now)
            self.assertTrue(entered.wait(1))
            try:
                self.assertEqual(bridge.run(self.cfg, self.plan, now=self.now)['status'], 'ALREADY_RUNNING')
            finally:
                release.set()
            self.finish()

    def test_thread_start_failure_not_stuck_running(self):
        with patch.object(bridge.threading.Thread, 'start', side_effect=RuntimeError('no thread')):
            result = bridge.run(self.cfg, self.plan, now=self.now)
        self.assertEqual(result['status'], 'WORKER_START_ERROR')
        self.assertFalse(bridge.state(self.cfg)['running'])

    def test_log_budget_preserves_existing(self):
        with patch.object(bridge, 'LOG_CAP', 1):
            with self.assertRaisesRegex(ValueError, 'LOG_BUDGET'):
                bridge._persist(dict(result='too big'))
        self.assertFalse((bridge.directory()/'events.jsonl').exists())

    def test_http_routes_cannot_enable_via_request_body(self):
        # Execute only these route definitions, not server startup/notification code.
        source = Path(__file__).with_name('server.py').read_text(encoding='utf-8')
        nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                 and n.name in ('api_joint_shadow_state','api_joint_shadow_run')]
        app = Flask('shadow_route_test')
        env = dict(app=app, jsonify=jsonify, get_cfg=lambda: {},
                   pentacles_demo=SimpleNamespace(monitoring_plan=lambda: self.plan))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'server_shadow_routes','exec'), env)
        client = app.test_client()
        with patch.object(bridge.subprocess, 'run', side_effect=AssertionError('no API bypass')):
            self.assertEqual(client.post('/api/joint-shadow/run', json={'enabled':True}).json['status'],'DISABLED')
            self.assertFalse(client.get('/api/joint-shadow/state').json['enabled'])


if __name__ == '__main__':
    unittest.main()
