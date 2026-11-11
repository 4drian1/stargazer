"""Isolated tests: never import production server or call an external API."""
import ast
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from flask import Flask, jsonify, request
import assistant_workflow as workflow
import deepseek_api
import pentacles_demo
import pentacles_review_data
from seven_day_review import evaluate


def response(value):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(
        content=json.dumps(value)))])


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.patch = patch('paths.APP_DIR', str(self.base))
        self.patch.start()
        self.tomorrow = int((datetime.now() + timedelta(days=1)).strftime('%Y%m%d'))
        self.snap = {'schema': 'PAGE_OF_PENTACLES_DEMO_V1', 'plan_ready': True,
                     'data_date': int(datetime.now().strftime('%Y%m%d')), 'monitor_date': self.tomorrow,
                     'rows': [{'code': '000001', 'name': 'A', 'setup': 'A_SDP',
                               'reference_price': 10., 'zone': {'zone_upper': 9.5, 'zone_lower': 9.}}]}
        file = self.base / 'data/pentacles_demo/snapshot.json'
        file.parent.mkdir(parents=True)
        file.write_text(json.dumps(self.snap), encoding='utf-8')
        workflow.JOB.update(running=False, error=None, draft=None)

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def draft(self, codes=None):
        with patch.object(workflow, 'complete', return_value=response({'reply': '先观察，不自动买入',
                                                                      'selected_codes': codes or ['000001']})):
            return workflow.plan_chat({}, '制定次日计划')['draft']

    def test_draft_does_not_enable_or_write_monitor(self):
        d = self.draft()
        self.assertTrue(workflow.load_draft('plan', d['id']))
        self.assertFalse((self.base / 'data/pentacles_demo/assistant_plan.json').exists())
        self.assertFalse((self.base / 'data/pentacles_demo/monitor.json').exists())

    def test_confirmation_and_idempotence(self):
        d = self.draft()
        with self.assertRaises(ValueError):
            workflow.confirm_plan(d['id'], 'true')
        result = workflow.confirm_plan(d['id'], True)
        self.assertEqual(result['count'], 1)
        workflow.confirm_plan(d['id'], True)
        self.assertEqual(len(list(workflow.root().glob('confirmed_*.json'))), 1)
        self.assertEqual(pentacles_demo.monitoring_plan()['assistant_plan_id'], d['id'])

    def test_snapshot_change_rejects_confirmation(self):
        d = self.draft()
        self.snap['rows'][0]['reference_price'] = 11
        (self.base / 'data/pentacles_demo/snapshot.json').write_text(json.dumps(self.snap))
        with self.assertRaises(ValueError):
            workflow.confirm_plan(d['id'], True)

    def test_unknown_stock_rejected(self):
        with self.assertRaises(ValueError):
            self.draft(['999999'])

    def test_tamper_and_path_traversal_rejected(self):
        d = self.draft()
        (workflow.root() / f'plan_{d["id"]}.json').write_text('{}')
        with self.assertRaises(ValueError):
            workflow.load_draft('plan', d['id'])
        with self.assertRaises(ValueError):
            workflow.load_draft('plan', '../config')

    def test_api_failure_no_draft_written(self):
        with patch.object(workflow, 'complete', side_effect=ValueError('API不可用')):
            with self.assertRaises(ValueError):
                workflow.plan_chat({}, '计划')
        self.assertFalse(list(workflow.root().glob('plan_*.json')))

    def test_recovery_and_historical_expiry(self):
        d = self.draft()
        self.assertEqual(workflow.latest('plan')['id'], d['id'])
        self.snap['monitor_date'] = 20200101
        (self.base / 'data/pentacles_demo/snapshot.json').write_text(json.dumps(self.snap))
        with self.assertRaises(ValueError):
            workflow.plan_chat({}, '计划')

    def test_review_append_only_and_dedup(self):
        learning = self.base / 'PENTACLES_LEARNING.md'
        learning.write_text('旧篇不可改\n', encoding='utf-8')
        old_bytes = learning.read_bytes()
        d = workflow.store('review', {'day': 20261013, 'facts': {'data_date': 20261012,
             'observe_date': 20261013, 'candidate_count': 1,
             'summary': {'valid_daily_rows': 1, 'daily_touch_count': 1, 'touch_close_mean_pct': -2}},
             'commentary': '只观察，未证明收益'})
        with self.assertRaises(ValueError):
            workflow.confirm_review(d['id'], False)
        workflow.confirm_review(d['id'], True)
        before = learning.read_bytes()
        self.assertTrue(before.startswith(old_bytes))
        self.assertTrue(workflow.confirm_review(d['id'], True)['duplicate'])
        self.assertEqual(before, learning.read_bytes())

    def test_actual_api_routes_without_server_threads(self):
        tree = ast.parse(Path(__file__).with_name('server.py').read_text(encoding='utf-8-sig'))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith('api_assistant_')]
        app = Flask('assistant-tests')
        ns = dict(app=app, request=request, jsonify=jsonify, assistant_workflow=workflow,
                  deepseek_api=deepseek_api, get_cfg=lambda: {}, pentacles_demo=pentacles_demo)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'server.py', 'exec'), ns)
        client = app.test_client()
        self.assertEqual(client.get('/api/assistant/state').status_code, 200)
        self.assertEqual(client.post('/api/assistant/plan/confirm', json={'id':'../key', 'confirmed':True}).status_code, 400)
        self.assertEqual(client.post('/api/assistant/review', json={'day':20261399}).status_code, 400)

    def test_budget_no_draft(self):
        with patch.object(workflow, 'LIMIT', 1):
            with self.assertRaises(ValueError):
                self.draft()
        self.assertFalse(list(workflow.root().glob('plan_*.json')))

    def test_actual_trade_gross_not_simulation(self):
        db = self.base / 'trades.db'
        with sqlite3.connect(db) as conn:
            conn.execute('CREATE TABLE trades(id INTEGER,code TEXT,trade_date TEXT,action TEXT,price REAL,shares INTEGER)')
            conn.executemany('INSERT INTO trades VALUES(?,?,?,?,?,?)', [
                (1,'000001','2026-10-08','buy',10,100), (2,'000001','2026-10-09','sell',11,100)])
        conn.close()
        value = pentacles_review_data.actual_trades({'agent':{'trades_db':str(db)}}, {'000001'}, 20261009)
        self.assertEqual(value['realized_gross_pnl'], 100)
        self.assertIsNone(value['realized_net_pnl'])

    def test_old_draft_rejected_after_newer_discussion(self):
        first = self.draft()
        second = self.draft()
        self.assertNotEqual(first['id'], second['id'])
        with self.assertRaises(ValueError):
            workflow.confirm_plan(first['id'], True)

    def test_review_job_uses_archived_population_and_requires_confirm(self):
        value = dict(self.snap, monitor_date=20261013, data_date=20261012)
        archive = self.base / f'data/pentacles_demo/journal/plans/{workflow.digest(value)}.json'
        archive.parent.mkdir(parents=True)
        archive.write_bytes(workflow.encoded(value))
        facts = {'day':20261013,'data_date':20261012,'observe_date':20261013,
                 'rows':[],'seven_day':[],
                 'candidate_count':1,'summary':{'valid_daily_rows':1,'daily_touch_count':1,
                                               'touch_close_mean_pct':-2}}
        with (patch.object(pentacles_review_data, 'collect', return_value=facts) as read,
              patch.object(workflow, 'complete', return_value=response({'commentary':'复盘草案'}))):
            workflow.review_start({},20261013)
            for _ in range(50):
                if not workflow.JOB['running']:
                    break
                time.sleep(.01)
        self.assertIsNone(workflow.JOB['error'])
        self.assertEqual(read.call_args.args[0]['monitor_date'],20261013)
        self.assertFalse(workflow.learning_path().exists())

    def test_local_fallback_exact_day_no_stale_substitution(self):
        import struct
        source = self.base / 'sz000001.day'
        source.write_bytes(struct.pack('<IIIIIfII',20261013,1000,1100,900,1000,10000.,1000,0))
        with patch('tdx_daily_source_v2.day_path', return_value=source):
            self.assertEqual(len(pentacles_review_data.local_day({}, {'000001'},20261013)[0]),1)
            self.assertEqual(len(pentacles_review_data.local_day({}, {'000001'},20261014)[0]),0)

    def test_fallback_duplicate_conflict_blocks(self):
        import struct
        source = self.base / 'sz000001.day'
        source.write_bytes(b''.join(struct.pack('<IIIIIfII',20261013,1000,1200,900,p,10000.,1000,0) for p in [1000,1100]))
        with patch('tdx_daily_source_v2.day_path', return_value=source):
            values, evidence = pentacles_review_data.local_day({}, {'000001'},20261013)
        self.assertFalse(values)
        self.assertEqual(evidence['missing'][0]['code'],'000001')
        self.assertEqual(evidence['missing'][0]['reason'],'TDX_DAY_ORDER_OR_DUPLICATE')


class SevenDayTests(unittest.TestCase):
    def setUp(self):
        self.calendar = [20261008,20261009,20261012,20261013,20261014,20261015,20261016]
        self.record = {'entry_ts':20261008093600, 'entry_price':10,
                       'days':{str(d):{'high':12,'low':8,'close':11} for d in self.calendar}}

    def test_unverified_scale_null(self):
        value = evaluate(self.record, self.calendar, self.calendar[-1])
        self.assertIsNone(value['simulated_net_return_pct'])
        self.assertEqual(value['status'], 'SCALE_UNVERIFIED')

    def test_pending_never_zero(self):
        value = evaluate(self.record, self.calendar, 20261012)
        self.assertEqual(value['observed_sessions'], 3)
        self.assertIsNone(value['day7_reference_return_pct'])

    def test_fixed_exit_not_maximum(self):
        self.record.update(scale_verified=True, corporate_action_verified=True,
                           entry_execution_verified=True, day7_exit_execution_verified=True,
                           buy_cost_rate=.0013, sell_cost_rate=.0013)
        value = evaluate(self.record, self.calendar, self.calendar[-1])
        self.assertAlmostEqual(value['day7_reference_return_pct'], 10.)
        self.assertAlmostEqual(value['mfe_pct'], 20.)
        self.assertAlmostEqual(value['simulated_net_return_pct'], (11*.9987/(10*1.0013)-1)*100)
        self.assertIsNone(value['actual_pnl'])

    def test_missing_day_not_deleted(self):
        del self.record['days']['20261012']
        self.assertEqual(evaluate(self.record, self.calendar, self.calendar[-1])['status'], 'DATA_INSUFFICIENT')

    def test_nan_no_return(self):
        self.record['days']['20261016']['close'] = float('nan')
        self.assertEqual(evaluate(self.record, self.calendar, self.calendar[-1])['status'], 'DATA_INSUFFICIENT')

    def test_pre_entry_extrema_excluded(self):
        self.record['scale_verified'] = True
        self.record['days']['20261008']['high'] = 1000
        value = evaluate(self.record, self.calendar, self.calendar[-1])
        self.assertAlmostEqual(value['mfe_pct'], 20.)


class GraphTests(unittest.TestCase):
    def test_persistent_interrupt_restart_and_stale(self):
        from assistant_graph_shadow import build_graph
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.types import Command
        with tempfile.TemporaryDirectory() as directory:
            file = str(Path(directory) / 'checkpoint.sqlite')
            config = {'configurable': {'thread_id':'test-plan'}}
            with SqliteSaver.from_conn_string(file) as saver:
                graph = build_graph(saver)
                result = graph.invoke({'snapshot_id':'s1','expires_at':'2026-10-15T09:30:00+08:00',
                                       'now':'2026-10-14T20:00:00+08:00',
                                       'towers':{'risk':'NOT_READY'}, 'draft':{'codes':['000001']}}, config)
                self.assertIn('__interrupt__', result)
            with SqliteSaver.from_conn_string(file) as saver:
                graph = build_graph(saver)
                result = graph.invoke(Command(resume={'approved':True,'now':'2026-10-15T09:31:00+08:00'}), config)
                self.assertEqual(result['decision'], 'STALE_REQUIRES_NEW_DRAFT')

    def test_thin_and_graph_decision_equal(self):
        from assistant_graph_shadow import build_graph, deterministic_decision
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import Command
        state = {'snapshot_id':'s','now':'2026-10-14T19:00:00+08:00',
                 'expires_at':'2026-10-15T09:30:00+08:00','draft':{},'towers':{}}
        graph = build_graph(InMemorySaver())
        cfg = {'configurable':{'thread_id':'fresh'}}
        graph.invoke(state, cfg)
        result = graph.invoke(Command(resume={'approved':True,'now':state['now']}), cfg)
        self.assertEqual(result['decision'], deterministic_decision(state, True))


if __name__ == '__main__':
    unittest.main()
