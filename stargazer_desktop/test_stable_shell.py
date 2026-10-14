"""Zero-network shell regression. Never start production notification threads."""
import ast
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from flask import Flask, request, jsonify
import assistant_context as context
import assistant_workflow as workflow
import assistant_runtime as runtime
import assistant_graph_worker as worker
import chat_archive
import model_slots


class ContextTests(unittest.TestCase):
    def test_actual_206_snapshot_retains_all_codes_prices_and_sources(self):
        file = Path(__file__).parent / 'dist/data/pentacles_demo/snapshot.json'
        value = json.loads(file.read_text(encoding='utf-8'))
        table = context.candidate_table(value)
        self.assertEqual(len(table['rows']), len(value['rows']))
        for original, row in zip(value['rows'], table['rows']):
            self.assertEqual(row[:4], [original.get(k) for k in table['columns'][:4]])
            self.assertEqual(row[4:7], [(original.get('zone') or {}).get(k) for k in table['columns'][4:7]])
        messages = context.bounded_messages([{'role':'system','content':'人工确认，不下单'},
            {'role':'user','content':context.serialized(table)}], [{'role':'user','content':'制定计划'}])
        self.assertLess(len(context.serialized(messages)), 58000)

    def test_long_history_preserves_all_user_conditions(self):
        history = []
        for i in range(30):
            history += [{'role':'user','content':f'条件{i}：价格不改，不自动买入'},
                        {'role':'assistant','content':'旧说明' * 800}]
        value = context.bounded_messages([{'role':'system','content':'持仓硬风控不可撤销'}], history)
        self.assertLess(len(context.serialized(value)), 58000)
        for i in range(30):
            self.assertIn(f'条件{i}：', context.serialized(value))
        self.assertEqual(value[0]['content'], '持仓硬风控不可撤销')
        self.assertEqual(len(history), 60)

    def test_tool_calls_keep_matching_responses(self):
        group = [{'role':'user','content':'查行情'},
                 {'role':'assistant','tool_calls':[{'id':'a'}]},
                 {'role':'tool','tool_call_id':'a','content':'{}'},
                 {'role':'assistant','content':'人工复核'}]
        self.assertEqual(context.bounded_messages([], group), group)

    def test_incomplete_tool_chain_keeps_user_but_not_orphan_call(self):
        value = context.bounded_messages([], [{'role':'user','content':'我的条件不可删除'},
                        {'role':'assistant','tool_calls':[{'id':'a'}]}])
        self.assertIn('我的条件不可删除', context.serialized(value))
        self.assertNotIn('tool_calls', context.serialized(value))

    def test_overlong_user_conditions_refused_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, '原记录保留'):
            context.bounded_messages([], [{'role':'user','content':'条件' * 40000}])


class StorageAndGraphTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.p = patch('paths.APP_DIR', str(self.base))
        self.p.start()

    def tearDown(self):
        self.p.stop()
        self.tmp.cleanup()

    def test_restart_and_new_session_preserve_old_originals(self):
        file, messages = chat_archive.open_session({'role':'system','content':'old'})
        chat_archive.append(file, [{'role':'user','content':'原始条件'}])
        before = file.read_bytes()
        again, messages = chat_archive.open_session({'role':'system','content':'new'})
        self.assertEqual(file, again)
        self.assertEqual(messages[0]['content'], 'new')
        self.assertEqual(messages[1]['content'], '原始条件')
        newer, _ = chat_archive.open_session({'role':'system','content':'new'}, fresh=True)
        self.assertNotEqual(file, newer)
        self.assertEqual(before, file.read_bytes())

    def test_large_tool_result_valid_json_and_full_evidence(self):
        file, _ = chat_archive.open_session({'role':'system','content':'s'})
        result = {'prices': ['完整原始字段'] * 5000}
        packed = json.loads(chat_archive.tool_content(result, file))
        self.assertEqual(packed['status'], 'RESULT_ARCHIVED_NOT_LOADED')
        evidence = json.loads((workflow.root() / f'tool_evidence_{packed["evidence_id"]}.json').read_text(encoding='utf-8'))
        self.assertEqual(evidence['result'], result)

    def test_archive_budget_refuses_without_destroying_history(self):
        file, _ = chat_archive.open_session({'role':'system','content':'s'})
        before = file.read_bytes()
        with patch.object(workflow, 'LIMIT', 1), self.assertRaises(ValueError):
            chat_archive.append(file, [{'role':'user','content':'x'}])
        self.assertEqual(before, file.read_bytes())

    def body(self, kind='plan'):
        draft = workflow.store(kind, {'snapshot_hash':'s', 'snapshot':{'monitor_date':20990101},
                                     'created_at': datetime.now().isoformat()})
        return {'root':str(workflow.root()), 'draft_id':draft['id'], 'kind':kind,
                'action':'prepare', 'expires_at':'2099-01-01T09:30:00+08:00',
                'now':'2026-10-14T20:00:00+08:00', 'approved':True}

    def test_real_sqlite_restart_human_confirm_and_idempotence(self):
        body = self.body()
        self.assertEqual(worker.run(body)['decision'], 'WAITING_FOR_HUMAN')
        body['action'] = 'confirm'
        first = worker.run(body)
        self.assertEqual(first['decision'], 'HUMAN_CONFIRMED_OBSERVATION_ONLY')
        self.assertEqual(worker.run(body), first)

    def test_expired_graph_does_not_approve(self):
        body = self.body()
        worker.run(body)
        body.update(action='confirm', now='2099-01-01T09:31:00+08:00')
        self.assertEqual(worker.run(body)['decision'], 'STALE_REQUIRES_NEW_DRAFT')

    def test_review_has_independent_thread_and_human_confirmation(self):
        body = self.body('review')
        self.assertEqual(worker.run(body)['decision'], 'WAITING_FOR_HUMAN')
        body.update(action='confirm', approved=False)
        self.assertEqual(worker.run(body)['decision'], 'REJECTED')

    def test_graph_refuses_tampered_business_draft(self):
        body = self.body()
        (workflow.root() / f'plan_{body["draft_id"]}.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'DRAFT_HASH'):
            worker.run(body)

    def test_missing_graph_environment_fails_closed_only_for_assistant(self):
        draft = {'id':'a'*64, 'snapshot':{'monitor_date':20990101}}
        with self.assertRaisesRegex(ValueError, '持仓风控不受影响'):
            runtime.gate('plan', draft, approved=True)

    def test_isolated_worker_subprocess_restart_and_confirmation(self):
        root = workflow.root()
        body = self.body()
        draft = {'id': body['draft_id'], **workflow.load_draft('plan', body['draft_id'])}
        desktop = str(Path(__file__).resolve().parent)
        with patch('paths.APP_DIR', desktop), patch.object(workflow, 'root', return_value=root):
            first = runtime.gate('plan', draft)
            self.assertEqual(first['decision'], 'WAITING_FOR_HUMAN')
            second = runtime.gate('plan', draft, approved=True)
            self.assertEqual(second['decision'], 'HUMAN_CONFIRMED_OBSERVATION_ONLY')

    def test_actual_plan_request_206_candidates_and_full_dialogue_archive(self):
        value = json.loads(Path(__file__).parent.joinpath('dist/data/pentacles_demo/snapshot.json').read_text(encoding='utf-8'))
        value['monitor_date'] = 20990101
        def complete(cfg, messages, **kwargs):
            self.assertLess(len(json.dumps(messages, ensure_ascii=False)), 60000)
            table = json.loads(messages[1]['content'])['candidates']
            self.assertEqual(len(table['rows']), len(value['rows']))
            answer = json.dumps({'reply':'人工确认', 'selected_codes':[value['rows'][0]['code']]})
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=answer))])
        with patch.object(workflow, 'snapshot', return_value=value), patch.object(workflow, 'complete', side_effect=complete):
            first = workflow.plan_chat({}, '条件1')
            second = workflow.plan_chat({}, '条件2', first['draft']['id'])
        self.assertEqual(len(second['draft']['dialogue']), 4)
        self.assertEqual(workflow.load_draft('plan', first['draft']['id'])['dialogue'][0]['content'], '条件1')

    def test_actual_brain_restart_uses_bounded_view_and_keeps_full_turns(self):
        tree = ast.parse(Path(__file__).parent.joinpath('agent/brain.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Brain')
        ns = {'load_config': lambda: {'agent':{'max_tool_rounds':1}},
              'load_system_prompt':lambda cfg:'人工确认，不自动交易',
              'TOOLS':[], 'json':json}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'brain.py', 'exec'), ns)
        def complete(cfg, messages, **kwargs):
            self.assertLess(len(context.serialized(messages)), 58000)
            msg = types.SimpleNamespace(content='解释' * 1200, tool_calls=None,
                model_dump=lambda **kwargs: {'role':'assistant','content':'解释' * 1200})
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
        with patch('deepseek_api.complete', side_effect=complete):
            brain = ns['Brain']()
            for i in range(12):
                brain.chat(f'原始条件{i}')
        restored = ns['Brain']()
        self.assertEqual(restored.archive, brain.archive)
        self.assertEqual(len(restored.messages), 25)
        self.assertEqual(restored.messages[1]['content'], '原始条件0')


class ShellTests(unittest.TestCase):
    def test_snapshot_v2_slots_no_fake_risk_or_information(self):
        value = model_slots.slots({'schema':'PAGE_OF_PENTACLES_DEMO_V1', 'rows':[]})
        self.assertFalse(value['interaction_enabled'])
        self.assertFalse(value['trading_authority'])
        for slot in ('information', 'risk'):
            self.assertEqual(value['models'][slot]['header']['quality_state'], 'UNKNOWN')
            self.assertEqual(value['models'][slot]['payload'], {})
            self.assertIsNone(value['models'][slot]['prediction'])

    def test_unavailable_selection_not_claimed_ready(self):
        self.assertEqual(model_slots.slots({'schema':'UNAVAILABLE','rows':[]})['models']['selection']['header']['quality_state'], 'UNKNOWN')

    def test_navigation_retirement_and_holdings_controls(self):
        from bs4 import BeautifulSoup
        html = BeautifulSoup(Path(__file__).with_name('web').joinpath('index.html').read_text(encoding='utf-8'), 'html.parser')
        self.assertEqual([b.get('data-page') for b in html.select('#mainNav .nav-item')], ['monitor','pentacles','chat','context'])
        self.assertIsNone(html.find(id='pmSaveBtn'))
        for identifier in ('addPosBtn','addWatchBtn','addTradeBtn','swSaveBtn','pentaclesLearning'):
            self.assertIsNotNone(html.find(id=identifier))
        self.assertTrue(html.find(id='pagePlan').has_attr('hidden'))
        self.assertIsNotNone(html.find(id='pageMonitor').find(id='overviewPositions'))

    def test_legacy_mutation_routes_refuse_without_touching_services(self):
        tree = ast.parse(Path(__file__).with_name('server.py').read_text(encoding='utf-8-sig'))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'retire_legacy_mutations']
        app = Flask('shell-retirement')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'server.py', 'exec'), dict(app=app, request=request, jsonify=jsonify))
        for url in ('/api/plan/settle','/api/plan/draft','/api/plan/freeze_export','/api/plan/dual_list/run'):
            self.assertEqual(app.test_client().post(url).status_code, 410)

    def test_holdings_structure_stop_and_watch_target_remain_active(self):
        tree = ast.parse(Path(__file__).with_name('notifier.py').read_text(encoding='utf-8-sig'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compose_stop_watch')
        services = types.ModuleType('services')
        services.wl_load = lambda cfg: {'positions':[{'code':'000001','low250':10}],
                                      'watch':[{'code':'000002','target_buy':20}]}
        market = types.SimpleNamespace(quotes=lambda codes: {
            '000001':{'price':9,'pct':-2,'name':'A'}, '000002':{'price':19,'pct':-1,'name':'B'}})
        tools = types.ModuleType('tools')
        tools.market = market
        ns = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'notifier.py', 'exec'), ns)
        with patch.dict('sys.modules', {'services':services, 'tools':tools}):
            alerts = ns['compose_stop_watch']({'stable_shell':True})
        self.assertEqual({key for key, _ in alerts}, {('000001','stop'),('000002','buy_trigger')})

    def test_minute_monitor_keeps_holdings_watch_quotes_without_legacy_models(self):
        import intraday_monitor
        import services
        from tools import market
        with tempfile.TemporaryDirectory() as directory:
            cfg = {'stable_shell':True, 'model':{'intraday_db':str(Path(directory) / 'minute.sqlite')}}
            wl = {'positions':[{'code':'000001','low250':10}],
                  'watch':[{'code':'000002','target_buy':20}]}
            quotes = {'000001':{'price':9,'pct':-2,'name':'A'},
                      '000002':{'price':19,'pct':-1,'name':'B'}}
            with patch.object(intraday_monitor.ml_bridge, 'advice', side_effect=AssertionError('legacy called')), \
                 patch.object(services, 'wl_load', return_value=wl), patch.object(market, 'quotes', return_value=quotes) as query:
                self.assertEqual(intraday_monitor.IntradayMonitor(lambda:cfg).snapshot_once(), 1)
                self.assertEqual(set(query.call_args.args[0]), {'000001','000002'})


if __name__ == '__main__':
    unittest.main()
