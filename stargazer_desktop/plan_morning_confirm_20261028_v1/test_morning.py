import ast
import base64
from contextlib import ExitStack
from datetime import datetime, timezone, timedelta
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch, Mock

from flask import Flask, jsonify, request
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import assistant_workflow as w
import assistant_runtime as runtime
import assistant_graph_worker as worker
import plan_evidence


def clock(value):
    instant=datetime.fromisoformat(value)
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):
            return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)
    return Clock


class MorningTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.base=Path(self.tmp.name)
        case=json.loads(gzip.decompress((HERE/'CASE.json.gz').read_bytes()))
        self.case=case;self.snapshot=case['draft']['snapshot'];self.evidence=case['draft']['model_evidence']
        self.stack=ExitStack()
        self.stack.enter_context(patch('paths.APP_DIR',str(self.base)))
        self.stack.enter_context(patch.object(w,'snapshot',return_value=self.snapshot))
        self.stack.enter_context(patch.object(plan_evidence,'for_snapshot',return_value=self.evidence))
        self.stack.enter_context(patch.dict(os.environ,{'STARGAZER_PYTHON':str(HERE.parents[1]/'_dev_venv/Scripts/python.exe')}))
        self.draft=w.store('plan',case['draft'])
        self.source=(w.root()/f'plan_{self.draft["id"]}.json')
        self.before=self.source.read_bytes()

    def tearDown(self):
        self.stack.close();self.tmp.cleanup()

    def test_old_midnight_failure_reproduced_against_real_five_stock_draft(self):
        baseline=json.loads(gzip.decompress((HERE/'BASELINE.json.gz').read_bytes()))
        source=base64.b64decode(baseline['stargazer_desktop/assistant_workflow.py'])
        old=types.ModuleType('old_midnight_workflow')
        exec(compile(source,'old_assistant_workflow.py','exec'),old.__dict__)
        with patch.object(old,'datetime',clock('2026-10-28T08:50:00+08:00')),patch.object(old,'snapshot',return_value=self.snapshot):
            self.assertFalse(old.plan_readiness(self.case['draft'],self.snapshot,self.evidence)['current'])
            with self.assertRaisesRegex(ValueError,'不是未来观察日'):old.plan_chat({},'重新制定',self.draft['id'])
            with self.assertRaisesRegex(ValueError,'过期'):old.confirm_plan(self.draft['id'],True)
        self.assertEqual(self.source.read_bytes(),self.before)

    def test_one_deadline_all_timezones_and_boundaries(self):
        self.assertEqual(w.plan_deadline(self.snapshot).isoformat(),'2026-10-28T09:30:00+08:00')
        for stamp,expired in [('2026-10-27T23:59:59+08:00',False),('2026-10-28T00:00:00+08:00',False),
              ('2026-10-28T08:50:00+08:00',False),('2026-10-28T09:29:59.999999+08:00',False),
              ('2026-10-28T09:30:00+08:00',True),('2026-10-28T09:31:00+08:00',True),
              ('2026-10-29T00:00:00+08:00',True),('2026-10-28T01:29:59+00:00',False),
              ('2026-10-28T01:30:00+00:00',True)]:
            with self.subTest(stamp=stamp):self.assertEqual(w.plan_expired(self.snapshot,datetime.fromisoformat(stamp)),expired)
        with self.assertRaisesRegex(ValueError,'时区'):w.plan_expired(self.snapshot,datetime(2026,10,28,8,50))

    def test_preopen_display_and_regeneration_preserve_five_stocks_seven_boards(self):
        answer=types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps({
            'reply':'按原要求重新讨论，风险仍未知', 'selected_codes':self.draft['selected_codes'],'boards':self.draft['boards']})))])
        with patch.object(w,'datetime',clock('2026-10-28T08:50:00+08:00')):
            self.assertTrue(w.state()['plan_current'])
            self.assertEqual(w.state()['plan_readiness']['risk_available'],0)
            with patch.object(w,'complete',return_value=answer) as api:
                new=w.plan_chat({},'按最新证据重新制定次日观察计划，保留数量和板块',self.draft['id'])['draft']
        self.assertEqual(api.call_count,1);self.assertEqual(new['selected_codes'],self.draft['selected_codes'])
        self.assertEqual(new['boards'],self.draft['boards']);self.assertEqual(new['manual_observations'],[])
        self.assertFalse((self.base/'dist/data/trade_plan.json').exists())
        self.assertEqual(self.source.read_bytes(),self.before)

    def test_real_http_confirmation_and_isolated_graph_preopen(self):
        tree=ast.parse((HERE.parent/'server.py').read_text(encoding='utf8'))
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('api_assistant_plan_confirm','api_assistant_state')]
        app=Flask('morning-confirm-real-route');enabled=Mock()
        namespace=dict(app=app,request=request,jsonify=jsonify,assistant_workflow=w,
            pentacles_demo=types.SimpleNamespace(set_enabled=enabled),deepseek_api=types.SimpleNamespace(status=lambda:{}))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'server.py','exec'),namespace)
        fixed=clock('2026-10-28T08:50:00+08:00')
        with patch.object(w,'datetime',fixed),patch.object(runtime,'datetime',fixed):
            status=app.test_client().get('/api/assistant/state').get_json()
            self.assertTrue(status['plan_current'])
            response=app.test_client().post('/api/assistant/plan/confirm',json={'id':self.draft['id'],'confirmed':True})
        self.assertEqual(response.status_code,200,response.get_json())
        self.assertEqual(response.get_json()['count'],5);enabled.assert_called_once_with(True)
        plan=json.loads((self.base/'dist/data/trade_plan.json').read_bytes())
        self.assertEqual([r['code'] for r in plan['stocks']],self.draft['selected_codes'])
        self.assertEqual(plan['boards'],self.draft['boards'])
        self.assertTrue(next(r for r in plan['stocks'] if r['code']=='000573')['observe_only'])
        self.assertEqual(self.source.read_bytes(),self.before)

    def test_0930_closes_all_mutation_paths_without_paid_calls(self):
        fixed=clock('2026-10-28T09:30:00+08:00')
        with patch.object(w,'datetime',fixed),patch.object(runtime,'datetime',fixed),patch.object(w,'complete') as api,patch.object(runtime.subprocess,'run') as sub:
            self.assertFalse(w.plan_readiness(self.draft,self.snapshot,self.evidence)['current'])
            with self.assertRaisesRegex(ValueError,'09:30'):w.plan_chat({},'重新制定',self.draft['id'])
            with self.assertRaisesRegex(ValueError,'过期'):w.confirm_plan(self.draft['id'],True)
            with self.assertRaisesRegex(ValueError,'过期'):runtime.gate('plan',self.draft,approved=True)
            api.assert_not_called();sub.assert_not_called()
        self.assertFalse((self.base/'dist/data/trade_plan.json').exists())

    def test_preopen_changed_snapshot_evidence_and_no_approval_still_rejected(self):
        with patch.object(w,'datetime',clock('2026-10-28T08:50:00+08:00')):
            with self.assertRaisesRegex(ValueError,'人工确认'):w.confirm_plan(self.draft['id'],False)
            changed=json.loads(json.dumps(self.evidence));changed[self.draft['selected_codes'][0]]['risk']['reason']='changed'
            self.assertFalse(w.plan_readiness(self.draft,self.snapshot,changed)['current'])
            current=json.loads(json.dumps(self.snapshot));current['rows'][0]['reference_price']+=.01
            with patch.object(w,'snapshot',return_value=current),self.assertRaisesRegex(ValueError,'快照变化'):
                w.confirm_plan(self.draft['id'],True)


if __name__=='__main__':unittest.main()
