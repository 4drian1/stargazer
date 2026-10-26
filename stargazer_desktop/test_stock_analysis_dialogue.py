import json
import ast
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

import stock_analysis_dialogue as dialogue
import assistant_workflow as workflow
from flask import Flask, request, jsonify


class AnalysisDialogueTests(unittest.TestCase):
    def test_pure_analysis_and_mixed_edit(self):
        self.assertEqual(dialogue.analysis_only('请分析600792，不修改计划条件'), ['600792'])
        self.assertEqual(dialogue.analysis_only('分析600792，不修改条件，同时加入600621'), [])
        self.assertEqual(dialogue.analysis_only('分析600792并调整板块'), [])

    def test_empty_ai_keeps_fact_and_no_retry(self):
        fact={'code':'002310','status':'AVAILABLE','data_date':20261021,
              'selection':{'personal_score':.54},
              'risk':{'status':'AVAILABLE','context':{'amp_232':.012,'amp_266':.018}}}
        with patch.object(dialogue.services,'model_stock_analysis',return_value=fact), patch.object(
                dialogue,'complete',side_effect=ValueError('DeepSeek返回了空正文（EMPTY_CONTENT）')) as ai, patch.object(
                dialogue.workflow,'store') as store:
            result=dialogue.discuss({},'分析002310',['002310'])
        ai.assert_called_once()
        self.assertEqual(ai.call_args.kwargs['purpose'],'chat')
        self.assertIn('0.012',result['reply'])
        self.assertIn('AI解释失败：EMPTY_CONTENT',result['reply'])
        self.assertEqual(store.call_args.args[0],'stock_analysis')
        self.assertNotIn('selected_codes',json.dumps(result))

    def test_uncovered_stocks_never_gain_score(self):
        def lookup(code):
            return {'code':code,'status':'NOT_COVERED','requires_feature_rebuild':True}
        with patch.object(dialogue.services,'model_stock_analysis',side_effect=lookup), patch.object(
                dialogue,'complete',side_effect=ValueError('EMPTY_CONTENT')), patch.object(dialogue.workflow,'store'):
            result=dialogue.discuss({},'分析600792、600621和000712',['600792','600621','000712'])
        for code in ('600792','600621','000712'):
            self.assertIn(code,result['reply'])
        self.assertNotIn('232主版',result['reply'])

    def test_llm_gets_only_requested_facts(self):
        from types import SimpleNamespace
        rsp=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='缺评分，待核对。'))])
        with patch.object(dialogue.services,'model_stock_analysis',return_value={'code':'600792','status':'NOT_COVERED'}), patch.object(
                dialogue,'complete',return_value=rsp) as ai, patch.object(dialogue.workflow,'store'):
            result=dialogue.discuss({},'分析600792',['600792'])
        self.assertEqual(len(ai.call_args.args[1]),2)
        self.assertIn('缺评分，待核对',result['reply'])
        self.assertEqual(result['analysis']['ai_status'],'AVAILABLE')

    def test_actual_plan_mode_route_does_not_create_draft(self):
        tree=ast.parse(Path(__file__).with_name('server.py').read_text(encoding='utf8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='chat')
        app=Flask('stock-analysis-route')
        namespace=dict(app=app,request=request,jsonify=jsonify,_brain_lock=threading.RLock(),
                       get_cfg=lambda:{}, get_brain=lambda: type('BrainStub', (), {'chat': lambda _, msg, on_tool=None: '600792 普通对话'})())
        exec(compile(ast.Module(body=[node],type_ignores=[]),'server.py','exec'),namespace)
        with patch.object(workflow,'plan_chat',side_effect=AssertionError('PLAN_MUST_NOT_RUN')):
            reply=app.test_client().post('/api/chat',json={'message':'分析600792，不修改计划条件',
                                                            'mode':'next_day_plan','draft_id':'old-draft'})
        self.assertEqual(reply.status_code,200)
        self.assertNotIn('draft',reply.get_json())
        self.assertIn('普通对话',reply.get_json()['reply'])


if __name__=='__main__': unittest.main()
