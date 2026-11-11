import ast
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

from flask import Flask, jsonify, request

import assistant_workflow
from conversation_intent import asks_for_model, model_scope, plan_work


class ConversationIntentTests(unittest.TestCase):
    def test_plan_edits_survive_but_questions_are_free_chat(self):
        for message in ('请根据当前名单制定次日计划', '5只', '把600621从人工观察删除',
                        '监控板块加上煤炭', '请把002310从计划里移除',
                        '人工观察和人工选择移出计划', '把002908移出人工观察',
                        '请把688661移出人工选择'):
            self.assertTrue(plan_work(message, drafting=True), message)
        for message in ('你好', '看看今天的计划', '我的持仓该卖吗',
                        '请分析600792，不修改计划条件', '解释一下Pyxis模型',
                        '如果把600792加入计划会怎么样', '人工观察和人工选择能移出计划吗？',
                        '不要把688661移出计划'):
            self.assertFalse(plan_work(message, drafting=True), message)
        self.assertTrue(asks_for_model('请用Pyxis模型分析600792'))
        self.assertFalse(asks_for_model('请帮我分析600792'))
        self.assertEqual(model_scope('请用Pyxis模型分析600792'), 'pyxis')
        self.assertEqual(model_scope('请用星币侍从分析600792'), 'tarot')

    def test_server_routes_plan_edits_and_free_chat_separately(self):
        tree = ast.parse(Path(__file__).with_name('server.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'chat')
        app = Flask('conversation-route')
        class BrainStub:
            archive = Path("isolated-route-test.jsonl")
            def chat(self, message, on_tool=None):
                return '普通对话：' + message
        namespace = dict(app=app, request=request, jsonify=jsonify, assistant_workflow=assistant_workflow,
                         _brain_lock=threading.RLock(),
                         get_cfg=lambda: {}, get_brain=lambda: BrainStub())
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'server.py', 'exec'), namespace)
        with patch.object(assistant_workflow, 'plan_chat', return_value={
                'reply': '计划草案', 'draft': {'id': 'draft', 'awaiting_count': True}}) as plan:
            normal = app.test_client().post('/api/chat', json={
                'message': '我的持仓该卖吗', 'mode': 'next_day_plan', 'draft_id': 'old'})
            self.assertEqual(normal.get_json()['reply'], '普通对话：我的持仓该卖吗')
            self.assertNotIn('draft', normal.get_json())
            self.assertEqual(plan.call_count, 0)
            edit = app.test_client().post('/api/chat', json={
                'message': '把600621从人工观察删除', 'mode': 'next_day_plan', 'draft_id': 'old'})
            self.assertEqual(edit.get_json()['draft']['id'], 'draft')
            self.assertEqual(plan.call_count, 1)
            # 已确认计划后的普通对话里说“移出”也必须进计划草案，不能落到自由对话拒绝
            edit2 = app.test_client().post('/api/chat', json={
                'message': '人工观察和人工选择移出计划', 'mode': 'chat', 'draft_id': 'old'})
            self.assertEqual(edit2.get_json()['draft']['id'], 'draft')
            self.assertEqual(plan.call_count, 2)
            plan.assert_called_with({}, '人工观察和人工选择移出计划', 'old')


if __name__ == '__main__':
    unittest.main()
