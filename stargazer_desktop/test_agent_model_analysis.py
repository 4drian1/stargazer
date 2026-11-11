"""Agent queries actual bound model evidence, not a renamed legacy screener."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.brain import Brain, TOOLS
import assistant_workflow as workflow
import plan_evidence
import services


class AgentModelTests(unittest.TestCase):
    def test_confirmed_plan_read_includes_bounded_model_facts(self):
        import trade_plan_v1
        plan = {'data_date': '20261021', 'monitor_date': '20261022',
                'stocks': [{'code': '000629'}], 'boards': [],
                'model_evidence': {'000629': {'selection': {'personal_score': .54},
                    'risk': {'status': 'AVAILABLE', 'context': {'amp_232': .012, 'amp_266': .014}}}}}
        with patch.object(trade_plan_v1, 'load_plan', return_value=plan):
            result = services.confirmed_plan()
        self.assertEqual(result['stocks'], [{'code': '000629'}])
        self.assertEqual(result['model_facts']['000629']['amp_232'], .012)
        self.assertNotIn('model_evidence', result)

    def test_confirmed_plan_view_keeps_manual_group_source(self):
        import conversation_plans
        value = {'stocks': [{'code': '000629', 'source': 'USER_MANUAL_REVIEW', 'selection_qualified': False},
                            {'code': '000510', 'source': 'USER', 'observe_only': True}],
                 'manual_observations': [{'code': '000510', 'source': 'USER'}], 'model_facts': {}}
        view = conversation_plans.tool_view(value)
        self.assertEqual(view['stocks'][0]['source'], 'USER_MANUAL_REVIEW')
        self.assertFalse(view['stocks'][0]['selection_qualified'])
        self.assertEqual(view['stocks'][1]['source'], 'USER')

    def test_generic_stock_question_does_not_run_models(self):
        answer = SimpleNamespace(content='先看行情和公司事实。', tool_calls=[],
                                 model_dump=lambda **kw: {'role': 'assistant', 'content': '先看行情和公司事实。'})
        with tempfile.TemporaryDirectory() as tmp, patch('paths.APP_DIR', tmp):
            brain = Brain.__new__(Brain)
            brain.cfg, brain.max_rounds = {}, 1
            brain.archive = Path(tmp) / 'conversation.jsonl'
            brain.messages = [{'role': 'system', 'content': '自然对话'}]
            with patch('deepseek_api.complete', return_value=SimpleNamespace(choices=[SimpleNamespace(message=answer)])) as ai, patch.object(
                    services, 'model_stock_inference', side_effect=AssertionError('MODEL_MUST_NOT_RUN')):
                self.assertIn('行情', brain.chat('请帮我分析600792'))
        self.assertNotIn('model_stock_inference',
                         [t['function']['name'] for t in ai.call_args.kwargs['tools']])

    def test_confirmed_plan_question_reads_plan_without_new_draft(self):
        answer = SimpleNamespace(content='今天已确认计划有五只模型股。', tool_calls=[],
                                 model_dump=lambda **kw: {'role': 'assistant', 'content': '今天已确认计划有五只模型股。'})
        with tempfile.TemporaryDirectory() as tmp, patch('paths.APP_DIR', tmp):
            brain = Brain.__new__(Brain)
            brain.cfg, brain.max_rounds = {}, 1
            brain.archive = Path(tmp) / 'conversation.jsonl'
            brain.messages = [{'role': 'system', 'content': '自然对话'}]
            with patch('deepseek_api.complete', return_value=SimpleNamespace(choices=[SimpleNamespace(message=answer)])), patch.object(
                    services, 'confirmed_plan', return_value={'status': 'CONFIRMED', 'stocks': [{'code': '000629'}]}) as plan:
                brain.chat('看看今天的计划')
        plan.assert_called_once()
        self.assertEqual(json.loads(brain.messages[2]['tool_calls'][0]['function']['arguments']), {})

    def test_explicit_model_request_uses_bound_result_without_recompute(self):
        with patch.object(services, 'model_stock_analysis', return_value={
                'code': '000629', 'status': 'AVAILABLE', 'computed_now': False}) as bound, patch(
                'subprocess.run', side_effect=AssertionError('NO_WORKER')):
            value = services.model_stock_inference('000629')
        bound.assert_called_once_with('000629')
        self.assertFalse(value['computed_now'])

    def test_unbound_stock_runs_one_local_worker_and_keeps_risk_unknown(self):
        worker = {'code': '600792', 'selection': {'quality_state': 'OK', 'selection_score': .42},
                  'risk': {'status': 'NOT_COMPUTED'}, 'scope': 'FROZEN_SELECTION_HEAD_SINGLE_STOCK_RESEARCH_ONLY'}
        with patch.object(services, 'model_stock_analysis', return_value={
                'code': '600792', 'status': 'NOT_COVERED'}), patch.object(
                workflow, 'snapshot', return_value={'data_date': 20261021, 'monitor_date': 20261022}), patch(
                'subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps(worker))) as run:
            value = services.model_stock_inference('600792')
        self.assertEqual(value['inference_status'], 'SELECTION_ONLY')
        self.assertEqual(value['on_demand']['risk']['status'], 'NOT_COMPUTED')
        self.assertEqual(run.call_count, 1)

    def test_pyxis_only_refuses_unbound_context_without_running_tarot(self):
        with patch.object(services, 'model_stock_analysis', return_value={
                'code': '600792', 'status': 'NOT_COVERED'}), patch('subprocess.run',
                side_effect=AssertionError('NO_TAROT_WORKER')):
            result = services.model_stock_inference('600792', model='pyxis')
        self.assertEqual(result['inference_status'], 'PYXIS_CONTEXT_NOT_BOUND')
        self.assertNotIn('on_demand', result)

    def test_bound_evidence_has_explicit_lookup_identity(self):
        snap = {'data_date': 20261021, 'rows': [{'code': '000629'}]}
        evidence = {'000629': {'selection': {'personal_score': .52}, 'risk': {
            'status': 'AVAILABLE', 'asof_ts': '20261021150000', 'context': {'amp_232': .012}}}}
        with patch.object(workflow, 'snapshot', return_value=snap), patch.object(
                plan_evidence, 'for_snapshot', return_value=evidence):
            result = services.model_stock_analysis('000629')
        self.assertEqual(result['status'], 'AVAILABLE')
        self.assertFalse(result['computed_now'])
        self.assertEqual(result['risk']['context']['amp_232'], .012)
        self.assertEqual(result['selection']['personal_score'], .52)

    def test_uncovered_stock_never_borrows_another_score(self):
        with patch.object(workflow, 'snapshot', return_value={'data_date': 20261021, 'rows': [{'code': '000629'}]}), patch.object(
                plan_evidence, 'for_snapshot', side_effect=AssertionError('DO_NOT_INFER_UNBOUND')):
            result = services.model_stock_analysis('600621')
        self.assertEqual(result['status'], 'NOT_COVERED')
        self.assertNotIn('amp_232', json.dumps(result))
        self.assertTrue(result['requires_feature_rebuild'])

    def test_invalid_code_and_no_snapshot(self):
        for code in ('../../secret', '000629;1', 629):
            with self.assertRaises(ValueError):
                services.model_stock_analysis(code)
        with patch.object(workflow, 'snapshot', side_effect=FileNotFoundError):
            self.assertEqual(services.model_stock_analysis('000629')['status'], 'UNAVAILABLE')

    def test_agent_tool_call_round_trip_is_real_and_read_only(self):
        self.assertTrue(any(t['function']['name'] == 'model_stock_inference' for t in TOOLS))
        toolcall = SimpleNamespace(id='model-query', function=SimpleNamespace(name='model_stock_inference', arguments='{"code":"000629"}'))
        first = SimpleNamespace(content=None, tool_calls=[toolcall], model_dump=lambda **kw: {
            'role': 'assistant', 'tool_calls': [{'id': 'model-query', 'type': 'function',
              'function': {'name': 'model_stock_inference', 'arguments': '{"code":"000629"}'}}]})
        last = SimpleNamespace(content='风险：按已绑定结果分析，不是重新推理。', tool_calls=[],
                               model_dump=lambda **kw: {'role': 'assistant', 'content': '风险：按已绑定结果分析，不是重新推理。'})
        with tempfile.TemporaryDirectory() as tmp, patch('paths.APP_DIR', tmp):
            brain = Brain.__new__(Brain)
            brain.cfg, brain.max_rounds = {}, 2
            brain.archive = Path(tmp)/'conversation.jsonl'
            brain.messages = [{'role': 'system', 'content': '工具事实与判断分开'}]
            with patch('deepseek_api.complete', side_effect=[SimpleNamespace(choices=[SimpleNamespace(message=first)]),
                    SimpleNamespace(choices=[SimpleNamespace(message=last)])]), patch.object(
                    services, 'model_stock_inference', return_value={'code': '000629', 'mode': 'BOUND_MODEL_RESULT_LOOKUP', 'computed_now': False}) as actual:
                reply = brain.chat('请用模型分析000629怎么样')
            actual.assert_called_once_with(code='000629', model='both')
            self.assertIn('不是重新推理', reply)
            tool = next(m for m in brain.messages if m['role'] == 'tool')
            self.assertFalse(json.loads(tool['content'])['computed_now'])


if __name__ == '__main__':
    unittest.main()
