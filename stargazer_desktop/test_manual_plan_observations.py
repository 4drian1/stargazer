"""User-authorized observations are not model candidates or executable rules."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import assistant_workflow as workflow
import plan_evidence
import trade_plan_v1
from test_assistant_workflow import response


MESSAGE = ('我还想关注pcb,军工，创新药，房地产，煤炭，存储，cpo这几个板块，'
           '然后我想把600621和000712加入计划，600621下杀到13.80以下看有无止跌，出现信号考虑介入，'
           '然后000712如果不停，跌超4%以上止跌考虑介入')
BOARDS = [{'name': name, 'note': '用户要求关注'} for name in ('pcb', '军工', '创新药', '房地产', '煤炭', '存储', 'cpo')]


class ManualPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.day = datetime.now().strftime('%Y%m%d')
        self.codes = [str(i).zfill(6) for i in range(1, 6)]
        self.snap = dict(schema='PAGE_OF_PENTACLES_DEMO_V1', plan_ready=True,
            data_date=int(self.day), monitor_date=int((datetime.now()+timedelta(days=1)).strftime('%Y%m%d')),
            rows=[dict(code=c, name='模型候选', setup='A_SDP', reference_price=10.,
                       zone={'zone_lower': 9., 'zone_upper': 9.5}) for c in self.codes])
        self.patches = [patch('paths.APP_DIR', str(self.base)), patch.object(workflow, 'snapshot', return_value=self.snap),
                        patch.object(plan_evidence.shadow_ui, 'find_pack', return_value=(None, 'MISSING'))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def chat(self, message=MESSAGE, previous=None, returned=None):
        answer = dict(reply='人工条件已记录，风险未评分，待核对跌幅参考基准。',
                      selected_codes=returned if returned is not None else self.codes + ['600621', '000712'],
                      boards=BOARDS, stock_comments=[])
        with patch.object(workflow, 'complete', return_value=response(answer)) as api:
            result = workflow.plan_chat({}, message, previous)['draft']
        return result, json.loads(api.call_args.args[1][1]['content'])

    def test_exact_screenshot_request_and_provenance(self):
        draft, context = self.chat()
        self.assertEqual(draft['selected_codes'], self.codes)
        self.assertEqual([r['code'] for r in draft['manual_observations']], ['600621', '000712'])
        self.assertEqual(len(draft['boards']), 7)
        self.assertNotIn('600621', draft['model_evidence'])
        self.assertNotIn('600621', str(context['candidates']))
        for row in draft['manual_observations']:
            self.assertEqual(row['source'], 'USER')
            self.assertEqual(row['user_requests'], [MESSAGE])
            self.assertEqual(row['conditions_status'], 'UNSTRUCTURED_NOT_EXECUTABLE')
        self.assertTrue(workflow.plan_readiness(draft, self.snap)['current'])
        self.assertEqual(draft['changes']['manual_added'], ['000712', '600621'])
        self.assertEqual(len(draft['changes']['boards_added']), 7)

    def test_confirm_save_and_read_keeps_manual_separate(self):
        draft, _ = self.chat()
        self.assertEqual(workflow.confirm_plan(draft['id'], True)['count'], 7)
        file = self.base/'dist/data/trade_plan.json'
        formal = json.loads(file.read_bytes())
        for stock in formal['stocks'][-2:]:
            self.assertEqual(stock['source'], 'USER')
            self.assertTrue(stock['observe_only'])
            self.assertIsNone(stock['buy_zone'])
            self.assertIsNone(stock['entry_rule'])
            self.assertEqual(stock['user_requests'], [MESSAGE])
            self.assertTrue(stock['evidence_refs'][0].startswith('human:'))
        monitor = json.loads((self.base/'data/pentacles_demo/assistant_plan.json').read_bytes())
        self.assertEqual([r['code'] for r in monitor['rows']], self.codes)
        with patch.object(trade_plan_v1, 'TRADE_PLAN_FILE', file):
            view = plan_evidence.model_view('risk')
        self.assertEqual(len(view['items']), 7)
        self.assertEqual(view['items'][-1]['facts']['status'], 'MISSING')
        self.assertIn('人工观察', view['items'][-1]['facts']['reason'])

    def test_later_dialogue_preserves_conditions_without_repeating_codes(self):
        draft, _ = self.chat()
        second, context = self.chat('保留当前人工观察，板块不变', draft['id'], self.codes)
        self.assertEqual(second['manual_observations'], draft['manual_observations'])
        self.assertEqual(len(context['manual_observations']), 2)

    def test_update_then_remove_does_not_rewrite_previous_draft(self):
        draft, _ = self.chat()
        before = (workflow.root()/f'plan_{draft["id"]}.json').read_bytes()
        updated, _ = self.chat('000712跌幅以昨日收盘为准，先只记录', draft['id'], self.codes)
        self.assertEqual(len(updated['manual_observations'][1]['user_requests']), 2)
        self.assertEqual(updated['changes']['conditions_updated'], ['000712'])
        with patch.object(workflow, 'complete', side_effect=AssertionError('移出人工观察不应调用AI')):
            removed = workflow.plan_chat({}, '请从计划中删除000712', updated['id'])['draft']
        self.assertEqual([r['code'] for r in removed['manual_observations']], ['600621'])
        self.assertEqual(removed['changes']['manual_removed'], ['000712'])
        self.assertEqual((workflow.root()/f'plan_{draft["id"]}.json').read_bytes(), before)

    def test_llm_cannot_invent_third_stock(self):
        with self.assertRaisesRegex(ValueError, '未知或重复'):
            self.chat(returned=self.codes + ['600621', '000712', '600000'])

    def test_negated_addition_is_not_authorization(self):
        with self.assertRaisesRegex(ValueError, '未知或重复'):
            self.chat('不要把600621加入计划', returned=self.codes + ['600621'])

    def test_mention_without_addition_is_not_authorization(self):
        with self.assertRaisesRegex(ValueError, '未知或重复'):
            self.chat('600621风险怎么样', returned=self.codes + ['600621'])

    def test_duplicates_still_rejected(self):
        with self.assertRaisesRegex(ValueError, '未知或重复'):
            self.chat(returned=self.codes + ['600621', '600621'])

    def test_user_addition_does_not_depend_on_llm_echo(self):
        draft, _ = self.chat(returned=self.codes)
        self.assertEqual(len(draft['manual_observations']), 2)

    def test_empty_model_pool_can_have_only_manual_observations(self):
        self.snap['rows'] = []
        draft, _ = self.chat(returned=['600621', '000712'])
        self.assertEqual(draft['selected_codes'], [])
        self.assertEqual(workflow.confirm_plan(draft['id'], True)['count'], 2)

    def test_analysis_in_plan_discussion_calls_actual_tool(self):
        first, _ = self.chat()
        with patch('services.model_stock_analysis', return_value={'code': '600621', 'status': 'NOT_COVERED'}) as tool:
            second, context = self.chat('分析一下600621怎么样', first['id'], self.codes)
        tool.assert_called_once_with('600621')
        self.assertEqual(context['requested_stock_analysis'][0]['status'], 'NOT_COVERED')
        self.assertEqual(second['stock_analyses'][0]['status'], 'NOT_COVERED')
        self.assertEqual(second['changes']['conditions_updated'], [])

    def test_analysis_explicitly_not_editing_keeps_conditions_unchanged(self):
        draft, _ = self.chat()
        with patch('services.model_stock_analysis', return_value={'code': '600621', 'status': 'NOT_COVERED'}):
            newer, _ = self.chat('分析600621怎么样，不修改计划条件', draft['id'], self.codes)
        self.assertEqual(newer['manual_observations'], draft['manual_observations'])
        self.assertEqual(newer['changes']['conditions_updated'], [])

    def seed_groups(self, selected_manual=('000510',), observations=('002908', '000400'),
                    snapshot_hash=None):
        snap = dict(self.snap, manual_review_rows=[dict(code='000510', name='人工复核票', rank=1,
            selection_score=70., reference_price=10., zone={'zone_lower': 9., 'zone_upper': 9.5})])
        previous = dict(origin='MANUAL_REVIEW_PLAN', snapshot=snap,
            snapshot_hash=snapshot_hash or workflow.digest(snap), selected_codes=['000001'],
            selected_manual_codes=list(selected_manual),
            manual_observations=[dict(code=c, source='USER', name='人工观察（身份未核验）',
                status='OBSERVATION_ONLY', conditions_status='UNSTRUCTURED_NOT_EXECUTABLE',
                user_requests=[MESSAGE]) for c in observations],
            requested_count=10, boards=[dict(b) for b in BOARDS], manual_review_candidates=[],
            dialogue=[], model_evidence={}, model_commentary={})
        return snap, workflow.store('plan', previous)

    def test_group_removal_moves_both_manual_groups_without_ai(self):
        snap, previous = self.seed_groups()
        with patch.object(workflow, 'snapshot', return_value=snap), patch.object(
                workflow, 'complete', side_effect=AssertionError('移出不应调用AI')):
            draft = workflow.plan_chat({}, '人工观察和人工选择移出计划', previous['id'])['draft']
        self.assertEqual(draft['selected_manual_codes'], [])
        self.assertEqual(draft['manual_observations'], [])
        self.assertIn('人工选择整组已移出', draft['reply'])
        self.assertIn('人工观察整组已移出', draft['reply'])
        self.assertEqual(draft['changes']['manual_removed'], ['000400', '002908'])
        self.assertTrue(draft['confirmable'])

    def test_observation_group_removal_keeps_manual_selection(self):
        snap, previous = self.seed_groups()
        with patch.object(workflow, 'snapshot', return_value=snap):
            draft = workflow.plan_chat({}, '人工观察移出计划', previous['id'])['draft']
        self.assertEqual(draft['selected_manual_codes'], ['000510'])
        self.assertEqual(draft['manual_observations'], [])
        self.assertNotIn('人工选择整组已移出', draft['reply'])

    def test_manual_selection_group_removal_keeps_observations(self):
        snap, previous = self.seed_groups()
        with patch.object(workflow, 'snapshot', return_value=snap):
            draft = workflow.plan_chat({}, '人工选择移出计划', previous['id'])['draft']
        self.assertEqual(draft['selected_manual_codes'], [])
        self.assertEqual([r['code'] for r in draft['manual_observations']], ['002908', '000400'])
        self.assertNotIn('人工观察整组已移出', draft['reply'])

    def test_explicit_add_uses_current_review_list_after_snapshot_change(self):
        snap, previous = self.seed_groups(selected_manual=(), observations=(), snapshot_hash='f' * 64)
        with patch.object(workflow, 'snapshot', return_value=snap):
            draft = workflow.plan_chat({}, '把000510加入人工复核', previous['id'])['draft']
        self.assertEqual(draft['selected_manual_codes'], ['000510'])
        self.assertIn('加入待确认草案', draft['reply'])

    def test_removal_question_or_negation_does_not_change_groups(self):
        previous = {'selected_manual_codes': ['000510'],
                    'manual_observations': [{'code': '002908'}, {'code': '000400'}]}
        for message in ('人工观察和人工选择能移出计划吗？', '不要移出人工观察和人工选择'):
            self.assertEqual(workflow.manual_review_codes(message, {}, previous), ['000510'])
            self.assertEqual([r['code'] for r in workflow.manual_observations(
                message, previous, set())], ['002908', '000400'])


if __name__ == '__main__':
    unittest.main()
