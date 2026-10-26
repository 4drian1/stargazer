import json
import tempfile
import unittest
import io
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import assistant_workflow as workflow
import plan_evidence
import plan_board_monitor as boards
import trade_plan_v1
from test_assistant_workflow import response
from test_shadow_ui import pack_dict
from oracles_v1 import EvidencePack


class PlanConversationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.app_patch = patch('paths.APP_DIR', str(self.base))
        self.app_patch.start()
        self.day = datetime.now().strftime('%Y%m%d')
        self.snap = {'schema': 'PAGE_OF_PENTACLES_DEMO_V1', 'plan_ready': True,
            'data_date': int(self.day),
            'monitor_date': int((datetime.now() + timedelta(days=1)).strftime('%Y%m%d')),
            'rows': [{'code': str(i).zfill(6), 'name': '测试股', 'setup': 'A_SDP',
                      'reference_price': 10., 'zone': {'zone_lower': 9., 'zone_upper': 9.5}}
                     for i in range(1, 51)]}
        file = self.base / 'data/pentacles_demo/snapshot.json'
        file.parent.mkdir(parents=True)
        file.write_text(json.dumps(self.snap))
        self.pack_patch = patch.object(plan_evidence.shadow_ui, 'find_pack', return_value=(None, 'MISSING'))
        self.pack_patch.start()

    def tearDown(self):
        self.pack_patch.stop()
        self.app_patch.stop()
        self.tmp.cleanup()

    def test_ask_then_respect_five_and_save_boards_views(self):
        with patch.object(workflow, 'complete', side_effect=AssertionError('QUESTION_MUST_BE_LOCAL')):
            pending = workflow.plan_chat({}, '五十个看不过来')['draft']
        self.assertTrue(pending['awaiting_count'])
        with self.assertRaises(ValueError):
            workflow.confirm_plan(pending['id'], True)
        codes = [r['code'] for r in self.snap['rows'][:5]]
        answer = {'reply': '先观察，风险证据不足，不建议直接买入', 'selected_codes': codes,
                  'boards': [{'name': '创新药', 'note': '观察相对大盘强弱'}],
                  'stock_comments': [{'code': codes[0], 'selection': '共性形态入围',
                    'risk': '非常安全', 'evidence_refs': [f'tarot:{self.day}:{codes[0]}']}]}
        with patch.object(workflow, 'complete', return_value=response(answer)) as api:
            draft = workflow.plan_chat({}, '五只，监控创新药板块', pending['id'])['draft']
        context = json.loads(api.call_args.args[1][1]['content'])
        self.assertIn(codes[0], context['risk_missing_codes'])
        self.assertEqual(draft['requested_count'], 5)
        self.assertNotEqual(draft['model_commentary'][codes[0]]['risk'], '非常安全')
        workflow.confirm_plan(draft['id'], True)
        file = self.base / 'dist/data/trade_plan.json'
        plan = json.loads(file.read_bytes())
        self.assertEqual(len(plan['stocks']), 5)
        self.assertEqual(plan['boards'][0]['name'], '创新药')
        with patch.object(trade_plan_v1, 'TRADE_PLAN_FILE', file):
            view = plan_evidence.model_view('risk')
        self.assertEqual(len(view['items']), 5)
        self.assertEqual(view['items'][0]['facts']['status'], 'MISSING')

    def test_limit_violation_not_silently_saved(self):
        pending = workflow.plan_chat({}, '太多了')['draft']
        with patch.object(workflow, 'complete', return_value=response({
                'reply': '观察', 'selected_codes': [r['code'] for r in self.snap['rows'][:6]]})):
            with self.assertRaisesRegex(ValueError, '超出'):
                workflow.plan_chat({}, '5只', pending['id'])
        self.assertEqual(workflow.latest('plan')['id'], pending['id'])

    def test_same_day_risk_is_in_actual_prompt(self):
        raw = pack_dict()
        raw.update(market_date=self.day, asof_ts=self.day + '150000', query_ts=self.day + '150000')
        raw['candidates'][0].update(code='000001', trigger=None)
        raw['risk_context']['asof_ts'] = raw['asof_ts']
        pack = EvidencePack(**raw).bind()
        with patch.object(plan_evidence.shadow_ui, 'find_pack', return_value=(pack, None)):
            with patch.object(workflow, 'complete', return_value=response({'reply': '根据两侧证据讨论', 'selected_codes': ['000001']})) as api:
                workflow.plan_chat({}, '只要1只')
            ctx = json.loads(api.call_args.args[1][1]['content'])
        self.assertEqual(ctx['risk_evidence']['000001']['context']['amp_232'], .014)
        self.assertIn('120', ctx['risk_evidence']['000001']['context']['horizon'])
        raw['risk_context']['asof_ts'] = self.day + '150100'
        with patch.object(plan_evidence.shadow_ui, 'find_pack', return_value=(EvidencePack(**raw).bind(), None)):
            self.assertEqual(plan_evidence.for_snapshot(self.snap)['000001']['risk']['status'], 'MISSING')
        raw.update(asof_ts=self.day + '110000', query_ts=self.day + '110000')
        raw['risk_context']['asof_ts'] = raw['asof_ts']
        with patch.object(plan_evidence.shadow_ui, 'find_pack', return_value=(EvidencePack(**raw).bind(), None)):
            self.assertEqual(plan_evidence.for_snapshot(self.snap)['000001']['risk']['status'], 'MISSING')

    def test_count_language(self):
        self.assertEqual(workflow.requested_count('只看五只'), 5)
        self.assertEqual(workflow.requested_count('不要5只，要3只'), 3)
        self.assertIsNone(workflow.requested_count('50只看不过来'))

    def plan_answer(self, comments, codes=None):
        answer = {'reply': '先观察再人工确认', 'selected_codes': codes or ['000001'], 'stock_comments': comments}
        with patch.object(workflow, 'complete', return_value=response(answer)) as api:
            draft = workflow.plan_chat({}, '只要1只')['draft']
        self.assertEqual(api.call_count, 1)  # No automatic paid retry.
        return draft

    def test_numeric_comment_is_isolated_not_entire_plan_rejected(self):
        draft = self.plan_answer([{'code': '000001', 'selection': 'RSI14为57，评分0.99',
            'risk': '120分钟下探幅度', 'evidence_refs': [f'tarot:{self.day}:000001']}])
        self.assertNotIn('selection', draft['model_commentary']['000001'])
        self.assertNotIn('0.99', json.dumps(draft, ensure_ascii=False))
        self.assertTrue(draft['commentary_warnings'])
        self.assertEqual(workflow.load_draft('plan', draft['id'])['selected_codes'], ['000001'])
        self.assertTrue(workflow.confirm_plan(draft['id'], True)['ok'])

    def test_same_snapshot_new_evidence_invalidates_old_draft_without_rewrite(self):
        draft = self.plan_answer([])
        file = workflow.root() / f'plan_{draft["id"]}.json'
        before = file.read_bytes()
        evidence = plan_evidence.for_snapshot(self.snap)
        evidence['000001']['risk'].update(status='AVAILABLE', asof_ts=self.day+'150000',
                                         context={'amp_232': .01}, pack_hash='new-pack')
        with patch.object(plan_evidence, 'for_snapshot', return_value=evidence):
            state = workflow.state()
            self.assertFalse(state['plan_current'])
            self.assertEqual(state['plan_readiness']['risk_available'], 1)
            with self.assertRaisesRegex(ValueError, '证据已更新'):
                workflow.confirm_plan(draft['id'], True)
            with patch.object(workflow, 'complete', return_value=response({
                    'reply': '依据最新证据讨论', 'selected_codes': ['000001']})) as api:
                new = workflow.plan_chat({}, '按最新证据重新制定', draft['id'])['draft']
            context = json.loads(api.call_args.args[1][1]['content'])
            self.assertTrue(context['evidence_refresh'])
            self.assertIsNone(context['current_draft']['reply'])
            self.assertFalse(any(m['role']=='assistant' for m in api.call_args.args[1]))
            self.assertTrue(workflow.state()['plan_current'])
            self.assertEqual(new['model_evidence']['000001']['risk']['pack_hash'], 'new-pack')
        self.assertEqual(before, file.read_bytes())

    def test_equal_coverage_changed_values_still_require_refresh(self):
        draft = self.plan_answer([])
        evidence = plan_evidence.for_snapshot(self.snap)
        evidence['000001']['selection']['personal_score'] = .5
        self.assertFalse(workflow.plan_readiness(draft, self.snap, evidence)['current'])

    def test_evidence_change_during_provider_call_does_not_save_stale_response(self):
        evidence = plan_evidence.for_snapshot(self.snap)
        changed = json.loads(json.dumps(evidence))
        changed['000001']['selection']['personal_score'] = .5
        def api(*a, **kw):
            evidence['000001'] = changed['000001']
            return response({'reply': '观察', 'selected_codes': ['000001']})
        # Separate copies emulate an atomic evidence pointer change while a call is in flight.
        original = json.loads(json.dumps(evidence))
        with patch.object(plan_evidence, 'for_snapshot', side_effect=[original, original, changed]), patch.object(
                workflow, 'complete', side_effect=api):
            with self.assertRaisesRegex(ValueError, '讨论期间模型证据已更新'):
                workflow.plan_chat({}, '只要1只')
        self.assertIsNone(workflow.latest('plan'))

    def test_bad_optional_comment_shapes_fall_back(self):
        for comments in (None, {}, ['bad'], [{'code': []}], [{'code': '999999'}],
                [{'code': '000001', 'evidence_refs': [{}]}],
                [{'code': '000001', 'selection': '背离成立', 'evidence_refs': ['FAKE']}],
                [{'code': '000001', 'selection': '太' * 1001}]):
            with self.subTest(comments=str(comments)[:60]):
                draft = self.plan_answer(comments)
                self.assertTrue(draft['commentary_warnings'])

    def test_missing_group_reference_cannot_support_comment(self):
        draft = self.plan_answer([{'code': '000001', 'selection': '形态入围',
            'evidence_refs': [f'pyxis:{self.day}:000001']}])
        self.assertNotIn('selection', draft['model_commentary']['000001'])

    def test_valid_optional_comment_is_kept(self):
        draft = self.plan_answer([{'code': '000001', 'selection': '共性形态入围',
            'evidence_refs': [f'tarot:{self.day}:000001']}])
        self.assertEqual(draft['model_commentary']['000001']['selection'], '共性形态入围')
        self.assertFalse(draft['commentary_warnings'])

    def test_malformed_main_answer_preserves_previous_draft(self):
        old = self.plan_answer([])['id']
        for answer in ([], {'reply': '观察', 'selected_codes': [[]]},
                {'reply': '观察', 'selected_codes': [None]}):
            with patch.object(workflow, 'complete', return_value=response(answer)):
                with self.assertRaises(ValueError):
                    workflow.plan_chat({}, '只要1只', old)
            self.assertEqual(workflow.latest('plan')['id'], old)

    def test_no_zone_confirmation_remains_observation_only(self):
        self.snap['rows'][0]['zone'] = None
        (self.base / 'data/pentacles_demo/snapshot.json').write_text(json.dumps(self.snap))
        draft = self.plan_answer([])
        workflow.confirm_plan(draft['id'], True)
        plan = json.loads((self.base / 'dist/data/trade_plan.json').read_bytes())
        self.assertIsNone(plan['stocks'][0]['buy_zone'])
        self.assertTrue(plan['stocks'][0]['observe_only'])
        for px in (0.01, 10., 10000.):
            self.assertEqual(trade_plan_v1.plan_check('000001', px,
                str(self.snap['monitor_date'])+'100000', plan)['status'], 'OBSERVE_ONLY')

    def test_invalid_zone_does_not_silently_become_observe_only(self):
        self.snap['rows'][0]['zone'] = {'zone_lower': 10, 'zone_upper': float('inf')}
        plan = {'version': 1, 'created_ts': self.day+'150000', 'data_date': self.day,
            'monitor_date': str(self.snap['monitor_date']), 'boards': [], 'stocks': [
            {'code': '000001', 'name': 'A', 'buy_zone': [9., float('inf')],
             'invalidations': [], 'valid_until': str(self.snap['monitor_date'])+'150000', 'evidence_refs': []}]}
        self.assertTrue(trade_plan_v1.validate_plan(plan))
        self.assertEqual(trade_plan_v1.plan_check('000001', 10., self.day+'160000', plan)['reason'], 'BUY_ZONE_INVALID')
        plan['stocks'][0]['buy_zone'] = [9., 10.]
        for px in ('nan', 'inf', '-1', '0'):
            self.assertEqual(trade_plan_v1.plan_check('000001', px, self.day+'160000', plan)['reason'], 'PRICE_INVALID')

    def test_old_above_close_zone_cannot_reach_discussion_or_confirmation(self):
        draft = self.plan_answer([])
        formal = self.base / 'dist/data/trade_plan.json'
        self.snap['rows'][0]['zone']['zone_upper'] = 10.1
        (self.base / 'data/pentacles_demo/snapshot.json').write_text(json.dumps(self.snap))
        with patch.object(workflow, 'complete', side_effect=AssertionError('NO_AI_CALL')):
            with self.assertRaisesRegex(ValueError, '重新运行盘后选股'):
                workflow.plan_chat({}, '只要1只')
        with self.assertRaisesRegex(ValueError, '重新运行盘后选股'):
            workflow.confirm_plan(draft['id'], True)
        self.assertFalse(formal.exists())

    def test_board_unknown_is_not_flat(self):
        row = boards.quote({'name': '未知板块', 'bk': None}, datetime.now(boards.ZONE))
        self.assertEqual(row['status'], 'UNRESOLVED')
        self.assertIsNone(row['pct'])
        file = self.base / 'data/board_list.json'
        file.write_text(json.dumps({'boards': {'创新药': 'BK1234'}}))
        self.assertEqual(boards.resolve(['创新药'])[0]['bk'], 'BK1234')
        self.assertIsNone(boards.resolve(['不存在'])[0]['bk'])

    def test_board_quote_ratio_and_stale_are_separate(self):
        now = datetime.now(boards.ZONE)
        data = {'data': {'f43': 12345, 'f60': 12000, 'f86': int(now.timestamp()), 'f58': '创新药'}}
        with patch.object(boards.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(data).encode())):
            row = boards.quote({'name': '创新药', 'bk': 'BK1234'}, now)
        self.assertEqual(row['status'], 'OK')
        self.assertAlmostEqual(row['pct'], 2.875)
        data['data']['f86'] -= 600
        with patch.object(boards.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(data).encode())):
            row = boards.quote({'name': '创新药', 'bk': 'BK1234'}, now)
        self.assertEqual(row['status'], 'STALE')
        self.assertIsNone(row['pct'])

    def test_future_plan_does_not_fetch_boards(self):
        plan = {'monitor_date': self.snap['monitor_date'], 'boards': [{'name': '创新药'}]}
        with patch.object(boards.trade_plan_v1, 'load_plan', return_value=plan), patch.object(
                boards, 'quote', side_effect=AssertionError('NO_PREMATURE_FETCH')):
            result = boards.state(datetime.now(boards.ZONE))
        self.assertEqual(result['status'], 'WAIT_SESSION')


if __name__ == '__main__':
    unittest.main()
