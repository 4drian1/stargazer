"""Readable, bound risk facts before human confirmation; no model changes."""
import copy
import unittest
from unittest.mock import patch

import assistant_workflow as workflow
import plan_evidence
import trade_plan_v1


class RiskDisplayTests(unittest.TestCase):
    def setUp(self):
        self.risk = dict(status='AVAILABLE', asof_ts='20261021150000', context={
            'amp_232': .012, 'amp_266': .013,
            'market_state': {'strength': -.006, 'cyb_strength': -.016},
            'board_states': [{'strength': -.001, 'coverage': .65}],
            'leader_states': [{'strength': .067, 'return_change': -.0007}],
            'feature_missing': {'232': ['x'], '266': ['x']}, 'trigger_state': None})
        self.draft = dict(id='d', snapshot={'data_date': 20261021, 'monitor_date': 20261022,
            'rows': [{'code': '000629', 'name': '钒钛股份'}]}, selected_codes=['000629'],
            model_evidence={'000629': {'risk': self.risk, 'selection': {}}}, reply='原文不改')

    def test_available_amplitude_not_hidden_by_unknown_trigger(self):
        lines = '\n'.join(plan_evidence.risk_lines(self.risk))
        for text in ('1.20%', '1.30%', '65.00%', '-0.60%', '6.70%', '不是概率', '模型已评分', '232缺1列'):
            self.assertIn(text, lines)
        self.assertNotIn('风险证据缺失', lines)

    def test_invalid_values_are_unknown_not_zero(self):
        for bad in (None, float('nan'), float('inf'), True, '0.01'):
            self.risk['context']['amp_232'] = bad
            self.assertIn('232主版 未知', '\n'.join(plan_evidence.risk_lines(self.risk)))
        self.risk['context']['amp_232'] = 0
        self.assertIn('232主版 0.00%', '\n'.join(plan_evidence.risk_lines(self.risk)))

    def test_missing_does_not_display_numbers(self):
        self.risk['status'] = 'MISSING'
        self.risk['reason'] = '源不匹配'
        self.assertEqual(plan_evidence.risk_lines(self.risk), ['风险证据缺失：源不匹配'])

    def test_presentation_is_read_only_and_keeps_all_stocks(self):
        original = copy.deepcopy(self.draft)
        result = plan_evidence.presentation(self.draft)
        self.assertIn('钒钛股份', result['draft'])
        self.assertEqual(len(result['risks']), 1)
        self.assertIn('沪指', ''.join(result['market_risk_summary']))
        self.assertIn('人工确认前', result['confirmation'])
        self.assertEqual(self.draft, original)

    def test_market_risk_summary_uses_only_bound_facts(self):
        self.risk['context']['market_state']['model_inputs'] = {
            'sse_5m_momentum_change': -.004, 'sse_15m_momentum_change': -.002}
        summary = ' '.join(plan_evidence.presentation(self.draft)['market_risk_summary'])
        self.assertIn('减少主动介入', summary)
        self.assertIn('沪指当日-0.60%', summary)
        self.assertIn('状态未知', summary)
        self.assertNotIn('已无量跌破', summary)

    def test_market_risk_missing_or_mixed_cutoff_is_unknown(self):
        second = copy.deepcopy(self.risk)
        self.draft['selected_codes'].append('000001')
        self.draft['model_evidence']['000001'] = {'risk': second, 'selection': {}}
        second['asof_ts'] = '20261020150000'
        self.assertIn('截止或内容不一致', ' '.join(plan_evidence.presentation(self.draft)['market_risk_summary']))
        second['status'] = 'MISSING'
        self.assertIn('仅有1/2只', ' '.join(plan_evidence.presentation(self.draft)['market_risk_summary']))

    def test_unconfirmed_detail_is_labelled_and_read_only(self):
        with patch.object(trade_plan_v1, 'load_plan', return_value=None), patch.object(
                workflow, 'latest', return_value=self.draft), patch.object(workflow, 'snapshot', return_value={}), patch.object(
                workflow, 'plan_readiness', return_value={'current': True}), patch.object(
                workflow, 'confirm_plan', side_effect=AssertionError('NO_CONFIRM')):
            view = plan_evidence.model_view('risk')
        self.assertEqual(view['status'], 'UNCONFIRMED_DRAFT')
        self.assertIn('只读', view['source_note'])
        self.assertIn('1.20%', '\n'.join(view['items'][0]['risk_lines']))

    def test_stale_detail_cannot_look_current(self):
        with patch.object(trade_plan_v1, 'load_plan', return_value=None), patch.object(
                workflow, 'latest', return_value=self.draft), patch.object(workflow, 'snapshot', return_value={}), patch.object(
                workflow, 'plan_readiness', return_value={'current': False}):
            view = plan_evidence.model_view('risk')
        self.assertEqual(view['status'], 'STALE_DRAFT')
        self.assertIn('旧草案', view['source_note'])

    def test_confirmed_plan_never_replaced_by_draft(self):
        confirmed = dict(data_date=20261021, stocks=[{'code': '000629'}],
                         model_evidence=self.draft['model_evidence'])
        with patch.object(trade_plan_v1, 'load_plan', return_value=confirmed), patch.object(
                workflow, 'latest', side_effect=AssertionError('DO_NOT_SUBSTITUTE')):
            view = plan_evidence.model_view('risk')
        self.assertEqual(view['status'], 'CONFIRMED_PLAN')
        self.assertEqual(view['items'][0]['code'], '000629')
        self.assertIn('沪指当日-0.60%', ' '.join(view['market_risk_summary']))


if __name__ == '__main__':
    unittest.main()
