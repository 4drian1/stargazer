"""Golden positions stay available independently of model admission."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import golden_stock_analysis as golden
from agent.brain import Brain
import assistant_workflow as workflow
import test_manual_plan_observations as manual_tests


def mapping(sources):
    return dict(policy='GOLDEN_LADDER_STOP_V3',bands=[dict(node=9.,lower=8.99,upper=9.01,
        sources=sources,shared_peak_groups=1)],primary_wave={},temporary_waves=[],cluster_max_span_fraction=.008)


def source(low_day=20261001,ratio=.618):
    return dict(low_day=low_day,high_day=20261012,low=8.,high=12.,ratio=ratio,
                kind='RETRACEMENT',known_at=20261014)


class GoldenToolTests(unittest.TestCase):
    def test_zone_is_node_to_plus_point2atr_capped_at_close(self):
        value=golden.summarize_map(mapping([source()]),9.05,1.)
        zone=value['node_zones'][0]
        self.assertEqual((zone['zone_lower'],zone['zone_upper']), (9.,9.05))
        self.assertAlmostEqual(zone['upper_unclipped'],9.2)

    def test_two_ratios_of_one_wave_are_not_two_sources(self):
        value=golden.summarize_map(mapping([source(),source(ratio=.5)]),10.,1.)
        self.assertEqual(value['node_zones'][0]['source_waves'],1)
        self.assertEqual(value['resonance_bands'],[])

    def test_shared_peak_is_explicitly_correlated(self):
        value=golden.summarize_map(mapping([source(),source(low_day=20260920)]),10.,1.)
        band=value['resonance_bands'][0]
        self.assertEqual((band['source_waves'],band['shared_peak_groups']),(2,1))

    def test_bad_atr_or_above_close_never_makes_zone(self):
        for atr in [0,float('nan')]:
            with self.assertRaises(ValueError):golden.summarize_map(mapping([source()]),10.,atr)
        with self.assertRaises(ValueError):golden.summarize_map(mapping([source()]),8.5,1.)

    def test_invalid_code_never_launches_worker(self):
        for code in ['688500;echo x','../secret',688500]:
            with self.assertRaises(ValueError):golden.stock_golden_levels(code)

    def test_explicit_three_stock_request_gets_positions_despite_admission(self):
        codes=['688500','002388','002815'];seen=[]
        def dispatch(name,args):
            seen.append((name,args['code']))
            if name=='model_stock_inference':
                return dict(code=args['code'],status='NOT_COVERED',on_demand=dict(selection=dict(quality_state='NOT_ADMITTED',selection_score=None)))
            return dict(code=args['code'],status='AVAILABLE',node_zones=[dict(node=9.,zone_upper=9.2)],plan_changed=False)
        answer=SimpleNamespace(content='三股位置分别列出，未通过选股不等于位置不可算。',tool_calls=[],
            model_dump=lambda **kw:dict(role='assistant',content='原始位置事实'))
        with tempfile.TemporaryDirectory() as tmp,patch('paths.APP_DIR',tmp):
            brain=Brain.__new__(Brain);brain.cfg={};brain.max_rounds=1
            brain.archive=Path(tmp)/'session.jsonl'
            brain.messages=[dict(role='system',content='位置与准入分列'),dict(role='assistant',content='此前提到600178')]
            with patch.object(brain,'_dispatch',side_effect=dispatch),patch('deepseek_api.complete',return_value=SimpleNamespace(choices=[SimpleNamespace(message=answer)])),patch.object(workflow,'confirm_plan',side_effect=AssertionError('NO_PLAN_CHANGE')):
                brain.chat('688500 002388 002815 请调用模型分析并给黄金共振位')
        self.assertEqual(seen,[(name,c) for c in codes for name in ('model_stock_inference','stock_golden_levels')])

    def test_golden_only_request_does_not_claim_model_inference(self):
        seen=[]
        answer=SimpleNamespace(content='位置事实',tool_calls=[],model_dump=lambda **kw:dict(role='assistant',content='位置事实'))
        with tempfile.TemporaryDirectory() as tmp,patch('paths.APP_DIR',tmp):
            brain=Brain.__new__(Brain);brain.cfg={};brain.max_rounds=1;brain.archive=Path(tmp)/'chat.jsonl'
            brain.messages=[dict(role='system',content='位置')]
            with patch.object(brain,'_dispatch',side_effect=lambda n,a:seen.append(n) or dict(status='AVAILABLE')),patch('deepseek_api.complete',return_value=SimpleNamespace(choices=[SimpleNamespace(message=answer)])):
                brain.chat('688500的黄金共振位在哪里')
        self.assertEqual(seen,['stock_golden_levels'])


class ManualZoneTests(manual_tests.ManualPlanTests):
    def test_explicit_region_is_saved_only_after_confirmation(self):
        draft,_=self.chat('请把688500加入次日计划 关注区间12.30-12.50',returned=self.codes)
        self.assertEqual(draft['manual_observations'][0]['price_zone'],[12.3,12.5])
        self.assertFalse((self.base/'dist/data/trade_plan.json').exists())
        workflow.confirm_plan(draft['id'],True)
        formal=json.loads((self.base/'dist/data/trade_plan.json').read_bytes())
        stock=next(s for s in formal['stocks'] if s['code']=='688500')
        self.assertEqual(stock['buy_zone'],[12.3,12.5]);self.assertEqual(stock['source'],'USER')
        self.assertIsNone(stock['entry_rule']);self.assertNotIn('688500',formal['model_evidence'])
        import trade_plan_v1
        result=trade_plan_v1.plan_check('688500',12.4,formal['monitor_date']+'100000',plan=formal)
        self.assertEqual(result['status'],'TRIGGERED')

    def test_analysis_and_negation_do_not_assign_region(self):
        self.assertEqual(workflow.manual_observations('不要把688500加入次日计划 关注区间12.30-12.50',None,set()),[])
        previous=dict(manual_observations=[dict(code='688500',price_zone=[9.,9.2],user_requests=['原条件'])])
        result=workflow.manual_observations('分析688500关注区间12.30-12.50',previous,set())
        self.assertEqual(result[0]['price_zone'],[9.,9.2])

    def test_reversed_region_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'下沿'):
            workflow.manual_observations('请把688500加入计划 关注区间13-12',None,set())


if __name__=='__main__':unittest.main()
