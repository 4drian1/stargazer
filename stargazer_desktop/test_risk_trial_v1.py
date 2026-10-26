"""Trial integration, scheduling, mixed clocks and paid-attempt boundaries."""
from datetime import datetime
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock

import risk_outlook_v1 as outlook
import tarot_monitor_v2 as live
import tarot_event_v1 as events
from trade_facts_v1 import fact_table,validate_and_render


class TrialTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.redirect=patch.object(live.runtime,'directory',return_value=self.root)
        self.redirect.start()
        self.cutoff=20261023113000
        self.facts={'index.sse.band.0':dict(asof=self.cutoff,status='KNOWN',value={'lower':3580.,'upper':3603.},text='真实带3580—3603'),
            'trial.sse.raw_state':dict(asof=self.cutoff,status='TRIAL',value='STRONG_REPAIR',text='试运行强修复；量能未核销'),
            'index.sse.volume':dict(asof=self.cutoff,status='UNKNOWN',value=None,text='量能未知')}
        self.reply=dict(asof=self.cutoff,bias='REPAIR_BIAS',sections={k:'价格修复有迹象，继续核验配合' for k in ('market','boards','leaders','holdings','plan')},
            focus_ids=['index.sse.band.0'],supporting_ids=['trial.sse.raw_state'],opposing_ids=[],unknown_ids=['index.sse.volume'],
            danger_signals=['TWO_5M_CLOSES_BELOW'],repair_signals=['TWO_5M_RECLAIM'])

    def tearDown(self):
        self.redirect.stop();self.temp.cleanup()

    def test_half_hour_has_eight_session_slots(self):
        slots=[h*10000+m*100 for h in range(9,16) for m in range(60) if outlook.half_hour(20261023000000+h*10000+m*100)]
        self.assertEqual(slots,[100000,103000,110000,113000,133000,140000,143000,150000])
        self.assertEqual(events.cutoff5(datetime(2026,10,23,9,50)),20261023095000)
        self.assertEqual(events.cutoff5(datetime(2026,10,23,13,4)),20261023113000)

    def test_forecast_only_selects_existing_price_positions(self):
        text=outlook.validate(self.reply,self.facts,self.cutoff)
        self.assertIn('3580—3603',text);self.assertIn('试运行',text)
        bad=dict(self.reply,focus_ids=['invented.position'])
        with self.assertRaises(ValueError):outlook.validate(bad,self.facts,self.cutoff)

    def test_reply_structure_normalization_changes_no_values(self):
        original=json.loads(json.dumps(self.reply))
        for k in ('focus_ids','supporting_ids','opposing_ids','unknown_ids','danger_signals','repair_signals'):
            original['sections'][k]=original.pop(k)
        parsed,notes=outlook.parse_reply(json.dumps(original)[:-1])
        self.assertEqual(parsed,self.reply);self.assertEqual(len(notes),7)
        with self.assertRaises(ValueError):outlook.parse_reply('{"asof":1,"asof":2}')
        with self.assertRaises(ValueError):outlook.parse_reply('{"asof":')

    def test_trial_opinion_can_be_left_uncertain(self):
        value=dict(self.reply,supporting_ids=[],unknown_ids=['index.sse.volume','trial.sse.raw_state'])
        self.assertIn('试运行',outlook.validate(value,self.facts,self.cutoff))
        bad=dict(self.reply,sections=dict(self.reply['sections'],market='明天必涨5%'))
        with self.assertRaises(ValueError):outlook.validate(bad,self.facts,self.cutoff)

    def test_unknown_not_used_as_confirmed_but_trial_allowed(self):
        bad=dict(self.reply,supporting_ids=['index.sse.volume'],unknown_ids=[])
        with self.assertRaises(ValueError):outlook.validate(bad,self.facts,self.cutoff)
        self.assertIn('试运行',outlook.validate(self.reply,self.facts,self.cutoff))

    def test_attempt_saved_and_failure_never_retried(self):
        with patch.object(outlook,'collect',return_value=self.facts),patch('deepseek_api.complete',side_effect=ValueError('offline')) as api:
            one=outlook.generate({},dict(items=[]),self.cutoff,overlay=False)
            two=outlook.generate({},dict(items=[]),self.cutoff,overlay=False)
        self.assertEqual(api.call_count,1);self.assertEqual(one,two)
        self.assertEqual(one['computed_facts'],self.facts);self.assertEqual(one['status'],'AI_FAILED')

    def test_half_hour_keeps_one_reserved_input_even_if_later_records_change(self):
        response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.reply)))])
        with patch.object(outlook,'collect',return_value=self.facts),patch('deepseek_api.complete',return_value=response) as api,patch.object(live,'fresh',return_value=True):
            one=outlook.generate({},dict(items=[]),self.cutoff,'INTRADAY',overlay=False,plan_id='same')
            two=outlook.generate({},dict(items=[{'changed':True}]),self.cutoff,'INTRADAY',overlay=False,plan_id='same')
        self.assertEqual(api.call_count,1);self.assertEqual(one,two);self.assertEqual(one['status'],'READY')

    def test_mixed_cutoff_requires_current_both_indices_and_bounded_age(self):
        cutoff=20261023113500
        record=dict(status='AVAILABLE',asof_ts=self.cutoff,mixed_cutoff_observation=cutoff,
            context=dict(market_price_evidence={n:dict(asof=cutoff,status='PRICE_FACTS_ONLY') for n in ['sse','cyb']}))
        with patch.object(live,'risk_ready',return_value=True):
            self.assertTrue(live.event_risk_ready(record,cutoff))
            self.assertFalse(live.event_risk_ready(record,20261023114500))
            del record['context']['market_price_evidence']['cyb']
            self.assertFalse(live.event_risk_ready(record,cutoff))

    def test_half_hour_report_push_once(self):
        from pentacles_monitor import PriceMonitor
        monitor=PriceMonitor(lambda:self.root);monitor.enable(True)
        plan=dict(assistant_plan_id='p',plan_ready=True,monitor_date=20261023,rule_version=live.VERSION)
        (self.root/'assistant_plan.json').write_text(json.dumps(plan))
        value=dict(status='MONITORING',cutoff=self.cutoff,rows={})
        bulletin=dict(status='READY',asof=self.cutoff,plan_id='p',text='市场和持仓巡检',key='once')
        broadcast,send=Mock(),Mock(return_value=(True,'ok'))
        with patch.object(live,'evaluate',return_value=value),patch.object(outlook,'latest',return_value=bulletin):
            for _ in range(2):monitor._tick_tarot_v2(plan,{'push':{'enabled':True}},datetime(2026,10,23,11,31),broadcast,send,None)
        self.assertEqual(broadcast.call_count,1);self.assertIn('半小时风险巡检',broadcast.call_args.args[0]['title'])

    def test_new_repair_evidence_can_reassess_previous_defer(self):
        plan=dict(assistant_plan_id='p',rows=[dict(code='688004')]);signal=dict(status='RULE_TRIGGERED',branch='RECLAIM',node=34.,trigger_time=20261023112500)
        risk=dict(asof_ts=self.cutoff,context=dict(stock_close=34.13))
        def response(cutoff):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(dict(asof=cutoff,decision='DEFER',supporting_ids=['stock.price'],opposing_ids=[],unknown_ids=[],observed_rule_status='RULE_TRIGGERED',observed_branch='RECLAIM',rationale='等待背景进一步配合'))))])
        with patch.object(live,'risk_ready',return_value=True),patch.object(live,'fresh',return_value=True),patch.object(live,'enabled',return_value=True),patch('deepseek_api.complete',side_effect=[response(self.cutoff),response(20261023131500)]) as api:
            first=live.advise({},plan,'688004',self.cutoff,risk,signal=signal)
            again=live.advise({},plan,'688004',self.cutoff,risk,signal=signal)
            later=live.advise({},plan,'688004',20261023131500,dict(risk,asof_ts=20261023131500),signal=signal)
        self.assertEqual(api.call_count,2);self.assertEqual(first,again);self.assertEqual(later['status'],'ADVICE_READY')

    def test_old_amplitude_not_relabeled_as_current_prediction(self):
        facts=fact_table(20261023113500,dict(asof_ts=self.cutoff,context=dict(amp_232=.03,stock_close=34.5,stock_price_asof=20261023113500,amp_reference_price=34.13)))
        self.assertEqual(facts['risk.amp_232']['evidence_time'],self.cutoff)
        self.assertIn('非当前新推理',facts['risk.amp_232']['text'])
        self.assertEqual(facts['risk.reference_price']['value'],34.13)
        self.assertEqual(facts['risk.reference_price']['evidence_time'],self.cutoff)
        self.assertEqual(facts['stock.price']['value'],34.5)
        self.assertEqual(facts['stock.price']['evidence_time'],20261023113500)

    def test_late_index_prefix_waits_without_poisoning_whole_day(self):
        plan=dict(assistant_plan_id='p',monitor_date=20261023,rows=[dict(code='688004')])
        with patch('services.wl_load',return_value={'positions':[]}),patch.object(events,'evaluate',return_value={}),patch.object(live,'index_prefix_ready',return_value=False),patch.object(live,'_thread',None),patch.object(live.subprocess,'run') as worker:
            result=live.evaluate(plan,{},datetime(2026,10,23,9,46))
        worker.assert_not_called();self.assertEqual(result['status'],'WAIT_INDEX_FEED')
        self.assertFalse((self.root/'tarot_v2/failed_20261023.json').exists())

    def test_index_prefix_requires_both_current_sessions(self):
        mirror=self.root/'index_minutes_v1';mirror.mkdir()
        for name in ['sse','cyb']:(mirror/f'{name}_5min.meta.json').write_text(json.dumps(dict(last_complete_ts=20261023112500)))
        with patch('joint_risk_bridge._worker_paths',return_value=(self.root/'python',self.root/'worker.py')):
            self.assertFalse(live.index_prefix_ready(self.cutoff))
            for name in ['sse','cyb']:(mirror/f'{name}_5min.meta.json').write_text(json.dumps(dict(last_complete_ts=self.cutoff)))
            self.assertTrue(live.index_prefix_ready(self.cutoff))

    def test_report_stale_fields_remain_unknown(self):
        view=dict(items=[dict(code='688004',facts=dict(status='AVAILABLE',asof_ts=20261022150000,context=dict(amp_232=.03)))])
        facts=outlook.collect(view,self.cutoff,overlay=False)
        self.assertEqual(facts['688004.risk.amp_232']['status'],'UNKNOWN')
        self.assertIsNone(facts['688004.risk.amp_232']['value'])

    def test_report_reuses_same_cutoff_event_progress(self):
        folder=self.root/'tarot_events';folder.mkdir()
        event=dict(cutoff=self.cutoff,plan_hash='bound',rows={'688004':dict(status='WAIT_SIGNAL',last_two_closes=[34.1,34.2],next_lower_node=34.)})
        (folder/'latest.json').write_text(json.dumps(event))
        view=dict(plan_hash='bound',items=[dict(code='688004',facts={})])
        facts=outlook.collect(view,self.cutoff,overlay=False)
        self.assertIn('34.2000',facts['688004.progress']['text'])
        view['plan_hash']='other'
        self.assertEqual(outlook.collect(view,self.cutoff,overlay=False)['688004.progress']['status'],'UNKNOWN')

    def test_same_day_holding_exit_is_not_actionable(self):
        plan=dict(assistant_plan_id='p',rows=[dict(code='688004')])
        risk=dict(asof_ts=self.cutoff,context=dict(stock_close=34.13))
        reply=dict(asof=self.cutoff,decision='EXIT',supporting_ids=['stock.price'],opposing_ids=[],unknown_ids=[],
            observed_rule_status=None,observed_branch=None,rationale='风险扩大，建议人工退出')
        response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(reply)))])
        with patch.object(live,'risk_ready',return_value=True),patch.object(live,'fresh',return_value=True),patch.object(live,'enabled',return_value=True),patch('deepseek_api.complete',return_value=response):
            result=live.advise({},plan,'688004',self.cutoff,risk,position=dict(code='688004',entry_date='2026-10-23'))
        self.assertEqual(result['status'],'T1_WAIT');self.assertEqual(result['decision'],'DEFER')

    def test_partial_event_publishes_before_model_wait_and_delivers_once(self):
        from pentacles_monitor import PriceMonitor
        monitor=PriceMonitor(lambda:self.root);monitor.enable(True)
        plan=dict(assistant_plan_id='p',plan_ready=True,monitor_date=20261023,rule_version=live.VERSION)
        (self.root/'assistant_plan.json').write_text(json.dumps(plan))
        signal=dict(status='RULE_TRIGGERED',node=33.99,trigger_time=20261023112500,branch='RECLAIM',advice=dict(status='WAIT_RISK'))
        value=dict(status='PREWARMING',event_cutoff=self.cutoff,rows={'688004':signal})
        broadcast,send=Mock(),Mock(return_value=(True,'ok'))
        with patch.object(live,'evaluate',return_value=value),patch.object(outlook,'latest',return_value={}):
            for _ in range(2):monitor._tick_tarot_v2(plan,{'push':{'enabled':True}},datetime(2026,10,23,11,31),broadcast,send,None)
        self.assertEqual(broadcast.call_count,1);self.assertEqual(send.call_count,1)
        self.assertIn('风险核验中',broadcast.call_args.args[0]['title'])


if __name__=='__main__':unittest.main()
