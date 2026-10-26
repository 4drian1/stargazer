"""No network/fees/notifications: new runtime safety boundaries."""
from datetime import datetime
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import tarot_monitor_v2 as live
import golden_monitor
from golden_rules import digest


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.redirect = patch.object(live.runtime, 'directory', return_value=self.root)
        self.redirect.start()
        self.plan = dict(assistant_plan_id='confirmed', rows=[dict(code='600121')])
        self.risk = dict(status='AVAILABLE', asof_ts=20261023094500, context=dict(stock_close=9.8))
        self.signal = dict(status='RULE_TRIGGERED', node=9., branch='RECLAIM')

    def tearDown(self):
        self.redirect.stop()
        self.temp.cleanup()

    def test_freshness_rejects_future_and_old(self):
        self.assertTrue(live.fresh(20261023094500, datetime(2026,10,23,9,50)))
        self.assertFalse(live.fresh(20261023094500, datetime(2026,10,23,9,44)))
        self.assertFalse(live.fresh(20261023094500, datetime(2026,10,23,10)))

    def test_missing_risk_never_pays(self):
        with patch('deepseek_api.complete') as api:
            result = live.advise({},self.plan,'600121',20261023094500,{},self.signal)
        api.assert_not_called()
        self.assertEqual(result['status'],'WAIT_RISK')

    def test_no_price_confirmation_never_pays(self):
        with patch.object(live,'risk_ready',return_value=True), patch('deepseek_api.complete') as api:
            result = live.advise({},self.plan,'600121',20261023094500,self.risk,dict(status='WAIT_SIGNAL'))
        api.assert_not_called()
        self.assertEqual(result['status'],'WAIT_SIGNAL')

    def test_paid_failure_reserved_and_not_retried(self):
        with patch.object(live,'risk_ready',return_value=True), patch.object(live,'fresh',return_value=True), \
                patch.object(live,'enabled',return_value=True), patch('deepseek_api.complete',side_effect=ValueError('budget')) as api:
            one=live.advise({},self.plan,'600121',20261023094500,self.risk,self.signal)
            two=live.advise({},self.plan,'600121',20261023094500,self.risk,self.signal)
        self.assertEqual(api.call_count,1)
        self.assertEqual(one,two)
        self.assertEqual(one['status'],'AI_FAILED')
        self.assertEqual(one['risk'],self.risk)

    def test_late_ai_buy_not_actionable(self):
        response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"asof":20261023094500,"decision":"BUY","supporting_ids":["stock.price"],"opposing_ids":[],"unknown_ids":[],"observed_rule_status":"RULE_TRIGGERED","observed_branch":"RECLAIM","rationale":"价格条件支持人工观察"}'))])
        with patch.object(live,'risk_ready',return_value=True), patch.object(live,'fresh',side_effect=[True,False]), \
                patch.object(live,'enabled',return_value=True), patch('deepseek_api.complete',return_value=response):
            result=live.advise({},self.plan,'600121',20261023094500,self.risk,self.signal)
        self.assertEqual(result['status'],'ADVICE_EXPIRED')

    def test_holding_uses_current_risk_not_entry_signal(self):
        response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"asof":20261023094500,"decision":"HOLD","supporting_ids":["stock.price"],"opposing_ids":[],"unknown_ids":[],"observed_rule_status":null,"observed_branch":null,"rationale":"当前风险支持继续观察持仓"}'))])
        position=dict(code='600121',entry=8.,entry_date='2026-10-22')
        with patch.object(live,'risk_ready',return_value=True),patch.object(live,'fresh',return_value=True), \
                patch.object(live,'enabled',return_value=True),patch('deepseek_api.complete',return_value=response) as api:
            result=live.advise({},self.plan,'600121',20261023094500,self.risk,position=position)
        self.assertEqual(result['status'],'ADVICE_READY')
        facts=json.loads(api.call_args.args[1][1]['content'])
        self.assertIsNone(facts['signal'])
        self.assertEqual(facts['risk']['context']['stock_close'],9.8)
        self.assertEqual(facts['position']['entry'],8.)

    def test_wrong_plan_ui_read_never_uses_advice(self):
        folder=self.root/'tarot_v2';folder.mkdir()
        (folder/'latest.json').write_text(json.dumps(dict(plan_id='other',cutoff=20261023094500,rows={})))
        result=live.status_for(dict(assistant_plan_id='confirmed'),'600121','20261023094600')
        self.assertEqual(result['status'],'WAIT_MINUTE_EVIDENCE')

    def test_reclaim_requires_two_completed_closes(self):
        rule=dict(levels=[9.])
        plan=dict(rule_version=live.VERSION,rows=[dict(code='600121',entry_rule=rule)])
        value=dict(cutoff=20261023094500,rows={'600121':dict(rule_hash=digest(rule),status='RULE_TRIGGERED',
                   trigger_time=20261023094500,price_asof=20261023094500,node=9.,branch='RECLAIM',last_two_closes=[8.9,9.1])})
        with self.assertRaisesRegex(ValueError,'TWO_COMPLETED_RECLAIM'):
            golden_monitor.validate_result(value,plan)
        value['rows']['600121']['last_two_closes']=[9.,9.1]
        golden_monitor.validate_result(value,plan)

    def test_rsi_requires_stop_and_divergence(self):
        rule=dict(levels=[9.]);plan=dict(rule_version=live.VERSION,rows=[dict(code='600121',entry_rule=rule)])
        value=dict(cutoff=20261023094500,rows={'600121':dict(rule_hash=digest(rule),status='RULE_TRIGGERED',
                   trigger_time=20261023094500,price_asof=20261023094500,node=9.,branch='RSI',flags={})})
        with self.assertRaisesRegex(ValueError,'V2_RSI'):
            golden_monitor.validate_result(value,plan)

    def test_atr_exit_and_t1_keep_human_decision(self):
        position=dict(code='600121',entry=10.,entry_date='2026-10-22',
                      tarot_execution=dict(stop_price=9.5,horizon_sessions=3))
        risk=dict(self.risk,context=dict(stock_close=9.4),trading_days=[20261022,20261023])
        with patch.object(live,'risk_ready',return_value=True),patch('deepseek_api.complete') as api:
            result=live.advise({},self.plan,'600121',20261023094500,risk,position=position)
            same=live.advise({},self.plan,'600121',20261022094500,dict(risk,asof_ts=20261022094500),position=position)
        api.assert_not_called()
        self.assertEqual(result['decision'],'EXIT')
        self.assertEqual(same['status'],'T1_WAIT')
        self.assertFalse(result['automatic_trading'])

    def test_confirmed_monitor_delivers_once_and_disabled_stops(self):
        from pentacles_monitor import PriceMonitor
        monitor=PriceMonitor(lambda:self.root)
        monitor.enable(True)
        plan=dict(self.plan,plan_ready=True,monitor_date=20261023,rule_version=live.VERSION)
        (self.root/'assistant_plan.json').write_text(json.dumps(plan))
        review=dict(status='ADVICE_READY',decision='BUY',code='600121',cutoff=20261023094500,
                    reason='同截止风险已核验',risk=dict(context=dict(amp_232=.03,amp_266=.04)))
        value=dict(status='MONITORING',cutoff=20261023094500,rows={'600121':dict(advice=review)})
        broadcast,send=Mock(),Mock(return_value=(True,'ok'))
        with patch.object(live,'evaluate',return_value=value):
            for _ in range(2):monitor._tick_golden(plan,dict(push=dict(enabled=True)),datetime(2026,10,23,9,46),broadcast,send,None)
            monitor.enable(False)
            monitor._tick_golden(plan,{},datetime(2026,10,23,9,46),broadcast,send,None)
        self.assertEqual(broadcast.call_count,1)
        self.assertEqual(send.call_count,1)


if __name__=='__main__':unittest.main()
