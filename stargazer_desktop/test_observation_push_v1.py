"""Actual monitor entry: price notice independent of buy qualification and AI."""
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import tarot_monitor_v2 as live
from pentacles_monitor import PriceMonitor


class ObservationPushTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.monitor=PriceMonitor(lambda:self.root)
        self.monitor.enable(True)
        self.now=datetime(2026,10,23,11,26)
        self.cutoff=20261023112500
        self.plan=dict(assistant_plan_id='confirmed',plan_ready=True,monitor_date=20261023,
            rule_version=live.VERSION,rows=[dict(code='688004',reference_price=37.,
                entry_rule=dict(levels=[35.,34.,33.5]))])
        (self.root/'assistant_plan.json').write_text(json.dumps(self.plan))
        self.quote=dict(time='20261023112600',prev=37.,price=34.2,low=34.1)
        self.value=dict(status='WAIT_INDEX_FEED',rows={})
        self.broadcast=Mock()
        self.send=Mock(return_value=(True,'ok'))
        self.runtime=patch.object(live,'evaluate',side_effect=lambda *a:self.value)
        self.bulletin=patch('risk_outlook_v1.latest',return_value={})
        self.runtime.start();self.bulletin.start()

    def tearDown(self):
        self.runtime.stop();self.bulletin.stop();self.temp.cleanup()

    def tick(self):
        quotes=Mock(return_value={'688004':self.quote})
        self.monitor._tick(self.plan,{'push':{'enabled':True}},self.now,quotes,self.broadcast,self.send)
        return quotes

    def signal(self,advice):
        self.value=dict(status='MONITORING',event_cutoff=self.cutoff,rows={'688004':dict(
            status='RULE_TRIGGERED',node=34.,branch='RECLAIM',trigger_time=20261023112500,advice=advice)})

    def test_quote_arrival_delivered_without_rule_or_risk(self):
        self.tick();self.tick()
        self.assertEqual(self.broadcast.call_count,1);self.assertEqual(self.send.call_count,1)
        message=self.broadcast.call_args.args[0]
        self.assertIn('观察位已到达',message['title']);self.assertIn('尚不代表止跌',message['body'])
        self.assertIn('35.0000',message['body'])

    def test_each_lower_position_once_and_restart_dedup(self):
        self.tick()
        self.quote.update(low=33.4,price=33.6)
        self.tick();self.tick()
        self.monitor=PriceMonitor(lambda:self.root)
        self.tick()
        self.assertEqual(self.send.call_count,2)
        self.assertIn('34.0000、33.5000',self.send.call_args.args[2])

    def test_stale_bad_basis_or_invalid_low_never_alert(self):
        for change in [dict(time='20261023110000'),dict(prev=36.),dict(low=float('nan')),dict(low='bad')]:
            original=dict(self.quote);self.quote.update(change);self.tick();self.quote=original
        self.broadcast.assert_not_called()

    def test_buy_and_defer_follow_quote_notification_with_reason(self):
        self.tick()
        for decision in ['DEFER','BUY']:
            self.signal(dict(status='ADVICE_READY',decision=decision,code='688004',cutoff=self.cutoff,
                reason='板块仍弱，继续等待' if decision=='DEFER' else '价格与修复条件配合，人工考虑'))
            self.tick();self.tick()
        self.assertEqual(self.broadcast.call_count,3)
        self.assertIn('板块仍弱',self.broadcast.call_args_list[1].args[0]['body'])
        self.assertIn('建议考虑买入',self.broadcast.call_args_list[2].args[0]['title'])

    def test_pending_then_ai_failure_both_deliver_no_retry(self):
        self.quote.update(price=36.,low=36.)
        for status in ['WAIT_RISK','AI_ATTEMPT_RESERVED','AI_FAILED','ENTRY_SESSION_CLOSED']:
            self.signal(dict(status=status))
            self.tick();self.tick()
        self.assertEqual(self.send.call_count,4)
        self.assertIn('AI分析失败',self.send.call_args_list[2].args[2])
        self.assertIn('结束当日入场',self.send.call_args_list[3].args[2])

    def test_t1_wait_and_hold_opinion_are_not_filtered_out(self):
        self.quote.update(price=36.,low=36.)
        for status,decision in [('T1_WAIT','DEFER'),('ADVICE_READY','HOLD')]:
            self.value=dict(status='MONITORING',cutoff=self.cutoff,rows={},holdings={'688004':dict(
                status=status,decision=decision,code='688004',cutoff=self.cutoff,reason='持仓风险与T+1约束')})
            self.tick()
        self.assertEqual(self.send.call_count,2)

    def test_no_delivery_after_switch_off_plan_change_or_wrong_day(self):
        self.monitor.enable(False);self.tick();self.monitor.enable(True)
        (self.root/'assistant_plan.json').write_text('{}');self.tick()
        (self.root/'assistant_plan.json').write_text(json.dumps(self.plan))
        self.now=datetime(2026,10,26,11,31);self.quote['time']='20261026113100'
        self.tick();self.broadcast.assert_not_called()

    def test_quote_failure_does_not_block_defer_or_send_retry(self):
        self.signal(dict(status='ADVICE_READY',decision='DEFER',code='688004',cutoff=self.cutoff,reason='风险较高'))
        self.send.return_value=(False,'offline')
        for _ in range(2):
            self.monitor._tick(self.plan,{'push':{'enabled':True}},self.now,Mock(side_effect=OSError()),self.broadcast,self.send)
        self.assertEqual(self.broadcast.call_count,1);self.assertEqual(self.send.call_count,1)


if __name__=='__main__':unittest.main()
