"""Actual failed reply replay and provider receipt propagation; no network."""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import risk_outlook_v1 as outlook
import tarot_monitor_v2 as live
from pentacles_monitor import PriceMonitor

EVIDENCE=Path(__file__).parent/'push_outlook_fix_20261026'/'original_failed_outlook.json'


class ReplyTests(unittest.TestCase):
    def setUp(self):
        self.original=json.loads(EVIDENCE.read_bytes())
        self.raw=self.original['raw_reply']
        self.facts=self.original['computed_facts']
        self.cutoff=self.original['asof']

    def test_actual_reply_moves_only_three_missing_references(self):
        parsed,_=outlook.parse_reply(self.raw)
        before=deepcopy(parsed)
        with self.assertRaisesRegex(ValueError,'OUTLOOK_UNKNOWN_AS_FACT'):
            outlook.validate(parsed,self.facts,self.cutoff)
        qualified,notes=outlook.qualify_references(parsed,self.facts)
        self.assertEqual(parsed,before)
        self.assertEqual(qualified['bias'],before['bias'])
        self.assertEqual(qualified['sections'],before['sections'])
        self.assertEqual(qualified['focus_ids'],before['focus_ids'])
        self.assertEqual(len(notes),3)
        self.assertEqual(len(qualified['unknown_ids']),13)
        self.assertEqual(qualified['opposing_ids'],[])
        self.assertIn('不能把当刻未证当修复',outlook.validate(qualified,self.facts,self.cutoff))

    def test_fabrication_and_duplicates_are_not_repaired(self):
        for key in ['invented', '300225.plan.0']:
            value=json.loads(self.raw)
            value['unknown_ids'].append(key)
            with self.assertRaisesRegex(ValueError,'OUTLOOK_FACT_REFERENCE'):
                outlook.qualify_references(value,self.facts)

    def test_known_cannot_be_reclassified_as_unknown(self):
        value=json.loads(self.raw)
        key=value['supporting_ids'].pop()
        value['unknown_ids'].append(key)
        value,_=outlook.qualify_references(value,self.facts)
        with self.assertRaisesRegex(ValueError,'OUTLOOK_KNOWN_AS_UNKNOWN'):
            outlook.validate(value,self.facts,self.cutoff)

    def test_generate_uses_reference_policy_and_never_retries_slot(self):
        with tempfile.TemporaryDirectory() as folder:
            response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.raw))])
            with patch.object(live.runtime,'directory',return_value=Path(folder)), \
                 patch.object(outlook,'collect',return_value=self.facts), \
                 patch.object(live,'fresh',return_value=True), \
                 patch('deepseek_api.complete',return_value=response) as api:
                a=outlook.generate({},dict(items=[]),self.cutoff,'INTRADAY',overlay=False,plan_id='p')
                b=outlook.generate({},dict(items=[]),self.cutoff,'INTRADAY',overlay=False,plan_id='p')
            self.assertEqual(a,b)
            self.assertEqual(a['status'],'READY')
            self.assertEqual(a['raw_reply'],self.raw)
            self.assertEqual(len(a['reference_qualification']),3)
            self.assertIn('300225.missing',a['request']['reference_policy']['unknown_only_ids'])
            self.assertEqual(api.call_count,1)


class DeliveryTests(unittest.TestCase):
    def check_delivery(self,enabled,outcome,expected):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);monitor=PriceMonitor(lambda:root);monitor.enable(True)
            plan=dict(assistant_plan_id='p',plan_ready=True,monitor_date=20261023,rule_version=live.VERSION)
            (root/'assistant_plan.json').write_text(json.dumps(plan))
            cutoff=20261023113000
            bulletin=dict(status='READY',asof=cutoff,plan_id='p',text='风险事实待证',key='receipt-once')
            broadcast=Mock()
            send=Mock(side_effect=outcome) if isinstance(outcome,Exception) else Mock(return_value=outcome)
            with patch.object(live,'evaluate',return_value=dict(status='MONITORING',cutoff=cutoff,rows={})), \
                 patch.object(outlook,'latest',return_value=bulletin):
                for _ in range(2):
                    monitor._tick_tarot_v2(plan,{'push':{'enabled':enabled}},
                        datetime(2026,10,23,11,31),broadcast,send,None)
            self.assertEqual(send.call_count,1 if enabled else 0)
            self.assertEqual(broadcast.call_count,1)
            event=broadcast.call_args.args[0]
            self.assertEqual(event['delivery'],expected)
            self.assertEqual(event['wechat'],expected=='SENT')
            self.assertTrue(event['wechat_msg'])
            stored=json.loads((root/'monitor.json').read_bytes())
            alert=next(iter(stored['alerts'].values()))
            self.assertEqual(event['wechat'],alert['wechat'])
            self.assertEqual(event['delivery'],alert['delivery'])

    def test_success_reaches_ui_as_sent(self):
        self.check_delivery(True,(True,'ok'),'SENT')

    def test_failure_reaches_ui_with_reason(self):
        self.check_delivery(True,(False,'offline'),'FAILED')

    def test_sender_exception_still_has_local_notice_no_retry(self):
        self.check_delivery(True,OSError('offline'),'FAILED')

    def test_disabled_is_not_an_empty_failure(self):
        self.check_delivery(False,(True,'not-called'),'DISABLED')


if __name__=='__main__':unittest.main()
