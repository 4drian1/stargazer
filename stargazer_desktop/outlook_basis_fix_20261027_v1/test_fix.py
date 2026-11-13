"""Replay the actual empty-reference reply; no network, paid calls or pushes."""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

HERE=Path(__file__).resolve().parent;APP=HERE.parent
sys.path.insert(0,str(APP))
import risk_outlook_v1 as outlook
import tarot_monitor_v2 as live
import tarot_event_v1 as events
from golden_rules import digest
from pentacles_monitor import PriceMonitor


class BasisTests(unittest.TestCase):
    def setUp(self):
        self.failed=json.loads((HERE/'ORIGINAL_FAILED.json').read_bytes())
        self.cutoff=self.failed['asof']
        self.now=datetime(2026,10,27,10,1)

    def test_real_reply_still_rejected_but_local_facts_present_without_ai_claim(self):
        before=deepcopy(self.failed)
        parsed,_=outlook.parse_reply(self.failed['raw_reply'])
        with self.assertRaisesRegex(ValueError,'OUTLOOK_NO_BASIS'):
            outlook.validate(parsed,self.failed['computed_facts'],self.cutoff)
        result=outlook.with_local_failure(self.failed)
        self.assertEqual(result['status'],'AI_FAILED')
        self.assertEqual(result['raw_reply'],before['raw_reply'])
        self.assertEqual(self.failed,before)
        self.assertEqual(result['fallback_kind'],'CODE_FACTS_ONLY')
        for text in ['沪指实时报价 3690.9600','-0.18%','创业板实时报价 2634.6400','-0.14%',
                     '不是AI研判','没有当日已确认候选','当前登记为空','未自动重试']:
            self.assertIn(text,result['text'])
        self.assertNotIn('盘初快照小幅走弱',result['text'])

    def test_saved_failed_slot_is_read_only_and_never_retried(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(live.runtime,'directory',return_value=Path(temp)), \
                patch('deepseek_api.complete') as api:
            path=Path(temp)/'risk_outlook'/f'{self.failed["key"]}.json'
            path.parent.mkdir();raw=json.dumps(self.failed).encode();path.write_bytes(raw)
            result=outlook.generate({},dict(plan_hash=self.failed['plan_hash']),self.cutoff,
                mode='INTRADAY',plan_id=self.failed['plan_id'])
            api.assert_not_called()
            self.assertIn('沪指实时报价',result['text'])
            self.assertEqual(path.read_bytes(),raw)

    def test_valid_quote_references_pass_without_a_daily_plan(self):
        parsed,_=outlook.parse_reply(self.failed['raw_reply'])
        parsed['opposing_ids']=['market.sse','market.cyb']
        self.assertIn('行情快照：沪指实时报价',outlook.validate(parsed,self.failed['computed_facts'],self.cutoff))

    def test_generation_requires_basis_and_retains_raw_failure(self):
        response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.failed['raw_reply']))])
        with tempfile.TemporaryDirectory() as temp,patch.object(live.runtime,'directory',return_value=Path(temp)), \
                patch.object(outlook,'collect',return_value=self.failed['computed_facts']), \
                patch.object(live,'fresh',return_value=True), \
                patch('deepseek_api.complete',return_value=response) as api:
            view=dict(items=[],plan_hash='test',plan_status='NO_CURRENT_PLAN')
            first=outlook.generate({},view,self.cutoff,'INTRADAY',positions=[],overlay=False,plan_id='test')
            second=outlook.generate({},view,self.cutoff,'INTRADAY',positions=[],overlay=False,plan_id='test')
            self.assertEqual(first,second)
            self.assertEqual(api.call_count,1)
            request=json.loads(api.call_args.args[1][1]['content'])
            self.assertEqual(request['reference_policy']['minimum_total_references'],1)
            self.assertEqual(request['plan_status'],'NO_CURRENT_PLAN')
            self.assertEqual(first['validation_error'],'OUTLOOK_NO_BASIS')
            self.assertEqual(first['raw_reply'],self.failed['raw_reply'])
            self.assertIn('沪指实时报价',first['text'])

    def test_market_price_facts_collected_without_candidate_rows(self):
        evidence=dict(asof=self.cutoff,status='PRICE_FACTS_ONLY',
            five=dict(rsi6=40.,rsi15=45.,rsi_above=False,cross_up_times=[]),
            fifteen=dict(evidence_time=self.cutoff,rsi_above=False),
            price_stable_shape=False,levels=[],summary='本地完整五分钟价格证据',resonance_bands=[])
        with patch('plan_evidence.archived_price_overlay',return_value={'sse':evidence,'cyb':evidence}):
            facts=outlook.collect(dict(items=[]),self.cutoff)
        self.assertEqual(facts['index.sse.rsi6']['value'],40.)
        self.assertEqual(facts['index.cyb.crosses']['status'],'KNOWN')
        self.assertEqual(facts['index.sse.volume']['status'],'UNKNOWN')
        self.assertFalse(any(k.startswith(('None.','.missing','stock.')) for k in facts))

    def test_expired_empty_plan_still_schedules_market_but_not_stock_worker(self):
        plan=dict(assistant_plan_id='old',monitor_date=20261026,rows=[dict(code='002878')])
        for prefix in [True,False]:
            with self.subTest(prefix=prefix),patch('services.wl_load',return_value={'positions':[]}), \
                    patch.object(events,'evaluate',return_value={}),patch.object(events,'read_risk',return_value={}), \
                    patch.object(live,'index_prefix_ready',return_value=prefix), \
                    patch.object(live.threading,'Thread') as thread,patch.object(outlook,'queue') as queue:
                result=live.evaluate(plan,{},self.now)
            self.assertEqual(result['status'],'MARKET_ONLY')
            self.assertEqual(queue.call_count,1)
            view=queue.call_args.args[1]
            self.assertEqual(view['items'],[])
            self.assertEqual(view['plan_status'],'NO_CURRENT_PLAN')
            thread.assert_not_called()

    def test_failed_bulletin_push_contains_local_quotes_and_is_sent_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);monitor=PriceMonitor(lambda:root);monitor.enable(True)
            plan=dict(assistant_plan_id=self.failed['plan_id'],plan_ready=True,monitor_date=20261026,
                rule_version=live.VERSION,rows=[])
            (root/'assistant_plan.json').write_text(json.dumps(plan))
            # Match the monitor's real plan binding, without modifying the archived failure.
            bulletin=dict(self.failed,plan_hash=digest(plan))
            send=Mock(return_value=(True,'accepted'));broadcast=Mock()
            with patch.object(live,'evaluate',return_value=dict(status='MARKET_ONLY',cutoff=self.cutoff,rows={})), \
                    patch.object(outlook,'latest',return_value=bulletin):
                for _ in range(2):
                    monitor._tick_tarot_v2(plan,{'push':{'enabled':True}},self.now,broadcast,send,None)
            self.assertEqual(send.call_count,1)
            self.assertIn('沪指实时报价',send.call_args.args[2])
            self.assertIn('不是AI研判',send.call_args.args[2])
            self.assertEqual(broadcast.call_args.args[0]['delivery'],'SENT')


if __name__=='__main__':unittest.main()
