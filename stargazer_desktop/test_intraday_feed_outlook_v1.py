"""Actual failed reply plus real entry contracts; no paid requests or messages."""
from datetime import datetime
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import risk_outlook_v1 as outlook
import tarot_monitor_v2 as live
import tarot_event_v1 as events
import golden_monitor
from golden_rules import digest
from tools import market

STAGE=Path(__file__).parent/'intraday_feed_fix_20261026'
ROOT=Path(__file__).parents[1]


class FeedTests(unittest.TestCase):
    def setUp(self):
        self.cutoff=20261026110000
        self.now=datetime(2026,10,26,11,0,55)

    def test_actual_missing_echo_uses_bound_request_without_changing_opinion(self):
        original=json.loads((STAGE/'original_20261026110000.json').read_bytes())
        parsed,_=outlook.parse_reply(original['raw_reply'])
        self.assertNotIn('asof',parsed)
        with self.assertRaisesRegex(ValueError,'OUTLOOK_TIME_OR_BIAS'):
            outlook.validate(parsed,original['computed_facts'],self.cutoff)
        bound,notes=outlook.bind_reply_time(parsed,self.cutoff)
        self.assertEqual({k:v for k,v in bound.items() if k!='asof'},parsed)
        self.assertEqual(notes,['asof_from_bound_request_not_model_echo'])
        self.assertIn('当前未知',outlook.validate(bound,original['computed_facts'],self.cutoff))

    def test_wrong_explicit_clock_and_bias_are_never_repaired(self):
        original=json.loads((STAGE/'original_20261026110000.json').read_bytes())
        parsed=json.loads(original['raw_reply'])
        for change in (dict(asof=20261023110000),dict(asof=str(self.cutoff)),dict(bias='BUY')):
            bound,_=outlook.bind_reply_time(dict(parsed,**change),self.cutoff)
            with self.assertRaisesRegex(ValueError,'OUTLOOK_TIME_OR_BIAS'):
                outlook.validate(bound,original['computed_facts'],self.cutoff)

    def test_real_index_parser_retains_provider_time_and_single_request(self):
        fields=['']*46
        for k,v in {1:'上证指数',2:'000001',3:'3600',30:'20261026110050',32:'0.8',37:'10000'}.items():fields[k]=v
        with patch.object(market,'_get',return_value='v_sh000001="'+'~'.join(fields)+'";') as get:
            result=market.index_quotes(tries=1)
        self.assertEqual(result[0]['time'],'20261026110050')
        self.assertEqual(get.call_count,1)
        self.assertEqual(get.call_args.kwargs['tries'],1)
        self.assertIn('sh000001',get.call_args.args[0])
        self.assertIn('sz399006',get.call_args.args[0])

    def test_quote_time_after_bar_is_separate_not_future_bar_evidence(self):
        quotes=[dict(code='000001',price=3600,pct=-.8,time='20261026110050'),
                dict(code='399006',price=2650,pct=1.2,time='20261026110051')]
        with patch.object(market,'index_quotes',return_value=quotes) as fetch:
            facts=outlook.live_index_facts(self.cutoff,self.now)
        self.assertEqual(fetch.call_args.kwargs,dict(tries=1))
        self.assertEqual(facts['market.sse']['value'],-.008)
        self.assertEqual(facts['market.cyb']['value'],.012)
        self.assertEqual(facts['market.sse']['bar_cutoff'],self.cutoff)
        self.assertEqual(facts['market.sse']['evidence_time'],20261026110050)
        self.assertEqual(facts['market.sse']['observed_at'],20261026110055)
        self.assertEqual(facts['market.sse']['evidence_kind'],'QUOTE_SNAPSHOT')
        self.assertNotIn('trial.sse.raw_state',facts)

    def test_stale_future_missing_nonfinite_quotes_never_become_known(self):
        for delta in (dict(time='20261023150000'),dict(time='20261026110056'),
                      dict(time='20261026100000'),dict(time=None),dict(price=float('nan')),
                      dict(pct=float('inf')),dict(code='399001')):
            row=dict(code='000001',price=3600,pct=1,time='20261026110050');row.update(delta)
            with patch.object(market,'index_quotes',return_value=[row]):
                self.assertEqual(outlook.live_index_facts(self.cutoff,self.now),{})

    def test_historical_request_never_reads_todays_quotes(self):
        with patch.object(market,'index_quotes') as fetch:
            self.assertEqual(outlook.live_index_facts(20261023110000,self.now),{})
        fetch.assert_not_called()

    def test_no_live_evidence_no_paid_slot_but_local_wait_report(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            with patch.object(live.runtime,'directory',return_value=folder), \
                 patch.object(outlook,'collect',return_value={'a.plan.0':dict(status='KNOWN',text='观察',value=3)}), \
                 patch.object(outlook,'live_index_facts',return_value={}),patch('deepseek_api.complete') as api:
                value=outlook.generate({},dict(observe_live=True,plan_hash='p',items=[]),self.cutoff,'INTRADAY')
            api.assert_not_called()
            self.assertEqual(value['status'],'DATA_WAIT')
            self.assertFalse((folder/'risk_outlook'/f'{value["key"]}.json').exists())
            self.assertTrue((folder/'risk_outlook/latest_intraday.json').exists())

    def test_wait_then_evidence_uses_only_one_paid_attempt_and_keeps_failed_slot(self):
        original=json.loads((STAGE/'original_20261026110000.json').read_bytes())
        facts=original['computed_facts'];response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=original['raw_reply']))])
        with tempfile.TemporaryDirectory() as directory,patch.object(live.runtime,'directory',return_value=Path(directory)), \
             patch.object(outlook,'collect',return_value=facts),patch.object(live,'fresh',return_value=True), \
             patch('deepseek_api.complete',return_value=response) as api:
            view=dict(observe_live=True,plan_hash='p',items=[])
            with patch.object(outlook,'live_index_facts',return_value={}):
                self.assertEqual(outlook.generate({},view,self.cutoff,'INTRADAY')['status'],'DATA_WAIT')
            # A legitimate current stock fact unlocks this previously unpaid slot.
            facts['002878.stock.price']=dict(status='KNOWN',value=13.,text='已知当前价')
            with patch.object(outlook,'live_index_facts',return_value={}):
                first=outlook.generate({},view,self.cutoff,'INTRADAY')
                second=outlook.generate({},view,self.cutoff,'INTRADAY')
            self.assertEqual(first,second)
            self.assertEqual(first['status'],'AI_FAILED')  # Old reply calls that now-known price UNKNOWN.
            self.assertEqual(api.call_count,1)

    def test_prewarming_can_queue_market_facts_without_current_stock_risk(self):
        plan=dict(assistant_plan_id='p',monitor_date=20261026,rows=[dict(code='688004')])
        previous=dict(cutoff=20261026103000,risk=dict(records={'688004':dict(status='AVAILABLE')}))
        with tempfile.TemporaryDirectory() as directory,patch.object(live.runtime,'directory',return_value=Path(directory)), \
             patch('services.wl_load',return_value={'positions':[]}),patch.object(events,'evaluate',return_value={}), \
             patch.object(events,'read_risk',return_value=previous),patch.object(live,'index_prefix_ready',return_value=True), \
             patch.object(live,'_thread',SimpleNamespace(is_alive=lambda:True)),patch.object(outlook,'queue') as queue:
            value=live.evaluate(plan,{},self.now)
        self.assertEqual(value['status'],'PREWARMING');queue.assert_called_once()
        self.assertTrue(queue.call_args.kwargs['overlay'])
        self.assertTrue(all(not item['facts'] for item in queue.call_args.args[1]['items']))

    def test_missing_index_prefix_waits_without_queuing_analysis(self):
        plan=dict(assistant_plan_id='p',monitor_date=20261026,rows=[dict(code='688004')])
        with tempfile.TemporaryDirectory() as directory,patch.object(live.runtime,'directory',return_value=Path(directory)), \
             patch('services.wl_load',return_value={'positions':[]}),patch.object(events,'evaluate',return_value={}), \
             patch.object(events,'read_risk',return_value={}),patch.object(live,'index_prefix_ready',return_value=False), \
             patch.object(live,'datetime',SimpleNamespace(now=lambda:self.now,strptime=datetime.strptime)), \
             patch.object(outlook,'queue') as queue:
            value=live.evaluate(plan,{},self.now)
        self.assertEqual(value['status'],'WAIT_INDEX_FEED')
        queue.assert_not_called()  # W1: wait for mirror, never consume the slot with missing index facts.

    def test_worker_failure_exposes_bounded_validation_code(self):
        plan=dict(rule_version=live.VERSION)
        response=SimpleNamespace(returncode=1,stdout='',stderr='Traceback\nValueError: CURRENT_RULE_SOURCE_FINGERPRINT_MISMATCH\n')
        with patch('joint_risk_bridge._worker_paths',return_value=(Path('python'),ROOT/'worker.py')), \
             patch.object(golden_monitor.subprocess,'run',return_value=response):
            with self.assertRaisesRegex(ValueError,'GOLDEN_WORKER_FAILED:ValueError: CURRENT_RULE_SOURCE_FINGERPRINT_MISMATCH'):
                golden_monitor.evaluate(plan,self.now,five_minute=True)

    def test_event_failure_retains_specific_reason(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(live.runtime,'directory',return_value=Path(directory)), \
             patch.object(golden_monitor,'evaluate',side_effect=ValueError('GOLDEN_WORKER_FAILED:TEST')):
            plan=dict(assistant_plan_id='p')
            target=Path(directory)/'event.json'
            events.work(plan,{},self.now,{},target)
            self.assertEqual(json.loads(target.read_bytes())['reason'],'GOLDEN_WORKER_FAILED:TEST')

    def test_current_bound_plan_passes_worker_verifier_without_fetch(self):
        file=ROOT/'stargazer_ml/v5_hybrid/green_common_profile/gcp_golden_intraday_v1.py'
        spec=importlib.util.spec_from_file_location('bound_golden_worker_feed_test',file)
        worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
        from tarot_live_v2 import MANIFEST
        current=json.loads(gzip.decompress(MANIFEST.read_bytes()))
        # A user's persisted plan can be expired; current-source verification
        # uses an isolated current contract, never rewrites that business record.
        plan=dict(rows=[dict(entry_rule=dict(policy='GOLDEN_RESONANCE_CROSSDAY_V4'))],
            rule_inputs=[dict(path=p,sha256=h) for p,h in current['frozen_inputs'].items()])
        with patch.object(worker.urllib.request,'urlopen',side_effect=AssertionError('NO_NETWORK')):
            worker.verify_plan_sources(plan)
            bad=json.loads(json.dumps(plan));bad['rule_inputs'][0]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'RULE_SOURCE_FINGERPRINT_MISMATCH'):
                worker.verify_plan_sources(bad)


if __name__=='__main__':unittest.main()
