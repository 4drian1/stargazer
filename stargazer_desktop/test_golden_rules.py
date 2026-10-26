import copy
import io
import hashlib
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import msgpack

import golden_rules as rules
import golden_monitor
import assistant_workflow
from pentacles_monitor import PriceMonitor
import trade_plan_v1

GREEN = Path(__file__).resolve().parent.parent/'stargazer_ml/v5_hybrid/green_common_profile'
sys.path.insert(0,str(GREEN))
sys.path.insert(0,str(GREEN/'gcp_round_b_small_sample_v1/pentacles_golden_v1'))
import gcp_golden_intraday_v1 as worker
import test_trigger_replay_v3 as fixture


class GoldenRulesTests(unittest.TestCase):
    def rule(self):
        return rules.build_rule(7.7,8.,7.,1.,10.,5.)

    def test_band_boundary_and_nonfinite(self):
        self.assertTrue(self.rule()['eligible'])
        self.assertFalse(rules.build_rule(7.70001,8.,7.,1.,10.,5.)['eligible'])
        for value in (float('nan'),float('inf'),0.,-1.,True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rules.build_rule(7.7,8.,7.,value,10.,5.)

    def days(self, closes):
        return {i+1:dict(open=9.9,high=20.,low=5.,close=c) for i,c in enumerate(closes)}

    def test_seven_increases_not_six(self):
        before = [10.]*7
        six = rules.cycle_anchors(self.days(before+[10.1,10.2,10.3,10.4,10.5,10.6]))
        seven = rules.cycle_anchors(self.days(before+[10.1,10.2,10.3,10.4,10.5,10.6,10.7]))
        self.assertEqual(six['low'],5.)
        self.assertEqual(seven['low'],10.)

    def test_small_declines_do_not_shrink_and_large_single_decline_does(self):
        small = rules.cycle_anchors(self.days([10.]*7+[9.95,9.9,9.85]))
        large = rules.cycle_anchors(self.days([10.]*7+[9.85,9.8,9.75]))
        self.assertEqual(small['high'],20.)
        self.assertEqual(large['high'],10.)

    def test_intraday_break_does_not_redraw_close_break_changes_one_end(self):
        daily = self.days([10.]*8)
        daily[8].update(high=99.,low=1.)
        self.assertEqual(rules.cycle_anchors(daily)['high'],20.)
        self.assertEqual(rules.cycle_anchors(daily)['low'],5.)
        daily[8]['close']=21.
        self.assertEqual(rules.cycle_anchors(daily)['high'],21.)
        self.assertEqual(rules.cycle_anchors(daily)['low'],5.)

    def test_projection_is_subset_only_and_retains_exclusion(self):
        parent = dict(data_date=20261021,monitor_date=20261022,inputs=[],rows=[
            dict(code='000001',name='A',setup='B_LVB',date=20261021,reference_price=7.7,rsi14=20.,zone=None),
            dict(code='000002',name='B',setup='B_LVB',date=20261021,reference_price=7.7,rsi14=20.,zone=None)])
        value = rules.project(parent,{'000001':self.rule()})
        self.assertEqual(len(value['excluded_rows']),1)
        self.assertTrue(rules.is_projection_of(value,parent))
        self.assertIsNone(parent['rows'][0]['zone'])
        for key, changed in [('reference_price',9.),('code','999999')]:
            bad=copy.deepcopy(value);bad['rows'][0][key]=changed
            self.assertFalse(rules.is_projection_of(bad,parent))
        bad=copy.deepcopy(value);bad['inputs']=[{}]
        self.assertFalse(rules.is_projection_of(bad,parent))

    def test_next_day_band_is_capped_and_broken_f618_excluded(self):
        parent = dict(data_date=20261020, monitor_date=20261021, inputs=[], rows=[
            dict(code='600792', name='A', reference_price=4.26, zone=None),
            dict(code='002860', name='B', reference_price=17.15, zone=None)])
        capped = self.rule(); capped.update(f618=4.1802, atr_ref=.5)
        broken = self.rule(); broken.update(f618=17.3474)
        value = rules.project(parent, {'600792': capped, '002860': broken})
        self.assertEqual(value['rows'][0]['zone']['zone_lower'], 4.1802)
        self.assertEqual(value['rows'][0]['zone']['zone_upper'], 4.26)
        self.assertEqual(value['excluded_rows'][0]['reason'], 'PREVIOUS_F618_ABOVE_CLOSE')
        self.assertEqual(parent['rows'][0]['zone'], None)

    def test_old_snapshot_above_close_must_be_regenerated(self):
        with self.assertRaisesRegex(ValueError, '重新运行盘后选股'):
            assistant_workflow.require_next_day_zones({'rows': [
                {'reference_price': 4.26, 'zone': {'zone_lower': 4.1802, 'zone_upper': 4.2802}}]})

    def minutes(self):
        bars=fixture.full_day_bars(price=9.95);day=20261013;base=day*1000000
        for i,s in enumerate(fixture.m.FIVE_STOPS[:12]):
            p=([10.05,10.][i%2]);fixture.set_group(bars,base,s,p,p+.05,p-.05,p)
        for t,o,h,lo,c in [(103500,9.95,9.95,9.85,9.9),(104000,9.9,9.9,9.75,9.8),
                (104500,9.8,9.8,9.65,9.7),(105000,9.7,9.7,9.6,9.75),(130500,9.75,10.3,9.7,10.2)]:
            fixture.set_group(bars,base,t,o,max(h,o,c),min(lo,o,c),c)
        return [dict(code='000001',**v) for v in bars.values()]

    def test_prefix_signal_and_future_perturbation(self):
        row=dict(code='000001',reference_price=10.,entry_rule=dict(version=rules.VERSION,eligible=True,f618=10.,atr_ref=1.))
        minutes=self.minutes()
        before=worker.prefix_result(row,20261013,20261013110000,minutes,10.,[])
        self.assertEqual(before['status'],'WAIT_SIGNAL')
        after=worker.prefix_result(row,20261013,20261013130500,minutes,10.,[])
        self.assertEqual(after['status'],'RULE_TRIGGERED')
        for r in minutes:
            if r['date']>=20261013110000:
                r.update(open=100.,high=101.,low=99.,close=100.)
        self.assertEqual(before,worker.prefix_result(row,20261013,20261013110000,minutes,10.,[]))

    def test_scale_action_and_duplicate_conflict_block(self):
        row=dict(code='000001',reference_price=10.,entry_rule=dict(version=rules.VERSION,eligible=True,f618=10.,atr_ref=1.))
        for previous, adj in [(9.,[]),(float('nan'),[]),(10.,[('复权:000001:20261013',{})])]:
            with self.assertRaises(ValueError):
                worker.prefix_result(row,20261013,20261013130500,self.minutes(),previous,adj)
        minute=self.minutes();minute.append(dict(minute[0],close=10.01))
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):
            worker.prefix_result(row,20261013,20261013130500,minute,10.,[])

    def test_rebound_is_not_downside_signal(self):
        row=dict(code='000001',reference_price=10.,entry_rule=dict(version=rules.VERSION,eligible=True,f618=10.,atr_ref=1.))
        minute=self.minutes();minute[0].update(open=9.99,low=9.98)
        self.assertEqual(worker.prefix_result(row,20261013,20261013130500,minute,10.,[])['status'],'NON_DOWNSIDE')

    def test_quote_touch_cannot_trigger_new_rule(self):
        plan=dict(assistant_plan_id='test', stocks=[dict(code='000001',name='A',buy_zone=[7.,8.],
            valid_until='20261022150000',entry_rule=self.rule())])
        with patch.object(golden_monitor,'status_for',return_value=dict(status='WAIT_MINUTE_EVIDENCE')):
            self.assertEqual(trade_plan_v1.plan_check('000001',7.5,'20261022100000',plan)['status'],'WAIT_MINUTE_EVIDENCE')

    def test_unsupported_trigger_flags_rejected(self):
        rule=self.rule();plan={'rows':[dict(code='000001',entry_rule=rule)]}
        value=dict(cutoff=20261022100000,rows={'000001':dict(status='RULE_TRIGGERED',trigger_time=20261022095500,
            rule_hash=rules.digest(rule),flags={})})
        with self.assertRaisesRegex(ValueError,'EVIDENCE_INVALID'):
            golden_monitor.validate_result(value,plan)

    def test_monitor_wait_does_not_use_quote_or_send(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor=PriceMonitor(lambda:tmp);monitor.enable(True)
            plan=dict(rule_version=rules.VERSION,data_date=20261021,monitor_date=20261022,plan_ready=True,assistant_plan_id='test',
                rows=[dict(code='000001',name='A',reference_price=7.7,zone=None,entry_rule=self.rule())])
            quote=Mock(side_effect=AssertionError('QUOTE_NOT_A_TRIGGER'));send=Mock();broadcast=Mock()
            with patch.object(golden_monitor,'evaluate',return_value={'rows':{'000001':{'status':'WAIT_SIGNAL'}}}):
                monitor.tick(plan,{},datetime(2026,10,22,10,0),quote,broadcast,send)
            quote.assert_not_called();send.assert_not_called();broadcast.assert_not_called()

    def test_unconfirmed_new_rules_cannot_start_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor=PriceMonitor(lambda:tmp);monitor.enable(True)
            plan=dict(rule_version=rules.VERSION,data_date=20261021,monitor_date=20261022,plan_ready=True,rows=[])
            with patch.object(golden_monitor,'evaluate',side_effect=AssertionError('NO_UNCONFIRMED_FETCH')):
                monitor.tick(plan,{},datetime(2026,10,22,10,0),Mock(),Mock(),Mock())
            self.assertEqual(monitor.state()['status'],'WAIT_HUMAN_CONFIRMATION')

    def test_real_worker_entry_mock_transport_raw_archive_and_no_retry(self):
        row=dict(code='000001',reference_price=10.,entry_rule=dict(version=rules.VERSION,eligible=True,f618=10.,atr_ref=1.))
        source=Path(rules.__file__)
        plan=dict(rule_version=rules.VERSION,data_date=20261012,monitor_date=20261013,rows=[row],
            rule_inputs=[dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())])
        previous=[dict(code='000001',date=20261012145900,open=10.,high=10.,low=10.,close=10.)]
        opener=Mock()
        opener.open.side_effect=[io.BytesIO(msgpack.packb(x,use_bin_type=True)) for x in [previous,self.minutes(),[{'date':20250101,'mult':1.0}]]]
        with tempfile.TemporaryDirectory() as tmp,patch.object(worker.urllib.request,'build_opener',return_value=opener):
            body=dict(plan=plan,cutoff=20261013130500,runtime=tmp)
            value=worker.run(body)
            self.assertEqual(value['rows']['000001']['status'],'RULE_TRIGGERED')
            self.assertEqual(opener.open.call_count,3)
            self.assertEqual(len(value['raw_evidence']),3)
            golden_monitor.validate_result(value,plan)
            self.assertEqual(worker.run(body),value)
            self.assertEqual(opener.open.call_count,3)  # restart/cached cutoff never re-fetches.
        opener=Mock();opener.open.side_effect=TimeoutError()
        with tempfile.TemporaryDirectory() as tmp,patch.object(worker.urllib.request,'build_opener',return_value=opener):
            value=worker.run(dict(plan=plan,cutoff=20261013130500,runtime=tmp))
            self.assertEqual(value['rows']['000001']['status'],'DATA_INSUFFICIENT')
            self.assertEqual(opener.open.call_count,1)


if __name__=='__main__':
    unittest.main()
