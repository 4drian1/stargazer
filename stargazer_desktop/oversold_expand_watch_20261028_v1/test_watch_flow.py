"""Quotes cannot buy; touch dedup cannot swallow a later confirmed decision."""
from datetime import datetime
import json
from pathlib import Path
import sys
import unittest

H=Path(__file__).resolve().parent
R=H.parents[1]
sys.path[:0]=[str(R/'stargazer_desktop'),str(R/'stargazer_ml/v5_hybrid/green_common_profile')]
from golden_watch_groups import watch_groups
from test_observation_push_v1 import ObservationPushTests


class GroupTests(unittest.TestCase):
    def rule(self):
        return dict(close_p=26.72,atr_ref=1.3585714285714283,observation_map=dict(bands=[
            dict(lower=lo,upper=hi,source_waves=n,shared_peak_groups=n,observable=ok)
            for lo,hi,n,ok in [(26.3406,26.5216,3,False),(26.1013,26.1013,1,False),
                (25.85,26.0248,3,False),(25.555,25.555,1,False),
                (25.234,25.2684,2,False),(25.0087,25.0087,1,False),
                (24.8001,24.8001,1,False),(24.2308,24.3763,3,True)]]))

    def test_real_levels_fill_middle_without_renaming_sources(self):
        r=self.rule();before=json.dumps(r,sort_keys=True);v=watch_groups(r)
        self.assertEqual([b['lower'] for b in v],[26.3406,25.234,24.2308])
        self.assertEqual([b['source_waves'] for b in v],[3,2,3])
        self.assertFalse(v[1]['execution_qualified'])
        self.assertTrue(v[-1]['execution_qualified'])
        self.assertEqual(before,json.dumps(r,sort_keys=True))

    def test_atr_extensions_never_exceed_frozen_close(self):
        r=self.rule();r['close_p']=26.55
        self.assertTrue(all(b['zone_upper']<=26.55 for b in watch_groups(r)))

    def test_sparse_and_deterministic_under_source_order(self):
        r=self.rule();a=watch_groups(r);r['observation_map']['bands'].reverse();b=watch_groups(r)
        self.assertEqual(a,b)
        self.assertTrue(all(a[i]['lower']-a[i+1]['lower']>=.5*r['atr_ref'] for i in range(len(a)-1)))

    def test_no_fabricated_integer_level_or_weak_buy_gate(self):
        r=self.rule();self.assertNotIn(25.,[b['lower'] for b in watch_groups(r)])
        self.assertEqual(len([b for b in r['observation_map']['bands'] if b['observable']]),1)

    def test_no_map_retains_old_explicit_bands(self):
        b=dict(lower=9.,upper=9.,zone_upper=9.2,source_waves=3)
        self.assertEqual(watch_groups(dict(observation_bands=[b])),[b])


class ProgressTests(ObservationPushTests):
    def test_evidence_exception_cannot_hide_quote_touch(self):
        self.runtime.stop()
        from unittest.mock import patch
        import tarot_monitor_v2 as live
        with patch.object(live,'evaluate',side_effect=ValueError('test')):self.tick()
        self.assertEqual(self.send.call_count,1)
        self.assertIn('建议观察并等待',self.send.call_args.args[2])

    def test_same_tick_buy_does_not_send_contradictory_wait_message(self):
        self.signal(dict(status='ADVICE_READY',decision='BUY',code='688004',cutoff=self.cutoff,reason='有效综合意见'))
        self.tick();self.tick()
        self.assertEqual(self.send.call_count,1)
        self.assertEqual(self.send.call_args.args[2],'688004 建议买入。')

    def test_touch_wait_then_signal_buy_and_restart_once(self):
        self.value=dict(status='MONITORING',event_cutoff=self.cutoff,rows={'688004':dict(
            status='WAIT_SIGNAL',pending_conditions=['触位后两次RSI上穿尚未完成','跨日底背离尚未成立'])})
        self.tick();body=self.send.call_args.args[2]
        self.assertIn('跨日底背离尚未成立',body)
        self.assertIn('建议观察并等待',body)
        self.signal(dict(status='ADVICE_READY',decision='BUY',code='688004',cutoff=self.cutoff,reason='完整详细依据保留本地'))
        self.tick();self.tick()
        self.assertEqual(self.send.call_count,2)
        self.assertEqual(self.send.call_args.args[2],'688004 建议买入。')
        from pentacles_monitor import PriceMonitor
        self.monitor=PriceMonitor(lambda:self.root);self.tick()
        self.assertEqual(self.send.call_count,2)
        alerts=self.monitor._load()['alerts']
        self.assertTrue(any(a.get('review',{}).get('reason')=='完整详细依据保留本地' for a in alerts.values()))

    def test_advice_buy_without_rule_confirmation_is_never_sent(self):
        self.quote.update(low=36.,price=36.)
        self.value=dict(status='MONITORING',event_cutoff=self.cutoff,rows={'688004':dict(status='WAIT_SIGNAL',
            advice=dict(status='ADVICE_READY',decision='BUY',code='688004',cutoff=self.cutoff,reason='不能越过规则'))})
        self.tick();self.send.assert_not_called()

    def test_touched_middle_weak_band_observes_without_buy(self):
        r=GroupTests().rule();r.update(policy='GOLDEN_RESONANCE_CROSSDAY_V4',levels=[24.2308],observation_bands=[])
        self.plan['rows'][0].update(reference_price=26.72,entry_rule=r)
        (self.root/'assistant_plan.json').write_text(json.dumps(self.plan))
        self.quote.update(prev=26.72,price=26.66,low=25.18)
        self.tick()
        self.assertEqual(self.send.call_count,1)
        self.assertIn('25.23',self.send.call_args.args[2])
        self.assertNotIn('建议买入',self.send.call_args.args[2])
        self.assertEqual(self.plan['rows'][0]['entry_rule']['levels'],[24.2308])

    def test_new_complete_bar_can_push_buy_after_prior_defer(self):
        self.tick()
        self.signal(dict(status='ADVICE_READY',decision='DEFER',code='688004',cutoff=self.cutoff,reason='风险尚未通过'))
        self.tick()
        self.now=datetime(2026,10,23,13,6)
        # Use an actual current completed cutoff, rather than re-notifying a stale morning result.
        self.cutoff=20261023130500
        self.quote.update(time='20261023130600')
        self.signal(dict(status='ADVICE_READY',decision='BUY',code='688004',cutoff=self.cutoff,reason='新证据已通过'))
        self.tick()
        self.assertEqual(self.send.call_count,3)
        self.assertEqual(self.send.call_args.args[2],'688004 建议买入。')

if __name__=='__main__':unittest.main()
