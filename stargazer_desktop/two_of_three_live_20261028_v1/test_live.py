from datetime import datetime
import copy,json
from pathlib import Path
import sys,unittest
from unittest.mock import patch
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1];G=R/'stargazer_ml/v5_hybrid/green_common_profile'
sys.path[:0]=[str(R/'stargazer_desktop'),str(G)]
import golden_resonance_v4 as rule
from golden_ladder_v3 import STOPS
from golden_monitor import validate_combination
from test_observation_push_v1 import ObservationPushTests
from trade_facts_v1 import fact_table,CONTRACT

class KernelTests(unittest.TestCase):
    def run_case(self,fast,slow,close=None):
        self.day=20261028;self.previous=20261027
        self.band=dict(lower=12.,upper=12.1,zone_upper=12.2,source_waves=3,shared_peak_groups=3)
        self.row=dict(code='600121',reference_price=20.,entry_rule=dict(p_date=self.previous,levels=[12.],observation_bands=[self.band],atr_ref=1.))
        self.history=[dict(day=self.previous,t=t,open=21.,high=23.,low=20. if i==0 else 21.,close=22.,volume=100.) for i,t in enumerate(STOPS)]
        close=close or [12.3,12.4,12.5,12.6][:len(fast)]
        self.five=[dict(day=self.day,t=STOPS[i],open=c,high=max(c,12.5),low=11. if i==0 else 11.2,close=c,volume=10.) for i,c in enumerate(close)]
        a=np.array([40.]*48+fast);b=np.array([45.]*48+slow)
        with patch.object(rule,'_rsi',side_effect=lambda _,n:a if n==6 else b):
            out=rule.evaluate_bars(self.row,self.day,self.day*1000000+STOPS[len(fast)-1],self.five,self.history)
        if out['status']=='RULE_TRIGGERED':validate_combination(out,self.row,out['price_asof'])
        return out
    def test_one_actual_cross_with_divergence_and_repair_passes(self):
        r=self.run_case([50.,60.],[55.,55.])
        self.assertEqual(r['status'],'RULE_TRIGGERED');self.assertEqual(len(r['event_cross_times']),1)
        self.assertEqual(r['combination']['votes'],dict(repair=True,momentum=True,oversold=False))
    def test_r_plus_o_does_not_require_momentum_or_current_rsi_above(self):
        r=self.run_case([5.,6.,7.],[20.,20.,20.])
        self.assertEqual(r['status'],'RULE_TRIGGERED');self.assertFalse(r['rsi']['above'])
        self.assertEqual(r['combination']['votes'],dict(repair=True,momentum=False,oversold=True))
    def test_oversold_alone_is_not_buy(self):
        r=self.run_case([5.,6.],[20.,20.],[11.5,11.4])
        self.assertEqual(r['status'],'WAIT_SIGNAL');self.assertEqual(r['combination']['count'],1)
    def test_boundary_twenty_percent_is_strict(self):
        r=self.run_case([8.,9.],[20.,20.],[11.5,11.4])
        self.assertIsNone(r['oversold_evidence']);self.assertFalse(r['combination']['votes']['oversold'])
    def test_staying_above_is_not_a_new_cross(self):
        r=self.run_case([50.,60.],[30.,30.])
        # Previous40<=45 -> first bar upcross is outside the allowed event index1.
        self.assertNotEqual(r['status'],'RULE_TRIGGERED')
    def test_m_plus_o_can_pass_without_position_repair(self):
        a=np.array([40.]*48+[5.,50.,60.]);b=np.array([45.]*48+[20.,55.,55.])
        self.run_case([5.,50.,60.],[20.,55.,55.],[11.5,10.2,10.8])
        self.five[1]['low']=10.;self.five[2]['low']=10.1
        with patch.object(rule,'_rsi',side_effect=lambda _,n:a if n==6 else b):
            r=rule.evaluate_bars(self.row,self.day,20261028094500,self.five,self.history)
        self.assertEqual(r['status'],'RULE_TRIGGERED');self.assertEqual(r['branch'],'MOMENTUM_OVERSOLD')
        validate_combination(r,self.row,20261028094500)
    def test_single_remaining_vote_invalidates_pass(self):
        r=self.run_case([50.,60.,10.],[55.,55.,30.],[12.3,12.4,11.5])
        self.assertEqual(r['status'],'WAIT_SIGNAL')
    def test_no_position_arrival_cannot_pass(self):
        self.run_case([5.,6.],[20.,20.],[13.,13.1])
        for b in self.five:b['low']=13.;b['high']=14.
        with patch.object(rule,'_rsi',side_effect=lambda _,n:np.array([40.]*48+[5.,6.]) if n==6 else np.array([45.]*48+[20.,20.])):
            r=rule.evaluate_bars(self.row,self.day,20261028094000,self.five,self.history)
        self.assertNotEqual(r['status'],'RULE_TRIGGERED')
    def test_forged_oversold_ratio_and_vote_rejected(self):
        r=self.run_case([5.,6.],[20.,20.]);r['oversold_evidence']['rsi6']=10.
        with self.assertRaises(ValueError):validate_combination(r,self.row,r['price_asof'])
    def test_missing_third_vote_never_becomes_global_hard_gate(self):
        r=self.run_case([5.,6.],[20.,20.]);r['divergence']['event_rsi']=None;r['combination']['votes']['momentum']=None
        validate_combination(r,self.row,r['price_asof'])
    def test_facts_use_false_divergence_not_nonempty_dictionary(self):
        r=self.run_case([5.,6.],[20.,20.]);f=fact_table(r['price_asof'],{},r)
        self.assertFalse(f['signal.divergence']['value']);self.assertFalse(f['signal.vote.momentum']['value'])
        self.assertTrue(f['signal.combination']['value']);self.assertIn('V5',CONTRACT)
    def test_old_two_cross_trigger_cannot_bypass_new_contract(self):
        r=self.run_case([50.,60.],[55.,55.]);r.pop('combination_policy')
        with self.assertRaises(ValueError):validate_combination(r,self.row,r['price_asof'])

class DeliveryTests(ObservationPushTests):
    def new_signal(self,advice):
        self.signal(advice)
        self.value['rows']['688004'].update(combination_policy=rule.COMBINATION_POLICY,
            combination=dict(votes=dict(repair=True,momentum=False,oversold=True),count=2,required=2))
    def test_touch_failed_then_pass_defer_then_later_buy_and_restart(self):
        self.value=dict(status='MONITORING',event_cutoff=self.cutoff,rows={'688004':dict(status='WAIT_SIGNAL',pending_conditions=['动能未满足','严重超卖未满足'])})
        self.tick();self.assertIn('未满足',self.send.call_args.args[2])
        self.new_signal(dict(status='ADVICE_READY',decision='DEFER',code='688004',cutoff=self.cutoff,summary_reason='板块转弱',risk_reasons=['板块转弱']))
        self.tick();self.tick();self.assertEqual(self.send.call_count,2)
        self.assertIn('三选二已通过',self.send.call_args.args[2]);self.assertIn('板块转弱',self.send.call_args.args[2]);self.assertNotIn('尚未通过',self.send.call_args.args[2])
        self.new_signal(dict(status='ADVICE_READY',decision='BUY',code='688004',cutoff=self.cutoff,reason='新证据支持'))
        self.tick();self.assertEqual(self.send.call_count,3);self.assertIn('建议买入',self.send.call_args.args[2])
        from pentacles_monitor import PriceMonitor
        self.monitor=PriceMonitor(lambda:self.root);self.tick();self.assertEqual(self.send.call_count,3)
    def test_pass_is_sent_before_risk_or_ai_finishes(self):
        self.new_signal(dict(status='WAIT_RISK'));self.tick();self.tick()
        self.assertEqual(self.send.call_count,1);self.assertIn('三选二已通过',self.send.call_args.args[2])
        self.new_signal(dict(status='AI_ATTEMPT_RESERVED'));self.tick();self.assertEqual(self.send.call_count,1)
        self.new_signal(dict(status='AI_FAILED',reason='本次接口失败'));self.tick();self.assertEqual(self.send.call_count,2)
        self.assertIn('三选二已通过',self.send.call_args.args[2]);self.assertIn('失败',self.send.call_args.args[2])
    def test_same_tick_pass_defer_coalesces_quote_without_losing_touch_dedup(self):
        self.new_signal(dict(status='ADVICE_READY',decision='DEFER',code='688004',cutoff=self.cutoff,summary_reason='环境偏弱'))
        self.tick();self.tick();self.assertEqual(self.send.call_count,1)
        self.assertIn('三选二已通过',self.send.call_args.args[2])
        self.value=dict(status='WAIT_INDEX_FEED',rows={});self.tick();self.assertEqual(self.send.call_count,1)
    def test_quote_still_pushes_if_complete_signal_evidence_unknown(self):
        self.value=dict(status='MONITORING',event_cutoff=self.cutoff,rows={'688004':dict(status='DATA_INSUFFICIENT')})
        self.tick();self.assertEqual(self.send.call_count,1);self.assertNotIn('三选二已通过',self.send.call_args.args[2])
if __name__=='__main__':unittest.main()
