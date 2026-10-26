"""Price-path regressions: missing bars, retained events, new lows and factual output."""
import copy,sys,unittest
from pathlib import Path
import pandas as pd
import golden_ladder_v3 as g
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'stargazer_desktop'))
from trade_facts_v1 import fact_table,validate_and_render
import golden_monitor
from golden_rules import digest


def minutes(bars,opening=False):
    rows=[]
    for bar,stop in zip(bars,g.STOPS):
        end=stop//10000*60+stop%10000//100
        for m in range(end-4,end+1):
            rows.append(dict(code='688004',date=20261023000000+m//60*10000+m%60*100,
                open=bar[0],high=bar[1],low=bar[2],close=bar[3],volume=bar[4]/5))
    if opening:rows.insert(0,dict(rows[0],date=20261023093000))
    return rows


class Tests(unittest.TestCase):
    def setUp(self):
        self.row=dict(code='688004',reference_price=11,entry_rule=dict(policy=g.POLICY,levels=[10.,9.],atr_ref=1.))
        self.bars=[(10.2,10.3,10.1,10.15,1000),(10.15,10.25,10.12,10.18,500),(10.18,10.3,10.13,10.22,400)]

    def test_optional_open_and_240_endpoints(self):
        for opening in [False,True]:
            five,_,_,_=g.completed_bars(minutes(self.bars,opening),'688004',20261023,20261023094500)
            self.assertEqual(len(five),3)

    def test_missing_internal_bar_is_not_filled(self):
        source=minutes(self.bars);source=[r for r in source if r['date']!=20261023093200]
        self.assertEqual(g.evaluate(self.row,20261023,20261023094500,source)['status'],'DATA_INSUFFICIENT')

    def test_stop_without_break_or_rsi_is_allowed(self):
        value=g.evaluate(self.row,20261023,20261023094500,minutes(self.bars))
        self.assertEqual((value['status'],value['branch']),('RULE_TRIGGERED','STOP_ABOVE'))
        self.assertIsNone(value['rsi']['rsi6'])
        self.assertIsNone(value['divergence'])

    def test_no_volume_not_proven_contracting(self):
        source=minutes(self.bars)
        for r in source:r.pop('volume')
        self.assertEqual(g.evaluate(self.row,20261023,20261023094500,source)['status'],'WAIT_SIGNAL')

    def test_new_low_invalidates_old_stop(self):
        source=minutes(self.bars+[(10.1,10.2,9.7,9.8,1000)])
        self.assertEqual(g.evaluate(self.row,20261023,20261023095000,source)['status'],'WAIT_SIGNAL')

    def test_break_then_two_closes_reclaim(self):
        bars=[(10.2,10.3,9.8,9.9,1000),(9.9,10.2,9.9,10.05,500),(10.05,10.3,10.01,10.15,400)]
        value=g.evaluate(self.row,20261023,20261023094500,minutes(bars))
        self.assertEqual(value['branch'],'RECLAIM')

    def test_break_moves_observation_to_next_level(self):
        bars=[(10.2,10.3,9.1,9.15,1000),(9.15,9.3,9.12,9.18,500),(9.18,9.4,9.13,9.22,400)]
        value=g.evaluate(self.row,20261023,20261023094500,minutes(bars))
        self.assertEqual((value['node'],value['branch']),(9.,'STOP_ABOVE'))
        self.assertIn(10.,value['pierced_levels'])

    def test_future_bars_do_not_change_prefix(self):
        whole=minutes(self.bars+[(10.1,10.2,9.7,9.8,1000)])
        self.assertEqual(g.evaluate(self.row,20261023,20261023094500,whole),g.evaluate(self.row,20261023,20261023094500,minutes(self.bars)))

    def test_event_remains_for_later_same_cutoff_risk(self):
        source=minutes(self.bars+[(10.22,10.3,10.14,10.2,300)])
        value=g.evaluate(self.row,20261023,20261023095000,source)
        self.assertEqual(value['trigger_time'],20261023094500)
        plan=dict(rule_version='TAROT_A_SECOND_CASH_V2',rows=[self.row])
        value['rule_hash']=digest(self.row['entry_rule'])
        golden_monitor.validate_result(dict(cutoff=20261023095000,rows={'688004':value}),plan)

    def test_price_map_is_causal_and_below_close(self):
        daily=pd.DataFrame([dict(date=20261001+i,open=10,low=9+i*.1,high=12+i*.1,close=11+i*.1) for i in range(8)])
        daily.loc[5,['open','low','high','close']]=[10.,9.5,13.,12.]
        daily.loc[[3,4,6,7],'low']=10.
        value=g.price_map(daily)
        self.assertTrue(all(0<n<=daily.close.iloc[-1] for n in value['levels']))
        self.assertTrue(all(w['known_at']<=daily.date.iloc[-1] for w in value['waves']))

    def test_facts_do_not_reverse_price_or_rsi(self):
        signal=dict(node=33.4503,session_low=33.56,rsi=dict(rsi6=40.55,rsi15=36.29,above=True,cross_up_times=[112000]))
        facts=fact_table(20261023113000,dict(context={}),signal)
        advice=dict(asof=20261023113000,decision='DEFER',supporting_ids=['signal.node_relation'],opposing_ids=[],unknown_ids=[],reason='最低价已跌破黄金位')
        text=validate_and_render(advice,facts,20261023113000)
        self.assertIn('高于',text);self.assertNotIn('已跌破',text)

    def test_unknown_ids_or_wrong_time_are_rejected(self):
        facts=fact_table(20261023113000,dict(context={}))
        advice=dict(asof=20261023113000,decision='BUY',supporting_ids=['invented'],opposing_ids=[],unknown_ids=[])
        with self.assertRaises(ValueError):validate_and_render(advice,facts,20261023113000)
        advice.update(asof=20261023114500,supporting_ids=['stock.price'])
        with self.assertRaises(ValueError):validate_and_render(advice,facts,20261023113000)

    def test_extension_formula_matches_manual_3584(self):
        levels=g.wave_levels(3741.11,3995.18)
        extension=next(x for x in levels if x['kind']=='DOWNSIDE_EXTENSION_FROM_HIGH' and x['ratio']==1.618)
        self.assertAlmostEqual(extension['price'],3584.09474,places=5)
        self.assertAlmostEqual(extension['price'],3741.11-.618*(3995.18-3741.11),places=5)

    def test_retracement_and_rebound_are_named_separately(self):
        levels=g.wave_levels(31.,42.45)
        down=next(x for x in levels if x['kind']=='RETRACEMENT' and x['ratio']==.786)
        up=next(x for x in levels if x['kind']=='REBOUND' and x['ratio']==.786)
        self.assertAlmostEqual(down['price'],33.4503)
        self.assertAlmostEqual(up['price'],39.9997)

    def test_index_anchors_are_selected_without_manual_target(self):
        dates=[int(x) for x in pd.bdate_range('2026-07-01',periods=70).strftime('%Y%m%d')]
        rows=[]
        for i,day in enumerate(dates[:-1]):
            for stop in g.STOPS:
                rows.append(dict(date=day*1000000+stop,open=3850.,high=3995.18 if i==20 else 3980.,
                    low=3741.11 if i==10 else 3800.,close=3900.))
        rows.append(dict(date=dates[-1]*1000000+93500,open=3590.,high=3615.,low=3574.,close=3610.))
        result=g.index_map(pd.DataFrame(rows),dates[-1]*1000000+93500)
        self.assertEqual((result['low_day'],result['high_day']),(dates[10],dates[20]))
        extension=next(x for x in result['levels'] if x['kind']=='DOWNSIDE_EXTENSION_FROM_HIGH' and x['ratio']==1.618)
        self.assertEqual(extension['below_then_close_back_times'],[dates[-1]*1000000+93500])


if __name__=='__main__':unittest.main()
