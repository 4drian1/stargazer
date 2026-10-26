"""Real frozen feature recipe and hard gates remain intact for human supplements."""
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import unittest

GCP=Path(__file__).resolve().parent.parent/'stargazer_ml/v5_hybrid/green_common_profile'
sys.path.insert(0,str(GCP))
import tarot_live_v2 as live
from tarot_manual_review_v1 import latest_record
import risk_outlook_v1 as outlook
import pandas as pd


class ManualDisplayTests(unittest.TestCase):
    def setUp(self):
        self.core=live.frozen()
        source=GCP/'tarot_a_second_leg_fit_20261022_v1/day_replay_20261022_23_v1/golden_flow_fix_20261023_v1/selector_binding_replay_20261025/daily/part_bound.parquet'
        frame=pd.read_parquet(source)
        self.g=frame[frame.code=='300225'].sort_values('date').reset_index(drop=True)
        self.calendar=live.trading_calendar()

    def test_latest_anchor_filter_preserves_frozen_signals(self):
        for length in range(len(self.g)-7,len(self.g)+1):
            g=self.g.iloc[:length].copy()
            expected={k:v for k,v in self.core.second_candidates(g,max_p=int(g.date.iloc[-1])).items() if k==len(g)-1}
            self.assertEqual(live.latest_second(g,self.core),expected)

    def test_prefix_reader_keeps_cross_part_history_and_rejects_conflicts(self):
        g=self.g.copy();other=g.copy();other['code']='600593'
        base=pd.concat([g.iloc[:-2],other.iloc[:-2]],ignore_index=True)
        incr=pd.concat([g.iloc[-3:],other.iloc[-2:]],ignore_index=True)
        fields=list(g)
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'part_a.parquet';b=Path(folder)/'part_b.parquet'
            base.to_parquet(a,index=False);incr.to_parquet(b,index=False)
            actual=dict(live.daily_groups([a,b],fields,int(g.date.iloc[-1])))
            for code,expected in [('300225',g),('600593',other)]:
                pd.testing.assert_frame_equal(actual[code].reset_index(drop=True),expected.reset_index(drop=True))
            incr.loc[0,'close']+=.01;incr.to_parquet(b,index=False)
            with self.assertRaisesRegex(ValueError,'CONFLICTING_DAILY_SOURCE'):
                list(live.daily_groups([a,b],fields,int(g.date.iloc[-1])))

    def test_early_outer_and_down_gate_are_existing_necessities(self):
        for kind in ('UP','ST','ILLIQUID'):
            g=self.near_miss()
            if kind=='UP':
                g.loc[len(g)-1,'close']=g.pre_close.iloc[-1]*1.01
                g.loc[len(g)-1,'high']=max(g.high.iloc[-1],g.close.iloc[-1])
            if kind=='ST':g.loc[len(g)-1,'is_st']=True
            if kind=='ILLIQUID':g.loc[len(g)-1,'amount']=1
            self.assertIsNone(live.latest_record(g,self.core))
            self.assertIsNone(latest_record(g,self.core,[],self.calendar))

    def near_miss(self):
        g=self.g.copy();last=len(g)-1
        signal=self.core.second_candidates(g,max_p=int(g.date.iloc[-1]))[last]
        t=signal['anchor'];g.loc[t+2,'volume']=g.volume.iloc[t+1]*1.01
        return g

    def test_volume_only_near_miss_uses_exact_frozen_feature_recipe(self):
        g=self.near_miss()
        self.assertIsNone(live.latest_record(g,self.core))
        rec=latest_record(g,self.core,[],self.calendar)
        self.assertIn('FIRST_LEG_VOLUME_NOT_DECREASING',rec['failed_gates'])
        self.assertTrue(rec['position_ready']);self.assertEqual(rec['stage'],'SECOND')
        expected=self.core.indicators(g)
        for k in self.core.FEATURES:
            if k in expected:self.assertEqual(rec[k],float(expected[k].iloc[-1]),k)

    def test_st_and_triple_top_still_rejected(self):
        g=self.near_miss();g.loc[len(g)-1,'is_st']=True
        self.assertIsNone(latest_record(g,self.core,[],self.calendar))
        g=self.near_miss()
        with patch.object(self.core,'top_structure',return_value=[1,2,3]):
            self.assertIsNone(latest_record(g,self.core,[],self.calendar))

    def test_missing_trading_day_never_fills_manual_list(self):
        g=self.near_miss().drop(index=len(self.g)-4).reset_index(drop=True)
        self.assertIsNone(latest_record(g,self.core,[],self.calendar))

    def test_known_zero_stays_known_and_protocol_version_changes_cache(self):
        facts={'index.sse.crosses':dict(status='KNOWN',value=0,text='上穿次数0')}
        reply=dict(asof=20261023150000,bias='MIXED',sections={k:'等待配合' for k in ['market','boards','leaders','holdings','plan']},
            focus_ids=[],supporting_ids=[],opposing_ids=[],unknown_ids=['index.sse.crosses'],danger_signals=[],repair_signals=[])
        with self.assertRaisesRegex(ValueError,'OUTLOOK_KNOWN_AS_UNKNOWN'):outlook.validate(reply,facts,reply['asof'])
        self.assertIn('次数为零',outlook.PROMPT)
        view=dict(data_date=20261023,monitor_date=20261026,items=[])
        old=outlook.view_id(view)
        with patch.object(outlook,'CONTRACT','different'):
            self.assertNotEqual(old,outlook.view_id(view))


if __name__=='__main__':unittest.main()
