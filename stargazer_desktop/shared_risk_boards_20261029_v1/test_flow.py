"""Actual monitor delivery and per-stock acceptance paths, with external I/O mocked."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

H=Path(__file__).resolve().parent;A=H.parent;K=A.parent/'stargazer_ml/v5_hybrid/risk_r0_v1'
sys.path[:0]=[str(A),str(K)]
from pentacles_monitor import PriceMonitor
import tarot_monitor_v2 as live
import sector_watch_v1 as watch
import sector_rule_scan as scan
from test_postclose_risk_runtime import PostcloseRuntimeTests


class AcceptanceReuseTests(PostcloseRuntimeTests):
    def test_existing_stock_reused_after_other_candidate_changed(self):
        r=dict(self.snapshot['rows'][0],feature_values={'f':1},selection_model={'sha256':'frozen'},
               entry_rule=dict(policy='GOLDEN_RESONANCE_CROSSDAY_V4',version='V',eligible=True,atr_ref=1,stage='A',close_p=10,model_reference_node=9))
        r['rank']=1
        parent=dict(self.snapshot,rule_version='TAROT_A_SECOND_CASH_V2',inputs=[{'path':'daily','sha256':'same'}],
                    candidate_population_count=2,rule_inputs=[{'path':'frozen_model','sha256':'same'}],rows=[r,dict(r,code='000002')])
        raw=json.dumps(parent).encode();self.root.joinpath('run/input_snapshot.json').write_bytes(raw)
        self.state['snapshot_sha256']=hashlib.sha256(raw).hexdigest();self.write_state()
        current={**parent,'candidate_population_count':3,'rows':[dict(r,rank=2),dict(r,code='000003'),dict(r,code='000004')]}
        pack,reason=live.runtime.find_pack('000001',current)
        self.assertIsNotNone(pack);self.assertIsNone(reason)
        self.assertIsNone(live.runtime.find_pack('000002',current)[0])
        changed={**current,'rows':[dict(r,reference_price=11),dict(r,code='000003')]}
        self.assertIsNone(live.runtime.find_pack('000001',changed)[0])


class BoardDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=H);self.root=Path(self.temp.name)
        self.monitor=PriceMonitor(lambda:self.root);self.monitor.enable(True)
        self.now=datetime(2026,10,29,10,1);self.cutoff=20261029100000
        self.band=dict(lower=10.,upper=10.1,zone_upper=10.2,source_waves=3,shared_peak_groups=3)
        self.row=dict(code='600001',name='合成板块股',reference_price=12.,
            entry_rule=dict(policy='GOLDEN_RESONANCE_CROSSDAY_V4',eligible=True,p_date=20261028,levels=[10.],observation_bands=[self.band]))
        self.plan=dict(assistant_plan_id='confirmed',rule_version='TAROT_A_SECOND_CASH_V2',plan_ready=True,
            data_date=20261028,monitor_date=20261029,rows=[],boards=[dict(name='pcb')])
        (self.root/'assistant_plan.json').write_text(json.dumps(self.plan))
        self.bytes=(self.root/'assistant_plan.json').read_bytes()
        self.scan=dict(status='READY',board_requested='pcb',board='PCB概念',board_code='B',source_key='fixed',data_date=20261028,
            observation_date=20261029,monitor_rows=[self.row],monitor_rule_inputs=[dict(path='fixture',sha256='fixed')])
        self.broadcast=Mock();self.send=Mock(return_value=(True,'accepted'))

    def tearDown(self):
        watch.AUTHORIZED.clear();self.temp.cleanup()

    def tick(self,result,when=None):
        quotes={'600001':dict(price=10.3,low=10.,prev=12.,time=(when or self.now).strftime('%Y%m%d%H%M%S'))}
        with patch.object(scan,'monitor_state',return_value=[self.scan]),patch.object(live,'evaluate',return_value=result),patch('risk_outlook_v1.latest',return_value={}),patch('risk_outlook_v1.with_local_failure',side_effect=lambda x:x):
            self.monitor._tick_tarot_v2(self.plan,dict(push=dict(enabled=True)),when or self.now,self.broadcast,self.send,None,quotes=lambda codes:quotes)
        self.assertEqual((self.root/'assistant_plan.json').read_bytes(),self.bytes)

    def test_touch_then_later_signal_defer_both_delivered_once(self):
        waiting=dict(status='MONITORING',cutoff=self.cutoff,rows={'600001':dict(status='WAIT_SIGNAL',pending_conditions=['动能未确认'])})
        self.tick(waiting);self.tick(waiting)
        self.assertEqual(self.send.call_count,1)
        self.assertIn('动能未确认',self.send.call_args.args[2])
        future=datetime(2026,10,29,10,6)
        advice=dict(status='ADVICE_READY',decision='DEFER',code='600001',cutoff=20261029100500,reason='板块偏弱',risk_reasons=['板块偏弱'])
        row=dict(status='RULE_TRIGGERED',combination_policy='POSITION_TWO_OF_THREE_O20_CROSS1_V1',
                 combination=dict(votes=dict(repair=True,momentum=True,oversold=False)),advice=advice)
        self.tick(dict(status='MONITORING',cutoff=20261029100500,rows={'600001':row}),future)
        self.assertEqual(self.send.call_count,2)
        self.assertIn('三选二已通过',self.send.call_args.args[2]);self.assertIn('板块偏弱',self.send.call_args.args[2])
        self.assertIn('板块发现',self.send.call_args.args[1])

    def test_signal_with_missing_risk_still_delivered(self):
        row=dict(status='RULE_TRIGGERED',combination_policy='POSITION_TWO_OF_THREE_O20_CROSS1_V1',
                 combination=dict(votes=dict(repair=True,momentum=False,oversold=True)),advice=dict(status='WAIT_RISK'))
        self.tick(dict(status='MONITORING',cutoff=self.cutoff,rows={'600001':row}))
        self.assertEqual(self.send.call_count,1);self.assertIn('三选二已通过',self.send.call_args.args[2])


if __name__=='__main__':unittest.main()
