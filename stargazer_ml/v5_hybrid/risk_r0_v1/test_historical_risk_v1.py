import math
import unittest
from datetime import date,timedelta
from copy import deepcopy

import numpy as np

from historical_risk_core_v1 import (label_at,stock_features,market_features,segment,sample_gate,
                                    in_scope,STOCK_FIELDS,MARKET_FIELDS)
from prepare_historical_risk_v1 import structural_stock


class HistoricalTests(unittest.TestCase):
    def setUp(self):
        self.cal=[]
        d=date(2024,1,1)
        while len(self.cal)<80:
            if d.weekday()<5: self.cal.append(int(d.strftime('%Y%m%d')))
            d+=timedelta(days=1)
        self.p=self.cal[65]
        self.bars={d:dict(open=100.,high=101.,low=99.,close=100.) for d in self.cal}
        self.f={d:1. for d in self.cal}

    def label(self):
        return label_at(self.bars,self.cal,self.p,self.f,scale_verified=True,coverage_verified=True)

    def test_threshold_equal_and_above(self):
        self.bars[self.cal[66]]['low']=92.
        self.assertEqual(self.label()['y_tail_5d'],1)
        self.bars[self.cal[66]]['low']=92.0001
        self.assertEqual(self.label()['y_tail_5d'],0)

    def test_split_not_false_tail(self):
        for d in self.cal[66:71]:
            self.bars[d]=dict(open=50.,high=50.5,low=49.5,close=50.)
            self.f[d]=2.
        self.assertEqual(self.label()['y_tail_5d'],0)

    def test_pending_evidence_never_negative(self):
        r=label_at(self.bars,self.cal,self.p,self.f)
        self.assertIsNone(r['y_tail_5d'])

    def test_missing_future_no_compression(self):
        del self.bars[self.cal[67]]
        self.assertIsNone(self.label()['mae_5d'])

    def test_invalid_prices_and_factors(self):
        for bad in (0,-1,float('nan'),float('inf'),True):
            with self.subTest(bad=bad):
                self.f[self.cal[67]]=bad
                self.assertIsNone(self.label()['mae_5d'])

    def test_calendar_end(self):
        r=label_at(self.bars,self.cal,self.cal[-3],self.f,scale_verified=True,coverage_verified=True)
        self.assertEqual(r['status'],'CENSORED_TAIL')

    def test_invalid_calendar(self):
        with self.assertRaises(ValueError):
            stock_features(self.bars,self.cal+[self.cal[-1]],self.p,self.f)

    def test_feature_values(self):
        x,status=stock_features(self.bars,self.cal,self.p,self.f)
        self.assertEqual(set(x),set(STOCK_FIELDS))
        self.assertEqual(x['ret_20d'],0)
        self.assertEqual(x['vol_20d'],0)
        self.assertAlmostEqual(x['atr14_pct'],.02)
        self.assertAlmostEqual(x['range_pct_5d'],.02)

    def test_prefix_and_future_perturbation(self):
        baseline=stock_features(self.bars,self.cal,self.p,self.f)
        prefix={d:b for d,b in self.bars.items() if d<=self.p}
        self.assertEqual(baseline,stock_features(prefix,self.cal[:66],self.p,self.f))
        for d in self.cal[66:]:
            self.bars[d]={k:v*5 for k,v in self.bars[d].items()}
            self.f[d]=1000.
        self.assertEqual(baseline,stock_features(self.bars,self.cal,self.p,self.f))

    def test_missing_history_not_forward_filled(self):
        del self.bars[self.cal[20]]
        x,_=stock_features(self.bars,self.cal,self.p,self.f)
        self.assertTrue(all(v is None for v in x.values()))

    def test_boundary_purge_keeps_assignment(self):
        self.assertEqual(segment(20231229,20240105),('train',False))
        self.assertEqual(segment(20231220,20231227),('train',True))
        self.assertEqual(segment(20240628,20240705),('select',False))

    def test_scope(self):
        self.assertTrue(in_scope('688001'))
        self.assertFalse(in_scope('159001'))
        self.assertFalse(in_scope('920001'))

    def test_index_prefix_and_missing(self):
        closes={d:100+i for i,d in enumerate(self.cal)}
        actual=market_features(closes,self.cal,self.p)
        prefix={d:c for d,c in closes.items() if d<=self.p}
        self.assertEqual(actual,market_features(prefix,self.cal[:66],self.p))
        self.assertAlmostEqual(actual[0]['market_ret_20d'],165/145-1)
        del closes[self.cal[55]]
        self.assertTrue(all(v is None for v in market_features(closes,self.cal,self.p)[0].values()))

    def test_config_and_code_segments_match(self):
        import json
        from pathlib import Path
        config=json.loads((Path(__file__).parent/'contracts/historical_stock_day_experiment_v1.json').read_bytes())
        for name,(start,end) in config['segments'].items():
            self.assertEqual(segment(start,start+1)[0],name)
            self.assertEqual(segment(end,end+10000)[0],name)

    def test_gate_missing_labels_do_not_pass(self):
        row=dict(code='000001',P=20230103,label_end_date=20230110,
                 training_eligible=False,y_tail_5d=None)
        self.assertFalse(sample_gate([row])['training_ready'])
        row['training_eligible']=True
        with self.assertRaises(ValueError): sample_gate([row])

    def test_gate_duplicate(self):
        row=dict(code='000001',P=20230103,label_end_date=20230110,
                 training_eligible=True,y_tail_5d=0,
                 features={k:0. for k in STOCK_FIELDS+MARKET_FIELDS})
        with self.assertRaises(ValueError): sample_gate([row,row])

    def test_structural_conflict_blocks(self):
        a=np.array([[d,100,101,99,100] for d in self.cal],dtype=float)
        bad=a[30].copy();bad[-1]=100.5
        dates,valid,past,future,dup,conf=structural_stock([a,bad.reshape(1,-1)],np.array(self.cal))
        self.assertEqual((dup,conf),(1,1))
        self.assertFalse(valid[30]);self.assertFalse(past[65])
        _,_,_,_,dup,conf=structural_stock([a,a[30:31]],np.array(self.cal))
        self.assertEqual((dup,conf),(1,0))


if __name__=='__main__': unittest.main()
