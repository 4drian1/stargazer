"""Local-only context reuse and authorized sector execution regression tests."""
from datetime import datetime, timedelta
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

H=Path(__file__).resolve().parent;A=H.parent;R=A.parent;K=R/'stargazer_ml/v5_hybrid/risk_r0_v1'
sys.path[:0]=[str(A),str(K),str(K.parent/'green_common_profile')]
import numpy as np
import pandas as pd
from joint_shadow_context_v1 import RiskSnapshot
from live_postclose_source_v1 import LocalCapture,CapturedMinutes,digest
from shared_risk_context_v1 import SharedPreparedContext,prepared_snapshot
from stock_prefix_cache_v1 import PrefixFeatureCache
from build_joint_risk_features_v1 import aggregate_minutes,features_from_arrays
import sector_watch_v1 as watch


def raw(days):
    rows=[]
    for day in days:
        for hour in (9,13):
            start=datetime.strptime(str(day),'%Y%m%d').replace(hour=hour,minute=30 if hour==9 else 0)
            for j in range(1,121):
                p=10+len(rows)*.001
                rows.append(dict(code='002933',date=int((start+timedelta(minutes=j)).strftime('%Y%m%d%H%M%S')),open=p,high=p+.1,low=p-.1,close=p,volume=100.,amount=1000.))
    return pd.DataFrame(rows)


class SharedTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=H);self.root=Path(self.tmp.name)
        self.days=[20261026,20261027,20261028];self.cutoff=20261028150000
        self.output=self.root/'run_test';self.output.mkdir()
        self.c=LocalCapture(self.output,budget_root=self.root)
        frame=raw(self.days);seed=self.root/'seed.parquet';frame.to_parquet(seed,index=False)
        self.entries={'002933':dict(seed=dict(path=str(seed),sha256=digest(seed.read_bytes())),extensions=[])}
        index=frame.rename(columns={'code':'ignored'})
        self.snap=RiskSnapshot(self.cutoff,self.cutoff,self.days,CapturedMinutes(self.entries,self.c,self.cutoff),
            {'000001.SH':index,'399006.SZ':index}, {'B':['002933']},
            {'002933':dict(reference='fixed',clock='END_1M',price_basis='RAW',available_at=self.cutoff)},
            self.cutoff,'FORWARD_SNAPSHOT',observed_at=self.cutoff)
        self.binding=dict(path='test_source',sha256='test_hash')
        self.c.save('source_manifest.json',json.dumps(dict(entries=self.entries,failures={},index_source_binding=self.binding)).encode())

    def tearDown(self):self.tmp.cleanup()

    def prepare(self):
        cache=SharedPreparedContext(self.snap,self.c)
        rows,ret=cache.stock('002933',self.days,self.cutoff);cache.commit()
        self.assertTrue(cache.audit['committed'])
        out=self.root/'run_next';out.mkdir()
        return rows,ret,out

    def load(self,out,**kw):
        return prepared_snapshot(self.root,out,kw.get('cutoff',self.cutoff),kw.get('codes',['002933']),index_source=kw.get('binding',('test_source','test_hash')))

    def test_cached_day_values_exact_without_history_read(self):
        rows,ret,out=self.prepare();snap,c,_=self.load(out)
        with (patch.object(PrefixFeatureCache,'stock',side_effect=AssertionError('NO_FULL_HISTORY')),
              patch.object(CapturedMinutes,'__getitem__',side_effect=AssertionError('NO_RAW_DECODE'))):
            cache=SharedPreparedContext(snap,c);actual,native=cache.stock('002933',self.days,self.cutoff)
        expected=rows[rows.cutoff_index//16==len(self.days)-1].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,expected,check_exact=True)
        for f in (5,15):np.testing.assert_array_equal(native[f],ret[f][-(240//f):])
        self.assertEqual(cache.audit['prepared_hits'],1)

    def test_new_cutoff_not_served_old_snapshot(self):
        _,_,out=self.prepare();self.assertIsNone(self.load(out,cutoff=20261029100000))

    def test_supplied_board_order_preserved(self):
        self.snap.boards={'Z':['002933'],'A':['002933']}
        _,_,out=self.prepare();snap,_,_=self.load(out)
        self.assertEqual(list(snap.boards),['Z','A'])

    def test_new_candidate_missing_kept_unknown(self):
        _,_,out=self.prepare();self.assertIsNone(self.load(out,codes=['000001']))

    def test_index_revision_requires_fresh_capture(self):
        _,_,out=self.prepare();self.assertIsNone(self.load(out,binding=('test_source','different_hash')))

    def test_kernel_change_invalidates_preparation(self):
        _,_,out=self.prepare()
        with patch('shared_risk_context_v1.kernel',return_value='different'):
            self.assertIsNone(self.load(out))

    def test_corruption_not_reused(self):
        _,_,out=self.prepare();p=next(self.root.glob('shared_context_v1/*/*/native.npz'));p.write_bytes(b'bad')
        self.assertIsNone(self.load(out))

    def test_runtime_budget_skip_does_not_break_features(self):
        self.c.disk_cap=self.c.disk_bytes+1024
        cache=SharedPreparedContext(self.snap,self.c);rows,_=cache.stock('002933',self.days,self.cutoff);cache.commit()
        self.assertGreater(len(rows),0);self.assertFalse(cache.audit['committed'])


def row(code):
    return dict(code=code,entry_rule=dict(eligible=True,policy='GOLDEN_RESONANCE_CROSSDAY_V4',p_date=20261028,levels=[10]),reference_price=11)


class SectorTests(unittest.TestCase):
    def setUp(self):
        watch.AUTHORIZED.clear();self.now=datetime(2026,10,29,10,0)
        self.plan=dict(assistant_plan_id='confirmed',data_date=20261028,monitor_date=20261029,
                       rows=[row('002933')],boards=[dict(name='pcb'),dict(name='存储')])
    def scan(self,name='pcb',codes=None):
        return dict(status='READY',board_requested=name,board='PCB概念',board_code='B',data_date=20261028,observation_date=20261029,
                    source_key='source1',monitor_rule_inputs=[dict(path='current',sha256='hash')],
                    monitor_rows=[row(c) for c in (codes or ['000001'])])

    def test_all_formals_not_only_display_top_five(self):
        codes=[f'60000{i}' for i in range(7)]
        plan=watch.execution_plan(self.plan,self.now,[self.scan(codes=codes)])
        self.assertEqual(len(plan['rows']),8);self.assertTrue(watch.matches(plan,self.plan))
        self.assertEqual(len(self.plan['rows']),1);self.assertNotIn('board_parent_digest',self.plan)

    def test_shared_stock_deduplicated_across_boards(self):
        plan=watch.execution_plan(self.plan,self.now,[self.scan(),self.scan('存储')])
        self.assertEqual(len(plan['rows']),2);self.assertEqual(len(plan['board_origins']['000001']),2)

    def test_existing_plan_stock_is_not_a_second_board_discovery(self):
        result=watch.execution_plan(self.plan,self.now,[self.scan(codes=['002933'])])
        self.assertEqual(result,self.plan)

    def test_plan_change_revokes_derived_execution(self):
        plan=watch.execution_plan(self.plan,self.now,[self.scan()])
        changed={**self.plan,'assistant_plan_id':'new'}
        self.assertFalse(watch.matches(plan,changed))

    def test_tampered_execution_not_authorized(self):
        plan=watch.execution_plan(self.plan,self.now,[self.scan()]);plan['rows'].append(row('600001'))
        self.assertFalse(watch.matches(plan,self.plan))

    def test_unrequested_board_not_admitted(self):
        self.assertEqual(watch.execution_plan(self.plan,self.now,[self.scan('other')]),self.plan)

    def test_manual_review_not_promoted(self):
        s=self.scan();s['monitor_rows']=[];s['rows']=[row('000001')];s['mode']='MANUAL_REVIEW_ONLY'
        self.assertEqual(watch.execution_plan(self.plan,self.now,[s]),self.plan)

    def test_stale_board_daily_inputs_not_admitted(self):
        s=self.scan();s['observation_date']=20261028
        self.assertEqual(watch.execution_plan(self.plan,self.now,[s]),self.plan)

    def test_expired_stock_plan_does_not_stop_authorized_board_daily_refresh(self):
        plan={**self.plan,'monitor_date':20261028,'rows':[row('002933')]}
        result=watch.execution_plan(plan,self.now,[self.scan()])
        self.assertEqual([r['code'] for r in result['rows']],['000001'])
        self.assertTrue(watch.matches(result,plan))

    def test_population_overflow_reported_not_silently_truncated(self):
        s=self.scan(codes=[str(600000+i) for i in range(129)])
        result=watch.execution_plan(self.plan,self.now,[s])
        self.assertEqual(result['board_watch_status'],'POPULATION_LIMIT')
        self.assertEqual(result['rows'],self.plan['rows']);self.assertTrue(watch.matches(result,self.plan))


if __name__=='__main__':unittest.main()
