import base64,gzip,io,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from datetime import datetime,timedelta
import numpy as np,pandas as pd,msgpack
H=Path(__file__).resolve().parent;R=H.parents[1];K=R/'stargazer_ml/v5_hybrid/risk_r0_v1';sys.path.insert(0,str(K))
from live_postclose_source_v1 import LocalCapture,CapturedMinutes,digest
from joint_shadow_context_v1 import RiskSnapshot
from stock_prefix_cache_v1 import PrefixFeatureCache
from intraday_relative_stall_v2 import minute_returns
import build_joint_risk_features_v1 as current

def raw(days):
    rows=[]
    for day in days:
        for hour in (9,13):
            start=datetime.strptime(str(day),'%Y%m%d').replace(hour=hour,minute=30 if hour==9 else 0)
            for j in range(1,121):
                p=10+len(rows)*.001;rows.append(dict(code='002933',date=int((start+timedelta(minutes=j)).strftime('%Y%m%d%H%M%S')),open=p,high=p+.1,low=p-.1,close=p,volume=100.,amount=1000.))
    return pd.DataFrame(rows)
def legacy():
    backups=json.loads(gzip.decompress((H/'BASELINE.json.gz').read_bytes()));path='stargazer_ml/v5_hybrid/risk_r0_v1/build_joint_risk_features_v1.py'
    module=types.ModuleType('original_stock_kernel');module.__file__=str(R/path)
    exec(compile(base64.b64decode(backups[path]),module.__file__,'exec'),module.__dict__);return module

class PrefixCacheTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.original=legacy();self.days=[20261021,20261022,20261023]
    def tearDown(self):self.tmp.cleanup()
    def snapshot(self,frame,days,extension=None):
        seed=self.root/'seed.parquet'
        if not seed.exists():frame.to_parquet(seed,index=False)
        refs=[]
        if extension is not None:
            p=self.root/'new.msgpack.gz';rawbody=msgpack.packb(extension.to_dict('records'),use_bin_type=True);p.write_bytes(gzip.compress(rawbody))
            refs=[dict(path=str(p),sha256=digest(p.read_bytes()),raw_sha256=digest(rawbody),period='20261026*')]
        output=self.root/('out'+str(len(list(self.root.glob('out*')))));output.mkdir();capture=LocalCapture(output,budget_root=self.root)
        entries={'002933':dict(seed=dict(path=str(seed),sha256=digest(seed.read_bytes())),extensions=refs)};cutoff=days[-1]*1000000+150000
        view=CapturedMinutes(entries,capture,cutoff)
        snap=RiskSnapshot(cutoff,cutoff,days,view,{}, {'B':['002933']},{'002933':dict(reference='raw',clock='END_1M',price_basis='RAW',available_at=cutoff)},cutoff,'FORWARD_SNAPSHOT',observed_at=cutoff)
        return snap,capture
    def compare(self,frame,days,rows,ret):
        old,_,_=self.original.stock_features(frame,'002933',np.asarray(days),set());pd.testing.assert_frame_equal(rows,old,check_exact=True)
        refs=dict(zip(days,old.groupby(old.cutoff_index//16).stock_reference_close_raw.first().reindex(range(len(days))).to_numpy()))
        expected,_=minute_returns(frame,'002933',days,refs)
        for f in (5,15):np.testing.assert_array_equal(ret[f],expected[f])
    def test_refactor_is_bit_exact_with_missing_bars_and_days(self):
        frame=raw(self.days).drop(index=[1,10,243,365]);a=self.original.stock_features(frame,'002933',np.asarray(self.days),set());b=current.stock_features(frame,'002933',np.asarray(self.days),set());pd.testing.assert_frame_equal(a[0],b[0],check_exact=True);np.testing.assert_array_equal(a[1],b[1]);self.assertEqual(a[2],b[2])
    def test_cold_and_hot_match_original_no_raw_decode_on_hit(self):
        frame=raw(self.days);snap,c=self.snapshot(frame,self.days);cache=PrefixFeatureCache(snap,c);a,ret=cache.stock('002933',self.days,snap.cutoff);self.compare(frame,self.days,a,ret)
        snap,c=self.snapshot(frame,self.days);cache=PrefixFeatureCache(snap,c)
        with patch.object(CapturedMinutes,'__getitem__',side_effect=AssertionError('NO_HISTORY_DECODE')):b,ret=cache.stock('002933',self.days,snap.cutoff)
        self.compare(frame,self.days,b,ret);self.assertEqual(cache.audit['hit'],1)
    def test_append_one_day_keeps_full_rsi_history_bit_exact(self):
        base=raw(self.days);snap,c=self.snapshot(base,self.days);PrefixFeatureCache(snap,c).stock('002933',self.days,snap.cutoff)
        days=self.days+[20261026];new=raw([20261026]);snap,c=self.snapshot(base,days,new);cache=PrefixFeatureCache(snap,c)
        with patch.object(CapturedMinutes,'__getitem__',side_effect=AssertionError('NO_FULL_REBUILD')):rows,ret=cache.stock('002933',days,snap.cutoff)
        self.compare(pd.concat([base,new]),days,rows,ret);self.assertEqual(cache.audit['incremental'],1)
    def test_source_hash_tamper_never_uses_cache(self):
        frame=raw(self.days);snap,c=self.snapshot(frame,self.days);cache=PrefixFeatureCache(snap,c);cache.stock('002933',self.days,snap.cutoff);(self.root/'seed.parquet').write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'HASH_MISMATCH'):cache.stock('002933',self.days,snap.cutoff)
    def test_appended_revision_falls_back_to_full_path(self):
        frame=raw(self.days);snap,c=self.snapshot(frame,self.days);PrefixFeatureCache(snap,c).stock('002933',self.days,snap.cutoff)
        # Same old rows overlap; append fast path must reject even equal revisions.
        ext=pd.concat([frame.iloc[:3],raw([20261026])]);days=self.days+[20261026];snap,c=self.snapshot(frame,days,ext);cache=PrefixFeatureCache(snap,c);rows,ret=cache.stock('002933',days,snap.cutoff)
        self.compare(pd.concat([frame,raw([20261026])]),days,rows,ret);self.assertEqual(cache.audit['cold'],1)
    def test_future_cache_not_used_for_past_cutoff(self):
        frame=raw(self.days);snap,c=self.snapshot(frame,self.days);PrefixFeatureCache(snap,c).stock('002933',self.days,snap.cutoff)
        days=self.days[:2];snap,c=self.snapshot(frame,days);cache=PrefixFeatureCache(snap,c);rows,ret=cache.stock('002933',days,snap.cutoff)
        self.compare(frame.loc[frame.date<=snap.cutoff],days,rows,ret);self.assertEqual(cache.audit['cold'],1)
    def test_oversize_cache_does_not_break_scoring(self):
        frame=raw(self.days);snap,c=self.snapshot(frame,self.days);c.disk_cap=c.disk_bytes+1024;cache=PrefixFeatureCache(snap,c);rows,ret=cache.stock('002933',self.days,snap.cutoff)
        self.compare(frame,self.days,rows,ret);self.assertEqual(cache.audit['write_skipped_budget'],1)
if __name__=='__main__':unittest.main()
