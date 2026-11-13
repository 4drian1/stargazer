import gzip,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import msgpack
H=Path(__file__).resolve().parent;R=H.parents[1];K=R/'stargazer_ml/v5_hybrid/risk_r0_v1'
sys.path[:0]=[str(K),str(H.parent)]
import live_postclose_source_v1 as s
def bar(t=20261028150000):return dict(code='002933',date=t,open=25.,high=26.,low=24.,close=25.,volume=100.,amount=2500.)
class CacheTests(unittest.TestCase):
    def cache(self,root,rows,proof=False,time=20261028200000):
        old=root/'run_20261028_old';old.mkdir();p=old/'raw_002933_20261028all.msgpack.gz';p.write_bytes(gzip.compress(msgpack.packb(rows,use_bin_type=True)))
        if proof:p.with_name(p.name+'.receipt.json').write_text(json.dumps(dict(code='002933',period='20261028*',sha256=s.digest(p.read_bytes()),captured_at=time)))
        return p
    def capture(self,root):
        new=root/'run_20261028_new';new.mkdir();c=s.LocalCapture(new,budget_root=root);c.cache_day=20261028;c.required_cutoff=20261028150000;return c
    def test_empty_intraday_archive_requires_real_new_request(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.cache(root,[]);c=self.capture(root)
            with patch.object(c.opener,'open',return_value=io.BytesIO(msgpack.packb([bar()],use_bin_type=True))) as http:
                rows,ref=c.fetch('002933','20261028*')
            self.assertEqual(http.call_count,1);self.assertEqual(rows,[bar()]);self.assertNotIn('reused',ref)
            self.assertTrue(Path(ref['path']+'.receipt.json').exists())
    def test_nonempty_preclose_cache_still_not_postclose_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.cache(root,[bar(20261028094500)],True,20261028094600);c=self.capture(root)
            with patch.object(c.opener,'open',return_value=io.BytesIO(msgpack.packb([bar()],use_bin_type=True))) as http:c.fetch('002933','20261028*')
            self.assertEqual(http.call_count,1)
    def test_verified_postclose_cache_reused_without_request(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.cache(root,[['key',bar()]],True);c=self.capture(root)
            with patch.object(c.opener,'open',side_effect=AssertionError('NO_HTTP')):rows,ref=c.fetch('002933','20261028*')
            self.assertEqual(rows,[bar()]);self.assertTrue(ref['reused']);self.assertEqual(c.requests,0)
    def test_bad_proof_hash_is_not_reused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.cache(root,[bar()],True);p.write_bytes(gzip.compress(msgpack.packb([bar(20261028110000)],use_bin_type=True)));c=self.capture(root)
            with patch.object(c.opener,'open',return_value=io.BytesIO(msgpack.packb([bar()],use_bin_type=True))) as http:c.fetch('002933','20261028*')
            self.assertEqual(http.call_count,1)
    def test_missing_candidate_stops_after_one_request(self):
        c=Mock();c.fetch.return_value=([],{});c.requests=1
        with self.assertRaisesRegex(ValueError,'002933:CANDIDATE_CURRENT_CUTOFF_MISSING'):
            s.candidate_preflight(c,20261028,['002933'],20261028150000,{'002933'},lambda **kw:None)
        c.fetch.assert_called_once_with('002933','20261028*');c.save.assert_called_once()
    def test_no_membership_stops_without_request(self):
        c=Mock();c.requests=0
        with self.assertRaisesRegex(ValueError,'MEMBERSHIP_MISSING'):s.candidate_preflight(c,20261028,['002933'],20261028150000,set(),lambda **kw:None)
        c.fetch.assert_not_called()
    def test_partial_candidate_does_not_pass_as_completed_close(self):
        c=Mock();c.fetch.return_value=([bar(20261028110000)],{});c.requests=1
        with self.assertRaisesRegex(ValueError,'CUTOFF_MISSING'):s.candidate_preflight(c,20261028,['002933'],20261028150000,{'002933'},lambda **kw:None)
    def test_valid_candidate_is_prefetched_for_later_reuse(self):
        c=Mock();c.fetch.return_value=([bar()],{'sha256':'bound'});c.requests=1
        ready,fail=s.candidate_preflight(c,20261028,['002933'],20261028150000,{'002933'},lambda **kw:None)
        self.assertFalse(fail);self.assertEqual(ready['002933'][0],[bar()])
if __name__=='__main__':unittest.main()
