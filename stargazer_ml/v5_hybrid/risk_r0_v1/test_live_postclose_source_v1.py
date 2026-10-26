import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import msgpack

from live_postclose_source_v1 import LocalCapture, merge_minutes, months


def bar(ts, close=10.):
    return dict(code='000001',date=ts,open=10.,high=11.,low=9.,close=close,volume=10,amount=100)


class LiveSourceTests(unittest.TestCase):
    def test_months_across_year(self):
        self.assertEqual(list(months(20251220,20260203)),['202512*','202601*','202602*'])

    def test_no_future_or_auction_or_lunch(self):
        rows=[bar(t) for t in [20261021093000,20261021093100,20261021120000,
                               20261021130100,20261022093100]]
        result=merge_minutes(pd.DataFrame(),rows,'000001',20261021150000)
        self.assertEqual(list(result.date),[20261021093100,20261021130100])

    def test_conflict_not_last_write_wins(self):
        with self.assertRaisesRegex(ValueError,'OVERLAP_CONFLICT'):
            merge_minutes(pd.DataFrame([bar(20261021093100)]),
                          [bar(20261021093100,10.5)],'000001',20261021150000)

    def test_same_value_duplicate_is_unique(self):
        row=bar(20261021093100)
        self.assertEqual(len(merge_minutes(pd.DataFrame([row]),[row],'000001',20261021150000)),1)

    def test_identity_mismatch(self):
        with self.assertRaisesRegex(ValueError,'IDENTITY'):
            merge_minutes(pd.DataFrame(),[bar(20261021093100)],'000002',20261021150000)

    def test_request_and_disk_limits_before_action(self):
        with tempfile.TemporaryDirectory() as d:
            cap=LocalCapture(d)
            cap.requests=16000
            with patch.object(cap.opener,'open',side_effect=AssertionError('NO_IO')):
                with self.assertRaisesRegex(ValueError,'REQUEST_BUDGET'):
                    cap.fetch('000001','*')
            cap.disk_bytes=1024**3
            with self.assertRaisesRegex(ValueError,'DISK_BUDGET'):
                cap.save('x',b'x')
            self.assertFalse(Path(d,'x').exists())

    def test_response_archive_readback(self):
        import io,gzip
        raw=msgpack.packb([bar(20261021093100)],use_bin_type=True)
        with tempfile.TemporaryDirectory() as d:
            cap=LocalCapture(d)
            with patch.object(cap.opener,'open',return_value=io.BytesIO(raw)):
                rows,ref=cap.fetch('000001','202610*')
            self.assertEqual(gzip.decompress(Path(ref['path']).read_bytes()),raw)
            self.assertEqual(rows[0]['date'],20261021093100)
            self.assertEqual(cap.requests,1)

    def test_verified_same_day_cache_does_not_spend_request_quota(self):
        import gzip
        from live_postclose_source_v1 import digest
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=root/'run_20261021_old';new=root/'run_20261021_new'
            old.mkdir();new.mkdir()
            raw=msgpack.packb([bar(20261021093100)],use_bin_type=True)
            source=old/'raw_000001_202610all.msgpack.gz'
            source.write_bytes(gzip.compress(raw,mtime=0))
            cap=LocalCapture(new,budget_root=root);cap.cache_day=20261021;cap.requests=50
            with patch.object(cap.opener,'open',side_effect=AssertionError('NO_DUPLICATE_REQUEST')):
                rows,ref=cap.fetch('000001','202610*')
            self.assertEqual(cap.requests,50)
            self.assertEqual(ref['raw_sha256'],digest(raw))
            self.assertEqual(rows[0]['date'],20261021093100)
            self.assertTrue(ref['reused'])

    def test_windows_reader_lock_gets_bounded_replace_retry(self):
        from live_postclose_source_v1 import atomic_json
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'status.json';original=Path.replace;calls=[]
            def flaky(self,target):
                calls.append(1)
                if len(calls)==1:
                    raise PermissionError('simulated reader lock')
                return original(self,target)
            with patch.object(Path,'replace',flaky):
                atomic_json(path,{'ok':True})
            self.assertEqual(len(calls),2)
            self.assertTrue(path.exists())

    def test_ohlc_mask_is_bit_exact_to_frozen_invalid_bar_handling(self):
        from datetime import datetime,timedelta
        import numpy as np
        from live_postclose_source_v1 import clean_ohlc
        from build_joint_risk_features_v1 import stock_features
        rows=[]
        for day in [20261020,20261021]:
            for hour,minute in [(9,30),(13,0)]:
                start=datetime.strptime(str(day),'%Y%m%d').replace(hour=hour,minute=minute)
                for j in range(1,121):
                    ts=int((start+timedelta(minutes=j)).strftime('%Y%m%d%H%M%S'))
                    p=10+len(rows)*.001
                    rows.append(dict(code='000001',date=ts,open=p,high=p+.1,low=p-.1,
                                     close=p,volume=100,amount=1000))
        raw=pd.DataFrame(rows)
        raw.loc[10,'high']=raw.loc[10,'close']-.5
        cleaned,bad=clean_ohlc(raw)
        before=stock_features(raw,'000001',np.array([20261020,20261021]),set())[0]
        after=stock_features(cleaned,'000001',np.array([20261020,20261021]),set())[0]
        pd.testing.assert_frame_equal(before,after,check_exact=True)
        self.assertEqual(bad,[int(raw.loc[10,'date'])])

    def test_transport_failure_reservation_survives_new_instance(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'run';out.mkdir()
            for expected in [1,2]:
                cap=LocalCapture(out,budget_root=root);cap.cache_day=20261021
                with patch.object(cap.opener,'open',side_effect=TimeoutError('offline')):
                    with self.assertRaises(TimeoutError):
                        cap.fetch('000001','20261021*')
                row=json.loads((root/'attempts_20261021.jsonl').read_text().splitlines()[-1])
                self.assertEqual(row['quota_count'],expected)


if __name__=='__main__':
    unittest.main()
