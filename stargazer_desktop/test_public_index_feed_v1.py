"""Public index source through the real prefix validator and mirror writer."""
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow.parquet as pq
import public_index_feed_v1 as feed
import index_minutes_v1 as desktop

ROOT=Path(__file__).parents[1]
FILE=ROOT/'stargazer_ml/v5_hybrid/risk_r0_v1/index_minutes_v1/index_minutes_updater_v1.py'
spec=importlib.util.spec_from_file_location('public_feed_test_updater',FILE)
updater=importlib.util.module_from_spec(spec);spec.loader.exec_module(updater)


def body(count=6,day=20261026):
    values=[]
    for stamp in updater.grid(day)[:count]:
        values.append(dict(day=datetime.strptime(str(stamp),'%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S'),
            open='3600',high='3602',low='3599',close='3601',volume='999',amount='99900'))
    return json.dumps(values).encode()


class SourceTests(unittest.TestCase):
    def test_continuous_prefix_and_unfinished_tail(self):
        rows,check=feed.parse(body(7),'000001.SH',20261026100200,updater)
        self.assertEqual(len(rows),6)
        self.assertEqual(check['in_progress'],[20261026100500])
        self.assertEqual(check['verdict'],'OK_CURRENT_PREFIX')
        self.assertTrue(all(r['raw_volume'] is None and r['amount'] is None for r in rows))

    def test_bad_inputs_fail_closed(self):
        original=json.loads(body())
        variants=[original[:2]+original[3:],original[::-1],original+[original[-1]],
            [dict(original[0],low='9999')]+original[1:],
            [dict(original[0],close='NaN')]+original[1:],
            [dict(original[0],day='2026-10-27 09:35:00')]]
        for values in variants:
            with self.subTest(values=values[:1]),self.assertRaises(ValueError):
                feed.parse(json.dumps(values).encode(),'000001.SH',20261026100000,updater)

    def test_previous_day_rows_do_not_replace_current_day(self):
        combined=json.loads(body(48,20261023))+json.loads(body())
        rows,_=feed.parse(json.dumps(combined).encode(),'000001.SH',20261026100000,updater)
        self.assertEqual(len(rows),6)
        self.assertTrue(all(r['day']==20261026 for r in rows))

    def test_postclose_requires_all_completed_bars(self):
        with self.assertRaisesRegex(ValueError,'POSTCLOSE_FAIL'):
            feed.parse(body(47),'000001.SH',20261026150100,updater)
        rows,check=feed.parse(body(48),'000001.SH',20261026150100,updater)
        self.assertTrue(check['fully_closed']);self.assertEqual(len(rows),48)

    def check_sync(self,mirror,root,count=6,asof=20261026100200):
        with patch.object(feed,'fetch',side_effect=lambda symbol:(body(count),'https://quotes.sina.cn/fixed/'+symbol,asof+1)) as fetch:
            result=feed.sync_public(updater,mirror,root,asof,{})
        return result,fetch

    def test_real_mirror_writer_null_activity_and_no_intraday_release(self):
        with tempfile.TemporaryDirectory() as folder:
            mirror=Path(folder)/'mirror';mirror.mkdir();root=Path(folder)/'runtime';root.mkdir()
            result,fetch=self.check_sync(mirror,root,count=7)
            self.assertEqual(fetch.call_count,2)
            self.assertFalse(result['release_status']['evidence_complete'])
            for name in ('sse','cyb'):
                frame=pq.read_table(mirror/(name+'_5min.parquet')).to_pandas()
                self.assertEqual(len(frame),6);self.assertEqual(frame.ts.iloc[-1],20261026100000)
                self.assertTrue(frame.raw_volume.isna().all())
                meta=feed.read(mirror/(name+'_5min.meta.json'))
                self.assertEqual(meta['source_contract'],feed.CONTRACT)
                self.assertFalse(meta['volume_verified'])
                self.assertEqual(meta['available_at'],20261026100201)
            with patch.object(feed,'fetch') as again:
                self.assertEqual(feed.sync_public(updater,mirror,root,20261026100230,{}),result)
                # Fresh continuous prefix also avoids a request in the next minute.
                feed.sync_public(updater,mirror,root,20261026100330,{})
            again.assert_not_called()

    def test_missing_one_index_closes_release_and_preserves_old_mirror(self):
        with tempfile.TemporaryDirectory() as folder:
            mirror=Path(folder)/'mirror';mirror.mkdir();root=Path(folder)/'runtime';root.mkdir()
            self.check_sync(mirror,root)
            before=(mirror/'cyb_5min.parquet').read_bytes()
            def fetch(symbol):
                if symbol=='399006.SZ':raise OSError('offline')
                return body(7),'https://quotes.sina.cn/fixed',20261026100601
            with patch.object(feed,'fetch',side_effect=fetch):
                result=feed.sync_public(updater,mirror,root,20261026100600,{})
            self.assertFalse(result['release_status']['evidence_complete'])
            self.assertEqual(result['symbols']['399006.SZ']['status'],'PUBLIC_SOURCE_WAIT')
            self.assertEqual((mirror/'cyb_5min.parquet').read_bytes(),before)

    def test_two_closed_days_keep_qualification_on_repeated_poll(self):
        with tempfile.TemporaryDirectory() as folder:
            mirror=Path(folder)/'mirror';mirror.mkdir();root=Path(folder)/'runtime';root.mkdir()
            result,_=self.check_sync(mirror,root,48,20261026150100)
            self.assertTrue(result['release_status']['evidence_complete'])
            with patch.object(feed,'fetch') as fetch:
                repeat=feed.sync_public(updater,mirror,root,20261026150200,{})
            fetch.assert_not_called();self.assertTrue(repeat['release_status']['evidence_complete'])

    def test_overlap_conflict_does_not_replace_published_prices(self):
        with tempfile.TemporaryDirectory() as folder:
            mirror=Path(folder)/'mirror';mirror.mkdir();root=Path(folder)/'runtime';root.mkdir()
            self.check_sync(mirror,root)
            before=(mirror/'sse_5min.parquet').read_bytes()
            value=json.loads(body(7));value[0]['close']='3600.5'
            with patch.object(feed,'fetch',return_value=(json.dumps(value).encode(),'https://quotes.sina.cn/fixed',20261026100601)):
                result=feed.sync_public(updater,mirror,root,20261026100600,{})
            self.assertIn('OVERLAP_CONFLICT',result['symbols']['000001.SH']['reason'])
            self.assertEqual((mirror/'sse_5min.parquet').read_bytes(),before)

    def test_desktop_entry_uses_public_feed_without_starting_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            mirror=Path(folder)/'mirror';mirror.mkdir();root=Path(folder)/'runtime';root.mkdir()
            # Native inputs intentionally absent. Real public fallback writes both mirrors.
            with patch.object(desktop,'MIRROR',mirror),patch.object(desktop,'SOURCE_FILES',{'000001.SH':mirror/'missing1','399006.SZ':mirror/'missing2'}), \
                 patch.object(desktop,'_load_updater',return_value=updater),patch('postclose_risk_runtime.directory',return_value=root), \
                 patch.object(updater,'get_index_freshness',return_value=dict(last_date=None,stale=True,max_lag_days=None,today_trading_day=20261026,required_latest=20261023,calendar_source='test',symbols={},checked_at='2026-10-26T10:02:00')), \
                 patch.object(feed,'fetch',return_value=(body(),'https://quotes.sina.cn/fixed',20261026100201)):
                result=desktop.sync_once(20261026100200,public=True)
            self.assertEqual(result['status'],'OK')
            self.assertEqual(result['symbols']['000001.SH']['status'],'OK_CURRENT_PREFIX')
            self.assertFalse(result['release']['evidence_complete'])


if __name__=='__main__':unittest.main()
