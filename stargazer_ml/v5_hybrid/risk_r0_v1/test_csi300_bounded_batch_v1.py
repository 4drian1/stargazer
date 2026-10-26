import unittest
from unittest.mock import patch
import json
import tempfile
from pathlib import Path
import urllib.parse

from csi300_bounded_batch_v1 import fetch_segment,merge_required,preflight,SYMBOL
from independent_fetch_client import TencentDailyClient,FetchCounter,ByteCounter,RawResponseRejected,FetchBudgetExceeded


class IndexBatchTests(unittest.TestCase):
    def payload(self,*,name='沪深300',symbol=SYMBOL,key='day'):
        return json.dumps(dict(data={symbol:{key:[['2019-01-02','3000','3001','3002','2999','1']],
                                               'qt':{symbol:['1',name,'000300']}}}),ensure_ascii=False).encode()

    def client(self,raw):
        return TencentDailyClient(FetchCounter(6),ByteCounter(2*1024**2),max_retries=0,
                                  transport=lambda *a:raw,sleep=lambda *a:None,disk_counter=ByteCounter(100000))

    def test_exact_index_symbol_and_archive(self):
        raw=self.payload();client=self.client(raw);urls=[]
        def transport(url,*args):urls.append(url);return raw
        client._transport=transport
        with tempfile.TemporaryDirectory() as out:
            data,entry=fetch_segment(client,'20180102','20191231',out)
            self.assertEqual(data[20190102]['close'],3001)
            self.assertTrue(urllib.parse.parse_qs(urllib.parse.urlparse(urls[0]).query)['param'][0].startswith('sh000300,day,'))
            self.assertEqual(next(Path(out).rglob('*.bin')).read_bytes(),raw)
            self.assertEqual(client.counter.used,1)

    def test_wrong_name_rejected_and_archived(self):
        client=self.client(self.payload(name='平安银行'))
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaisesRegex(ValueError,'IDENTITY_NAME'):fetch_segment(client,'20180102','20191231',out)
            self.assertEqual(len(list(Path(out).rglob('*.bin'))),1)

    def test_wrong_market_symbol(self):
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(RawResponseRejected):
                fetch_segment(self.client(self.payload(symbol='sz000300')),'20180102','20191231',out)

    def test_adjusted_response_rejected(self):
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(RawResponseRejected):
                fetch_segment(self.client(self.payload(key='qfqday')),'20180102','20191231',out)

    def test_no_retry_on_transport_failure(self):
        client=self.client(b'');calls=[]
        def failed(*args):calls.append(1);raise TimeoutError('fixture')
        client._transport=failed
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(TimeoutError):fetch_segment(client,'20180102','20191231',out)
        self.assertEqual(len(calls),1);self.assertEqual(client.counter.used,1)

    def test_missing_required_preserves_available(self):
        all_prices={}
        with self.assertRaisesRegex(ValueError,'MISSING'):merge_required(all_prices,{20190102:{'close':1}},[20190102,20190103])
        self.assertIn(20190102,all_prices)

    def test_overlap_conflict(self):
        with self.assertRaisesRegex(ValueError,'CONFLICT'):
            merge_required({20190102:{'close':1}},{20190102:{'close':2}},[20190102])

    def test_superset_not_population_expansion(self):
        prices={};merge_required(prices,{20180102:{'close':1},20190102:{'close':2}},[20190102])
        self.assertEqual(prices[20190102]['close'],2)

    def test_request_hard_cap(self):
        client=self.client(self.payload());client.counter.used=6
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(FetchBudgetExceeded):fetch_segment(client,'20180102','20191231',out)
            self.assertFalse(list(Path(out).rglob('*.bin')))

    def test_missing_auth_refuses_before_network(self):
        with self.assertRaisesRegex(ValueError,'AUTHORIZATION_HASH_REQUIRED'):preflight(auth_sha=None)


if __name__=='__main__':unittest.main()
