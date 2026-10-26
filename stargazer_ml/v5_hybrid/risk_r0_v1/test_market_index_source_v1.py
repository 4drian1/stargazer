import unittest
import tempfile
import hashlib
import json
from pathlib import Path

from market_index_source_v1 import load_csi300_closes, MarketIndexSourceUnbound


class MarketIndexSourceTests(unittest.TestCase):
    def bound(self,folder,body='date,close\n20240102,3500\n',**overrides):
        path=Path(folder)/'index.csv'
        path.write_text(body,encoding='utf-8')
        meta=dict(symbol='SH000300',semantics='CSI300_PRICE_INDEX',adjustment='NONE',
                  source='SYNTHETIC_TEST_NOT_REAL_MARKET_EVIDENCE',
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        meta.update(overrides)
        path.with_name(path.name+'.binding.json').write_text(json.dumps(meta),encoding='utf-8')
        return path

    def test_unbound_raises(self):
        with self.assertRaises(MarketIndexSourceUnbound):
            load_csi300_closes(None)

    def test_missing_path_raises(self):
        with self.assertRaises(MarketIndexSourceUnbound):
            load_csi300_closes('D:/nonexistent_csi300.parquet')

    def test_accept_and_sort(self):
        with tempfile.TemporaryDirectory() as folder:
            path=self.bound(folder,'date,close\n20240103,3501\n20240102,3500\n')
            self.assertEqual(list(load_csi300_closes(path)),[20240102,20240103])

    def test_identity_hash_binding(self):
        for overrides in ({'symbol':'000001'},{'semantics':'STOCK_MEDIAN_PROXY'},
                          {'adjustment':'QFQ'},{'source':''},{'sha256':'bad'}):
            with self.subTest(overrides=overrides),tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(MarketIndexSourceUnbound):
                    load_csi300_closes(self.bound(folder,**overrides))

    def test_unbound_regular_stock_file_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=self.bound(folder)
            path.with_name(path.name+'.binding.json').unlink()
            with self.assertRaises(MarketIndexSourceUnbound): load_csi300_closes(path)

    def test_invalid_rows_never_silently_dropped(self):
        for row in ('20240102,nan','20240102,inf','20240102,0',
                    '20241302,3500','20240106,3500','20240102.5,3500'):
            with self.subTest(row=row),tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(MarketIndexSourceUnbound):
                    load_csi300_closes(self.bound(folder,'date,close\n'+row+'\n'))

    def test_duplicate_conflict(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(MarketIndexSourceUnbound):
                load_csi300_closes(self.bound(folder,'date,close\n20240102,3500\n20240102,3501\n'))

    def test_same_value_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(load_csi300_closes(self.bound(folder,'date,close\n20240102,3500\n20240102,3500\n')),
                             {20240102:3500.})


if __name__ == '__main__':
    unittest.main()
