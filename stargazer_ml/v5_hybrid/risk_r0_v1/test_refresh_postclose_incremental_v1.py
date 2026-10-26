import gzip
import unittest

import numpy as np

from refresh_postclose_incremental_v1 import pairs,valid_daily,contexts


def bar(code,day,close):
    return dict(code=code,date=day,open=close,high=close,low=close,close=close)


class IncrementalTests(unittest.TestCase):
    def test_single_dict_and_pairs(self):
        row=bar('000001',20261016,10)
        p=pairs(row,'日k')
        self.assertEqual(p,[('日k:000001:20261016',row)])
        self.assertEqual(valid_daily(p,20261016)['000001'],row)
        self.assertEqual(pairs(p,'日k'),p)

    def test_wrong_date_and_key_rejected(self):
        row=bar('000001',20261016,10)
        for key,day in [('日k:000002:20261016',20261016),('日k:000001:20261016',20261015)]:
            with self.assertRaises(ValueError):valid_daily([(key,row)],day)

    def test_invalid_ohlc_unknown_not_zero(self):
        for value in [0,np.nan,np.inf,True]:
            row=bar('000001',20261016,10);row['close']=value
            self.assertIsNone(valid_daily([('日k:000001:20261016',row)],20261016)['000001'])

    def test_duplicate_conflict(self):
        rows=[('日k:000001:20261016',bar('000001',20261016,c)) for c in [10,11]]
        with self.assertRaisesRegex(ValueError,'CONFLICT'):valid_daily(rows,20261016)

    def setup_context(self):
        members=[dict(code='A',symbols=['000001','000002']),dict(code='B',symbols=['000003'])]
        daily={d:{c:bar(c,d,p) for c,p in values.items()} for d,values in
            [(20261014,{'000001':10,'000002':10,'000003':10}),
             (20261015,{'000001':11,'000002':10,'000003':9}),
             (20261016,{'000001':11,'000002':11,'000003':9})]}
        return members,daily

    def test_self_exclusion_and_daily_winner(self):
        members,daily=self.setup_context()
        frame,_=contexts(daily,members,['000001','000002'])
        row=frame[(frame.code=='000002')&(frame.day==20261015)].iloc[0]
        self.assertAlmostEqual(row.context_sector_mean_return_loo_raw,.1)
        self.assertEqual(row.observed_leader_stock,'000001')
        self.assertAlmostEqual(row.context_current_leader_return_loo_raw,.1)
        own=frame[(frame.code=='000001')&(frame.day==20261015)].iloc[0]
        self.assertTrue(np.isnan(own.context_current_leader_return_loo_raw))

    def test_missing_member_changes_coverage_not_zero(self):
        members,daily=self.setup_context();daily[20261015]['000002']=None
        frame,_=contexts(daily,members,['000001'])
        row=frame[frame.day==20261015].iloc[0]
        self.assertTrue(np.isnan(row.context_sector_mean_return_loo_raw))
        self.assertEqual(row.context_sector_mean_coverage_loo,0)

    def test_future_day_does_not_change_previous_context(self):
        members,daily=self.setup_context();a,_=contexts(daily,members,['000001','000002'])
        daily[20261016]['000003']=bar('000003',20261016,100)
        b,_=contexts(daily,members,['000001','000002'])
        import pandas as pd
        pd.testing.assert_frame_equal(a[a.day==20261015],b[b.day==20261015])

    def test_gzip_exact_raw_reversible(self):
        raw=b'original msgpack bytes'
        self.assertEqual(gzip.decompress(gzip.compress(raw,mtime=0)),raw)


if __name__=='__main__':unittest.main()
