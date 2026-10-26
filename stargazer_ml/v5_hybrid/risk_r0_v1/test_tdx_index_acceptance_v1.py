from copy import deepcopy
import struct
import unittest

from tdx_index_acceptance_v1 import aggregate_15, grid, price_features, read_binary


def rows(day=20250102):
    return [dict(date=t,open=10.,high=11.,low=9.,close=10.,volume=100,amount=10000.) for t in grid(day,5)]


class IndexAcceptanceTests(unittest.TestCase):
    def test_float_ohlc_and_signed_integer_volume_layout(self):
        packed=(2025-2004)*2048+102
        result=read_binary(struct.pack('<HHfffffiI',packed,575,10.,11.,9.,10.,10000.,100,0))[0]
        self.assertEqual(result['date'],20250102093500)
        self.assertEqual(result['open'],10.)
        self.assertEqual(result['volume'],100)

    def test_bad_record_size_and_clock(self):
        with self.assertRaisesRegex(ValueError,'RECORD_SIZE'): read_binary(b'x')
        with self.assertRaisesRegex(ValueError,'CLOCK'):
            read_binary(struct.pack('<HHfffffiI',43110,1440,10.,11.,9.,10.,10000.,100,0))

    def test_three_native_bars_no_lunch_stitch(self):
        b=aggregate_15(rows(),[20250102])
        self.assertEqual(len(b),16)
        self.assertEqual(b[7]['bar_end'],20250102113000)
        self.assertEqual(b[8]['bar_start'],20250102130000)
        self.assertTrue(all(r['volume'] is None for r in b))

    def test_missing_source_bar_is_not_partial_price(self):
        source=rows();del source[1]
        b=aggregate_15(source,[20250102])
        self.assertEqual(b[0]['status'],'INCOMPLETE')
        self.assertIsNone(b[0]['close'])
        self.assertEqual(b[0]['missing_source_bars'],[20250102094000])

    def test_duplicate_rejected_even_identical(self):
        source=rows()
        with self.assertRaisesRegex(ValueError,'DUPLICATE'): aggregate_15(source+[source[0]],[20250102])

    def test_nonfinite_and_bad_ohlc(self):
        for change in (dict(close=float('nan')),dict(low=12.),dict(volume=-1)):
            source=rows();source[0].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError): aggregate_15(source,[20250102])

    def test_rogue_clock_and_date_rejected(self):
        for ts in (20250102120000,20250102093600,20250103093500):
            source=rows();source[0]['date']=ts
            with self.subTest(ts=ts),self.assertRaisesRegex(ValueError,'GRID'): aggregate_15(source,[20250102])

    def test_wilder_flat_seed_and_warmup(self):
        source=rows()+rows(20250103)+rows(20250106)
        fs=price_features(aggregate_15(source,[20250102,20250103,20250106]))
        self.assertEqual(sum(f['rsi14'] is not None for f in fs),34)
        self.assertIsNone(fs[13]['rsi14'])
        self.assertEqual(fs[14]['rsi14'],50.)
        self.assertIsNone(fs[16]['return15m'])  # no overnight-as-15m return

    def test_missing_resets_warmup(self):
        source=rows();del source[20]
        fs=price_features(aggregate_15(source,[20250102]))
        self.assertTrue(all(f['rsi14'] is None for f in fs))

    def test_no_volume_or_live_admission(self):
        fs=price_features(aggregate_15(rows(),[20250102]))
        self.assertTrue(all(f['volume_feature'] is None and f['available_at'] is None and not f['admitted'] for f in fs))

    def test_prefix_and_future(self):
        source=rows()
        full=price_features(aggregate_15(source,[20250102]))
        prefix=[r for r in source if r['date']<=20250102100000]
        self.assertEqual(full[:2],price_features(aggregate_15(prefix,[20250102]))[:2])
        altered=deepcopy(source)
        for r in altered:
            if r['date']>20250102100000:
                for k in ('open','high','low','close'):r[k]*=2
        self.assertEqual(full[:2],price_features(aggregate_15(altered,[20250102]))[:2])


if __name__=='__main__': unittest.main()
