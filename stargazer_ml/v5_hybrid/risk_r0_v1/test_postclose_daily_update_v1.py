import copy
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_postclose_background_v1 as bg
import postclose_daily_update_v1 as update
from postclose_research_scorer_v1 import strict_json
from postclose_daily_job_v1 import plan,read_input


class UpdateTests(unittest.TestCase):
    def test_01_actual_prepare_and_consumer(self):
        with tempfile.TemporaryDirectory() as root,patch.object(update,'INPUTS',Path(root)/'inputs'):
            result=update.prepare(20261016,write=True)
            b,_=read_input(result['background'],'POSTCLOSE_MARKET_GOLDEN')
            c,_=read_input(result['calendar'],'TRADING_CALENDAR')
            self.assertEqual(len(b['features']),78);self.assertEqual(c['days'][-1],20261016)
            self.assertFalse(b['provenance_verified'])

    def test_02_existing_background_not_overwritten(self):
        with tempfile.TemporaryDirectory() as root,patch.object(update,'INPUTS',Path(root)/'inputs'):
            update.prepare(20261016,write=True)
            with self.assertRaisesRegex(ValueError,'NO_OVERWRITE'):update.prepare(20261016,write=True)

    def test_03_future_unavailable_never_falls_back(self):
        with self.assertRaisesRegex(ValueError,'NO_FALLBACK'):update.prepare(20261019)

    def write_source(self,path,rows):
        body=b''
        for row in rows:
            day=row['date']//1000000;year=day//10000;month=day//100%100;date=day%100
            time=row['date']%1000000;minute=(time//10000)*60+(time//100)%100
            body+=struct.pack('<HHfffffiI',(year-2004)*2048+month*100+date,minute,
                row['open'],row['high'],row['low'],row['close'],row['amount'],row['volume'],0)
        path.write_bytes(body)

    def test_04_synthetic_growth_and_checkpoint_survives_later_trim(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);sources={s:root/(s+'.lc5') for s in bg.SOURCES}
            original={s:bg.read_binary(p.read_bytes()) for s,p in bg.SOURCES.items()}
            with patch.object(bg,'SOURCES',sources),patch.object(update,'INPUTS',root/'inputs'),patch.object(update,'clock_now',return_value='2026-10-20T18:00:00+08:00'):
                for symbol,rows in original.items():
                    new=[dict(r,date=20261019000000+r['date']%1000000) for r in rows[-48:]]
                    self.write_source(sources[symbol],rows+new)
                prepared=update.prepare(20261019,write=True)
                self.assertTrue(plan(20261019,prepared['calendar'],prepared['background'])[0]['ready'])
                for symbol,rows in original.items():
                    # Drop 10/19 from the next export: its accepted delta must retain it.
                    new=[dict(r,date=20261020000000+r['date']%1000000) for r in rows[-48:]]
                    self.write_source(sources[symbol],rows[-48*30:]+new)
                prepared=update.prepare(20261020,write=True)
                calendar=strict_json(Path(prepared['calendar']).read_bytes())
                self.assertEqual(calendar['days'][-3:],[20261016,20261019,20261020])
                manifest=strict_json((root/'inputs/20261020/manifest.json').read_bytes())
                self.assertEqual(manifest['new_days'],{'000001.SH':1,'399006.SZ':1})

    def test_05_checkpoint_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as root,patch.object(update,'INPUTS',Path(root)/'inputs'):
            update.prepare(20261016,write=True)
            (update.INPUTS/'20261016/daily_delta.json').write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError,'CHECKPOINT_HASH'):update.checkpoint_history(20261016)


if __name__=='__main__':unittest.main()
