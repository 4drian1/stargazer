import tempfile
import unittest
from pathlib import Path

import pandas as pd

from postclose_local_source_bridge_v1 import (bound_read,overlay_features,load_local_features,
                                            latest_day_payload,clock_now)
from postclose_research_scorer_v1 import ResearchScorer,sha


class SourceBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer=ResearchScorer()
        cls.data,cls.receipts=load_local_features(cls.scorer)

    def test_01_all_fields_match_existing_accepted_reader(self):
        from read_postclose_daily_context_repair_v1 import read_repaired_population
        expected,_=read_repaired_population()
        columns=['code','day',*self.scorer.fields]
        pd.testing.assert_frame_equal(self.data,expected[columns],check_exact=True)

    def test_02_labels_not_projected(self):
        self.assertEqual(set(self.data.columns),{'code','day',*self.scorer.fields})
        self.assertNotIn('next_session_adverse_excursion',self.data)

    def test_03_actual_receipts_not_historical_arrival(self):
        self.assertEqual(len(self.receipts),3)
        for row in self.receipts:
            self.assertIsNone(row['original_data_arrival_ts'])
            self.assertLessEqual(row['observation_started_at'],row['observation_finished_at'])
            self.assertIn('+08:00',row['observation_started_at'])

    def test_04_unknown_arrival_and_full_scope(self):
        payload,_=latest_day_payload(self.data,self.scorer)
        self.assertEqual(len(payload['rows']),43)
        self.assertTrue(all(row['decision_available_ts'] is None for row in payload['rows']))

    def test_05_missing_source_not_dropped(self):
        latest=self.data[self.data.day.eq(self.data.day.max())]
        payload,missing=latest_day_payload(latest.iloc[1:],self.scorer)
        self.assertEqual(len(payload['rows']),43)
        self.assertIn(latest.iloc[0].code,missing)
        row=next(r for r in payload['rows'] if r['code']==latest.iloc[0].code)
        self.assertTrue(all(v is None for v in row['features'].values()))

    def test_06_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'source';path.write_bytes(b'bad')
            receipts=[]
            with self.assertRaisesRegex(ValueError,'HASH_DRIFT'):
                bound_read(path,sha(b'expected'),receipts)
            self.assertEqual(receipts,[])

    def test_07_overlay_duplicate_or_order_rejected(self):
        base=self.data.iloc[:2].copy()
        for overlay in [base.iloc[::-1],pd.concat([base,base.iloc[:1]])]:
            with self.assertRaises(ValueError):
                overlay_features(base,overlay,self.scorer.fields)

    def test_08_observation_clock_is_current_not_asof(self):
        from datetime import datetime
        now=datetime.fromisoformat(clock_now())
        self.assertIsNotNone(now.tzinfo)
        self.assertEqual(now.utcoffset().total_seconds(),8*3600)


if __name__=='__main__':
    unittest.main()
