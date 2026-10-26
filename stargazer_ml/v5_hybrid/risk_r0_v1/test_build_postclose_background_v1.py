import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import build_postclose_background_v1 as bg
from postclose_research_scorer_v1 import ResearchScorer,encode
from postclose_local_source_bridge_v1 import load_local_features
from postclose_daily_job_v1 import background_fields,read_input
from read_postclose_incremental_v2 import load_refreshed


class BackgroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer=ResearchScorer();cls.fields=background_fields(cls.scorer.fields)
        cls.history={s:bg.aggregate_days(bg.read_binary(p.read_bytes())) for s,p in bg.SOURCES.items()}
        cls.frame=bg.compute_background(cls.history,cls.fields)

    def test_01_actual_all_days_fields_match_training_input(self):
        expected,_=load_local_features(self.scorer)
        expected=expected.drop_duplicates('day').set_index('day')[self.fields]
        np.testing.assert_allclose(self.frame.set_index('day').loc[expected.index,self.fields],expected,
                                   rtol=0,atol=1e-12,equal_nan=True)
        self.assertEqual(len(self.fields),78)

    def test_02_prefix_reconstruction(self):
        for end in [20,220,len(self.frame)]:
            prefix={s:rows[:end] for s,rows in self.history.items()}
            actual=bg.compute_background(prefix,self.fields)
            pd.testing.assert_frame_equal(actual,self.frame.iloc[:end].reset_index(drop=True))

    def test_03_future_price_amount_perturbation(self):
        cutoff=220;future=copy.deepcopy(self.history)
        for rows in future.values():
            for row in rows[cutoff:]:
                for key in ['open','high','low','close','amount']:row[key]*=1.2
        actual=bg.compute_background(future,self.fields)
        pd.testing.assert_frame_equal(actual.iloc[:cutoff],self.frame.iloc[:cutoff])

    def test_04_missing_duplicate_bar_not_complete(self):
        rows=bg.read_binary(next(iter(bg.SOURCES.values())).read_bytes())[:48]
        for changed in [rows[:-1],rows+[rows[0]]]:
            with self.assertRaisesRegex(ValueError,'INCOMPLETE_OR_DUPLICATE'):bg.aggregate_days(changed)

    def test_05_invalid_price_rejected(self):
        rows=bg.read_binary(next(iter(bg.SOURCES.values())).read_bytes())[:48]
        for value in [0,np.nan,np.inf]:
            changed=copy.deepcopy(rows);changed[0]['close']=value
            with self.assertRaises(ValueError):bg.aggregate_days(changed)

    def test_06_bad_amount_unknown_not_fake_volume(self):
        rows=bg.read_binary(next(iter(bg.SOURCES.values())).read_bytes())[:48]
        rows[0]['amount']=-1
        self.assertIsNone(bg.aggregate_days(rows)[0]['amount'])

    def test_07_rolling_file_does_not_reset_indicators(self):
        merged={s:bg.merge_history(rows,rows[100:],20261016) for s,rows in self.history.items()}
        pd.testing.assert_frame_equal(bg.compute_background(merged,self.fields),self.frame)

    def test_08_overlap_change_rejected(self):
        seed=self.history['000001.SH'];changed=copy.deepcopy(seed[-3:]);changed[0]['close']+=1
        with self.assertRaisesRegex(ValueError,'HISTORY_CHANGED'):bg.merge_history(seed,changed,20261016)

    def test_09_no_overlap_is_not_verified_identity(self):
        seed=self.history['000001.SH'];new=[dict(seed[-1],day=20261019)]
        with self.assertRaisesRegex(ValueError,'OVERLAP'):bg.merge_history(seed,new,20261019)

    def test_10_two_indices_need_same_session_grid(self):
        h=dict(self.history);h['399006.SZ']=h['399006.SZ'][:-1]
        with self.assertRaisesRegex(ValueError,'GRID_MISMATCH'):bg.compute_background(h,self.fields)

    def test_11_no_seed_corruption(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);(root/'daily_seed.json').write_bytes(b'{}')
            (root/'manifest.json').write_bytes(encode(dict(daily_seed_sha256='wrong')))
            with patch.object(bg,'SEED',root),self.assertRaisesRegex(ValueError,'SEED_HASH_DRIFT'):
                bg.seed_history()

    def test_12_real_background_input_contract_and_missing_day(self):
        background,calendar,_,_=bg.background_for_day(20261016)
        self.assertEqual(set(background['features']),set(self.fields))
        self.assertIn('OBSERVED',calendar['verification'])
        self.assertIsNone(background['original_available_at'])
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'background.json';path.write_bytes(encode(background))
            actual,_=read_input(path,'POSTCLOSE_MARKET_GOLDEN')
            self.assertEqual(actual['day'],20261016)
        with self.assertRaisesRegex(ValueError,'TARGET_DAY_NOT_IN_SOURCE_NO_FALLBACK'):
            bg.background_for_day(20261019)

    def test_13_rebuilt_background_reproduces_saved_scores(self):
        frame,scorer=load_refreshed();original=frame[frame.day.eq(20261016)].copy()
        from postclose_local_source_bridge_v1 import latest_day_payload
        before=scorer.score(latest_day_payload(original,scorer)[0])
        updated=original.copy();row=self.frame[self.frame.day.eq(20261016)].iloc[0]
        for f in self.fields:updated[f]=row[f]
        after=scorer.score(latest_day_payload(updated,scorer)[0])
        self.assertEqual(before['rows'],after['rows'])
        self.assertEqual(sum(r['status']=='RESEARCH_SCORE' for r in after['rows']),39)


if __name__=='__main__':unittest.main()
