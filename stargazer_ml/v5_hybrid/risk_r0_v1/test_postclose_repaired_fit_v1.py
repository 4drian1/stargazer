import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import pandas as pd
import postclose_repaired_training_v1 as training
from compare_postclose_repaired_fit_v1 import partitions


class RepairedFitTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(code=['000001','000002'],day=[20260101,20260101],
            label_end_time=[20260102150000]*2,next_session_adverse_excursion=[.01,.02]))

    def test_authorization_required_before_data_or_fit(self):
        with patch.object(training,'prepare',side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError,'AUTHORIZATION_REQUIRED'): training.train()

    def test_existing_output_refused_before_data_or_fit(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(training,'OUT',Path(folder)), patch.object(training,'prepare',side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError,'NO_OVERWRITE'): training.train(True)

    def test_common_added_removed_partition(self):
        old=self.frame(); new=old.copy(); new.loc[1,'code']='000003'
        common,added,removed=partitions(old,new)
        self.assertEqual((len(common),len(added),len(removed)),(1,1,1))
        self.assertEqual(added.code.iloc[0],'000003')
        self.assertEqual(removed.code.iloc[0],'000002')

    def test_common_target_drift_refused(self):
        old=self.frame(); new=old.copy(); new.loc[0,'next_session_adverse_excursion']=.03
        with self.assertRaisesRegex(ValueError,'TARGET_OR_END_DRIFT'): partitions(old,new)

    def test_duplicate_keys_refused(self):
        frame=self.frame()
        with self.assertRaisesRegex(ValueError,'DUPLICATE_COMPARE_KEYS'): partitions(pd.concat([frame,frame]),frame)


if __name__=='__main__': unittest.main()
