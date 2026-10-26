import unittest
import numpy as np
import pandas as pd

from decision_after_labels_v1 import minute_clock
from read_decision_after_labels_v1 import attach_label_times


class LogicalLabelTests(unittest.TestCase):
    def test_logical_end_and_eligibility_are_separate(self):
        days=[20250102,20250103,20250106,20250107,20250108,20250109]
        times=minute_clock(days)[14::15]
        frame=pd.DataFrame(dict(code=['000001','000001'],cutoff_index=[0,len(times)-1],
                                label_state=[0,1],proposed_segment=[0,0]))
        result=attach_label_times(frame,times,[dict(name='train')])
        self.assertEqual(result.label_end_time.iloc[0],20250109094500)
        self.assertTrue(pd.isna(result.label_end_time.iloc[1]))
        self.assertTrue(result.label_research_admitted.iloc[0])
        self.assertFalse(result.training_eligible.any())
        self.assertTrue(result.data_complete_time.isna().all())

    def test_bad_cutoff_rejected(self):
        frame=pd.DataFrame(dict(cutoff_index=[-1]))
        with self.assertRaisesRegex(ValueError,'INVALID_LABEL_CUTOFF'):
            attach_label_times(frame,np.array([20250102094500]),[])

    def test_clock_offset_rejected(self):
        frame=pd.DataFrame(dict(cutoff_index=[0]))
        with self.assertRaisesRegex(ValueError,'CLOCK_MISMATCH'):
            attach_label_times(frame,np.array([20250102094600]),[])


if __name__=='__main__':unittest.main()
