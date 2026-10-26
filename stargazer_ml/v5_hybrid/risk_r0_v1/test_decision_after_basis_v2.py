import unittest
import numpy as np
import pandas as pd

from decision_after_labels_v1 import aligned_days,minute_clock
from decision_after_basis_v2 import native_aligned_days
from read_decision_after_labels_v2 import apply_correction


class BasisCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.days=np.array([20250102]);self.grid=np.tile([100.,101.,99.,100.],(240,1))
        self.native=pd.DataFrame(dict(date=[20250102093000],open=[102.],high=[103.],low=[100.],close=[101.]))
        self.bars={20250102:dict(open=102.,high=103.,low=99.,close=100.)}

    def test_native_0930_corrects_false_basis_rejection(self):
        self.assertFalse(aligned_days(self.grid,self.days,self.bars)[0])
        self.assertTrue(native_aligned_days(self.grid,self.days,self.bars,self.native)[0])

    def test_clock_and_target_grid_not_changed(self):
        clock=minute_clock(self.days);before=self.grid.copy()
        native_aligned_days(self.grid,self.days,self.bars,self.native)
        np.testing.assert_array_equal(before,self.grid)
        self.assertEqual(clock[0],20250102093100);self.assertEqual(len(clock),240)

    def test_real_close_mismatch_still_rejected(self):
        self.bars[20250102]['close']=98
        self.assertFalse(native_aligned_days(self.grid,self.days,self.bars,self.native)[0])

    def test_native_conflict_rejected(self):
        self.native=pd.concat([self.native,self.native.assign(close=102)],ignore_index=True)
        self.assertFalse(native_aligned_days(self.grid,self.days,self.bars,self.native)[0])

    def test_no_boundary_record_does_not_invent_one(self):
        self.assertFalse(native_aligned_days(self.grid,self.days,self.bars,self.native.iloc[:0])[0])

    def test_only_formal_columns_overridden(self):
        frame=pd.DataFrame(dict(code=['000001'],cutoff_index=[0],mae_1200m=[np.nan],
            y_tail_1200m=pd.array([None],dtype='Int8'),label_state=np.array([5],dtype='uint8'),
            diagnostic_mae_raw_1200m=[-.1]))
        patch=frame[['code','cutoff_index','mae_1200m','y_tail_1200m','label_state']].copy()
        patch['mae_1200m']=-.1;patch['y_tail_1200m']=pd.array([1],dtype='Int8');patch['label_state']=0
        result=apply_correction(frame,patch)
        self.assertEqual(result.diagnostic_mae_raw_1200m.iloc[0],-.1)
        self.assertEqual(result.y_tail_1200m.iloc[0],1)
        self.assertEqual(frame.label_state.iloc[0],5)

    def test_outside_population_correction_rejected(self):
        frame=pd.DataFrame(dict(code=['000001'],cutoff_index=[0]))
        with self.assertRaisesRegex(ValueError,'OUTSIDE_POPULATION'):
            apply_correction(frame,pd.DataFrame(dict(code=['000002'],cutoff_index=[0])))


if __name__=='__main__':unittest.main()
