import unittest
import numpy as np
import pandas as pd

from decision_after_labels_v1 import minute_clock,price_grid,aligned_days,label_rows,proposed_split


class FutureLabelTests(unittest.TestCase):
    def setUp(self):
        self.days=np.array([20250102,20250103,20250106,20250107,20250108,20250109,20250110])
        self.clock=minute_clock(self.days)
        self.grid=np.tile([100.,101.,100.,100.],(len(self.clock),1))
        self.row=pd.DataFrame(dict(code=['000001'],cutoff_index=[0],stock_close=[100.]))
        self.evidence=dict(aligned=np.ones(7,dtype=bool),issues=np.zeros(7,dtype=bool),factors=np.ones(7))

    def result(self,grid=None,evidence=None):
        return label_rows(self.row,self.grid if grid is None else grid,self.days,evidence)[0].iloc[0]

    def test_decision_bar_low_excluded(self):
        self.grid[14,2]=1
        self.assertEqual(self.result().diagnostic_tail_raw_1200m,0)

    def test_first_future_bar_included_and_boundary(self):
        self.grid[15,2]=92
        self.assertEqual(self.result(evidence=self.evidence).y_tail_1200m,1)

    def test_1200th_included_1201st_excluded(self):
        self.grid[1214,2]=92
        self.assertEqual(self.result().diagnostic_tail_raw_1200m,1)
        self.grid[1214,2]=100;self.grid[1215,2]=1
        self.assertEqual(self.result().diagnostic_tail_raw_1200m,0)

    def test_missing_slot_even_when_tail_already_seen(self):
        self.grid[15,2]=80;self.grid[20]=np.nan
        result=self.result(evidence=self.evidence)
        self.assertEqual(result.label_state,3);self.assertTrue(pd.isna(result.y_tail_1200m))
        self.assertEqual(result.diagnostic_tail_raw_1200m,-1)

    def test_future_price_after_window_has_no_effect(self):
        before=self.result();self.grid[1215:]=1
        pd.testing.assert_series_equal(before,self.result())

    def test_lunch_and_overnight_are_not_slots(self):
        self.assertEqual(self.clock[119],20250102113000)
        self.assertEqual(self.clock[120],20250102130100)
        self.assertEqual(self.clock[240],20250103093100)
        self.assertEqual(self.clock[14+1200],20250109094500)

    def test_raw_is_only_diagnostic_without_evidence(self):
        self.grid[15,2]=90;result=self.result()
        self.assertEqual(result.diagnostic_tail_raw_1200m,1)
        self.assertEqual(result.label_state,4);self.assertTrue(pd.isna(result.y_tail_1200m))

    def test_normalization_removes_artificial_split_tail(self):
        self.grid[240:,:]/=2;self.evidence['factors'][1:]=2
        result=self.result(evidence=self.evidence)
        self.assertEqual(result.diagnostic_tail_raw_1200m,1)
        self.assertEqual(result.y_tail_1200m,0)

    def test_real_tail_survives_normalization(self):
        self.grid[240:,:]/=2;self.grid[250,2]=45;self.evidence['factors'][1:]=2
        self.assertEqual(self.result(evidence=self.evidence).y_tail_1200m,1)

    def test_partial_daily_basis_blocks_label(self):
        self.evidence['aligned'][1]=False
        result=self.result(evidence=self.evidence)
        self.assertEqual(result.label_state,5);self.assertTrue(pd.isna(result.y_tail_1200m))

    def test_unresolved_action_blocks_label(self):
        self.evidence['issues'][1]=True
        self.assertEqual(self.result(evidence=self.evidence).label_state,6)

    def test_source_tail_not_negative(self):
        self.grid[300:]=np.nan
        result=self.result();self.assertEqual(result.label_state,2)
        self.assertEqual(result.diagnostic_tail_raw_1200m,-1)

    def test_calendar_tail_not_negative(self):
        self.row['cutoff_index']=len(self.clock)//15-1
        result=self.result();self.assertEqual(result.label_state,1)
        self.assertTrue(pd.isna(result.y_tail_1200m))

    def test_reference_change_is_hard_failure(self):
        self.row['stock_close']=99
        with self.assertRaisesRegex(ValueError,'REFERENCE'):self.result()

    def test_invalid_factor_rejected(self):
        self.evidence['factors'][1]=np.inf
        with self.assertRaisesRegex(ValueError,'INVALID_FACTOR'):self.result(evidence=self.evidence)

    def test_grid_duplicate_conflict_and_quarantine(self):
        frame=pd.DataFrame(dict(code=['000001']*3,date=[self.clock[0],self.clock[0],self.clock[240]],
            open=[100,100,100],high=[101,101,101],low=[99,99,99],close=[100,100.1,100],volume=[1,1,1],amount=[1,1,1]))
        grid=price_grid(frame,'000001',self.clock,{20250103})
        self.assertTrue(np.isnan(grid[[0,240]]).all())

    def test_alignment_is_positive_evidence_not_exclusion_list(self):
        bars={int(d):dict(open=100,high=101,low=100,close=100) for d in self.days}
        self.assertTrue(aligned_days(self.grid,self.days,bars).all())
        bars[int(self.days[0])]['close']=99
        self.assertFalse(aligned_days(self.grid,self.days,bars)[0])

    def test_fold_boundaries_and_equal_end_purged(self):
        segments=[dict(name='train',start=20250101,end=20251231),dict(name='test',start=20260101,end=20261231)]
        assignment,purged=proposed_split(np.array([20251230150000]*2),np.array([20260101000000.,20251231150000.]),segments)
        self.assertEqual(assignment.tolist(),[0,0]);self.assertEqual(purged.tolist(),[True,False])


if __name__=='__main__':unittest.main()
