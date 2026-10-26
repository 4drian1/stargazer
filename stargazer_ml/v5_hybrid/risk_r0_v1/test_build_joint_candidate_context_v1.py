import unittest
import numpy as np
import pandas as pd
from build_joint_candidate_context_v1 import excluded_means,candidate_context


class ExclusionTests(unittest.TestCase):
    def test_exclude_own_stock_from_board(self):
        mean,n=excluded_means(np.array([.1]),np.array([2]),np.array([.1]),np.array([True]))
        self.assertEqual(mean[0],0)
        self.assertEqual(n[0],1)

    def test_only_candidate_is_empty_not_zero(self):
        mean,n=excluded_means(np.array([.1]),np.array([1]),np.array([.1]),np.array([True]))
        self.assertTrue(np.isnan(mean[0]))
        self.assertEqual(n[0],0)

    def test_candidate_outside_board_not_subtracted(self):
        mean,n=excluded_means(np.array([.1]),np.array([2]),np.array([.1]),np.array([False]))
        self.assertEqual(mean[0],.05)
        self.assertEqual(n[0],2)

    def test_missing_own_return_not_subtracted(self):
        mean,n=excluded_means(np.array([.1]),np.array([2]),np.array([np.nan]),np.array([True]))
        self.assertEqual(mean[0],.05)
        self.assertEqual(n[0],2)

    def test_negative_count_rejected(self):
        with self.assertRaisesRegex(ValueError,'NEGATIVE'):
            excluded_means(np.array([0.]),np.array([0]),np.array([.1]),np.array([True]))

    def test_actual_context_self_leader_and_single_member(self):
        frame=pd.DataFrame(dict(code=['000001'],cutoff_index=[0],stock_day_return_raw=[.1]))
        leaders=pd.DataFrame(dict(leader_observed_stock=['000001'],leader_observed_stock_return_raw=[.1],
            day=[20250102],captured_stock_count_today=[1],captured_stock_missing_today=[0],
            captured_stock_mean_return_raw=[.1],board_index=[0]))
        out=candidate_context(frame,np.array([[.1]]),np.array([[1]]),np.array([1]),
                              {'000001':[0]},leaders,{('000001',20250102):0})
        for col in ['context_sector_mean_return_loo_raw','context_leader_board_return_loo_raw',
                    'context_current_leader_return_loo_raw','context_captured_leader_mean_return_loo_raw']:
            self.assertTrue(pd.isna(out.iloc[0][col]),col)
        self.assertTrue(out.iloc[0].context_current_leader_is_candidate)

    def test_capture_after_cutoff_does_not_exclude_past(self):
        frame=pd.DataFrame(dict(code=['000001'],cutoff_index=[0],stock_day_return_raw=[.1]))
        leaders=pd.DataFrame(dict(leader_observed_stock=['000002'],leader_observed_stock_return_raw=[.2],
            day=[20250102],captured_stock_count_today=[1],captured_stock_missing_today=[0],
            captured_stock_mean_return_raw=[.2],board_index=[0]))
        out=candidate_context(frame,np.array([[.3]]),np.array([[2]]),np.array([2]),
                              {'000001':[0]},leaders,{('000001',20250102):1})
        self.assertEqual(out.iloc[0].context_captured_leader_mean_return_loo_raw,.2)

    def test_common_member_momentum_and_breadth_exclude_candidate(self):
        frame=pd.DataFrame(dict(code=['000001'],cutoff_index=[1],stock_day_return_raw=[.1]))
        leaders=pd.DataFrame(dict(leader_observed_stock=['000002','000002'],leader_observed_stock_return_raw=[.2,.21],
            day=[20250102,20250102],captured_stock_count_today=[1,1],captured_stock_missing_today=[0,0],
            captured_stock_mean_return_raw=[.2,.21],board_index=[0,0]))
        extras=dict(up_totals=np.array([[1.,2.]]),common_totals=np.array([[0.,.11]]),
            common_counts=np.array([[0,2]]),returns=np.array([[0.,.1],[.2,.21]]),
            code_index={'000001':0,'000002':1},current_leader_metrics=np.ones((2,4))*50,
            current_leader_change=np.array([np.nan,.01]),captured_change_total=np.array([0.,.01]),
            captured_change_count=np.array([0,1]))
        out=candidate_context(frame,np.array([[.2,.31]]),np.array([[2,2]]),np.array([2]),
            {'000001':[0]},leaders,{('000002',20250102):0},extras)
        self.assertAlmostEqual(out.iloc[0].context_sector_common_change_loo_raw,.01)
        self.assertEqual(out.iloc[0].context_sector_breadth_loo,1.)
        self.assertAlmostEqual(out.iloc[0].context_current_leader_return_change_loo_raw,.01)
        self.assertAlmostEqual(out.iloc[0].context_captured_leader_common_change_loo_raw,.01)


class SharedStorageTests(unittest.TestCase):
    def test_lossless_context_storage_including_null(self):
        from build_joint_candidate_context_v1 import context_bytes
        import pyarrow as pa
        import pyarrow.parquet as pq
        frame=pd.DataFrame(dict(code=['000001','000001'],cutoff_index=[0,1],
                                value=np.array([.12345,np.nan],dtype='float32'),flag=[True,False]))
        restored=pq.read_table(pa.BufferReader(context_bytes(frame))).to_pandas()
        pd.testing.assert_frame_equal(frame,restored)

    def test_arrow_missing_leader_is_not_a_code(self):
        from build_joint_candidate_context_v1 import known_leader
        for value in [None,np.nan,pd.NA,'']:
            self.assertFalse(known_leader(value))
        self.assertTrue(known_leader('000001'))

    def test_compact_and_wide_have_same_logical_context(self):
        from build_joint_candidate_context_v1 import attach_shared_context
        compact=pd.DataFrame(dict(code=['000001'],cutoff_index=[0],stock_close=[10.]))
        shared=pd.DataFrame(dict(cutoff_index=[0],decision_ts_assumed_end=[20250102094500],sse_close=[3200.]))
        logical=attach_shared_context(compact,shared)
        self.assertEqual(logical.iloc[0].sse_close,3200.)
        self.assertFalse(logical.iloc[0].joint_training_admitted)
        self.assertTrue(np.isnan(logical.iloc[0].index_volume_feature))
        pd.testing.assert_frame_equal(attach_shared_context(logical,shared),logical)

    def test_shared_cutoff_duplicates_rejected(self):
        from build_joint_candidate_context_v1 import attach_shared_context
        compact=pd.DataFrame(dict(code=['000001'],cutoff_index=[0]))
        shared=pd.DataFrame(dict(cutoff_index=[0,0],decision_ts_assumed_end=[1,2]))
        with self.assertRaises(pd.errors.MergeError):attach_shared_context(compact,shared)


if __name__=='__main__':unittest.main()
