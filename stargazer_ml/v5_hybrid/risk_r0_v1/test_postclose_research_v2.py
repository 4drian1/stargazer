import unittest
import numpy as np
import pandas as pd
from prepare_postclose_research_v2 import complete_daily_close,family_masks


class DailyResearchTests(unittest.TestCase):
    def test_full_day_and_missing_bar(self):
        frame=pd.DataFrame(dict(code=['A']*32,cutoff_index=np.arange(32),stock_close=np.arange(32)+10.))
        np.testing.assert_allclose(complete_daily_close(frame,2)['A'],[25.,41.])
        frame=frame[frame.cutoff_index!=1]
        self.assertTrue(np.isnan(complete_daily_close(frame,2)['A'][0]))
        self.assertEqual(complete_daily_close(frame,2)['A'][1],41.)

    def test_duplicate_rejected(self):
        frame=pd.DataFrame(dict(code=['A','A'],cutoff_index=[15,15],stock_close=[10.,11.]))
        with self.assertRaises(ValueError):complete_daily_close(frame,1)

    def test_full_family_is_not_repaired_by_another_family(self):
        frame=pd.DataFrame(dict(stock_daily_rsi14=[20.],sse_daily_rsi14=[30.],cyb_daily_rsi14=[40.],
            context_sector_mean_return_loo_raw=[np.nan],context_current_leader_daily_rsi14_raw=[50.],
            context_captured_leader_daily_rsi14_raw=[np.nan],golden_position=[.3]))
        masks=family_masks(frame)
        self.assertFalse(masks['sector_daily_observation'].iloc[0])
        self.assertTrue(masks['leader_daily_observation'].iloc[0])


if __name__=='__main__':unittest.main()
