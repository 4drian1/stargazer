"""Latest-day optimization must preserve frozen rule results, including NaNs."""
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from gcp_v0_1_pools import process_code
from gcp_demo_pools import process_latest_code


class LatestPoolTests(unittest.TestCase):
    def test_full_history_latest_day_equivalence(self):
        for seed in range(40):
            rng = np.random.default_rng(seed)
            n = 500
            close = 10 * np.exp(np.cumsum(rng.normal(-.002, .025, n)))
            volume = rng.integers(100, 10000, n).astype(float)
            if seed % 3 == 0:
                volume[-65:] = 100
            if seed % 4 == 0:
                volume[150] = np.nan
            if seed % 5 == 0:
                volume[200] = 0
            if seed % 7 == 0:
                volume[-2] = np.inf
            g = pd.DataFrame(dict(date=np.arange(n), code='600000', close=close,
                open=close * 1.01, high=close * 1.04, low=close * .96,
                pre_close=np.r_[close[0], close[:-1]], volume=volume,
                amount=1e9, is_st=0, float_mv=1e9))
            expected = process_code(g)
            actual = process_latest_code(g)
            for old, new in zip(expected, actual):
                old = old[old.date == n - 1] if not old.empty else old
                if old.empty and new.empty:
                    continue
                assert_frame_equal(old.reset_index(drop=True), new.reset_index(drop=True))

    def test_short_history_remains_empty(self):
        self.assertTrue(all(frame.empty for frame in process_latest_code(pd.DataFrame(index=range(69)))))


if __name__ == '__main__':
    unittest.main()
