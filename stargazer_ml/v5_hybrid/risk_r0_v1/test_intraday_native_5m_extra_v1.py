import unittest

import numpy as np
import pandas as pd

from build_joint_risk_features_v1 import stock_features
from intraday_relative_stall_v2 import basket, minute_returns
from test_intraday_relative_stall_v2 import bars


class NativeFiveMinuteTests(unittest.TestCase):
    def test_native_fifteen_matches_existing_stock_features(self):
        first, second = bars(day=20250102), bars(day=20250103)
        second['close'], second['high'] = 102., 103.
        frame = pd.concat([first, second], ignore_index=True)
        _, expected, _ = stock_features(frame, 'X', np.array([20250102, 20250103]), set(), returns_only=True)
        actual, _ = minute_returns(frame, 'X', [20250102, 20250103], {20250103: 101.})
        np.testing.assert_allclose(expected, actual[15], rtol=0, atol=1e-12, equal_nan=True)

    def test_same_member_count_does_not_certify_same_members(self):
        got = basket(np.array([[.1, np.nan], [np.nan, .2]]), [0, 1], 5)
        self.assertEqual(list(got['present']), [1, 1])
        self.assertTrue(np.isnan(got['common_change'][1]))
        self.assertFalse(got['full_members'].any())


if __name__ == '__main__':
    unittest.main()
