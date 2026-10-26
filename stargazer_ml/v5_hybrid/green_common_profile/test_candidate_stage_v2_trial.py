import unittest

import pandas as pd

from candidate_stage_v2_trial import long_base_trial, short_trial


class StageV2Tests(unittest.TestCase):
    def test_short_decline_exits_on_sixth_day(self):
        close = [10, 9, 8, 7, 6, 5]
        g = pd.DataFrame(dict(close=close, pre_close=[9]+close[:-1],
                              volume=[10, 9, 8, 7, 6, 5]))
        self.assertTrue(short_trial(g)['eligible'])
        sixth = pd.concat([g, pd.DataFrame([dict(close=4, pre_close=5, volume=4)])],
                          ignore_index=True)
        self.assertFalse(short_trial(sixth)['eligible'])
        g.loc[4, 'volume'] = 8
        self.assertFalse(short_trial(g)['eligible'])

    def test_long_base_needs_five_days_without_new_low(self):
        close = [20.] * 68 + [19.,18.,17.,16.,15.,14.,13.] + [13.,13.1,13.05,13.1,13.2]
        volume = [100.] * 68 + [90.,80.,70.,60.,55.,50.,45.] + [50.] * 5
        g = pd.DataFrame(dict(close=close, pre_close=[close[0]]+close[:-1],
                              high=[x+.2 for x in close],low=[x-.2 for x in close],
                              volume=volume,amount=[1e8]*80,is_st=[0]*80,float_mv=[1e9]*80))
        self.assertTrue(long_base_trial(g)['eligible'])
        g.loc[78, 'low'] = g.loc[75, 'low']-.01
        self.assertFalse(long_base_trial(g)['eligible'])


if __name__ == '__main__':
    unittest.main()
