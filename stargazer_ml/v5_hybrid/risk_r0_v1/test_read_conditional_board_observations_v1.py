import unittest

import numpy as np
import pandas as pd

from read_conditional_board_observations_v1 import attach, build


class ReaderTests(unittest.TestCase):
    def test_shared_replicates_without_changing_old_fields(self):
        frame=pd.DataFrame({'code':['A','B'],build.TIME:[1,1],'mb5_mean':[.1,.1],'label':[0,1]})
        diff=pd.DataFrame({'code':['__SHARED__'],build.TIME:[1],'feature':['mb5_mean'],
                           'old_value':[.1],'new_value':[.2]})
        result=attach(frame,diff)
        pd.testing.assert_frame_equal(result[list(frame)],frame)
        np.testing.assert_allclose(result.ar_mb5_mean,.2)

    def test_own_patch_does_not_change_other_candidate(self):
        frame=pd.DataFrame({'code':['A','B'],build.TIME:[1,1],'own5_mean':[np.nan,.1]})
        diff=pd.DataFrame({'code':['A'],build.TIME:[1],'feature':['own5_mean'],
                           'old_value':[np.nan],'new_value':[.2]})
        np.testing.assert_allclose(attach(frame,diff).ar_own5_mean,[.2,.1])

    def test_base_drift_refused(self):
        frame=pd.DataFrame({'code':['A'],build.TIME:[1],'own5_mean':[.3]})
        diff=pd.DataFrame({'code':['A'],build.TIME:[1],'feature':['own5_mean'],
                           'old_value':[.1],'new_value':[.2]})
        with self.assertRaisesRegex(ValueError,'BASE_VALUE_CHANGED'):
            attach(frame,diff)

    def test_duplicate_and_field_collision_refused(self):
        frame=pd.DataFrame({'code':['A'],build.TIME:[1],'own5_mean':[.1],'ar_own5_mean':[.2]})
        diff=pd.DataFrame({'code':['A'],build.TIME:[1],'feature':['own5_mean'],
                           'old_value':[.1],'new_value':[.2]})
        with self.assertRaisesRegex(ValueError,'COLLISION'):
            attach(frame,diff)
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):
            attach(frame,pd.concat([diff,diff]))


if __name__=='__main__':
    unittest.main()
