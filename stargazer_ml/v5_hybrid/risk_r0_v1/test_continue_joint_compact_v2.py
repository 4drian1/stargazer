import tempfile
import unittest
from pathlib import Path

import pandas as pd

from continue_joint_compact_v2 import completed_codes,calendar_days


class CompactContinuationTests(unittest.TestCase):
    def test_days_support_source_builder_vector_indexing(self):
        days=calendar_days(pd.DataFrame({'decision_ts_assumed_end':[20250103094500,20250102094500]}))
        self.assertEqual(days[[1,0]].tolist(),[20250103,20250102])

    def test_whole_codes_reused_not_rebuilt(self):
        with tempfile.TemporaryDirectory() as folder:
            paths=[Path(folder)/f'{i}.parquet' for i in range(2)]
            pd.DataFrame({'code':['000001','000001']}).to_parquet(paths[0])
            pd.DataFrame({'code':['000002']}).to_parquet(paths[1])
            self.assertEqual(completed_codes(paths),{'000001','000002'})

    def test_code_spanning_parts_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            paths=[Path(folder)/f'{i}.parquet' for i in range(2)]
            for path in paths:pd.DataFrame({'code':['000001']}).to_parquet(path)
            with self.assertRaisesRegex(ValueError,'CODE_SPANS'):
                completed_codes(paths)


if __name__=='__main__':unittest.main()
