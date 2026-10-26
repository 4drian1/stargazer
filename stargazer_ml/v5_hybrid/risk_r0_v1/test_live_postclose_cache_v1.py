import json
from pathlib import Path
import tempfile
import unittest
from live_postclose_cache_v1 import prior_entries,remaining_periods


class IncrementalCacheTests(unittest.TestCase):
    def test_new_trading_dates_only(self):
        self.assertEqual(remaining_periods(20261016,[20261016,20261019,20261020]),
                         ['20261019*','20261020*'])

    def test_only_completed_past_days_can_seed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for day,status in [(20261020,'COMPLETE'),(20261021,'FAILED'),(20261022,'COMPLETE')]:
                run=root/f'run_{day}_test';run.mkdir()
                (run/'terminal.json').write_text(json.dumps(dict(data_date=day,status=status,run_dir=str(run))))
                (run/'source_manifest.json').write_text(json.dumps({'entries':{'000001':{'seed_day':day}}}))
            self.assertEqual(prior_entries(root,20261021)['000001']['seed_day'],20261020)
            self.assertEqual(prior_entries(root,20261020),{})


if __name__=='__main__':
    unittest.main()
