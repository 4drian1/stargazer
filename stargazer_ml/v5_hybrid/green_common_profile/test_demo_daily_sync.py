"""Zero-network regressions for the desktop CLI's daily-sync preflight."""
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import gcp_demo_snapshot as snapshot

spec = importlib.util.spec_from_file_location(
    'pentacles_refresh_test', Path(__file__).resolve().parent.parent / 'green_v2/refresh_pool.py')
refresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refresh)


class DailySyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.calendar = self.root / 'calendar.json'
        self.calendar.write_text(json.dumps([20261009, 20261012, 20261013]))
        self.calpatch = patch.object(snapshot, 'calendar_file', return_value=self.calendar)
        self.calpatch.start()
        self.now = datetime(2026, 10, 12, 19)

    def tearDown(self):
        self.calpatch.stop()
        self.tmp.cleanup()

    def daily(self, date):
        pd.DataFrame({'date': [date]}).to_parquet(self.root / 'part_history.parquet')

    def test_current_does_not_contact_stockdb(self):
        self.daily(20261012)
        with patch.object(snapshot.subprocess, 'run') as run:
            snapshot.ensure_daily_current(self.root, self.now)
            run.assert_not_called()

    def test_stale_calls_only_daily_sync_with_correct_directory(self):
        self.daily(20261009)
        result = subprocess.CompletedProcess([], 0, b'[done]', b'')
        with patch.object(snapshot.subprocess, 'run', return_value=result) as run:
            snapshot.ensure_daily_current(self.root, self.now)
        args, kwargs = run.call_args
        self.assertEqual(args[0][-3:], ['--daily-only', '--through-date', '20261012'])
        self.assertEqual(kwargs['env']['STARGAZER_DAILY_PARTS'], str(self.root))
        self.assertTrue(kwargs['capture_output'])

    def test_failed_sync_is_visible_and_preserves_history(self):
        self.daily(20261009)
        original = (self.root / 'part_history.parquet').read_bytes()
        result = subprocess.CompletedProcess([], 1, b'', b'source unavailable')
        with patch.object(snapshot.subprocess, 'run', return_value=result):
            with self.assertRaisesRegex(ValueError, '自动同步失败'):
                snapshot.ensure_daily_current(self.root, self.now)
        self.assertEqual((self.root / 'part_history.parquet').read_bytes(), original)

    def test_intraday_rejected_before_sync(self):
        self.daily(20261009)
        with patch.object(snapshot.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, '收盘后'):
                snapshot.ensure_daily_current(self.root, datetime(2026, 10, 12, 14))
            run.assert_not_called()

    def test_daily_only_never_rebuilds_old_pool(self):
        with patch.object(refresh, 'latest_stockdb_date', return_value=20261012), \
                patch.object(refresh, 'latest_parts_date', return_value=20261012), \
                patch.object(subprocess, 'run') as run:
            refresh.main(daily_only=True, through_date=20261012)
            run.assert_not_called()

    def test_source_outdated_does_not_fetch(self):
        with patch.object(refresh, 'latest_stockdb_date', return_value=20261009), \
                patch.object(refresh, 'fetch_day') as fetch:
            with self.assertRaises(SystemExit):
                refresh.main(daily_only=True, through_date=20261012)
            fetch.assert_not_called()

    def test_failed_symbol_never_publishes_partial_partition(self):
        def get(key):
            if key == '股票代码':
                return {'stocks': ['600000', '600001']}
            if '600001' in key:
                raise ConnectionError('offline')
            record = {f: 1 for f in refresh.FIELDS}
            record.update(date=20261012, code='600000', name='test', is_st=False)
            return {'data': [['key', record]]}

        with patch.object(refresh, 'PARTS', self.root), patch.object(refresh, 'get', side_effect=get), \
                patch.object(refresh._t, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, '请求失败'):
                refresh.fetch_day('20261012')
        self.assertFalse((self.root / 'part_incr_20261012.parquet').exists())

    def test_success_is_atomic_and_existing_partition_not_overwritten(self):
        record = {f: 1 for f in refresh.FIELDS}
        record.update(date=20261012, code='600000', name='test', is_st=False)
        with patch.object(refresh, 'PARTS', self.root), patch.object(
                refresh, 'get', side_effect=[{'stocks': ['600000']}, {'data': [['key', record]]}]):
            refresh.fetch_day('20261012')
        out = self.root / 'part_incr_20261012.parquet'
        original = out.read_bytes()
        self.assertFalse(out.with_suffix('.parquet.tmp').exists())
        with patch.object(refresh, 'PARTS', self.root), patch.object(refresh, 'get') as get:
            refresh.fetch_day('20261012')
            get.assert_not_called()
        self.assertEqual(out.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
