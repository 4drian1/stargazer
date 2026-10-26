"""Bound TDX source sync; real parser and mirrors, no network."""
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import index_minutes_v1 as service


def binary(day=20261016, price=100.):
    year, month, date = day // 10000, day // 100 % 100, day % 100
    packed = (year - 2004) * 2048 + month * 100 + date
    minutes = list(range(575, 691, 5)) + list(range(785, 901, 5))
    return b''.join(struct.pack('<HHfffffiI', packed, m, price, price+1,
                               price-1, price, 1000., 100, 0) for m in minutes)


class DirectSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sources = {s: self.root / (s + '.lc5') for s in service.SOURCE_FILES}
        for source in self.sources.values():
            source.write_bytes(binary())
        self.updater = service._load_updater()
        self.updater.MIRROR = self.root
        self.patches = [patch.object(service, 'MIRROR', self.root),
                        patch.object(service, 'SOURCE_FILES', self.sources),
                        patch.object(service, '_last_success', None),
                        patch.object(service, '_load_updater', return_value=self.updater),
                        patch.object(self.updater, 'trading_days_via_stockdb',
                                     return_value=[20261021, 20261020, 20261019, 20261016])]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def test_success_read_only_and_unchanged(self):
        before = [p.read_bytes() for p in self.sources.values()]
        self.assertEqual(service.sync_once()['status'], 'OK')
        self.assertEqual(before, [p.read_bytes() for p in self.sources.values()])
        self.assertEqual(service.sync_once()['status'], 'UNCHANGED')
        self.assertTrue(service.get_index_freshness()['stale'])
        self.assertEqual(service.get_index_freshness()['last_date'], 20261016)

    def test_missing_source_does_not_use_old_export(self):
        next(iter(self.sources.values())).unlink()
        self.assertEqual(service.sync_once()['status'], 'ERROR')
        self.assertFalse((self.root / 'sse_5min.parquet').exists())

    def test_incomplete_source_rejected_before_either_write(self):
        list(self.sources.values())[1].write_bytes(binary()[:-32])
        result = service.sync_once()
        self.assertIn('SOURCE_VALIDATION_FAILED', result['error'])
        self.assertFalse((self.root / 'sse_5min.parquet').exists())

    def test_overlap_conflict_keeps_mirrors(self):
        self.assertEqual(service.sync_once()['status'], 'OK')
        before = (self.root / 'sse_5min.parquet').read_bytes()
        next(iter(self.sources.values())).write_bytes(binary(price=101.))
        result = service.sync_once()
        self.assertIn('OVERLAP_CONFLICT', result['error'])
        self.assertEqual(before, (self.root / 'sse_5min.parquet').read_bytes())
        self.assertTrue(service.get_index_freshness()['stale'])

    def test_retry_after_error(self):
        source = next(iter(self.sources.values()))
        source.write_bytes(b'bad')
        self.assertEqual(service.sync_once()['status'], 'ERROR')
        source.write_bytes(binary())
        self.assertEqual(service.sync_once()['status'], 'OK')

    def test_changed_during_read_rejected(self):
        original = self.updater.parse_lc5
        def changing(body):
            next(iter(self.sources.values())).write_bytes(binary() + b'x')
            return original(body)
        with patch.object(self.updater, 'parse_lc5', side_effect=changing):
            result = service.sync_once()
        self.assertIn('SOURCE_CHANGED', result['error'])
        self.assertFalse((self.root / 'sse_5min.parquet').exists())

    def test_periodic_worker_started_only_once(self):
        with patch.object(service, '_worker', None), patch.object(service.threading, 'Thread') as thread:
            thread.return_value.is_alive.return_value = True
            service.start_source_sync()
            service.start_source_sync()
            self.assertEqual(thread.call_count, 1)
            thread.return_value.start.assert_called_once()


if __name__ == '__main__':
    unittest.main()
