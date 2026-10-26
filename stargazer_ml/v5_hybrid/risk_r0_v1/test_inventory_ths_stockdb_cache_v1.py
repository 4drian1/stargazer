import tempfile
from pathlib import Path
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from inventory_ths_stockdb_cache_v1 import board_file_coverage, footer_info


class CacheInventoryTests(unittest.TestCase):
    def test_missing_members_do_not_shrink_universe(self):
        boards = [dict(code='x', name='主题', symbols=['000001', '000002'])]
        row = board_file_coverage(boards, {'000001'})[0]
        self.assertEqual((row['members'], row['file_candidates'], row['file_missing']), (2, 1, 1))
        self.assertFalse(row['ranking_admitted'])

    def test_files_for_all_members_are_not_slot_or_history_proof(self):
        row = board_file_coverage([dict(code='x', name='主题', symbols=['000001'])], {'000001'})[0]
        self.assertEqual(row['file_candidate_ratio'], 1)
        self.assertFalse(row['slot_coverage_verified'])
        self.assertFalse(row['historical_membership_verified'])

    def test_shared_members_not_counted_as_independent_stocks(self):
        boards = [dict(code=c, name=c, symbols=['000001']) for c in ['x', 'y']]
        self.assertEqual(len(board_file_coverage(boards, {'000001'})), 2)
        self.assertEqual(len({m for b in boards for m in b['symbols']}), 1)

    def test_metadata_date_range_does_not_prove_intervening_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'000001.parquet'
            pq.write_table(pa.table({'date': [20250102093100, 20250106150000]}), path)
            info = footer_info(path)
            self.assertEqual(info['rows'], 2)
            self.assertEqual(info['date_min'], 20250102093100)
            self.assertFalse(info['slot_coverage_verified'])
            self.assertEqual(info['hash_scope'], 'PARQUET_FOOTER_ONLY')
            self.assertFalse(info['content_hash_verified'])

    def test_missing_date_stats_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'000001.parquet'
            pq.write_table(pa.table({'date': [20250102093100]}), path, write_statistics=False)
            info = footer_info(path)
            self.assertIsNone(info['date_min'])
            self.assertFalse(info['date_stats_all_groups'])

    def test_invalid_binary_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'000001.parquet'
            path.write_bytes(b'not parquet')
            with self.assertRaisesRegex(ValueError, 'NOT_PARQUET'):
                footer_info(path)


if __name__ == '__main__':
    unittest.main()
