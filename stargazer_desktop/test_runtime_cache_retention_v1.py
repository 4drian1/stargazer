"""#18.188-W4 cache lifecycle: daily-version retention, journal budget policy."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pentacles_journal
import tdx_daily_source_v2 as daily


class PruneDayVersionsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.day = Path(self.temp.name) / 'tdx_daily_v2' / '20261103'
        self.day.mkdir(parents=True)
        self.keeper = self._version('keeper')
        self.old = self._version('old1')
        self.older = self._version('old2')
        self.building = self._version('abc.building.123.456')
        (self.building / 'part_0000.parquet').write_bytes(b'x')

    def tearDown(self):
        self.temp.cleanup()

    def _version(self, name):
        folder = self.day / name
        folder.mkdir()
        (folder / 'COMPLETE.json').write_text('{}', encoding='utf-8')
        (folder / 'part_0000.parquet').write_bytes(b'x')
        return folder

    def test_prune_keeps_only_verified_and_building(self):
        daily._prune_day_versions(self.day, self.keeper)
        remaining = sorted(p.name for p in self.day.iterdir())
        self.assertEqual(remaining, ['abc.building.123.456', 'keeper'])
        self.assertTrue((self.keeper / 'COMPLETE.json').exists())

    def test_prune_missing_keep_untouched(self):
        daily._prune_day_versions(self.day, self.day / 'nonexistent')
        # Missing keeper must never authorize deleting the remaining evidence.
        remaining = sorted(p.name for p in self.day.iterdir())
        self.assertEqual(remaining, ['abc.building.123.456', 'keeper', 'old1', 'old2'])


class JournalBudgetPolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _journal(self, policy=None):
        if policy is not None:
            (self.root / 'journal_budget.json').write_text(
                json.dumps(policy), encoding='utf-8')
        return pentacles_journal.ObservationJournal(lambda: self.root)

    def test_default_two_mib(self):
        self.assertEqual(self._journal().budget, 2 * 1024 * 1024)

    def test_authorized_four_mib(self):
        journal = self._journal(dict(bytes=4 * 1024 * 1024, user_authorized=True))
        self.assertEqual(journal.budget, 4 * 1024 * 1024)

    def test_unauthorized_policy_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self._journal(dict(bytes=8 * 1024 * 1024, user_authorized=True)).budget
        self.assertIn('UNAUTHORIZED_JOURNAL_BUDGET', str(ctx.exception))
        with self.assertRaises(ValueError):
            self._journal(dict(bytes=4 * 1024 * 1024)).budget  # missing user_authorized

    def test_explicit_budget_argument_still_wins(self):
        journal = pentacles_journal.ObservationJournal(lambda: self.root, budget=1024)
        self.assertEqual(journal.budget, 1024)


if __name__ == '__main__':
    unittest.main()
