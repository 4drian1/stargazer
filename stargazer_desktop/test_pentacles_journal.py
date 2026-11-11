"""Offline journal preservation, quality, future-label and budget tests."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from pentacles_journal import ObservationJournal
from pentacles_monitor import PriceMonitor


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'journal'
        self.journal = ObservationJournal(lambda: self.root)
        self.plan = {'data_date': 20260930, 'monitor_date': 20261008, 'created_at': 'first',
                     'plan_ready': True, 'rows': [
                         {'code': '600121', 'name': '测试', 'setup': 'A_SDP', 'reference_price': 10.,
                          'zone': {'zone_lower': 9., 'zone_upper': 9.5}},
                         {'code': '600103', 'name': '无区域', 'setup': 'B_LVB', 'zone': None}]}
        self.now = datetime(2026, 10, 8, 10)

    def tearDown(self):
        self.temp.cleanup()

    def data(self):
        return json.loads((self.root / 'days/20261008.json').read_text(encoding='utf-8'))

    def row(self):
        return next(row for row in self.data()['candidates'].values() if row['code'] == '600121')

    def record(self, price=9.4, stamp='20261008100000', status='WAIT_PRICE'):
        self.journal.record(self.plan, {'status': 'MONITORING',
                            'rows': {'600121': {'status': status, 'price': price, 'time': stamp}}}, self.now)

    def test_complete_population_before_observation(self):
        self.journal.register(self.plan)
        data = self.data()
        self.assertEqual(len(data['candidates']), 2)
        self.assertEqual(data['attempts'], {})
        self.assertEqual([r['samples'] for r in data['candidates'].values()], [0, 0])
        self.assertIn('NO_ZONE', [r['last_status'] for r in data['candidates'].values()])

    def test_immutable_versions_and_actual_hash(self):
        self.journal.register(self.plan)
        self.plan['created_at'] = 'rerun'
        self.journal.register(self.plan)
        self.assertEqual(len(self.data()['plans']), 1)
        self.plan['rows'][0]['zone']['zone_upper'] = 9.6
        self.journal.register(self.plan)
        self.assertEqual(len(self.data()['plans']), 2)
        for file in (self.root / 'plans').glob('*.json'):
            self.assertEqual(file.stem, hashlib.sha256(file.read_bytes()).hexdigest())

    def test_touch_followup_excludes_pre_touch_extremes(self):
        self.record(price=10.2)
        self.record(price=9.4, stamp='20261008100100')
        self.record(price=9.8, stamp='20261008100200')
        row = self.row()
        self.assertEqual(row['sample_max'], 10.2)
        self.assertEqual(row['after_touch_sample_max'], 9.8)
        self.assertEqual(row['first_observed_touch']['time'], '20261008100100')
        self.assertAlmostEqual(row['after_touch_sample_return'], 9.8 / 9.4 - 1)
        self.assertTrue(all(label['value'] is None for label in row['exact_labels'].values()))

    def test_restart_dedup_and_out_of_order(self):
        self.record()
        self.journal = ObservationJournal(lambda: self.root)
        self.record()
        self.record(price=9.1, stamp='20261008095900')
        row = self.row()
        self.assertEqual(row['samples'], 1)
        self.assertEqual(row['sample_min'], 9.4)
        self.assertEqual(row['quality_counts']['OUT_OF_ORDER_QUOTE'], 1)

    def test_bad_quote_null_not_failure_label(self):
        self.record(price=None, status='QUOTE_STALE')
        row = self.row()
        self.assertIsNone(row['first_observed_touch'])
        self.assertEqual(row['samples'], 0)
        self.assertEqual(row['quality_counts']['QUOTE_STALE'], 1)
        self.assertIsNone(row['exact_labels']['H3']['value'])

    def test_future_day_and_disabled_status(self):
        health = {'status': 'DISABLED', 'rows': {}}
        self.journal.record(self.plan, health, datetime(2026, 10, 7, 20))
        self.assertEqual(self.data()['attempts'], {})
        self.journal.record(self.plan, health, self.now)
        self.assertEqual(self.data()['attempts'], {'DISABLED': 1})
        self.assertEqual(next(iter(self.data()['candidates'].values()))['samples'], 0)

    def test_corruption_retained_and_error_visible(self):
        self.journal.register(self.plan)
        file = self.root / 'days/20261008.json'
        file.write_text('{broken', encoding='utf-8')
        self.record()
        self.assertEqual(file.read_text(), '{broken')
        self.assertTrue(self.journal.state()['error'])

    def test_budget_no_partial_new_plan(self):
        self.journal.register(self.plan)
        files = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.journal = ObservationJournal(lambda: self.root, budget=self.journal.state()['bytes'] + 1)
        self.plan['rows'][0]['zone']['zone_upper'] = 9.6
        self.journal.register(self.plan)
        self.assertTrue(self.journal.state()['error'])
        self.assertTrue(all(p.read_bytes() == value for p, value in files.items()))
        self.assertEqual(len(self.data()['plans']), 1)

    def test_immutable_plan_corruption_not_overwritten(self):
        self.journal.register(self.plan)
        file = next((self.root / 'plans').glob('*.json'))
        file.write_text('{}', encoding='utf-8')
        self.record()
        self.assertEqual(file.read_text(), '{}')
        self.assertIn('哈希', self.journal.state()['error'])

    def test_report_path_and_unknown_dates(self):
        self.journal.register(self.plan)
        for day in ('../../config', '20260230', '20261009'):
            with self.assertRaises(ValueError):
                self.journal.report(day)
        self.assertIn('不是买卖建议', self.journal.report()['markdown'])

    def test_recorder_failure_does_not_block_alert(self):
        monitor = PriceMonitor(lambda: self.temp.name)
        monitor.journal = ObservationJournal(lambda: Path(self.temp.name) / 'journal', budget=1)
        monitor.enable(True)
        quotes = Mock(return_value={'600121': {'price': 9.4, 'prev': 10., 'time': '20261008100000'}})
        broadcast, send = Mock(), Mock(return_value=(True, 'ok'))
        monitor.tick(self.plan, {'push': {'enabled': True}}, self.now, quotes, broadcast, send)
        self.assertTrue(monitor.journal.state()['error'])
        broadcast.assert_called_once()
        send.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
