"""Clock/quote/push injection only: never contact market or notification services."""
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

from pentacles_monitor import PriceMonitor


class ReminderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.monitor = PriceMonitor(lambda: self.temp.name)
        self.monitor.enable(True)
        self.now = datetime(2026, 10, 8, 10, 0)
        self.plan = {'data_date': 20260930, 'monitor_date': 20261008, 'plan_ready': True,
                     'rows': [{'code': '600121', 'name': '测试', 'reference_price': 10.,
                               'zone': {'zone_lower': 9., 'zone_upper': 9.5}}]}
        self.quote = {'price': 9.4, 'prev': 10., 'time': '20261008100000'}
        self.provider = Mock(side_effect=lambda codes: {'600121': self.quote})
        self.broadcast = Mock()
        self.send = Mock(return_value=(True, 'ok'))

    def tearDown(self):
        self.temp.cleanup()

    def tick(self, **kwargs):
        self.monitor.tick(kwargs.get('plan', self.plan), kwargs.get('cfg', {'push': {'enabled': True}}),
                          kwargs.get('now', self.now), self.provider, self.broadcast, self.send,
                          kwargs.get('clock'))

    def test_next_day_wait_and_expiry(self):
        self.tick(now=datetime(2026, 10, 7, 10))
        self.assertEqual(self.monitor.state()['status'], 'WAITING')
        self.tick(now=datetime(2026, 10, 9, 10))
        self.assertEqual(self.monitor.state()['status'], 'PLAN_EXPIRED')
        self.provider.assert_not_called()

    def test_disabled_persistent_restart(self):
        self.monitor.enable(False)
        self.monitor = PriceMonitor(lambda: self.temp.name)
        self.tick()
        self.assertFalse(self.monitor.state()['enabled'])
        self.provider.assert_not_called()

    def test_stale_plan_holiday_lunch(self):
        self.plan['plan_ready'] = False
        self.tick()
        self.assertEqual(self.monitor.state()['status'], 'PLAN_NOT_READY')
        self.plan['plan_ready'] = True
        self.tick(now=datetime(2026, 10, 8, 12))
        self.provider.assert_not_called()

    def test_price_hit_restart_dedup(self):
        self.tick()
        self.monitor = PriceMonitor(lambda: self.temp.name)
        self.tick()
        self.assertEqual(self.broadcast.call_count, 1)
        self.assertEqual(self.send.call_count, 1)
        self.assertIn('请人工复核', self.broadcast.call_args[0][0]['body'])
        self.assertIn('不是买入建议', self.broadcast.call_args[0][0]['body'])

    def test_price_above_no_alert_then_hit(self):
        self.quote['price'] = 9.6
        self.tick()
        self.broadcast.assert_not_called()
        self.quote['price'] = 9.5
        self.tick()
        self.broadcast.assert_called_once()

    def test_below_region_warns_not_buy(self):
        self.quote['price'] = 8.
        self.tick()
        self.assertIn('跌破关注区域', self.send.call_args[0][1])
        self.assertIn('警惕破位', self.send.call_args[0][2])

    def test_invalid_quotes_never_notify(self):
        for changes in ({'time': '20261007100000'}, {'time': '20261008095800'},
                        {'price': 0}, {'price': float('nan')}, {'prev': 9.9}, {'time': 'bad'}):
            with self.subTest(changes=changes):
                old = self.quote.copy()
                self.quote.update(changes)
                self.tick()
                self.quote = old
        self.send.assert_not_called()
        self.broadcast.assert_not_called()

    def test_wechat_off_records_local_notification(self):
        self.tick(cfg={'push': {'enabled': False}})
        self.broadcast.assert_called_once()
        self.send.assert_not_called()
        self.assertEqual(next(iter(self.monitor.state()['alerts'].values()))['delivery'], 'WECHAT_DISABLED')
        self.tick()
        self.send.assert_called_once()
        self.broadcast.assert_called_once()

    def test_retry_budget_survives_restart(self):
        self.send.return_value = False, 'failure'
        for _ in range(5):
            self.tick()
            self.monitor = PriceMonitor(lambda: self.temp.name)
        self.assertEqual(self.send.call_count, 3)
        self.assertEqual(self.broadcast.call_count, 1)

    def test_quote_latency_and_off_during_fetch(self):
        self.provider.side_effect = lambda codes: (self.monitor.enable(False) or {'600121': self.quote})
        self.tick()
        self.send.assert_not_called()
        self.monitor.enable(True)
        self.provider.side_effect = lambda codes: {'600121': self.quote}
        self.quote['time'] = '20261008100001'
        self.tick(clock=lambda: datetime(2026, 10, 8, 10, 0, 2))
        self.send.assert_called_once()

    def test_corrupt_dedup_not_reset(self):
        file = Path(self.temp.name) / 'monitor.json'
        file.write_text('{broken', encoding='utf-8')
        self.assertEqual(self.monitor.state()['status'], 'STATE_ERROR')
        with self.assertRaises(json.JSONDecodeError):
            self.tick()
        self.send.assert_not_called()

    def test_calendar_holidays_and_before_close(self):
        directory = Path(__file__).parent.parent / 'stargazer_ml/v5_hybrid/green_common_profile'
        sys.path.insert(0, str(directory))
        try:
            from gcp_demo_snapshot import plan_dates
            days = [20260928, 20260929, 20260930, 20261008, 20261009]
            self.assertEqual(plan_dates(days, datetime(2026, 10, 7, 20)), (20260930, 20261008))
            self.assertEqual(plan_dates(days, datetime(2026, 10, 8, 16)), (20261008, 20261009))
            with self.assertRaises(ValueError):
                plan_dates(days, datetime(2026, 10, 8, 12))
            with self.assertRaises(ValueError):
                plan_dates(days, datetime(2026, 10, 10, 16))
        finally:
            sys.path.remove(str(directory))


if __name__ == '__main__':
    unittest.main(verbosity=2)
