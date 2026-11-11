"""#18.188 intraday outlook slot: WAITING_SOURCE deferral, once-per-slot records."""
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tarot_monitor_v2 as live

CUTOFF = 20261103133000


class FrozenDatetime(datetime):
    current = None

    @classmethod
    def now(cls, tz=None):
        return cls.current


class SlotWaitTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.redirect = mock.patch.object(live.runtime, 'directory', return_value=self.root)
        self.redirect.start()
        self.writes = []

        def fake_save(file, value):
            self.writes.append(Path(file))
            Path(file).parent.mkdir(parents=True, exist_ok=True)
            Path(file).write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')

        self.save = mock.patch.object(live, 'save', side_effect=fake_save)
        self.save.start()
        quotes = {'market.sse': dict(status='KNOWN', text='沪指实时报价 3580；快照不代表完整五分钟确认')}
        self.outlook = mock.patch.dict(sys.modules, {
            'risk_outlook_v1': SimpleNamespace(live_index_facts=lambda cutoff: quotes)})
        self.outlook.start()
        self.ready = mock.patch.object(live, 'index_prefix_ready', return_value=False)
        self.ready.start()
        self.clock = mock.patch.object(live, 'datetime', FrozenDatetime)
        self.clock.start()
        FrozenDatetime.current = datetime(2026, 11, 3, 13, 31, 0)
        self.view = dict(plan_hash='h', plan_id='p1')

    def tearDown(self):
        self.clock.stop()
        self.ready.stop()
        self.outlook.stop()
        self.save.stop()
        self.redirect.stop()
        self.temp.cleanup()

    def _marker(self):
        key, slot = live.outlook_slot(CUTOFF, 'p1', 'h')
        return slot.with_name(key + '.waiting.json')

    def test_waiting_records_once_and_shows_quotes(self):
        self.assertFalse(live.intraday_slot_wait(self.view, CUTOFF, 'p1'))
        latest = json.loads((self.root / 'risk_outlook' / 'latest_intraday.json').read_bytes())
        self.assertEqual(latest['status'], 'WAITING_SOURCE')
        self.assertEqual(latest['computed_facts']['market.sse']['status'], 'KNOWN')
        self.assertIn('不以旧K线充当当前截止证据', latest['text'])
        self.assertTrue(self._marker().exists())
        count = len(self.writes)
        self.assertFalse(live.intraday_slot_wait(self.view, CUTOFF, 'p1'))
        self.assertEqual(len(self.writes), count)  # dedupe: no rewrite

    def test_ready_proceeds_without_records(self):
        self.ready.stop()
        self.ready = mock.patch.object(live, 'index_prefix_ready', return_value=True)
        self.ready.start()
        self.assertTrue(live.intraday_slot_wait(self.view, CUTOFF, 'p1'))
        self.assertFalse((self.root / 'risk_outlook' / 'latest_intraday.json').exists())
        self.assertEqual(self.writes, [])

    def test_existing_slot_never_rewaits(self):
        key, slot = live.outlook_slot(CUTOFF, 'p1', 'h')
        slot.parent.mkdir(parents=True, exist_ok=True)
        slot.write_text('{}', encoding='utf-8')
        self.assertTrue(live.intraday_slot_wait(self.view, CUTOFF, 'p1'))
        self.assertEqual(self.writes, [])

    def test_stale_marker_cleaned_when_source_catches_up(self):
        self.assertFalse(live.intraday_slot_wait(self.view, CUTOFF, 'p1'))
        self.assertTrue(self._marker().exists())
        self.ready.stop()
        self.ready = mock.patch.object(live, 'index_prefix_ready', return_value=True)
        self.ready.start()
        self.assertTrue(live.intraday_slot_wait(self.view, CUTOFF, 'p1'))
        self.assertFalse(self._marker().exists())

    def test_late_after_fresh_window_reported_once(self):
        FrozenDatetime.current = datetime(2026, 11, 3, 13, 40, 0)  # cutoff+420s
        self.assertFalse(live.intraday_slot_wait(self.view, CUTOFF, 'p1'))
        latest = json.loads((self.root / 'risk_outlook' / 'latest_intraday.json').read_bytes())
        self.assertEqual(latest['status'], 'SOURCE_LATE')
        self.assertFalse(latest['automatic_retry'])
        count = len(self.writes)
        self.assertFalse(live.intraday_slot_wait(self.view, CUTOFF, 'p1'))
        self.assertEqual(len(self.writes), count)  # late fault recorded once

    def test_waiting_to_late_transition_writes_both_states_once(self):
        self.assertFalse(live.intraday_slot_wait(self.view, CUTOFF, 'p1'))
        FrozenDatetime.current = datetime(2026, 11, 3, 13, 40, 0)
        self.assertFalse(live.intraday_slot_wait(self.view, CUTOFF, 'p1'))
        marker = json.loads(self._marker().read_bytes())
        self.assertEqual(marker['status'], 'SOURCE_LATE')
        count = len(self.writes)
        self.assertFalse(live.intraday_slot_wait(self.view, CUTOFF, 'p1'))
        self.assertEqual(len(self.writes), count)


class EvaluateDrivenSlotTest(unittest.TestCase):
    """Waiting -> timeout through the real evaluate() entry, with AI/push forbidden."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.redirect = mock.patch.object(live.runtime, 'directory', return_value=self.root)
        self.redirect.start()
        self.writes = []

        def fake_save(file, value):
            self.writes.append(Path(file))
            Path(file).parent.mkdir(parents=True, exist_ok=True)
            Path(file).write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')

        self.save = mock.patch.object(live, 'save', side_effect=fake_save)
        self.save.start()
        self.queued = []
        stub = SimpleNamespace(
            half_hour=lambda cutoff: cutoff % 1000000 in (100000, 103000, 110000, 113000, 133000, 140000, 143000, 150000),
            queue=lambda cfg, view, cutoff, **kw: self.queued.append((cutoff, kw)) or dict(status='QUEUED'),
            live_index_facts=lambda cutoff: {'market.sse': dict(status='KNOWN', text='沪指实时报价 3580')})
        self.outlook = mock.patch.dict(sys.modules, {'risk_outlook_v1': stub})
        self.outlook.start()
        self.bridge = mock.patch.dict(sys.modules, {'joint_risk_bridge': SimpleNamespace(completed_cutoff=lambda now: CUTOFF)})
        self.bridge.start()
        self.services = mock.patch.dict(sys.modules, {'services': SimpleNamespace(wl_load=lambda cfg: {'positions': []})})
        self.services.start()
        self.events = mock.patch.dict(sys.modules, {'tarot_event_v1': SimpleNamespace(
            evaluate=lambda plan, cfg, now: dict(status='OK', rows={}),
            read_risk=lambda plan: {})})
        self.events.start()
        self.ready = mock.patch.object(live, 'index_prefix_ready', return_value=False)
        self.ready.start()
        self.clock = mock.patch.object(live, 'datetime', FrozenDatetime)
        self.clock.start()
        self.plan = dict(assistant_plan_id='p1', monitor_date=20261103, rows=[])

    def tearDown(self):
        self.clock.stop()
        self.ready.stop()
        self.events.stop()
        self.services.stop()
        self.bridge.stop()
        self.outlook.stop()
        self.save.stop()
        self.redirect.stop()
        self.temp.cleanup()

    def _marker(self):
        key, slot = live.outlook_slot(CUTOFF, 'p1', live.digest(self.plan))
        return slot.with_name(key + '.waiting.json')

    def test_evaluate_waits_then_times_out_without_ai_or_push(self):
        FrozenDatetime.current = datetime(2026, 11, 3, 13, 31, 0)
        live.evaluate(self.plan, {}, datetime(2026, 11, 3, 13, 31, 0))
        self.assertTrue(self._marker().exists())
        latest = json.loads((self.root / 'risk_outlook' / 'latest_intraday.json').read_bytes())
        self.assertEqual(latest['status'], 'WAITING_SOURCE')
        self.assertEqual(self.queued, [])  # not queued while source lags
        FrozenDatetime.current = datetime(2026, 11, 3, 13, 40, 0)
        live.evaluate(self.plan, {}, datetime(2026, 11, 3, 13, 40, 0))
        latest = json.loads((self.root / 'risk_outlook' / 'latest_intraday.json').read_bytes())
        self.assertEqual(latest['status'], 'SOURCE_LATE')
        self.assertEqual(self.queued, [])  # stale slot never queues AI/push
        count = len(self.writes)
        live.evaluate(self.plan, {}, datetime(2026, 11, 3, 13, 41, 0))
        self.assertEqual(len(self.writes), count)  # late fault recorded once


if __name__ == '__main__':
    unittest.main()
