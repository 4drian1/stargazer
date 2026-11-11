"""#18.188-W3 isolated acceptance: failure classes surfaced, price evidence independent."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import push_presenter_v1 as presenter
import push_layout_v1 as layout

ASOF = 20261103133000
CODE = '300652'


def _runtime_with(latest):
    temp = tempfile.TemporaryDirectory()
    root = Path(temp.name)
    risk = root / 'postclose_risk' / 'tarot_v2'
    risk.mkdir(parents=True)
    if latest is not None:
        (risk / 'latest.json').write_text(json.dumps(latest, ensure_ascii=False),
                                          encoding='utf-8')
    return temp, SimpleNamespace(directory=lambda: root / 'postclose_risk')


class WaitClassTest(unittest.TestCase):
    def _wait(self, latest):
        temp, stub = _runtime_with(latest)
        try:
            with mock.patch.dict(sys.modules, {'postclose_risk_runtime': stub}):
                return presenter._risk_lane_wait()
        finally:
            temp.cleanup()

    def test_data_failure_class(self):
        text = self._wait(dict(status='EVIDENCE_FAILED',
                               reason='ValueError:REALTIME_CONTEXT_SUPPLY_PENDING'))
        self.assertTrue(text.startswith('数据失败：'), text)

    def test_compute_failure_class(self):
        text = self._wait(dict(status='EVIDENCE_FAILED',
                               reason='ValueError:RUNTIME_FINGERPRINT_MISMATCH:x.py'))
        self.assertTrue(text.startswith('计算失败：'), text)

    def test_legitimate_unknown_class(self):
        text = self._wait(dict(status='WAITING_SOURCE', reason='指数五分钟镜像尚未到达'))
        self.assertTrue(text.startswith('正常未知：'), text)

    def test_healthy_lane_no_wait_text(self):
        self.assertIsNone(self._wait(dict(status='MONITORING')))
        self.assertIsNone(self._wait(None))


class ReportDecouplingTest(unittest.TestCase):
    def _report(self, latest, signal_rows):
        from tarot_monitor_v2 import _flow_version
        from golden_rules import digest
        plan = dict(assistant_plan_id='p1', monitor_date=ASOF // 1000000,
                    rows=[dict(code=CODE, name='雷迪克', entry_rule=dict(levels=[26.15]))], boards=[])
        if latest is not None:
            latest = dict(latest, flow_version=_flow_version(), cutoff=ASOF,
                          plan_id='p1', plan_hash=digest(plan))
        temp, stub = _runtime_with(latest)
        try:
            bulletin = dict(asof=ASOF, computed_facts={}, analysis={})
            result = dict(rows=signal_rows)
            quotes = {CODE: dict(price=26.15, time=str(ASOF))}
            with mock.patch.dict(sys.modules, {'postclose_risk_runtime': stub}):
                return presenter.report(bulletin, plan, result, quotes, None,
                                        sector_events=[], observed_at=None)
        finally:
            temp.cleanup()

    def test_price_shown_when_risk_lane_data_failed(self):
        rendered = self._report(dict(status='EVIDENCE_FAILED',
                                     reason='ValueError:REALTIME_HISTORY_DAY_EMPTY'),
                                signal_rows={})
        body = rendered['body']
        self.assertIn('数据失败：', body)
        self.assertIn('26.15', body)  # 报价独立展示，不被Pyxis失败吞掉

    def test_signal_progress_not_masked_by_risk_failure(self):
        rendered = self._report(dict(status='EVIDENCE_FAILED',
                                     reason='ValueError:REALTIME_HISTORY_DAY_EMPTY'),
                                signal_rows={CODE: dict(status='RULE_TRIGGERED', node=26.15,
                                                        combination=dict(votes={}))})
        body = rendered['body']
        self.assertIn('⚡组合通过', body)
        self.assertNotIn('数据失败：', body.split('\n')[2])  # 该行走信号状态

    def test_lane_status_line_splits_three_stages(self):
        rendered = self._report(dict(status='EVIDENCE_FAILED',
                                     reason='ValueError:REALTIME_CONTEXT_SUPPLY_PENDING'),
                                signal_rows={})
        self.assertIn('环节状态: 规则：未运行｜Pyxis：数据失败：', rendered['body'])
        self.assertIn('｜DeepSeek：未完成', rendered['body'])


class LayoutPriorityTest(unittest.TestCase):
    def test_status_text_overrides_unknown(self):
        row = dict(code=CODE, name='n', price=26.15, status='UNKNOWN',
                   status_text='数据失败：REALTIME_CONTEXT_SUPPLY_PENDING')
        self.assertIn('数据失败：', layout.plan_line(row))

    def test_known_status_keeps_symbol(self):
        row = dict(code=CODE, name='n', price=26.15, status='RULE_TRIGGERED')
        self.assertIn('⚡组合通过', layout.plan_line(row))

    def test_no_status_text_falls_back(self):
        row = dict(code=CODE, name='n', price=26.15, status='UNKNOWN')
        self.assertTrue(layout.plan_line(row).endswith('待核验'))


if __name__ == '__main__':
    unittest.main()
