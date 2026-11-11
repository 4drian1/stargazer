"""Counterexample tests for the independent index-price evidence line (#18.187).
1 Pyxis missing but prices complete -> price facts still enter the pack.
2 Only one index ready -> keep known side; gate names the missing one.
3 Gap/stale/over-cutoff -> still refused, with a specific reason.
4 Fingerprint mismatch -> specific error; overlay failure cache recovers."""
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for sub in ('stargazer_desktop', 'stargazer_ml/v5_hybrid/green_common_profile',
            'stargazer_ml/v5_hybrid/risk_r0_v1'):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd

import plan_evidence
import push_presenter_v1
import risk_outlook_v1

CUTOFF = 20261102143000
DAY = 20261102


def _evidence(state='REPAIRED', asof=CUTOFF):
    return dict(status='PRICE_FACTS_ONLY', asof=asof, last_close=3635.78,
                critical_position=dict(state=state, lower=3600.0, upper=3600.0,
                                       reference_asof=20261101150000, asof=asof),
                volume_verified=False)


def _collect(prices, error=None):
    with mock.patch('trade_facts_v1.fact_table', return_value={}), \
            mock.patch('plan_evidence.archived_price_overlay',
                       lambda cutoff, err=None: prices):
        return risk_outlook_v1.collect({'items': []}, CUTOFF, overlay=True)


class CounterexampleTests(unittest.TestCase):
    def test_pyxis_missing_prices_complete(self):
        facts = _collect({'sse': _evidence('REPAIRED'), 'cyb': _evidence('PENDING')})
        self.assertEqual(facts['index.sse.critical_state']['value'], 'REPAIRED')
        self.assertEqual(facts['index.cyb.critical_state']['value'], 'PENDING')
        self.assertEqual(facts['market.sse.price']['status'], 'KNOWN')
        self.assertNotIn('index.price_overlay', facts)

    def test_single_index_gate_names_missing(self):
        facts = _collect({'sse': _evidence('REPAIRED')})
        self.assertEqual(facts['index.sse.critical_state']['value'], 'REPAIRED')
        self.assertNotIn('index.cyb.critical_state', facts)
        states = {'sse': facts['index.sse.critical_state']['value'], 'cyb': None}
        text = push_presenter_v1.gate(states)
        self.assertIn('⚪灰', text)
        self.assertIn('创业板', text)
        self.assertNotIn('沪指关键位置状态）', text)

    def test_stale_evidence_refused_with_reason(self):
        facts = _collect({'sse': _evidence(asof=20261102140000)})
        self.assertNotIn('index.sse.critical_state', facts)
        self.assertEqual(facts['index.sse.price_state']['status'], 'UNKNOWN')
        self.assertIn('未就绪', facts['index.sse.price_state']['text'])

    def test_observe_gap_prefix_refused(self):
        rows = []
        for t in (935, 940):  # 945 missing: gap at cutoff 945
            ts = DAY * 1000000 + t * 100
            rows.append(dict(date=ts, open=10.0, high=10.1, low=9.9, close=10.05))
        frame = pd.DataFrame(rows)
        from market_price_evidence_v1 import observe
        result = observe(frame, DAY * 1000000 + 94500)
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertEqual(result['reason'], 'CURRENT_PREFIX_GAP_OR_CUTOFF_MISMATCH')

    def test_verify_fingerprints_reports_mismatch(self):
        from joint_shadow_service_v1 import verify_fingerprints
        mismatch, ok = verify_fingerprints({'stargazer_desktop/notifier.py': '0' * 64})
        self.assertFalse(ok)
        self.assertTrue(any('notifier.py' in m for m in mismatch))

    def test_overlay_failure_recovers(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            worker = base / 'risk_r0_v1' / 'tarot_intraday_risk_v2.py'
            mirror = worker.parent / 'index_minutes_v1'
            mirror.mkdir(parents=True)
            for name in ('sse', 'cyb'):
                (mirror / (name + '_5min.parquet')).write_bytes(b'x')
            good = types.SimpleNamespace(returncode=0, stderr='', stdout=json.dumps(
                dict(cutoff=CUTOFF, evidence={'sse': _evidence(), 'cyb': _evidence()})))
            bad = types.SimpleNamespace(returncode=1, stdout='', stderr='boom')
            calls = []

            def fake_run(*args, **kwargs):
                calls.append(1)
                return bad if len(calls) == 1 else good

            plan_evidence._price_cache.clear()
            with mock.patch('joint_risk_bridge._worker_paths', return_value=('python', worker)), \
                    mock.patch.object(plan_evidence.subprocess, 'run', side_effect=fake_run):
                error = {}
                self.assertEqual(plan_evidence.archived_price_overlay(CUTOFF, error), {})
                self.assertTrue(error['reason'].startswith('CALCULATION_FAILED'))
                again = plan_evidence.archived_price_overlay(CUTOFF, {})
                self.assertEqual(again['sse']['status'], 'PRICE_FACTS_ONLY')
                plan_evidence.archived_price_overlay(CUTOFF, {})  # success cached
            self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
