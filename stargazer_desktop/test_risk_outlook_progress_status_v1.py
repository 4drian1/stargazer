"""Regression: .progress facts with DATA_INSUFFICIENT value must carry UNKNOWN status.
Otherwise the AI legally-correct instinct (bucket unusable progress into unknown_ids)
trips OUTLOOK_KNOWN_AS_UNKNOWN and kills the whole half-hour bulletin (2026-11-02)."""
import unittest
from unittest import mock

import risk_outlook_v1

CUTOFF = 20261102113000


def _signals(status):
    return {'002585': {'status': status, 'next_lower_node': 10.0, 'last_two_closes': [10.5]}}


class ProgressStatusTest(unittest.TestCase):
    def _collect(self, status):
        with mock.patch('trade_facts_v1.fact_table', return_value={}), \
                mock.patch('plan_evidence.archived_price_overlay', return_value={}):
            return risk_outlook_v1.collect(
                {'items': [], 'plan_hash': 'x'}, CUTOFF, signals=_signals(status), overlay=False)

    def test_data_insufficient_is_unknown(self):
        facts = self._collect('DATA_INSUFFICIENT')
        self.assertEqual(facts['002585.progress']['status'], 'UNKNOWN')
        self.assertEqual(facts['002585.progress']['value'], 'DATA_INSUFFICIENT')

    def test_real_status_stays_known(self):
        facts = self._collect('RULE_PASSED')
        self.assertEqual(facts['002585.progress']['status'], 'KNOWN')
        self.assertEqual(facts['002585.progress']['value'], 'RULE_PASSED')

    def test_unknown_ids_accepts_data_insufficient(self):
        facts = self._collect('DATA_INSUFFICIENT')
        value = dict(asof=CUTOFF, bias='MIXED',
                     sections={k: '横盘整理观望' for k in ('market', 'boards', 'leaders', 'holdings', 'plan')},
                     supporting_ids=[], opposing_ids=[], unknown_ids=['002585.progress'],
                     focus_ids=[], danger_signals=[], repair_signals=[])
        try:
            risk_outlook_v1.validate(value, facts, CUTOFF)
        except ValueError as exc:
            self.assertNotEqual(str(exc), 'OUTLOOK_KNOWN_AS_UNKNOWN')
            raise
        self.assertTrue(True)  # full validate() passed


if __name__ == '__main__':
    unittest.main()
