import copy
import unittest

from test_candidate_resonance_history_v1 import market_records, evidence_factors, source_state


class MarketReplayParserTests(unittest.TestCase):
    def fixture(self):
        return dict(data=dict(sh000300=dict(day=[['2026-01-05', '100', '101', '102', '99']],
                    qt=dict(sh000300=['', '沪深300', '000300']))))

    def test_explicit_ohlc_mapping(self):
        self.assertEqual(market_records(self.fixture())[20260105], dict(open=100, close=101, high=102, low=99))

    def test_adjusted_key_refused(self):
        p = self.fixture(); p['data']['sh000300']['qfqday'] = []
        with self.assertRaisesRegex(ValueError, 'NOT_RAW'): market_records(p)

    def test_wrong_identity_refused(self):
        p = self.fixture(); p['data']['sh000300']['qt']['sh000300'][2] = '000001'
        with self.assertRaisesRegex(ValueError, 'IDENTITY'): market_records(p)

    def test_conflict_and_invalid_price_refused(self):
        p = self.fixture(); p['data']['sh000300']['day'].append(['2026-01-05', '100', '100', '102', '99'])
        with self.assertRaisesRegex(ValueError, 'CONFLICT'): market_records(p)
        p = copy.deepcopy(self.fixture()); p['data']['sh000300']['day'][0][2] = 'inf'
        with self.assertRaisesRegex(ValueError, 'INVALID'): market_records(p)

    def test_unresolved_audit_not_a_factor_and_still_blocks(self):
        cal = [20260105, 20260106, 20260107]
        ev = dict(events=[dict(date=20260106, multiplier=None)],
                  issues=[dict(date=20260106, reason='PARAM_GAP')])
        self.assertEqual(evidence_factors(ev, cal), dict.fromkeys(cal, 1.))
        self.assertEqual(source_state(ev, cal, 2), 'SOURCE_EVENT_PENDING')
        ev['issues'] = []
        with self.assertRaisesRegex(ValueError, 'WITHOUT_ISOLATION'): evidence_factors(ev, cal)

    def test_future_dated_issue_does_not_change_past_source_gate(self):
        ev = dict(events=[], issues=[dict(date=20260107, reason='UNRESOLVED')])
        self.assertIsNone(source_state(ev, [20260105, 20260106, 20260107], 1))


if __name__ == '__main__':
    unittest.main()
