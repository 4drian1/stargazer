from copy import deepcopy
import unittest

from two_stage_risk_v1 import freeze_plan, evaluate, fingerprint

HASH = 'a' * 64


class TwoStageTests(unittest.TestCase):
    def plan(self):
        return freeze_plan(dict(code='002437', P=20261015, E=20261016,
            created_at='2026-10-15T19:00:00+08:00', daily_bar_completed_at='2026-10-15T15:00:00+08:00',
            daily_data_available_at='2026-10-15T18:00:00+08:00', candidate_known_at='2026-10-15T18:30:00+08:00',
            watch_lower=10., watch_upper=11., index_symbol='SH000001', sector_id='MOCK_SECTOR',
            leader_list_hash=HASH, rule_version='SYNTHETIC_ONLY', source_hashes=[HASH]),
            [20261015, 20261016, 20261019])

    def quote(self, price=10.5):
        return dict(code='002437', price=price, observed_at='2026-10-16T10:02:00+08:00',
                    available_at='2026-10-16T10:02:01+08:00', evidence_hash=HASH)

    def facts(self):
        facts = dict(code='002437', index_symbol='SH000001', sector_id='MOCK_SECTOR', leader_list_hash=HASH)
        for name in ('market_location', 'momentum', 'sector', 'leader'):
            facts[name] = dict(status='PASS', bar_start='2026-10-16T10:00:00+08:00',
                bar_end='2026-10-16T10:15:00+08:00', available_at='2026-10-16T10:15:03+08:00', evidence_hash=HASH)
        return facts

    def run_at(self, p=None, **kwargs):
        options = dict(decision_at='2026-10-16T10:15:05+08:00', max_quote_age_seconds=60, max_bar_age_seconds=60)
        options.update(kwargs)
        return evaluate(p or self.plan(), **options)

    def triggered(self):
        return self.run_at(quote=self.quote(), decision_at='2026-10-16T10:02:02+08:00')

    def test_plan_hash_is_immutable_and_owns_copy(self):
        p = self.plan(); p['content']['watch_lower'] = 9
        with self.assertRaisesRegex(ValueError, 'PLAN_INTEGRITY'): self.run_at(p)

    def test_calendar_next_session_and_arrival_order(self):
        c = self.plan()['content']; c.pop('calendar_content_hash')
        c['E'] = 20261019
        with self.assertRaisesRegex(ValueError, 'NEXT_TRADING'): freeze_plan(c, [20261015, 20261016, 20261019])
        c['E'] = 20261016; c['daily_data_available_at'] = '2026-10-15T20:00:00+08:00'
        with self.assertRaisesRegex(ValueError, 'ARRIVAL_ORDER'): freeze_plan(c, [20261015, 20261016])

    def test_before_open_lunch_and_expiry(self):
        for now in ('2026-10-16T09:00:00+08:00', '2026-10-16T12:00:00+08:00'):
            self.assertEqual(self.run_at(decision_at=now)['content']['state'], 'OUT_OF_SESSION')
        self.assertEqual(self.run_at(decision_at='2026-10-16T15:01:00+08:00')['content']['state'], 'EXPIRED')

    def test_outside_region_does_not_trigger(self):
        r = self.run_at(quote=self.quote(12), decision_at='2026-10-16T10:02:02+08:00')
        self.assertEqual(r['content']['state'], 'WAIT_TRIGGER')
        self.assertIsNone(r['content']['trigger_available_at'])

    def test_touch_alone_never_passes(self):
        self.assertEqual(self.triggered()['content']['state'], 'DATA_INSUFFICIENT')
        self.assertEqual(self.triggered()['content']['action'], 'NONE')

    def test_full_facts_without_touch_still_wait(self):
        self.assertEqual(self.run_at(observations=self.facts())['content']['state'], 'WAIT_TRIGGER')

    def test_all_pass_is_not_order_permission(self):
        r = self.run_at(previous=self.triggered(), observations=self.facts())['content']
        self.assertEqual((r['state'], r['action']), ('CONDITIONS_MET', 'NONE'))
        self.assertTrue(r['human_confirmation_required'])

    def test_wait_block_and_unknown_are_distinct(self):
        for status, expected in [('WAIT', 'WATCH_WAIT'), ('BLOCK', 'INVALIDATED'), ('UNKNOWN', 'DATA_INSUFFICIENT')]:
            facts = self.facts(); facts['sector']['status'] = status
            self.assertEqual(self.run_at(previous=self.triggered(), observations=facts)['content']['state'], expected)

    def test_future_available_block_is_not_known_block(self):
        facts = self.facts(); facts['leader']['status'] = 'BLOCK'
        facts['leader']['available_at'] = '2026-10-16T10:16:00+08:00'
        r = self.run_at(previous=self.triggered(), observations=facts)['content']
        self.assertEqual(r['state'], 'DATA_INSUFFICIENT')
        self.assertEqual(r['components']['leader'], 'UNKNOWN')

    def test_late_quote_and_wrong_code_do_not_trigger(self):
        for q in [self.quote(), dict(self.quote(), code='000001')]:
            r = self.run_at(quote=q)['content']
            self.assertEqual(r['state'], 'DATA_INSUFFICIENT')
            self.assertIsNone(r['trigger_available_at'])

    def test_unfinished_and_pretrigger_bars_refused(self):
        for start, end in [('10:15', '10:30'), ('09:45', '10:00')]:
            facts = self.facts()
            for key in ('market_location', 'momentum', 'sector', 'leader'):
                facts[key].update(bar_start=f'2026-10-16T{start}:00+08:00', bar_end=f'2026-10-16T{end}:00+08:00')
            self.assertEqual(self.run_at(previous=self.triggered(), observations=facts)['content']['state'], 'DATA_INSUFFICIENT')

    def test_stale_and_mixed_bar_cutoffs_refused(self):
        self.assertEqual(self.run_at(previous=self.triggered(), observations=self.facts(),
            decision_at='2026-10-16T10:17:00+08:00')['content']['state'], 'DATA_INSUFFICIENT')
        facts = self.facts(); facts['leader'].update(bar_start='2026-10-16T10:15:00+08:00',
            bar_end='2026-10-16T10:30:00+08:00', available_at='2026-10-16T10:30:02+08:00')
        r = self.run_at(previous=self.triggered(), observations=facts,
            decision_at='2026-10-16T10:30:03+08:00', max_bar_age_seconds=1200)
        self.assertIn('MIXED_BAR_CUTOFFS', r['content']['reason_codes'])

    def test_changed_plan_or_state_and_clock_rewind_refused(self):
        p, prev = self.plan(), self.triggered()
        other = deepcopy(p); other['content']['watch_upper'] = 12; other['plan_hash'] = fingerprint(other['content'])
        with self.assertRaisesRegex(ValueError, 'PLAN_VERSION'): self.run_at(other, previous=prev)
        with self.assertRaisesRegex(ValueError, 'CLOCK_REWIND'):
            self.run_at(previous=prev, decision_at='2026-10-16T10:01:00+08:00')
        prev['content']['state'] = 'CONDITIONS_MET'
        with self.assertRaisesRegex(ValueError, 'STATE_INTEGRITY'): self.run_at(previous=prev)

    def test_invalidated_plan_cannot_silently_reactivate(self):
        facts = self.facts(); facts['leader']['status'] = 'BLOCK'
        old = self.run_at(previous=self.triggered(), observations=facts)
        new = self.run_at(previous=old, observations=self.facts())
        self.assertEqual(new['content']['state'], 'INVALIDATED')

    def test_future_injection_does_not_change_past_snapshot(self):
        facts = self.facts(); p = self.plan(); old_hash = p['plan_hash']
        result = self.run_at(p, previous=self.triggered(), observations=facts)
        unrelated_future = dict(available_at='2026-10-19T10:00:00+08:00', status='BLOCK')
        facts['future_unselected_version'] = unrelated_future
        self.assertEqual(result, self.run_at(p, previous=self.triggered(), observations=facts))
        self.assertEqual(old_hash, p['plan_hash'])


if __name__ == '__main__':
    unittest.main()
