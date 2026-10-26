import unittest

import numpy as np

from prior_anchor_evidence_v1 import classify, normalized_return


class EvidenceTests(unittest.TestCase):
    def args(self):
        return dict(anchor=100., previous_day=20250102, current_day=20250103,
                    bars={20250102: dict(open=100, high=101, low=99, close=100)},
                    audit=dict(events=[], issues=[], missing_records=[]), source_bound=True)

    def test_conditional_not_full_certification(self):
        self.assertEqual(classify(**self.args()), ('CONDITIONAL_RESEARCH',
            'PRIOR_CLOSE_MATCH_AND_BOUND_ORDINARY_SCOPE', 1.))

    def test_missing_evidence_and_source_remain_pending(self):
        for key, value in [('audit', None), ('source_bound', False), ('bars', {})]:
            a = self.args(); a[key] = value
            self.assertEqual(classify(**a)[0], 'PENDING')

    def test_mismatch_and_invalid_daily(self):
        for change in [dict(close=99.98), dict(close=float('nan')), dict(low=102)]:
            a = self.args(); a['bars'][20250102].update(change)
            self.assertEqual(classify(**a)[0], 'CONFLICT')

    def test_price_tick_tolerance(self):
        a = self.args(); a['anchor'] = 100.01
        self.assertEqual(classify(**a)[0], 'CONDITIONAL_RESEARCH')

    def test_unknown_issue_date_blocks_whole_code(self):
        a = self.args(); a['audit']['issues'] = [dict(date=None)]
        self.assertEqual(classify(**a)[0], 'CONFLICT')

    def test_dependency_issues_and_quarantine(self):
        for date in [20250102, 20250103]:
            a = self.args(); a['audit']['issues'] = [dict(date=date)]
            self.assertEqual(classify(**a)[0], 'CONFLICT')
            a = self.args(); a['quarantine'] = [date]
            self.assertEqual(classify(**a)[0], 'CONFLICT')

    def test_action_normalization_direction(self):
        a = self.args(); a['audit']['events'] = [dict(date=20250103,
            multiplier=2., status='ORDINARY_PARAMETERS_CORROBORATED')]
        status, _, m = classify(**a)
        self.assertEqual(status, 'CONDITIONAL_RESEARCH')
        self.assertEqual(normalized_return(50., 100., m), 0.)
        self.assertAlmostEqual(normalized_return(45., 100., m), -.1)

    def test_future_perturbation_and_current_daily_not_used(self):
        a = self.args(); before = classify(**a)
        a['bars'][20250103] = dict(close=-999)
        a['audit']['events'] = [dict(date=20250106, multiplier=-999, status='UNKNOWN')]
        a['audit']['issues'] = [dict(date=20250106)]
        self.assertEqual(before, classify(**a))

    def test_invalid_or_duplicate_action_not_admitted(self):
        for m in [None, np.nan, np.inf, 0., -1., True]:
            a = self.args(); a['audit']['events'] = [dict(date=20250103, multiplier=m,
                status='ORDINARY_PARAMETERS_CORROBORATED')]
            self.assertEqual(classify(**a)[0], 'CONFLICT')
        a = self.args(); e = dict(date=20250103, multiplier=2., status='ORDINARY_PARAMETERS_CORROBORATED')
        a['audit']['events'] = [e,e]
        self.assertEqual(classify(**a)[0], 'CONFLICT')

    def test_missing_action_record_blocks(self):
        a = self.args(); a['audit']['missing_records'] = [dict(date=20250103)]
        self.assertEqual(classify(**a)[0], 'CONFLICT')

    def test_noncausal_and_nonfinite_return_refused(self):
        a = self.args(); a['previous_day'] = a['current_day']
        with self.assertRaisesRegex(ValueError, 'NOT_STRICTLY_PAST'):
            classify(**a)
        for value in [np.nan, np.inf, 0., -1.]:
            with self.assertRaises(ValueError):
                normalized_return(value, 100., 1.)


if __name__ == '__main__':
    unittest.main()
