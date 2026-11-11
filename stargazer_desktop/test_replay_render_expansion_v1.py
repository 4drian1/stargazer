"""#18.189 replay render expansion: rule reasons, votes, day move, MFE must show."""
import unittest

import minute_plan_review as M


def _row(**over):
    row = dict(code='002585', name='双星新材', status='REPLAYED', origin=None,
               touch=None, first_signal=None, signal_count=0,
               state_changes=[dict(cutoff=20261102103000, status='RULE_PASSED',
                                   reason='未到达三重以上合格买入位置',
                                   combination=dict(votes=dict(repair=True, momentum=False, oversold=False)),
                                   pending_conditions=['严重超卖'])],
               simulation=None,
               day_move=dict(close=8.24, reference=8.10, change=8.24 / 8.10 - 1, high=8.40, low=7.95))
    row.update(over)
    return row


class RenderExpansionTest(unittest.TestCase):
    def test_unconfirmed_shows_reason_votes_daymove(self):
        result = dict(context=dict(day=20261102, cutoff=20261102150000), rows=[_row()])
        text = M.render(result)
        self.assertIn('未到达三重以上合格买入位置', text)
        self.assertIn('位置✓', text)
        self.assertIn('动能✗', text)
        self.assertIn('超卖✗', text)
        self.assertIn('待严重超卖', text)
        self.assertIn('收盘8.24', text)
        self.assertIn('+1.73%', text)
        self.assertIn('历史规则回放', text)

    def test_triggered_shows_votes_and_mfe(self):
        row = _row(first_signal=dict(trigger_time=20261102103500,
                                     combination=dict(votes=dict(repair=True, momentum=True, oversold=False))),
                   signal_count=1,
                   simulation=dict(entry_price=8.20, mark_to_market_return=0.01,
                                   maximum_adverse_excursion=-0.02, maximum_favorable_excursion=0.05))
        text = M.render(dict(context=dict(day=20261102, cutoff=20261102150000), rows=[_row(**row)]))
        self.assertIn('规则触发20261102103500', text)
        self.assertIn('最大上行5.00%', text)
        self.assertIn('最大下探-2.00%', text)

    def test_hypothetical_origin_kept(self):
        text = M.render(dict(context=dict(day=20261102, cutoff=20261102150000),
                             rows=[_row(origin='MANUAL_REVIEW_HYPOTHETICAL')]))
        self.assertIn('未过门股票，仅假设回放', text)
        self.assertIn('不宣称成功拦截', text)


if __name__ == '__main__':
    unittest.main()
