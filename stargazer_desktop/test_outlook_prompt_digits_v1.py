"""Prompt hardening regression: sections must never contain digits, including
counts like '跌超4家' (2026-11-02 14:30 slot rejection)."""
import re
import unittest

import risk_outlook_v1


class PromptDigitGuardTest(unittest.TestCase):
    def test_prompt_forbids_counts_with_example(self):
        self.assertIn('跌超4家', risk_outlook_v1.PROMPT)
        self.assertIn('计数', risk_outlook_v1.PROMPT)

    def test_validate_rejects_count_digits(self):
        facts = {'market.sse': dict(status='KNOWN', text='沪指实时报价快照，不代表完整五分钟确认')}
        value = dict(asof=20261102143000, bias='MIXED',
                     sections={'market': '偏红修复', 'boards': '跌超4家转弱', 'leaders': '待核验',
                               'holdings': '未登记', 'plan': '观察'},
                     supporting_ids=[], opposing_ids=[], unknown_ids=['market.sse'],
                     focus_ids=[], danger_signals=[], repair_signals=[])
        with self.assertRaises(ValueError) as ctx:
            risk_outlook_v1.validate(value, facts, 20261102143000)
        self.assertEqual(str(ctx.exception), 'OUTLOOK_SECTION_INVALID')

    def test_validate_allows_reworded_count(self):
        facts = {'market.sse': dict(status='UNKNOWN', text='沪指关键位置状态待证')}
        value = dict(asof=20261102143000, bias='MIXED',
                     sections={'market': '偏红修复', 'boards': '跌超阈值的家数转弱', 'leaders': '待核验',
                               'holdings': '未登记', 'plan': '观察'},
                     supporting_ids=[], opposing_ids=[], unknown_ids=['market.sse'],
                     focus_ids=[], danger_signals=[], repair_signals=[])
        self.assertIsInstance(risk_outlook_v1.validate(value, facts, 20261102143000), str)


if __name__ == '__main__':
    unittest.main()
