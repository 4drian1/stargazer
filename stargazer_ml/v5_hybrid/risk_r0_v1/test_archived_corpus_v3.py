import unittest
from build_archived_corpus_v3 import normalize_external,decimal_number


class CorpusNormalizationTests(unittest.TestCase):
    def test_units_and_allotment_isolation_marker(self):
        parsed=dict(bonus=[{'进度':'实施','除权除息日':'2025-07-29','派息(税前)(元)':'39.74',
                           '送股(股)':'8','转增(股)':'12'}],allotment=[{'除权日':'2023-12-08'}])
        rows=normalize_external('002594',parsed)
        self.assertEqual(rows[0]['per_share_cash'],3.974)
        self.assertEqual(rows[0]['bonus_ratio_per10']['song'],8)
        self.assertEqual(rows[1]['table'],'allotment')

    def test_missing_cash_stays_none(self):
        rows=normalize_external('000001',dict(bonus=[{'进度':'实施','派息(税前)(元)':'--'}],allotment=[]))
        self.assertIsNone(rows[0]['per_share_cash'])

    def test_not_distributed_not_action(self):
        self.assertEqual(normalize_external('000001',dict(bonus=[{'进度':'不分配'}],allotment=[])),[])

    def test_parser_failure_not_no_event(self):
        with self.assertRaises(ValueError):normalize_external('000001',dict(bonus=None,allotment=[]))


if __name__=='__main__':unittest.main()
