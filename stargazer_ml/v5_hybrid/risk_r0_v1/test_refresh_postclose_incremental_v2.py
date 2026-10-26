import unittest
from refresh_postclose_incremental_v2 import structured_pairs,valid_daily


class StructuredTests(unittest.TestCase):
    def test_nested_daily(self):
        row=dict(code='000001',date=20261016,open=10,high=10,low=10,close=10)
        for obj in [{'000001':{'20261016':row}},[['000001',[['20261016',row]]]],
                    [['日k:000001:20261016',row]],[row]]:
            pairs=structured_pairs(obj,'日k','*')
            self.assertEqual(pairs,[('日k:000001:20261016',row)])
            self.assertIn('000001',valid_daily(pairs,20261016))

    def test_adjustment_nested_date(self):
        row=dict(mult=1.02)
        for obj in [[['复权:000001:20261016',row]],{'20261016':row}]:
            self.assertEqual(structured_pairs(obj,'复权','000001'),[('复权:000001:20261016',row)])

    def test_unknown_shape_or_identity(self):
        for obj in [42,{'mult':1.02}]:
            with self.assertRaises(ValueError):structured_pairs(obj,'复权','000001')

    def test_nested_code_conflict(self):
        row=dict(code='000002',date=20261016)
        with self.assertRaisesRegex(ValueError,'CODE_MISMATCH'):
            structured_pairs({'000001':{'20261016':row}},'日k','*')

    def test_empty_stays_empty(self):
        self.assertEqual(structured_pairs([],'日k','*'),[])


if __name__=='__main__':unittest.main()
