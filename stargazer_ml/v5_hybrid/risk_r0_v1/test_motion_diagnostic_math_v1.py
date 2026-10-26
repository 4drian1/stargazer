import unittest
import numpy as np
import pandas as pd

from motion_diagnostic_math_v1 import static_membership,old_linear_contributions,outcome_mix


class DiagnosticTests(unittest.TestCase):
    def test_one_unbound_member_blocks_full_but_not_all_quotes(self):
        boards=[dict(code='X',symbols=['A','B']),dict(code='Y',symbols=['A','C'])]
        r=static_membership(boards,{'A','B'},['A','B'])
        self.assertEqual(r['A']['related'],2);self.assertEqual(r['A']['potentially_complete'],1)

    def test_exclusion_is_not_no_related_or_no_source(self):
        r=static_membership([dict(code='X',symbols=['A','B'])],{'B'},['A','C'])
        self.assertEqual(r['A']['potentially_complete'],1)
        self.assertEqual(r['C']['related'],0)

    def test_coefficient_missing_indicator_offset_uses_full_width(self):
        state=dict(columns=['x','new'],median=[2,0],mean=[1,0],scale=[2,1])
        a,z=old_linear_contributions(pd.DataFrame({'x':[3,np.nan]}),['x'],state,[4,5,6,7])
        np.testing.assert_allclose(a[:,0],[4,8]);np.testing.assert_allclose(z[:,0],[1,.5])

    def test_shape_rejects_misleading_coefficient_mapping(self):
        with self.assertRaises(ValueError):old_linear_contributions(pd.DataFrame({'x':[1]}),['x'],
            dict(columns=['new','x'],median=[0,0],mean=[0,0],scale=[1,1]),[1]*4)

    def test_population_mixture_identity(self):
        r=outcome_mix(['A','A','B','B'],[1,1,3,3],['A','A','A','B'],[2,2,2,4])
        self.assertEqual(r['population_mix_component'],-.5)
        self.assertEqual(r['within_code_outcome_shift'],1)
        self.assertEqual(r['observed_mean']-r['train_mean'],r['population_mix_component']+r['within_code_outcome_shift'])

    def test_unseen_code_disclosed_not_silently_dropped(self):
        r=outcome_mix(['A'],[1],['B'],[4]);self.assertEqual(r['unseen_code_rows'],1)
        self.assertEqual(r['within_code_outcome_shift'],3)


if __name__=='__main__':unittest.main()
