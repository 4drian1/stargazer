import unittest

import numpy as np
import pandas as pd

from direct_warning_comparison_v1 import specification, warning_fields, inputs, train_patterns


class SpecificationTests(unittest.TestCase):
    def base(self):
        return [f'base_{i}' for i in range(200)]

    def test_exact_two_arms_and_explicit_mapping(self):
        base=self.base();new=warning_fields();columns=base+new+['ar_'+f for f in new]
        arms,mapping=specification(base,new,columns)
        self.assertEqual([len(arms[a]) for a in ['C0','C1']],[200,232])
        self.assertEqual(arms['C1'][:200],arms['C0'])
        self.assertTrue(all(mapping[f]=='ar_'+f for f in new))

    def test_missing_reordered_or_added_state_not_silently_accepted(self):
        base=self.base();new=warning_fields()
        for altered in [new[:-1],new[::-1],new+['nh_sse5_close_new_day_high']]:
            with self.assertRaisesRegex(ValueError,'SCHEMA_CHANGED'):
                specification(base,altered,base+altered)
        with self.assertRaisesRegex(ValueError,'MISSING_BOUND'):
            specification(base,new,base+new[:-1])

    def test_forbidden_baseline_and_duplicate_refused(self):
        base=self.base();new=warning_fields();base[0]='y_tail_1200m'
        with self.assertRaises(ValueError):
            specification(base,new,base+new)
        base=self.base();base[-1]=base[0]
        with self.assertRaisesRegex(ValueError,'BASELINE'):
            specification(base,new,base+new)

    def test_nan_retained_and_input_not_mutated(self):
        f=pd.DataFrame(dict(a=[1.,2.],ar_a=[3.,np.nan],y_tail=[0,1]))
        before=f.copy()
        x=inputs(f,['a'],{'a':'ar_a'})
        self.assertEqual(x.a.iloc[0],3.)
        self.assertTrue(np.isnan(x.a.iloc[1]))
        pd.testing.assert_frame_equal(f,before)

    def test_noncanonical_future_mapping_and_infinity_refused(self):
        with self.assertRaisesRegex(ValueError,'FUTURE'):
            inputs(pd.DataFrame(dict(a=[1.],y_tail=[0])),['a'],{'a':'y_tail'})
        with self.assertRaisesRegex(ValueError,'INFINITE'):
            inputs(pd.DataFrame(dict(a=[np.inf])),['a'],{'a':'a'})

    def test_constant_duplicate_not_automatically_dropped(self):
        f=pd.DataFrame(dict(a=[1.,np.nan,1.],b=[1.,np.nan,1.],c=[np.nan]*3))
        result=train_patterns(f)
        self.assertEqual(result['all_null_columns'],['c'])
        self.assertEqual(result['constant_nonmissing_columns'],['a','b'])
        self.assertIn(['a','b'],result['exact_duplicate_groups_including_missing_pattern'])
        self.assertFalse(result['automatic_column_removal'])

    def test_missingness_differences_are_not_exact_duplicates(self):
        result=train_patterns(pd.DataFrame(dict(a=[1.,np.nan],b=[1.,1.])))
        self.assertEqual(result['exact_duplicate_groups_including_missing_pattern'],[])


if __name__=='__main__':
    unittest.main()
