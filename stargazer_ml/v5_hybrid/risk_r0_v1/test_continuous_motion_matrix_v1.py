import unittest
import numpy as np
import pandas as pd

from continuous_motion_matrix_v1 import path_arrays,context,schema,board_motion
from continuous_motion_features_v1 import entity_motion
from intraday_relative_stall_v1 import TIME,slots,close_path


class MatrixTests(unittest.TestCase):
    def test_exact_schema(self):
        fields=[n for f in (5,15) for own in (False,True) for n in schema(f,own)]
        self.assertEqual(len(fields),118);self.assertEqual(len(set(fields)),118)

    def test_scalar_path_equivalence(self):
        for f in (5,15):
            for gap in (False,True):
                n=240//f;x=100+np.sin(np.arange(n*2))
                proof=np.ones(n*2,bool)
                if gap:proof[3]=False
                times=slots(20250102,f)+slots(20250103,f)
                frame=pd.DataFrame({TIME:times,'level':x,'day_return':x/100-1,'path_ok':proof})
                old=close_path(frame,f).rename(columns={'drawdown_from_close_day_high':'close_drawdown'})
                old['code']='A';old['interval_known']=proof;old['peak_known']=old.path_prefix_complete
                expected=entity_motion(old,['code'],f);actual=path_arrays(x,proof,f)
                for a,b in [('return_','motion_interval_return'),('deceleration','motion_deceleration'),('giveback','motion_giveback')]:
                    np.testing.assert_allclose(actual[a],expected[b],atol=1e-12,rtol=0,equal_nan=True)

    def test_future_and_truncation_both_native_grids(self):
        for f in (5,15):
            n=240//f;r=np.vstack([np.linspace(.01,.03,n),np.linspace(.02,.04,n)])
            markets={'sse':np.linspace(100,101,n),'cyb':np.linspace(100,102,n)}
            for own in (False,True):
                a=context(r,[[0,1]],markets,f,own=own)
                changed=r.copy();changed[:,6:]=.5
                b=context(changed,[[0,1]],markets,f,own=own)
                short=r.copy();short[:,6:]=np.nan
                c=context(short,[[0,1]],markets,f,own=own)
                for key in a:
                    np.testing.assert_allclose(a[key][:6],b[key][:6],equal_nan=True)
                    np.testing.assert_allclose(a[key][:6],c[key][:6],equal_nan=True)
                self.assertTrue(np.isfinite(a[next(iter(a))][2:6]).any())

    def test_candidate_not_promoted_to_runner_up(self):
        r=np.vstack([np.linspace(.1,.05,48),np.linspace(.01,.02,48)])
        stocks=path_arrays(1+r,np.isfinite(r),5)
        out=board_motion(r,[0,1],5,stocks,exclude=0)
        self.assertTrue(np.isnan(out['current_deceleration']).all())

    def test_missing_members_do_not_manufacture_basket_motion(self):
        r=np.full((2,48),np.nan);r[0,0]=.1;r[1,1]=.2
        markets={'sse':np.ones(48)*100,'cyb':np.ones(48)*100}
        out=context(r,[[0,1]],markets,5)
        self.assertTrue(np.isnan(out['cm_all5_deceleration_mean']).all())
        self.assertEqual(out['cm_all5_deceleration_coverage'][1],0)

    def test_empty_related_boards(self):
        out=context(np.zeros((1,48)),[],{'sse':np.ones(48),'cyb':np.ones(48)},5,own=True)
        self.assertTrue(all(np.isnan(v).all() for v in out.values()))


if __name__=='__main__':unittest.main()
