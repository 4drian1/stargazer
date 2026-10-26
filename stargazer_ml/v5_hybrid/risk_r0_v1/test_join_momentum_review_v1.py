import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from join_momentum_review_v1 import project_market, attach, METRICS, TIME
from joint_fit_preparation_v1 import whitelist


class JoinMomentumTests(unittest.TestCase):
    def source(self):
        rows=[]
        for symbol in ['000001.SH','399006.SZ']:
            for step in [5,15]:
                for ts in [20260105094500,20260105100000]:
                    row=dict(symbol=symbol,frequency_minutes=step,decision_ts_assumed_end=ts)
                    for layer in ['intraday','prior_daily','provisional_daily']:
                        for metric in METRICS:
                            row[layer+'_'+metric]=1.
                    row['provisional_daily_rsi14_change_from_prior']=.5
                    rows.append(row)
        return pd.DataFrame(rows)

    def existing(self):
        return ['sse_5m_rsi14','sse_15m_rsi14','cyb_15m_rsi14','sse_15m_daily_rsi14_asof_prior_day']

    def base(self):
        return pd.DataFrame(dict(code=['B','A','C'],cutoff_index=[1,0,0],
            decision_ts_assumed_end=[20260105100000,20260105094500,20260105094500],
            y_tail_1200m=[1,0,1],old_feature=[np.nan,2.,3.]))

    def test_projection_and_rsi_aliases(self):
        context,aliases=project_market(self.source(),self.existing())
        self.assertEqual(len(context.columns)-1,54)
        self.assertEqual(set(aliases)-{TIME},set(self.existing()))
        self.assertFalse(set(context)-{TIME} & set(self.existing()))
        self.assertFalse(any('macd_dif_change_from_prior' in c for c in context))

    def test_no_absolute_price_or_admin_as_features(self):
        context,_=project_market(self.source(),self.existing())
        forbidden={'bbi','close','macd_dif','macd_dea','macd_hist','training_admitted','confirmed_divergence'}
        self.assertFalse(any(c.split('_',2)[-1] in forbidden for c in context if c != TIME))
        spec=json.loads((Path(__file__).parent/'contracts/JOINT_RISK_FIT_REVIEW_V1.json').read_bytes())
        spec['feature_groups']['market'].extend(c for c in context if c != TIME)
        self.assertEqual(len(whitelist(spec)),100)

    def test_exact_join_keeps_order_labels_and_missing(self):
        context,_=project_market(self.source(),self.existing())
        base=self.base()
        result=attach(base,context)
        pd.testing.assert_frame_equal(result[list(base)],base)
        self.assertEqual(len(result),3)
        self.assertTrue(pd.isna(result.iloc[0].old_feature))

    def test_missing_exact_cutoff_not_nearest_filled(self):
        base=self.base()
        context=pd.DataFrame({TIME:[20260105095000],'new_feature':[999.]})
        result=attach(base,context)
        self.assertTrue(result.new_feature.isna().all())
        self.assertEqual(len(result),len(base))

    def test_duplicate_market_keys_rejected(self):
        source=self.source()
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):
            project_market(pd.concat([source,source.iloc[:1]]),self.existing())

    def test_duplicate_context_and_cohort_rejected(self):
        base=self.base()
        context=pd.DataFrame({TIME:[20260105094500]*2,'new':[1.,2.]})
        with self.assertRaisesRegex(ValueError,'DUPLICATE_CONTEXT'):
            attach(base,context)
        with self.assertRaisesRegex(ValueError,'DUPLICATE_COHORT'):
            attach(pd.concat([base,base.iloc[:1]]),context.iloc[:1])

    def test_old_column_collision_rejected(self):
        context=pd.DataFrame({TIME:[20260105094500],'y_tail_1200m':[999]})
        with self.assertRaisesRegex(ValueError,'COLLISION'):
            attach(self.base(),context)

    def test_future_changes_do_not_alter_earlier_join(self):
        source=self.source()
        expected,_=project_market(source,self.existing())
        source.loc[source[TIME]>20260105094500,'provisional_daily_macd_hist_over_close']=999.
        changed,_=project_market(source,self.existing())
        pd.testing.assert_frame_equal(expected[expected[TIME]<=20260105094500].reset_index(drop=True),
                                      changed[changed[TIME]<=20260105094500].reset_index(drop=True))


if __name__=='__main__':
    unittest.main()
