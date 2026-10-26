import unittest
from unittest.mock import patch
from pathlib import Path

from stockdb_adjustment_v2 import rebuild_mult_factors,continuity_diagnostic
from historical_risk_core_v1 import label_at
from stockdb_source_v1 import read_adjustment,build_factor_series
from verify_corporate_action_v1 import verify_code
from historical_stream_v1 import bound_rows
from market_index_source_v1 import load_csi300_closes,MarketIndexSourceUnbound


class AdjustmentRegressionTests(unittest.TestCase):
    def test_cum_reset_does_not_create_false_factor_drop(self):
        recs=[dict(date=20260105,mult=2.,cum=2.),dict(date=20260106,mult=1.1,cum=1.1)]
        self.assertEqual(rebuild_mult_factors(recs,[20260102,20260105,20260106]),
                         {20260102:1.,20260105:2.,20260106:2.2})
        self.assertEqual(build_factor_series(recs,[20260106]),{20260106:2.2})

    def test_invalid_mult_not_replaced_by_one(self):
        for value in (None,0,-1,True,float('nan'),float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rebuild_mult_factors([dict(date=20260105,mult=value)],[20260106])

    def test_adjustment_parser_does_not_swallow_invalid(self):
        for value in (None,0,True,'nan'):
            with self.subTest(value=value),patch('stockdb_source_v1._fetch',return_value=[
                ['复权:000001:20260105',dict(mult=value,cum=2.)]]):
                with self.assertRaises(ValueError): read_adjustment('000001')

    def test_duplicate_conflicting_events_rejected(self):
        with self.assertRaises(ValueError):
            rebuild_mult_factors([dict(date=20260105,mult=2),dict(date=20260105,mult=3)],[20260106])

    def test_no_events_is_not_coverage_pass(self):
        bars=[dict(date=20260105,close=100.),dict(date=20260106,close=90.)]
        result=continuity_diagnostic(bars,[])
        self.assertEqual(result['diagnostic_status'],'NOT_TESTED')
        self.assertEqual(result['corporate_action_coverage_verdict'],'UNVERIFIED')
        self.assertEqual(verify_code('000001',bars=bars,adjustments=[])['verdict'],'DIAGNOSTIC_ONLY')

    def test_ten_percent_move_inside_twelve_percent_band_is_not_admission(self):
        result=continuity_diagnostic([dict(date=20260105,close=100.),dict(date=20260106,close=45.)],
                                    [dict(date=20260106,mult=2.)])
        self.assertEqual(result['diagnostic_status'],'WITHIN_DIAGNOSTIC_BAND')
        self.assertFalse(result['training_eligible'])

    def test_one_day_offset_can_flip_five_day_tail_label(self):
        cal=[20260105,20260106,20260107,20260108,20260109,20260112,20260113]
        bars={d:dict(open=100.,high=101.,low=99.,close=100.) for d in cal}
        for d in cal[2:]: bars[d]=dict(open=50.,high=50.5,low=49.5,close=50.)
        premature=rebuild_mult_factors([dict(date=20260106,mult=2.)],cal)
        correct=rebuild_mult_factors([dict(date=20260107,mult=2.)],cal)
        old=label_at(bars,cal,20260106,premature,scale_verified=True,coverage_verified=True)
        fixed=label_at(bars,cal,20260106,correct,scale_verified=True,coverage_verified=True)
        self.assertEqual(old['y_tail_5d'],1)
        self.assertEqual(fixed['y_tail_5d'],0)

    def test_existing_real_binding_is_quarantined(self):
        root=Path(__file__).parent
        binding=root/'historical_stock_day_preparation_v1/real_binding_2code_v1/source_binding.json'
        if not binding.exists(): self.skipTest('real legacy artifact not available')
        with self.assertRaisesRegex(ValueError,'QUARANTINED'):
            next(bound_rows(binding,root/'historical_stock_day_preparation_v1/csi300_tracker_510300.csv'))

    def test_raw_etf_needs_separate_scope_and_distribution_evidence(self):
        import tempfile,json,hashlib
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'etf.csv'
            path.write_text('date,close\n20260105,100\n20260106,95\n',encoding='utf-8')
            meta=dict(symbol='SH510300',semantics='CSI300_ETF_TRACKER',adjustment='NONE',
                      source='SYNTHETIC_TEST',sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            path.with_name(path.name+'.binding.json').write_text(json.dumps(meta),encoding='utf-8')
            with self.assertRaises(MarketIndexSourceUnbound): load_csi300_closes(path)
            with self.assertRaises(MarketIndexSourceUnbound): load_csi300_closes(path,allow_etf_tracker=True)


if __name__=='__main__': unittest.main()
