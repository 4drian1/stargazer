import unittest
from ordinary_action_reconciliation_v3 import reconcile_event,ordinary_multiplier
from build_restricted_history_v3 import reconcile_code


class OrdinaryActionTests(unittest.TestCase):
    def example(self,cash=.105,song=0.,zhuan=0.):
        return dict(table='bonus',reconcile_status='eligible',progress='实施',per_share_cash=cash,
                    bonus_ratio_per10=dict(song=song,zhuan=zhuan))

    def test_small_cash_mult_one_is_not_no_event(self):
        r=reconcile_event(dict(div=.105,give=0,trans=0,mult=1),self.example(),325.43)
        self.assertEqual(r['status'],'ORDINARY_PARAMETERS_CORROBORATED')
        self.assertTrue(r['stored_rounded_to_one']);self.assertGreater(r['multiplier'],1)

    def test_byd_cash_and_share_formula(self):
        r=reconcile_event(dict(div=3.974,give=.8,trans=1.2,mult=3.036),self.example(3.974,8.,12.),337.)
        self.assertAlmostEqual(r['multiplier'],3.035799006684163)

    def test_missing_parameters_not_zero(self):
        r=reconcile_event(dict(div=0,give=0,trans=0,mult=1),self.example(None),100.)
        self.assertIsNone(r['multiplier'])

    def test_cross_source_unit_mismatch(self):
        r=reconcile_event(dict(div=1.05,give=0,trans=0,mult=1),self.example(),325.43)
        self.assertEqual(r['status'],'CROSS_SOURCE_PARAMETER_MISMATCH')

    def test_special_or_rights_not_plain_bonus(self):
        e=self.example();e['table']='rights'
        self.assertIsNone(reconcile_event(dict(div=0,give=0,trans=0,mult=1),e,100.)['multiplier'])

    def test_dividend_ge_price_rejected(self):
        with self.assertRaises(ValueError):ordinary_multiplier(1.,1.,0.,0.)

    def test_single_source_date_is_not_shifted(self):
        native=[['复权:000001:20260106',dict(mult=1.01,div=1,give=0,trans=0)]]
        e=self.example(1.);e.update(code='000001',dates={'除权除息日':'2026-01-07'})
        bars={20260105:{'close':100.},20260106:{'close':99.},20260107:{'close':98.}}
        events,audit,issues=reconcile_code('000001',bars,[e],native)
        self.assertEqual(events,[]);self.assertEqual(len(issues),2)


if __name__=='__main__':unittest.main()
