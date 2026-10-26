import unittest
from types import SimpleNamespace
from live_postclose_job_v1 import pack_from_row
from accept_live_postclose_v1 import price_matches
from jev_adapter_v1 import _risk_evidence_incomplete,opportunity_triggered


class PacketTests(unittest.TestCase):
    def setUp(self):
        self.snapshot=SimpleNamespace(cutoff=20261021150000,observed_at=20261021180000,
            boards={'b':['000001','000002']},stock_minutes={'000001':None,'000002':None})
        self.row={'sse_15m_day_return_prev_close':0.,'cyb_15m_day_return_prev_close':-.01,
            'new_sse_golden_position':.618,'context_sector_mean_return_loo_raw':0.,
            'context_sector_mean_coverage_loo':1.,'context_current_leader_return_loo_raw':.01,
            'context_current_leader_return_change_loo_raw':-.001}

    def make(self):
        return pack_from_row(self.row,{'code':'000001','atr_ref':1.,'distance_atr':.2},
                             [0.,.01],self.snapshot,{'232':['missing'],'266':[]},'run/test')

    def test_zero_risk_amplitude_and_flat_market_are_known(self):
        pack=self.make()
        self.assertEqual(_risk_evidence_incomplete(pack),[])
        self.assertEqual(pack.risk_context['amp_232'],0.)
        self.assertEqual(pack.asof_ts,'20261021150000')

    def test_no_implicit_intraday_buy_trigger(self):
        pack=self.make()
        self.assertFalse(opportunity_triggered(pack))
        self.assertIsNone(pack.candidates[0]['trigger'])
        self.assertIsNone(pack.candidates[0]['p_close_score'])

    def test_unknown_leader_not_replaced(self):
        self.row['context_current_leader_return_loo_raw']=float('nan')
        pack=self.make()
        self.assertIn('leader_states',pack.risk_missing)
        self.assertIn('LEADER_STATES_UNKNOWN',_risk_evidence_incomplete(pack))

    def test_cross_source_scale_gate(self):
        self.assertTrue(price_matches(10.,10.01))
        self.assertFalse(price_matches(10.,10.02))
        self.assertFalse(price_matches(float('nan'),10.))
        self.assertFalse(price_matches(0.,10.))

    def test_only_actual_input_cells_are_exposed_missing_remains_unknown(self):
        from live_postclose_explain_v1 import attach_model_signals
        pack=self.make()
        attach_model_signals(pack,{'sse_5m_rsi14':25.,'mb5_positive_share':0.},
                             {'ccm_v1_5_positive_deceleration':float('nan')})
        self.assertEqual(pack.risk_context['market_state']['model_inputs']['sse_5m_rsi14'],25.)
        self.assertEqual(pack.risk_context['all_board_signals']['mb5_positive_share'],0.)
        self.assertIsNone(pack.risk_context['all_board_signals']['ccm_v1_5_positive_deceleration'])

    def test_selection_next_session_calendar_bound_to_candidate_snapshot(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from accept_live_postclose_v1 import plan_calendar
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'calendar.json';raw=json.dumps(['20261016','20261019']).encode();path.write_bytes(raw)
            snapshot={'data_date':20261016,'monitor_date':20261019,
                      'inputs':[{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}]}
            self.assertEqual(plan_calendar(snapshot)[0],[20261016,20261019])
            snapshot['monitor_date']=20261018
            with self.assertRaisesRegex(ValueError,'NEXT_TRADING_DAY'):
                plan_calendar(snapshot)

    def test_selection_calendar_drift_rejected(self):
        import tempfile,json
        from pathlib import Path
        from accept_live_postclose_v1 import plan_calendar
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'calendar.json';path.write_text('["20261021","20261022"]')
            with self.assertRaisesRegex(ValueError,'HASH_MISMATCH'):
                plan_calendar({'inputs':[{'path':str(path),'sha256':'wrong'}]})


if __name__=='__main__':
    unittest.main()
