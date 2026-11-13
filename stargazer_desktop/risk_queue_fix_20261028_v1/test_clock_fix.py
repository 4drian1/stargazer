import sys,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'stargazer_ml/v5_hybrid/risk_r0_v1'))
from joint_shadow_context_v1 import observation_time
from joint_shadow_compute_v2 import ComputeUnavailable
class ClockTests(unittest.TestCase):
    def call(self,cutoff,observed,scope):return observation_time(SimpleNamespace(cutoff=cutoff,observed_at=observed,membership_scope=scope))
    def test_explicit_closed_capture_keeps_real_next_day_arrival(self):
        self.assertEqual(self.call(20261027150000,20261028085000,'POSTCLOSE_CAPTURE_OBSERVED_LATER'),20261028085000)
    def test_forward_scope_still_rejects_cross_day(self):
        with self.assertRaises(ComputeUnavailable):self.call(20261027150000,20261028085000,'FORWARD_SNAPSHOT')
    def test_intraday_never_gets_cross_day_exception(self):
        with self.assertRaises(ComputeUnavailable):self.call(20261027113000,20261028085000,'POSTCLOSE_CAPTURE_OBSERVED_LATER')
    def test_future_data_cutoff_never_allowed(self):
        with self.assertRaises(ComputeUnavailable):self.call(20261028150000,20261028085000,'POSTCLOSE_CAPTURE_OBSERVED_LATER')
    def test_current_day_forward_unchanged(self):
        self.assertEqual(self.call(20261028150000,20261028194000,'FORWARD_SNAPSHOT'),20261028194000)
