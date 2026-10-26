import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
import fit_postclose_temporal_v1 as module


class TemporalFitTests(unittest.TestCase):
    def test_no_authorization_no_input_or_fit(self):
        with patch.object(module,'frozen_parts',side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError,'AUTHORIZATION_REQUIRED'):module.run()

    def test_existing_output_refused_before_fit(self):
        with tempfile.TemporaryDirectory() as d,patch.object(module,'OUT',Path(d)),patch.object(module,'frozen_parts',side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError,'NO_OVERWRITE'):module.run(True)

    def test_budget_refuses_before_output_creation(self):
        with tempfile.TemporaryDirectory() as d,patch.object(module,'OUT',Path(d)/'new'),patch.object(module,'frozen_parts',return_value=([],{}, {},{})),patch.object(module,'bytes_used',return_value=module.TOTAL_CAP):
            with self.assertRaisesRegex(ValueError,'RESERVATION_FAILED'):module.run(True)
            self.assertFalse((Path(d)/'new').exists())


if __name__=='__main__':unittest.main()
