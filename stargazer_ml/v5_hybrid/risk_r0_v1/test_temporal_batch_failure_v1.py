import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
import fit_postclose_temporal_v1 as module


class BatchFailureTests(unittest.TestCase):
    def test_first_failure_stops_before_D3_and_cannot_restart(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(module,'OUT',Path(folder)/'batch'),patch.object(module,'bytes_used',return_value=0),patch.object(module,'frozen_parts',return_value=([],{},dict(D1={},D3={}),{})),patch.object(module,'fit_one',side_effect=RuntimeError('injected')) as fit:
            with self.assertRaisesRegex(RuntimeError,'injected'):module.run(True)
            self.assertEqual(fit.call_count,1)
            self.assertFalse((module.OUT/'D3').exists())
            stop=json.loads((module.OUT/'stop_record.json').read_bytes())
            self.assertEqual(stop['status'],'FAILED_NO_AUTOMATIC_CONTINUATION')
            self.assertEqual(stop['completed_folds'],[])
            with self.assertRaisesRegex(ValueError,'NO_OVERWRITE'):module.run(True)

    def test_second_failure_preserves_first_fold(self):
        fake=({'model.txt':b'unit-test-placeholder'}, {'unit_test':True})
        with tempfile.TemporaryDirectory() as folder,patch('builtins.print'),patch.object(module,'OUT',Path(folder)/'batch'),patch.object(module,'bytes_used',return_value=0),patch.object(module,'frozen_parts',return_value=([],{},dict(D1={},D3={}),{})),patch.object(module,'fit_one',side_effect=[fake,RuntimeError('second-injected')]) as fit:
            with self.assertRaisesRegex(RuntimeError,'second-injected'):module.run(True)
            self.assertEqual(fit.call_count,2)
            self.assertEqual((module.OUT/'D1/model.txt').read_bytes(),b'unit-test-placeholder')
            self.assertEqual(json.loads((module.OUT/'stop_record.json').read_bytes())['completed_folds'],['D1'])


if __name__=='__main__':unittest.main()
