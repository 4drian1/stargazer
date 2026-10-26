"""Reuse v1 boundary tests and compare every real row, retaining the v1 evidence."""
import io
import json
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

import read_conditional_board_observations_v1 as legacy
import read_conditional_board_observations_v2 as new


class WideTests(unittest.TestCase):
    def test_wide_batch_has_no_fragmentation_warning(self):
        frame=pd.DataFrame({'code':['A'],new.build.TIME:[1],**{f'x{i}':[.1] for i in range(150)}})
        diff=pd.DataFrame({'code':['A']*150,new.build.TIME:[1]*150,
            'feature':[f'x{i}' for i in range(150)],'old_value':[.1]*150,'new_value':[.2]*150})
        with warnings.catch_warnings():
            warnings.simplefilter('error',pd.errors.PerformanceWarning)
            result=new.attach(frame,diff)
        self.assertEqual(result.shape[1],302)
        np.testing.assert_allclose(result[[f'ar_x{i}' for i in range(150)]],.2)


def run():
    build=new.build
    destination=build.HERE/'conditional_board_reader_acceptance_20261018_v2.json'
    if destination.exists():
        raise ValueError('ACCEPTANCE_EXISTS_NO_OVERWRITE')
    log=io.StringIO()
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName('test_read_conditional_board_observations_v1'),
                             unittest.defaultTestLoader.loadTestsFromTestCase(WideTests)])
    with patch('test_read_conditional_board_observations_v1.attach',new.attach):
        tests=unittest.TextTestRunner(stream=log).run(suite)
    if not tests.wasSuccessful() or tests.skipped:
        raise ValueError('READER_V2_TEST_FAILED:'+log.getvalue())
    summaries={}
    for segment in build.old.prep.SEGMENTS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',pd.errors.PerformanceWarning)
            old=legacy.read_segment(segment)
        with warnings.catch_warnings():
            warnings.simplefilter('error',pd.errors.PerformanceWarning)
            actual=new.read_segment(segment)
        pd.testing.assert_frame_equal(actual,old)
        names=[n for n in actual if n.startswith('ar_')]
        summaries[segment]=dict(rows=len(actual),conditional_fields=len(names),
            changed_cells=sum(int((~np.isclose(actual[n],actual[n[3:]],equal_nan=True,atol=1e-12,rtol=0)).sum()) for n in names),
            legacy_performance_warnings=sum(issubclass(w.category,pd.errors.PerformanceWarning) for w in caught),
            v2_performance_warnings=0)
    if sum(s['rows'] for s in summaries.values())!=100704:
        raise ValueError('ORIGINAL_POPULATION_CHANGED')
    result=dict(status='PASS_BATCH_READER_IDENTICAL_TO_V1_NOT_FIT',tests=tests.testsRun,
        test_output=log.getvalue(),segments=summaries,
        reader_sha256=build.old.prep.sha(build.HERE/'read_conditional_board_observations_v2.py'),
        auditor_sha256=build.old.prep.sha(build.HERE/'audit_conditional_board_reader_v2.py'),
        prior_acceptance_sha256=build.old.prep.sha(build.HERE/'conditional_board_observations_acceptance_20261018_v1.json'),
        old_reader_preserved=True,fit=False,http_requests=0)
    body=build.old.prep.encoded(result)
    used=sum(p.stat().st_size for p in build.OUT.iterdir() if p.is_file())
    if used+len(body)+build.RESERVE>build.CAP or build.old.bytes_used()+len(body)+build.RESERVE>build.old.TOTAL_CAP:
        raise ValueError('READER_ACCEPTANCE_BUDGET_STOP')
    with destination.open('xb') as stream:
        stream.write(body)
    print(json.dumps({k:v for k,v in result.items() if k!='test_output'}))


if __name__=='__main__':
    run()
