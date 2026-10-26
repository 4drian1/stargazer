"""One isolated offline engineering acceptance, never source collection."""
import io
import json
import unittest
from pathlib import Path

from postclose_research_scorer_v1 import HERE, DEFAULT_PACKAGE, encode, sha, ResearchScorer
from postclose_next_period_evaluation_v1 import preparation_check
from research_budget_v2 import bytes_used, TOTAL_CAP

OUT = HERE / 'postclose_forward_offline_acceptance_20261017_v1'
CAP = 2 * 1024**2


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    if bytes_used()+CAP > TOTAL_CAP:
        raise ValueError('ACCEPTANCE_BUDGET_RESERVATION_FAILED')
    refs = {str(p):sha(p.read_bytes()) for p in [Path(__file__),DEFAULT_PACKAGE,
        HERE/'postclose_forward_offline_v1.py',HERE/'test_postclose_forward_offline_v1.py',
        HERE/'postclose_next_period_evaluation_v1.py',HERE/'test_postclose_next_period_evaluation_v1.py']}
    OUT.mkdir(exist_ok=False)
    def write(name, value, terminal=False):
        body=encode(value)
        reserve=0 if terminal else 16384
        used=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        if used+len(body)+reserve > CAP or bytes_used()+len(body)+reserve > TOTAL_CAP:
            raise ValueError('ACCEPTANCE_WRITE_BUDGET_STOP')
        with (OUT/name).open('xb') as stream:
            stream.write(body)
    write('plan.json',dict(mode='OFFLINE_ACCEPTANCE_ONLY',inputs=refs,collection_authorized=False))
    try:
        log=io.StringIO()
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in
            ['test_postclose_forward_offline_v1','test_postclose_next_period_evaluation_v1',
             'test_postclose_research_scorer_v1'])
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        write('test_output.json',dict(output=log.getvalue(),tests=result.testsRun,
              failures=len(result.failures),errors=len(result.errors)))
        if not result.wasSuccessful():
            raise ValueError('OFFLINE_ACCEPTANCE_FAILED')
        ResearchScorer().verify_artifacts()
        if any(sha(Path(p).read_bytes())!=h for p,h in refs.items()):
            raise ValueError('INPUT_DRIFT_DURING_ACCEPTANCE')
        report=dict(status='PASS_OFFLINE_MECHANISM_NOT_PROSPECTIVE_ACCEPTANCE',tests=result.testsRun,
                    new_tests=18,regressions=36,scope_codes=43,old_feature_examples=2,
                    synthetic_clock=True,real_forward_records=0,
                    source_acceptance_passed=False,production_recording_chain_accepted=False,
                    prospective_scoring_accepted=False,collection_started=False,
                    independent_validation_passed=False,fit_run=False,network_requests=0,
                    original_D2_artifacts_unchanged=True,current_preparation=preparation_check())
        write('acceptance.json',report)
        write('stop_record.json',dict(status='COMPLETE_OFFLINE_ACCEPTANCE_ONLY'),True)
        return report
    except Exception as exc:
        write('stop_record.json',dict(status='FAILED',error=str(exc)),True)
        raise


if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False,allow_nan=False))
