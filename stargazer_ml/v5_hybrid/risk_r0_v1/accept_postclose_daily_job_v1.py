"""Offline/manual-entry acceptance. No scheduler or real new-date collection."""
import io
import unittest
from pathlib import Path

from postclose_daily_job_v1 import plan,run
from postclose_research_scorer_v1 import HERE,ResearchScorer,encode,sha
from research_budget_v2 import bytes_used,TOTAL_CAP

OUT=HERE/'postclose_daily_job_acceptance_20261018_v1'
CAP=2*1024**2


def accept():
    if OUT.exists():raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    if bytes_used()+CAP>TOTAL_CAP:raise ValueError('ACCEPTANCE_BUDGET_RESERVATION_FAILED')
    refs={str(p):sha(p.read_bytes()) for p in [Path(__file__),HERE/'postclose_daily_job_v1.py',
        HERE/'test_postclose_daily_job_v1.py',HERE/'contracts/RISK_ONLY_EVALUATION_BOUNDARY_20261018.json']}
    OUT.mkdir(exist_ok=False)
    def write(name,value,terminal=False):
        body=encode(value);reserve=0 if terminal else 16384
        current=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        if current+len(body)+reserve>CAP or bytes_used()+len(body)+reserve>TOTAL_CAP:raise ValueError('ACCEPTANCE_WRITE_BUDGET_STOP')
        with (OUT/name).open('xb') as stream:stream.write(body)
    write('plan.json',dict(inputs=refs,mode='OFFLINE_MANUAL_ENTRY_ACCEPTANCE',real_new_date_fetch=False))
    try:
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in
            ['test_postclose_daily_job_v1','test_refresh_postclose_incremental_v1',
             'test_refresh_postclose_incremental_v2','test_read_postclose_incremental_v2',
             'test_postclose_local_source_bridge_v1','test_postclose_forward_offline_v1',
             'test_postclose_next_period_evaluation_v1','test_postclose_research_scorer_v1'])
        log=io.StringIO();tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        write('tests.json',dict(tests=tests.testsRun,failures=len(tests.failures),errors=len(tests.errors),output=log.getvalue()))
        if not tests.wasSuccessful():raise ValueError('DAILY_JOB_TESTS_FAILED')
        cached=run(20261016);future=plan(20261019)[0]
        ResearchScorer().verify_artifacts()
        if any(sha(Path(p).read_bytes())!=h for p,h in refs.items()):raise ValueError('ACCEPTANCE_INPUT_DRIFT')
        report=dict(status='PASS_MANUAL_DAILY_JOB_ENGINEERING_NOT_DEPLOYMENT',tests=tests.testsRun,
            cached_day=20261016,cached_scores=sum(r['status']=='RESEARCH_SCORE' for r in cached['rows']),
            cache_reused=cached['cache_reused'],real_http_requests=0,
            future_day_check=future,new_date_execution_test='MOCK_TRANSPORT_TEMPORARY_DIRECTORY_ONLY',
            trading_profit_required=False,risk_validation_still_required=True,
            numeric_risk_thresholds_approved=False,scheduler_installed=False,
            original_D2_unchanged=True,fit_run=False,independent_validation_passed=False)
        write('acceptance.json',report);write('stop_record.json',dict(status='COMPLETE_OFFLINE_ACCEPTANCE_ONLY'),True)
        return report
    except Exception as exc:
        write('stop_record.json',dict(status='FAILED',error=str(exc)),True)
        raise


if __name__=='__main__':print(accept())
