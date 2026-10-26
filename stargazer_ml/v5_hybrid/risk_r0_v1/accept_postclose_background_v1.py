"""Real native-file background readback and offline pipeline regression acceptance."""
import io
import unittest
from pathlib import Path

import postclose_daily_update_v1 as update
from postclose_research_scorer_v1 import HERE,ResearchScorer,encode,sha,strict_json
from postclose_daily_job_v1 import read_input,run
from research_budget_v2 import bytes_used,TOTAL_CAP

OUT=HERE/'postclose_background_acceptance_20261018_v1'
CAP=2*1024**2


def accept():
    if OUT.exists():raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    if bytes_used()+4*1024**2>TOTAL_CAP:raise ValueError('BACKGROUND_ACCEPTANCE_RESERVATION_FAILED')
    refs={str(p):sha(p.read_bytes()) for p in [Path(__file__),HERE/'build_postclose_background_v1.py',
        HERE/'postclose_daily_update_v1.py',HERE/'test_build_postclose_background_v1.py',
        HERE/'test_postclose_daily_update_v1.py']}
    OUT.mkdir(exist_ok=False)
    def write(name,value,terminal=False):
        body=encode(value);reserve=0 if terminal else 16384
        used=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        if used+len(body)+reserve>CAP or bytes_used()+len(body)+reserve>TOTAL_CAP:raise ValueError('BACKGROUND_ACCEPTANCE_BUDGET_STOP')
        with (OUT/name).open('xb') as stream:stream.write(body)
    write('plan.json',dict(inputs=refs,mode='NATIVE_LOCAL_BACKGROUND_ACCEPTANCE',new_http=False,fit_authorized=False))
    try:
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in
            ['test_build_postclose_background_v1','test_postclose_daily_update_v1',
             'test_postclose_daily_cli_v1','test_postclose_daily_job_v1',
             'test_refresh_postclose_incremental_v1','test_refresh_postclose_incremental_v2',
             'test_read_postclose_incremental_v2','test_postclose_local_source_bridge_v1',
             'test_postclose_forward_offline_v1','test_postclose_next_period_evaluation_v1',
             'test_postclose_research_scorer_v1'])
        log=io.StringIO();tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        write('tests.json',dict(tests=tests.testsRun,failures=len(tests.failures),errors=len(tests.errors),output=log.getvalue()))
        if not tests.wasSuccessful():raise ValueError('BACKGROUND_ACCEPTANCE_TESTS_FAILED')
        prepared=update.prepare(20261016,write=True)
        background,_=read_input(prepared['background'],'POSTCLOSE_MARKET_GOLDEN')
        calendar,_=read_input(prepared['calendar'],'TRADING_CALENDAR')
        cached=run(20261016)
        refused=None
        try:update.prepare(20261019)
        except ValueError as exc:refused=str(exc)
        if refused is None or 'NO_FALLBACK' not in refused:raise ValueError('MISSING_DAY_NOT_REFUSED')
        for p,h in refs.items():
            if sha(Path(p).read_bytes())!=h:raise ValueError('BACKGROUND_ACCEPTANCE_CODE_DRIFT')
        for item in background['source_refs']:
            if sha(Path(item['path']).read_bytes())!=item['sha256']:raise ValueError('BACKGROUND_ACCEPTANCE_SOURCE_DRIFT')
        ResearchScorer().verify_artifacts()
        report=dict(status='PASS_NATIVE_MARKET_GOLDEN_TO_DAILY_INPUT_NOT_PROSPECTIVE_GATE',tests=tests.testsRun,
            actual_source_day=20261016,source_days=len(calendar['days']),fields=len(background['features']),
            same_day_background_bound=True,historical_parity_days=448,historical_parity_fields=78,
            actual_cached_scored=sum(r['status']=='RESEARCH_SCORE' for r in cached['rows']),
            missing_day_refusal=refused,observed_calendar_not_exchange_certified=True,
            source_files_auto_updated=False,original_availability_known=False,
            http_requests=0,fit_run=False,scheduler_installed=False,independent_validation_passed=False,
            output_refs={p.name:sha(p.read_bytes()) for p in (update.INPUTS/'20261016').iterdir() if p.is_file()})
        write('acceptance.json',report);write('stop_record.json',dict(status='COMPLETE_LOCAL_BACKGROUND_ACCEPTANCE'),True)
        return report
    except Exception as exc:
        write('stop_record.json',dict(status='FAILED',error=str(exc)),True)
        raise


if __name__=='__main__':print(accept())
