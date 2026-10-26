"""One real local-cache replay with exclusive evidence; no source collection."""
import io
import json
import unittest
from pathlib import Path

from postclose_local_source_bridge_v1 import observe_existing_cache
from postclose_research_scorer_v1 import HERE,ResearchScorer,DEFAULT_PACKAGE,encode,sha,strict_json
from research_budget_v2 import bytes_used,TOTAL_CAP

OUT=HERE/'postclose_local_source_bridge_20261017_v1'
CAP=2*1024**2


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    if bytes_used()+CAP>TOTAL_CAP:
        raise ValueError('SOURCE_BRIDGE_BUDGET_RESERVATION_FAILED')
    refs={str(p):sha(p.read_bytes()) for p in [Path(__file__),DEFAULT_PACKAGE,
        HERE/'postclose_local_source_bridge_v1.py',HERE/'test_postclose_local_source_bridge_v1.py']}
    OUT.mkdir(exist_ok=False)
    def write(name,value,terminal=False):
        body=encode(value);reserve=0 if terminal else 16384
        current=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        if current+len(body)+reserve>CAP or bytes_used()+len(body)+reserve>TOTAL_CAP:
            raise ValueError('SOURCE_BRIDGE_WRITE_BUDGET_STOP')
        with (OUT/name).open('xb') as stream:
            stream.write(body)
    write('plan.json',dict(mode='REAL_LOCAL_CACHE_REPLAY_NOT_PROSPECTIVE',inputs=refs,
          collection_authorized=False,fit_authorized=False))
    try:
        log=io.StringIO()
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in
            ['test_postclose_local_source_bridge_v1','test_postclose_forward_offline_v1',
             'test_postclose_next_period_evaluation_v1','test_postclose_research_scorer_v1'])
        tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        write('tests.json',dict(output=log.getvalue(),tests=tests.testsRun,
              failures=len(tests.failures),errors=len(tests.errors)))
        if not tests.wasSuccessful():
            raise ValueError('SOURCE_BRIDGE_TESTS_FAILED')
        content=observe_existing_cache()
        envelope=dict(content=content,content_sha256=sha(encode(content)))
        write('observation.json',envelope)
        readback=strict_json((OUT/'observation.json').read_bytes())
        if sha(encode(readback['content']))!=readback['content_sha256'] or readback!=envelope:
            raise ValueError('OBSERVATION_READBACK_FAILED')
        for item in content['receipts']:
            if sha(Path(item['path']).read_bytes())!=item['sha256']:
                raise ValueError('SOURCE_CHANGED_DURING_OBSERVATION')
        ResearchScorer().verify_artifacts()
        if any(sha(Path(p).read_bytes())!=h for p,h in refs.items()):
            raise ValueError('CODE_DRIFT_DURING_OBSERVATION')
        rows=content['scores']['rows']
        report=dict(status='PASS_REAL_LOCAL_CACHE_WIRING_NOT_LIVE_ACCEPTANCE',tests=tests.testsRun,
                    source_day=content['source_day'],source_population_rows=content['source_population_rows'],
                    output_rows=len(rows),research_scored=sum(r['status']=='RESEARCH_SCORE' for r in rows),
                    unscorable=sum(r['status']=='UNSCORABLE' for r in rows),
                    source_files=len(content['receipts']),labels_projected=False,
                    real_observation_clock=True,original_arrival_known=False,
                    real_forward_records=0,live_incremental_features_accepted=False,
                    original_D2_unchanged=True,network_requests=0,fit_run=False,
                    observation_sha256=sha((OUT/'observation.json').read_bytes()),
                    independent_validation_passed=False,deployment_authorized=False)
        write('acceptance.json',report)
        write('stop_record.json',dict(status='COMPLETE_LOCAL_REPLAY_NOT_FORWARD_VALIDATION'),True)
        return report
    except Exception as exc:
        write('stop_record.json',dict(status='FAILED',error=str(exc)),True)
        raise


if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False,allow_nan=False))
