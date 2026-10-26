"""Read-only evidence/ledger audit after the one authorized batch. No requests or labels."""
from datetime import datetime
import json
from pathlib import Path

from expansion_first30_v1 import HERE,OUTPUT,MANIFEST,REQUEST,CONTRACT,LIMITS,URL,validate_page,verify_inputs,require,preflight
from price_verification_guarded_run import encoded,sha
from research_budget_v2 import bytes_used,TOTAL_CAP


def run():
    destination=HERE/'expansion_first30_result_audit_v1.json'
    require(not destination.exists(),'RESULT_AUDIT_EXISTS_NO_OVERWRITE')
    stop=json.loads((OUTPUT/'stop_record.json').read_bytes())
    report=json.loads((OUTPUT/'report.json').read_bytes())
    claim=json.loads((OUTPUT/'claim.json').read_bytes())
    index=json.loads((OUTPUT/'index.json').read_bytes())['records']
    request=json.loads(REQUEST.read_bytes())
    manifest=json.loads(MANIFEST.read_bytes())
    verify_inputs(manifest['inputs'])
    verify_inputs([dict(path=str(OUTPUT/r['name']),sha256=r['sha256']) for r in stop['files']])
    require({p.name for p in OUTPUT.iterdir()}=={r['name'] for r in stop['files']}|{'stop_record.json'},'UNLISTED_BATCH_ARTIFACT')
    for ref in stop['files']:
        require((OUTPUT/ref['name']).stat().st_size==ref['bytes'],'ARTIFACT_SIZE_DRIFT')
    require(claim['limits']==report['limits']==LIMITS,'LIMITS_DRIFT')
    context=claim['context']
    require(sha(context['authorization_path'])==context['authorization_sha256'],'AUTHORIZATION_DRIFT')
    auth=json.loads(Path(context['authorization_path']).read_bytes())
    require(auth['status']=='AUTHORIZED' and auth['scope']=='INDEPENDENT_FETCH_R0_FIRST30_V1','AUTHORIZATION_SCOPE')
    require(auth['manifest_sha256']==context['check']['manifest_sha256']==sha(MANIFEST)
        and auth['request_sha256']==sha(REQUEST) and auth['contract_sha256']==sha(CONTRACT),'AUTHORIZATION_BINDING')
    require(stop['status']==report['status']=='completed' and not stop['error'] and not report['error'],'BATCH_NOT_COMPLETED')
    require(stop['requests']==report['request_attempts']==len(report['ledger'])==len(index)==30,'REQUEST_LEDGER_MISMATCH')
    times=[]
    pages=[]
    total=0
    for i,(expected,meta,record) in enumerate(zip(request['codes'],report['ledger'],index),1):
        code=expected['code']
        attempt=json.loads((OUTPUT/f'{i:02d}_{code}.attempt.json').read_bytes())
        stored=json.loads((OUTPUT/f'{i:02d}_{code}.json').read_bytes())
        raw=OUTPUT/f'{i:02d}_{code}.bin'
        require(meta==stored and all(attempt[k]==meta[k] for k in attempt),'SIDECAR_OR_ATTEMPT_DRIFT')
        require(meta['seq']==record['seq']==i and meta['code']==record['code']==code,'REQUEST_SEQUENCE')
        require(record['raw_file']==raw.name and record['meta_file']==f'{i:02d}_{code}.json','INDEX_FILENAME')
        require(meta['sha256']==record['sha256']==sha(raw) and raw.stat().st_size==meta['raw_bytes'],'RAW_BYTES_OR_HASH')
        require(meta['method']=='GET' and meta['url']==URL.format(code) and meta['status']==200
                and not meta['error'] and meta['smoke']==(i<=2),'HTTP_OR_SMOKE_IDENTITY')
        parsed=validate_page(raw.read_bytes(),code)
        require(all(meta[k]==v for k,v in parsed.items()),'PAGE_MECHANISM_READBACK')
        start=datetime.fromisoformat(meta['started_at'])
        end=datetime.fromisoformat(meta['finished_at'])
        require(start<=end and start>=datetime.fromisoformat(auth['authorized_at']),'TIME_ORDER')
        times.append(start)
        total+=raw.stat().st_size
        pages.append(dict(code=code,status=meta['status'],bonus_source_rows=parsed['bonus_rows'],
            allotment_source_rows=parsed['allotment_rows'],
            note='Source table rows including placeholders/non-implemented rows; NOT confirmed events.',
            raw_bytes=meta['raw_bytes'],sha256=meta['sha256'],mechanism_verified=True,
            event_coverage_verified=False,data_admission=False))
    require([r['code'] for r in report['ledger']]==report['codes_attempted'],'ATTEMPT_CODE_LEDGER')
    require(total==report['response_bytes']==stop['response_bytes']<=LIMITS['response'],'RESPONSE_BODY_LEDGER')
    disk=sum(p.stat().st_size for p in OUTPUT.iterdir() if p.is_file())
    require(disk<=LIMITS['disk'],'ACTUAL_BATCH_DISK_LIMIT')
    intervals=[(b-a).total_seconds() for a,b in zip(times,times[1:])]
    require(min(intervals)>=1,'START_INTERVAL')
    try:
        preflight()
    except ValueError as exc:
        require(str(exc)=='BATCH_DIRECTORY_EXISTS_NO_RESTART','UNEXPECTED_FINALITY_GUARD')
    else:
        raise ValueError('FINAL_BATCH_WOULD_RESTART')
    verify_inputs(manifest['inputs'])
    result=dict(status='FIRST30_EVIDENCE_LEDGER_AUDIT_PASS_NOT_LABEL_ADMISSION',batch_status='completed',
        http_200_responses=30,request_attempts=30,smoke_requests=2,smoke_in_total=True,
        response_body_bytes=total,batch_actual_disk_bytes=disk,batch_disk_cap=LIMITS['disk'],
        response_cap=LIMITS['response'],frozen_inputs_verified=len(manifest['inputs']),
        min_recorded_start_interval_seconds=min(intervals),all_raw_sidecar_attempt_index_hashes_verified=True,
        stopped_batch_refuses_restart=True,additional_network_requests_in_audit=0,
        local_http_requests=0,labels_generated=0,fit_performed=False,event_coverage_verified=False,
        population_training_ready=False,pages=pages,registered_actual_bytes_before_audit=bytes_used(),
        audit_code_sha256=sha(__file__),batch_stop_sha256=sha(OUTPUT/'stop_record.json'))
    data=encoded(result)
    require(disk+len(data)<=LIMITS['disk'] and bytes_used()+len(data)<=TOTAL_CAP,'AUDIT_OUTPUT_BUDGET')
    with destination.open('xb') as f:
        f.write(data)
    print(json.dumps({k:v for k,v in result.items() if k!='pages'},ensure_ascii=False))


if __name__=='__main__':
    run()
