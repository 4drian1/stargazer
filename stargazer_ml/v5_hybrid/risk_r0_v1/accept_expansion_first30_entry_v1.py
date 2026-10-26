"""Offline acceptance with socket connections forbidden. No production fetch or dataset build."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import expansion_first30_v1 as app
from build_archived_corpus_v3 import frozen_pages
from price_verification_guarded_run import encoded,sha
from research_budget_v2 import bytes_used,TOTAL_CAP
from test_expansion_first30_v1 import Stream


def run():
    destination=app.HERE/'expansion_first30_entry_acceptance_v2.json'
    app.require(not destination.exists(),'ACCEPTANCE_EXISTS_NO_OVERWRITE')
    with patch('socket.create_connection',side_effect=AssertionError('NETWORK_FORBIDDEN_IN_OFFLINE_ACCEPTANCE')):
        codes,manifest,check=app.preflight()
        cli={}
        for command in ('check','fetch'):
            with patch.object(app,'Batch') as constructor,patch.object(app,'open_response') as transport, \
                 redirect_stdout(io.StringIO()) as captured:
                rc=app.main([command])
                constructor.assert_not_called()
                transport.assert_not_called()
            data=json.loads(captured.getvalue())
            app.require(rc==2 and data['fetch_ready'] is False,'UNAUTHORIZED_CLI_NOT_BLOCKED')
            cli[command]=dict(exit_code=rc,blocked=data['blocked'],production_constructor_called=False)
        archive=[]
        for code,(raw,meta) in frozen_pages().items():
            parsed=app.validate_page(raw.read_bytes(),code)
            archive.append(dict(code=code,raw_sha256=sha(raw),meta_sha256=sha(meta),
                bonus_rows=parsed['bonus_rows'],allotment_rows=parsed['allotment_rows'],
                status='ARCHIVED_PAGE_MECHANISM_PASS_NOT_NEW30_EVIDENCE'))
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            state={'clock':0.,'calls':0}
            def sleep(n):
                state['clock']+=n
            def opener(url,timeout):
                state['calls']+=1
                return Stream(url.split('/')[-1].split('.')[0])
            batch=app.Batch(base/'success',app.LIMITS,{'SYNTHETIC_ACCEPTANCE_NOT_REAL_DATA':True},
                opener=opener,budget_used=lambda:sum(p.stat().st_size for p in base.rglob('*') if p.is_file()),
                clock=lambda:state['clock'],sleep=sleep)
            terminal=batch.run(codes)
            app.require(terminal['status']=='completed' and terminal['requests']==state['calls']==30,'MOCK_BATCH_FLOW')
            app.require(all(sha(batch.root/r['name'])==r['sha256'] for r in terminal['files']),'MOCK_EVIDENCE_DRIFT')
            raw_bytes=sum((batch.root/f'{r["seq"]:02d}_{r["code"]}.bin').stat().st_size for r in batch.ledger)
            app.require(raw_bytes==terminal['response_bytes'],'MOCK_BYTE_LEDGER_DRIFT')
            success=dict(status=terminal['status'],requests=terminal['requests'],response_bytes=raw_bytes,
                actual_disk_bytes=batch.actual_bytes(),all_evidence_hashes_verified=True,
                smoke_response_reused=True,fixture_only=True)
    app.verify_inputs(manifest['inputs'])
    app.require(not app.OUTPUT.exists(),'PRODUCTION_DIRECTORY_CREATED')
    report=dict(status='ENGINEERING_ENTRY_ACCEPTED_SEPARATE_FETCH_APPROVAL_REQUIRED',
        network_requests=0,local_http_requests=0,fetch_authorized=False,fit_authorized=False,
        production_directory_exists=False,manifest_sha256=sha(app.MANIFEST),
        request_sha256=sha(app.REQUEST),contract_sha256=sha(app.CONTRACT),frozen_inputs=len(manifest['inputs']),
        cli_guard=cli,archived_pages_checked=archive,mock_full30=success,limits=app.LIMITS,
        registry_bytes_before=bytes_used(),audit_code_sha256=sha(__file__),
        note='Socket connections forbidden. Mock success does not prove current remote availability, new30 coverage or label readiness.')
    data=encoded(report)
    app.require(len(data)<1024**2 and bytes_used()+len(data)<=TOTAL_CAP,'ACCEPTANCE_BUDGET')
    with destination.open('xb') as f:
        f.write(data)
    print(json.dumps(dict(output=str(destination),network_requests=0,archived_pages=len(archive),
        mock_requests=success['requests'],frozen_inputs=report['frozen_inputs'],fetch_authorized=False),ensure_ascii=False))


if __name__=='__main__':
    run()
