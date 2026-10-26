"""Freeze technical evidence-fetch entry and an unsigned approval template; zero network."""
from datetime import datetime,timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform

from expansion_first30_v1 import HERE,REQUEST,MANIFEST,CONTRACT,LIMITS,OUTPUT,require,verify_inputs
from price_verification_guarded_run import encoded,sha
from research_budget_v2 import bytes_used,TOTAL_CAP


def run():
    draft=HERE/'contracts/r0_first30_fetch_authorization_draft_v2.json'
    require(not MANIFEST.exists() and not draft.exists(),'FREEZE_EXISTS_NO_OVERWRITE')
    require(not OUTPUT.exists(),'PRODUCTION_BATCH_ALREADY_EXISTS')
    audit=json.loads((HERE/'population_expansion_plan_acceptance_v2.json').read_bytes())
    require(sha(REQUEST)==audit['pilot_draft_sha256'],'ORIGINAL_PILOT_REQUEST_DRIFT')
    plan=HERE/'risk_population_expansion_plan_v2_20261015'
    refs=json.loads((plan/'input_manifest.json').read_bytes())['inputs']
    verify_inputs(refs)
    old=json.loads((HERE/'contracts/crossyear_risk_manifest_v1.json').read_bytes())['inputs']
    verify_inputs(old)
    files={Path(r['path']) for r in refs+old}
    files.update(p for p in plan.iterdir() if p.is_file())
    files.update((REQUEST,CONTRACT,Path(__file__),HERE/'expansion_first30_v1.py',
        HERE/'test_expansion_first30_v1.py',HERE/'sina_div_client.py',HERE/'test_sina_div_client.py',
        HERE/'population_expansion_plan_acceptance_v2.json',HERE/'audit_risk_population_plan_v2.py'))
    files.update((HERE/'FIRST30_TECHNICAL_FREEZE_V1_WITHDRAWAL.md',
        HERE/'contracts/r0_first30_fetch_manifest_v1.json',HERE/'contracts/r0_first30_fetch_authorization_draft_v1.json',
        HERE/'expansion_first30_entry_acceptance_v1.json'))
    files.update(p for p in (HERE/'_first30_entry_freeze_v1_withdrawn').iterdir() if p.is_file())
    bindings=[dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
    data=encoded(dict(schema='R0_FIRST30_FETCH_MANIFEST_V1',status='TECHNICALLY_FROZEN_NOT_AUTHORIZED',
        frozen_at=datetime.now(timezone.utc).isoformat(),request_sha256=sha(REQUEST),contract_sha256=sha(CONTRACT),
        limits=LIMITS,python_version=platform.python_version(),
        runtime_versions={p:version(p) for p in ('numpy','pandas','lxml')},inputs=bindings))
    template=encoded(dict(status='DRAFT',scope='INDEPENDENT_FETCH_R0_FIRST30_V1',signer=None,
        authorized_at=None,approval_reference=None,request_sha256=sha(REQUEST),
        manifest_sha256=sha256(data).hexdigest(),contract_sha256=sha(CONTRACT),
        note='Separate actual user approval required; preserve this draft. No DATA_BUILD/FIT authorization.'))
    require(len(data)+len(template)<=1024**2 and bytes_used()+len(data)+len(template)<=TOTAL_CAP,'FREEZE_BUDGET')
    verify_inputs(bindings)
    for path,body in ((MANIFEST,data),(draft,template)):
        with path.open('xb') as f:
            f.write(body)
    print(json.dumps(dict(manifest=str(MANIFEST),sha256=sha(MANIFEST),frozen_inputs=len(bindings),
        fetch_authorized=False,network_requests=0),ensure_ascii=False))


if __name__=='__main__':
    run()
