"""Freeze one technical draft, not a signature or permission to run a real fit."""
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

from price_verification_guarded_run import encoded, sha
from research_budget_v2 import bytes_used, TOTAL_CAP
from risk_fit_v1 import CONFIG, CONTRACT, HERE, PACKAGES


def run():
    config = json.loads(CONFIG.read_bytes())
    manifest_path = HERE / config['manifest']
    auth_path = HERE / 'contracts/restricted_risk_fit_authorization_draft_v1.json'
    if manifest_path.exists() or auth_path.exists():
        raise ValueError('FREEZE_EXISTS_NO_OVERWRITE')
    acceptance_path = HERE / config['acceptance']
    acceptance = json.loads(acceptance_path.read_bytes())
    files = {CONFIG, CONTRACT, acceptance_path, Path(__file__)}
    files.update(HERE / n for n in ('risk_fit_v1.py', 'risk_fit_eval_v1.py', 'test_risk_fit_v1.py',
                 'historical_risk_core_v1.py', 'research_budget_v2.py', 'prepare_historical_risk_v1.py',
                 'price_verification_guarded_run.py', 'contracts/research_storage_4gib_20261015.json'))
    files.update(Path(r['path']) for r in acceptance['matrix_files'] + acceptance['controls'] + [acceptance['audit_code']])
    batch = HERE / 'historical_archived_corpus_v4_20261015'
    plan = json.loads((batch / 'plan.json').read_bytes())
    files.update(Path(r['path']) for r in plan['inputs'])
    files.update(p for p in batch.rglob('*') if p.is_file())
    refs = [dict(path=str(p.resolve()), sha256=sha(p), bytes=p.stat().st_size) for p in sorted(files)]
    manifest = dict(schema='RESTRICTED_R0_FIT_MANIFEST_V1', status='TECHNICAL_DRAFT_NOT_FIT_AUTHORIZATION',
                    frozen_at=datetime.now(timezone.utc).isoformat(), inputs=refs,
                    config_sha256=sha(CONFIG), contract_sha256=sha(CONTRACT),
                    runtime_versions={p: version(p) for p in PACKAGES})
    data = encoded(manifest)
    auth = encoded(dict(status='DRAFT', scope='FIT_RESTRICTED_R0_V1', signer=None, authorized_at=None,
                        approval_reference=None, manifest_sha256=__import__('hashlib').sha256(data).hexdigest(),
                        config_sha256=sha(CONFIG), contract_sha256=sha(CONTRACT),
                        note='Do not toggle config flags. A separate actual user FIT approval is required.'))
    if len(data) + len(auth) > 1024**2 or bytes_used() + len(data) + len(auth) > TOTAL_CAP:
        raise ValueError('FREEZE_BUDGET')
    for path, content in ((manifest_path, data), (auth_path, auth)):
        with path.open('xb') as f:
            f.write(content)
    print(json.dumps(dict(manifest=str(manifest_path), sha256=sha(manifest_path),
                          frozen_inputs=len(refs), fit_authorized=False), ensure_ascii=False))


if __name__ == '__main__':
    run()
