"""Freeze technical inputs and a NON-authorizing template. Never creates a FIT approval."""
from datetime import datetime, timezone
from importlib.metadata import version
import json
from hashlib import sha256
from pathlib import Path

from price_verification_guarded_run import encoded, sha
from research_budget_v2 import bytes_used, TOTAL_CAP
from risk_crossyear_v1 import CONFIG, MANIFEST, CONTRACT, HERE, PACKAGES, load_data, require, verify_refs


def run():
    auth = HERE/'contracts/crossyear_risk_authorization_draft_v1.json'
    require(not MANIFEST.exists() and not auth.exists(), 'FREEZE_EXISTS_NO_OVERWRITE')
    config, _, _, _, inventories, fingerprints = load_data()
    old = json.loads((HERE/'risk_failure_diagnostics_v1_20261015/input_manifest.json').read_bytes())
    verify_refs(old['inputs'])
    files = {Path(r['path']) for r in old['inputs']}
    files.update((CONFIG, CONTRACT, Path(__file__), HERE/'risk_crossyear_v1.py',
        HERE/'risk_crossyear_eval_v1.py', HERE/'test_risk_crossyear_v1.py',
        HERE/'risk_failure_diagnostics_v1_20261015/fold_inventory.json',
        HERE/'risk_failure_diagnostics_v1_20261015/stop_record.json'))
    files.update(p for p in (HERE/config['reuse_run']).iterdir() if p.is_file())
    refs = [dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(files)]
    manifest = dict(schema='R0_CROSSYEAR_MANIFEST_V1', status='TECHNICALLY_FROZEN_NOT_FIT_AUTHORIZATION',
        frozen_at=datetime.now(timezone.utc).isoformat(), config_sha256=sha(CONFIG),
        contract_sha256=sha(CONTRACT), runtime_versions={p:version(p) for p in PACKAGES},
        inputs=refs, fold_inventory=inventories, fold_populations=fingerprints)
    data = encoded(manifest)
    template = encoded(dict(status='DRAFT', scope='FIT_R0_CROSSYEAR_V1', signer=None,
        authorized_at=None, approval_reference=None, new_fit_folds=['F1','F2'], reuse_only_folds=['F3'],
        manifest_sha256=sha256(data).hexdigest(), config_sha256=sha(CONFIG), contract_sha256=sha(CONTRACT),
        note='Separate user FIT approval required; preserve this draft and create a separate authorized record.'))
    require(len(data)+len(template) <= 1024**2 and bytes_used()+len(data)+len(template) <= TOTAL_CAP,
            'FREEZE_BUDGET_STOP')
    verify_refs(refs)
    for path, content in ((MANIFEST,data),(auth,template)):
        with path.open('xb') as f:
            f.write(content)
    print(json.dumps(dict(manifest=str(MANIFEST), frozen_inputs=len(refs), sha256=sha(MANIFEST),
        fit_authorized=False, fit_performed=False), ensure_ascii=False))


if __name__ == '__main__':
    run()
