"""Independent read-only acceptance: reuse F3 predictions, never fit any model."""
import json
from pathlib import Path

import numpy as np

from price_verification_guarded_run import encoded, sha
from research_budget_v2 import bytes_used, TOTAL_CAP
from risk_crossyear_v1 import HERE, MANIFEST, OUTPUT, preflight, authorize, require, verify_refs
from risk_crossyear_eval_v1 import evaluate_test


def run():
    destination = HERE/'risk_crossyear_entry_acceptance_v1.json'
    require(not destination.exists(), 'ACCEPTANCE_EXISTS_NO_OVERWRITE')
    config, base, groups, reused, check = preflight()
    require(authorize(None,check)[0] is False and authorize(
        HERE/'contracts/crossyear_risk_authorization_draft_v1.json',check)[0] is False,
        'DRAFT_OR_ABSENT_AUTHORIZATION_ACCEPTED')
    p = {k:np.array([r[k] for r in reused]) for k in ('lr_raw','lr','lightgbm_raw','lightgbm')}
    result = evaluate_test(groups['F3']['test'],p,reused[0]['p_constant'],base['evaluation'])
    old = json.loads((HERE/config['reuse_run']/'report.json').read_bytes())['segments']['test']
    # Reproduce every pre-existing evaluation item before accepting additional comparisons.
    for key in ('n','dates','unfiltered_tail_rate','unfiltered_mean_mae','probability','filtering','risk_groups','bootstrap'):
        require(result[key] == old[key], 'F3_EXISTING_METRIC_DRIFT:'+key)
    manifest = json.loads(MANIFEST.read_bytes())
    verify_refs(manifest['inputs'])
    require(not OUTPUT.exists(), 'REAL_TRAINING_DIRECTORY_CREATED')
    report = dict(status='ENGINEERING_ENTRY_ACCEPTED_PENDING_SEPARATE_FIT_AUTHORIZATION',
        data_ready=True, fit_ready=False, fit_authorized=False, actual_or_synthetic_fit_performed=False,
        network_requests=0, all_folds_sample_gate_passed=True, frozen_inputs_verified=len(manifest['inputs']),
        manifest_sha256=sha(MANIFEST), original_f3_metrics_reproduced_exactly=True,
        f3_new_readonly_comparisons=result['paired_additions'], f3_same_day_ranking=result['same_day_ranking'],
        new_output_cap=config['new_output_cap'], registered_actual_bytes_before=bytes_used(),
        cumulative_cap=TOTAL_CAP, original_f3_run_untouched=True, training_directory_exists=False,
        audit_code=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        note='New comparisons are post-label historical research; no new F1/F2 predictions or fit have been created.')
    data = encoded(report)
    require(len(data) <= 2*1024**2 and bytes_used()+len(data) <= TOTAL_CAP, 'ACCEPTANCE_BUDGET')
    with destination.open('xb') as f:
        f.write(data)
    print(json.dumps(dict(output=str(destination), f3_metrics_match=True, fit_ready=False,
        frozen_inputs=len(manifest['inputs']), f3_new_comparisons=result['paired_additions']),ensure_ascii=False))


if __name__ == '__main__':
    run()
