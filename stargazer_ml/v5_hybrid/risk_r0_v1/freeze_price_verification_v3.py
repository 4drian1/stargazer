"""Freeze only new REV3 metadata; retain the REV2 point set and source bytes."""
import json
from pathlib import Path
import sys

import price_verification_batch_44win_v3 as api
from price_verification_guarded_run import encoded, sha, verify_bundle as verify_v2
from price_verification_guarded_run_v3 import POLICY, verify_bundle


def main():
    parent = api.BATCH / 'price_verification_manifest_v2.json'
    target = api.MANIFEST
    if target.exists():
        raise ValueError('REV3_MANIFEST_EXISTS_NO_OVERWRITE')
    manifest,windows,points,local = verify_v2(parent,api)
    # Copy metadata in memory, not large data inputs.
    manifest['runtime_revision'] = 3
    manifest['response_scope_policy'] = POLICY
    manifest['parent_manifest_sha256'] = sha(parent)
    extra = {
        'rev3_runner_code':Path(api.__file__),
        'rev3_guard_code':api.HERE / 'price_verification_guarded_run_v3.py',
        'rev3_freeze_code':Path(__file__),
        'rev3_request_order':api.HERE / 'price_verification_request_order_44win_REV3.md',
        'parent_manifest':parent,
    }
    for role,path in extra.items():
        path = path.resolve()
        manifest['inputs'].append({'role':role,'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)})
    manifest['network_authorized'] = False
    manifest['label_build_authorized'] = False
    manifest['fit_authorized'] = False
    raw = encoded(manifest)
    if len(raw) > 1024*1024:
        raise ValueError('ZERO_NETWORK_ARTIFACT_BUDGET_STOP')
    with target.open('xb') as stream:
        stream.write(raw)
    verify_bundle(target,api)
    print(json.dumps({'status':'REV3_FROZEN_NOT_AUTHORIZED','windows':len(windows),
                      'points':len(points),'input_files':len(manifest['inputs']),
                      'new_bytes':len(raw),'sha256':sha(target)},ensure_ascii=False))


if __name__ == '__main__':
    main()
