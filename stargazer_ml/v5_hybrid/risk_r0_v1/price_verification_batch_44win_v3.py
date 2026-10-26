"""REV3 CLI, reusing unchanged REV2 price and client contracts."""
import argparse
import json
from pathlib import Path
import sys

from price_verification_batch_44win import (
    HERE, BATCH, REQUEST_CAP, BYTE_CAP, REPORT_RESERVE, BSE_GAP, TOL,
    TencentDailyClient, FetchCounter, ByteCounter, compare_point,
    boundary_sensitivity, plan_from_points, load_local_prices, tencent_rows_to_map,
)
from price_verification_guarded_run_v3 import execute, verify_bundle, POLICY
from price_verification_guarded_run import sha

MANIFEST = BATCH / 'price_verification_manifest_v3.json'


def dry_run(manifest_path=MANIFEST):
    manifest, windows, points, local = verify_bundle(manifest_path, sys.modules[__name__])
    result = {'status':'REV3_ZERO_NETWORK_PREFLIGHT_PASS_AUTHORIZATION_REQUIRED',
              'policy':POLICY,'windows':len(windows),'references':6*len(windows),
              'unique_points':len(points),'stocks':len({p['code'] for p in points}),
              'inputs':len(manifest['inputs']),'manifest_sha256':sha(manifest_path),
              'normal_requests':30,'request_cap':REQUEST_CAP,'retries':0,
              'response_cap':BYTE_CAP,'disk_cap':BYTE_CAP,'report_reserve':REPORT_RESERVE,
              'network_authorized':False,'label_build_authorized':False,'fit_authorized':False}
    print(json.dumps(result,ensure_ascii=False))
    return result


def fetch_run(output, authorization=None, authorization_sha256=None, manifest_path=MANIFEST):
    return execute(output,authorization,authorization_sha256,manifest_path,sys.modules[__name__])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REV3：额外历史仅归档；标签/训练不授权')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--dry-run',action='store_true')
    modes.add_argument('--fetch',metavar='NEW_APPROVED_BATCH_DIRECTORY')
    parser.add_argument('--manifest',type=Path,default=MANIFEST)
    parser.add_argument('--authorization')
    parser.add_argument('--authorization-sha256')
    args = parser.parse_args()
    try:
        if args.dry_run:
            dry_run(args.manifest)
        else:
            result = fetch_run(args.fetch,args.authorization,args.authorization_sha256,args.manifest)
            print(json.dumps(result,ensure_ascii=False,allow_nan=False))
            raise SystemExit(0 if result['stop_reason'] == 'completed' else 2)
    except (ValueError,OSError,KeyError) as exc:
        print(json.dumps({'status':'BLOCKED','reason':str(exc),'network_authorized':False},ensure_ascii=False))
        raise SystemExit(2)
