"""One-command fingerprint release (#18.188-W4): check -> switch -> regen -> verify.

Fixed procedure, no hand-ordered steps:
  1. CHECK   read current manifests from disk, verify them, list current versions.
  2. SWITCH  point every flow reference at the successor flow manifest and the
             Tarot reference at the successor Tarot manifest (in ONE pass, so
             cross-manifest files such as tarot_live_v2.py are switched before
             any regeneration reads them).
  3. REGEN   recompute SHA256 for EVERY key from disk (full recompute even on
             add-key rounds) and write both successors; existing versions are
             never overwritten.
  4. VERIFY  run the release gate (verify_live_release); non-zero exit forbids
             packaging.

Usage:  python release_fingerprints_v1.py <new_flow_version> <new_tarot_version>
  e.g.  python release_fingerprints_v1.py v111 v100
Dates are taken from today; version labels must be newer than the current ones.
"""
import gzip
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1'
GCP = ROOT / 'stargazer_ml/v5_hybrid/green_common_profile'
FLOW_REFERENCES = (
    'stargazer_desktop/golden_stock_analysis.py',
    'stargazer_desktop/on_demand_model_worker.py',
    'stargazer_desktop/sector_rule_scan.py',
    'stargazer_ml/v5_hybrid/green_common_profile/market_price_evidence_v1.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/tarot_intraday_risk_v2.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/live_postclose_job_v1.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/accept_live_postclose_v1.py')
FLOW_NAME = 'live_monitor_fingerprints_{date}_{ver}_flow.json.gz'
TAROT_NAME = 'tarot_live_manifest_{ver}_{date}_rel.json.gz'
FLOW_RE = re.compile(r'live_monitor_fingerprints_\d+_v\d+_flow\.json\.gz')
TAROT_RE = re.compile(r'tarot_live_manifest_v\d+_\d+_rel\.json\.gz')
FLOW_REF_RE = re.compile(r'live_monitor_fingerprints_\d+_v(\d+)_flow\.json\.gz')
TAROT_REF_RE = re.compile(r'tarot_live_manifest_v(\d+)_\d+_rel\.json\.gz')
# New modules introduced after the current manifests were frozen; they join
# both successor manifests on the next release (#18.199-补).
NEW_FILES = (
    'stargazer_desktop/persistent_records_v1.py',
    'stargazer_desktop/web/local_storage.html',
    'release_audit_v1/release_fingerprints_v1.py',
    'release_audit_v1/verify_frozen_package_v1.py',
    'stargazer_desktop/storage_roots_v1.py',
    'stargazer_desktop/cache_layer_keys_v1.py',
    'stargazer_desktop/asset_migrate_v1.py',
    'stargazer_desktop/index_map_layers_v1.py',
    'stargazer_desktop/touch_alert_lane_v1.py',
    'stargazer_desktop/self_check_realtime_v1.py',
    'stargazer_desktop/strong_pullback_policy_v1.py',
    'stargazer_desktop/strong_pullback_context_v1.py',
    'stargazer_desktop/strong_pullback_boards_v1.py',
    'stargazer_desktop/strong_pullback_positions_v1.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/intraday_public_context_v1.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/postclose_shared_source_v1.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/tdx_shared_prepare_v1.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/postclose_feature_cache_v1.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/intraday_native_context_v1.py',
)

# Scope: ONLY the confirmed-plan source bridge in gcp_golden_intraday_v1.
# Current release files still ALL verify. These do not change plan thresholds,
# Fibonacci maps, RSI/cross rules, or permit reusing old risk predictions.
PLAN_RUNTIME_ADDITIONS = {
    'stargazer_desktop/app.py':'Application launcher/self-check commands',
    'stargazer_desktop/push_compact_v1.py':'Presentation only',
    'stargazer_ml/v5_hybrid/risk_r0_v1/tdx_native_kernel_v1.py':'Risk preparation; separately SHA-bound risk inputs/results',
    'stargazer_ml/v5_hybrid/risk_r0_v1/tdx_postclose_runtime_v1.py':'Risk preparation/cache/budgets; not the plan trigger kernel',
}
PROTECTED_PLAN_INPUTS = {
    'stargazer_desktop/plan_rule_profiles.py',
    'stargazer_ml/v5_hybrid/green_common_profile/gcp_golden_intraday_v1.py',
    'stargazer_ml/v5_hybrid/green_common_profile/golden_ladder_v3.py',
    'stargazer_ml/v5_hybrid/green_common_profile/golden_resonance_v4.py',
}


def register_runtime_predecessors(manifest, published):
    """Accept known published wrapper hashes, never arbitrary claimed hashes.

    No trading-kernel exemptions are introduced. Historical plan bytes and
    current model/market/source validators remain unchanged.
    """
    approved=(set(manifest.get('approved_runtime_updates',[]))|set(PLAN_RUNTIME_ADDITIONS))-PROTECTED_PLAN_INPUTS
    known={k:set(v) for k,v in manifest.get('approved_predecessor_inputs',{}).items()}
    for prior in published:
        for path,value in prior.get('frozen_inputs',{}).items():
            if path in approved and path in manifest['frozen_inputs'] and re.fullmatch(r'[0-9a-f]{64}',str(value)):
                known.setdefault(path,set()).add(value)
    manifest['approved_runtime_updates']=sorted(approved)
    manifest['approved_predecessor_inputs']={k:sorted(v) for k,v in known.items()}
    manifest['plan_runtime_compatibility_scope']=PLAN_RUNTIME_ADDITIONS
    manifest['strict_plan_input_keys']=sorted(PROTECTED_PLAN_INPUTS)
    return manifest


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gz_json(path: Path):
    return json.loads(gzip.decompress(path.read_bytes()))


def fail(message):
    print('RELEASE_REFUSED:' + message)
    raise SystemExit(1)


def check():
    """Step 1: resolve current versions and verify them clean. After a refused
    or crashed release the references may point at a never-created successor;
    fall back to the newest existing manifest of each family."""
    flow_names = set()
    for name in FLOW_REFERENCES:
        found = set(FLOW_RE.findall((ROOT / name).read_text(encoding='utf-8')))
        if len(found) != 1:
            fail('RUNTIME_REFERENCE_INVALID:' + name)
        flow_names.update(found)
    if len(flow_names) != 1:
        fail('RUNTIME_REFERENCES_DIVERGE:' + ','.join(sorted(flow_names)))
    tarot_names = set(TAROT_RE.findall((GCP / 'tarot_live_v2.py').read_text(encoding='utf-8')))
    if len(tarot_names) != 1:
        fail('TAROT_REFERENCE_INVALID')
    flow_path = RISK / flow_names.pop()
    tarot_path = GCP / tarot_names.pop()
    if not flow_path.is_file():
        candidates = sorted(RISK.glob('live_monitor_fingerprints_*_v*_flow.json.gz'),
                            key=lambda p: int(FLOW_REF_RE.search(p.name).group(1)))
        if not candidates:
            fail('NO_FLOW_MANIFEST')
        flow_path = candidates[-1]
        print('CHECK recover: flow reference dangling; using', flow_path.name)
    if not tarot_path.is_file():
        candidates = sorted(GCP.glob('tarot_live_manifest_v*_ *_rel.json.gz'.replace(' ', '')),
                            key=lambda p: int(TAROT_REF_RE.search(p.name).group(1)))
        if not candidates:
            fail('NO_TAROT_MANIFEST')
        tarot_path = candidates[-1]
        print('CHECK recover: tarot reference dangling; using', tarot_path.name)
    print('CHECK ok: current =', flow_path.name, '+', tarot_path.name)
    return flow_path, tarot_path


def current_versions(flow_path, tarot_path):
    flow_ver = int(FLOW_REF_RE.fullmatch(flow_path.name).group(1))
    tarot_ver = int(TAROT_REF_RE.fullmatch(tarot_path.name).group(1))
    return flow_ver, tarot_ver


def switch(flow_name, tarot_name):
    """Step 2: every reference switches before any regeneration reads them."""
    for name in FLOW_REFERENCES:
        path = ROOT / name
        text = path.read_text(encoding='utf-8')
        updated = FLOW_RE.sub(lambda m: flow_name, text)
        if updated != text:
            path.write_text(updated, encoding='utf-8')
            print('SWITCH', name, '->', flow_name)
    path = GCP / 'tarot_live_v2.py'
    text = path.read_text(encoding='utf-8')
    updated = TAROT_RE.sub(lambda m: tarot_name, text)
    if updated != text:
        path.write_text(updated, encoding='utf-8')
        print('SWITCH', path.relative_to(ROOT), '->', tarot_name)


def regen(flow_path, tarot_path, flow_name, tarot_name, tarot_ver_label):
    """Step 3: full SHA recompute for every key; never overwrite old files."""
    flow_out = RISK / flow_name
    tarot_out = GCP / tarot_name
    if flow_out.exists() or tarot_out.exists():
        fail('SUCCESSOR_EXISTS:refusing to overwrite ' +
             (flow_name if flow_out.exists() else tarot_name))
    old_flow = load_gz_json(flow_path)
    missing = [k for k in old_flow if not (ROOT / k).is_file()]
    if missing:
        fail('FLOW_KEY_MISSING:' + ','.join(missing[:5]))
    flow_keys = list(old_flow) + [k for k in NEW_FILES
                                  if k not in old_flow and (ROOT / k).is_file()]
    new_flow = {k: sha(ROOT / k) for k in flow_keys}
    flow_out.write_bytes(gzip.compress(
        json.dumps(new_flow, ensure_ascii=False, sort_keys=True).encode(), mtime=0))
    old_tarot = load_gz_json(tarot_path)
    frozen = old_tarot['frozen_inputs']
    missing = [k for k in frozen if not (ROOT / k).is_file()]
    if missing:
        fail('TAROT_KEY_MISSING:' + ','.join(missing[:5]))
    tarot_keys = list(frozen) + [k for k in NEW_FILES
                                 if k not in frozen and (ROOT / k).is_file()]
    new_tarot = dict(old_tarot)
    new_tarot['frozen_inputs'] = {k: sha(ROOT / k) for k in tarot_keys}
    register_runtime_predecessors(new_tarot,[load_gz_json(p) for p in GCP.glob('tarot_live_manifest_v*_rel.json.gz')])
    new_tarot['supersedes'] = tarot_path.name
    new_tarot['supersedes_sha256'] = sha(tarot_path)
    note = new_tarot.get('release_note', '')
    new_tarot['release_note'] = note + ' | %s: 指纹发布脚本全键重算' % tarot_ver_label
    tarot_out.write_bytes(gzip.compress(
        json.dumps(new_tarot, indent=1, ensure_ascii=False).encode(), mtime=0))
    print('REGEN ok:', flow_name, len(new_flow), 'keys;', tarot_name, len(frozen), 'keys')


def verify():
    """Step 4: total release gate; failure forbids packaging."""
    sys.path.insert(0, str(ROOT / 'release_audit_v1'))
    from verify_live_release import verify_release
    result = verify_release(ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get('ok'):
        fail('RELEASE_GATE_FAILED:packaging forbidden')
    print('VERIFY ok: release gate passed')


def main(argv):
    if len(argv) != 3 or not re.fullmatch(r'v\d+', argv[1]) or not re.fullmatch(r'v\d+', argv[2]):
        fail('usage: release_fingerprints_v1.py <new_flow_version> <new_tarot_version>')
    date = datetime.now().strftime('%Y%m%d')
    flow_path, tarot_path = check()
    flow_ver, tarot_ver = current_versions(flow_path, tarot_path)
    new_flow_ver, new_tarot_ver = int(argv[1][1:]), int(argv[2][1:])
    if new_flow_ver <= flow_ver or new_tarot_ver <= tarot_ver:
        fail('VERSION_NOT_NEWER:current flow v%d tarot v%d' % (flow_ver, tarot_ver))
    flow_name = FLOW_NAME.format(date=date, ver=argv[1])
    tarot_name = 'tarot_live_manifest_%s_%s_rel.json.gz' % (argv[2], date)
    switch(flow_name, tarot_name)
    regen(flow_path, tarot_path, flow_name, tarot_name, argv[2])
    verify()
    print('RELEASED', flow_name, '+', tarot_name,
          '(old versions preserved: %s, %s)' % (flow_path.name, tarot_path.name))


if __name__ == '__main__':
    main(sys.argv)
