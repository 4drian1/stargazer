"""Read-back/hash audit and independent Wilder seed arithmetic; no fetch or fit."""
from collections import Counter
import json
from pathlib import Path

from expansion_first30_v1 import MANIFEST, verify_inputs
from price_verification_guarded_run import sha
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = Path(__file__).resolve().parent


def run():
    out = HERE / 'intraday_cache_probe_20261016_v1'
    manifest = json.loads((out / 'manifest.json').read_bytes())
    verify_inputs(manifest['inputs'])
    report = json.loads((out / 'diagnostics.json').read_bytes())
    counts = Counter()
    for case in report['results']:
        for scenario in case['scenarios']:
            bars = scenario['e_day']
            counts['prefix_equal'] += sum(c['prefix_equal'] for c in scenario['checks'])
            counts.update(c['future_perturbation'] for c in scenario['checks'])
            for row in bars:
                counts[scenario['semantics_assumption'] + '_' + row['bar']['status']] += 1
                counts[scenario['semantics_assumption'] + '_rsi_nonnull'] += row['fields']['rsi14'] is not None
                assert row['fields']['available_at'] is None and not row['fields']['live_feature_admitted']
            if scenario['semantics_assumption'] == 'END':
                closes = [r['bar']['close'] for r in bars]
                changes = [b - a for a, b in zip(closes, closes[1:])]
                gain = sum(max(v, 0) for v in changes[:14]) / 14
                loss = sum(max(-v, 0) for v in changes[:14]) / 14
                for index in (14, 15):
                    if index == 15:
                        gain = (13 * gain + max(changes[14], 0)) / 14
                        loss = (13 * loss + max(-changes[14], 0)) / 14
                    expected = 50 if gain == loss == 0 else 100 if loss == 0 else 100 - 100 / (1 + gain / loss)
                    assert abs(expected - bars[index]['fields']['rsi14']) < 1e-10
                    counts['independent_rsi_values_checked'] += 1
    assert counts['prefix_equal'] == 192
    assert counts['PASS'] == 186 and counts['NOT_TESTED_NO_FUTURE_ROWS'] == 6
    assert counts['END_rsi_nonnull'] == 12 and counts['START_rsi_nonnull'] == 0
    old_refs = 0
    for name in ('first_fit_result_audit_v1.json', 'crossyear_fit_result_audit_v1.json'):
        refs = json.loads((HERE / name).read_bytes())['files']
        verify_inputs(refs)
        old_refs += len(refs)
    for path in (MANIFEST, HERE / 'candidate_resonance_test_20261016_v1/manifest.json',
                 HERE / 'two_stage_risk_acceptance_20261016_v1/manifest.json'):
        verify_inputs(json.loads(path.read_bytes())['inputs'])
    exe = HERE.parents[2] / 'stargazer_desktop/dist/Stargazer.exe'
    assert sha(exe) == '06a2c63e513ff131a6f16d7dd74b78811cef6b2625c0c87b23cbfd235bd738d0'
    audit = dict(counts=counts, frozen_old_fit_files_verified=old_refs, old_manifests_verified=3,
        canonical_exe_unchanged=True, research_bytes=bytes_used(), training_ready=False,
        inputs=manifest['inputs'], outputs=[dict(path=str(p), sha256=sha(p))
                                           for p in sorted(out.iterdir())],
        audit_code_sha256=sha(Path(__file__)))
    target = HERE / 'intraday_cache_probe_audit_20261016_v1.json'
    body = (json.dumps(audit, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf8')
    if bytes_used() + len(body) > TOTAL_CAP:
        raise ValueError('REGISTERED_BUDGET')
    with target.open('xb') as stream:
        stream.write(body)
    print(json.dumps(dict(counts=counts, old_fit_files=old_refs, bytes=len(body))))


if __name__ == '__main__':
    run()
