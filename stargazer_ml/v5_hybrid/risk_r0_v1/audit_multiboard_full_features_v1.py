"""Full-cohort engineering readback, not a model-effectiveness or trading gate."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq

from build_multiboard_full_features_v1 import HERE, OUT, CAP, SEGMENTS, sha
from intraday_relative_stall_v1 import TIME
from multiboard_risk_features_v1 import FLAGS
from read_multiboard_full_features_v1 import checked_table, metadata, read_segment
from research_budget_v2 import bytes_used, TOTAL_CAP

ACCEPTANCE = HERE / 'multiboard_full_features_acceptance_20261018_v1.json'


def audit():
    if ACCEPTANCE.exists():
        raise ValueError('ACCEPTANCE_EXISTS_NO_OVERWRITE')
    report, spec = metadata()
    previous_receipt = HERE / 'relative_stall_5m_acceptance_20261018_v1.json'
    modules = json.loads(previous_receipt.read_bytes())['tests_modules'] + [
        'test_multiboard_risk_features_v1', 'test_build_multiboard_full_features_v1',
        'test_read_multiboard_full_features_v1']
    tested = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-m', 'unittest', *modules],
                            cwd=HERE, capture_output=True, text=True, check=True)
    if 'skipped' in tested.stderr:
        raise ValueError('NO_SKIPPED_REAL_TESTS_ALLOWED')
    for path, digest in report['inputs'].items():
        if sha(Path(path)) != digest:
            raise ValueError('FROZEN_INPUT_CHANGED:' + path)
    raw_ledger = json.loads((HERE / 'relative_stall_5m_20261018_v1/raw_input_ledger.json').read_bytes())
    for item in raw_ledger:
        if sha(Path(item['path'])) != item['sha256']:
            raise ValueError('RAW_ARCHIVE_CHANGED:' + item['code'])
    for name, item in report['outputs'].items():
        if sha(OUT / name) != item['sha256']:
            raise ValueError('FEATURE_OUTPUT_CHANGED:' + name)
    fields = spec['shared'] + spec['own']
    assert len(fields) == 106 and len(set(fields)) == 106
    for frequency in (5, 15):
        for prefix in ('mb', 'own'):
            for benchmark in ('sse', 'cyb'):
                for flag in FLAGS:
                    assert f'{prefix}{frequency}_{benchmark}_positive_{flag}_rate' in fields
                    assert f'{prefix}{frequency}_{benchmark}_positive_{flag}_coverage' in fields
    assert not set(fields) & {'y_tail_1200m', 'mae_1200m', 'label_end_time', TIME, 'code'}
    actual_rows, summaries = 0, {}
    for segment in SEGMENTS:
        frame = read_segment(segment)
        assert len(frame) == report['segment_rows'][segment]
        assert (frame.golden_asof_day < frame[TIME] // 1000000).all()
        assert not np.isinf(frame[fields].to_numpy(dtype=float)).any()
        for field in fields:
            stats = report['audit'][segment][field]
            assert stats['rows'] == len(frame)
            assert stats['finite'] == int(np.isfinite(frame[field]).sum())
            if field.endswith(('_rate', '_coverage', '_share')) or field.endswith('member_coverage'):
                observed = frame[field].dropna()
                assert (observed >= -1e-12).all() and (observed <= 1 + 1e-12).all(), field
        summaries[segment] = dict(rows=len(frame), dates=int(frame[TIME].floordiv(1000000).nunique()),
            own_mean5_known=int(frame.own5_mean.notna().sum()), own_mean15_known=int(frame.own15_mean.notna().sum()),
            global5_known=int(frame.mb5_mean.notna().sum()), global15_known=int(frame.mb15_mean.notna().sum()))
        actual_rows += len(frame)
    assert actual_rows == 100704
    # Independent whole-universe 5m pilot comparison, exact common cutoffs.
    # Leader capture statistics have a declared prior-capture policy, so these
    # comparisons target distribution and price-stall fields rather than aliasing them.
    proof_count = 0
    pilot_root = HERE / 'relative_stall_5m_20261018_v1'
    pilot_report = json.loads((pilot_root / 'report.json').read_bytes())
    january = next(p for p in report['parts'] if p['month'] == 202501)
    shared = checked_table(report, january['shared']).set_index(TIME)
    for prefix in ('sse', 'cyb'):
        path = pilot_root / (prefix + '_board_evidence_5m.parquet')
        assert sha(path) == pilot_report['outputs'][path.name]['sha256']
        pilot = pq.read_table(path).to_pandas()
        for ts, group in pilot.groupby(TIME):
            if ts not in shared.index:
                continue
            row = shared.loc[ts]
            valid = group.day_return.dropna()
            mean = float(valid.mean()) if len(valid) else np.nan
            np.testing.assert_allclose(row.mb5_mean, mean, rtol=0, atol=1e-12, equal_nan=True)
            for q, name in ((.1, 'q10'), (.5, 'q50'), (.9, 'q90')):
                value = float(valid.quantile(q)) if len(valid) else np.nan
                np.testing.assert_allclose(row['mb5_' + name], value, rtol=0, atol=1e-12, equal_nan=True)
            selected = group[group.day_return > 0]
            for flag, column in [('nonpositive', 'market_rising_sector_nonpositive'),
                                  ('falling', 'market_rising_sector_falling'),
                                  ('not_new_high', 'market_new_high_sector_not_new_high')]:
                known = selected[column].dropna()
                rate = float(known.mean()) if len(known) else np.nan
                coverage = len(known) / len(selected) if len(selected) else np.nan
                np.testing.assert_allclose(row[f'mb5_{prefix}_positive_{flag}_rate'], rate, rtol=0, atol=1e-12, equal_nan=True)
                np.testing.assert_allclose(row[f'mb5_{prefix}_positive_{flag}_coverage'], coverage, rtol=0, atol=1e-12, equal_nan=True)
            proof_count += 1
    result = dict(status='PASS_FULL_COHORT_FEATURE_ENGINEERING_NOT_FIT_OR_RISK_EFFECTIVENESS',
        test_output=tested.stderr, tests_modules=modules, segment_coverage=summaries,
        pilot_distribution_and_denominator_comparisons=proof_count,
        rows=actual_rows, new_fields=len(fields), all_null_train=report['all_null_train'],
        constant_train=report['constant_train'], raw15_valid_compared=report['raw15_valid_compared'],
        raw_ledger_sha256=sha(pilot_root / 'raw_input_ledger.json'),
        report_sha256=sha(OUT / 'report.json'), full_report_inputs_verified=len(report['inputs']),
        raw_files_verified_at_acceptance=len(raw_ledger),
        code_refs={str(HERE / name): sha(HERE / name) for name in [
            'multiboard_risk_features_v1.py', 'build_multiboard_full_features_v1.py',
            'read_multiboard_full_features_v1.py', 'test_multiboard_risk_features_v1.py',
            'test_build_multiboard_full_features_v1.py', 'test_read_multiboard_full_features_v1.py', Path(__file__).name]},
        pilot_bytes=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()),
        labels_changed=False, model_unchanged=True, http_requests=0, fit=False,
        prospective_or_unbiased_gate=False)
    body = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    if len(body) > 256 * 1024 or len(body) + result['pilot_bytes'] > CAP or bytes_used() + len(body) > TOTAL_CAP:
        raise ValueError('ACCEPTANCE_BUDGET_STOP')
    with ACCEPTANCE.open('xb') as stream:
        stream.write(body)
    print(json.dumps({k: result[k] for k in ['status', 'rows', 'new_fields', 'pilot_bytes',
        'all_null_train', 'constant_train', 'segment_coverage']}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    audit()
