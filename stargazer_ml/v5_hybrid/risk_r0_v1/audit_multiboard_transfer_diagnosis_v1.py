"""Diagnostic entry checks and immutable readback; no training/network access."""
import argparse
import io
import json
import unittest

import pandas as pd

import diagnose_multiboard_transfer_v1 as diag
from research_budget_v2 import bytes_used, TOTAL_CAP

ENTRY = diag.HERE / 'multiboard_transfer_diagnosis_entry_20261018_v1.json'
ACCEPT = diag.HERE / 'multiboard_transfer_diagnosis_acceptance_20261018_v1.json'


def save(path, record):
    body = diag.prep.encoded(record)
    used = sum(p.stat().st_size for p in diag.OUT.rglob('*') if p.is_file()) if diag.OUT.exists() else 0
    if used + len(body) + diag.RESERVE > diag.CAP or bytes_used() + len(body) + diag.RESERVE > TOTAL_CAP:
        raise ValueError('DIAGNOSTIC_ACCEPTANCE_BUDGET_STOP')
    with path.open('xb') as stream:
        stream.write(body)


def entry():
    if ENTRY.exists():
        raise ValueError('ENTRY_EXISTS_NO_OVERWRITE')
    output = io.StringIO()
    tests = unittest.defaultTestLoader.loadTestsFromNames([
        'test_multiboard_transfer_diagnosis_v1', 'test_prepare_multiboard_comparison_v1'])
    result = unittest.TextTestRunner(stream=output).run(tests)
    if not result.wasSuccessful() or result.skipped:
        raise ValueError('DIAGNOSTIC_TESTS_FAILED:' + output.getvalue())
    refs = {str(diag.HERE / n): diag.prep.sha(diag.HERE / n) for n in [
        'diagnose_multiboard_transfer_v1.py', 'test_multiboard_transfer_diagnosis_v1.py',
        'audit_multiboard_transfer_diagnosis_v1.py']}
    record = dict(status='PASS_READONLY_DIAGNOSTIC_ENTRY', tests=result.testsRun,
                  test_output=output.getvalue(), refs=refs, actual_fit=False, http_requests=0)
    save(ENTRY, record)
    print(json.dumps({k: v for k, v in record.items() if k not in ('refs', 'test_output')}))


def result():
    if ACCEPT.exists():
        raise ValueError('ACCEPTANCE_EXISTS_NO_OVERWRITE')
    accepted = json.loads(ENTRY.read_bytes())
    for path, expected in accepted['refs'].items():
        if diag.prep.sha(path) != expected:
            raise ValueError('ENTRY_CODE_DRIFT')
    terminal = json.loads((diag.OUT / 'stop_record.json').read_bytes())
    manifest = json.loads((diag.OUT / 'manifest.json').read_bytes())
    if terminal['status'] != 'COMPLETE_DIAGNOSTICS_NOT_NEW_FIT' or diag.prep.sha(diag.OUT / 'manifest.json') != terminal['manifest_sha256']:
        raise ValueError('DIAGNOSTIC_TERMINAL_OR_MANIFEST_DRIFT')
    for path, expected in manifest['inputs'].items():
        if diag.prep.sha(path) != expected:
            raise ValueError('FROZEN_INPUT_DRIFT:' + path)
    for name, item in manifest['outputs'].items():
        if diag.prep.sha(diag.OUT / name) != item['sha256']:
            raise ValueError('DIAGNOSTIC_OUTPUT_DRIFT:' + name)
    data = json.loads((diag.OUT / 'diagnostics.json').read_bytes())
    positions = pd.read_parquet(diag.OUT / 'OUTCOME_ONLY_positive_positions.parquet')
    frozen = pd.read_parquet(diag.FIT / 'predictions.parquet')
    expected = frozen[frozen.y_tail_1200m.eq(1)].rename(columns={'segment': 'proposed_segment_name'})
    keys = ['code', diag.prep.TIME, 'proposed_segment_name']
    pd.testing.assert_frame_equal(positions[keys].sort_values(keys).reset_index(drop=True),
                                 expected[keys].sort_values(keys).reset_index(drop=True))
    if positions.duplicated(['code', diag.prep.TIME]).any():
        raise ValueError('DUPLICATE_POSITIVE_RECORD')
    if not positions.first_touch_trading_minute.between(1, 1200).all() or not positions.minimum_trading_minute.between(1, 1200).all():
        raise ValueError('OUTCOME_OFFSET_OUTSIDE_FROZEN_WINDOW')
    if not (positions.minimum_trading_minute >= positions.first_touch_trading_minute).all():
        raise ValueError('MINIMUM_PRECEDES_FIRST_TOUCH')
    for segment in diag.prep.SEGMENTS:
        part = positions[positions.proposed_segment_name.eq(segment)]
        if diag.duration_summary(part) != data['timing'][segment]:
            raise ValueError('TIMING_READBACK_DIFFERENT')
        for signal in data['signals'][segment].values():
            states = signal['states']
            if sum(v['observations'] for v in states.values()) != signal['observations']:
                raise ValueError('SIGNAL_UNIT_COUNTS_DIFFERENT')
            if sum(v['candidate_rows'] for v in states.values()) != data['ranking'][segment]['M1_lr']['all']['rows']:
                raise ValueError('SIGNAL_STATES_DO_NOT_PARTITION_POPULATION')
    # Existing fit metadata/labels/predictions are hashes in manifest inputs; never rewrite them.
    original_d2 = diag.prep.sha(diag.HERE / 'postclose_repaired_research_fit_20261017_v1/model.txt')
    if original_d2 != '5d8402cd530cb83778d901e33eb87788f21ebc8619414f335b6d56905a81e005':
        raise ValueError('OLD_MODEL_DRIFT')
    record = dict(status='PASS_FIXED_TRANSFER_AND_OUTCOME_POSITION_DIAGNOSTICS',
        tests=accepted['tests'], original_rows=manifest['original_population_rows'],
        common_rows=manifest['common_rows'], existing_positive_records=len(positions),
        report_sha256=diag.prep.sha(diag.OUT / 'diagnostics.json'),
        manifest_sha256=diag.prep.sha(diag.OUT / 'manifest.json'),
        old_d2_sha256=original_d2, new_fit=False, new_labels=False, http_requests=0,
        diagnostic_future_offsets_eligible_for_X=False, causal_root_cause_proven=False)
    save(ACCEPT, record)
    print(json.dumps(record))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--entry', action='store_true')
    mode.add_argument('--result', action='store_true')
    args = parser.parse_args()
    entry() if args.entry else result()
