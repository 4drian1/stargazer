"""Freeze a candidate aggregation proposal and verify old population; NEVER FIT."""
import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import unittest

import pandas as pd

import common_candidate_aggregation_v1 as aggregation
import prepare_auxiliary_fixed_fit_v1 as parent
from multiboard_three_arm_fit_v1 import EvidenceWriter
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = Path(__file__).resolve().parent
OUT = HERE / 'common_candidate_preparation_20261019_v1'
CONTRACT = HERE / 'contracts/COMMON_MOTION_CANDIDATE_COMPARISON_20261019.md'
CAP, RESERVE = 128 * 1024, 16 * 1024
PARENT_SHA = '8a3ffb0cb2ea6f04f4f6d9cb32030d6fd7281f96ed92c1fd94c9aadcc7d0baaa'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_sha(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()


def prepare():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    used = bytes_used()
    if used + CAP > TOTAL_CAP:
        raise ValueError('PREPARATION_RESERVATION_UNAVAILABLE')
    parent_path = parent.OUT / 'config.json'
    if sha(parent_path) != PARENT_SHA:
        raise ValueError('PARENT_CONFIG_DRIFT')
    original = json.loads(parent_path.read_bytes())
    accepted_path = HERE / 'acceptance_common_20261018/report.json'
    accepted = json.loads(accepted_path.read_bytes())
    if accepted['verdict'] != 'PASS' or accepted['meta']['n_candidates'] != 86319:
        raise ValueError('CANDIDATE_BOARD_COVERAGE_REQUIRED')
    refs = dict(accepted['input_fingerprints'])
    # The old execution verifier differs by documented post-run corrections; do not
    # replace its fingerprint. The four calculation modules must still match.
    for name, expected in accepted['code_fingerprints'].items():
        if Path(name).name != 'verify_continuous_motion_common_coverage_v1.py':
            path = Path(name) if Path(name).is_absolute() else HERE / name
            if sha(path) != expected:
                raise ValueError('ACCEPTED_CALCULATION_DRIFT:' + str(path))
    for path, expected in refs.items():
        if sha(path) != expected:
            raise ValueError('FROZEN_INPUT_DRIFT:' + path)
    for path in [parent_path, accepted_path, accepted_path.with_name('fingerprints.json'),
                 accepted_path.with_name('caveats_and_correction_v1.md'),
                 accepted_path.with_name('caveats_and_correction_v2.md'), CONTRACT,
                 Path(__file__), HERE / 'common_candidate_aggregation_v1.py',
                 HERE / 'test_common_candidate_preparation_v1.py',
                 HERE / 'common_loo_derive_v1.py', HERE / 'continuous_motion_common_v1.py',
                 HERE / 'verify_continuous_motion_common_coverage_v1.py']:
        refs[str(path)] = sha(path)
    config = deepcopy(original)
    config.update(version='COMMON_CANDIDATE_COMPARISON_PREPARATION_V1',
                  status='SPEC_READY_REAL_NEW_FEATURE_ACCEPTANCE_PENDING',
                  fields=original['fields'] + aggregation.learning_fields(),
                  numeric_dimensions=266, transformed_dimensions=532,
                  new_fields=aggregation.learning_fields(),
                  parent_config_sha256=PARENT_SHA, fit_authorized=False,
                  current_stage_authorizes_fit=False,
                  primary_saved_comparator=str(parent.FIT_OUT),
                  new_support_fields_audit_only=True)
    if len(config['fields']) != 266 or len(set(config['fields'])) != 266:
        raise ValueError('LEARNING_FIELD_SCHEMA_DRIFT')
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log).run(
        unittest.defaultTestLoader.loadTestsFromName('test_common_candidate_preparation_v1'))
    if not tests.wasSuccessful():
        raise ValueError('AGGREGATION_TEST_FAILED:' + log.getvalue())
    writer = EvidenceWriter(OUT, CAP, RESERVE)
    writer.json('plan.json', dict(inputs=refs, preparation_cap=CAP,
                                 scope='NO_FIT_NO_RAW_REPLAY', original_budget_remaining=TOTAL_CAP-used))
    try:
        parts = parent.load_parts(original)
        population = pd.concat([f[['code', parent.TIME]].assign(segment=s)
                                for s, f in parts.items()], ignore_index=True)
        aggregation.validate_population(population)
        profiles = {}
        for segment, frame in parts.items():
            labels = ['code', parent.TIME, parent.TARGET, 'label_end_time', 'auxiliary_end_time']
            profiles[segment] = dict(rows=len(frame), stocks=int(frame.code.nunique()),
                keys_labels_sha256=frame_sha(frame[labels]),
                original_232_sha256=frame_sha(frame[original['fields']]))
        for path, expected in refs.items():
            if sha(path) != expected:
                raise ValueError('INPUT_CHANGED_DURING_PREPARATION:' + path)
        writer.json('config.json', config)
        writer.json('report.json', dict(
            status='PASS_AGGREGATION_UNIT_AND_ORIGINAL_POPULATION_ONLY',
            rows=len(population), segments=profiles, tests_run=tests.testsRun,
            test_output=log.getvalue(), new_numeric_columns=34,
            real_new_values_computed=False, real_candidate_sidecar_accepted=False,
            training_ready=False, real_fits=0, new_http_requests=0, raw_minute_replays=0,
            missing_gates=['REAL_34_COLUMN_CANDIDATE_SIDECAR_ACCEPTANCE',
                           'HASH_BOUND_TRAINING_ENTRY_AND_SEPARATE_FIT_AUTHORIZATION',
                           'SUFFICIENT_REGISTERED_FIT_STORAGE'],
            fit_budget_proposal=3 * 1024**2, fit_budget_reserved=False))
        writer.json('manifest.json', dict(outputs=writer.outputs.copy()))
        writer.json('stop_record.json', dict(status='COMPLETE_PREPARATION_NOT_FIT',
                    manifest_sha256=sha(OUT / 'manifest.json')), terminal=True)
    except Exception as exc:
        writer.json('stop_record.json', dict(status='FAILED_NO_RESTART',
                    reason=f'{type(exc).__name__}:{exc}'), terminal=True)
        raise
    return dict(status='COMPLETE_PREPARATION_NOT_FIT', rows=len(population),
                training_ready=False, bytes=writer.used)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(prepare()))
