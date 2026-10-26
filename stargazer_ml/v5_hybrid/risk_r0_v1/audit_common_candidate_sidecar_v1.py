"""Independent saved-artifact readback; no raw-minute replay or model fit."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common_candidate_aggregation_v1 import KEYS, learning_fields, validate_population
from intraday_relative_stall_v1 import TIME
from research_budget_common_v3 import EvidenceWriter, bytes_used, TOTAL_CAP

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'common_candidate_sidecar_20261019_v1'
OUT = HERE / 'common_candidate_sidecar_acceptance_20261019_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    stop = json.loads((SOURCE / 'stop_record.json').read_bytes())
    if stop['status'] != 'COMPLETE_NOT_FIT' or stop['manifest_sha256'] != sha(SOURCE / 'manifest.json'):
        raise ValueError('SOURCE_NOT_COMPLETE')
    manifest = json.loads((SOURCE / 'manifest.json').read_bytes())
    for name, item in manifest['outputs'].items():
        path = SOURCE / name
        if sha(path) != item['sha256'] or path.stat().st_size != item['bytes']:
            raise ValueError('SAVED_OUTPUT_DRIFT:' + name)
    plan = json.loads((SOURCE / 'plan.json').read_bytes())
    for path, expected in plan['inputs'].items():
        if sha(path) != expected:
            raise ValueError('BOUND_SOURCE_DRIFT:' + path)
    config = json.loads((HERE / 'common_candidate_preparation_20261019_v1/config.json').read_bytes())
    base = json.loads((HERE / 'auxiliary_120m_fit_preparation_20261018_v1/config.json').read_bytes())
    if config['fields'] != base['fields'] + learning_fields() or config['fit_authorized'] is not False:
        raise ValueError('FIXED_CONFIG_DRIFT')
    side = pd.read_parquet(SOURCE / 'candidate_features.parquet')
    validate_population(side)
    if side.columns.tolist() != KEYS + learning_fields() or len(side) != 86319:
        raise ValueError('CANDIDATE_SIDECAR_SCHEMA_DRIFT')
    if np.isinf(side[learning_fields()].to_numpy()).any():
        raise ValueError('INFINITE_FEATURE')
    labels = pd.read_parquet(base['label_reference']['path'])
    report = json.loads((SOURCE / 'report.json').read_bytes())
    segments = {}
    for segment in ('train', 'select', 'calibration', 'test'):
        frame = side[side.segment.eq(segment)]
        truth = labels[labels.segment.eq(segment)]
        pd.testing.assert_frame_equal(frame[['code', TIME]].reset_index(drop=True),
                                      truth[['code', TIME]].reset_index(drop=True))
        for field in learning_fields():
            actual = int(np.isfinite(frame[field]).sum())
            if actual != report['profiles'][segment][field]['finite']:
                raise ValueError('FINITE_RATE_DISAGREEMENT:' + field)
        segments[segment] = dict(rows=len(frame), min_field_finite=int(
            frame[learning_fields()].notna().sum().min()))
    if segments['train']['min_field_finite'] == 0:
        raise ValueError('ALL_NULL_TRAIN')
    audit = pd.read_parquet(SOURCE / 'candidate_audit.parquet')
    if audit.duplicated(KEYS + ['frequency']).any() or len(audit) != 2 * len(side):
        raise ValueError('AUDIT_KEY_DRIFT')
    for frequency in (5, 15):
        frame = audit[audit.frequency.eq(frequency)]
        pd.testing.assert_frame_equal(frame[KEYS].sort_values(KEYS).reset_index(drop=True),
                                      side[KEYS].sort_values(KEYS).reset_index(drop=True))
        denominator = frame[f'ccm_v1_{frequency}_own_expected_boards']
        if not denominator.gt(0).all():
            raise ValueError('INVALID_EXPECTED_BOARD_DENOMINATOR')
        for column in frame.filter(regex=f'^ccm_v1_{frequency}_own_.*_known_boards$'):
            if not ((frame[column] >= 0) & (frame[column] <= denominator)).all():
                raise ValueError('INVALID_KNOWN_BOARD_COUNT')
    # Saved original comparator and the superseded 350-field experiment stay intact.
    frozen_outputs = 0
    for name in ('auxiliary_120m_fixed_fit_20261018_v1', 'continuous_motion_fixed_fit_20261018_v1'):
        root = HERE / name
        old = json.loads((root / 'manifest.json').read_bytes())
        for filename, item in old['outputs'].items():
            if sha(root / filename) != item['sha256']:
                raise ValueError('OLD_MODEL_ARTIFACT_DRIFT:' + filename)
            frozen_outputs += 1
    writer = EvidenceWriter(OUT, cap=64*1024, reserve=8*1024)
    writer.json('acceptance.json', dict(
        status='PASS_SAVED_REAL_CANDIDATE_SIDECAR_NOT_FIT',
        source_manifest_sha256=sha(SOURCE / 'manifest.json'), source_report_sha256=sha(SOURCE / 'report.json'),
        audit_code_sha256=sha(Path(__file__)), bound_inputs=len(plan['inputs']),
        rows=len(side), numeric_fields=len(learning_fields()), segments=segments,
        old_model_outputs_unchanged=frozen_outputs, model_fits=0, http_requests=0,
        raw_minute_replays=0, training_authorized=False,
        training_entry_acceptance='PENDING',
        real_prefix_and_loo_cases=report['real_prefix_and_loo_cases'],
        registered_bytes_used_before_receipt=bytes_used(), registered_cap=TOTAL_CAP))
    writer.json('stop_record.json', dict(status='COMPLETE_NOT_FIT',
                receipt_sha256=sha(OUT / 'acceptance.json')), terminal=True)
    return dict(status='PASS_SAVED_REAL_CANDIDATE_SIDECAR_NOT_FIT', rows=len(side), fields=34,
                segments=segments, old_outputs_unchanged=frozen_outputs)


if __name__ == '__main__':
    print(json.dumps(run()))
