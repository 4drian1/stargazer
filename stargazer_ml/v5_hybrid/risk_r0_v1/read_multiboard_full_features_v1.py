"""Original risk cohort + keyed, factorized research features. Not a fit entry."""
import argparse
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from build_multiboard_full_features_v1 import HERE, OUT, SEGMENTS, sha
from intraday_relative_stall_v1 import TIME


def metadata():
    report = json.loads((OUT / 'report.json').read_bytes())
    terminal = json.loads((OUT / 'stop_record.json').read_bytes())
    if report['status'] != 'FULL_COHORT_THREE_GROUP_CANDIDATES_NOT_FIT' or terminal['status'] != 'COMPLETE':
        raise ValueError('THREE_GROUP_BUILD_NOT_COMPLETE')
    spec_path = OUT / 'feature_spec.json'
    if sha(spec_path) != report['outputs'][spec_path.name]['sha256']:
        raise ValueError('FEATURE_SPEC_HASH_MISMATCH')
    spec = json.loads(spec_path.read_bytes())
    if spec['shared'] != report['shared_fields'] or spec['own'] != report['own_fields']:
        raise ValueError('FEATURE_LIST_CONFLICT')
    return report, spec


def checked_table(report, name, filters=None):
    if sha(OUT / name) != report['outputs'][name]['sha256']:
        raise ValueError('FEATURE_PART_HASH_MISMATCH:' + name)
    return pq.read_table(OUT / name, filters=filters).to_pandas()


def attach(original, own, shared, fields):
    keys = ['code', TIME]
    if original.duplicated(keys).any() or own.duplicated(keys).any() or shared.duplicated(TIME).any():
        raise ValueError('DUPLICATE_JOIN_KEY')
    if set(fields) & set(original):
        raise ValueError('ORIGINAL_COLUMN_COLLISION')
    identity = original[keys].merge(own[keys], on=keys, how='outer', indicator=True, validate='one_to_one')
    if not (identity['_merge'] == 'both').all():
        raise ValueError('COHORT_KEYS_NOT_IDENTICAL')
    if not set(original[TIME]) <= set(shared[TIME]):
        raise ValueError('MISSING_EXACT_SHARED_CUTOFF_NO_ASOF')
    result = original.merge(own, on=keys, how='left', sort=False, validate='one_to_one').merge(
        shared, on=TIME, how='left', sort=False, validate='many_to_one')
    pd.testing.assert_frame_equal(result[list(original)], original.reset_index(drop=True))
    if np.isinf(result[fields].to_numpy(dtype=float)).any():
        raise ValueError('INFINITE_NEW_FEATURE')
    return result


def read_segment(segment):
    if segment not in SEGMENTS:
        raise ValueError('UNKNOWN_SEGMENT')
    from join_momentum_review_v1 import read_segment as original_reader
    report, spec = metadata()
    base_path = HERE / 'joint_fit_review_20261017_v1' / (segment + '.parquet')
    if sha(base_path) != report['inputs'][str(base_path.resolve())]:
        raise ValueError('ORIGINAL_COHORT_HASH_MISMATCH')
    original = original_reader(segment)
    own, shared = [], []
    for part in report['parts']:
        frame = checked_table(report, part['own'], [('segment', '=', segment)])
        if len(frame):
            own.append(frame[['code', TIME, *spec['own']]])
            shared.append(checked_table(report, part['shared']))
    if not own:
        raise ValueError('EMPTY_ORIGINAL_SEGMENT_CONTEXT')
    return attach(original, pd.concat(own, ignore_index=True), pd.concat(shared, ignore_index=True),
                  spec['shared'] + spec['own'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment', choices=SEGMENTS, required=True)
    args = parser.parse_args()
    frame = read_segment(args.segment)
    report, spec = metadata()
    print(json.dumps(dict(status='RESEARCH_FEATURE_VIEW_NOT_FIT', segment=args.segment,
        rows=len(frame), new_features=len(spec['shared']) + len(spec['own']),
        all_null_train=report['all_null_train'], constant_train=report['constant_train'],
        fit_authorized=False, labels_unchanged=True), ensure_ascii=False))
