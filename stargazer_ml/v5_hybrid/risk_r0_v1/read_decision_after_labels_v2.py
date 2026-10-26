"""Canonical label view: immutable v1 slots plus same-key basis corrections."""
import json
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from build_decision_after_labels_v1 import SOURCE,CONTRACT
from inventory_ths_stockdb_cache_v1 import HERE,digest
from read_decision_after_labels_v1 import attach_label_times

OUT=HERE/'joint_decision_after_labels_20261017_v2'


def read_stored_part(part):
    body=(OUT/part['path']).read_bytes()
    if digest(body)!=part['sha256']:raise ValueError('BASE_LABEL_PART_HASH_MISMATCH')
    frame=pq.read_table(pa.BufferReader(body)).to_pandas()
    if 'correction' not in part:return frame
    body=(OUT/part['correction']).read_bytes()
    if digest(body)!=part['correction_sha256']:raise ValueError('BASIS_CORRECTION_HASH_MISMATCH')
    correction=pq.read_table(pa.BufferReader(body)).to_pandas()
    return apply_correction(frame,correction)


def apply_correction(frame,correction):
    keys=['code','cutoff_index']
    if correction.duplicated(keys).any() or frame.duplicated(keys).any():raise ValueError('DUPLICATE_CORRECTION_KEY')
    where=pd.MultiIndex.from_frame(frame[keys]).get_indexer(pd.MultiIndex.from_frame(correction[keys]))
    if (where<0).any():raise ValueError('CORRECTION_OUTSIDE_POPULATION')
    result=frame.copy()
    for column in ['mae_1200m','y_tail_1200m','label_state']:
        result.iloc[where,result.columns.get_loc(column)]=correction[column].array
    return result


def load_label_part(index):
    if json.loads((OUT/'stop_record.json').read_bytes())['status']!='COMPLETE':raise ValueError('BASIS_BUILD_NOT_COMPLETE')
    manifest=json.loads((OUT/'manifest.json').read_bytes())
    for path in [CONTRACT,SOURCE/'manifest.json']:
        if digest(path.read_bytes())!=manifest['inputs'][str(path)]:raise ValueError('LABEL_BINDING_DRIFT')
    body=(SOURCE/'shared_market_and_leaders.parquet').read_bytes()
    if digest(body)!=json.loads((SOURCE/'manifest.json').read_bytes())['output_hashes']['shared_market_and_leaders.parquet']['sha256']:
        raise ValueError('LABEL_CLOCK_DRIFT')
    times=pq.read_table(pa.BufferReader(body),columns=['decision_ts_assumed_end']).column(0).to_numpy()
    return attach_label_times(read_stored_part(manifest['parts'][index]),times,json.loads(CONTRACT.read_bytes())['proposed_segments'])
