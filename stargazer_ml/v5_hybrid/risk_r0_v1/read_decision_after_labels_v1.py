"""Logical label view: explicit times without repeating them on disk."""
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from build_decision_after_labels_v1 import OUT,SOURCE,CONTRACT
from inventory_ths_stockdb_cache_v1 import digest
from decision_after_labels_v1 import minute_clock


def attach_label_times(labels,cutoff_times,segments):
    result=labels.copy();t=result.cutoff_index.to_numpy(dtype=int)
    if (t<0).any() or (t>=len(cutoff_times)).any():raise ValueError('INVALID_LABEL_CUTOFF')
    clock=minute_clock(np.unique(cutoff_times//1000000))
    if not np.array_equal(clock[14::15],cutoff_times):raise ValueError('LABEL_CLOCK_MISMATCH')
    ends=t*15+14+1200;valid=ends<len(clock)
    end_values=np.full(len(result),np.nan);end_values[valid]=clock[ends[valid]]
    result['decision_ts_assumed_end']=cutoff_times[t]
    result['label_end_time']=pd.array(end_values,dtype='Int64')
    result['data_complete_time']=None
    result['label_research_admitted']=result.label_state==0
    result['training_eligible']=False
    names={i:s['name'] for i,s in enumerate(segments)}
    result['proposed_segment_name']=result.proposed_segment.map(names).fillna('OUT_OF_SCOPE')
    return result


def load_label_part(index):
    if json.loads((OUT/'stop_record.json').read_bytes())['status']!='COMPLETE':
        raise ValueError('LABEL_BUILD_NOT_COMPLETE')
    metadata=json.loads((OUT/'manifest.json').read_bytes())
    for path in [CONTRACT,SOURCE/'manifest.json']:
        if digest(path.read_bytes())!=metadata['inputs'][str(path)]:raise ValueError('LABEL_BINDING_DRIFT')
    part=metadata['parts'][index];body=(OUT/part['path']).read_bytes()
    if digest(body)!=part['sha256']:raise ValueError('LABEL_PART_HASH_MISMATCH')
    labels=pq.read_table(pa.BufferReader(body)).to_pandas()
    source_meta=json.loads((SOURCE/'manifest.json').read_bytes())
    shared=(SOURCE/'shared_market_and_leaders.parquet').read_bytes()
    if digest(shared)!=source_meta['output_hashes']['shared_market_and_leaders.parquet']['sha256']:
        raise ValueError('LABEL_SHARED_CLOCK_DRIFT')
    times=pq.read_table(pa.BufferReader(shared),columns=['decision_ts_assumed_end']).column(0).to_numpy()
    return attach_label_times(labels,times,json.loads(CONTRACT.read_bytes())['proposed_segments'])
