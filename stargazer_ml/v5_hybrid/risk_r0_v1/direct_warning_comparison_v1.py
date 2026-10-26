"""Fixed direct-warning schema and train-only descriptive profiles. No estimators."""
import hashlib

import numpy as np
import pandas as pd

from joint_fit_preparation_v1 import whitelist


def warning_fields():
    return [f'nh_{scope}{frequency}_{benchmark}_positive_common_{flag}_{kind}'
        for scope in ['mb','own'] for frequency in [5,15] for benchmark in ['sse','cyb']
        for flag in ['nonpositive','falling'] for kind in ['rate','coverage']]


def specification(baseline,new_fields,columns):
    if len(baseline)!=200 or len(set(baseline))!=200:
        raise ValueError('BASELINE_NOT_FROZEN_200_FIELDS')
    if new_fields!=warning_fields() or set(baseline)&set(new_fields):
        raise ValueError('DIRECT_WARNING_SCHEMA_CHANGED')
    fields=baseline+new_fields
    whitelist({'feature_groups':{'explicit':fields}})
    if any(f.startswith(('ar_','nh_')) for f in baseline):
        raise ValueError('BASELINE_NOT_ORIGINAL_M1')
    mapping={f:'ar_'+f if 'ar_'+f in columns else f for f in fields}
    if not set(mapping.values())<=set(columns):
        raise ValueError('MISSING_BOUND_SOURCE_FIELD')
    return dict(C0=list(baseline),C1=fields),mapping


def inputs(frame,fields,mapping):
    if not fields or len(fields)!=len(set(fields)):
        raise ValueError('DUPLICATE_OR_EMPTY_FIELDS')
    for f in fields:
        if mapping.get(f) not in [f,'ar_'+f]:
            raise ValueError('NONCANONICAL_OR_FUTURE_SOURCE_MAPPING')
    values=np.column_stack([frame[mapping[f]].to_numpy(dtype=float) for f in fields])
    if np.isinf(values).any():
        raise ValueError('INFINITE_MODEL_FIELD')
    return pd.DataFrame(values,columns=fields,index=frame.index)


def train_patterns(frame):
    groups={};empty=[];constant=[]
    for field in frame:
        values=frame[field].to_numpy(dtype=float,copy=True)
        if np.isinf(values).any():
            raise ValueError('INFINITE_TRAIN_FIELD')
        if np.isnan(values).all():
            empty.append(field)
        elif frame[field].nunique(dropna=True)==1:
            constant.append(field)
        values[np.isnan(values)]=np.nan
        values[values==0]=0.
        digest=hashlib.sha256(values.astype('<f8').tobytes()).hexdigest()
        groups.setdefault(digest,[]).append(field)
    duplicate=[v for v in groups.values() if len(v)>1]
    for group in duplicate:
        for field in group[1:]:
            np.testing.assert_array_equal(frame[group[0]].to_numpy(),frame[field].to_numpy())
    return dict(all_null_columns=empty,constant_nonmissing_columns=constant,
                exact_duplicate_groups_including_missing_pattern=duplicate,
                automatic_column_removal=False,scope='TRAIN_ONLY_DESCRIPTIVE_NOT_FEATURE_SELECTION')
