"""Explicit research reader: ar_* fields coexist with every unchanged old field."""
import json

import numpy as np
import pandas as pd

import build_conditional_board_observations_v1 as build


def attach(frame,diff):
    keys=['code',build.TIME]
    if frame.duplicated(keys).any() or diff.duplicated([*keys,'feature']).any():
        raise ValueError('DUPLICATE_OVERLAY_KEY')
    out=frame.copy()
    for field,group in diff.groupby('feature',sort=False):
        name='ar_'+field
        if field not in frame or name in frame:
            raise ValueError('OVERLAY_FIELD_MISSING_OR_COLLISION')
        out[name]=frame[field].to_numpy(dtype=float,copy=True)
        for shared,part in [(True,group[group.code.eq('__SHARED__')]),
                            (False,group[~group.code.eq('__SHARED__')])]:
            if part.empty:
                continue
            if shared:
                patch=frame[keys].merge(part.drop(columns='code'),on=build.TIME,how='inner',validate='many_to_one')
            else:
                patch=frame[keys].merge(part,on=keys,how='inner',validate='one_to_one')
            pos=pd.MultiIndex.from_frame(frame[keys]).get_indexer(pd.MultiIndex.from_frame(patch[keys]))
            actual=frame[field].to_numpy(dtype=float)[pos]
            if not np.isclose(actual,patch.old_value,equal_nan=True,rtol=0,atol=1e-12).all():
                raise ValueError('OVERLAY_BASE_VALUE_CHANGED')
            out.loc[out.index[pos],name]=patch.new_value.to_numpy()
    pd.testing.assert_frame_equal(out[list(frame)],frame)
    return out


def read_segment(segment):
    if segment not in build.old.prep.SEGMENTS:
        raise ValueError('UNKNOWN_SEGMENT')
    acceptance=json.loads((build.HERE/'conditional_board_observations_acceptance_20261018_v1.json').read_bytes())
    report=json.loads((build.OUT/'report.json').read_bytes())
    if build.old.prep.sha(build.OUT/'report.json')!=acceptance['report_sha256']:
        raise ValueError('ACCEPTED_REPORT_DRIFT')
    name='observation_differences.parquet'
    if build.old.prep.sha(build.OUT/name)!=report['outputs'][name]['sha256']:
        raise ValueError('ACCEPTED_OVERLAY_DRIFT')
    diff=pd.read_parquet(build.OUT/name)
    original=build.read_segment(segment)
    side=build.old.read_context(segment).drop(columns='segment')
    original=original.merge(side,on=['code',build.TIME],validate='one_to_one',sort=False)
    result=attach(original,diff)
    result.attrs['scope']='CONDITIONAL_RESEARCH_NOT_FIT_OR_DEPLOYMENT'
    return result
