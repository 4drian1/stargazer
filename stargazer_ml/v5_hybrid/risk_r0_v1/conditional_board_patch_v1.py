"""Explicitly scoped missing-only path patches and sparse observation differences."""
import numpy as np
import pandas as pd

from multiboard_risk_features_v1 import day_high
from newhigh_common_motion_v1 import combine


def patch_missing(original, fresh, allowed):
    original, fresh = np.asarray(original, dtype=float), np.asarray(fresh, dtype=float)
    allowed = np.asarray(allowed, dtype=bool)
    if original.shape != fresh.shape or allowed.shape != original.shape:
        raise ValueError('PATCH_SHAPE_MISMATCH')
    if np.isinf(original).any() or np.isinf(fresh).any():
        raise ValueError('INFINITE_PATH')
    if (allowed & np.isfinite(original)).any():
        raise ValueError('PATCH_WOULD_REPLACE_KNOWN_VALUE')
    result = original.copy()
    take = allowed & np.isfinite(fresh)
    result[take] = fresh[take]
    return result, int(take.sum())


def with_new_high(summary, markets, frequency, prefix):
    result = dict(summary)
    for benchmark, close in markets.items():
        high = day_high(close, np.isfinite(close), frequency)
        for flag in ('nonpositive', 'falling'):
            base = prefix + benchmark + '_positive_' + flag
            name = 'nh_' + prefix + benchmark + '_positive_common_' + flag
            result[name+'_rate'], result[name+'_coverage'] = combine(
                summary[base+'_rate'], summary[base+'_coverage'], summary[prefix+'positive_share'], high)
    return result


def differences(identity, before, after):
    if set(before) != set(after):
        raise ValueError('OBSERVATION_SCHEMA_CHANGED')
    frames = []
    for feature in before:
        a, b = np.asarray(before[feature], dtype=float), np.asarray(after[feature], dtype=float)
        if len(identity) != len(a) or a.shape != b.shape or np.isinf(a).any() or np.isinf(b).any():
            raise ValueError('INVALID_OBSERVATION_DIFFERENCE')
        changed = ~np.isclose(a, b, equal_nan=True, rtol=0, atol=1e-12)
        if changed.any():
            f = identity.loc[changed].copy()
            f['feature'], f['old_value'], f['new_value'] = feature, a[changed], b[changed]
            frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=[*identity.columns, 'feature','old_value','new_value'])
