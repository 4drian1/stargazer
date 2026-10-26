"""Candidate-row aggregation only. No source IO, label creation or model fitting."""
import numpy as np
import pandas as pd

from continuous_motion_common_v1 import METRICS_COMMON
from intraday_relative_stall_v1 import TIME

KEYS = ['segment', 'code', TIME]
OWN_METRICS = (*METRICS_COMMON, 'current_deceleration', 'captured_deceleration')


def learning_fields():
    return [f'ccm_v1_{freq}_{scope}_{metric}' for freq in (5, 15)
            for scope in ('all', 'positive', 'own')
            for metric in (OWN_METRICS if scope == 'own' else METRICS_COMMON)]


def validate_population(population):
    if population[KEYS].isna().any().any() or population.duplicated(KEYS).any():
        raise ValueError('NULL_OR_DUPLICATE_CANDIDATE_KEY')
    if population.duplicated(['code', TIME]).any():
        raise ValueError('CANDIDATE_REPEATED_ACROSS_SEGMENTS')


def aggregate_own(population, assignments, frequency, memberships):
    """One equal-weight mean per candidate and field, never one training row per board.

    assignments must contain EVERY expected board, including all-NaN observations.
    memberships maps code to frozen board identities; duplicate or absent boards fail.
    Relative metrics must be computed on each board at the candidate cutoff beforehand.
    Audit counts report missing support; no counts become learning columns.
    """
    if frequency not in (5, 15):
        raise ValueError('INVALID_FREQUENCY')
    validate_population(population)
    a = assignments[KEYS + ['board'] + list(OWN_METRICS)].copy()
    if a[KEYS + ['board']].isna().any().any() or a.duplicated(KEYS + ['board']).any():
        raise ValueError('NULL_OR_DUPLICATE_BOARD_ASSIGNMENT')
    if np.isinf(a[list(OWN_METRICS)].to_numpy(dtype=float)).any():
        raise ValueError('INFINITE_BOARD_FEATURE')
    expected = {code: tuple(memberships.get(code, ())) for code in population.code.unique()}
    if any(not boards or len(set(boards)) != len(boards) for boards in expected.values()):
        raise ValueError('MISSING_OR_DUPLICATE_FROZEN_MEMBERSHIP')
    if any(board not in expected.get(code, ()) for code, board in
           a[['code', 'board']].drop_duplicates().itertuples(index=False, name=None)):
        raise ValueError('UNEXPECTED_BOARD_IDENTITY')
    idx = pd.MultiIndex.from_frame(population[KEYS])
    groups = a.groupby(KEYS, sort=False, dropna=False)
    counts = groups.size()
    if len(counts.index.difference(idx)) or len(idx.difference(counts.index)):
        raise ValueError('BOARD_ASSIGNMENT_CANDIDATE_KEYS_DIFFER')
    wanted = population.code.map(lambda code: len(expected[code])).to_numpy()
    if not np.array_equal(counts.reindex(idx).to_numpy(), wanted):
        raise ValueError('INCOMPLETE_BOARD_ASSIGNMENTS_KEEP_NULL_ROWS')
    result = population[KEYS].copy()
    means = groups[list(OWN_METRICS)].mean().reindex(idx)
    audit = population[KEYS].copy()
    audit[f'ccm_v1_{frequency}_own_expected_boards'] = wanted
    finite = groups[list(OWN_METRICS)].count().reindex(idx)
    for metric in OWN_METRICS:
        result[f'ccm_v1_{frequency}_own_{metric}'] = means[metric].to_numpy()
        audit[f'ccm_v1_{frequency}_own_{metric}_known_boards'] = finite[metric].to_numpy()
    return result, audit


def attach_sidecars(parts, sidecar, base_fields, expected_rows=86319):
    """Attach a completed 34-column candidate sidecar without changing any old cell.

    The shared all/positive scopes must already be produced by context_common at
    the same completed cutoff. This function does not manufacture those inputs.
    """
    population = pd.concat([f[['code', TIME]].assign(segment=s)
                            for s, f in parts.items()], ignore_index=True)
    validate_population(population)
    validate_population(sidecar)
    fields = learning_fields()
    if set(sidecar.columns) != set(KEYS + fields):
        raise ValueError('LEARNING_SIDECAR_SCHEMA_AUDIT_NOT_ALLOWED')
    if len(population) != expected_rows or len(sidecar) != expected_rows:
        raise ValueError('CANDIDATE_ROW_COUNT_CHANGED')
    source = sidecar.set_index(KEYS)
    idx = pd.MultiIndex.from_frame(population[KEYS])
    if len(source.index.difference(idx)) or len(idx.difference(source.index)):
        raise ValueError('SIDECAR_CANDIDATE_KEYS_DIFFER')
    if np.isinf(source[fields].to_numpy(dtype=float)).any():
        raise ValueError('INFINITE_SIDECAR_FEATURE')
    output = {}
    for segment, frame in parts.items():
        if set(fields) & set(frame.columns) or not set(base_fields).issubset(frame.columns):
            raise ValueError('OLD_SCHEMA_CHANGED_OR_NEW_FIELD_COLLISION')
        keys = frame[['code', TIME]].assign(segment=segment)[KEYS]
        addition = source.reindex(pd.MultiIndex.from_frame(keys))[fields].copy()
        addition.index = frame.index
        output[segment] = pd.concat([frame, addition], axis=1)
        pd.testing.assert_frame_equal(output[segment][frame.columns], frame)
    if output['train'][fields].isna().all().any():
        raise ValueError('ALL_NULL_TRAIN_NO_AUTOMATIC_COLUMN_DROP')
    return output
