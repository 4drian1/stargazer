"""Index close-new-high AND observed common-member nonpositive/falling motion.

This does not certify a whole sector's own high, and never recovers missing bars.
"""
import numpy as np


def combine(parent_rate, parent_coverage, positive_share, market_high):
    r, coverage, positive, high = np.broadcast_arrays(*[
        np.asarray(v, dtype=float) for v in (parent_rate, parent_coverage, positive_share, market_high)])
    for values in (r, coverage, positive):
        valid = np.isfinite(values)
        if np.isinf(values).any() or ((values[valid] < 0) | (values[valid] > 1)).any():
            raise ValueError('INVALID_BOUND_PARENT_RATIO')
    if np.isinf(high).any() or not np.isin(high[np.isfinite(high)], [0, 1]).all():
        raise ValueError('INVALID_INDEX_HIGH_STATE')
    if (np.isfinite(r) & ((positive <= 0) | ~np.isfinite(positive) | (coverage <= 0) | ~np.isfinite(coverage))).any():
        raise ValueError('PARENT_RATE_WITHOUT_EVALUABLE_SELECTED_GROUP')
    known = np.isfinite(r) & np.isfinite(high)
    rate = np.where(known, r * high, np.nan)
    # Unknown index => no evaluable conjunction, not an empty group or no risk.
    result_coverage = np.where(np.isfinite(high), coverage,
                               np.where(positive > 0, 0., np.nan))
    return rate, result_coverage


def sector_weights(codes, memberships):
    """Mean board member coverage = sum(stock observed * frozen stock weight)."""
    index = {code: i for i, code in enumerate(codes)}
    weights = np.zeros(len(codes))
    if not memberships:
        raise ValueError('EMPTY_BOARD_CATALOG')
    for members in memberships:
        if not members or len(set(members)) != len(members):
            raise ValueError('EMPTY_OR_DUPLICATE_BOARD_MEMBERS')
        for code in members:
            weights[index[code]] += 1. / len(members) / len(memberships)
    np.testing.assert_allclose(weights.sum(), 1., atol=1e-12, rtol=0)
    return weights


def footer_bounds(file):
    """All row groups must have date min/max. Missing metadata stays unknown."""
    column = file.schema_arrow.names.index('date')
    bounds = []
    for i in range(file.metadata.num_row_groups):
        group = file.metadata.row_group(i)
        if not group.num_rows:
            continue
        stats = group.column(column).statistics
        if stats is None or not stats.has_min_max:
            return None, None
        bounds.append((int(stats.min), int(stats.max)))
    return (min(a for a, _ in bounds), max(b for _, b in bounds)) if bounds else (None, None)
