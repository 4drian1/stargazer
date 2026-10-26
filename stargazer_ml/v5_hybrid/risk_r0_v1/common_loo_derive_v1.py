"""Fast, faithful derivation of per-board common-member caliber AND every member's
leave-one-out (LOO) output, without re-running board_motion_common per membership.

WHY: the real board_motion_common is correct but O(M * bars) and must be called per
(board, member) to get LOO -- with ~80 members/board and ~80k memberships that is
prohibitively slow for a one-time population acceptance. This module computes the board's
intermediates ONCE, then derives each member's LOO output by the closed-form adjustment
that the module's `exclude` branch performs (drop the excluded member's row from the
common set / leader mask, recompute the scalar sums). It is validated by
test_continuous_motion_common_v1.py to match board_motion_common exactly (with and without
exclude) on synthetic boards, including large-M boards that mimic the real concept boards.

It imports ONLY the module's `divide` and the frozen matrix `path_arrays` -- no fit /
population / IO -- so it is safe to import from both the verifier and the unit test.
"""
import numpy as np

from continuous_motion_common_v1 import divide


def compute_board(returns, members, frequency, stocks, min_common=1):
    """Return the board's intermediates needed for board-level fields and LOO derivation."""
    if frequency not in (5, 15):
        raise ValueError('INVALID_FREQUENCY')
    values = returns[members]
    valid = np.isfinite(values)
    raw_valid = valid.copy()
    M = len(members)
    width = 240 // frequency
    n_bars = values.shape[-1]
    days = n_bars // width
    half = width // 2
    v = valid.reshape(M, days, width)
    common3 = v & np.roll(v, 1, axis=2) & np.roll(v, 2, axis=2)
    dead = np.array([0, 1, 2, half, half + 1, half + 2])
    common3[:, :, dead] = False
    common = common3.reshape(M, n_bars).copy()
    n = common.sum(axis=0)                                  # (n_bars,) int
    safe = n >= min_common
    denom_n = np.where(safe, n, 1)
    Sx0 = np.where(common, values, 0.0).sum(axis=0)
    Sx1 = np.where(common, np.roll(values, 1, axis=1), 0.0).sum(axis=0)
    Sx2 = np.where(common, np.roll(values, 2, axis=1), 0.0).sum(axis=0)
    any_valid = raw_valid.any(axis=0)
    leader_val = np.where(raw_valid, values, -np.inf)
    leader_idx = np.argmax(leader_val, axis=0)
    cols = np.arange(n_bars)
    winner = np.zeros_like(values, dtype=bool)
    winner[leader_idx, cols] = True
    winner &= any_valid[None, :]
    capture = np.logical_or.accumulate(winner.reshape(M, -1, width), axis=-1)
    capture = np.roll(capture, 1, axis=-1)
    capture[..., 0] = False
    capture = capture.reshape(values.shape)
    a = stocks['deceleration'][members]                     # (M, n_bars)
    return dict(M=M, n_bars=n_bars, width=width, values=values, common=common, n=n,
                safe=safe, denom_n=denom_n, Sx0=Sx0, Sx1=Sx1, Sx2=Sx2,
                winner=winner, capture=capture, a=a, min_common=min_common)


def board_fields(inter):
    """Board-level fields (no exclusion) -- equals board_motion_common's output dict for the
    four kinetic fields + common_count."""
    x0 = divide(inter['Sx0'], inter['denom_n'])
    x1 = divide(inter['Sx1'], inter['denom_n'])
    x2 = divide(inter['Sx2'], inter['denom_n'])
    cur = divide(1 + x0, 1 + x1) - 1
    prev = divide(1 + x1, 1 + x2) - 1
    deceleration = np.where(inter['safe'], prev - cur, np.nan)
    r = np.where(inter['safe'], cur, np.nan)

    def _leader(mask):
        publish = mask & inter['common']
        known = publish & np.isfinite(inter['a'])
        count = publish.sum(axis=0)
        known_n = known.sum(axis=0)
        known_sum = np.where(known, inter['a'], 0.0).sum(axis=0)
        result = divide(known_sum, np.where(count > 0, count, 1))
        return np.where((count > 0) & (known_n == count), result, np.nan)

    return dict(return_=r, deceleration=deceleration,
                current_deceleration=_leader(inter['winner']),
                captured_deceleration=_leader(inter['capture']),
                common_count=inter['n'].astype(float))


def loo_field(inter, i, field):
    """LOO field for excluding member row `i` -- equals board_motion_common(..., exclude=...)."""
    common = inter['common']; values = inter['values']; n = inter['n']; a = inter['a']
    min_common = inter['min_common']
    cmi = common[i]
    n_excl = (n - cmi.astype(int)).astype(float)
    safe_excl = n_excl >= min_common
    denom_n_excl = np.where(safe_excl, n_excl, 1)
    Sx0 = inter['Sx0'] - np.where(cmi, values[i], 0.0)
    Sx1 = inter['Sx1'] - np.where(cmi, np.roll(values, 1, axis=1)[i], 0.0)
    Sx2 = inter['Sx2'] - np.where(cmi, np.roll(values, 2, axis=1)[i], 0.0)
    if field in ('return_', 'deceleration'):
        x0 = divide(Sx0, denom_n_excl)
        x1 = divide(Sx1, denom_n_excl)
        x2 = divide(Sx2, denom_n_excl)
        cur = divide(1 + x0, 1 + x1) - 1
        prev = divide(1 + x1, 1 + x2) - 1
        if field == 'return_':
            return np.where(safe_excl, cur, np.nan)
        return np.where(safe_excl, prev - cur, np.nan)
    # leader fields (current / captured): drop row i from the leader mask and common set
    if field == 'current_deceleration':
        mask = inter['winner'].copy()
    else:
        mask = inter['capture'].copy()
    mask[i] = False
    common_excl = common.copy()
    common_excl[i] = False
    publish = mask & common_excl
    known = publish & np.isfinite(a)
    count = publish.sum(axis=0)
    known_n = known.sum(axis=0)
    known_sum = np.where(known, a, 0.0).sum(axis=0)
    result = divide(known_sum, np.where(count > 0, count, 1))
    return np.where((count > 0) & (known_n == count), result, np.nan)


def loo_all_members(inter, field):
    """Vectorized version of loo_field: returns a (M, n_bars) array where row i is the LOO
    output for excluding member i. Equivalent to stacking [loo_field(inter, i, field) for i],
    but computes all members at once (no per-member python loop / no per-call allocation).
    Validated against loo_field in the unit test."""
    common = inter['common']; values = inter['values']; n = inter['n']; a = inter['a']
    min_common = inter['min_common']
    if field in ('return_', 'deceleration'):
        contrib0 = np.where(common, values, 0.0)
        contrib1 = np.where(common, np.roll(values, 1, axis=1), 0.0)
        contrib2 = np.where(common, np.roll(values, 2, axis=1), 0.0)
        Sx0_adj = inter['Sx0'][None, :] - contrib0
        Sx1_adj = inter['Sx1'][None, :] - contrib1
        Sx2_adj = inter['Sx2'][None, :] - contrib2
        n_excl = n[None, :] - common.astype(int)
        safe_excl = n_excl >= min_common
        denom = np.where(safe_excl, n_excl, 1)
        x0 = divide(Sx0_adj, denom)
        x1 = divide(Sx1_adj, denom)
        x2 = divide(Sx2_adj, denom)
        cur = divide(1 + x0, 1 + x1) - 1
        prev = divide(1 + x1, 1 + x2) - 1
        if field == 'return_':
            return np.where(safe_excl, cur, np.nan)
        return np.where(safe_excl, prev - cur, np.nan)
    # leader fields: drop row i from the leader mask and common set, per member.
    # The published value is the MEAN of the per-stock deceleration over the published AND
    # known members (excluding i) -- NOT member i's own deceleration.
    if field == 'current_deceleration':
        mask = inter['winner']
    else:
        mask = inter['capture']
    pub = mask & common                                   # (M, n_bars) publish flag per member
    K = pub & np.isfinite(a)                              # (M, n_bars) known flag per member
    count_total = pub.sum(axis=0)                          # (n_bars,)
    known_n_total = K.sum(axis=0)                          # (n_bars,)
    known_sum_total = np.where(K, a, 0.0).sum(axis=0)      # (n_bars,)
    count_excl = count_total[None, :] - pub                # (M, n_bars)
    known_n_excl = known_n_total[None, :] - K
    known_sum_excl = known_sum_total[None, :] - np.where(K, a, 0.0)
    result = divide(known_sum_excl, np.where(count_excl > 0, count_excl, 1))
    return np.where((count_excl > 0) & (known_n_excl == count_excl), result, np.nan)
