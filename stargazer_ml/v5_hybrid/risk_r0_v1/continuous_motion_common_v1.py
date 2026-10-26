"""Common-member (3-bar stable) LOCAL momentum caliber -- NEW, independent of the
frozen continuous_motion_matrix_v1 (diagnosis #18.109 §4; reviewed and defect-fixed
after first pass). Independent module so the frozen original and this new version
keep distinct identities; it does NOT replace the old caliber and is not yet wired
into any model/admission.

Contract & the five defect fixes (see report §6):
1. FIXED common set: at bar t the level is computed over the SAME member set
   common(t) for t, t-1 and t-2, so adjacent baskets are never compared across
   different memberships. Deceleration is likewise computed ENTIRELY within
   common(t) (previous_return vs current_return, both over common(t)), so it cannot
   jump when membership changes (the first review's roll-based decel crossed two
   different member sets and produced spurious ~0.48pp spikes -- fixed).
2. SESSION-local window: the 3-bar window looks two bars back, so any common bar
   whose window REACHES a session-open gap return (overnight / lunch) would carry
   that gap into the local momentum. The gap bars AND the two following bars of each
   session are therefore zeroed (not just the open bars) -- so the lunch gap (and the
   symmetric AM-open gap) is never bridged into the basket level or deceleration.
3. all / positive scopes write DISTINCT field names (the old draft overwrote all
   with positive).
4. Leader identity is recorded from the REAL leader over the full board EVERY bar
   (independent of whether a common window exists), then published only where that
   identity is itself in the stable common set -- so an early (session-open) leader
   is still tracked/captured, and it is never silently promoted to 2nd place.
5. The RETRACEMENT (giveback) is deliberately NOT published here (neither the basket
   giveback nor the leader giveback): the 3-bar local momentum cannot prove a
   full-day high retracement (that needs separate stable full-day evidence); it is
   separated and not released as a completed field.

denominator / common_count / coverage are AUDIT / credibility fields (support
disclosure), NOT auto-entered into any model whitelist. min_common only guards
empty (NaN) output, it is NOT a credibility gate (no threshold tuned by risk).
"""
import numpy as np

from continuous_motion_matrix_v1 import path_arrays, divide

# Local caliber metric set: LOCAL deceleration + local basket-vs-benchmark relatives.
# 'giveback' (retracement) is intentionally EXCLUDED -- see contract point 5.
METRICS_COMMON = ('deceleration', *[f'{b}_relative_{m}' for b in ('sse', 'cyb')
                                    for m in ('return', 'deceleration')])


def board_motion_common(returns, members, frequency, stocks, denominator=None,
                        exclude=None, min_common=1):
    if frequency not in (5, 15):
        raise ValueError('INVALID_FREQUENCY')
    values = returns[members]
    valid = np.isfinite(values)
    raw_valid = valid.copy()  # full board BEFORE exclude handling (leader identity)
    width = 240 // frequency
    if values.shape[-1] % width:
        raise ValueError('INVALID_NATIVE_PATH_SHAPE')
    n_bars = values.shape[-1]
    days = n_bars // width
    half = width // 2
    # 3-bar common window, SESSION-local. Layout (members, bars): window along the
    # TIME axis, reshaped to (members, days, width); roll is along the width axis
    # only (cannot bridge across days). The gap returns sit at each session open
    # (bar 0 and bar `half`: overnight / lunch gap), so a bar whose 3-bar window
    # REACHES a gap bar would still carry that gap into the local momentum. We
    # therefore clear not just the gap bars but the two following bars of each
    # session too, so every remaining common bar's window is purely intraday
    # (defect #2, hardened -- also fixes the symmetric AM-open bridge at bars 2,3).
    v = valid.reshape(len(members), days, width)
    common3 = v & np.roll(v, 1, axis=2) & np.roll(v, 2, axis=2)
    dead = np.array([0, 1, 2, half, half + 1, half + 2])
    common3[:, :, dead] = False
    common = common3.reshape(len(members), n_bars).copy()
    total = len(members) - int(exclude in members)
    if exclude in members:
        i = members.index(exclude)
        common[i] = False
        valid = valid.copy(); valid[i] = False
    denom = denominator if denominator is not None else total
    n = common.sum(axis=0)
    safe = n >= min_common
    denom_n = np.where(safe, n, 1)
    # FIXED common set (defect #1): at bar t, level uses the SAME member set
    # common(t) for t, t-1 and t-2, so adjacent baskets are never compared across
    # different memberships.
    x0 = divide(np.where(common, values, 0).sum(axis=0), denom_n)                      # level at t
    x1 = divide(np.where(common, np.roll(values, 1, axis=1), 0).sum(axis=0), denom_n)  # at t-1
    x2 = divide(np.where(common, np.roll(values, 2, axis=1), 0).sum(axis=0), denom_n)  # at t-2
    # Same-set deceleration (defect #1, hardened): current_return and previous_return
    # BOTH use the identical common(t) membership, so the two compared moments never
    # span different member sets. previous_return[t] = (1+x1[t])/(1+x2[t])-1 spans
    # t-1 vs t-2 over common(t); current_return[t] = (1+x0[t])/(1+x1[t])-1 spans
    # t vs t-1 over common(t); deceleration = previous_return - current_return.
    current_return = divide(1 + x0, 1 + x1) - 1
    previous_return = divide(1 + x1, 1 + x2) - 1
    deceleration = np.where(safe, previous_return - current_return, np.nan)
    r = np.where(safe, current_return, np.nan)
    level = np.where(safe, x0, np.nan)
    output = dict(return_=r, deceleration=deceleration, day_return=level)
    # Leader identity is recorded from the REAL leader over the FULL board (raw_valid),
    # EVERY bar -- independent of whether a common window exists at that bar (defect #2
    # hardening: a leader at the day's first bars is still tracked and can be captured;
    # publication is gated separately by the common set, so the identity is never
    # silently lost, nor reassigned to 2nd place when unavailable -- defect #4).
    # A bar with NO valid quote on the whole board must NOT produce a phantom leader:
    # argmax over an all -inf column still returns index 0, which would be permanently
    # added to the captured set (corrupting downstream leader-basket deceleration). Only
    # record a leader where at least one real quote exists this bar.
    any_valid = raw_valid.any(axis=0)                       # (n_bars,): real quote present
    leader_val = np.where(raw_valid, values, -np.inf)
    leader_idx = np.argmax(leader_val, axis=0)
    cols = np.arange(n_bars)
    winner = np.zeros_like(values, dtype=bool)
    winner[leader_idx, cols] = True            # real leader every bar, full board
    winner &= any_valid[None, :]               # no leader at an all-invalid bar (no phantom)
    capture = np.logical_or.accumulate(winner.reshape(len(members), -1, width), axis=-1)
    capture = np.roll(capture, 1, axis=-1); capture[..., 0] = False
    capture = capture.reshape(values.shape)
    if exclude in members:
        j = members.index(exclude)
        winner = winner.copy(); capture = capture.copy()
        winner[j] = capture[j] = False   # excluded candidate never a leader/captured
    for label, mask in (('current', winner), ('captured', capture)):
        for metric in ('deceleration',):   # giveback NOT published here (defect #5)
            a = stocks[metric][members]
            # publish only where the designated leader stock is itself in the common
            # set (and its metric is known) -- identity preserved, publication gated.
            publish = mask & common
            known = publish & np.isfinite(a)
            count = publish.sum(axis=0)
            result = divide(np.where(known, a, 0).sum(axis=0), np.where(count > 0, count, 1))
            output[label + '_' + metric] = np.where(
                (count > 0) & (known.sum(axis=0) == count), result, np.nan)
    # disclosure fields: full roster denominator, common-member count, coverage.
    output['denominator'] = np.full(n_bars, float(denom))
    output['common_count'] = n.astype(float)
    output['coverage'] = divide(n, denom)
    return output


def schema_common(frequency):
    scopes = ['all', 'positive']
    names = [f'cm_common{frequency}_{scope}_{m}_{stat}' for scope in scopes
             for m in METRICS_COMMON for stat in ('mean', 'coverage')]
    for group in ('current', 'captured'):
        names += [f'cm_common{frequency}_{group}_deceleration_{stat}'
                  for stat in ('mean', 'coverage')]
    names += [f'cm_common{frequency}_denominator', f'cm_common{frequency}_common_count',
              f'cm_common{frequency}_coverage']
    return names


def context_common(returns, memberships, markets, frequency, exclude=None, stocks=None,
                  denominator=None, min_common=1):
    """Aggregate board_motion_common across boards (parity with context).

    Only the time-shaped metric fields are aggregated; the per-board denominator /
    common_count are pooled (summed). 'all' and 'positive' scopes write DISTINCT
    field names (defect #3).
    """
    if stocks is None:
        stocks = path_arrays(1 + returns, np.isfinite(returns), frequency)
    boards = [board_motion_common(returns, m, frequency, stocks, denominator=denominator,
                                  exclude=exclude, min_common=min_common) for m in memberships]
    if not boards:
        return {n: np.full(returns.shape[-1], np.nan) for n in schema_common(frequency)}
    keys = ('deceleration', 'return_', 'day_return', 'current_deceleration',
            'captured_deceleration')
    values = {k: np.stack([b[k] for b in boards]) for k in keys}
    metrics = {'deceleration': values['deceleration']}
    for benchmark, x in markets.items():
        market = path_arrays(x, np.isfinite(x), frequency)
        for metric, key in (('return', 'return_'), ('deceleration', 'deceleration')):
            metrics[f'{benchmark}_relative_{metric}'] = values[key] - market[key]
    result = {}

    def stats(name, a, selected):
        known = selected & np.isfinite(a)
        n = known.sum(axis=0)
        result[name + '_mean'] = divide(np.where(known, a, 0).sum(axis=0), n)
        result[name + '_coverage'] = divide(n, selected.sum(axis=0))

    all_rows = np.ones(values['day_return'].shape, dtype=bool)
    for scope in ('all', 'positive'):
        selected = all_rows if scope != 'positive' else (
            np.isfinite(values['day_return']) & (values['day_return'] > 0))
        for metric in METRICS_COMMON:
            stats(f'cm_common{frequency}_{scope}_{metric}', metrics[metric], selected)
    for group in ('current', 'captured'):
        stats(f'cm_common{frequency}_{group}_deceleration',
              values[group + '_deceleration'], all_rows)
    pooled_denom = np.stack([b['denominator'] for b in boards]).sum(axis=0)
    pooled_common = np.stack([b['common_count'] for b in boards]).sum(axis=0)
    result[f'cm_common{frequency}_denominator'] = pooled_denom
    result[f'cm_common{frequency}_common_count'] = pooled_common
    result[f'cm_common{frequency}_coverage'] = divide(pooled_common, pooled_denom)
    if list(result) != schema_common(frequency):
        raise ValueError('CONTINUOUS_COMMON_SCHEMA_DRIFT')
    return result
