"""Three as-of research groups; no labels, fitted statistics or trading rules."""
import numpy as np

FLAGS = ('nonpositive', 'falling', 'not_new_high')


def divide(total, count):
    return np.divide(total, count, out=np.full(np.broadcast_shapes(np.shape(total), np.shape(count)), np.nan), where=np.asarray(count) > 0)


def avg(values):
    valid = np.isfinite(values)
    return divide(np.where(valid, values, 0).sum(axis=0), valid.sum(axis=0))


def rate(flag, selected):
    """Unknown never false. Coverage denominator is the entire selected group."""
    known = selected & np.isfinite(flag)
    return divide(np.where(known, flag, 0).sum(axis=0), known.sum(axis=0)), divide(known.sum(axis=0), selected.sum(axis=0))


def prior(values, frequency):
    if frequency not in (5, 15) or values.shape[-1] % (240 // frequency):
        raise ValueError('INVALID_COMPLETE_DAILY_GRID')
    out = np.roll(values, 1, axis=-1).copy()
    out[..., np.arange(out.shape[-1]) % (240 // frequency) == 0] = np.nan
    return out


def day_high(level, complete, frequency):
    shape = level.shape
    grid = 240 // frequency
    x = level.reshape(*shape[:-1], -1, grid)
    proof = (complete & np.isfinite(level)).reshape(x.shape)
    prefix = np.logical_and.accumulate(proof, axis=-1)
    high = np.fmax.accumulate(np.where(prefix, x, np.nan), axis=-1)
    previous_high = np.roll(high, 1, axis=-1)
    known = prefix & np.isfinite(previous_high)
    known[..., 0] = False
    return np.where(known, x > previous_high, np.nan).reshape(shape)


def board_arrays(returns, members, frequency, exclude_index=None):
    """Count-aware basket and identity-preserving leaders, optionally leave one out.

    Leader selection/capture uses the original observed winners, then excludes
    the candidate; it never silently promotes a runner-up as the old leader.
    """
    total_members = len(members) - int(exclude_index in members)
    values = returns[members]
    valid = np.isfinite(values)
    prev = prior(values, frequency)
    pair = valid & np.isfinite(prev)
    # Same open-session reset as the accepted basket builder.
    pair[:, np.arange(values.shape[1]) % (120 // frequency) == 0] = False
    change = values - prev
    maximum = np.max(np.where(valid, values, -np.inf), axis=0)
    winner = valid & (values == maximum)
    daily = winner.reshape(len(members), -1, 240 // frequency)
    captured = np.logical_or.accumulate(daily, axis=-1)
    captured_before = np.roll(captured, 1, axis=-1)
    captured_before[..., 0] = False
    captured_before = captured_before.reshape(values.shape)
    if exclude_index in members:
        i = members.index(exclude_index)
        valid = valid.copy(); pair = pair.copy(); winner = winner.copy(); captured_before = captured_before.copy()
        valid[i] = pair[i] = winner[i] = captured_before[i] = False
    count = valid.sum(axis=0)
    mean = divide(np.where(valid, values, 0).sum(axis=0), count)
    motion = divide(np.where(pair, change, 0).sum(axis=0), pair.sum(axis=0))
    current_pair, captured_pair = winner & pair, captured_before & pair
    cur_n, cap_n = winner.sum(axis=0), captured_before.sum(axis=0)
    cur_mean = divide(np.where(current_pair, change, 0).sum(axis=0), current_pair.sum(axis=0))
    cap_mean = divide(np.where(captured_pair, change, 0).sum(axis=0), captured_pair.sum(axis=0))
    cur_mean = np.where((cur_n > 0) & (cur_n == current_pair.sum(axis=0)), cur_mean, np.nan)
    cap_mean = np.where((cap_n > 0) & (cap_n == captured_pair.sum(axis=0)), cap_mean, np.nan)
    return dict(mean=mean, breadth=divide(((values > 0) & valid).sum(axis=0), count),
        motion=motion, coverage=divide(count, total_members),
        high=day_high(1 + mean, (count == total_members) & (total_members > 0), frequency),
        current_change=cur_mean, captured_change=cap_mean,
        current_count=cur_n, captured_count=cap_n, common_count=pair.sum(axis=0))


def matrix_features(returns, memberships, frequency, exclude_index=None):
    rows = [board_arrays(returns, m, frequency, exclude_index) for m in memberships]
    return {k: np.stack([row[k] for row in rows]) for k in rows[0]}


def stall_flags(board, market_close, frequency):
    prev = prior(market_close, frequency)
    market_step = divide(market_close, prev) - 1
    motion = board['motion']
    known = np.isfinite(motion) & np.isfinite(market_step)
    market_high = day_high(market_close, np.isfinite(market_close), frequency)
    return dict(nonpositive=np.where(known, (market_step > 0) & (motion <= 0), np.nan),
        falling=np.where(known, (market_step > 0) & (motion < 0), np.nan),
        not_new_high=np.where(np.isfinite(board['high']) & np.isfinite(market_high),
                             (market_high == 1) & (board['high'] == 0), np.nan))


def summarize(board, markets, frequency, prefix, own=False):
    """Fixed candidate whitelist, not feature selection based on risk outcomes."""
    values = board['mean']
    valid = np.isfinite(values)
    selected = valid & (values > 0)
    out = {'observed_share': valid.mean(axis=0), 'positive_share': divide(selected.sum(axis=0), valid.sum(axis=0)),
           'mean': avg(values), 'member_coverage': avg(board['coverage'])}
    if not own:
        # Explicit columns, not warning-suppressed all-NULL nanquantiles.
        for q in (.1, .5, .9):
            result = np.full(values.shape[1], np.nan)
            has = valid.any(axis=0)
            result[has] = np.nanquantile(values[:, has], q, axis=0)
            out['q' + str(int(q * 100))] = result
    else:
        best = np.max(np.where(valid, values, -np.inf), axis=0)
        out['best'] = np.where(np.isfinite(best), best, np.nan)
        out['breadth'] = avg(board['breadth'])
        out['common_change'] = avg(board['motion'])
        out['current_leader_change'] = avg(board['current_change'])
        out['captured_leader_change'] = avg(board['captured_change'])
        out['current_leader_known_share'] = np.isfinite(board['current_change']).mean(axis=0)
        out['captured_leader_known_share'] = np.isfinite(board['captured_change']).mean(axis=0)
    for metric in ('current_change', 'captured_change'):
        flag = np.where(np.isfinite(board[metric]), board[metric] < 0, np.nan)
        out[metric + '_negative_rate'], out[metric + '_negative_coverage'] = rate(flag, selected)
    for benchmark, close in markets.items():
        flags = stall_flags(board, close, frequency)
        for flag, values in flags.items():
            out[benchmark + '_positive_' + flag + '_rate'], out[benchmark + '_positive_' + flag + '_coverage'] = rate(values, selected)
    return {prefix + name: values for name, values in out.items()}
