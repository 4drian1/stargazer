"""Batch causal implementation of v1 observations; v1/frozen pilot are untouched."""
import numpy as np
import pandas as pd

from intraday_relative_stall_v1 import TIME, slots


def paths(frame, frequency, identity='code'):
    """Reindex necessary grid before shifts/cummax; output only input observations.

    All shifts and day-high prefixes group by identity AND trading day. Missing
    slots break adjacency and invalidate that day's remaining high-prefix proof.
    """
    slots(20250102, frequency)
    if frame.duplicated([identity, TIME]).any():
        raise ValueError('DUPLICATE_ENTITY_CUTOFF')
    if frame.empty:
        raise ValueError('EMPTY_PATH_INPUT')
    source = frame.copy()
    source['_day'] = source[TIME] // 1000000
    days = sorted(source._day.unique())
    grid = [t for day in days for t in slots(int(day), frequency)]
    if not set(source[TIME]) <= set(grid):
        raise ValueError('OFF_GRID_OR_INCOMPLETE_BAR_END')
    keys = pd.MultiIndex.from_product([sorted(source[identity].unique()), grid], names=[identity, TIME])
    source['_observed'] = True
    f = source.set_index([identity, TIME]).reindex(keys).reset_index()
    groups = [f[identity], f[TIME] // 1000000]
    level = pd.to_numeric(f.level, errors='coerce')
    ret = pd.to_numeric(f.get('day_return', pd.Series(np.nan, index=f.index)), errors='coerce')
    level = level.where(np.isfinite(level) & (level > 0))
    ret = ret.where(np.isfinite(ret) & (ret > -1))
    ok = f.path_ok.fillna(False).astype(bool) & level.notna()
    prefix = ok.groupby(groups).cummin()
    previous = level.where(ok).groupby(groups).shift()
    previous_ret = ret.where(ok).groupby(groups).shift()
    high = level.where(prefix).groupby(groups).cummax()
    prior_high = high.groupby(groups).shift()
    new_high = (level > prior_high).astype(object).where(prefix & prior_high.notna(), None)
    out = f[[identity, TIME]].copy()
    out['interval_return'] = (level / previous - 1).where(ok & previous.notna())
    out['day_return_change'] = (ret - previous_ret).where(ok & ret.notna())
    out['close_new_day_high'] = new_high
    out['drawdown_from_close_day_high'] = (level / high - 1).where(prefix & prior_high.notna())
    out['path_prefix_complete'] = prefix
    return out.loc[f._observed.fillna(False).astype(bool)].reset_index(drop=True)


def board_evidence(market, boards, frequency):
    """Same fields and semantics as v1; all boards ranked, never topK-selected."""
    if boards.duplicated(['board_code', TIME]).any():
        raise ValueError('DUPLICATE_BOARD_CUTOFF')
    if boards.empty:
        return boards.copy()
    market_path = paths(market.assign(code='BENCHMARK'), frequency).drop(columns='code')
    market_path = market_path.rename(columns={c: 'market_' + c for c in market_path if c != TIME})
    sector_input = boards.assign(level=1 + boards.day_return, path_ok=boards.full_members)
    sector_path = paths(sector_input, frequency, 'board_code')
    sector_path = sector_path.rename(columns={c: 'sector_' + c for c in sector_path
                                               if c not in ['board_code', TIME]})
    f = boards.sort_values(['board_code', TIME]).merge(sector_path, on=['board_code', TIME],
        validate='one_to_one').merge(market_path, on=TIME, how='left', validate='many_to_one')
    grid = {t: i for day in sorted(set(f[TIME] // 1000000))
            for i, t in enumerate(slots(int(day), frequency))}
    grouped = f.groupby('board_code', sort=False)
    previous_ts = grouped[TIME].shift()
    adjacent = ((f[TIME] // 1000000 == previous_ts // 1000000)
                & (f[TIME].map(grid) == previous_ts.map(grid) + 1))
    change = pd.to_numeric(f.common_change, errors='coerce')
    f['sector_day_return_change_common'] = change.where(adjacent & np.isfinite(change))
    f['sector_minus_market_day_return_change'] = f.sector_day_return_change_common - f.market_day_return_change
    full_pair = f.full_members.fillna(False) & grouped.full_members.shift().fillna(False)
    f['sector_breadth_change_full'] = grouped.breadth.diff().where(adjacent & full_pair)
    mh, sh = f.market_close_new_day_high, f.sector_close_new_day_high
    f['market_new_high_sector_not_new_high'] = (
        mh.astype('boolean') & ~sh.astype('boolean')).where(mh.notna() & sh.notna()).astype('boolean')
    mr, sc = f.market_interval_return, f.sector_day_return_change_common
    known = mr.notna() & sc.notna()
    f['market_rising_sector_nonpositive'] = ((mr > 0) & (sc <= 0)).where(known).astype('boolean')
    f['market_rising_sector_falling'] = ((mr > 0) & (sc < 0)).where(known).astype('boolean')
    f['observed_day_gain_rank'] = f.groupby(TIME).day_return.rank(method='min', ascending=False)
    f['positive_day_gain_observed'] = (f.day_return > 0).where(f.day_return.notna()).astype('boolean')
    f['frequency_minutes'] = frequency
    return f


def minute_returns(frame, code, days, references, excluded_days=()):
    """Exact 1m-end grids -> independent complete 5/15m closes and raw day returns.

    references bind prior complete day-close values from accepted derived rows.
    No adjustment, pre_close substitution, source merging or missing-slot fill.
    """
    fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
    if not (frame.code.astype(str) == code).all():
        raise ValueError('SOURCE_CODE_MISMATCH')
    duplicate = frame[frame.duplicated('date', keep=False)]
    conflicts = set(duplicate.groupby('date')[fields].nunique(dropna=False).max(axis=1).loc[lambda s: s > 1].index)
    f = frame.drop_duplicates('date').sort_values('date')
    ts = f.date.to_numpy(dtype=np.int64)
    day = ts // 1000000
    minute = ts // 10000 % 100 * 60 + ts // 100 % 100
    am, pm = (minute > 570) & (minute <= 690), (minute > 780) & (minute <= 900)
    numbers = f[fields].to_numpy(dtype=float)
    o, h, l, c, v, a = numbers.T
    valid = (np.isfinite(numbers).all(axis=1) & (np.minimum.reduce([o, h, l, c]) > 0)
        & (l <= np.minimum(o, c)) & (h >= np.maximum(o, c)) & (v >= 0) & (a >= 0)
        & (ts % 100 == 0) & (ts // 100 % 100 < 60) & (am | pm)
        & np.isin(day, days) & ~np.isin(day, list(excluded_days)) & ~np.isin(ts, list(conflicts)))
    locations = {d: i for i, d in enumerate(days)}
    positions = np.array([locations[d] for d in day[valid]], dtype=int) * 240
    positions += np.where(am[valid], minute[valid] - 571, 120 + minute[valid] - 781)
    closes = np.full(len(days) * 240, np.nan)
    closes[positions] = c[valid]
    result = {}
    for step in (5, 15):
        groups = closes.reshape(-1, step)
        close = np.where(np.isfinite(groups).all(axis=1), groups[:, -1], np.nan)
        reference = np.repeat(np.array([references.get(d, np.nan) for d in days], dtype=float), 240 // step)
        returns = np.divide(close, reference, out=np.full_like(close, np.nan),
                            where=np.isfinite(reference) & (reference > 0)) - 1
        result[step] = np.where(np.isfinite(returns) & (returns > -1), returns, np.nan)
    return result, dict(conflicting_timestamps=len(conflicts), invalid_or_excluded_rows=int((~valid).sum()))


def basket(returns, member_indexes, frequency):
    """Observed raw basket with fixed membership; missing members are not zero."""
    slots(20250102, frequency)
    values = returns[member_indexes]
    valid = np.isfinite(values)
    n = valid.sum(axis=0)
    total = np.where(valid, values, 0).sum(axis=0)
    mean = np.divide(total, n, out=np.full(total.shape, np.nan), where=n > 0)
    up = np.divide(((values > 0) & valid).sum(axis=0), n,
                   out=np.full(total.shape, np.nan), where=n > 0)
    common = valid[:, 1:] & valid[:, :-1]
    count = common.sum(axis=0)
    delta = np.where(common, values[:, 1:] - values[:, :-1], 0).sum(axis=0)
    motion = np.r_[np.nan, np.divide(delta, count, out=np.full(delta.shape, np.nan), where=count > 0)]
    motion[np.arange(len(motion)) % (120 // frequency) == 0] = np.nan
    return dict(day_return=mean, breadth=up, present=n,
                coverage=n / len(member_indexes) if member_indexes else np.zeros_like(n, dtype=float),
                common_change=motion, full_members=(n == len(member_indexes)) & bool(member_indexes))
