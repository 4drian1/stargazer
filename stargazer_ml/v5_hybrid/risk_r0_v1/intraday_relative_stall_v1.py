"""Causal observation candidates, not trading rules or a model. No market IO.

Sector common-member day-return changes are NOT index interval returns.
Close-based day highs are explicitly a research definition, not pivot confirmation.
"""
from datetime import datetime
import math

import pandas as pd

TIME = 'decision_ts_assumed_end'


def slots(day, frequency):
    if type(frequency) is not int or frequency not in (5, 15):
        raise ValueError('FREQUENCY_MUST_BE_5_OR_15')
    datetime.strptime(str(day), '%Y%m%d')
    return [day * 1000000 + (m // 60) * 10000 + (m % 60) * 100
            for start, end in ((570, 690), (780, 900))
            for m in range(start + frequency, end + 1, frequency)]


def number(value):
    return None if pd.isna(value) or not math.isfinite(float(value)) else float(value)


def close_path(frame, frequency):
    """One entity. Missing slots invalidate comparisons; day-high prefix never heals.

    Input level>0, day_return>-1; path_ok is positive provenance supplied by caller.
    Afternoon first interval uses 11:30 as previous trading cutoff (not lunch bars).
    """
    slots(20250102, frequency)
    if frame[TIME].duplicated().any():
        raise ValueError('DUPLICATE_CUTOFF')
    rows = []
    for day, group in frame.groupby(frame[TIME] // 1000000, sort=True):
        grid = slots(int(day), frequency)
        if not set(group[TIME]) <= set(grid):
            raise ValueError('OFF_GRID_OR_INCOMPLETE_BAR_END')
        indexed = group.set_index(TIME)
        previous = previous_return = high = None
        prefix = True
        for ts in grid:
            if ts not in indexed.index:
                previous = previous_return = None
                prefix = False
                continue
            row = indexed.loc[ts]
            level = number(row['level'])
            ret = number(row.get('day_return'))
            if level is not None and level <= 0:
                level = None
            if ret is not None and ret <= -1:
                ret = None
            ok = pd.notna(row['path_ok']) and bool(row['path_ok']) and level is not None
            prefix = prefix and ok
            new_high = bool(level > high) if prefix and high is not None else None
            drawdown = level / max(high, level) - 1 if prefix and high is not None else None
            rows.append({TIME: ts,
                'interval_return': level / previous - 1 if ok and previous is not None else None,
                'day_return_change': ret - previous_return
                    if ok and ret is not None and previous_return is not None else None,
                'close_new_day_high': new_high, 'drawdown_from_close_day_high': drawdown,
                'path_prefix_complete': prefix})
            if prefix:
                high = level if high is None else max(high, level)
            previous = level if ok else None
            previous_return = ret if ok else None
    return pd.DataFrame(rows)


def board_evidence(market, boards, frequency):
    """One benchmark, all boards, exact-time left join; no topK or winner selection.

    Boards: board_code, cutoff, day_return, common_change, breadth, full_members.
    full_members must certify the original frozen membership, not a moving denominator.
    """
    if boards.duplicated(['board_code', TIME]).any():
        raise ValueError('DUPLICATE_BOARD_CUTOFF')
    market_path = close_path(market, frequency).add_prefix('market_').rename(
        columns={'market_' + TIME: TIME})
    parts = []
    for code, group in boards.groupby('board_code', sort=True):
        group = group.sort_values(TIME).copy()
        path_input = group.assign(level=1 + group.day_return, path_ok=group.full_members)
        path = close_path(path_input, frequency).add_prefix('sector_').rename(
            columns={'sector_' + TIME: TIME})
        joined = group.merge(path, on=TIME, validate='one_to_one').merge(
            market_path, on=TIME, how='left', validate='many_to_one')
        positions = {ts: i for day in set(group[TIME] // 1000000)
                     for i, ts in enumerate(slots(int(day), frequency))}
        adjacent = ((joined[TIME] // 1000000 == joined[TIME].shift() // 1000000)
                    & (joined[TIME].map(positions) == joined[TIME].shift().map(positions) + 1))
        change = pd.to_numeric(joined.common_change, errors='coerce')
        joined['sector_day_return_change_common'] = change.where(adjacent & change.map(
            lambda x: pd.notna(x) and math.isfinite(x)))
        joined['sector_minus_market_day_return_change'] = (
            joined.sector_day_return_change_common - joined.market_day_return_change)
        full_pair = joined.full_members.fillna(False) & joined.full_members.shift().fillna(False)
        joined['sector_breadth_change_full'] = joined.breadth.diff().where(adjacent & full_pair)
        flags = {k: [] for k in ('market_new_high_sector_not_new_high',
                               'market_rising_sector_nonpositive', 'market_rising_sector_falling')}
        for row in joined.to_dict('records'):
            mh, sh = row.get('market_close_new_day_high'), row.get('sector_close_new_day_high')
            flags['market_new_high_sector_not_new_high'].append(
                bool(mh and not sh) if pd.notna(mh) and pd.notna(sh) else None)
            mr, sc = number(row.get('market_interval_return')), number(row.get('sector_day_return_change_common'))
            for key, falling in [('market_rising_sector_nonpositive', False),
                                 ('market_rising_sector_falling', True)]:
                flags[key].append(bool(mr > 0 and (sc < 0 if falling else sc <= 0))
                                  if mr is not None and sc is not None else None)
        for key, values in flags.items():
            joined[key] = pd.array(values, dtype='boolean')
        parts.append(joined)
    if not parts:
        return boards.copy()
    result = pd.concat(parts, ignore_index=True)
    result['observed_day_gain_rank'] = result.groupby(TIME).day_return.rank(method='min', ascending=False)
    result['positive_day_gain_observed'] = pd.array(
        [value > 0 if (value := number(x)) is not None else None for x in result.day_return],
        dtype='boolean')
    result['frequency_minutes'] = frequency
    return result


def leader_watch(boards, membership, stocks):
    """All supplied boards, ties retained; captures are explicitly within-day only.

    Partial ranks are observations, never certified full-universe leadership.
    Recovered stocks do not turn an incomplete observation into a strict rank.
    """
    if stocks.duplicated(['code', TIME]).any():
        raise ValueError('DUPLICATE_STOCK_CUTOFF')
    available = {}
    for row in stocks.to_dict('records'):
        value = number(row['day_return'])
        if value is not None and value > -1:
            available.setdefault(row[TIME], {})[row['code']] = value
    watched, records = {}, []
    for ts in sorted(boards[TIME].unique()):
        day = ts // 1000000
        quotes = available.get(ts, {})
        for board in sorted(boards.loc[boards[TIME] == ts, 'board_code'].unique()):
            members = membership.get(board, [])
            if len(set(members)) != len(members):
                raise ValueError('DUPLICATE_MEMBER')
            eligible = {c: quotes[c] for c in members if c in quotes}
            best = max(eligible.values()) if eligible else None
            current = {c for c, value in eligible.items() if value == best}
            captured = watched.setdefault((day, board), {})
            for code in current:
                captured.setdefault(code, ts)
            for code, capture in sorted(captured.items()):
                records.append(dict(board_code=board, code=code, **{TIME: ts},
                    current_observed_leader=code in current, capture_ts_within_day=capture,
                    day_return=quotes.get(code),
                    member_quote_coverage=len(eligible) / len(members) if members else None,
                    strict_member_rank=bool(members) and len(eligible) == len(members),
                    current_tie_count=len(current), unknown_reseal=True))
    return pd.DataFrame(records)
