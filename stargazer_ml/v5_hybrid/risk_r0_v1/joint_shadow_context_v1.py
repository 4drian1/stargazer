"""Raw-prefix risk context; no historical feature lookup, I/O, fitting or HTTP.

The caller supplies immutable-source snapshots from its data worker. Stock minutes
must use the frozen risk END clock (not the trigger kernel's START clock). A source
receipt is an input assertion, not an independent certification by this module.
"""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_relative_stall_v1 import TIME, slots
from joint_shadow_compute_v2 import ComputeUnavailable


@dataclass
class RiskSnapshot:
    cutoff: int
    available_at: int | None
    days: list
    stock_minutes: dict
    index_5m: dict
    boards: dict
    receipts: dict
    membership_known_at: int | None
    membership_scope: str
    observed_at: int | None = None


def load_snapshot(path):
    """Read a local data-worker manifest and SHA-bound raw files, never a URL.

    Manifest includes the RiskSnapshot scalar fields, boards and receipts. Each
    stock_minutes/index_5m entry is {path, sha256, format}; format is parquet or
    native lc5. Metadata receipt timestamps must be genuine producer records;
    old files without such records cannot be promoted into a forward snapshot.
    """
    path = Path(path).resolve()
    spec = json.loads(path.read_bytes())
    def read(entry):
        source = Path(entry['path'])
        source = source if source.is_absolute() else path.parent / source
        body = source.read_bytes()
        if hashlib.sha256(body).hexdigest() != entry['sha256']:
            raise ComputeUnavailable('RAW_SNAPSHOT_HASH_MISMATCH:' + str(source))
        if entry['format'] == 'parquet':
            return pd.read_parquet(io.BytesIO(body))
        if entry['format'] == 'lc5':
            from tdx_index_acceptance_v1 import read_binary
            return pd.DataFrame(read_binary(body))
        raise ComputeUnavailable('UNSUPPORTED_SNAPSHOT_FORMAT')
    return RiskSnapshot(
        cutoff=spec['cutoff'], available_at=spec['available_at'], days=spec['days'],
        stock_minutes={c: read(r) for c, r in spec['stock_minutes'].items()},
        index_5m={c: read(r) for c, r in spec['index_5m'].items()},
        boards=spec['boards'], receipts=spec['receipts'],
        membership_known_at=spec['membership_known_at'], membership_scope=spec['membership_scope'],
        observed_at=spec.get('observed_at'))


def _prefix(frame, cutoff, receipt, stock=False, research=False, observed_at=None):
    if not receipt or not receipt.get('reference'):
        raise ComputeUnavailable('SOURCE_RECEIPT_MISSING')
    if receipt.get('clock') != ('END_1M' if stock else 'END_5M'):
        raise ComputeUnavailable('SOURCE_CLOCK_UNBOUND')
    if receipt.get('price_basis') != 'RAW':
        raise ComputeUnavailable('SOURCE_SCALE_UNBOUND')
    available = receipt.get('available_at')
    assumed = research and receipt.get('availability_basis') == 'HISTORICAL_RECONSTRUCTION_ASSUMPTION'
    if available is None and not assumed:
        raise ComputeUnavailable('SOURCE_ARRIVAL_UNKNOWN')
    if available is not None and int(available) > (observed_at or cutoff):
        raise ComputeUnavailable('SOURCE_NOT_AVAILABLE_AS_OF')
    # Arrival timestamps are supplied by the producer; historical snapshots are
    # explicitly labelled research, never manufactured here from a bar timestamp.
    out = frame.loc[frame.date <= cutoff].copy()
    required = ['date', 'open', 'high', 'low', 'close']
    if stock:
        required += ['code', 'volume', 'amount']
    if not set(required) <= set(out):
        raise ComputeUnavailable('SOURCE_COLUMNS_MISSING')
    if out.empty or out.date.duplicated().any():
        raise ComputeUnavailable('EMPTY_OR_DUPLICATE_SOURCE')
    prices = out[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ComputeUnavailable('INVALID_SOURCE_PRICE')
    o, h, l, c = prices.T
    if ((l > np.minimum(o, c)) | (h < np.maximum(o, c))).any():
        raise ComputeUnavailable('INVALID_SOURCE_OHLC')
    if stock:
        activity = out[['volume', 'amount']].to_numpy(dtype=float)
        if not np.isfinite(activity).all() or (activity < 0).any():
            raise ComputeUnavailable('INVALID_SOURCE_ACTIVITY')
    return out.sort_values('date').reset_index(drop=True)


def _market(snapshot):
    from materialize_local_market_context_v1 import build, bars_and_days
    from golden_anchor_rules_v2 import replay
    from materialize_multitimeframe_momentum_v1 import build as momentum
    from join_momentum_review_v1 import project_market

    rows = {}
    for symbol in ('000001.SH', '399006.SZ'):
        if symbol not in snapshot.index_5m:
            raise ComputeUnavailable('INDEX_SOURCE_MISSING:' + symbol)
        frame = _prefix(snapshot.index_5m[symbol], snapshot.cutoff,
                        snapshot.receipts.get(symbol), research=(
                            snapshot.membership_scope == 'SNAPSHOT_RECONSTRUCTION_RESEARCH'),
                        observed_at=snapshot.observed_at)
        rows[symbol] = frame.to_dict('records')
        _, complete_days = bars_and_days(rows[symbol], 5)
        if not set(snapshot.days[:-1]) <= set(complete_days):
            raise ComputeUnavailable('INDEX_PAST_DAY_INCOMPLETE:' + symbol)
    _, daily = bars_and_days(rows['000001.SH'], 5)
    # The current incomplete session is never used to update the daily anchors.
    bars = [dict(day=d, open=v['close'], completed=True, **v) for d, v in sorted(daily.items())
            if d < snapshot.cutoff // 1000000]
    # Anchor contraction uses real daily opens, not close-as-open surrogates.
    opens = {r['date'] // 1000000: r['open'] for r in reversed(rows['000001.SH'])}
    for bar in bars:
        bar['open'] = opens[bar['day']]
    states = replay(bars)
    if not states:
        raise ComputeUnavailable('GOLDEN_HISTORY_NOT_READY')
    market = pd.DataFrame([r for symbol in rows for f in (5, 15)
                           for r in build(rows[symbol], symbol, states, f)])
    # Keep the entire supplied seed history; no arbitrary rolling truncation of
    # Wilder/EMA state. Completion/missing slots remain the frozen kernel's job.
    shared = pd.DataFrame({TIME: [t for d in snapshot.days for t in slots(d, 15)]})
    base = ['rsi14', 'rsi14_delta1', 'return_adjacent_bar', 'momentum_change',
            'day_return_prev_close', 'day_return_change_adjacent',
            'drawdown_from_observed_day_high', 'daily_return5_asof_prior_day',
            'daily_return20_asof_prior_day', 'daily_rsi14_asof_prior_day',
            'anchor_low', 'anchor_high', 'anchor_position', 'anchor_asof_day',
            'reference_close_day', 'session_fraction']
    for symbol, name in (('000001.SH', 'sse'), ('399006.SZ', 'cyb')):
        for f in (5, 15):
            data = market[(market.symbol == symbol) & (market.frequency_minutes == f)]
            cols = base + [c for c in data if c.startswith(
                ('golden_distance_', 'golden_level_', 'time_fraction_distance_'))]
            block = data[[TIME, *cols]].rename(columns={c: f'{name}_{f}m_{c}' for c in cols})
            shared = shared.merge(block, on=TIME, how='left', validate='one_to_one')
    projected, _ = project_market(momentum(market), set(shared.columns))
    shared = shared.merge(projected, on=TIME, how='left', validate='one_to_one')
    return market, states, shared


def _context(frame, codes, members, returns, stock_rows, days, board_codes):
    from build_joint_risk_features_v1 import basket_features, observed_leaders
    from build_joint_candidate_context_v1 import candidate_context

    baskets = [basket_features(returns, m) for m in members]
    means = np.stack([b['observed_return_raw'] for b in baskets])
    counts = np.stack([b['present'] for b in baskets])
    sizes = np.array([len(m) for m in members])
    totals = np.where(counts > 0, means * counts, 0)
    leaders = observed_leaders(means, counts, sizes, members, returns, board_codes, codes)
    leaders['day'] = np.repeat(days, 16)
    leaders['board_index'] = leaders.leader_observed_board.map(
        {c: i for i, c in enumerate(board_codes)}).fillna(-1).astype(int)
    ci = {c: i for i, c in enumerate(codes)}
    first, captured = {}, set()
    length = returns.shape[1]
    change_total, change_count = np.zeros(length), np.zeros(length, dtype=int)
    metrics, current_change = np.full((length, 4), np.nan), np.full(length, np.nan)
    metric_cols = ['stock_rsi14_5m_raw', 'stock_rsi14_15m_raw',
                   'stock_return5m_raw', 'stock_return15m_raw']
    metric_map = {c: g.set_index('cutoff_index')[metric_cols]
                  for c, g in stock_rows.groupby('code')}
    for row in leaders.itertuples():
        t, code = row.cutoff_index, row.leader_observed_stock
        if t % 16 == 0:
            captured = set()
        if t % 8 and captured:
            indexes = np.array([ci[c] for c in captured])
            valid = np.isfinite(returns[indexes, t]) & np.isfinite(returns[indexes, t-1])
            change_count[t] = valid.sum()
            change_total[t] = (returns[indexes[valid], t] - returns[indexes[valid], t-1]).sum()
        if isinstance(code, str) and code in ci:
            first.setdefault((code, row.day), t)
            captured.add(code)
            if code in metric_map and t in metric_map[code].index:
                metrics[t] = metric_map[code].loc[t].to_numpy(dtype=float)
            if t % 8:
                current_change[t] = returns[ci[code], t] - returns[ci[code], t-1]
    common_counts = np.zeros_like(counts)
    for b, m in enumerate(members):
        valid = np.isfinite(returns[m])
        common_counts[b, 1:] = (valid[:, 1:] & valid[:, :-1]).sum(axis=0)
    common_counts[:, np.arange(length) % 8 == 0] = 0
    common_mean = np.stack([b['common_member_return_change'] for b in baskets])
    up = np.stack([np.sum(np.isfinite(returns[m]) & (returns[m] > 0), axis=0) for m in members])
    extras = dict(up_totals=up, common_counts=common_counts,
                  common_totals=np.where(common_counts > 0, common_mean * common_counts, 0),
                  returns=returns, code_index=ci, current_leader_metrics=metrics,
                  current_leader_change=current_change, captured_change_total=change_total,
                  captured_change_count=change_count)
    affiliations = {c: [i for i, m in enumerate(members) if ci[c] in m] for c in frame.code}
    context = candidate_context(frame, totals, counts, sizes, affiliations, leaders, first, extras)
    # Frozen context sidecar stored floats as float32; preserve that boundary.
    for c in context.select_dtypes('float64'):
        context[c] = context[c].astype('float32')
    return frame.merge(context, on=['code', 'cutoff_index'], validate='one_to_one')


def build_features(snapshot, candidates):
    """Return source, ordered 232 and ordered 266 matrices plus audit metadata.

    One snapshot/decision cutoff per call. Membership is the supplied, explicitly
    disclosed comparison universe; missing constituents are NaN, not deleted.
    """
    import rebuild_features_v1 as frozen
    from build_joint_risk_features_v1 import stock_features
    from intraday_relative_stall_v2 import minute_returns
    from build_multiboard_full_features_v1 import market_arrays, golden_fields
    from multiboard_risk_features_v1 import matrix_features, summarize
    from conditional_board_patch_v1 import with_new_high
    from joint_fit_preparation_v1 import derive
    from build_common_candidate_sidecar_v1 import month_features

    cutoff = int(snapshot.cutoff)
    pd.to_datetime(str(cutoff), format='%Y%m%d%H%M%S', errors='raise')
    observed = snapshot.observed_at or cutoff
    pd.to_datetime(str(observed), format='%Y%m%d%H%M%S', errors='raise')
    if observed < cutoff or observed // 1000000 != cutoff // 1000000:
        raise ComputeUnavailable('INVALID_OBSERVATION_TIME')
    day = cutoff // 1000000
    days = list(snapshot.days)
    if (days != sorted(set(days)) or len(days) < 2 or days[-1] != day
            or cutoff not in slots(day, 15)):
        raise ComputeUnavailable('INVALID_PREFIX_CALENDAR_OR_CUTOFF')
    if snapshot.membership_scope not in ('SNAPSHOT_RECONSTRUCTION_RESEARCH', 'FORWARD_SNAPSHOT'):
        raise ComputeUnavailable('MEMBERSHIP_SCOPE_UNBOUND')
    research = snapshot.membership_scope == 'SNAPSHOT_RECONSTRUCTION_RESEARCH'
    for arrival in (snapshot.available_at, snapshot.membership_known_at):
        if arrival is None and not research:
            raise ComputeUnavailable('SNAPSHOT_ARRIVAL_UNKNOWN')
        if arrival is not None and arrival > observed:
            raise ComputeUnavailable('SNAPSHOT_NOT_AVAILABLE_AS_OF')
    if not snapshot.boards or any(not m or len(m) != len(set(m)) for m in snapshot.boards.values()):
        raise ComputeUnavailable('INVALID_BOARD_MEMBERSHIP')
    for c in candidates:
        if (int(c['cutoff_ts']) != cutoff or int(c['E']) != day or int(c['P']) != days[-2]):
            raise ComputeUnavailable('CANDIDATE_SNAPSHOT_TIME_MISMATCH')
    keys = pd.DataFrame([{'code': str(c['code']).zfill(6), TIME: cutoff} for c in candidates])
    if keys.duplicated().any():
        raise ComputeUnavailable('DUPLICATE_CANDIDATE_KEY')
    board_codes = list(snapshot.boards)
    codes = sorted({c for m in snapshot.boards.values() for c in m})
    if not set(keys.code) <= set(codes):
        raise ComputeUnavailable('CANDIDATE_MEMBERSHIP_MISSING')
    ci = {c: i for i, c in enumerate(codes)}
    members = [[ci[c] for c in snapshot.boards[b]] for b in board_codes]
    # Board paths/capture reset every day. Keep their matrices for E only, while
    # stock RSI/EMA/previous-session features still consume the full seed history.
    # This avoids holding N stocks x years of 5/15m returns and stock rows in RAM.
    context_days = [day]
    clock = {f: slots(day, f) for f in (5, 15)}
    native = {f: np.full((len(codes), len(clock[f])), np.nan) for f in (5, 15)}
    stock_rows, source_missing = [], []
    for code in codes:
        if code not in snapshot.stock_minutes:
            source_missing.append(code)
            continue
        raw = _prefix(snapshot.stock_minutes[code], cutoff, snapshot.receipts.get(code),
                      stock=True, research=research, observed_at=observed)
        rows, _, _ = stock_features(raw, code, np.array(days), set())
        # Reference close must be a complete previous day, exactly as the frozen
        # stock kernel. Never substitute pre_close, ETF or a stale daily close.
        refs = dict(zip(days,
                        rows.groupby(rows.cutoff_index // 16).stock_reference_close_raw.first()
                        .reindex(range(len(days))).to_numpy()))
        ret, _ = minute_returns(raw, code, days, refs)
        for f in native:
            native[f][ci[code]] = ret[f][-(240 // f):]
        rows = rows[rows.cutoff_index // 16 == len(days)-1].copy()
        rows['cutoff_index'] %= 16
        stock_rows.append(rows)
    if not stock_rows:
        raise ComputeUnavailable('NO_STOCK_INPUT')
    stocks = pd.concat(stock_rows, ignore_index=True)
    keys['cutoff_index'] = clock[15].index(cutoff)
    frame = keys.merge(stocks, on=['code', 'cutoff_index'], how='left', validate='one_to_one')
    market, states, shared = _market(snapshot)
    frame = frame.merge(shared, on=TIME, how='left', validate='many_to_one')
    frame = _context(frame, codes, members, native[15], stocks, context_days, board_codes)
    market_data, board_columns = {}, {}
    for f in (5, 15):
        grid, benchmarks = market_arrays(market, context_days, f)
        market_data[f] = benchmarks
        at = grid.index(cutoff)
        summary = matrix_features(native[f], members, f)
        shared_board = with_new_high(summarize(summary, benchmarks, f, f'mb{f}_'), benchmarks, f, f'mb{f}_')
        for name, values in shared_board.items():
            board_columns[name] = np.full(len(frame), values[at])
        for i, code in enumerate(frame.code):
            related = [m for m in members if ci[code] in m]
            own = matrix_features(native[f], related, f, exclude_index=ci[code])
            summary = with_new_high(summarize(own, benchmarks, f, f'own{f}_', own=True),
                                    benchmarks, f, f'own{f}_')
            for name, values in summary.items():
                board_columns.setdefault(name, np.full(len(frame), np.nan))[i] = values[at]
    for name, values in golden_fields(context_days, market_data[15]['sse'], states).items():
        board_columns[name] = np.full(len(frame), values[clock[15].index(cutoff)])
    frame = pd.concat([frame, pd.DataFrame(board_columns, index=frame.index)], axis=1)
    frame = derive(frame)
    mapping, fields232, fields266, _ = frozen.load_spec()
    # ar_ means repaired observation semantics. Recompute those semantics above;
    # do not join a historical corrections overlay for a new trading day.
    aliases = {source: frame[source[3:]] for source in set(mapping.values())
               if source.startswith('ar_') and source[3:] in frame}
    frame = pd.concat([frame, pd.DataFrame(aliases, index=frame.index)], axis=1)
    x232 = frozen.resolve_logical(frame, mapping, fields232)
    x266 = x232.copy()
    population = keys[['code', TIME]].assign(segment='forward_snapshot')
    for f in (5, 15):
        values, _, _ = month_features(population, codes, members, native[f], market_data[f], clock[f], f)
        for name in values:
            if name in fields266[232:]:
                x266[name] = values[name].to_numpy()
    x266 = x266[fields266]
    if np.isinf(x266.to_numpy(dtype=float)).any():
        raise ComputeUnavailable('INFINITE_FEATURE_VALUE')
    audit = dict(cutoff=cutoff, observed_at=snapshot.observed_at, available_at=snapshot.available_at,
                 membership_scope=snapshot.membership_scope, boards=len(members), stocks=len(codes),
                 missing_stock_sources=source_missing,
                 evidence_refs=sorted({r['reference'] for r in snapshot.receipts.values()}),
                 feature_count=[len(fields232), len(fields266)],
                 seed_policy='ENTIRE_SUPPLIED_PREFIX_NO_SYNTHETIC_WARMUP')
    return frame, x232, x266, audit
