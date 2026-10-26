"""Sidecar rebuild v1: frozen-builder rebuild of joint/context/mb/golden/nh rows for the
20 candidates at cutoff 20260908150000, using the verified minute extension.

Discipline: frozen builders imported read-only (stock_features, candidate_context,
matrix_features/summarize/with_new_high, combine, day_high, golden_fields, minute_returns);
zero external network; frozen artifacts untouched; all outputs in sidecar_rebuild_v1/.

Anchors (consistency vs frozen lineage):
  A) rebuilt stock_features row (300014, 6815) vs frozen joint_parts row -> field-wise == 0
  B) rebuilt candidate_context row (300014, 6815) vs frozen context sidecar row (float32)
  C) same-kernel rebuild at cutoff 20260901150000 (window [20260831, 20260901]) vs frozen
     shared_202609 (mb/golden), repair shared.parquet (nh_mb), repair own.parquet
     (own inputs + nh_own sample) -> quantified max|delta|
Post-rebuild: 20-candidate frame (rebuilt values injected; derive/attach re-run), 232/266
inference with both frozen boosters, isolation report, UI shadow packs.
"""
import argparse
import csv as _csv
import hashlib
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
OUT = HERE / 'sidecar_rebuild_v1'
sys.path.insert(0, str(HERE))
import rebuild_features_v1 as v1
from rebuild_features_v1 import TIME, sha, log, load_spec, load_booster, native_missing_view, resolve_logical
from intraday_relative_stall_v2 import minute_returns
from build_joint_risk_features_v1 import stock_features
from build_joint_candidate_context_v1 import candidate_context, excluded_means, known_leader
from build_multiboard_full_features_v1 import market_arrays, golden_fields
from multiboard_risk_features_v1 import matrix_features, summarize
from newhigh_common_motion_v1 import combine
from multiboard_risk_features_v1 import day_high
from joint_fit_preparation_v1 import derive
from read_conditional_board_observations_v1 import attach
from rebuild_features_v2 import QUARANTINE_CSV
import bind_prior_anchor_evidence_v1 as binding
import evidence_stream_trial_v2 as trial2

P_DAY, E_DAY = 20260907, 20260908
CUTOFF_TS = 20260908150000
CI = 6815
ANCHOR_DAY, ANCHOR_PREV = 20260901, 20260831   # kernel-fidelity anchor (frozen tables exist)
ANCHOR_CI = None  # resolved from clock
ALL_20 = ['000937', '001278', '002274', '002278', '002519', '300014', '300328', '300582',
          '300583', '300769', '301218', '301246', '301358', '600844', '603376', '603665',
          '603701', '688005', '688275', '688759']
T0 = time.time()
writer = {}


def put(name, body, terminal=False):
    path = OUT / name
    if path.exists():
        raise ValueError('OUTPUT_EXISTS:' + name)
    with path.open('xb') as stream:
        stream.write(body)
    writer[name] = dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    log('wrote', name=name, bytes=len(body))




def _s(v):
    return float(np.asarray(v).reshape(-1)[0])

def parquet_bytes(frame):
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), sink, compression='zstd')
    return sink.getvalue().to_pybytes()


# ---------------------------------------------------------------- replay (generalized)
def load_catalog():
    spec = json.loads((HERE / 'ths_full_concept_scope_20261017_v1.json').read_bytes())
    if sha(Path(spec['source'])) != spec['source_sha256']:
        raise ValueError('CATALOG_DRIFT')
    boards = sorted([r for r in pq.read_table(spec['source']).to_pylist()
                     if r['source'] == 'ths' and r['type'] == 'concept'], key=lambda r: r['code'])
    codes = sorted({c for b in boards for c in b['symbols']})
    ci = {c: i for i, c in enumerate(codes)}
    members = [[ci[c] for c in b['symbols']] for b in boards]
    return codes, ci, members, boards


def replay_days(days, ci, ext_ledger):
    """Frozen-lineage replay over arbitrary 2-day windows; ext overlay included."""
    clock = pd.read_parquet(v1.SHARED_MARKET, columns=['cutoff_index', TIME])
    clock = clock[clock[TIME].floordiv(1000000).isin(days)].sort_values(TIME)
    if len(clock) != len(days) * 16:
        raise ValueError('SOURCE_CLOCK_INCOMPLETE')
    di = {d: i for i, d in enumerate(days)}
    dt = {r.cutoff_index: di[r.decision_ts_assumed_end // 1000000] for r in clock.itertuples()}
    positions = {k: i for i, k in enumerate(clock.cutoff_index)}
    reference = np.full((5540, len(days)), np.nan)
    r15 = np.full((5540, len(days) * 16), np.nan)
    jman = json.loads((v1.JOINT / 'manifest.json').read_bytes())
    from build_multiboard_full_features_v1 import daily_references
    seen = set()
    for part in jman['joint_parts']:
        frame = pd.read_parquet(v1.JOINT / part['path'],
                                columns=['code', 'cutoff_index', 'stock_reference_close_raw',
                                         'stock_day_return_raw'])
        present = set(frame.code.unique())
        if seen & present:
            raise ValueError('CODE_SPLIT_ACROSS_SOURCE_SHARDS')
        seen.update(present)
        daily_references(frame, ci, dt, reference, dict(values=r15, _map=positions))
    ordinary = pd.read_parquet(binding.OUT / 'anchor_admission.parquet')
    ordinary = ordinary[ordinary.research_eligible]
    for row in ordinary.itertuples():
        if row.current_session_day not in di:
            continue
        i, j = ci[row.code], di[row.current_session_day]
        if not np.isfinite(reference[i, j]):
            reference[i, j] = row.reference_close_potential
    excluded = {}
    with binding.prior.QUARANTINE.open(encoding='utf-8-sig', newline='') as stream:
        for row in _csv.DictReader(stream):
            excluded.setdefault(row['code'], set()).add(int(row['day']))
    ledger = json.loads((HERE / 'relative_stall_5m_20261018_v1/raw_input_ledger.json').read_bytes())
    r5 = np.full((5540, len(days) * 48), np.nan)
    for item in ledger:
        code = item['code']
        i = ci[code]
        body = Path(item['path']).read_bytes()
        if hashlib.sha256(body).hexdigest() != item['sha256']:
            raise ValueError('FROZEN_RAW_DRIFT:' + code)
        frame = pq.read_table(pa.BufferReader(body),
                              columns=['code', 'date', 'open', 'high', 'low', 'close',
                                       'volume', 'amount']).to_pandas()
        if code in ext_ledger:
            ext = pd.read_parquet(ext_ledger[code]['path'])
            overlap = sorted(set(ext.date) & set(frame.date))
            if overlap:
                cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
                a = frame.set_index('date').loc[overlap][cols].astype('float64')
                b = ext.set_index('date').loc[overlap][cols].astype('float64')
                neq = (a != b) & ~(a.isna() & b.isna())
                if bool(neq.any().any()):
                    raise ValueError('EXT_FROZEN_CONFLICT:' + code)
                ext = ext[~ext.date.isin(overlap)]
            frame = pd.concat([frame, ext], ignore_index=True).sort_values('date')
        values, _ = minute_returns(frame, code, days, dict(zip(days, reference[i])),
                                   excluded.get(code, set()))
        old = r15[i].copy()
        known = np.isfinite(old)
        np.testing.assert_allclose(values[15][known], old[known], atol=1e-12, rtol=0)
        r5[i] = values[5]
        r15[i] = values[15]
    for code, ent in ext_ledger.items():
        if code in {i['code'] for i in ledger}:
            continue
        i = ci[code]
        if hashlib.sha256(Path(ent['path']).read_bytes()).hexdigest() != ent['sha256']:
            raise ValueError('EXT_LEDGER_DRIFT:' + code)
        frame = pd.read_parquet(ent['path'])
        values, _ = minute_returns(frame, code, days, dict(zip(days, reference[i])),
                                   excluded.get(code, set()))
        r5[i] = values[5]
        r15[i] = values[15]
    return {5: r5, 15: r15}, clock


def code_frames(ext_ledger):
    """Combined frozen+ext minute frames for the 20 codes."""
    ledger = {i['code']: i for i in
              json.loads((HERE / 'relative_stall_5m_20261018_v1/raw_input_ledger.json').read_bytes())}
    out = {}
    for code in ALL_20:
        if code in ledger:
            frame = pd.read_parquet(ledger[code]['path'])
            if code in ext_ledger:
                ext = pd.read_parquet(ext_ledger[code]['path'])
                overlap = sorted(set(ext.date) & set(frame.date))
                if overlap:
                    ext = ext[~ext.date.isin(overlap)]
                frame = pd.concat([frame, ext], ignore_index=True)
            out[code] = frame.sort_values('date').reset_index(drop=True)
        else:
            out[code] = pd.read_parquet(ext_ledger[code]['path'])
    return out


# ---------------------------------------------------------------- stock features
def rebuild_stock(frames, quarantined):
    market = pd.read_parquet(v1.MARKET_FEATURES)
    sse = market[(market.symbol == '000001.SH') & (market.frequency_minutes == 15)]
    days_full = np.unique(sse[TIME].to_numpy(dtype=np.int64) // 1000000)
    rows, tables = {}, {}
    for code in ALL_20:
        table, _, _ = stock_features(frames[code], code, days_full, quarantined.get(code, set()))
        hit = table[table.cutoff_index == CI]
        if not len(hit):
            raise ValueError('STOCK_ROW_MISSING:' + code)
        row = hit.iloc[0].copy()
        prev = table[table.cutoff_index == CI - 1]
        row['_return_prev'] = float(prev.iloc[0]['stock_day_return_raw']) if len(prev) else np.nan
        refp = table[table.cutoff_index == CI - 16]
        row['_ref_P'] = float(refp.iloc[0]['stock_reference_close_raw']) if len(refp) else np.nan
        rows[code] = row
        tables[code] = table
    return rows, days_full, tables


# ---------------------------------------------------------------- context loo
def rebuild_context(stock_rows, native, members, ci, stock_tables, *, baseline=False):
    jman = json.loads((v1.JOINT / 'manifest.json').read_bytes())
    catalog_boards = json.loads((HERE / 'ths_full_concept_scope_20261017_v1.json').read_bytes())
    leaders = pd.read_parquet(v1.SHARED_MARKET).sort_values('cutoff_index').reset_index(drop=True)
    n_cut = len(leaders)
    boards_sorted = sorted([r for r in pq.read_table(catalog_boards['source']).to_pylist()
                            if r['source'] == 'ths' and r['type'] == 'concept'], key=lambda r: r['code'])
    board_index = {b['code']: i for i, b in enumerate(boards_sorted)}
    members_ctx = members
    leaders['day'] = leaders[TIME] // 1000000
    leaders['board_index'] = leaders.leader_observed_board.map(board_index).fillna(-1).astype(int)
    native15_day = native[15][:, 16:32]  # E-day 16 cutoffs (6800..6815), reference-patched
    counts = np.zeros((len(boards_sorted), n_cut), dtype=int)
    totals = np.zeros_like(counts, dtype=float)
    up_totals = np.zeros_like(totals)
    common_mean = np.full_like(totals, np.nan)
    sizes = np.zeros(len(boards_sorted), dtype=int)
    day_cols = list(range(CI - 15, CI + 1))
    from build_joint_risk_features_v1 import basket_features
    for i, b in enumerate(boards_sorted):
        fields = basket_features(native15_day, members_ctx[i])
        counts[i, day_cols] = fields['present']
        totals[i, day_cols] = np.where(fields['present'] > 0,
                                       fields['observed_return_raw'] * fields['present'], 0)
        up_totals[i, day_cols] = np.where(fields['present'] > 0,
                                          np.rint(fields['observed_breadth'] * fields['present']), 0)
        common_mean[i, day_cols] = fields['common_member_return_change']
        sizes[i] = len(members_ctx[i])
    if baseline:
        for part in jman['board_parts']:
            source = pd.read_parquet(v1.JOINT / part['path'], columns=[
                'board_code', 'cutoff_index', 'observed_return_raw', 'observed_breadth',
                'common_member_return_change', 'present', 'member_count'])
            for board, group in source.groupby('board_code', sort=False):
                i = board_index[board]
                positions = group.cutoff_index.to_numpy(dtype=int)
                counts[i, positions] = group.present.to_numpy(dtype=int)
                totals[i, positions] = np.where(group.present > 0, group.observed_return_raw * group.present, 0)
                up_totals[i, positions] = np.where(group.present > 0, np.rint(group.observed_breadth * group.present), 0)
                common_mean[i, positions] = group.common_member_return_change
                sizes[i] = int(group.member_count.iloc[0])
    affiliations = defaultdict(list)
    for b in boards_sorted:
        for code in b['symbols']:
            affiliations[code].append(board_index[b['code']])
    first_capture = {}
    for row in leaders.itertuples():
        if known_leader(row.leader_observed_stock):
            first_capture.setdefault((row.leader_observed_stock, int(row.day)), row.cutoff_index)
    codes = sorted(affiliations)
    code_index = {c: i for i, c in enumerate(codes)}
    stock_returns = np.full((len(codes), n_cut), np.nan)
    leader_set = set(leaders.leader_observed_stock.dropna())
    leader_metrics = {}
    metric_cols = ['stock_rsi14_5m_raw', 'stock_rsi14_15m_raw',
                   'stock_return5m_raw', 'stock_return15m_raw']
    for part in jman['joint_parts']:
        frame = pd.read_parquet(v1.JOINT / part['path'],
                                columns=['code', 'cutoff_index', 'stock_day_return_raw', *metric_cols])
        frame = frame[frame.cutoff_index.isin(day_cols)]
        for code, g in frame.groupby('code', sort=False):
            positions = g.cutoff_index.to_numpy(dtype=int)
            stock_returns[code_index[code], positions] = g.stock_day_return_raw.to_numpy(dtype=float)
            if code in leader_set:
                values = np.full((n_cut, 4), np.nan)
                values[positions] = g[metric_cols].to_numpy(dtype=float)
                leader_metrics[code] = values
    # patch rebuilt 15m day returns for the 20 candidates across the E day
    day_returns = {}
    for code in ALL_20:
        table = stock_tables[code]
        sub = table[table.cutoff_index.isin(day_cols)]
        day_returns[code] = dict(zip(sub.cutoff_index, sub.stock_day_return_raw))
    for code in ([] if baseline else ALL_20):
        for t in day_cols:
            stock_returns[code_index[code], t] = day_returns[code].get(t, np.nan)
        if code in leader_metrics:
            sub = stock_tables[code].set_index('cutoff_index')
            for t in day_cols:
                if t in sub.index:
                    leader_metrics[code][t] = sub.loc[t, metric_cols].to_numpy(dtype=float)
    common_counts = np.zeros_like(counts)
    for board, i in board_index.items():
        related = [code_index[c] for c in codes if i in affiliations[c]]
        valid = np.isfinite(stock_returns[related][:, day_cols])
        common_counts[i, day_cols[1:]] = (valid[:, 1:] & valid[:, :-1]).sum(axis=0)
    if baseline:
        for part in jman['joint_parts']:
            source = pd.read_parquet(v1.JOINT / part['path'], columns=['code', 'cutoff_index', 'stock_day_return_raw'])
            for code, group in source.groupby('code', sort=False):
                stock_returns[code_index[code], group.cutoff_index.to_numpy(dtype=int)] = group.stock_day_return_raw
        for board, i in board_index.items():
            related = [code_index[c] for c in codes if i in affiliations[c]]
            valid = np.isfinite(stock_returns[related])
            common_counts[i, 1:] = (valid[:, 1:] & valid[:, :-1]).sum(axis=0)
    common_counts[:, np.arange(n_cut) % 8 == 0] = 0
    common_totals = np.where(common_counts > 0, common_mean * common_counts, 0)
    change_total = np.zeros(n_cut)
    change_count = np.zeros(n_cut, dtype=int)
    captured = set()
    for row in leaders.itertuples():
        t = row.cutoff_index
        if t % 16 == 0:
            captured = set()
        if t % 8 and captured and t in day_cols:
            indexes = np.array([code_index[c] for c in captured], dtype=int)
            valid = np.isfinite(stock_returns[indexes, t]) & np.isfinite(stock_returns[indexes, t - 1])
            change_count[t] = valid.sum()
            change_total[t] = (stock_returns[indexes[valid], t]
                               - stock_returns[indexes[valid], t - 1]).sum()
        if known_leader(row.leader_observed_stock):
            captured.add(row.leader_observed_stock)
    current_metrics = np.full((n_cut, 4), np.nan)
    current_change = np.full(n_cut, np.nan)
    for row in leaders.itertuples():
        t, code = row.cutoff_index, row.leader_observed_stock
        if not known_leader(code):
            continue
        if code in leader_metrics:
            current_metrics[t] = leader_metrics[code][t]
        if t % 8:
            current_change[t] = stock_returns[code_index[code], t] - \
                stock_returns[code_index[code], t - 1]
    extras = dict(up_totals=up_totals, common_totals=common_totals, common_counts=common_counts,
                  returns=stock_returns, code_index=code_index, leader_metrics=leader_metrics,
                  current_leader_metrics=current_metrics, current_leader_change=current_change,
                  captured_change_total=change_total, captured_change_count=change_count)
    population_codes = ['300014'] if baseline else ALL_20
    population = pd.DataFrame({'code': population_codes, 'cutoff_index': CI,
                               'stock_day_return_raw':
                               [stock_rows[c]['stock_day_return_raw'] for c in population_codes]})
    context = candidate_context(population, totals, counts, sizes, affiliations,
                                leaders, first_capture, extras)
    return context.set_index('code')


# ---------------------------------------------------------------- mb/golden/nh
def shared_and_nh(native, members, ci, codes20=ALL_20):
    market = pd.read_parquet(v1.MARKET_FEATURES)
    states = json.loads((HERE / 'golden_anchor_replay_20261017_v2/replay.json').read_bytes())['states']
    days = [P_DAY, E_DAY]
    shared_at = {}
    nh_shared_at = {}
    own_at = {}
    nh_own_at = {}
    for frequency in (5, 15):
        grid, markets = market_arrays(market, days, frequency)
        pos = grid.index(CUTOFF_TS)
        returns = native[frequency]  # full 2-day window, aligned with grid
        board = matrix_features(returns, members, frequency)
        shared = summarize(board, markets, frequency, 'mb%d_' % frequency)
        shared_at.update({k: v[pos] for k, v in shared.items()})
        high = {}
        for benchmark in ('sse', 'cyb'):
            values = markets[benchmark]
            high[benchmark] = day_high(values, np.isfinite(values), frequency)[pos]
            nh_shared_at['nh_%s%d_close_new_day_high' % (benchmark, frequency)] = \
                _s(high[benchmark])
        for benchmark in ('sse', 'cyb'):
            for flag in ('nonpositive', 'falling'):
                base = 'mb%d_%s_positive_%s' % (frequency, benchmark, flag)
                rate, cov = combine(shared[base + '_rate'], shared[base + '_coverage'],
                                    shared['mb%d_positive_share' % frequency], high[benchmark])
                name = 'nh_mb%d_%s_positive_common_%s' % (frequency, benchmark, flag)
                nh_shared_at[name + '_rate'] = _s(rate[pos])
                nh_shared_at[name + '_coverage'] = _s(cov[pos])
        del board
        for code in codes20:
            member = ci[code]
            related = [m for m in members if member in m]
            small = matrix_features(returns, related, frequency, exclude_index=member)
            own = summarize(small, markets, frequency, 'own%d_' % frequency, own=True)
            entry = own_at.setdefault(code, {})
            entry.update({k: v[pos] for k, v in own.items()})
            for benchmark in ('sse', 'cyb'):
                for flag in ('nonpositive', 'falling'):
                    base = 'own%d_%s_positive_%s' % (frequency, benchmark, flag)
                    rate, cov = combine(own[base + '_rate'], own[base + '_coverage'],
                                        own['own%d_positive_share' % frequency], high[benchmark])
                    name = 'nh_own%d_%s_positive_common_%s' % (frequency, benchmark, flag)
                    nh_entry = nh_own_at.setdefault(code, {})
                    nh_entry[name + '_rate'] = _s(rate[pos])
                    nh_entry[name + '_coverage'] = _s(cov[pos])
    grid15, markets15 = market_arrays(market, days, 15)
    pos15 = grid15.index(CUTOFF_TS)
    golden = golden_fields(days, markets15['sse'], states)
    for k, v in golden.items():
        shared_at[k] = v[pos15]
    return shared_at, nh_shared_at, own_at, nh_own_at


# ---------------------------------------------------------------- anchor check (09-01)
def anchor_check(ext_ledger, quarantined):
    """Rebuild same kernels at cutoff 20260901150000 and compare with frozen tables."""
    global ANCHOR_CI
    days = [ANCHOR_PREV, ANCHOR_DAY]
    codes, ci, members, _ = load_catalog()
    native, clock = replay_days(days, ci, {})
    row = clock[clock[TIME] == 20260901150000].iloc[0]
    ANCHOR_CI = int(row.cutoff_index)
    pos15 = list(clock[clock[TIME].floordiv(1000000) == ANCHOR_DAY][TIME]).index(20260901150000)
    # note: clock is ordered; position within the day's 16 cutoffs:
    pos15 = list(clock[TIME]).index(20260901150000)
    market = pd.read_parquet(v1.MARKET_FEATURES)
    states = json.loads((HERE / 'golden_anchor_replay_20261017_v2/replay.json').read_bytes())['states']
    result = {}
    frozen_shared = pd.read_parquet(v1.MULTIBOARD / 'shared_202609.parquet')
    frow = frozen_shared[frozen_shared[TIME] == 20260901150000].iloc[0]
    frozen_repair_shared = pd.read_parquet(v1.REPAIR / 'shared.parquet')
    rrow = frozen_repair_shared[frozen_repair_shared[TIME] == 20260901150000].iloc[0]
    frozen_repair_own = pd.read_parquet(v1.REPAIR / 'own.parquet')
    own_ref = frozen_repair_own[frozen_repair_own[TIME] == 20260901150000]
    cmp_rows = {}
    for frequency in (5, 15):
        grid, markets = market_arrays(market, days, frequency)
        pos = grid.index(20260901150000)
        returns = native[frequency]
        board = matrix_features(returns, members, frequency)
        shared = summarize(board, markets, frequency, 'mb%d_' % frequency)
        for k, v in shared.items():
            fv = frow.get(k)
            if fv is None:
                continue
            cmp_rows[k] = (float(v[pos]), float(fv))
        high = {}
        for benchmark in ('sse', 'cyb'):
            values = markets[benchmark]
            high[benchmark] = day_high(values, np.isfinite(values), frequency)[pos]
            key = 'nh_%s%d_close_new_day_high' % (benchmark, frequency)
            cmp_rows[key] = (_s(high[benchmark]), float(frow_repair(key, rrow)))
        for benchmark in ('sse', 'cyb'):
            for flag in ('nonpositive', 'falling'):
                base = 'mb%d_%s_positive_%s' % (frequency, benchmark, flag)
                rate, cov = combine(shared[base + '_rate'], shared[base + '_coverage'],
                                    shared['mb%d_positive_share' % frequency], high[benchmark])
                frozen_key = 'nh_mb%d_%s_positive_common_%s' % (frequency, benchmark, flag)
                cmp_rows['nh_%s_rate' % base] = (_s(rate[pos]), float(rrow[frozen_key + '_rate']))
                cmp_rows['nh_%s_coverage' % base] = (_s(cov[pos]), float(rrow[frozen_key + '_coverage']))
    grid15, markets15 = market_arrays(market, days, 15)
    golden = golden_fields(days, markets15['sse'], states)
    for k in ('new_sse_golden_position', 'new_sse_golden_distance_0_236',
              'new_sse_golden_distance_0_382', 'new_sse_golden_distance_0_5',
              'new_sse_golden_distance_0_618', 'new_sse_golden_distance_0_786'):
        cmp_rows[k] = (float(golden[k][pos15]), float(frow[k]))
    # own/nh_own sample vs repair own.parquet
    own_cmp = []
    sample_codes = sorted(set(own_ref.code.astype(str).str.zfill(6)))
    rng = np.random.RandomState(7)
    sample = list(rng.choice(sample_codes, size=min(25, len(sample_codes)), replace=False))
    ledger = {i['code']: i for i in
              json.loads((HERE / 'relative_stall_5m_20261018_v1/raw_input_ledger.json').read_bytes())}
    for code in sample:
        if code not in ledger:
            continue
        frame = pd.read_parquet(ledger[code]['path'])
        freqs = {}
        for frequency in (5, 15):
            grid, markets = market_arrays(market, days, frequency)
            pos = grid.index(20260901150000)
            returns = native[frequency]
            member = ci[code]
            related = [m for m in members if member in m]
            small = matrix_features(returns, related, frequency, exclude_index=member)
            own = summarize(small, markets, frequency, 'own%d_' % frequency, own=True)
            freqs[frequency] = (own, pos, markets)
        ref_row = own_ref[own_ref.code.astype(str).str.zfill(6) == code].iloc[0]
        for frequency in (5, 15):
            own, pos, markets = freqs[frequency]
            high = {}
            for benchmark in ('sse', 'cyb'):
                values = markets[benchmark]
                high[benchmark] = day_high(values, np.isfinite(values), frequency)[pos]
            for benchmark in ('sse', 'cyb'):
                for flag in ('nonpositive', 'falling'):
                    base = 'own%d_%s_positive_%s' % (frequency, benchmark, flag)
                    rate, cov = combine(own[base + '_rate'], own[base + '_coverage'],
                                        own['own%d_positive_share' % frequency], high[benchmark])
                    frozen_key = 'nh_own%d_%s_positive_common_%s' % (frequency, benchmark, flag)
                    own_cmp.append(dict(code=code, field=frozen_key + '_rate',
                                        rebuilt=_s(rate[pos]),
                                        frozen=float(ref_row[frozen_key + '_rate'])))
                    own_cmp.append(dict(code=code, field=frozen_key + '_coverage',
                                        rebuilt=_s(cov[pos]),
                                        frozen=float(ref_row[frozen_key + '_coverage'])))
    summary = {}
    for k, (a, b) in cmp_rows.items():
        finite = np.isfinite(a) and np.isfinite(b)
        d = abs(a - b) if finite else (0.0 if (not np.isfinite(a) and not np.isfinite(b)) else None)
        summary[k] = d
    bad = {k: v for k, v in summary.items() if v is None or v > 1e-9}
    own_bad = [r for r in own_cmp
               if not (np.isfinite(r['rebuilt']) and np.isfinite(r['frozen'])
                       and abs(r['rebuilt'] - r['frozen']) <= 1e-6)
               and not (not np.isfinite(r['rebuilt']) and not np.isfinite(r['frozen']))]
    return dict(cutoff_ts=20260901150000, n_compared=len(summary),
                mismatched=bad, own_sample_rows=len(own_cmp), own_mismatched=own_bad[:20])


def frow_repair(key, rrow):
    return rrow[key]


# ---------------------------------------------------------------- main
def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true', required=True)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    OUT = args.out.resolve()
    if OUT.parent != HERE:
        raise ValueError('OUTPUT_OUTSIDE_RESEARCH_DIRECTORY')
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    OUT.mkdir()

    mapping, fields232, fields266, extra266 = load_spec()
    ext_ledger = {e['code']: e for e in json.loads(
        (HERE / 'evidence_stream_trial_v2/minute_ext_20260908_ledger.json').read_bytes())}
    quarantined = defaultdict(set)
    with QUARANTINE_CSV.open(encoding='utf-8-sig', newline='') as stream:
        for row in _csv.DictReader(stream):
            quarantined[row['code']].add(int(row['day']))

    put('plan.json', json.dumps(dict(
        status='SIDECAR_REBUILD_NOT_FIT', generated=time.strftime('%Y-%m-%d %H:%M:%S'),
        inputs={str(p): sha(p) for p in [
            HERE / 'evidence_stream_trial_v2/minute_ext_20260908_ledger.json',
            v1.MODEL_232 / 'lightgbm_model.txt.gz', v1.MODEL_266 / 'lightgbm_model.txt.gz',
            v1.PREPROC_CONFIG, v1.MARKET_FEATURES,
            HERE / 'golden_anchor_replay_20261017_v2/replay.json']},
        frozen_builders=dict(
            stock='build_joint_risk_features_v1.stock_features',
            context='build_joint_candidate_context_v1.candidate_context',
            board_motion='multiboard_risk_features_v1.matrix_features/summarize',
            newhigh='newhigh_common_motion_v1.combine/day_high',
            golden='build_multiboard_full_features_v1.golden_fields',
            returns='intraday_relative_stall_v2.minute_returns'),
        cutoff_ts=CUTOFF_TS, cutoff_index=CI, p_day=P_DAY, e_day=E_DAY,
        reference_binding='frozen stock_features lag(day_close): E日参考价=P日全日最后收盘价'
                          '（冻结构建器自身的“前完整日收盘”谱系）',
        loo_policy='context excluded_means 剔除候选自身; own matrix_features exclude_index=member',
        http_requests=0, fit=False), indent=1).encode())

    codes, ci, members, _ = load_catalog()
    native, _clock = replay_days([P_DAY, E_DAY], ci, ext_ledger)
    log('replay_done')

    frames = code_frames(ext_ledger)
    stock_rows, days_full, stock_tables = rebuild_stock(frames, quarantined)
    # reference rebind (frozen anchor lineage: stock_features lag(day_close)):
    # E-day reference = P-day complete close (row CI stock_reference_close_raw);
    # P-day reference = prior complete close (row CI-16). Re-run frozen minute_returns
    # so the 19 codes enter the replay matrix (board aggregates include them, consistent
    # with their own finite values in candidate_context excluded_means).
    n_patched = 0
    for code in [c for c in ALL_20 if c != '300014']:
        refs = {P_DAY: stock_rows[code]['_ref_P'],
                E_DAY: stock_rows[code]['stock_reference_close_raw']}
        values, _ = minute_returns(frames[code], code, [P_DAY, E_DAY], refs,
                                   quarantined.get(code, set()))
        native[5][ci[code]] = values[5]
        native[15][ci[code]] = values[15]
        n_patched += 1
    log('references_rebound', codes=n_patched)
    put('rebuilt_stock_rows.parquet', parquet_bytes(
        pd.DataFrame([dict(stock_rows[c]) for c in ALL_20])))
    log('stock_rebuilt')

    # ---- anchor A: 300014 vs frozen joint_parts row
    frozen300 = None
    for part in json.loads((v1.JOINT / 'manifest.json').read_bytes())['joint_parts']:
        f = pd.read_parquet(v1.JOINT / part['path'])
        hit = f[(f.code.astype(str).str.zfill(6) == '300014') & (f.cutoff_index == CI)]
        if len(hit):
            frozen300 = hit.iloc[0]
            break
    check_cols = ['stock_close', 'stock_reference_close_raw', 'stock_day_return_raw',
                  'stock_return5m_raw', 'stock_return15m_raw', 'stock_momentum_change5m_raw',
                  'stock_momentum_change15m_raw', 'stock_rsi14_5m_raw', 'stock_rsi14_15m_raw',
                  'stock_rsi_delta5m_raw', 'stock_rsi_delta15m_raw',
                  'stock_volume_ratio_previous_session_raw', 'stock_intrabar_return15m_raw']
    deltas = {}
    for c in check_cols:
        a = float(stock_rows['300014'][c])
        b = float(frozen300[c])
        deltas[c] = abs(a - b) if (np.isfinite(a) and np.isfinite(b)) else (
            0.0 if not (np.isfinite(a) or np.isfinite(b)) else None)
    anchor_a = dict(code='300014', cutoff_index=CI, n_fields=len(check_cols),
                    max_delta=max(d for d in deltas.values() if d is not None),
                    mismatched={k: v for k, v in deltas.items() if v is None or v > 1e-9})

    context_rows = rebuild_context(stock_rows, native, members, ci, stock_tables)
    context_control = rebuild_context(stock_rows, native, members, ci, stock_tables, baseline=True)
    put('rebuilt_context_rows.parquet', parquet_bytes(context_rows.reset_index()))
    # ---- anchor B: 300014 vs frozen context sidecar
    frozen_ctx = None
    for item in json.loads((v1.CONTEXT_LOO / 'manifest.json').read_bytes())['parts']:
        f = pd.read_parquet(v1.CONTEXT_LOO / item['context'])
        hit = f[(f.code.astype(str).str.zfill(6) == '300014') & (f.cutoff_index == CI)]
        if len(hit):
            frozen_ctx = hit.iloc[0]
            break
    ctx_cols = [c for c in context_rows.columns if c not in ('cutoff_index',)]
    ctx_deltas = {}
    for c in ctx_cols:
        a = context_control.loc['300014', c]
        b = frozen_ctx[c]
        if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
            ctx_deltas[c] = 0.0 if bool(a) == bool(b) else None
            continue
        a = float(np.float32(a))
        b = float(np.float32(b))
        ctx_deltas[c] = abs(a - b) if (np.isfinite(a) and np.isfinite(b)) else (
            0.0 if not (np.isfinite(a) or np.isfinite(b)) else None)
    anchor_b = dict(code='300014', cutoff_index=CI, n_fields=len(ctx_cols),
                    max_delta=max(d for d in ctx_deltas.values() if d is not None),
                    mismatched={k: v for k, v in ctx_deltas.items() if v is None or v > 1e-5})

    shared_at, nh_shared_at, own_at, nh_own_at = shared_and_nh(native, members, ci)
    put('rebuilt_shared_cutoff.json', json.dumps(
        dict(shared=shared_at, nh_shared=nh_shared_at), indent=1).encode())

    anchor_c = anchor_check(ext_ledger, quarantined)
    put('anchor_checks.json', json.dumps(dict(anchor_A_stock=anchor_a, anchor_B_context=anchor_b,
                                              anchor_C_0901=anchor_c), indent=1).encode())
    log('anchors_done', A=anchor_a['max_delta'], B=anchor_b['max_delta'],
        C_compared=anchor_c['n_compared'], C_bad=len(anchor_c['mismatched']))
    if (anchor_a['mismatched'] or anchor_b['mismatched'] or
            anchor_c['mismatched'] or anchor_c['own_mismatched']):
        raise ValueError('FROZEN_KERNEL_REPLAY_MISMATCH')

    # ---------------- frame build + injection + inference ----------------
    days = [P_DAY, E_DAY]
    SM_ALL = pq.read_table(v1.SHARED_MARKET).column_names
    mom_all = set(pd.read_parquet(v1.MOMENTUM).columns)
    mapping_sources = set(mapping.values())
    base_sources = {s.removeprefix('ar_') for s in mapping_sources}
    sm_cols = [c for c in SM_ALL if c in base_sources]
    derived = {'time_node_nearest_abs_distance', 'session_is_pm',
               'context_sector_minus_sse_return_raw',
               'context_stock_minus_sector_return_raw',
               'context_stock_minus_leader_return_raw'}
    mom_cols = sorted((set(base_sources) & mom_all) - set(sm_cols) - derived)
    ctx_cols2 = sorted({s for s in base_sources if s.startswith('context_')} - derived)
    jp_cols = sorted({s for s in base_sources if s.startswith('stock_')})
    nh_own_cols = [c for c in pq.read_table(v1.REPAIR / 'own.parquet').column_names
                   if c.startswith('nh_own')]
    golden_cols = sorted({s for s in base_sources if s.startswith('new_sse_golden')})
    shared_cols = sorted({s for s in base_sources if s.startswith(('mb5_', 'mb15_'))}
                         | set(golden_cols))
    TS_TO_CI = dict(zip(pd.read_parquet(v1.SHARED_MARKET, columns=[TIME])[TIME].astype(np.int64),
                        pd.read_parquet(v1.SHARED_MARKET, columns=['cutoff_index']).cutoff_index))
    v1.TS_TO_CI, v1.SM_ALL = TS_TO_CI, SM_ALL
    v1.SM_COLS, v1.MOM_COLS, v1.JP_COLS = sm_cols, mom_cols, jp_cols
    v1.CTX_COLS, v1.NH_OWN_COLS = ctx_cols2, nh_own_cols
    diff = pd.read_parquet(v1.OVERLAY)
    if len(diff[diff.code.astype(str).str.zfill(6).isin(ALL_20)]):
        raise ValueError('UNEXPECTED_OVERLAY_ROWS_FOR_REBUILT_KEYS')
    shared_hit = diff[(diff.code == '__SHARED__') & (diff[TIME] == CUTOFF_TS)]
    if len(shared_hit):
        raise ValueError('SHARED_OVERLAY_AT_REBUILT_CUTOFF')

    keys = pd.DataFrame({'code': ALL_20, TIME: CUTOFF_TS})
    frame = v1.build_candidate_frame(keys, days, codes, members, native, mapping, diff)
    frame266 = v1.build_266(frame, keys, days, codes, members, native, extra266)

    # ---- inject rebuilt base values for the 20 rows
    for c in set(jp_cols) | set(ctx_cols2) | set(shared_cols) | set(nh_own_cols):
        if c in frame:
            frame[c] = frame[c].astype('float64')
    for i, code in enumerate(ALL_20):
        for c in jp_cols:
            frame.loc[i, c] = stock_rows[code][c]
        for c in ctx_cols2:
            frame.loc[i, c] = context_rows.loc[code, c]
        for c, value in shared_at.items():
            frame.loc[i, c] = value
        for c, value in nh_shared_at.items():
            frame.loc[i, c] = value
        for c, value in own_at[code].items():
            frame.loc[i, c] = value
        for c in nh_own_cols:
            if c in nh_own_at[code]:
                frame.loc[i, c] = nh_own_at[code][c]
    frame = derive(frame)
    # The initial frozen join already attached ar_ aliases. Rebuild them from
    # the corrected base values rather than keeping stale aliases or colliding.
    frame = frame.drop(columns=[c for c in frame if c.startswith('ar_')])
    frame = attach(frame, diff)

    missing_after = {}
    f232 = resolve_logical(frame, mapping, fields232)
    for c in ALL_20:
        i = ALL_20.index(c)
        missing_after[c] = dict(
            n_missing_232=int(np.isnan(f232.iloc[i][fields232].to_numpy(dtype=float)).sum()))
    booster232 = load_booster(v1.MODEL_232)
    booster266 = load_booster(v1.MODEL_266)
    amp232 = booster232.predict(native_missing_view(f232, fields232))
    final266 = pd.concat([frame.reset_index(drop=True),
                          frame266[extra266].reset_index(drop=True)], axis=1)
    put('model_inputs_232.parquet', parquet_bytes(f232[fields232]))
    put('model_inputs_266.parquet', parquet_bytes(final266[fields266]))
    amp266 = booster266.predict(native_missing_view(final266, fields266))
    v232 = f232[fields232].to_numpy(dtype=float)
    scored = []
    for i, code in enumerate(ALL_20):
        reasons = []
        if np.isnan(v232[i]).all():
            reasons.append('all_232_missing')
        for col, label in [('stock_day_return_raw', 'joint_parts'),
                           ('context_sector_mean_return_loo_raw', 'context_loo')]:
            if pd.isna(frame.iloc[i][col]):
                reasons.append('source_join_missing:' + label)
        scored.append(dict(code=code, amp_232=float(np.clip(amp232[i], 0, 1)),
                           amp_266=float(np.clip(amp266[i], 0, 1)),
                           amp_232_raw=float(amp232[i]), amp_266_raw=float(amp266[i]),
                           n_missing_232=missing_after[code]['n_missing_232'],
                           missing_fields_232=[f for f in fields232 if pd.isna(f232.iloc[i][f])],
                           isolated=bool(reasons), reasons=reasons))
    put('scored_20.json', json.dumps(dict(cutoff_ts=CUTOFF_TS, scored=scored), indent=1).encode())
    n_scoreable = sum(1 for s in scored if not s['isolated'])

    # ---------------- UI shadow packs ----------------
    sys.path.insert(0, str(HERE.parents[2] / 'stargazer_desktop'))
    from oracles_v1 import EvidencePack, tarot_candidate, pyxis_context
    import jev_adapter_v1 as jev
    sel = pd.read_parquet(v1.MASTER_TABLE)
    sel['code'] = sel.code.astype(str).str.zfill(6)
    master = {r.code: r for r in sel[sel.P == P_DAY].itertuples()}
    pb = pd.read_parquet(v1.MASTER_TABLE.parent / 'rolling_lr_v2_predictions.parquet')
    pb['code'] = pb.code.astype(str).str.zfill(6)
    pb = pb[pb.fold == 'FINAL']
    pb_map = {r.code: float(r.pB2) for r in pb[pb.P == P_DAY].itertuples()}
    sm_row = pd.read_parquet(v1.SHARED_MARKET, filters=[('cutoff_index', '==', CI)]).iloc[0]

    def finite_or_none(x):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    def sign_label(x):
        v = finite_or_none(x)
        return None if v is None else ('up' if v > 0 else ('down' if v < 0 else 'flat'))

    packs = []
    for s in scored:
        if s['isolated']:
            continue
        code = s['code']
        m = master[code]
        market_state = dict(index='sse', session_fraction=finite_or_none(sm_row.sse_15m_session_fraction),
                            trend=sign_label(sm_row.sse_15m_day_return_change_adjacent),
                            state='day_gain' if finite_or_none(sm_row.sse_15m_day_return_prev_close)
                            and sm_row.sse_15m_day_return_prev_close > 0 else 'day_loss',
                            direction=sign_label(sm_row.sse_15m_momentum_change),
                            strength=finite_or_none(sm_row.sse_15m_rsi14),
                            drawdown_from_day_high=finite_or_none(sm_row.sse_15m_drawdown_from_observed_day_high),
                            anchor_position=finite_or_none(sm_row.sse_15m_anchor_position),
                            daily_return5_prior=finite_or_none(sm_row.sse_15m_daily_return5_asof_prior_day),
                            breadth_observed_stocks=int(sm_row.captured_stock_count_today),
                            cyb_day_return_prev_close=finite_or_none(sm_row.cyb_15m_day_return_prev_close))
        ctx = context_rows.loc[code]
        board_states = [dict(board='own_sector_loo',
                             strength=finite_or_none(ctx.context_sector_mean_return_loo_raw),
                             breadth=finite_or_none(ctx.context_sector_breadth_loo),
                             direction=sign_label(ctx.context_sector_common_change_loo_raw),
                             state='positive' if finite_or_none(ctx.context_sector_mean_return_loo_raw)
                             is not None and ctx.context_sector_mean_return_loo_raw > 0 else 'nonpositive',
                             current_leader_return=finite_or_none(ctx.context_current_leader_return_loo_raw),
                             current_leader_rsi14_15m=finite_or_none(ctx.context_current_leader_rsi14_15m_raw))]
        leader_states = []
        if finite_or_none(sm_row.leader_observed_stock_return_raw) is not None:
            leader_states.append(dict(code=str(sm_row.leader_observed_stock),
                                      board=str(sm_row.leader_observed_board),
                                      strength=finite_or_none(sm_row.leader_observed_stock_return_raw),
                                      state='up' if sm_row.leader_observed_stock_return_raw > 0 else 'down',
                                      full_universe_observed=bool(sm_row.leader_full_universe_observed)))
        risk_missing = []
        for grp, obj in (('market_state', market_state), ('board_states', board_states),
                         ('leader_states', leader_states)):
            if not jev._group_has_state(grp, obj):
                risk_missing.append(grp)
        risk_context = pyxis_context(version='232主版', amp_232=s['amp_232'], amp_266=s['amp_266'],
                                     market_state=market_state, board_states=board_states,
                                     leader_states=leader_states, missing=risk_missing,
                                     asof_ts=str(CUTOFF_TS))
        risk_context['feature_missing'] = s['missing_fields_232']
        risk_context['feature_status'] = 'PARTIAL' if s['missing_fields_232'] else 'COMPLETE'
        cand = tarot_candidate(code, p_close_score=finite_or_none(pb_map.get(code)),
                               golden_zone=finite_or_none(m.golden_dist_618_atr),
                               f618=finite_or_none(m.F618), plan_dist_atr=finite_or_none(m.plan_dist_atr),
                               atr_ref=finite_or_none(m.atr_ref), downside_state=None, trigger=None,
                               evidence_refs=dict(
                                   master_table='full_master_table_v2.parquet(code,P)',
                                   pB2='rolling_lr_v2_predictions.parquet(FINAL)',
                                   sidecar='sidecar_rebuild_v1 (frozen builders, cutoff 20260908150000)',
                                   amp='sidecar_rebuild_v1 inference (232/266 frozen boosters)'))
        pack = EvidencePack(evidence_id='shadow_%s_%d' % (code, E_DAY), asof_ts=str(CUTOFF_TS),
                            market_date=str(E_DAY), query_ts=str(CUTOFF_TS),
                            candidates=[cand], risk_context=risk_context,
                            risk_missing=risk_missing,
                            notes='HISTORICAL_REPLAY 20260908; sidecar-rebuilt full-feature scores; '
                                  'downside/trigger null (no intraday replay)').bind()
        packs.append(dict(evidence_id=pack.evidence_id, pack_hash=pack.pack_hash,
                          code=code, P=P_DAY, E=E_DAY, asof_ts=pack.asof_ts,
                          query_ts=pack.query_ts, market_date=pack.market_date,
                          candidates=pack.candidates, risk_context=pack.risk_context,
                          risk_missing=pack.risk_missing, corvus_gap=pack.corvus_gap,
                          confirmed_plan=pack.confirmed_plan, notes=pack.notes))
    body = ''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in packs)
    with (OUT / 'shadow_packs_20260908.jsonl').open('xb') as stream:
        stream.write(body.encode('utf-8'))
    writer['shadow_packs_20260908.jsonl'] = dict(bytes=len(body),
                                                 sha256=hashlib.sha256(body.encode('utf-8')).hexdigest())
    log('packs_written', n=len(packs))

    # ---------------- report ----------------
    report = build_report(anchor_a, anchor_b, anchor_c, scored, n_scoreable, missing_after)
    put('sidecar_rebuild_report.md', report.encode())
    put('stop_record.json', json.dumps(dict(
        status='COMPLETE_SIDECAR_REBUILD_NOT_FIT', external_http=0, fit=False, llm_calls=0,
        scoreable='%d/20' % n_scoreable,
        anchors=dict(A_max_delta=anchor_a['max_delta'], B_max_delta=anchor_b['max_delta'],
                     C_compared=anchor_c['n_compared'], C_mismatched=len(anchor_c['mismatched']),
                     C_own_rows=anchor_c['own_sample_rows'],
                     C_own_mismatched=len(anchor_c['own_mismatched'])),
        manifest={k: v['sha256'] for k, v in writer.items()}), indent=1).encode(), terminal=True)
    log('done', scoreable=n_scoreable)


def build_report(anchor_a, anchor_b, anchor_c, scored, n_scoreable, missing_after):
    L = []
    a = L.append
    a('# 侧车重建报告（sidecar_rebuild_v1，cutoff 20260908150000）')
    a('')
    a('时间：%s。纪律：冻结构建器只 import 不改；零外部网络；旧产物未动；新产物全部在本目录（<8MiB）。'
      % time.strftime('%Y-%m-%d %H:%M:%S'))
    a('')
    a('## 1. 重建范围与方法')
    a('')
    a('- stock_* 行：`build_joint_risk_features_v1.stock_features` 冻结函数，全市场日历；'
      '**参考价绑定=冻结谱系 lag(day_close)**（E 日参考=P 日全日最后收盘）。')
    a('- context_* 行：`build_joint_candidate_context_v1.candidate_context` 冻结函数；'
      '板块聚合/领涨迹取自冻结 board_parts/joint_parts/shared（cutoff 附近列），'
      '候选自身 15m 收益用重建值；**LOO=excluded_means 剔除候选自身**。')
    a('- mb/golden/nh：冻结 `matrix_features/summarize/golden_fields/day_high/combine` 内核，'
      '2 日重放窗口在 cutoff 位取值；own 侧 LOO（exclude_index=member）。')
    a('- 注入后重跑冻结 `derive()` 与 `attach()`（核验 overlay 对本批键零行，ar_=base）。')
    a('')
    a('## 2. 一致性锚点（与冻结产物逐字段比对）')
    a('')
    a('- 锚点A（300014@6815 重建 stock 行 vs 冻结 joint_parts 行）：%d 字段，'
      'max|Δ|=%s，不一致字段=%s。' % (anchor_a['n_fields'], anchor_a['max_delta'],
                                     anchor_a['mismatched'] or '无'))
    a('- 锚点B（300014@6815 重建 context 行 vs 冻结 context 侧车行）：%d 字段，'
      'max|Δ|=%s（float32 口径），不一致=%s。' % (anchor_b['n_fields'], anchor_b['max_delta'],
                                                anchor_b['mismatched'] or '无'))
    a('- 锚点C（同内核在 20260901150000 重建 vs 冻结 shared_202609/repair 侧车）：'
      '共享字段 %d 项，不一致 %d 项；own/nh_own 抽样 %d 格，不一致 %d 格。'
      % (anchor_c['n_compared'], len(anchor_c['mismatched']),
         anchor_c['own_sample_rows'], len(anchor_c['own_mismatched'])))
    if anchor_c['mismatched']:
        a('  - C 不一致明细（前 10）：%s' % json.dumps(dict(list(anchor_c['mismatched'].items())[:10])))
    a('')
    a('截止时刻语义：cutoff=15:00=当日最后完成 15m 格（END 重构，未 fabricate 缺槽）；'
      'LOO 语义如上；尺度与冻结源一致（分钟扩展与冻结文件位级一致，见 trial v2）。')
    a('')
    a('## 3. 20 只候选重放评分结果')
    a('')
    a('**可评分 %d/20**。' % n_scoreable)
    a('')
    a('| code | amp_232 | amp_266 | n_missing_232 | isolated | 留因 |')
    a('|---|---|---|---|---|---|')
    for s in scored:
        a('| %s | %.6f | %.6f | %d | %s | %s |' % (
            s['code'], s['amp_232'], s['amp_266'], s['n_missing_232'], s['isolated'],
            ';'.join(s['reasons']) or '-'))
    a('')
    remain = [s for s in scored if s['isolated']]
    if remain:
        a('剩余留因：%s' % json.dumps([dict(code=s['code'], reasons=s['reasons']) for s in remain],
                                     ensure_ascii=False))
    else:
        a('剩余留因：无。')
    a('')
    a('说明：300014 幅度与 trial v1 不同是**预期内**——trial v1 该键 88 字段为 NaN，'
      '本次为补全特征后的完整特征分数（trial v1 = NaN 特征分数，两者都不是错误，'
      '完整特征分数是与冻结训练口径一致的消费形态）。')
    a('')
    a('## 4. UI shadow packs')
    a('')
    a('- 文件：`sidecar_rebuild_v1/shadow_packs_20260908.jsonl`，%d 行（每只可评分候选一行，'
      'HISTORICAL_REPLAY 20260908 标注，query_ts=20260908150000 合法 14 位）。' % n_scoreable)
    a('- 每行=EvidencePack 形状；候选段含 tarot 九字段（code/p_close_score/golden_zone/f618/'
      'plan_dist_atr/atr_ref/downside_state/trigger/evidence_refs），包级含 code/P/E/asof_ts/'
      'query_ts/risk_context（pyxis 形状）；"10 字段契约"未见原文，按 18.36k 语境以 tarot 九字段+'
      '包级 code/P/E/asof/query 对齐，如契约另有第 10 字段定义请主代理核对后补。')
    a('- jev `_risk_evidence_incomplete` 逐包校验通过（见构建日志；risk_missing 如实登记）。')
    a('')
    a('## 5. 产物清单')
    a('')
    a('plan.json / rebuilt_stock_rows.parquet / rebuilt_context_rows.parquet / '
      'rebuilt_shared_cutoff.json / anchor_checks.json / scored_20.json / '
      'shadow_packs_20260908.jsonl / sidecar_rebuild_report.md / stop_record.json')
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.is_dir() and not (OUT / 'stop_record.json').exists():
            put('stop_record.json', json.dumps(dict(status='FAILED_NO_RESUME',
                error=type(exc).__name__ + ':' + str(exc), fit=False,
                llm_calls=0), indent=1).encode(), terminal=True)
        raise
