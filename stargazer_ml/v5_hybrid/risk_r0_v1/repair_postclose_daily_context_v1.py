"""Daily-close-only context sidecar. Old minute completeness and fit stay frozen."""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from build_joint_risk_features_v1 import wilder_rsi
from finalize_postclose_cohort_v3 import read_population, corrected_masks
from inventory_ths_stockdb_cache_v1 import HERE, digest
from research_budget_v2 import TOTAL_CAP, bytes_used

SOURCE = HERE / 'joint_risk_features_20261017_v3'
OUT = HERE / 'postclose_daily_context_repair_20261017_v1'
CAP = 16 * 1024**2


def closing_panel(frame, day_count):
    if frame[['code', 'cutoff_index']].duplicated().any():
        raise ValueError('DUPLICATE_SOURCE_KEY')
    if not frame.cutoff_index.between(0, day_count * 16 - 1).all():
        raise ValueError('SOURCE_DAY_OUT_OF_BOUNDS')
    result = {}
    final = frame[frame.cutoff_index.mod(16).eq(15)]
    for code, group in final.groupby('code', sort=False):
        values = group.stock_close.to_numpy(dtype=float)
        valid = np.isfinite(values) & (values > 0)
        closes = np.full(day_count, np.nan)
        closes[group.cutoff_index.to_numpy(dtype=int)[valid] // 16] = values[valid]
        result[code] = closes
    return result


def daily_returns(closes):
    result = np.full_like(closes, np.nan, dtype=float)
    valid = np.isfinite(closes[:, 1:]) & np.isfinite(closes[:, :-1])
    valid &= (closes[:, 1:] > 0) & (closes[:, :-1] > 0)
    np.divide(closes[:, 1:], closes[:, :-1], out=result[:, 1:], where=valid)
    result[:, 1:] -= 1
    return result


def board_panel(returns, members):
    means, counts, ups = [], [], []
    for indexes in members:
        values = returns[indexes]
        valid = np.isfinite(values)
        n = valid.sum(axis=0)
        total = np.where(valid, values, 0).sum(axis=0)
        means.append(np.divide(total, n, out=np.full(len(n), np.nan), where=n > 0))
        counts.append(n)
        ups.append(((values > 0) & valid).sum(axis=0))
    return np.asarray(means), np.asarray(counts), np.asarray(ups)


def unique_leaders(returns, board_returns, members, codes):
    boards = np.full(returns.shape[1], -1, dtype=int)
    stocks = np.full_like(boards, -1)
    for d in range(len(boards)):
        values = board_returns[:, d]
        if not np.isfinite(values).any():
            continue
        winners = np.flatnonzero(values == np.nanmax(values))
        if len(winners) != 1:
            continue
        b = int(winners[0]); boards[d] = b
        allowed = [i for i in members[b] if np.isfinite(returns[i, d]) and
                   codes[i].startswith(('000','001','002','003','300','301','600','601','603','605','688','920','83'))]
        if allowed:
            best = max(returns[i, d] for i in allowed)
            winners = [i for i in allowed if returns[i, d] == best]
            if len(winners) == 1:
                stocks[d] = winners[0]
    return boards, stocks


def excluded_board(mean, count, up, own):
    valid = np.isfinite(own)
    remaining = count - valid.astype(int)
    if (remaining < 0).any():
        raise ValueError('NEGATIVE_REMAINING_COUNT')
    total = np.where(count > 0, mean * count, 0) - np.where(valid, own, 0)
    reduced = np.divide(total, remaining, out=np.full_like(total, np.nan), where=remaining > 0)
    breadth = np.divide(up - (valid & (own > 0)).astype(int), remaining,
                        out=np.full_like(total, np.nan), where=remaining > 0)
    return reduced, breadth, remaining


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    population, frozen = read_population()
    source = json.loads((SOURCE / 'manifest.json').read_bytes())
    refs = {str(SOURCE / 'manifest.json'): digest((SOURCE / 'manifest.json').read_bytes()),
            frozen['population']['path']: frozen['population']['sha256'],
            str(Path(__file__)): digest(Path(__file__).read_bytes())}
    def bound(path, sha):
        body = path.read_bytes()
        if digest(body) != sha:
            raise ValueError('INPUT_HASH_DRIFT:' + str(path))
        refs[str(path)] = sha
        return body
    catalog_path = HERE / 'ths_full_concept_scope_20261017_v1.json'
    catalog = json.loads(bound(catalog_path, source['inputs'][str(catalog_path)]))
    membership_path = Path(catalog['source'])
    membership = pq.read_table(io.BytesIO(bound(membership_path, source['inputs'][str(membership_path)]))).to_pylist()
    membership = sorted([b for b in membership if b['source'] == 'ths' and b['type'] == 'concept'], key=lambda b: b['code'])
    if [b['code'] for b in membership] != [b['code'] for b in catalog['boards']]:
        raise ValueError('BOARD_UNIVERSE_DRIFT')
    shared = pd.read_parquet(io.BytesIO(bound(SOURCE / 'shared_market_and_leaders.parquet',
        source['output_hashes']['shared_market_and_leaders.parquet']['sha256'])), columns=['cutoff_index', 'decision_ts_assumed_end'])
    days = np.sort(population.day.unique())
    closing_times = shared[shared.cutoff_index.mod(16).eq(15)]
    if not np.array_equal(closing_times.decision_ts_assumed_end.to_numpy(), days * 1000000 + 150000):
        raise ValueError('DAILY_CLOSE_TIME_GRID_DRIFT')
    codes = sorted({code for b in membership for code in b['symbols']})
    code_index = {c: i for i, c in enumerate(codes)}
    closes = np.full((len(codes), len(days)), np.nan)
    seen = set(); previous_return_counts = np.zeros(len(days), dtype=int)
    for index, part in enumerate(source['joint_parts']):
        body = bound(SOURCE / part['path'], source['output_hashes'][part['path']]['sha256'])
        frame = pq.read_table(io.BytesIO(body), columns=['code','cutoff_index','stock_close','stock_day_return_raw']).to_pandas()
        part_codes = set(frame.code)
        if seen & part_codes:
            raise ValueError('CODE_SPANS_SOURCE_PARTS')
        seen |= part_codes
        for code, values in closing_panel(frame, len(days)).items():
            closes[code_index[code]] = values
        final = frame[frame.cutoff_index.mod(16).eq(15) & frame.stock_day_return_raw.notna()]
        previous_return_counts += np.bincount(final.cutoff_index.to_numpy(dtype=int) // 16, minlength=len(days))
        if index % 15 == 0:
            print(json.dumps(dict(stage='daily_close_replay', parts=index+1, codes=len(seen))), flush=True)
    returns = daily_returns(closes)
    members = [np.asarray([code_index[c] for c in b['symbols']], dtype=int) for b in membership]
    board_returns, counts, ups = board_panel(returns, members)
    boards, leaders = unique_leaders(returns, board_returns, members, codes)
    rsi = np.asarray([wilder_rsi(c) for c in closes])
    affiliations = {c: [] for c in population.code.unique()}
    for b, item in enumerate(membership):
        for code in set(item['symbols']) & affiliations.keys():
            affiliations[code].append(b)
    rows = []
    for code, group in population.groupby('code', sort=False):
        s = code_index[code]
        actual = group.set_index('day').reindex(days)
        reference = actual.reference_close_raw.to_numpy(dtype=float)
        aligned = np.isfinite(closes[s]) & np.isfinite(reference) & (np.abs(closes[s] - reference) <= .01 + 1e-12)
        both_aligned = aligned & np.r_[False, aligned[:-1]]
        related = np.asarray(affiliations[code], dtype=int)
        reduced, breadth, remaining = excluded_board(board_returns[related], counts[related], ups[related], returns[s][None, :])
        sizes = np.asarray([len(members[b]) - 1 for b in related])[:, None]
        coverage = np.divide(remaining, sizes, out=np.full_like(reduced, np.nan), where=sizes > 0)
        for d, day in enumerate(days):
            row = dict(code=code, day=int(day), daily_reference_pair_aligned=bool(both_aligned[d]),
                       observed_leader_board=membership[boards[d]]['code'] if boards[d] >= 0 else None,
                       observed_leader_stock=codes[leaders[d]] if leaders[d] >= 0 else None,
                       context_sector_mean_return_loo_raw=None, context_sector_breadth_loo=None,
                       context_sector_mean_coverage_loo=None, context_leader_board_return_loo_raw=None,
                       context_current_leader_return_loo_raw=None, context_current_leader_daily_rsi14_raw=None,
                       formal_full_universe_leader=False, intraday_capture_reconstructed=False)
            if both_aligned[d]:
                for name, values in [('context_sector_mean_return_loo_raw', reduced[:, d]),
                                     ('context_sector_breadth_loo', breadth[:, d]),
                                     ('context_sector_mean_coverage_loo', coverage[:, d])]:
                    row[name] = float(np.nanmean(values)) if np.isfinite(values).any() else None
                b, leader = boards[d], leaders[d]
                if b >= 0:
                    if b in related:
                        value = reduced[np.flatnonzero(related == b)[0], d]
                    else:
                        value = board_returns[b, d]
                    row['context_leader_board_return_loo_raw'] = float(value) if np.isfinite(value) else None
                if leader >= 0 and leader != s:
                    row['context_current_leader_return_loo_raw'] = float(returns[leader, d])
                    row['context_current_leader_daily_rsi14_raw'] = float(rsi[leader, d]) if np.isfinite(rsi[leader, d]) else None
            rows.append(row)
    overlay = pd.DataFrame(rows).set_index(['code','day']).reindex(pd.MultiIndex.from_frame(population[['code','day']])).reset_index()
    repaired = population.copy()
    replacements = [c for c in overlay if c.startswith('context_')]
    for column in replacements:
        repaired[column] = overlay[column].to_numpy()
    repaired['context_stock_minus_sector_daily_return'] = repaired.stock_return1d - repaired.context_sector_mean_return_loo_raw
    repaired['context_stock_minus_leader_daily_return'] = repaired.stock_return1d - repaired.context_current_leader_return_loo_raw
    masks = corrected_masks(repaired)
    eligible = repaired.label_state.eq('CONDITIONAL_NEXT_SESSION_COMPLETE') & ~repaired.proposed_purged & np.logical_and.reduce(list(masks.values()))
    monthly = []
    for month, indexes in population.groupby(population.day.astype(str).str[:6]).groups.items():
        g = repaired.loc[indexes]
        monthly.append(dict(month=month, rows=len(g), previous_cohort=int(population.loc[indexes].research_cohort_eligible.sum()),
            repaired_cohort=int(eligible.loc[indexes].sum()),
            sector_available=int(g.context_sector_mean_return_loo_raw.notna().sum()),
            leader_available=int(masks['leader_daily_observation'].loc[indexes].sum()),
            previous_close_return_stocks=int(previous_return_counts[np.isin(days, g.day.unique())].sum()),
            daily_close_return_stocks=int(np.isfinite(returns[:, np.isin(days, g.day.unique())]).sum())))
    overlay['proposed_repaired_cohort_eligible'] = eligible.to_numpy()
    evidence = dict(status='DAILY_CLOSE_CONTEXT_REBUILT_NOT_FIT_AUTHORIZED', inputs=refs,
        fields_preserved=frozen['fields'], source_codes=len(seen), board_count=len(membership),
        population_rows=len(population), monthly=monthly,
        old_cohort=int(population.research_cohort_eligible.sum()), proposed_cohort=int(eligible.sum()),
        assumptions=['archived snapshot membership; partial raw context, not corporate-action certified',
                     '15:00 close sufficient for daily context, not minute-bar completeness certification',
                     'global observed rank includes candidate; context arithmetic excludes its contribution; self leader stays NULL',
                     'no new intraday captured leaders invented; old captured fields retained'],
        fit_run=False, old_labels_unchanged=True, original_training_outputs_unchanged=True)
    sink = io.BytesIO(); overlay.to_parquet(sink, index=False, compression='zstd')
    payloads = {'daily_context_overlay.parquet': sink.getvalue()}
    evidence['outputs'] = {k: dict(bytes=len(v), sha256=digest(v)) for k, v in payloads.items()}
    payloads['manifest.json'] = json.dumps(evidence, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    if sum(map(len, payloads.values())) + 1024**2 > CAP or bytes_used() + sum(map(len, payloads.values())) + 1024**2 > TOTAL_CAP:
        raise ValueError('REPAIR_BUDGET_STOP')
    for path, sha in refs.items():
        if digest(Path(path).read_bytes()) != sha:
            raise ValueError('INPUT_CHANGED_DURING_REPLAY:' + path)
    OUT.mkdir(exist_ok=False)
    for name, body in payloads.items():
        with (OUT / name).open('xb') as stream:
            stream.write(body)
    print(json.dumps({k: v for k, v in evidence.items() if k not in ['inputs','fields_preserved']}, ensure_ascii=False))
    return evidence


if __name__ == '__main__':
    run()
