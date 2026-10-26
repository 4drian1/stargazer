"""Same engineering days/snapshot; native 1m -> real 5m, never interpolate 15m."""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from intraday_relative_stall_v1 import TIME, leader_watch, slots
from intraday_relative_stall_v2 import basket, board_evidence, minute_returns, paths
from inventory_ths_stockdb_cache_v1 import ROOTS
from materialize_relative_stall_pilot_v1 import HERE, DAYS, SOURCE, sha, parquet_body
from research_budget_v2 import bytes_used, TOTAL_CAP

OUT = HERE / 'relative_stall_5m_20261018_v1'
CAP, RESERVE = 24 * 1024**2, 256 * 1024


def semantic_equal(left, right, columns, tolerance=1e-12):
    """Exact NULL masks; permit only storage dtype variation, no filled unknowns."""
    for column in columns:
        a, b = left[column].reset_index(drop=True), right[column].reset_index(drop=True)
        pd.testing.assert_series_equal(a.isna(), b.isna())
        pd.testing.assert_series_equal(a[a.notna()].reset_index(drop=True), b[b.notna()].reset_index(drop=True),
            check_dtype=False, check_exact=False, rtol=0, atol=tolerance)


def materialize():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if bytes_used() + CAP > TOTAL_CAP:
        raise ValueError('REGISTERED_BUDGET_STOP')
    OUT.mkdir(exist_ok=False)
    used, outputs, inputs = 0, {}, {}

    def encoded(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')

    def write(name, body, terminal=False):
        nonlocal used
        reserve = 0 if terminal else RESERVE
        if used + len(body) + reserve > CAP or bytes_used() + len(body) + reserve > TOTAL_CAP:
            raise ValueError('SERIALIZED_BUDGET_STOP')
        with (OUT / name).open('xb') as stream:
            stream.write(body)
        used += len(body)
        outputs[name] = dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())

    def bind(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('INPUT_HASH_MISMATCH:' + str(path))
        inputs[str(path.resolve())] = digest
        return path

    try:
        old_pilot = HERE / 'relative_stall_pilot_20261018_v1'
        for p in [Path(__file__), HERE / 'intraday_relative_stall_v2.py',
                  HERE / 'test_intraday_relative_stall_v2.py', HERE / 'intraday_relative_stall_v1.py',
                  SOURCE / 'manifest.json', old_pilot / 'report.json', old_pilot / 'stop_record.json']:
            bind(p)
        if json.loads((old_pilot / 'stop_record.json').read_bytes())['status'] != 'COMPLETE':
            raise ValueError('ORIGINAL_PILOT_NOT_COMPLETE')
        previous = json.loads((old_pilot / 'report.json').read_bytes())
        model = bind(HERE / 'postclose_repaired_research_fit_20261017_v1/model.txt')
        write('plan.json', encoded(dict(days=DAYS, boards=978, frequency_minutes=5,
            original_15m_preserved=True, maximum_bytes=CAP, http_requests=0, fit=False,
            daily_reference='EXISTING_FROZEN_STOCK_REFERENCE_CLOSE_RAW', model_sha256=inputs[str(model.resolve())])))
        manifest = json.loads((SOURCE / 'manifest.json').read_bytes())
        shared = bind(SOURCE / 'shared_market_and_leaders.parquet',
                      manifest['output_hashes']['shared_market_and_leaders.parquet']['sha256'])
        calendar = pq.read_table(shared, columns=['cutoff_index', TIME]).to_pandas()
        selected = calendar[calendar[TIME].floordiv(1000000).isin(DAYS)].copy()
        cutoffs = [t for day in DAYS for t in slots(day, 5)]
        selected_indexes = selected.cutoff_index.tolist()
        catalog_path = bind(HERE / 'ths_full_concept_scope_20261017_v1.json')
        catalog = json.loads(catalog_path.read_bytes())
        snapshot = bind(Path(catalog['source']), catalog['source_sha256'])
        boards = sorted([r for r in pq.read_table(snapshot).to_pylist()
                         if r['source'] == 'ths' and r['type'] == 'concept'], key=lambda r: r['code'])
        codes = sorted({c for b in boards for c in b['symbols']})
        code_index = {c: i for i, c in enumerate(codes)}
        old_chunks = []
        for item in manifest['joint_parts']:
            path = bind(SOURCE / item['path'], manifest['output_hashes'][item['path']]['sha256'])
            old_chunks.append(pq.read_table(path, columns=['code', 'cutoff_index', 'stock_day_return_raw',
                'stock_reference_close_raw', 'source_cache'], filters=[('cutoff_index', 'in', selected_indexes)]).to_pandas())
        old = pd.concat(old_chunks, ignore_index=True).merge(selected, on='cutoff_index', validate='many_to_one')
        old['day'] = old[TIME] // 1000000
        if old.duplicated(['code', TIME]).any():
            raise ValueError('DUPLICATE_OLD_STOCK_KEY')
        grouped = old.groupby(['code', 'day']).stock_reference_close_raw
        if (grouped.nunique() > 1).any():
            raise ValueError('DAILY_REFERENCE_CONFLICT')
        references = grouped.first().to_dict()
        old_sources = old.groupby('code').source_cache
        if (old_sources.nunique() != 1).any():
            raise ValueError('OLD_SOURCE_PRIORITY_CONFLICT')
        inventory_path = bind(HERE / 'ths_stockdb_cache_inventory_20261017_v1.json',
                              manifest['inputs'][str(HERE / 'ths_stockdb_cache_inventory_20261017_v1.json')])
        inventory = json.loads(inventory_path.read_bytes())
        chosen = {}
        for priority in ('old', 'car', 'ext', 'clean'):
            for item in inventory['files']:
                if item['root'] == priority and item['code'] in code_index:
                    chosen[item['code']] = priority
        if any(chosen.get(code) != label for code, label in old_sources.first().items()):
            raise ValueError('FROZEN_SOURCE_CACHE_PRIORITY_MISMATCH')
        quarantine = bind(HERE.parents[1] / 'pullback_train/quarantine_2026.csv')
        excluded = {}
        with quarantine.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                excluded.setdefault(row['code'], set()).add(int(row['day']))
        bound_raw = {r['code']: r for r in manifest['source_files']}
        r5 = np.full((len(codes), len(cutoffs)), np.nan)
        old15 = np.full((len(codes), len(selected)), np.nan)
        time15 = {t: i for i, t in enumerate(selected[TIME])}
        for code, t, value in old[['code', TIME, 'stock_day_return_raw']].itertuples(index=False, name=None):
            old15[code_index[code], time15[t]] = value
        ledger, comparisons, new15_only = [], 0, 0
        for i, (code, label) in enumerate(sorted(chosen.items()), 1):
            path = ROOTS[label] / (code + '.parquet')
            body = path.read_bytes()
            digest = hashlib.sha256(body).hexdigest()
            known = bound_raw.get(code)
            if known and (known['root'] != label or known['sha256'] != digest):
                raise ValueError('PREVIOUSLY_BOUND_RAW_INPUT_CHANGED:' + code)
            frame = pq.read_table(pa.BufferReader(body), filters=[('date', '>=', DAYS[0] * 1000000),
                ('date', '<=', DAYS[-1] * 1000000 + 150000)],
                columns=['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']).to_pandas()
            values, quality = minute_returns(frame, code, list(DAYS),
                {day: references.get((code, day), np.nan) for day in DAYS}, excluded.get(code, set()))
            original = old15[code_index[code]]
            valid_old = np.isfinite(original)
            np.testing.assert_allclose(values[15][valid_old], original[valid_old], rtol=0, atol=1e-12)
            comparisons += int(valid_old.sum())
            new15_only += int((np.isfinite(values[15]) & ~valid_old).sum())
            r5[code_index[code]] = values[5]
            ledger.append(dict(code=code, path=str(path), sha256=digest, bytes=len(body),
                original_raw_hash_available=known is not None, **quality))
            if i % 200 == 0:
                print(json.dumps(dict(stage='native_minute_to_5m', processed=i, total=len(chosen))), flush=True)
        if comparisons != int(np.isfinite(old15).sum()):
            raise ValueError('OLD_VALID_POPULATION_NOT_FULLY_COMPARED')
        write('raw_input_ledger.json', encoded(ledger))
        board_chunks = []
        for board in boards:
            values = basket(r5, [code_index[c] for c in board['symbols']], 5)
            chunk = pd.DataFrame(values)
            chunk.insert(0, TIME, cutoffs)
            chunk.insert(0, 'board_code', board['code'])
            chunk['member_count'] = len(board['symbols'])
            chunk['price_scale_verified'] = False
            board_chunks.append(chunk)
        board_frame = pd.concat(board_chunks, ignore_index=True)
        stocks = pd.DataFrame(dict(code=np.repeat(codes, len(cutoffs)),
            **{TIME: np.tile(cutoffs, len(codes))}, day_return=r5.ravel()))
        leaders = leader_watch(board_frame, {b['code']: b['symbols'] for b in boards}, stocks)
        watched = stocks[stocks.code.isin(leaders.code.unique())]
        leader_paths = paths(watched.assign(level=1 + watched.day_return, path_ok=watched.day_return.notna()), 5)
        leaders = leaders.merge(leader_paths, on=['code', TIME], how='left', validate='many_to_one')
        leaders = leaders.merge(board_frame[['board_code', TIME, 'common_change']],
                                on=['board_code', TIME], how='left', validate='many_to_one')
        leaders['leader_minus_sector_day_return_change'] = leaders.day_return_change - leaders.common_change
        write('leader_watch_5m.parquet', parquet_body(leaders))
        market_path = HERE / 'local_market_context_20261017_v1/market_price_features.parquet'
        bind(market_path, previous['inputs'][str(market_path.resolve())])
        market = pq.read_table(market_path, filters=[('day', 'in', list(DAYS)),
                                                     ('frequency_minutes', '=', 5)]).to_pandas()
        counts = {}
        for symbol, prefix in [('000001.SH', 'sse'), ('399006.SZ', 'cyb')]:
            source = market[market.symbol == symbol].rename(columns={'close': 'level', 'day_return_prev_close': 'day_return'})
            evidence = board_evidence(source.assign(path_ok=source.source_complete), board_frame, 5)
            evidence['benchmark_symbol'] = symbol
            write(prefix + '_board_evidence_5m.parquet', parquet_body(evidence))
            counts[symbol] = {f: {'true': int(evidence[f].sum()), 'unknown': int(evidence[f].isna().sum())}
                for f in ['market_new_high_sector_not_new_high', 'market_rising_sector_nonpositive', 'market_rising_sector_falling']}
        # Exact source-row references, not duplicate labels in a new training X.
        cohort = []
        for segment in ('train', 'select', 'calibration', 'test'):
            path = bind(HERE / 'joint_fit_review_20261017_v1' / (segment + '.parquet'))
            keys = pq.read_table(path, columns=['code', TIME]).to_pandas()
            take = keys[TIME].floordiv(1000000).isin(DAYS)
            selected_keys = keys.loc[take].copy()
            selected_keys['source_row'] = np.flatnonzero(take)
            selected_keys['segment'] = segment
            cohort.append(selected_keys)
        cohort = pd.concat(cohort, ignore_index=True)
        if cohort.duplicated(['code', TIME]).any() or not set(cohort[TIME]) <= set(cutoffs):
            raise ValueError('COHORT_REFERENCE_KEY_CONFLICT')
        write('cohort_source_rows.parquet', parquet_body(cohort))
        for path, digest in inputs.items():
            if sha(Path(path)) != digest:
                raise ValueError('INPUT_CHANGED_DURING_BUILD:' + path)
        for name, item in outputs.items():
            if sha(OUT / name) != item['sha256']:
                raise ValueError('OUTPUT_READBACK_HASH_MISMATCH')
        report = dict(status='COMPLETE_NATIVE_5M_OBSERVATIONS_NOT_FIT', days=DAYS, boards=len(boards),
            cutoffs=len(cutoffs), board_rows=len(board_frame), leader_watch_rows=len(leaders),
            retained_noncurrent_leader_rows=int((~leaders.current_observed_leader).sum()),
            captured_leader_quote_unknown_rows=int(leaders.day_return.isna().sum()),
            full_member_board_rows=int(board_frame.full_members.sum()),
            raw_files=len(ledger), original_raw_hash_bound=sum(r['original_raw_hash_available'] for r in ledger),
            raw15_valid_rows_compared=comparisons, raw15_extra_valid_not_in_old_rows=new15_only,
            cohort_rows=len(cohort), cohort_segments=cohort.segment.value_counts().to_dict(),
            evidence_counts=counts, inputs=inputs, outputs=outputs.copy(), http_requests=0, fit=False,
            actual_historical_arrival=None, model_unchanged=True,
            selection_topk=None, labels_not_rewritten=True, historical_membership_bias_retained=True,
            remaining=['Multi-board selection/aggregation policy and full-cohort risk effect comparison',
                       'Snapshot/partial-price/source-lineage limits retained; no prospective validation'])
        write('report.json', encoded(report))
        write('stop_record.json', encoded(dict(status='COMPLETE', bytes_before_stop=used)), terminal=True)
        print(json.dumps({k: report[k] for k in ['status', 'boards', 'cutoffs', 'board_rows',
            'raw_files', 'raw15_valid_rows_compared', 'leader_watch_rows', 'cohort_rows', 'evidence_counts']}), flush=True)
        return report
    except Exception as exc:
        write('stop_record.json', encoded(dict(status='FAILED', error=str(exc))), terminal=True)
        raise


if __name__ == '__main__':
    materialize()
