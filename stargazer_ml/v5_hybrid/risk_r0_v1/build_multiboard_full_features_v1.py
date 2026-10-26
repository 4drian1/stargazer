"""Full original cohort, factorized monthly features. No HTTP, label rebuild or fit."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from intraday_relative_stall_v1 import TIME, slots
from intraday_relative_stall_v2 import minute_returns
from inventory_ths_stockdb_cache_v1 import ROOTS
from materialize_relative_stall_pilot_v1 import HERE, SOURCE, sha, parquet_body
from multiboard_risk_features_v1 import matrix_features, summarize
from research_budget_v2 import bytes_used, TOTAL_CAP

OUT = HERE / 'multiboard_full_features_20261018_v1'
CAP, RESERVE = 28 * 1024**2, 256 * 1024
SEGMENTS = ('train', 'select', 'calibration', 'test')


def daily_references(frame, code_index, time_to_day, reference, old15):
    """Fill bound references from old derived shards; reject within-day conflicts."""
    local = frame.cutoff_index.map(time_to_day).notna()
    f = frame.loc[local].copy()
    if f.empty:
        return
    f['_day'] = f.cutoff_index.map(time_to_day).astype(int)
    grouped = f.groupby(['code', '_day']).stock_reference_close_raw
    if (grouped.nunique() > 1).any():
        raise ValueError('OLD_DAILY_REFERENCE_CONFLICT')
    for (code, day), value in grouped.first().items():
        reference[code_index[code], day] = value
    t = f.cutoff_index.map(old15['_map']).to_numpy(dtype=int)
    c = f.code.map(code_index).to_numpy(dtype=int)
    old15['values'][c, t] = f.stock_day_return_raw.to_numpy(dtype=float)


def market_arrays(frame, days, frequency):
    grid = [t for day in days for t in slots(int(day), frequency)]
    result = {}
    for symbol, name in [('000001.SH', 'sse'), ('399006.SZ', 'cyb')]:
        f = frame[(frame.symbol == symbol) & (frame.frequency_minutes == frequency)]
        if f[TIME].duplicated().any():
            raise ValueError('DUPLICATE_MARKET_CUTOFF')
        f = f.set_index(TIME).reindex(grid)
        result[name] = np.where(f.source_complete.fillna(False), f.close, np.nan).astype(float)
    return grid, result


def golden_fields(days, close, states):
    output = {name: [] for name in ['new_sse_golden_position', 'golden_asof_day',
        *['new_sse_golden_distance_' + str(r).replace('.', '_') for r in (.236, .382, .5, .618, .786)]]}
    for day, prices in zip(days, close.reshape(-1, 16)):
        prior = [s for s in states if s['asof_day'] < day]
        state = max(prior, key=lambda s: s['asof_day']) if prior else None
        low, high = (state['low'], state['high']) if state else (np.nan, np.nan)
        output['new_sse_golden_position'].extend((prices - low) / (high - low))
        output['golden_asof_day'].extend([state['asof_day'] if state else np.nan] * 16)
        for r in (.236, .382, .5, .618, .786):
            output['new_sse_golden_distance_' + str(r).replace('.', '_')].extend(prices / (low + r * (high - low)) - 1)
    return output


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if bytes_used() + CAP > TOTAL_CAP:
        raise ValueError('REGISTERED_STORAGE_RESERVATION_FAILED')
    OUT.mkdir(exist_ok=False)
    used, outputs, inputs = 0, {}, {}

    def encoded(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')

    def write(name, body, terminal=False):
        nonlocal used
        reserve = 0 if terminal else RESERVE
        if used + len(body) + reserve > CAP or bytes_used() + len(body) + reserve > TOTAL_CAP:
            raise ValueError('SERIALIZED_REGISTERED_BUDGET_STOP')
        with (OUT / name).open('xb') as stream:
            stream.write(body)
        used += len(body)
        outputs[name] = dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())

    def bind(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('FROZEN_INPUT_CHANGED:' + str(path))
        inputs[str(path.resolve())] = digest
        return path

    try:
        for path in [Path(__file__), HERE / 'multiboard_risk_features_v1.py',
                     HERE / 'test_multiboard_risk_features_v1.py', HERE / 'test_build_multiboard_full_features_v1.py',
                     HERE / 'intraday_relative_stall_v2.py',
                     HERE / 'contracts/INTRADAY_MULTI_BOARD_CONTEXT_V1.json', SOURCE / 'manifest.json']:
            bind(path)
        model = bind(HERE / 'postclose_repaired_research_fit_20261017_v1/model.txt')
        ledger_path = bind(HERE / 'relative_stall_5m_20261018_v1/raw_input_ledger.json')
        ledger = json.loads(ledger_path.read_bytes())
        old_manifest = json.loads((SOURCE / 'manifest.json').read_bytes())
        source_scope = json.loads(bind(HERE / 'ths_full_concept_scope_20261017_v1.json').read_bytes())
        snapshot = bind(Path(source_scope['source']), source_scope['source_sha256'])
        boards = sorted([b for b in pq.read_table(snapshot).to_pylist()
                         if b['source'] == 'ths' and b['type'] == 'concept'], key=lambda b: b['code'])
        codes = sorted({c for b in boards for c in b['symbols']})
        code_index = {c: i for i, c in enumerate(codes)}
        members = [[code_index[c] for c in b['symbols']] for b in boards]
        cohorts = []
        for segment in SEGMENTS:
            path = bind(HERE / 'joint_fit_review_20261017_v1' / (segment + '.parquet'))
            frame = pq.read_table(path, columns=['code', TIME]).to_pandas()
            frame['segment'], frame['source_row'] = segment, np.arange(len(frame))
            cohorts.append(frame)
        cohort = pd.concat(cohorts, ignore_index=True)
        if len(cohort) != 100704 or cohort.duplicated(['code', TIME]).any():
            raise ValueError('ORIGINAL_COHORT_CHANGED')
        days = sorted(set(cohort[TIME] // 1000000))
        estimate = dict(cohort_rows=len(cohort), days=len(days), codes=len(codes), boards=len(boards),
            r5_memory_bytes=len(codes) * len(days) * 48 * 8,
            old15_memory_bytes=len(codes) * len(days) * 16 * 8,
            uncompressed_candidate_feature_upper_bytes=len(cohort) * 60 * 8,
            strategy='MONTHLY_SHARED_CONTEXT_AND_KEYED_OWN_CONTEXT_NO_FULL_BOARD_OUTPUT',
            output_cap_bytes=CAP, available_registered_bytes=TOTAL_CAP - bytes_used())
        print(json.dumps(dict(stage='prewrite_estimate', **estimate)), flush=True)
        write('plan.json', encoded(dict(estimate=estimate, status='RESEARCH_FEATURE_IMPLEMENTATION_NOT_FIT',
            source_policy='BIND_EXISTING_RAW_LEDGER_NO_NEW_FETCH', model_sha256=inputs[str(model.resolve())],
            risk_profit_gate=False, fit=False, timestamp_policy='ASSUMED_END_HISTORICAL_RECONSTRUCTION')))
        # Freeze candidate field names before any true values or labels are inspected.
        shared_names, own_names = [], []
        for frequency in (5, 15):
            shape = 240 // frequency
            small = matrix_features(np.full((2, shape), .01), [[0, 1]], frequency)
            markets = dict(sse=np.full(shape, 100.), cyb=np.full(shape, 100.))
            shared_names.extend(summarize(small, markets, frequency, f'mb{frequency}_'))
            own_names.extend(summarize(small, markets, frequency, f'own{frequency}_', own=True))
        shared_names.extend(['new_sse_golden_position', *['new_sse_golden_distance_' + str(r).replace('.', '_') for r in (.236, .382, .5, .618, .786)]])
        write('feature_spec.json', encoded(dict(status='CANDIDATE_NUMERIC_WHITELIST_NOT_SIGNED_FIT',
            shared=shared_names, own=own_names, metadata=['golden_asof_day', 'code', TIME, 'source_row', 'segment'],
            distribution_quantiles=[.1, .5, .9], negative_leader_requires_all_selected_refs_known=True,
            original_golden_v1_policy='KEEP_IN_ORIGINAL_BASELINE; DO_NOT_TREAT_AS_EQUIVALENT_TO_V2_OR_AN_INDEPENDENT_VOTE',
            shared_market_may_include_candidate=True, own_context_excludes_candidate=True,
            no_topk=True, unknown_is_not_zero=True, fit_authorized=False)))
        shared_source = bind(SOURCE / 'shared_market_and_leaders.parquet',
                            old_manifest['output_hashes']['shared_market_and_leaders.parquet']['sha256'])
        clock = pq.read_table(shared_source, columns=['cutoff_index', TIME]).to_pandas()
        selected = clock[clock[TIME].floordiv(1000000).isin(days)]
        if len(selected) != len(days) * 16 or set(selected[TIME]) != set(cohort[TIME]):
            raise ValueError('EXACT_COHORT_CUTOFF_GRID_MISMATCH')
        day_index = {day: i for i, day in enumerate(days)}
        time_to_day = {row.cutoff_index: day_index[row.decision_ts_assumed_end // 1000000] for row in selected.itertuples()}
        t15 = {t: i for i, t in enumerate(selected.cutoff_index)}
        reference = np.full((len(codes), len(days)), np.nan)
        old15 = np.full((len(codes), len(days) * 16), np.nan)
        seen, source_cache = set(), {}
        for item in old_manifest['joint_parts']:
            path = bind(SOURCE / item['path'], old_manifest['output_hashes'][item['path']]['sha256'])
            frame = pq.read_table(path, columns=['code', 'cutoff_index', 'stock_reference_close_raw',
                                                 'stock_day_return_raw', 'source_cache']).to_pandas()
            present = set(frame.code.unique())
            if seen & present:
                raise ValueError('SOURCE_CODE_SPLIT_ACROSS_SHARDS')
            seen.update(present)
            if (frame.groupby('code').source_cache.nunique() > 1).any():
                raise ValueError('SOURCE_PRIORITY_CONFLICT')
            source_cache.update(frame.groupby('code').source_cache.first().to_dict())
            daily_references(frame, code_index, time_to_day, reference, dict(values=old15, _map=t15))
        quarantine = bind(HERE.parents[1] / 'pullback_train/quarantine_2026.csv')
        excluded = {}
        with quarantine.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                excluded.setdefault(row['code'], set()).add(int(row['day']))
        r5 = np.full((len(codes), len(days) * 48), np.nan)
        compared, raw_extra, raw_bindings = 0, 0, []
        for i, row in enumerate(ledger, 1):
            code, path = row['code'], Path(row['path'])
            if code in source_cache:
                if str((ROOTS[source_cache[code]] / (code + '.parquet')).resolve()) != str(path.resolve()):
                    raise ValueError('SOURCE_CACHE_IDENTITY_CHANGED')
            elif np.isfinite(reference[code_index[code]]).any():
                raise ValueError('REFERENCE_WITHOUT_ORIGINAL_SOURCE_IDENTITY')
            body = path.read_bytes()
            if hashlib.sha256(body).hexdigest() != row['sha256']:
                raise ValueError('RAW_FROZEN_LEDGER_CHANGED:' + code)
            frame = pq.read_table(pa.BufferReader(body), columns=['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']).to_pandas()
            values, quality = minute_returns(frame, code, days,
                dict(zip(days, reference[code_index[code]])), excluded.get(code, set()))
            original = old15[code_index[code]]
            valid = np.isfinite(original)
            np.testing.assert_allclose(values[15][valid], original[valid], rtol=0, atol=1e-12)
            compared += int(valid.sum())
            raw_extra += int((np.isfinite(values[15]) & ~valid).sum())
            r5[code_index[code]] = values[5]
            raw_bindings.append(dict(code=code, **quality))
            if i % 200 == 0:
                print(json.dumps(dict(stage='full_period_native', processed=i, total=len(ledger))), flush=True)
        if compared != int(np.isfinite(old15).sum()):
            raise ValueError('FULL_OLD_VALID_POPULATION_NOT_COMPARED')
        write('raw_quality.json', encoded(raw_bindings))
        market_path = bind(HERE / 'local_market_context_20261017_v1/market_price_features.parquet')
        market = pq.read_table(market_path).to_pandas()
        golden_path = bind(HERE / 'golden_anchor_replay_20261017_v2/replay.json')
        states = json.loads(golden_path.read_bytes())['states']
        months = sorted({day // 100 for day in days})
        parts, audits = [], {s: {} for s in SEGMENTS}
        for month in months:
            month_days = [day for day in days if day // 100 == month]
            start, stop = day_index[month_days[0]], day_index[month_days[-1]] + 1
            keys = cohort[cohort[TIME].floordiv(100000000).eq(month)].copy()
            shared_frame = pd.DataFrame({TIME: [t for d in month_days for t in slots(d, 15)]})
            own_frame = keys.copy()
            for name in own_names:
                own_frame[name] = np.nan
            for frequency, array in ((5, r5), (15, old15)):
                grid = 240 // frequency
                returns = array[:, start * grid:stop * grid]
                times, markets = market_arrays(market, month_days, frequency)
                positions = {t: i for i, t in enumerate(times)}
                take = [positions[t] for t in shared_frame[TIME]]
                board = matrix_features(returns, members, frequency)
                shared = summarize(board, markets, frequency, f'mb{frequency}_')
                for name, values in shared.items():
                    shared_frame[name] = values[take]
                del board
                for code, indexes in own_frame.groupby('code', sort=False).groups.items():
                    member = code_index[code]
                    related = [m for m in members if member in m]
                    if not related:
                        continue
                    board = matrix_features(returns, related, frequency, exclude_index=member)
                    own = summarize(board, markets, frequency, f'own{frequency}_', own=True)
                    at = [positions[t] for t in own_frame.loc[indexes, TIME]]
                    for name, values in own.items():
                        own_frame.loc[indexes, name] = values[at]
                if frequency == 15:
                    for name, values in golden_fields(month_days, markets['sse'], states).items():
                        shared_frame[name] = values
            if own_frame.empty or len(shared_frame) != len(month_days) * 16:
                raise ValueError('MONTH_POPULATION_MISMATCH')
            merged = own_frame.merge(shared_frame, on=TIME, validate='many_to_one')
            for segment, group in merged.groupby('segment'):
                for field in shared_names + own_names:
                    data = group[field].to_numpy(dtype=float)
                    if np.isinf(data).any():
                        raise ValueError('INFINITE_NEW_FEATURE')
                    stats = audits[segment].setdefault(field, dict(rows=0, finite=0, min=None, max=None))
                    stats['rows'] += len(data); finite = data[np.isfinite(data)]; stats['finite'] += len(finite)
                    if len(finite):
                        lo, hi = float(finite.min()), float(finite.max())
                        stats['min'] = lo if stats['min'] is None else min(lo, stats['min'])
                        stats['max'] = hi if stats['max'] is None else max(hi, stats['max'])
            own_name, shared_name = f'own_{month}.parquet', f'shared_{month}.parquet'
            write(own_name, parquet_body(own_frame)); write(shared_name, parquet_body(shared_frame))
            parts.append(dict(month=month, own=own_name, shared=shared_name, rows=len(own_frame), cutoffs=len(shared_frame)))
            print(json.dumps(dict(stage='monthly_features', month=month, cohort_rows=len(own_frame), bytes_written=used)), flush=True)
        for path, digest in inputs.items():
            if sha(Path(path)) != digest:
                raise ValueError('INPUT_CHANGED_DURING_BUILD:' + path)
        for name, item in outputs.items():
            if sha(OUT / name) != item['sha256']:
                raise ValueError('OUTPUT_READBACK_HASH_MISMATCH')
        report = dict(status='FULL_COHORT_THREE_GROUP_CANDIDATES_NOT_FIT', inputs=inputs, outputs=outputs.copy(),
            parts=parts, rows=len(cohort), days=len(days), candidate_codes=int(cohort.code.nunique()),
            segment_rows=cohort.segment.value_counts().to_dict(), raw_files=len(ledger), raw15_valid_compared=compared,
            raw15_extra_valid_not_admitted=raw_extra, shared_fields=shared_names, own_fields=own_names,
            audit=audits, labels_changed=False, http_requests=0, fit=False, model_unchanged=True,
            global_market_stats_not_candidate_independent=True, own_references_exclude_candidate=True,
            historical_actual_arrival=None, membership_snapshot_bias_retained=True,
            all_null_train=[name for name, stats in audits['train'].items() if stats['finite'] == 0],
            constant_train=[name for name, stats in audits['train'].items() if stats['finite'] and stats['min'] == stats['max']])
        write('report.json', encoded(report))
        write('stop_record.json', encoded(dict(status='COMPLETE', bytes_before_stop=used)), terminal=True)
        print(json.dumps({k: report[k] for k in ['status', 'rows', 'days', 'raw15_valid_compared', 'all_null_train', 'constant_train']}), flush=True)
        return report
    except Exception as exc:
        write('stop_record.json', encoded(dict(status='FAILED', error=str(exc))), terminal=True)
        raise


if __name__ == '__main__':
    run()
