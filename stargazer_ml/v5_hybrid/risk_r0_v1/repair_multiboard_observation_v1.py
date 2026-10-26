"""Bounded coverage provenance and exact-parent new-high/motion sidecar; no fit."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from build_multiboard_full_features_v1 import SOURCE, market_arrays
from intraday_relative_stall_v1 import TIME
from multiboard_risk_features_v1 import day_high, prior
from newhigh_common_motion_v1 import combine, sector_weights, footer_bounds
from read_multiboard_full_features_v1 import read_segment, metadata
import prepare_multiboard_comparison_v1 as prep
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = prep.HERE
OUT = HERE / 'multiboard_observation_repair_20261018_v1'
CAP, RESERVE = 6 * 1024**2, 128 * 1024
REASONS = ('no_bound_source', 'before_bound_source_start', 'after_bound_source_end',
           'current_bar_not_admitted_within_span', 'prior_close_unavailable',
           'valid_return', 'span_unknown_current_not_admitted')


def parquet(frame):
    sink = io.BytesIO()
    frame.to_parquet(sink, index=False, compression='zstd')
    return sink.getvalue()


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if bytes_used() + CAP > TOTAL_CAP:
        raise ValueError('REGISTERED_REPAIR_BUDGET_RESERVATION_FAILED')
    report, _ = metadata()
    refs = dict(report['inputs'])
    old_build = HERE / 'multiboard_full_features_20261018_v1'
    for name in ['report.json', 'stop_record.json']:
        refs[str(old_build / name)] = prep.sha(old_build / name)
    for name, item in report['outputs'].items():
        refs[str(old_build / name)] = item['sha256']
    for name in ['repair_multiboard_observation_v1.py', 'newhigh_common_motion_v1.py',
                 'test_multiboard_observation_repair_v1.py', 'audit_multiboard_observation_repair_v1.py',
                 'research_budget_v2.py', 'read_multiboard_full_features_v1.py']:
        refs[str(HERE / name)] = prep.sha(HERE / name)
    for path, expected in refs.items():
        if prep.sha(path) != expected:
            raise ValueError('FROZEN_REPAIR_INPUT_DRIFT:' + path)
    used, outputs = 0, {}

    def write(name, body, terminal=False):
        nonlocal used
        reserve = 0 if terminal else RESERVE
        if used + len(body) + reserve > CAP or bytes_used() + len(body) + reserve > TOTAL_CAP:
            raise ValueError('REPAIR_SERIALIZED_BUDGET_STOP')
        if not OUT.exists():
            OUT.mkdir(exist_ok=False)
        with (OUT / name).open('xb') as stream:
            stream.write(body)
        used += len(body)
        outputs[name] = dict(sha256=prep.sha(OUT / name), bytes=len(body))

    write('plan.json', prep.encoded(dict(inputs=refs, cap=CAP, raw_footer_file_cap=4196,
        old_shards_cap=84, no_minute_reaggregation=True, new_fields_not_fit_authorized=True,
        reason_order=list(REASONS), fit=False, http_requests=0)))
    try:
        frames = {s: read_segment(s) for s in prep.SEGMENTS}
        cohort = pd.concat([f.assign(segment=s) for s, f in frames.items()], ignore_index=True)
        if len(cohort) != 100704 or cohort.duplicated(['code', TIME]).any():
            raise ValueError('ORIGINAL_POPULATION_CHANGED')
        days = sorted(set(cohort[TIME] // 1000000))
        times = np.array(sorted(cohort[TIME].unique()), dtype=np.int64)
        scope = json.loads((HERE / 'ths_full_concept_scope_20261017_v1.json').read_bytes())
        boards = sorted([b for b in pq.read_table(scope['source']).to_pylist()
                         if b['source'] == 'ths' and b['type'] == 'concept'], key=lambda b: b['code'])
        members = [b['symbols'] for b in boards]
        codes = sorted({c for group in members for c in group})
        code_index = {c: i for i, c in enumerate(codes)}
        weights = sector_weights(codes, members)
        state = np.zeros((len(codes), len(times)), dtype=np.uint8)
        ledger = json.loads((HERE / 'relative_stall_5m_20261018_v1/raw_input_ledger.json').read_bytes())
        if len(ledger) != 4196 or len(codes) != 5540 or len(boards) != 978:
            raise ValueError('FROZEN_CATALOG_OR_RAW_SCOPE_CHANGED')
        footers = []
        for i, entry in enumerate(ledger):
            body = Path(entry['path']).read_bytes()
            if hashlib.sha256(body).hexdigest() != entry['sha256']:
                raise ValueError('BOUND_RAW_SOURCE_CHANGED:' + entry['code'])
            # Verify bytes once; decode only the footer, never the minute table.
            file = pq.ParquetFile(pa.BufferReader(body))
            first, last = footer_bounds(file)
            idx = code_index[entry['code']]
            state[idx] = 6 if first is None else 3
            if first is not None:
                state[idx, times < first] = 1
                state[idx, times > last] = 2
            footers.append(dict(code=entry['code'], first_timestamp=first, last_timestamp=last,
                                rows=file.metadata.num_rows, bytes=len(body), sha256=entry['sha256']))
            if (i + 1) % 500 == 0:
                print(json.dumps(dict(stage='verified_footer_only', files=i+1, total=len(ledger))), flush=True)
        footer = {r['code']: r for r in footers}
        old = json.loads((SOURCE / 'manifest.json').read_bytes())
        if len(old['joint_parts']) != 84:
            raise ValueError('DERIVED_SHARD_SCOPE_CHANGED')
        clock = pd.read_parquet(SOURCE / 'shared_market_and_leaders.parquet', columns=['cutoff_index', TIME])
        clock = clock[clock[TIME].isin(times)]
        if len(clock) != len(times) or set(clock[TIME]) != set(times):
            raise ValueError('EXACT_CLOCK_DIFFERENT')
        lookup = dict(zip(clock.cutoff_index, np.searchsorted(times, clock[TIME])))
        seen = np.zeros_like(state, dtype=bool)
        for i, part in enumerate(old['joint_parts']):
            path = SOURCE / part['path']
            # Old source hashes were already checked above via report inputs.
            f = pd.read_parquet(path, columns=['code', 'cutoff_index', 'stock_close',
                                             'stock_reference_close_raw', 'stock_day_return_raw'])
            f = f[f.cutoff_index.isin(lookup)]
            row = f.code.map(code_index).to_numpy(dtype=int)
            col = f.cutoff_index.map(lookup).to_numpy(dtype=int)
            if seen[row, col].any() or f.duplicated(['code', 'cutoff_index']).any():
                raise ValueError('DUPLICATE_DERIVED_KEYS')
            if (state[row, col] < 3).any():
                raise ValueError('DERIVED_CLOSE_OUTSIDE_BOUND_SOURCE_SPAN')
            if not (np.isfinite(f.stock_close) & (f.stock_close > 0)).all():
                raise ValueError('DERIVED_CURRENT_CLOSE_INVALID')
            reference = f.stock_reference_close_raw.to_numpy(dtype=float)
            ref_ok = np.isfinite(reference) & (reference > 0)
            valid = np.isfinite(f.stock_day_return_raw.to_numpy(dtype=float))
            if not np.array_equal(ref_ok, valid):
                raise ValueError('RETURN_REFERENCE_IDENTITY_NOT_EXPLAINED')
            seen[row, col] = True
            state[row, col] = np.where(valid, 5, 4)
            if (i+1) % 14 == 0:
                print(json.dumps(dict(stage='old_derived_provenance', shards=i+1, total=84)), flush=True)
        attribution = pd.DataFrame({TIME: times})
        for idx, reason in enumerate(REASONS):
            attribution[reason] = np.einsum('i,ij->j', weights, state == idx)
        np.testing.assert_allclose(attribution[list(REASONS)].sum(axis=1), 1., atol=1e-12, rtol=0)
        joined = cohort[[TIME, 'mb15_member_coverage']].merge(attribution, on=TIME, validate='many_to_one')
        np.testing.assert_allclose(joined.mb15_member_coverage, joined.valid_return, atol=1e-12, rtol=0)
        population = pd.read_parquet(prep.OUT / 'population_audit.parquet')
        pd.testing.assert_frame_equal(cohort[['code', TIME]].reset_index(drop=True), population[['code', TIME]])
        summary = {}
        for segment in prep.SEGMENTS:
            mask = population.proposed_segment_name.eq(segment) & population.common_main_eligible
            summary[segment] = {reason: float(joined.loc[mask.to_numpy(), reason].mean()) for reason in REASONS}
        market_path = HERE / 'local_market_context_20261017_v1/market_price_features.parquet'
        market = pd.read_parquet(market_path)
        shared = pd.DataFrame({TIME: times})
        high_cols = []
        for frequency in (5, 15):
            grid, markets = market_arrays(market, days, frequency)
            choose = np.searchsorted(np.asarray(grid), times)
            for benchmark, values in markets.items():
                high = day_high(values, np.isfinite(values), frequency)
                step = values / prior(values, frequency) - 1
                if not (step[high == 1] > 0).all():
                    raise ValueError('NEW_CLOSE_HIGH_DOES_NOT_IMPLY_PRICE_RISE')
                # Actual data prefix and future perturbation, not an empty pass.
                point = len(values) // 2 + 3
                truncated = values.copy(); truncated[point+1:] = np.nan
                disturbed = values.copy(); disturbed[point+1:] *= 7
                if not np.isfinite(high[:point+1]).any():
                    raise ValueError('EMPTY_REAL_MARKET_PREFIX')
                for revised in (truncated, disturbed):
                    np.testing.assert_allclose(day_high(revised, np.isfinite(revised), frequency)[:point+1],
                                               high[:point+1], equal_nan=True, atol=0, rtol=0)
                name = f'nh_{benchmark}{frequency}_close_new_day_high'
                shared[name] = high[choose]
                high_cols.append(name)
        context = cohort[['code', TIME, 'segment']].merge(shared, on=TIME, validate='many_to_one')
        shared_fields, own_fields = [], []
        for frequency in (5, 15):
            for benchmark in ('sse', 'cyb'):
                high = context[f'nh_{benchmark}{frequency}_close_new_day_high'].to_numpy()
                for scope_name in ('mb', 'own'):
                    prefix = f'{scope_name}{frequency}_'
                    for flag in ('nonpositive', 'falling'):
                        base = prefix + benchmark + '_positive_' + flag
                        name = f'nh_{prefix}{benchmark}_positive_common_{flag}'
                        r, cov = combine(cohort[base+'_rate'], cohort[base+'_coverage'],
                                         cohort[prefix+'positive_share'], high)
                        context[name+'_rate'], context[name+'_coverage'] = r, cov
                        target = shared_fields if scope_name == 'mb' else own_fields
                        target.extend([name+'_rate', name+'_coverage'])
        # Shared ratios must be identical for every candidate at the same cutoff.
        if (context.groupby(TIME)[shared_fields].nunique(dropna=False) > 1).any().any():
            raise ValueError('SHARED_NEW_HIGH_RATE_CONFLICT')
        shared = context[[TIME, *high_cols, *shared_fields]].drop_duplicates(TIME)
        own = context[['code', TIME, 'segment', *own_fields]]
        coverage = {}
        for segment in prep.SEGMENTS:
            mask = population.proposed_segment_name.eq(segment) & population.common_main_eligible
            f = context.loc[mask.to_numpy()]
            coverage[segment] = {name: dict(finite=int(f[name].notna().sum()), active=int((f[name] > 0).sum()),
                                          rows=len(f)) for name in shared_fields + own_fields + high_cols}
        write('bound_source_footers.parquet', parquet(pd.DataFrame(footers)))
        write('coverage_attribution.parquet', parquet(attribution))
        write('shared.parquet', parquet(shared))
        write('own.parquet', parquet(own))
        write('feature_spec.json', prep.encoded(dict(status='OBSERVATION_REPAIR_CANDIDATES_NOT_FIT',
            shared_fields=shared_fields, own_fields=own_fields, market_state_fields=high_cols,
            semantic='INDEX_COMPLETED_CLOSE_NEW_HIGH_AND_BOARD_COMMON_MEMBER_NONPOSITIVE_OR_FALLING',
            not_equivalent_to_whole_sector_not_new_high=True, old_fields_unchanged=True,
            source_group='same currently positive observed boards; own excludes candidate',
            selection_uses_no_outcomes=True, fit_authorized=False)))
        tail_hist = pd.DataFrame(footers).last_timestamp.dropna().floordiv(100000000).astype(int).value_counts().sort_index().to_dict()
        write('report.json', prep.encoded(dict(status='COMPLETE_COVERAGE_ATTRIBUTION_AND_NEW_HIGH_MOTION_SIDECAR',
            inputs=refs, outputs=outputs.copy(), rows=len(cohort), unique_cutoffs=len(shared),
            raw_footer_files=len(footers), raw_minute_tables_reparsed=0, old_derived_shards=84,
            scope_codes=len(codes), boards=len(boards), reason_order=list(REASONS),
            bound_source_last_month_histogram={str(k): int(v) for k,v in tail_hist.items()},
            common_population_weighted_coverage_reasons=summary, feature_coverage=coverage,
            new_rate_and_coverage_fields=len(shared_fields)+len(own_fields), market_state_fields=len(high_cols),
            old_coverage_unchanged=True, labels_changed=False, fit=False, http_requests=0,
            provenance_is_ordered_partition_not_unique_causal_diagnosis=True)))
        for path, expected in refs.items():
            if prep.sha(path) != expected:
                raise ValueError('FROZEN_INPUT_CHANGED_DURING_REPAIR:' + path)
        write('manifest.json', prep.encoded(dict(outputs=outputs.copy())))
        write('stop_record.json', prep.encoded(dict(status='COMPLETE_NOT_FIT_OR_DEPLOYMENT',
            manifest_sha256=prep.sha(OUT/'manifest.json'), bytes_before_stop=used)), terminal=True)
        return dict(status='COMPLETE_NOT_FIT_OR_DEPLOYMENT', rows=len(cohort), bytes=used)
    except Exception as exc:
        write('stop_record.json', prep.encoded(dict(status='FAILED_NO_RESTART', error=f'{type(exc).__name__}:{exc}')), terminal=True)
        raise


def read_context(segment):
    if segment not in prep.SEGMENTS:
        raise ValueError('UNKNOWN_SEGMENT')
    terminal = json.loads((OUT/'stop_record.json').read_bytes())
    if terminal['status'] != 'COMPLETE_NOT_FIT_OR_DEPLOYMENT':
        raise ValueError('REPAIR_NOT_COMPLETE')
    report = json.loads((OUT/'report.json').read_bytes())
    for name in ['shared.parquet', 'own.parquet']:
        if prep.sha(OUT/name) != report['outputs'][name]['sha256']:
            raise ValueError('REPAIR_SIDECAR_DRIFT')
    own = pd.read_parquet(OUT/'own.parquet', filters=[('segment', '=', segment)])
    shared = pd.read_parquet(OUT/'shared.parquet')
    if own.duplicated(['code', TIME]).any() or shared.duplicated(TIME).any() or not set(own[TIME]) <= set(shared[TIME]):
        raise ValueError('INVALID_EXACT_REPAIR_KEYS')
    return own.merge(shared, on=TIME, validate='many_to_one', sort=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--segment', choices=prep.SEGMENTS)
    args = parser.parse_args()
    print(json.dumps(run() if args.run else dict(rows=len(read_context(args.segment)), fit=False)))
