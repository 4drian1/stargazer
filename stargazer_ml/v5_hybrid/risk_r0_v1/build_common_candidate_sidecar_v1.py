"""One frozen-source replay -> 34 candidate columns; no estimators or HTTP."""
import argparse
import io
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import common_candidate_aggregation_v1 as aggregation
import common_loo_derive_v1 as derive
import continuous_motion_common_v1 as common
from continuous_motion_matrix_v1 import path_arrays, divide
from intraday_relative_stall_v1 import TIME

HERE = Path(__file__).resolve().parent
OUT = HERE / 'common_candidate_sidecar_20261019_v1'
FIELDS = aggregation.learning_fields()


def month_features(population, codes, memberships, returns, markets, clock, frequency,
                   check_real_prefix=False):
    """Pure monthly kernel, candidate rows in input order, board identity is frozen order."""
    aggregation.validate_population(population)
    if frequency not in (5, 15) or len(clock) != returns.shape[1] or len(set(clock)) != len(clock):
        raise ValueError('INVALID_SOURCE_CLOCK')
    positions = {int(t): i for i, t in enumerate(clock)}
    if any(int(t) not in positions for t in population[TIME]):
        raise ValueError('CANDIDATE_CLOCK_MISS')
    ci = {c: i for i, c in enumerate(codes)}
    if any(c not in ci for c in population.code):
        raise ValueError('CANDIDATE_SOURCE_CODE_MISS')
    stocks = path_arrays(1 + returns, np.isfinite(returns), frequency)
    market = {name: path_arrays(values, np.isfinite(values), frequency)
              for name, values in markets.items()}
    if set(market) != {'sse', 'cyb'}:
        raise ValueError('BENCHMARK_IDENTITIES_REQUIRED')
    metrics = common.METRICS_COMMON
    totals = {(scope, m): np.zeros(len(clock)) for scope in ('all', 'positive') for m in metrics}
    known = {key: np.zeros(len(clock), dtype=np.int64) for key in totals}
    selected_n = {s: np.zeros(len(clock), dtype=np.int64) for s in ('all', 'positive')}
    by_board, roster = {}, {c: [] for c in population.code.unique()}
    for board, members in enumerate(memberships):
        identity = f'frozen_board_{board:04d}'
        local_rows = {gi: i for i, gi in enumerate(members)}
        for code in roster:
            if ci[code] in local_rows:
                roster[code].append(identity)
        selected = population.code.map(lambda c: ci[c] in local_rows).to_numpy()
        rows = population.loc[selected, aggregation.KEYS].copy()
        if len(rows):
            by_board[board] = (rows, np.array([local_rows[ci[c]] for c in rows.code]),
                               np.array([positions[int(t)] for t in rows[TIME]]))
    if any(not boards for boards in roster.values()):
        raise ValueError('CANDIDATE_HAS_NO_FROZEN_BOARD')
    assignments, checked = [], []
    for board, members in enumerate(memberships):
        inter = derive.compute_board(returns, members, frequency, stocks)
        bf = derive.board_fields(inter)
        level = np.where(inter['safe'], divide(inter['Sx0'], inter['denom_n']), np.nan)
        bmetrics = {'deceleration': bf['deceleration']}
        for benchmark in ('sse', 'cyb'):
            bmetrics[f'{benchmark}_relative_return'] = bf['return_'] - market[benchmark]['return_']
            bmetrics[f'{benchmark}_relative_deceleration'] = (
                bf['deceleration'] - market[benchmark]['deceleration'])
        for scope in ('all', 'positive'):
            select = np.ones(len(clock), dtype=bool) if scope == 'all' else np.isfinite(level) & (level > 0)
            selected_n[scope] += select
            for metric, values in bmetrics.items():
                present = select & np.isfinite(values)
                totals[scope, metric] += np.where(present, values, 0)
                known[scope, metric] += present
        if board not in by_board:
            continue
        rows, row_i, bar = by_board[board]
        loo = {f: derive.loo_all_members(inter, f) for f in
               ('return_', 'deceleration', 'current_deceleration', 'captured_deceleration')}
        rows['board'] = f'frozen_board_{board:04d}'
        rows['deceleration'] = loo['deceleration'][row_i, bar]
        for benchmark in ('sse', 'cyb'):
            for metric, key in (('return', 'return_'), ('deceleration', 'deceleration')):
                rows[f'{benchmark}_relative_{metric}'] = loo[key][row_i, bar] - market[benchmark][key][bar]
        for metric in ('current_deceleration', 'captured_deceleration'):
            rows[metric] = loo[metric][row_i, bar]
        assignments.append(rows)
        informative = np.flatnonzero(np.isfinite(loo['deceleration'][row_i, bar]))
        if check_real_prefix and not checked and len(informative):
            choice = int(informative[0])
            i, at = int(row_i[choice]), int(bar[choice])
            small = returns[members].copy()
            calc = lambda x: common.board_motion_common(
                x, list(range(len(members))), frequency,
                path_arrays(1 + x, np.isfinite(x), frequency), exclude=i)
            original = calc(small)
            mutated = small.copy(); mutated[:, at+1:] = .123
            truncated = small.copy(); truncated[:, at+1:] = np.nan
            for changed in (mutated, truncated):
                replay = calc(changed)
                for field in loo:
                    np.testing.assert_allclose(original[field][:at+1], replay[field][:at+1],
                                               atol=1e-12, rtol=0, equal_nan=True)
            for field in loo:
                np.testing.assert_allclose(original[field][at], loo[field][i, at],
                                           atol=1e-12, rtol=0, equal_nan=True)
            if not np.isfinite(original['deceleration'][:at+1]).any():
                # Do not count an all-missing prefix as a successful non-vacuous case.
                continue
            checked.append(dict(code=codes[members[i]], cutoff=int(clock[at]), board=board))
    own, audit = aggregation.aggregate_own(population, pd.concat(assignments, ignore_index=True),
                                           frequency, roster)
    result = own.copy()
    at = np.array([positions[int(t)] for t in population[TIME]])
    for scope in ('all', 'positive'):
        audit[f'ccm_v1_{frequency}_{scope}_selected_boards'] = selected_n[scope][at]
        for metric in metrics:
            result[f'ccm_v1_{frequency}_{scope}_{metric}'] = divide(
                totals[scope, metric], known[scope, metric])[at]
            audit[f'ccm_v1_{frequency}_{scope}_{metric}_known_boards'] = known[scope, metric][at]
    audit['frequency'] = frequency
    return result, audit, checked


def run():
    from build_continuous_motion_population_v1 import arrays
    from build_multiboard_full_features_v1 import market_arrays
    import prepare_auxiliary_fixed_fit_v1 as parent
    from prepare_common_candidate_comparison_v1 import sha, frame_sha
    from research_budget_common_v3 import EvidenceWriter, ADDENDUM

    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    prepared = HERE / 'common_candidate_preparation_20261019_v1'
    config = json.loads((prepared / 'config.json').read_bytes())
    refs = dict(json.loads((prepared / 'plan.json').read_bytes())['inputs'])
    for path in [prepared / n for n in ('config.json', 'report.json', 'manifest.json', 'stop_record.json')]:
        refs[str(path)] = sha(path)
    for name in ('build_common_candidate_sidecar_v1.py', 'test_common_candidate_sidecar_v1.py',
                 'research_budget_common_v3.py'):
        refs[str(HERE / name)] = sha(HERE / name)
    refs[str(ADDENDUM)] = sha(ADDENDUM)
    for path, expected in refs.items():
        if sha(path) != expected:
            raise ValueError('FROZEN_INPUT_DRIFT:' + path)
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromNames(
        ['test_common_candidate_sidecar_v1', 'test_common_candidate_preparation_v1']))
    if not tests.wasSuccessful():
        raise ValueError('SIDECAR_UNIT_ACCEPTANCE_FAILED:' + log.getvalue())
    writer = EvidenceWriter(OUT)
    writer.json('plan.json', dict(inputs=refs, columns=FIELDS, real_fit_authorized=False,
                                 raw_replay_scope='ONE_BOUND_4196_FILE_PASS', output_cap=writer.cap))
    try:
        base = json.loads((parent.OUT / 'config.json').read_bytes())
        parts = parent.load_parts(base)
        old_report = json.loads((prepared / 'report.json').read_bytes())
        for segment, frame in parts.items():
            label_cols = ['code', TIME, parent.TARGET, 'label_end_time', 'auxiliary_end_time']
            if (frame_sha(frame[label_cols]) != old_report['segments'][segment]['keys_labels_sha256']
                    or frame_sha(frame[config['fields'][:232]]) != old_report['segments'][segment]['original_232_sha256']):
                raise ValueError('ORIGINAL_POPULATION_OR_LABEL_OR_FEATURE_DRIFT')
        population = pd.concat([f[['code', TIME]].assign(segment=s)
                                for s, f in parts.items()], ignore_index=True)
        days = sorted(set(population[TIME] // 1000000))
        codes, members, native, raw_report = arrays(days)
        market = pd.read_parquet(HERE / 'local_market_context_20261017_v1/market_price_features.parquet')
        di = {d: i for i, d in enumerate(days)}
        sidecar = population[aggregation.KEYS].copy()
        audits, checks = [], []
        checked_segments = set()
        for freq in (5, 15):
            width = 240 // freq
            for month in sorted({d // 100 for d in days}):
                md = [d for d in days if d // 100 == month]
                sub = population[population[TIME].floordiv(100000000).eq(month)]
                clock, markets = market_arrays(market, md, freq)
                values, audit, case = month_features(
                    sub, codes, members, native[freq][:, di[md[0]]*width:(di[md[-1]]+1)*width],
                    markets, clock, freq,
                    check_real_prefix=(freq, str(sub.segment.iloc[0])) not in checked_segments)
                for column in values.columns.difference(aggregation.KEYS):
                    sidecar.loc[sub.index, column] = values[column].to_numpy()
                audits.append(audit)
                if case:
                    checks.extend([dict(frequency=freq, segment=str(sub.segment.iloc[0]), **x) for x in case])
                    checked_segments.add((freq, str(sub.segment.iloc[0])))
                print(json.dumps(dict(stage='common_candidate_month', frequency=freq,
                                      month=month, rows=len(sub))), flush=True)
        result = aggregation.attach_sidecars(parts, sidecar, config['fields'][:232])
        profiles = {}
        for segment, frame in result.items():
            profiles[segment] = {field: dict(finite=int(frame[field].notna().sum()), rows=len(frame),
                sha256=frame_sha(frame[[field]])) for field in FIELDS}
        # Schema order and float64 must survive lossless serialization and readback.
        sidecar = sidecar[aggregation.KEYS + FIELDS]
        sink = io.BytesIO(); sidecar.to_parquet(sink, index=False, compression='zstd')
        body = sink.getvalue()
        pd.testing.assert_frame_equal(pd.read_parquet(io.BytesIO(body)), sidecar)
        writer.write('candidate_features.parquet', body)
        audit_frame = pd.concat(audits, ignore_index=True)
        sink = io.BytesIO(); audit_frame.to_parquet(sink, index=False, compression='zstd')
        writer.write('candidate_audit.parquet', sink.getvalue())
        for path, expected in refs.items():
            if sha(path) != expected:
                raise ValueError('INPUT_CHANGED_DURING_ACCEPTANCE:' + path)
        if len(checked_segments) != 8:
            raise ValueError('NONVACUOUS_REAL_PREFIX_CASES_INCOMPLETE')
        writer.json('report.json', dict(status='PASS_REAL_34_COLUMN_CANDIDATE_SIDECAR_NOT_FIT',
            rows=len(sidecar), columns=FIELDS, profiles=profiles, original_232_and_labels_unchanged=True,
            unit_tests=tests.testsRun, unit_output=log.getvalue(), real_prefix_and_loo_cases=checks,
            source_replay=raw_report, http_requests=0, model_fits=0, training_authorized=False,
            audit_columns_not_learned=True, member_snapshot_and_restricted_scale_limits_unchanged=True))
        writer.json('manifest.json', dict(outputs=writer.outputs.copy()))
        writer.json('stop_record.json', dict(status='COMPLETE_NOT_FIT',
                    manifest_sha256=sha(OUT / 'manifest.json')), terminal=True)
    except Exception as exc:
        writer.json('stop_record.json', dict(status='FAILED_NO_RESTART',
                    reason=f'{type(exc).__name__}:{exc}'), terminal=True)
        raise
    return dict(status='COMPLETE_NOT_FIT', rows=len(sidecar), fields=len(FIELDS), bytes=writer.used)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-and-accept', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(run()))
