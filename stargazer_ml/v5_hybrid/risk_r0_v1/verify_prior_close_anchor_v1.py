"""Exact preceding-session final-bar anchor potential; not scale admission or FIT."""
import io
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import repair_multiboard_observation_v1 as old
from build_joint_risk_features_v1 import QUARANTINE

HERE, TIME = old.HERE, old.TIME
OUT = HERE / 'prior_close_anchor_verification_20261018_v1'
CAP, RESERVE = 3 * 1024**2, 64 * 1024


def exact_previous(frame):
    """No ffill: index i uses exactly (i//16)*16-1 from the same code."""
    keys = ['code', 'cutoff_index']
    if frame.duplicated(keys).any():
        raise ValueError('DUPLICATE_SOURCE_KEY')
    if (frame.cutoff_index < 0).any():
        raise ValueError('INVALID_CUTOFF')
    last = frame.loc[frame.cutoff_index % 16 == 15].set_index(keys).stock_close
    index = pd.MultiIndex.from_arrays([frame.code, frame.cutoff_index // 16 * 16 - 1], names=keys)
    result = last.reindex(index).to_numpy(dtype=float, copy=True)
    result[~np.isfinite(result) | (result <= 0)] = np.nan
    return result


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if old.bytes_used() + CAP > old.TOTAL_CAP:
        raise ValueError('REGISTERED_BUDGET_RESERVATION_FAILED')
    manifest_path = old.SOURCE / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    clock_path = old.SOURCE / 'shared_market_and_leaders.parquet'
    clock = pd.read_parquet(clock_path, columns=['cutoff_index', TIME]).sort_values('cutoff_index')
    if not np.array_equal(clock.cutoff_index, np.arange(len(clock))) or len(clock) % 16:
        raise ValueError('NONCONTIGUOUS_FULL_CALENDAR')
    dates = clock[TIME].to_numpy() // 1000000
    if not np.all(dates.reshape(-1, 16) == dates.reshape(-1, 16)[:, :1]):
        raise ValueError('INVALID_SESSION_CLOCK')
    if not (clock[TIME].to_numpy()[15::16] % 1000000 == 150000).all():
        raise ValueError('FINAL_BAR_IS_NOT_SESSION_CLOSE')
    attrib_path = old.OUT / 'coverage_attribution.parquet'
    attrib = pd.read_parquet(attrib_path)
    times = attrib[TIME].to_numpy()
    scope_path = HERE / 'ths_full_concept_scope_20261017_v1.json'
    scope = json.loads(scope_path.read_bytes())
    boards = [b for b in pq.read_table(scope['source']).to_pylist()
              if b['source'] == 'ths' and b['type'] == 'concept']
    codes = sorted({c for b in boards for c in b['symbols']})
    weights = dict(zip(codes, old.sector_weights(codes, [b['symbols'] for b in boards])))
    if len(boards) != 978 or len(codes) != 5540 or len(manifest['joint_parts']) != 84:
        raise ValueError('FROZEN_SCOPE_CHANGED')
    inputs = {str(p): old.prep.sha(p) for p in [manifest_path, clock_path, attrib_path,
        scope_path, QUARANTINE, HERE/'verify_prior_close_anchor_v1.py',
        HERE/'test_prior_close_anchor_v1.py', HERE/'research_budget_v2.py']}
    inputs[scope['source']] = scope['source_sha256']
    used, outputs = 0, {}

    def write(name, body, terminal=False):
        nonlocal used
        reserve = 0 if terminal else RESERVE
        if used + len(body) + reserve > CAP or old.bytes_used() + len(body) + reserve > old.TOTAL_CAP:
            raise ValueError('SERIALIZED_BUDGET_STOP')
        if not OUT.exists():
            OUT.mkdir(exist_ok=False)
        with (OUT/name).open('xb') as stream:
            stream.write(body)
        used += len(body)
        outputs[name] = dict(sha256=old.prep.sha(OUT/name), bytes=len(body))

    write('plan.json', old.prep.encoded(dict(inputs=inputs, cap=CAP, fit=False,
        formal_admitted=False, policy='EXACT_PREVIOUS_CALENDAR_SESSION_FINAL_COMPLETE_15M',
        no_cross_missing_day_fill=True, old_fields_unchanged=True)))
    try:
        quarantine = pd.read_csv(QUARANTINE, dtype={'code': str})
        excluded = pd.MultiIndex.from_frame(quarantine[['code', 'day']])
        gain = np.zeros(len(times))
        missing = np.zeros(len(times))
        patches, owners = [], set()
        known_count, missing_count, potential_count = 0, 0, 0
        for number, part in enumerate(manifest['joint_parts']):
            path = old.SOURCE / part['path']
            expected = manifest['output_hashes'][part['path']]['sha256']
            if old.prep.sha(path) != expected:
                raise ValueError('SOURCE_SHARD_DRIFT')
            inputs[str(path)] = expected
            f = pd.read_parquet(path, columns=['code', 'cutoff_index', 'stock_close',
                'stock_reference_close_raw', 'stock_day_return_raw'])
            current_codes = set(f.code)
            if owners & current_codes:
                raise ValueError('CODE_SPLIT_ACROSS_SHARDS_REQUIRES_GLOBAL_LOOKUP')
            owners.update(current_codes)
            if not (np.isfinite(f.stock_close) & (f.stock_close > 0)).all():
                raise ValueError('INVALID_CURRENT_CLOSE')
            proposed = exact_previous(f)
            previous = f.stock_reference_close_raw.to_numpy(dtype=float)
            known = np.isfinite(previous) & (previous > 0)
            np.testing.assert_allclose(previous[known], proposed[known], rtol=0, atol=0)
            day = dates[f.cutoff_index.to_numpy()]
            qkey = pd.MultiIndex.from_arrays([f.code, day])
            if qkey.isin(excluded).any():
                raise ValueError('DERIVED_ROW_ON_QUARANTINED_DAY')
            f[TIME] = clock[TIME].to_numpy()[f.cutoff_index.to_numpy()]
            chosen = f[TIME].isin(times).to_numpy()
            bad = chosen & ~known
            potential = bad & np.isfinite(proposed)
            known_count += int((chosen & known).sum())
            missing_count += int(bad.sum())
            potential_count += int(potential.sum())
            loc = np.searchsorted(times, f.loc[bad, TIME])
            np.add.at(missing, loc, f.loc[bad, 'code'].map(weights).to_numpy())
            loc = np.searchsorted(times, f.loc[potential, TIME])
            np.add.at(gain, loc, f.loc[potential, 'code'].map(weights).to_numpy())
            # Factorized per-code/day evidence, not a duplicate full return table.
            p = f.loc[potential, ['code', 'cutoff_index']].copy()
            p['prior_cutoff_index'] = p.cutoff_index // 16 * 16 - 1
            p['reference_close_potential'] = proposed[potential]
            p['current_session_day'] = day[potential]
            p = p.drop(columns='cutoff_index').drop_duplicates(['code', 'current_session_day'])
            prior_keys = pd.MultiIndex.from_arrays([p.code, dates[p.prior_cutoff_index.to_numpy()]])
            if prior_keys.isin(excluded).any():
                raise ValueError('PRIOR_ANCHOR_ON_QUARANTINED_DAY')
            patches.append(p)
            if (number+1) % 14 == 0:
                print(json.dumps(dict(shards=number+1, potential_rows=potential_count)), flush=True)
        np.testing.assert_allclose(missing, attrib.prior_close_unavailable, rtol=0, atol=1e-12)
        if (gain > missing + 1e-12).any():
            raise ValueError('RECOVERY_EXCEEDS_MISSING_REFERENCE')
        patches = pd.concat(patches, ignore_index=True)
        patches['formal_admitted'] = False
        patches['scale_action_admission'] = 'NOT_INFERRED_FROM_CLOSE_PRESENCE'
        side = attrib[[TIME]].copy()
        side['old_valid_return_coverage'] = attrib.valid_return
        side['potential_anchor_gain'] = gain
        side['potential_only_coverage'] = attrib.valid_return + gain
        side['formal_added_coverage'] = 0.
        population = pd.read_parquet(old.prep.OUT/'population_audit.parquet')
        joined = population.merge(side, on=TIME, validate='many_to_one', sort=False)
        summary = {s: {c: float(joined.loc[joined.proposed_segment_name.eq(s)
            & joined.common_main_eligible, c].mean()) for c in side.columns if c != TIME}
            for s in old.prep.SEGMENTS}
        write('anchor_potential.parquet', old.parquet(patches))
        write('coverage_potential.parquet', old.parquet(side))
        for path, expected in inputs.items():
            if old.prep.sha(path) != expected:
                raise ValueError('INPUT_DRIFT_DURING_RUN:' + path)
        write('report.json', old.prep.encoded(dict(status='COMPLETE_STRUCTURAL_POTENTIAL_NOT_FORMAL_ADMISSION',
            inputs=inputs, shards=84, calendar_days=len(clock)//16, cutoffs=len(times),
            known_reference_rows_verified=known_count, missing_reference_rows=missing_count,
            potential_recovered_rows=potential_count, factorized_code_days=len(patches),
            common_population_summary=summary, old_fields_changed=False, labels_changed=False,
            formal_admitted_rows=0, fit=False, http_requests=0,
            next_step='Bind window-level scale/action evidence before adopting any anchor replacement')))
        write('manifest.json', old.prep.encoded(dict(outputs=outputs.copy())))
        write('stop_record.json', old.prep.encoded(dict(status='COMPLETE_NOT_FIT',
            manifest_sha256=old.prep.sha(OUT/'manifest.json'))), terminal=True)
        return dict(bytes=used, potential_rows=potential_count, summary=summary)
    except Exception as exc:
        write('stop_record.json', old.prep.encoded(dict(status='FAILED_NO_RESTART',
            error=f'{type(exc).__name__}:{exc}')), terminal=True)
        raise


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False))
