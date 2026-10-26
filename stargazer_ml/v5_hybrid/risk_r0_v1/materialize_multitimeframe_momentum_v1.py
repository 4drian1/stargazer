"""Independent proposed-parameter market sidecar; no label read or model fit."""
from pathlib import Path
import hashlib
import json
import io
import numpy as np
import pandas as pd
from multitimeframe_momentum_v1 import MomentumState, positive
from tdx_index_acceptance_v1 import grid
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = Path(__file__).parent
SOURCE = HERE / 'local_market_context_20261017_v1'
OUT = HERE / 'market_momentum_proposal_20261017_v1'
PARAMETERS = dict(rsi_period=14, macd_periods=(12, 26, 9), bbi_periods=(3, 6, 12, 24))
CAP, RESERVE = 16 * 1024**2, 1024**2
KEYS = ['symbol', 'frequency_minutes', 'decision_ts_assumed_end']


def digest(body):
    return hashlib.sha256(body).hexdigest()


def build(frame):
    if frame.duplicated(KEYS).any():
        raise ValueError('DUPLICATE_MARKET_KEY')
    if not set(frame.frequency_minutes.unique()) <= {5, 15}:
        raise ValueError('UNSUPPORTED_FREQUENCY')
    output = []
    for symbol, source in frame.groupby('symbol', sort=True):
        days = sorted(source.day.unique())
        five = source[source.frequency_minutes == 5].set_index('decision_ts_assumed_end')
        daily_state = MomentumState(**PARAMETERS)
        baseline, provisional = {}, {}
        previous_close = None
        previous_values = daily_state.branch(None)
        for day in days:
            baseline[day] = (previous_values.copy(), previous_close)
            for row in source[source.day == day].itertuples(index=False):
                valid = row.source_complete and positive(row.close)
                provisional[(row.frequency_minutes, row.decision_ts_assumed_end)] = daily_state.branch(row.close if valid else None)
            expected = grid(int(day), 5)
            complete = (all(t in five.index for t in expected)
                        and all(bool(five.loc[t, 'source_complete']) and positive(five.loc[t, 'close']) for t in expected))
            close = float(five.loc[expected[-1], 'close']) if complete else None
            previous_values = daily_state.update(close)
            previous_close = close
        for step, bars in source.groupby('frequency_minutes', sort=True):
            state = MomentumState(**PARAMETERS)
            bars = bars.set_index('decision_ts_assumed_end')
            for day in sorted(bars.day.unique()):
                for ts in grid(int(day), int(step)):
                    if ts not in bars.index:
                        # Missing past grid slots reset warmup; trailing slots cannot affect prior rows.
                        state.update(None)
                        continue
                    row = bars.loc[ts]
                    if int(row.day) != ts // 1000000:
                        raise ValueError('DAY_CLOCK_MISMATCH')
                    close = float(row.close) if row.source_complete and positive(row.close) else None
                    current = state.update(close)
                    prior, reference = baseline[day]
                    temp = provisional[(step, ts)]
                    record = dict(symbol=symbol, frequency_minutes=int(step), decision_ts_assumed_end=ts,
                                  daily_background_asof='STRICTLY_PRIOR_OBSERVED_DAY',
                                  daily_current_status='PROVISIONAL_AT_CUTOFF_NOT_CONFIRMED_DAILY_PIVOT',
                                  research_parameters_status='PROPOSED_NOT_FROZEN',
                                  time_semantics='ASSUMED_END_HISTORICAL_RECONSTRUCTION',
                                  actual_available_at=None, training_admitted=False,
                                  confirmed_divergence=None)
                    for prefix, values in [('intraday', current), ('prior_daily', prior), ('provisional_daily', temp)]:
                        for name, value in values.items():
                            record[prefix + '_' + name] = value
                        for name in ('macd_dif', 'macd_dea', 'macd_hist'):
                            denominator = reference if prefix == 'prior_daily' else close
                            record[prefix + '_' + name + '_over_close'] = (values[name] / denominator
                                if values[name] is not None and positive(denominator) else None)
                    for name in ('rsi14', 'macd_dif', 'macd_hist'):
                        record['provisional_daily_' + name + '_change_from_prior'] = (temp[name] - prior[name]
                            if temp[name] is not None and prior[name] is not None else None)
                    record['price_return_from_prior_daily_close'] = close / reference - 1 if close and reference else None
                    output.append(record)
    result = pd.DataFrame(output).sort_values(KEYS).reset_index(drop=True)
    # Stable numeric schema even when an early prefix is entirely in warmup.
    for name in result:
        if name.startswith(('intraday_', 'prior_daily_', 'provisional_daily_')) or name == 'price_return_from_prior_daily_close':
            result[name] = result[name].astype(float)
    if not result[KEYS].equals(frame[KEYS].sort_values(KEYS).reset_index(drop=True)):
        raise ValueError('MISSING_OR_OFF_GRID_INPUT_KEY')
    return result


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    manifest_path = SOURCE / 'manifest.json'
    old = json.loads(manifest_path.read_bytes())
    body = (SOURCE / 'market_price_features.parquet').read_bytes()
    if digest(body) != old['output']['sha256']:
        raise ValueError('ACCEPTED_MARKET_SOURCE_DRIFT')
    refs = {str(manifest_path): digest(manifest_path.read_bytes()),
            str(SOURCE / 'market_price_features.parquet'): digest(body)}
    for name in ['multitimeframe_momentum_v1.py', 'materialize_multitimeframe_momentum_v1.py',
                 'test_multitimeframe_momentum_v1.py', 'tdx_index_acceptance_v1.py',
                 'contracts/MULTITIMEFRAME_MOMENTUM_SPEC_V1.json']:
        path = HERE / name
        refs[str(path)] = digest(path.read_bytes())
    frame = pd.read_parquet(io.BytesIO(body))
    result = build(frame)
    cases = []
    for symbol, part in frame.groupby('symbol'):
        times = sorted(part.decision_ts_assumed_end.unique())
        for index in sorted({80, len(times)//2, len(times)-2}):
            cutoff = int(times[index])
            prefix = build(part[part.decision_ts_assumed_end <= cutoff])
            expected = result[(result.symbol == symbol) & (result.decision_ts_assumed_end <= cutoff)].reset_index(drop=True)
            pd.testing.assert_frame_equal(prefix, expected)
            changed = part.copy()
            changed.loc[changed.decision_ts_assumed_end > cutoff, 'close'] *= 1.2
            other = build(changed)
            pd.testing.assert_frame_equal(other[other.decision_ts_assumed_end <= cutoff].reset_index(drop=True), expected)
            cases.append(dict(symbol=symbol, cutoff=cutoff, prefix='PASS', future_perturbation='PASS'))
    numeric = result.select_dtypes(include='number')
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError('NONFINITE_OUTPUT')
    if result.training_admitted.any() or result.confirmed_divergence.notna().any():
        raise ValueError('UNAPPROVED_ADMISSION')
    buffer = io.BytesIO()
    result.to_parquet(buffer, index=False, compression='zstd')
    payload = buffer.getvalue()
    report = dict(status='MATERIALIZED_PROPOSED_MARKET_MOMENTUM_NOT_TRAINING_READY', inputs=refs,
                  parameters=PARAMETERS, initialization='EMA_SMA_SEED; WILDER_RSI; FULL_SMA_BBI; GAP_RESETS_ALL',
                  histogram='2*(DIF-DEA)', rows=len(result), frequencies=[5, 15],
                  acceptance_cases=cases, http_requests=0, fit_run=False, training_ready=False,
                  divergence_materialized=False, output=dict(path='market_momentum.parquet', bytes=len(payload), sha256=digest(payload)),
                  summary=[])
    for (symbol, step), data in result.groupby(['symbol', 'frequency_minutes']):
        report['summary'].append(dict(symbol=symbol, frequency_minutes=int(step), rows=len(data),
            nonnull={c:int(data[c].notna().sum()) for c in data if c.startswith(('intraday_', 'prior_daily_', 'provisional_daily_'))}))
    metadata = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2).encode()
    if len(payload) + len(metadata) + RESERVE > CAP or bytes_used() + len(payload) + len(metadata) + RESERVE > TOTAL_CAP:
        raise ValueError('RESEARCH_BUDGET_STOP')
    for name, sha in refs.items():
        if digest(Path(name).read_bytes()) != sha:
            raise ValueError('INPUT_CHANGED_DURING_BUILD')
    OUT.mkdir(exist_ok=False)
    for name, content in [('market_momentum.parquet', payload), ('manifest.json', metadata)]:
        with (OUT / name).open('xb') as stream:
            stream.write(content)
    reread = pd.read_parquet(OUT / 'market_momentum.parquet')
    pd.testing.assert_frame_equal(reread, result)
    assert digest((OUT / 'market_momentum.parquet').read_bytes()) == report['output']['sha256']
    print(json.dumps(dict(status=report['status'], rows=len(result), bytes=len(payload)+len(metadata), acceptance_cases=len(cases))))


if __name__ == '__main__':
    run()
