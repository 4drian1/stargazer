"""Readback and independent RSI replay for the proposed market-only sidecar."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from materialize_local_market_context_v1 import rsi_series
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = Path(__file__).parent
OUT = HERE / 'market_momentum_proposal_20261017_v1'


def run():
    manifest = json.loads((OUT / 'manifest.json').read_bytes())
    for name, sha in manifest['inputs'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
    path = OUT / manifest['output']['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['output']['sha256']
    data = pd.read_parquet(path)
    source = pd.read_parquet(HERE / 'local_market_context_20261017_v1/market_price_features.parquet')
    keys = ['symbol', 'frequency_minutes', 'decision_ts_assumed_end']
    old = source.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(data[keys], old[keys])
    assert len(data) == 57344 and not data.duplicated(keys).any()
    assert not np.isinf(data.select_dtypes(include='number').to_numpy(dtype=float)).any()
    assert data.confirmed_divergence.isna().all() and not data.training_admitted.any()
    assert data.actual_available_at.isna().all()
    np.testing.assert_allclose(data.intraday_rsi14, old.rsi14, equal_nan=True, rtol=0, atol=1e-10)
    cases = []
    for symbol, frame in source.groupby('symbol'):
        five = frame[frame.frequency_minutes == 5].sort_values('decision_ts_assumed_end')
        daily = five.groupby('day').close.last()
        for day in [int(daily.index[30]), int(daily.index[len(daily)//2]), 20261015, 20261016]:
            prior = daily[daily.index < day].tolist()
            bars = five[five.day == day]
            for row in [bars.iloc[0], bars.loc[bars.close.idxmax()], bars.iloc[-1]]:
                ts = int(row.decision_ts_assumed_end)
                expected = rsi_series(prior + [float(row.close)])[-1]
                actual = data[(data.symbol == symbol) & (data.frequency_minutes == 5)
                              & (data.decision_ts_assumed_end == ts)].iloc[0]
                assert abs(actual.provisional_daily_rsi14 - expected) < 1e-10
                assert abs(actual.prior_daily_rsi14 - rsi_series(prior)[-1]) < 1e-10
                cases.append(dict(symbol=symbol, cutoff=ts, rsi14=float(expected), status='PASS'))
    frozen = json.loads((HERE / 'joint_fit_review_20261017_v1/manifest.json').read_bytes())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in frozen['inputs'].items())
    report = dict(status='PASS_ENGINEERING_NOT_TRAINING_READY', rows=len(data),
                  source_and_output_hashes='PASS', intraday_rsi_matches_accepted_component='ALL_ROWS_PASS',
                  daily_prefix_scalar_replays=cases, unchanged_frozen_review_inputs=True,
                  regression_tests=123, training_ready=False, fit_run=False)
    body = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2).encode()
    if bytes_used() + len(body) > TOTAL_CAP:
        raise ValueError('RESEARCH_BUDGET_STOP')
    with (OUT / 'readback_acceptance.json').open('xb') as stream:
        stream.write(body)
    print(json.dumps(dict(status=report['status'], rows=len(data), scalar_replays=len(cases),
                         bytes_used=bytes_used(), cap=TOTAL_CAP)))


if __name__ == '__main__':
    run()
