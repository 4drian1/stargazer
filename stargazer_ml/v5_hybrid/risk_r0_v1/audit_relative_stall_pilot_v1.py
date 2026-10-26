"""Read back the real pilot and freeze engineering acceptance separately."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from intraday_relative_stall_v1 import TIME, board_evidence
from materialize_relative_stall_pilot_v1 import HERE, OUT, CAP, sha
from read_relative_stall_pilot_v1 import read
from research_budget_v2 import bytes_used, TOTAL_CAP

ACCEPTANCE = HERE / 'relative_stall_acceptance_20261018_v1.json'


def audit():
    if ACCEPTANCE.exists():
        raise ValueError('ACCEPTANCE_EXISTS_NO_OVERWRITE')
    report = json.loads((OUT / 'report.json').read_bytes())
    if json.loads((OUT / 'stop_record.json').read_bytes())['status'] != 'COMPLETE':
        raise ValueError('PILOT_NOT_COMPLETE')
    tests = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-m', 'unittest',
        'test_intraday_relative_stall_v1', 'test_read_relative_stall_pilot_v1',
        'test_intraday_sector_leaders_v1', 'test_build_joint_risk_features_v1',
        'test_golden_anchor_rules_v2', 'test_build_postclose_background_v1',
        'test_postclose_daily_update_v1', 'test_postclose_daily_cli_v1',
        'test_postclose_daily_job_v1', 'test_refresh_postclose_incremental_v1',
        'test_refresh_postclose_incremental_v2', 'test_read_postclose_incremental_v2',
        'test_postclose_local_source_bridge_v1', 'test_postclose_forward_offline_v1',
        'test_postclose_next_period_evaluation_v1', 'test_postclose_research_scorer_v1'],
        cwd=HERE, text=True, capture_output=True, check=True)
    if 'skipped' in tests.stderr:
        raise ValueError('REAL_TESTS_MUST_NOT_SKIP')
    for path, digest in report['inputs'].items():
        if sha(Path(path)) != digest:
            raise ValueError('INPUT_CHANGED:' + path)
    for name, item in report['outputs'].items():
        if sha(OUT / name) != item['sha256']:
            raise ValueError('OUTPUT_CHANGED:' + name)
    proof = {}
    for benchmark, prefix in [('000001.SH', 'sse'), ('399006.SZ', 'cyb')]:
        table = pq.read_table(OUT / (prefix + '_board_evidence_15m.parquet')).to_pandas()
        assert len(table) == 978 * 48
        assert not table.duplicated(['board_code', TIME]).any()
        assert table.loc[~table.full_members, 'sector_close_new_day_high'].isna().all()
        for flag, strict in [('market_rising_sector_nonpositive', False), ('market_rising_sector_falling', True)]:
            selected = table[table[flag].fillna(False)]
            assert (selected.market_interval_return > 0).all()
            assert ((selected.sector_day_return_change_common < 0) if strict else
                    (selected.sector_day_return_change_common <= 0)).all()
        market_path = HERE / 'local_market_context_20261017_v1/market_price_features.parquet'
        market = pq.read_table(market_path, filters=[('symbol', '=', benchmark),
            ('frequency_minutes', '=', 15), ('day', 'in', report['days'])]).to_pandas()
        market = market.rename(columns={'close': 'level', 'day_return_prev_close': 'day_return'})
        market['path_ok'] = market.source_complete
        # Fixed code-order samples, not selected by signal or outcome; real prefix replay.
        sample = table[table.board_code.isin(sorted(table.board_code.unique())[:3])].copy()
        columns = ['board_code', TIME, 'day_return', 'common_change', 'breadth', 'full_members']
        full = board_evidence(market, sample[columns], 15)
        cutoff = 20250103101500
        prefix_result = board_evidence(market[market[TIME] <= cutoff],
                                      sample.loc[sample[TIME] <= cutoff, columns], 15)
        expected = full[full[TIME] <= cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, prefix_result.reset_index(drop=True))
        modified = market.copy()
        modified.loc[modified[TIME] > cutoff, 'level'] *= 2
        altered = board_evidence(modified, sample[columns], 15)
        pd.testing.assert_frame_equal(expected, altered[altered[TIME] <= cutoff].reset_index(drop=True))
        compare = [c for c in full if c != 'observed_day_gain_rank']
        # Whole-population Parquet schema can be float while a tiny all-NULL
        # recomputation infers object. Normalize ONLY when both sides are NULL.
        for column in compare:
            if sample[column].isna().all() and full[column].isna().all():
                full[column] = full[column].astype(sample[column].dtype)
        pd.testing.assert_frame_equal(sample[compare].reset_index(drop=True), full[compare].reset_index(drop=True))
        info, boards, leaders = read(cutoff, benchmark)
        assert info['sse_golden_v2_asof_day'] < cutoff // 1000000
        assert len(boards) == 978 and not leaders.empty
        assert not np.isinf(table.select_dtypes(include='number').to_numpy(dtype=float)).any()
        proof[benchmark] = dict(real_rows=len(table), real_prefix_and_future_perturbation='PASS',
            context_asof_day=info['sse_golden_v2_asof_day'], observations_at_sample_cutoff=len(leaders))
    watched = pq.read_table(OUT / 'leader_watch_15m.parquet').to_pandas()
    assert not watched.duplicated(['board_code', 'code', TIME]).any()
    assert (watched.capture_ts_within_day <= watched[TIME]).all()
    assert (watched.capture_ts_within_day // 1000000 == watched[TIME] // 1000000).all()
    artifact_bytes = sum(path.stat().st_size for path in OUT.iterdir() if path.is_file())
    result = dict(status='PASS_ENGINEERING_ONLY_NOT_RISK_EFFECTIVENESS', proof=proof,
        test_output=tests.stderr, input_hashes_verified=len(report['inputs']),
        real_output_hashes_verified=len(report['outputs']), model_unchanged=True,
        pilot_bytes=artifact_bytes, prospective_records=0, http_requests=0, fit=False,
        five_minute_sector_actual=False, fifteen_minute_actual=True,
        code_refs={str(HERE / name): sha(HERE / name) for name in [
            'intraday_relative_stall_v1.py', 'materialize_relative_stall_pilot_v1.py',
            'read_relative_stall_pilot_v1.py', 'test_intraday_relative_stall_v1.py',
            'test_read_relative_stall_pilot_v1.py', Path(__file__).name]},
        report_sha256=sha(OUT / 'report.json'))
    body = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    if len(body) > 256 * 1024 or artifact_bytes + len(body) > CAP or bytes_used() + len(body) > TOTAL_CAP:
        raise ValueError('ACCEPTANCE_BUDGET_STOP')
    with ACCEPTANCE.open('xb') as stream:
        stream.write(body)
    print(json.dumps(dict(status=result['status'], pilot_bytes=artifact_bytes,
                          acceptance_bytes=len(body), proof=proof)), flush=True)
    return result


if __name__ == '__main__':
    audit()
