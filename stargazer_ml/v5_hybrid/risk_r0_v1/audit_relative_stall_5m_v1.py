"""Full regression + real 5m/old 15m equivalence, without labels/fit/HTTP."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq

from intraday_relative_stall_v1 import TIME
from intraday_relative_stall_v2 import board_evidence
from materialize_relative_stall_5m_v1 import CAP, HERE, OUT, semantic_equal, sha
from read_relative_stall_v2 import read, research_segment
from research_budget_v2 import bytes_used, TOTAL_CAP

ACCEPTANCE = HERE / 'relative_stall_5m_acceptance_20261018_v1.json'


def audit():
    if ACCEPTANCE.exists():
        raise ValueError('ACCEPTANCE_EXISTS_NO_OVERWRITE')
    report = json.loads((OUT / 'report.json').read_bytes())
    if report['status'] != 'COMPLETE_NATIVE_5M_OBSERVATIONS_NOT_FIT':
        raise ValueError('NATIVE_5M_NOT_COMPLETE')
    if json.loads((OUT / 'stop_record.json').read_bytes())['status'] != 'COMPLETE':
        raise ValueError('NATIVE_5M_NOT_TERMINAL')
    modules = ['test_intraday_relative_stall_v2', 'test_intraday_native_5m_extra_v1',
        'test_read_relative_stall_v2', 'test_intraday_relative_stall_v1',
        'test_read_relative_stall_pilot_v1', 'test_intraday_sector_leaders_v1',
        'test_build_joint_risk_features_v1', 'test_golden_anchor_rules_v2',
        'test_build_postclose_background_v1', 'test_postclose_daily_update_v1',
        'test_postclose_daily_cli_v1', 'test_postclose_daily_job_v1',
        'test_refresh_postclose_incremental_v1', 'test_refresh_postclose_incremental_v2',
        'test_read_postclose_incremental_v2', 'test_postclose_local_source_bridge_v1',
        'test_postclose_forward_offline_v1', 'test_postclose_next_period_evaluation_v1',
        'test_postclose_research_scorer_v1']
    tests = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-m', 'unittest', *modules],
        cwd=HERE, capture_output=True, text=True, check=True)
    if 'skipped' in tests.stderr:
        raise ValueError('REAL_TESTS_MUST_NOT_SKIP')
    for path, digest in report['inputs'].items():
        if sha(Path(path)) != digest:
            raise ValueError('DERIVED_INPUT_HASH_CHANGED:' + path)
    ledger = json.loads((OUT / 'raw_input_ledger.json').read_bytes())
    for row in ledger:
        if sha(Path(row['path'])) != row['sha256']:
            raise ValueError('RAW_INPUT_HASH_CHANGED:' + row['code'])
    for name, item in report['outputs'].items():
        if sha(OUT / name) != item['sha256']:
            raise ValueError('NATIVE_OUTPUT_HASH_CHANGED:' + name)
    old_root = HERE / 'relative_stall_pilot_20261018_v1'
    old_report = json.loads((old_root / 'report.json').read_bytes())
    market = pq.read_table(HERE / 'local_market_context_20261017_v1/market_price_features.parquet',
                          filters=[('day', 'in', report['days'])]).to_pandas()
    proofs = {}
    for symbol, prefix in [('000001.SH', 'sse'), ('399006.SZ', 'cyb')]:
        old_file = old_root / (prefix + '_board_evidence_15m.parquet')
        if sha(old_file) != old_report['outputs'][old_file.name]['sha256']:
            raise ValueError('OLD_15M_CHANGED')
        old = pq.read_table(old_file).to_pandas()
        previous = market[(market.symbol == symbol) & (market.frequency_minutes == 15)].rename(
            columns={'close': 'level', 'day_return_prev_close': 'day_return'})
        columns = ['board_code', TIME, 'day_return', 'common_change', 'breadth', 'full_members']
        recomputed15 = board_evidence(previous.assign(path_ok=previous.source_complete), old[columns], 15)
        semantic_equal(old, recomputed15, list(recomputed15))
        current = pq.read_table(OUT / (prefix + '_board_evidence_5m.parquet')).to_pandas()
        assert len(current) == 978 * 144
        assert not current.duplicated(['board_code', TIME]).any()
        assert current.loc[~current.full_members, 'sector_close_new_day_high'].isna().all()
        five = market[(market.symbol == symbol) & (market.frequency_minutes == 5)].rename(
            columns={'close': 'level', 'day_return_prev_close': 'day_return'})
        five = five.assign(path_ok=five.source_complete)
        cutoff = 20250103101000
        subset = current[current[TIME] <= cutoff].reset_index(drop=True)
        rebuilt = board_evidence(five[five[TIME] <= cutoff], current.loc[current[TIME] <= cutoff, columns], 5)
        semantic_equal(subset, rebuilt, list(rebuilt))
        altered_market = five.copy()
        altered_boards = current[columns].copy()
        altered_market.loc[altered_market[TIME] > cutoff, 'level'] *= 2
        altered_boards.loc[altered_boards[TIME] > cutoff, ['day_return', 'common_change']] = 10
        altered = board_evidence(altered_market, altered_boards, 5)
        semantic_equal(subset, altered[altered[TIME] <= cutoff].reset_index(drop=True), list(rebuilt))
        assert current.market_rising_sector_falling.sum() > 0
        assert not np.isinf(current.select_dtypes(include='number').to_numpy(dtype=float)).any()
        for flag, strict in [('market_rising_sector_nonpositive', False), ('market_rising_sector_falling', True)]:
            cases = current[current[flag].fillna(False)]
            assert (cases.market_interval_return > 0).all()
            assert ((cases.sector_day_return_change_common < 0) if strict else
                    (cases.sector_day_return_change_common <= 0)).all()
        info, boards, leaders = read(cutoff, symbol, 5)
        assert info['sse_golden_v2_asof_day'] < cutoff // 1000000
        assert len(boards) == 978 and not leaders.empty
        proofs[symbol] = dict(native5_rows=len(current), old15_all_board_rows_equal=len(old),
            real5_prefix_and_future_perturbation='PASS', full_member_rows=int(current.full_members.sum()),
            daily_positive_current_falling_rows=int(((current.day_return > 0) & current.market_rising_sector_falling.fillna(False)).sum()))
    base, context = research_segment('train')
    assert len(base) == 1200 and all(len(frame) == 978 * base[TIME].nunique() for frame in context.values())
    info, _, _ = read(20250103101000)
    external_context_refs = {str(path): sha(path) for path in [
        HERE / 'golden_anchor_replay_20261017_v2/replay.json',
        HERE / 'market_momentum_proposal_20261017_v1/market_momentum.parquet']}
    result = dict(status='PASS_NATIVE_5M_AND_UNCHANGED_15M_ENGINEERING_NOT_RISK_GATE',
        test_output=tests.stderr, tests_modules=modules, proofs=proofs, original_cohort_rows=len(base),
        context_tables=list(context), labels_changed=False, risk_effectiveness_evaluated=False,
        original_raw_hash_bound=report['original_raw_hash_bound'], newly_observed_raw_refs=len(ledger),
        http_requests=0, fit=False, model_unchanged=True, reseal_evidence='UNKNOWN',
        sse_golden_asof_day=info['sse_golden_v2_asof_day'], context_source_refs=external_context_refs,
        code_refs={str(HERE / name): sha(HERE / name) for name in [
            'intraday_relative_stall_v2.py', 'materialize_relative_stall_5m_v1.py',
            'read_relative_stall_v2.py', 'test_intraday_relative_stall_v2.py',
            'test_intraday_native_5m_extra_v1.py', 'test_read_relative_stall_v2.py', Path(__file__).name]},
        report_sha256=sha(OUT / 'report.json'),
        pilot_bytes=sum(path.stat().st_size for path in OUT.iterdir() if path.is_file()))
    body = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    if len(body) > 256 * 1024 or result['pilot_bytes'] + len(body) > CAP or bytes_used() + len(body) > TOTAL_CAP:
        raise ValueError('ACCEPTANCE_BUDGET_STOP')
    with ACCEPTANCE.open('xb') as stream:
        stream.write(body)
    print(json.dumps(dict(status=result['status'], pilot_bytes=result['pilot_bytes'], acceptance_bytes=len(body),
                         proofs=proofs, original_cohort_rows=len(base))), flush=True)
    return result


if __name__ == '__main__':
    audit()
