"""Existing three-day board/leader path replay; zero HTTP / labels / model fits."""
import io
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from continuous_motion_features_v1 import entity_motion, relative_motion, summarize_motion
from intraday_relative_stall_v1 import TIME
from materialize_relative_stall_pilot_v1 import sha
from multiboard_three_arm_fit_v1 import EvidenceWriter

HERE = Path(__file__).resolve().parent
OUT = HERE / 'continuous_motion_acceptance_20261018_v2'
CAP, RESERVE = 256*1024, 32*1024
SOURCES = {5: HERE/'relative_stall_5m_20261018_v1',
           15: HERE/'relative_stall_pilot_20261018_v1'}


def canonical(frame, prefix, known, peak):
    f = frame.copy()
    f['interval_return'] = f[prefix+'interval_return']
    f['close_drawdown'] = f[prefix+'drawdown_from_close_day_high']
    f['interval_known'], f['peak_known'] = known, peak
    return f


def independent_check(source, result, keys, frequency):
    """Scalar per-entity replay; verifies every returned non-null/unknown value."""
    indexes = [*keys, TIME]
    source = source.sort_values(indexes).reset_index(drop=True)
    pd.testing.assert_frame_equal(source[indexes], result[indexes])
    expected = np.full((len(source),3),np.nan)
    for _, group in source.groupby(keys, sort=False):
        previous = None
        for i, row in group.iterrows():
            ts = int(row[TIME]); minute = ts//10000%100*60+ts//100%100
            r = row.interval_return if pd.notna(row.interval_known) and row.interval_known else np.nan
            dd = max(0.,-row.close_drawdown) if pd.notna(row.peak_known) and row.peak_known and pd.notna(row.close_drawdown) else np.nan
            dec = np.nan
            if previous is not None:
                oldts, oldminute, oldr = previous
                if ts//1000000==oldts//1000000 and minute-oldminute==frequency and (minute<720)==(oldminute<720):
                    dec = oldr-r
            expected[i] = [r,dec,dd]
            previous = ts, minute, r
    np.testing.assert_allclose(result[['motion_interval_return','motion_deceleration','motion_giveback']].to_numpy(dtype=float),
                               expected,atol=1e-12,rtol=0,equal_nan=True)
    return int(expected.size)


def run():
    writer = EvidenceWriter(OUT, CAP, RESERVE)
    refs = {str(HERE/n):sha(HERE/n) for n in ['continuous_motion_features_v1.py',
        'test_continuous_motion_features_v1.py','verify_continuous_motion_v1.py',
        'intraday_relative_stall_v1.py','CONTINUOUS_MOTION_FEATURE_SPEC_V1.md']}
    reports = {}
    for frequency, root in SOURCES.items():
        report_path = root/'report.json'
        refs[str(report_path)] = sha(report_path)
        report = json.loads(report_path.read_bytes())
        refs[str(root/'stop_record.json')] = sha(root/'stop_record.json')
        if json.loads((root/'stop_record.json').read_bytes())['status'] != 'COMPLETE':
            raise ValueError('SOURCE_NOT_COMPLETE')
        for name in [f'{b}_board_evidence_{frequency}m.parquet' for b in ('sse','cyb')]+[f'leader_watch_{frequency}m.parquet']:
            if sha(root/name) != report['outputs'][name]['sha256']:
                raise ValueError('SOURCE_HASH_DRIFT:'+name)
            refs[str(root/name)] = sha(root/name)
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('test_continuous_motion_features_v1'))
    if not tests.wasSuccessful():
        raise ValueError(log.getvalue())
    writer.json('plan.json',dict(inputs=refs,tests=tests.testsRun,cap=CAP,
        dates=[20250102,20250103,20250106],scope='EXISTING_ALL_978_BOARD_THREE_DAY_ENGINEERING_PATHS',
        http_requests=0,fit=False,formal_training_admitted=False,full_86319_materialization=False))
    try:
        for frequency, root in SOURCES.items():
            leader = pd.read_parquet(root/f'leader_watch_{frequency}m.parquet')
            lp = canonical(leader,'',leader.path_prefix_complete,leader.path_prefix_complete)
            lm = entity_motion(lp,['board_code','code'],frequency)
            checks = independent_check(lp,lm,['board_code','code'],frequency)
            summary = dict(leader_rows=len(lm),leader_scalar_cells_checked=checks,benchmarks={})
            for benchmark in ('sse','cyb'):
                board = pd.read_parquet(root/f'{benchmark}_board_evidence_{frequency}m.parquet')
                bp = canonical(board,'sector_',board.full_members,board.sector_path_prefix_complete)
                bm = entity_motion(bp,['board_code'],frequency)
                checks = independent_check(bp,bm,['board_code'],frequency)
                cols = [TIME,'market_interval_return','market_drawdown_from_close_day_high','market_path_prefix_complete']
                market = board[cols].drop_duplicates()
                if market.duplicated(TIME).any():
                    raise ValueError('CONFLICTING_MARKET_PATH')
                market['code'] = benchmark
                mp = canonical(market,'market_',market.market_path_prefix_complete,market.market_path_prefix_complete)
                mm = entity_motion(mp,['code'],frequency)
                features = relative_motion(bm,mm,['board_code'])
                all_rows = features.merge(board[['board_code',TIME,'day_return']],on=['board_code',TIME],validate='one_to_one')
                selected = all_rows.day_return.gt(0).to_numpy()
                rising_slow = ((all_rows.motion_interval_return>0)&(all_rows.motion_deceleration>0))
                relative_weak = ((all_rows.motion_interval_return>0)&(all_rows.motion_relative_return<0))
                cases = {}
                for name, mask in [('rising_but_slowing',rising_slow),('rising_but_lagging',relative_weak),
                                    ('strengthening',all_rows.motion_deceleration<0),('giveback',all_rows.motion_giveback>0)]:
                    examples = all_rows.loc[mask].head(2).astype(object)
                    examples = examples.where(pd.notna(examples),None)
                    cases[name] = dict(rows=int(mask.sum()),fixed_first_examples=examples.to_dict('records'))
                lf = relative_motion(lm,mm,['board_code','code'])
                flags = leader[['board_code','code',TIME,'current_observed_leader']]
                lf = lf.merge(flags,on=['board_code','code',TIME],validate='one_to_one')
                current = lf.current_observed_leader.fillna(False).to_numpy(dtype=bool)
                # Both current and previous winners remain in the retained watch source.
                summary['benchmarks'][benchmark] = dict(board_rows=len(bm),boards=int(board.board_code.nunique()),
                    board_scalar_cells_checked=checks,all_boards=summarize_motion(all_rows,np.ones(len(all_rows),bool)),
                    positive_boards=summarize_motion(all_rows,selected),positive_unknown_day_return=int(board.day_return.isna().sum()),
                    all_watched_leaders=summarize_motion(lf,np.ones(len(lf),bool)),
                    current_observed_leaders=summarize_motion(lf,current),
                    retained_noncurrent_leaders=summarize_motion(lf,~current),cases=cases,
                    source_scale_verified_rows=int(board.price_scale_verified.fillna(False).sum()),
                    formal_training_admitted=False)
                print(json.dumps(dict(stage='continuous_motion_replayed',frequency=frequency,benchmark=benchmark,
                    board_rows=len(bm),leader_rows=len(lm))),flush=True)
            reports[str(frequency)] = summary
        for path, expected in refs.items():
            if sha(path) != expected:
                raise ValueError('INPUT_CHANGED:'+path)
        writer.json('report.json',dict(status='PASS_ENGINEERING_PATH_REPLAY_NOT_FULL_COHORT_NOT_FIT',
            tests=tests.testsRun,test_output=log.getvalue(),results=reports,
            all_86319_features_ready=False,formal_training_admitted=False,
            labels_or_risk_effect_examined=False,source_scale_limits_preserved=True,http_requests=0,fit=False))
        writer.json('manifest.json',dict(outputs=writer.outputs.copy()))
        writer.json('stop_record.json',dict(status='COMPLETE_NOT_TRAINING_READY',manifest_sha256=sha(OUT/'manifest.json')),terminal=True)
        return dict(status='COMPLETE_NOT_TRAINING_READY',bytes=writer.used,tests=tests.testsRun)
    except Exception as exc:
        writer.json('stop_record.json',dict(status='FAILED_NO_RESTART',reason=str(exc)),terminal=True)
        raise


if __name__ == '__main__':
    print(json.dumps(run()))
