"""Read-only same-cutoff observation/research views, not a fit or decision entry."""
import argparse
import json

import pandas as pd
import pyarrow.parquet as pq

from intraday_relative_stall_v1 import TIME, slots
from materialize_relative_stall_5m_v1 import HERE, OUT, sha
from read_relative_stall_pilot_v1 import golden_context, read as read15


def checked_report():
    report = json.loads((OUT / 'report.json').read_bytes())
    if (report['status'] != 'COMPLETE_NATIVE_5M_OBSERVATIONS_NOT_FIT'
            or json.loads((OUT / 'stop_record.json').read_bytes())['status'] != 'COMPLETE'):
        raise ValueError('NATIVE_5M_NOT_COMPLETE')
    return report


def read_output(report, name, filters=None):
    path = OUT / name
    if sha(path) != report['outputs'][name]['sha256']:
        raise ValueError('OUTPUT_HASH_MISMATCH:' + name)
    return pq.read_table(path, filters=filters).to_pandas()


def add_indicators(info):
    root = HERE / 'market_momentum_proposal_20261017_v1'
    manifest = json.loads((root / 'manifest.json').read_bytes())
    path = root / manifest['output']['path']
    if sha(path) != manifest['output']['sha256']:
        raise ValueError('MOMENTUM_HASH_MISMATCH')
    data = pq.read_table(path, filters=[(TIME, '=', info['cutoff']),
        ('frequency_minutes', '=', info['frequency_minutes'])]).to_pandas()
    if len(data) != 2 or set(data.symbol) != {'000001.SH', '399006.SZ'}:
        raise ValueError('EXACT_CUTOFF_MOMENTUM_MISSING')
    for row in data.to_dict('records'):
        prefix = 'sse' if row['symbol'] == '000001.SH' else 'cyb'
        for field in ['macd_dif_over_close', 'macd_dea_over_close', 'macd_hist_over_close',
                      'bbi_slope', 'close_over_bbi_minus_one', 'price_above_bbi']:
            info[prefix + '_' + field] = row['intraday_' + field]
    info['momentum_parameter_status'] = 'EXISTING_RESEARCH_PROPOSAL_NOT_CONFIRMED_DIVERGENCE'
    info['momentum_source_sha256'] = manifest['output']['sha256']
    return info


def read(cutoff, benchmark='000001.SH', frequency=5):
    if frequency not in (5, 15) or cutoff not in slots(cutoff // 1000000, frequency):
        raise ValueError('NOT_A_COMPLETED_FREQUENCY_CUTOFF')
    if benchmark not in ('000001.SH', '399006.SZ'):
        raise ValueError('UNSUPPORTED_BENCHMARK')
    if frequency == 15:
        info, boards, leaders = read15(cutoff, benchmark, 15)
        return add_indicators(info), boards, leaders
    report = checked_report()
    if cutoff // 1000000 not in report['days']:
        raise ValueError('DAY_OUTSIDE_PILOT_NO_FALLBACK')
    prefix = 'sse' if benchmark == '000001.SH' else 'cyb'
    boards = read_output(report, prefix + '_board_evidence_5m.parquet', [(TIME, '=', cutoff)])
    leaders = read_output(report, 'leader_watch_5m.parquet', [(TIME, '=', cutoff)])
    if len(boards) != 978:
        raise ValueError('BOARD_POPULATION_MISMATCH')
    market = HERE / 'local_market_context_20261017_v1/market_price_features.parquet'
    if sha(market) != report['inputs'][str(market.resolve())]:
        raise ValueError('MARKET_HASH_MISMATCH')
    context = pq.read_table(market, filters=[(TIME, '=', cutoff), ('frequency_minutes', '=', 5)]).to_pandas()
    if len(context) != 2 or set(context.symbol) != {'000001.SH', '399006.SZ'}:
        raise ValueError('EXACT_CUTOFF_MARKET_MISSING')
    root = HERE / 'golden_anchor_replay_20261017_v2'
    manifest = json.loads((root / 'manifest.json').read_bytes())
    if sha(root / 'replay.json') != manifest['result_sha256']:
        raise ValueError('GOLDEN_HASH_MISMATCH')
    states = json.loads((root / 'replay.json').read_bytes())['states']
    info = dict(status='RESEARCH_OBSERVATION_NOT_DECISION', cutoff=cutoff, benchmark=benchmark,
        frequency_minutes=5, scope='ALL_ARCHIVED_978_OBSERVED_BOARDS', selection_topk=None,
        actual_historical_arrival=None, reseal_evidence='UNKNOWN', source_frequency='NATIVE_1M_TO_5M',
        cross_day_capture_policy='UNCHANGED; THIS_VIEW_SHOWS_WITHIN_DAY_CAPTURE_ONLY',
        golden_replay_sha256=manifest['result_sha256'],
        **golden_context(context[context.symbol == '000001.SH'].iloc[0], states, cutoff // 1000000))
    for row in context.to_dict('records'):
        name = 'sse' if row['symbol'] == '000001.SH' else 'cyb'
        for field in ('rsi14', 'rsi14_delta1', 'return_adjacent_bar', 'momentum_change', 'source_complete'):
            info[name + '_' + field] = row[field]
    return add_indicators(info), boards.sort_values(['observed_day_gain_rank', 'board_code'], na_position='last'), leaders


def research_segment(segment):
    """Original 100-column cohort plus factored observation tables; no new fixed X.

    Labels are returned unchanged in the ORIGINAL research frame, never copied
    into observation tables. No aggregation/topK/feature whitelist is invented.
    """
    from join_momentum_review_v1 import read_segment
    report = checked_report()
    keys = read_output(report, 'cohort_source_rows.parquet')
    keys = keys[keys.segment == segment].reset_index(drop=True)
    base_path = HERE / 'joint_fit_review_20261017_v1' / (segment + '.parquet')
    if str(base_path.resolve()) not in report['inputs'] or sha(base_path) != report['inputs'][str(base_path.resolve())]:
        raise ValueError('ORIGINAL_COHORT_HASH_MISMATCH')
    original = read_segment(segment)
    base = original.iloc[keys.source_row.to_numpy(dtype=int)].reset_index(drop=True)
    pd.testing.assert_frame_equal(base[['code', TIME]], keys[['code', TIME]], check_dtype=False)
    cutoffs = keys[TIME].unique().tolist()
    views = {}
    for frequency in (5, 15):
        for prefix in ('sse', 'cyb'):
            name = prefix + '_board_evidence_' + str(frequency) + 'm.parquet'
            if frequency == 5:
                frame = read_output(report, name, [(TIME, 'in', cutoffs)])
            else:
                root = HERE / 'relative_stall_pilot_20261018_v1'
                previous = json.loads((root / 'report.json').read_bytes())
                if sha(root / name) != previous['outputs'][name]['sha256']:
                    raise ValueError('OLD_15M_OUTPUT_HASH_MISMATCH')
                frame = pq.read_table(root / name, filters=[(TIME, 'in', cutoffs)]).to_pandas()
            if len(frame) != len(cutoffs) * 978 or frame.duplicated(['board_code', TIME]).any():
                raise ValueError('EXACT_CUTOFF_FACTORED_CONTEXT_MISMATCH')
            views[prefix + '_' + str(frequency) + 'm'] = frame
    return base, views


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff', type=int, required=True)
    parser.add_argument('--frequency', type=int, choices=[5, 15], default=5)
    parser.add_argument('--benchmark', choices=['000001.SH', '399006.SZ'], default='000001.SH')
    args = parser.parse_args()
    info, boards, leaders = read(args.cutoff, args.benchmark, args.frequency)
    print(pd.Series({**info, 'board_rows': len(boards), 'leader_watch_rows': len(leaders)}).to_json(force_ascii=False))
