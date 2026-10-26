"""Exact-cutoff research view: multi-board evidence + daily golden + leader watch.

Returns evidence, never a buy/sell decision. No source requests or model changes.
"""
import argparse
import json
import pandas as pd
import pyarrow.parquet as pq

from intraday_relative_stall_v1 import TIME, slots
from materialize_relative_stall_pilot_v1 import HERE, OUT, sha


def golden_context(market_row, states, day):
    """Only a strictly prior completed daily version. Never today's close anchors."""
    prior = [state for state in states if state['asof_day'] < day]
    state = max(prior, key=lambda item: item['asof_day']) if prior else None
    result = {'sse_golden_v2_asof_day': None, 'sse_golden_v2_low': None,
              'sse_golden_v2_high': None, 'golden_proximity_rule_frozen': False}
    if state is not None:
        low, high = state['low'], state['high']
        result.update(sse_golden_v2_asof_day=state['asof_day'],
                      sse_golden_v2_low=low, sse_golden_v2_high=high)
    for ratio in (.236, .382, .5, .618, .786):
        key = str(ratio).replace('.', '_')
        level = low + ratio * (high - low) if state is not None else None
        result['sse_golden_v2_level_' + key] = level
        result['sse_golden_v2_distance_' + key] = (
            market_row['close'] / level - 1 if level is not None else None)
    return result


def read(cutoff, benchmark='000001.SH', frequency=15):
    if frequency not in (5, 15):
        raise ValueError('FREQUENCY_MUST_BE_5_OR_15')
    if frequency == 5:
        raise ValueError('5M_SECTOR_INPUTS_NOT_MATERIALIZED_NO_15M_FALLBACK')
    if cutoff not in slots(cutoff // 1000000, frequency):
        raise ValueError('NOT_A_COMPLETED_GRID_CUTOFF')
    if benchmark not in ('000001.SH', '399006.SZ'):
        raise ValueError('UNSUPPORTED_BENCHMARK')
    report = json.loads((OUT / 'report.json').read_bytes())
    terminal = json.loads((OUT / 'stop_record.json').read_bytes())
    if report['status'] != 'COMPLETE_OBSERVATION_PILOT_NOT_MODEL_EFFECTIVENESS' or terminal['status'] != 'COMPLETE':
        raise ValueError('PILOT_NOT_COMPLETE')
    if cutoff // 1000000 not in report['days']:
        raise ValueError('DAY_OUTSIDE_PILOT_NO_FALLBACK')
    prefix = 'sse' if benchmark == '000001.SH' else 'cyb'
    names = [prefix + '_board_evidence_15m.parquet', 'leader_watch_15m.parquet']
    for name in names:
        if sha(OUT / name) != report['outputs'][name]['sha256']:
            raise ValueError('PILOT_OUTPUT_HASH_MISMATCH')
    boards, leaders = [pq.read_table(OUT / name, filters=[(TIME, '=', cutoff)]).to_pandas() for name in names]
    if len(boards) != 978:
        raise ValueError('BOARD_POPULATION_MISMATCH')
    market = HERE / 'local_market_context_20261017_v1/market_price_features.parquet'
    if sha(market) != report['inputs'][str(market.resolve())]:
        raise ValueError('MARKET_INPUT_HASH_MISMATCH')
    context = pq.read_table(market, filters=[(TIME, '=', cutoff),
                                           ('frequency_minutes', '=', frequency)]).to_pandas()
    if set(context.symbol) != {'000001.SH', '399006.SZ'} or len(context) != 2:
        raise ValueError('SAME_CUTOFF_MARKET_CONTEXT_MISSING')
    golden_dir = HERE / 'golden_anchor_replay_20261017_v2'
    golden_manifest = json.loads((golden_dir / 'manifest.json').read_bytes())
    if sha(golden_dir / 'replay.json') != golden_manifest['result_sha256']:
        raise ValueError('GOLDEN_REPLAY_HASH_MISMATCH')
    states = json.loads((golden_dir / 'replay.json').read_bytes())['states']
    golden = golden_context(context[context.symbol == '000001.SH'].iloc[0], states, cutoff // 1000000)
    record = dict(status='RESEARCH_OBSERVATION_NOT_DECISION', cutoff=cutoff,
        benchmark=benchmark, frequency_minutes=frequency, scope='ALL_ARCHIVED_978_OBSERVED_BOARDS',
        selection_topk=None, actual_historical_arrival=None, reseal_evidence='UNKNOWN',
        cross_day_capture_policy='UNCHANGED; THIS_VIEW_SHOWS_WITHIN_DAY_CAPTURE_ONLY',
        golden_replay_sha256=golden_manifest['result_sha256'], **golden)
    # Carry same-cutoff market indicators; original v1 anchor columns deliberately excluded.
    for row in context.to_dict('records'):
        prefix = 'sse' if row['symbol'] == '000001.SH' else 'cyb'
        for field in ('rsi14', 'rsi14_delta1', 'return_adjacent_bar', 'momentum_change', 'source_complete'):
            record[prefix + '_' + field] = row[field]
    boards = boards.sort_values(['observed_day_gain_rank', 'board_code'], na_position='last')
    return record, boards.reset_index(drop=True), leaders.reset_index(drop=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff', type=int, required=True)
    parser.add_argument('--benchmark', choices=['000001.SH', '399006.SZ'], default='000001.SH')
    parser.add_argument('--frequency', type=int, choices=[5, 15], default=15)
    args = parser.parse_args()
    info, boards, leaders = read(args.cutoff, args.benchmark, args.frequency)
    # Metadata/counts only: do not select a topK or emit all histories by default.
    print(pd.Series({**info, 'board_rows': len(boards), 'leader_watch_rows': len(leaders)}).to_json(force_ascii=False))
