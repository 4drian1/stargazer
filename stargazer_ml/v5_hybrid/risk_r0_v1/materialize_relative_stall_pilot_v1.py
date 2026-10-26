"""Read-only three-day, all-board observation pilot. No labels, HTTP or fit."""
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from intraday_relative_stall_v1 import TIME, board_evidence, close_path, leader_watch
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'joint_risk_features_20261017_v3'
OUT = HERE / 'relative_stall_pilot_20261018_v1'
DAYS = (20250102, 20250103, 20250106)  # Existing engineering dates, never outcome selected.
CAP, RESERVE = 8 * 1024**2, 256 * 1024


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def parquet_body(frame):
    stream = io.BytesIO()
    frame.to_parquet(stream, index=False, compression='zstd')
    return stream.getvalue()


def materialize():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if bytes_used() + CAP > TOTAL_CAP:
        raise ValueError('REGISTERED_BUDGET_STOP')
    OUT.mkdir(exist_ok=False)
    used, outputs, inputs = 0, {}, {}

    def write(name, body, terminal=False):
        nonlocal used
        reserve = 0 if terminal else RESERVE
        if used + len(body) + reserve > CAP or bytes_used() + len(body) + reserve > TOTAL_CAP:
            raise ValueError('SERIALIZED_BUDGET_STOP')
        with (OUT / name).open('xb') as stream:
            stream.write(body)
        used += len(body)
        outputs[name] = {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}

    def record(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('FROZEN_INPUT_HASH_MISMATCH:' + str(path))
        inputs[str(path.resolve())] = digest

    def json_body(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')

    try:
        for path in [Path(__file__), HERE / 'intraday_relative_stall_v1.py',
                     HERE / 'test_intraday_relative_stall_v1.py', SOURCE / 'manifest.json']:
            record(path)
        model = HERE / 'postclose_repaired_research_fit_20261017_v1/model.txt'
        record(model)
        model_sha = inputs[str(model.resolve())]
        write('plan.json', json_body(dict(days=DAYS, frequency_minutes=15,
            scope='ALL_978_ARCHIVED_BOARDS_THREE_EXISTING_ENGINEERING_DAYS',
            new_high_definition='STRICT_INTRADAY_COMPLETED_CLOSE_RESEARCH_CANDIDATE',
            cap=CAP, reserve=RESERVE, model_sha256=model_sha, fit=False, http_requests=0)))
        manifest = json.loads((SOURCE / 'manifest.json').read_bytes())
        shared = SOURCE / 'shared_market_and_leaders.parquet'
        record(shared, manifest['output_hashes']['shared_market_and_leaders.parquet']['sha256'])
        times = pq.read_table(shared, columns=['cutoff_index', TIME]).to_pandas()
        selected = times[times[TIME].floordiv(1000000).isin(DAYS)].copy()
        indexes = selected.cutoff_index.tolist()
        board_chunks = []
        for item in manifest['board_parts']:
            path = SOURCE / item['path']
            record(path, manifest['output_hashes'][item['path']]['sha256'])
            board_chunks.append(pq.read_table(path, filters=[('cutoff_index', 'in', indexes)]).to_pandas())
        boards = pd.concat(board_chunks, ignore_index=True).merge(selected, on='cutoff_index', validate='many_to_one')
        boards = boards.rename(columns={'observed_return_raw': 'day_return',
            'common_member_return_change': 'common_change', 'observed_breadth': 'breadth'})
        boards['full_members'] = boards.present == boards.member_count
        if boards.board_code.nunique() != 978 or len(boards) != 978 * len(selected):
            raise ValueError('PILOT_POPULATION_MISMATCH')
        stock_chunks = []
        for i, item in enumerate(manifest['joint_parts']):
            path = SOURCE / item['path']
            record(path, manifest['output_hashes'][item['path']]['sha256'])
            chunk = pq.read_table(path, columns=['code', 'cutoff_index', 'stock_day_return_raw'],
                                  filters=[('cutoff_index', 'in', indexes)]).to_pandas()
            stock_chunks.append(chunk)
            if (i + 1) % 20 == 0:
                print(json.dumps({'stage': 'existing_stock_shards', 'read': i + 1}), flush=True)
        stocks = pd.concat(stock_chunks, ignore_index=True).merge(selected, on='cutoff_index', validate='many_to_one')
        stocks = stocks.rename(columns={'stock_day_return_raw': 'day_return'})
        catalog_path = HERE / 'ths_full_concept_scope_20261017_v1.json'
        record(catalog_path)
        catalog = json.loads(catalog_path.read_bytes())
        snapshot = Path(catalog['source'])
        record(snapshot, catalog['source_sha256'])
        membership = {r['code']: list(r['symbols']) for r in pq.read_table(snapshot).to_pylist()
                      if r['source'] == 'ths' and r['type'] == 'concept'}
        leaders = leader_watch(boards, membership, stocks)
        stock_paths = []
        # Follow each stock's own history, never splice successive winning stocks.
        for code, group in stocks.groupby('code', sort=True):
            path = close_path(group.assign(level=1 + group.day_return, path_ok=group.day_return.notna()), 15)
            path['code'] = code
            stock_paths.append(path)
        paths = pd.concat(stock_paths, ignore_index=True)
        leaders = leaders.merge(paths, on=['code', TIME], how='left', validate='many_to_one')
        sector_step = boards[['board_code', TIME, 'common_change']]
        leaders = leaders.merge(sector_step, on=['board_code', TIME], how='left', validate='many_to_one')
        leaders['leader_minus_sector_day_return_change'] = leaders.day_return_change - leaders.common_change
        write('leader_watch_15m.parquet', parquet_body(leaders))
        market_path = HERE / 'local_market_context_20261017_v1/market_price_features.parquet'
        market_manifest = json.loads((market_path.parent / 'manifest.json').read_bytes())
        record(market_path, market_manifest['output']['sha256'])
        market = pq.read_table(market_path, filters=[('day', 'in', list(DAYS)),
                                                    ('frequency_minutes', '=', 15)]).to_pandas()
        summaries = {}
        for symbol, prefix in [('000001.SH', 'sse'), ('399006.SZ', 'cyb')]:
            frame = market[market.symbol == symbol].rename(columns={'close': 'level',
                                                               'day_return_prev_close': 'day_return'})
            frame = frame.assign(path_ok=frame.source_complete)
            evidence = board_evidence(frame, boards, 15)
            evidence['benchmark_symbol'] = symbol
            write(prefix + '_board_evidence_15m.parquet', parquet_body(evidence))
            summaries[symbol] = {key: {'true': int(evidence[key].sum()),
                                       'unknown': int(evidence[key].isna().sum())}
                                for key in ['market_new_high_sector_not_new_high',
                                            'market_rising_sector_nonpositive', 'market_rising_sector_falling']}
        for name, item in outputs.items():
            if sha(OUT / name) != item['sha256']:
                raise ValueError('OUTPUT_READBACK_HASH_MISMATCH')
        if sha(model) != model_sha:
            raise ValueError('MODEL_CHANGED')
        report = dict(status='COMPLETE_OBSERVATION_PILOT_NOT_MODEL_EFFECTIVENESS',
            days=DAYS, boards=978, cutoffs=len(selected), board_rows=len(boards),
            stock_rows=len(stocks), stock_codes=int(stocks.code.nunique()), leader_watch_rows=len(leaders),
            observed_current_leader_rows=int(leaders.current_observed_leader.sum()),
            missing_captured_leader_quotes=int(leaders.day_return.isna().sum()),
            full_member_board_rows=int(boards.full_members.sum()), evidence_counts=summaries,
            inputs=inputs, outputs=outputs.copy(), model_unchanged=True, http_requests=0, fit=False,
            outputs_are_partial_snapshot_reconstruction=True, historical_actual_arrival=None,
            five_minute_sector_inputs_materialized=False, future1200m_labels_used=False,
            remaining=['Same-scope 5m board and leader inputs', 'Multi-board selection and ranking scope',
                       'Golden context and momentum join; same-cohort risk evaluation',
                       'Original raw lineage/scale and historical membership limitations retained'],
            disk_bytes_before_report=used)
        write('report.json', json_body(report))
        write('stop_record.json', json_body(dict(status='COMPLETE', bytes_before_stop=used)), terminal=True)
        print(json.dumps({k: report[k] for k in ['status', 'boards', 'cutoffs', 'board_rows',
                                                 'leader_watch_rows', 'evidence_counts']}), flush=True)
        return report
    except Exception as exc:
        write('stop_record.json', json_body(dict(status='FAILED', error=str(exc))), terminal=True)
        raise


if __name__ == '__main__':
    materialize()
