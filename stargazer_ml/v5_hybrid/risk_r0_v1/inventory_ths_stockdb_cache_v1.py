"""Zero HTTP: archived THS membership vs existing stock minute file candidates.

Parquet footer hashes bind metadata only, NOT full file contents or slot coverage.
No source merging, runtime imports of fetchers, synthetic prices or training.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'green_common_profile/gcp_round_b_small_sample_v1'
PULLBACK = HERE.parents[1]/'pullback_train'
SCOPE = HERE/'ths_full_concept_scope_20261017_v1.json'
OUT = HERE/'ths_stockdb_cache_inventory_20261017_v1.json'
ROOTS = {'clean': PULLBACK/'minute_cache_clean', 'old': BASE/'minute',
         'ext': BASE/'minute_ext', 'car': BASE/'minute_car'}
CAP = 8*1024**2


def digest(body):
    return hashlib.sha256(body).hexdigest()


def board_file_coverage(boards, available):
    result = []
    for row in boards:
        members = set(row['symbols'])
        present = members & available
        result.append(dict(code=row['code'], name=row['name'], members=len(members),
                           file_candidates=len(present), file_missing=len(members-present),
                           file_candidate_ratio=len(present)/len(members) if members else None,
                           slot_coverage_verified=False, historical_membership_verified=False,
                           ranking_admitted=False))
    return result


def footer_info(path):
    """Only read the footer, never represent a metadata digest as full-content hash."""
    start = path.stat()
    with path.open('rb') as stream:
        if start.st_size < 12 or stream.read(4) != b'PAR1':
            raise ValueError('NOT_PARQUET')
        stream.seek(-8, 2)
        trailer = stream.read(8)
        length = int.from_bytes(trailer[:4], 'little')
        if trailer[4:] != b'PAR1' or length > start.st_size-12:
            raise ValueError('INVALID_PARQUET_FOOTER')
        stream.seek(-8-length, 2)
        footer_hash = digest(stream.read(length+8))
    meta = pq.ParquetFile(path)
    fields = meta.schema_arrow.names
    ranges = []
    stats_complete = True
    for group in range(meta.metadata.num_row_groups):
        columns = meta.metadata.row_group(group)
        stat = next((columns.column(i).statistics for i in range(columns.num_columns)
                     if columns.column(i).path_in_schema == 'date'), None)
        if stat is None or not stat.has_min_max or type(stat.min) is not int or type(stat.max) is not int:
            stats_complete = False
        else:
            ranges.append((stat.min, stat.max))
    end = path.stat()
    return dict(bytes=start.st_size, rows=meta.metadata.num_rows,
                footer_sha256=footer_hash, hash_scope='PARQUET_FOOTER_ONLY',
                content_hash_verified=False, fields=fields,
                date_min=min(x[0] for x in ranges) if ranges else None,
                date_max=max(x[1] for x in ranges) if ranges else None,
                date_stats_all_groups=stats_complete,
                stat_unchanged=(start.st_size, start.st_mtime_ns) == (end.st_size, end.st_mtime_ns),
                code_identity='FILENAME_ONLY_NOT_ROW_VERIFIED', slot_coverage_verified=False,
                price_scale_verified=False, arrival_time_verified=False)


def run():
    if OUT.exists():
        raise RuntimeError('OUTPUT_EXISTS_NO_OVERWRITE')
    scope_body = SCOPE.read_bytes()
    scope = json.loads(scope_body)
    source = Path(scope['source'])
    source_body = source.read_bytes()
    if digest(source_body) != scope['source_sha256']:
        raise ValueError('CATALOG_SOURCE_HASH_MISMATCH')
    boards = [r for r in pq.read_table(source).to_pylist() if r['source'] == 'ths' and r['type'] == 'concept']
    if len(boards) != scope['board_count'] or len({r['code'] for r in boards}) != len(boards):
        raise ValueError('CATALOG_COUNT_OR_IDENTITY')
    for board in boards:
        if not board['symbols'] or len(set(board['symbols'])) != len(board['symbols']):
            raise ValueError('EMPTY_OR_DUPLICATE_MEMBERS')
        frozen = next(r for r in scope['boards'] if r['code'] == board['code'])
        if digest(json.dumps(sorted(board['symbols'])).encode()) != frozen['member_hash']:
            raise ValueError('FROZEN_MEMBER_HASH_MISMATCH')
    members = {code for board in boards for code in board['symbols']}
    files, counts, errors = [], {}, []
    for label, root in ROOTS.items():
        paths = sorted(root.glob('*.parquet'))
        selected = [p for p in paths if p.stem in members and re.fullmatch(r'\d{6}', p.stem)]
        counts[label] = dict(path=str(root.resolve()), exists=root.exists(), files=len(paths), relevant_files=len(selected))
        for path in selected:
            try:
                files.append(dict(root=label, code=path.stem, **footer_info(path)))
            except Exception as exc:
                errors.append(dict(path=str(path), error=f'{type(exc).__name__}:{exc}'))
        print(json.dumps(dict(root=label, scanned=len(selected), errors=len(errors))), flush=True)
    available = {r['code'] for r in files if r['stat_unchanged']}
    missing = sorted(members-available)
    coverage = board_file_coverage(boards, available)
    evidence_paths = [HERE.parent/'opportunity_v2/o5_sector_lag/pit_membership.parquet',
                      PULLBACK/'fetch_minute_2026.py', PULLBACK/'refetch_clean_2026.py',
                      PULLBACK/'proxy_fetch.py', PULLBACK/'minute_label_engine.py',
                      HERE/'stockdb_source_v1.py', HERE/'stockdb_adjustment_v2.py']
    refs = {str(p): digest(p.read_bytes()) for p in evidence_paths}
    batch = HERE/'sector_concept_prep/energy_storage_fetch_20261017_v1'
    terminal = batch/'stop_record.json'
    stop_body = terminal.read_bytes()
    stop = json.loads(stop_body)
    raw_entries = stop['entries']
    for entry in raw_entries:
        body = (batch/entry['file']).read_bytes()
        if len(body) != entry['bytes'] or digest(body) != entry['sha256']:
            raise ValueError('OLD_RAW_BATCH_HASH_MISMATCH')
    raw_codes = {r['code'] for r in raw_entries if r['table'] == '分钟k'}
    result = dict(status='FILE_CANDIDATE_INVENTORY_NOT_TRAINING_READY', http_requests=0,
                  catalog_id=scope['catalog_id'], board_count=len(boards), member_count=len(members),
                  source=str(source), source_sha256=digest(source_body),
                  membership_time_fields=[c for c in pq.read_schema(source).names
                                          if any(t in c.lower() for t in ['date','time','known','effective'])],
                  historical_ths_membership_verified=False, current_members_backtest_authorized=False,
                  roots=counts, files=files, metadata_errors=errors, boards=coverage,
                  member_file_candidates=len(available), missing_file_codes=missing,
                  missing_by_prefix3=dict(sorted(Counter(c[:3] for c in missing).items())),
                  price_reference='PREVIOUS_SESSION_CLOSE_SAME_VERIFIED_SCALE_REQUIRED_NOT_PRE_CLOSE',
                  old_raw_batch=dict(path=str(batch), terminal_sha256=digest(stop_body),
                                     terminal_status=stop.get('status'), raw_hashes_verified=len(raw_entries),
                                     minute_codes=len(raw_codes), ths_member_codes=len(raw_codes & members),
                                     historical_ths_identity_admitted=False),
                  other_taxonomy=dict(path=str(evidence_paths[0]), columns=pq.read_schema(evidence_paths[0]).names,
                                      scope='SW_INDUSTRY_NOT_THS', substitutes_ths=False),
                  references=refs,
                  inputs={str(SCOPE): digest(scope_body), str(Path(__file__)): digest(Path(__file__).read_bytes()),
                          str(HERE/'test_inventory_ths_stockdb_cache_v1.py'):
                          digest((HERE/'test_inventory_ths_stockdb_cache_v1.py').read_bytes())},
                  training_ready=False, new_fetch_authorized=False)
    body = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
    from research_budget_v2 import bytes_used, TOTAL_CAP
    if len(body) > CAP or bytes_used()+len(body) > TOTAL_CAP:
        raise RuntimeError('OUTPUT_OR_REGISTERED_BUDGET_STOP')
    with OUT.open('xb') as stream:
        stream.write(body)
    print(json.dumps(dict(output_bytes=len(body), board_count=len(boards), members=len(members),
                          file_candidate_codes=len(available), missing_codes=len(missing),
                          boards_all_files=sum(b['file_missing'] == 0 for b in coverage),
                          old_raw_batch=result['old_raw_batch'], metadata_errors=len(errors)), ensure_ascii=False))


if __name__ == '__main__':
    run()
