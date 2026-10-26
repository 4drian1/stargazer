"""Freeze an existing archived THS directory; emit a request proposal, never send it."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'opportunity_v2/o5_sector_lag/_upstream/stockdb_boards_snapshot.parquet'
OUTPUT = HERE / 'ths_full_concept_scope_20261017_v1.json'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run():
    if OUTPUT.exists():
        raise RuntimeError('OUTPUT_EXISTS_NO_OVERWRITE')
    source_hash = digest(SOURCE.read_bytes())
    rows = [r for r in pq.read_table(SOURCE).to_pylist() if r['source'] == 'ths' and r['type'] == 'concept']
    if not rows or len({r['code'] for r in rows}) != len(rows):
        raise ValueError('EMPTY_OR_DUPLICATE_CATALOG')
    boards = [dict(code=r['code'], name=r['name'], index_alias=r['index_code'],
                   member_count=len(r['symbols'] or []),
                   member_hash=digest(json.dumps(sorted(r['symbols'] or [])).encode()),
                   source_row_reference=r['code'], historical_known_at=None)
              for r in sorted(rows, key=lambda r: r['code'])]
    requests = []
    for name in ('储能', '创新药'):
        hits = [r for r in rows if r['name'] == name]
        if len(hits) != 1:
            raise ValueError('PROBE_IDENTITY_AMBIGUOUS')
        row = hits[0]
        for code in (row['code'], row['index_code']):
            for table in ('日k', '分钟k'):
                requests.append(dict(board=name, catalog_code=row['code'], table=table,
                                     query_code=code, date='20250102*'))
    references = [SOURCE, HERE / 'intraday_sector_leaders_v2.py', HERE / 'test_intraday_sector_leaders_v2.py',
                  HERE / 'intraday_sector_leaders_v1.py', HERE.parents[2] / 'stargazer_desktop/sector_pullback.py',
                  Path(__file__)]
    result = dict(scope='FULL_THS_CONCEPT', status='ARCHIVED_CATALOG_ONLY_NOT_TRAINING_READY',
                  inspected_at_utc=datetime.now(timezone.utc).isoformat(),
                  archived_fetch_time=None, source=str(SOURCE), source_sha256=source_hash,
                  board_count=len(rows), unique_member_count=len({c for r in rows for c in r['symbols'] or []}),
                  member_references=sum(len(r['symbols'] or []) for r in rows),
                  duplicate_names=[n for n, count in Counter(r['name'] for r in rows).items() if count > 1],
                  boards=boards, catalog_id='THS_ARCHIVE_' + source_hash,
                  source_columns=pq.read_schema(SOURCE).names,
                  current_refresh_verified=False, historical_membership_verified=False,
                  not_limited_by_monitor_watchlist=True, directory_versions_not_fixed_to_978=True,
                  proposed_local_probe=dict(status='DRAFT_NOT_AUTHORIZED_NOT_EXECUTED', requests=requests,
                                            request_cap=8, retries=0, response_cap_bytes=4*1024**2,
                                            disk_cap_bytes=8*1024**2,
                                            note='Two catalog identities/aliases and one fixed date only; empty results not global absence'),
                  inputs={str(p): digest(p.read_bytes()) for p in references}, http_requests=0, training_ready=False)
    body = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    if len(body) > 2*1024**2 or bytes_used() + len(body) > TOTAL_CAP:
        raise RuntimeError('BUDGET_STOP')
    if digest(SOURCE.read_bytes()) != source_hash:
        raise ValueError('SOURCE_DRIFT')
    with OUTPUT.open('xb') as f:
        f.write(body)
    print(json.dumps(dict(boards=len(boards), member_references=result['member_references'],
                          unique_members=result['unique_member_count'], output_bytes=len(body),
                          proposed_requests=len(requests), requests_executed=0), ensure_ascii=False))


if __name__ == '__main__':
    run()
