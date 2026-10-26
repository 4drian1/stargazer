"""Local files only: candidate name bridge, not historical identity/price admission."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
THS = Path('D:/同花顺')
SCOPE = HERE/'ths_full_concept_scope_20261017_v1.json'
NAME_TABLE = THS/'stockname/stockname_48_0.txt'
OUT = HERE/'ths_desktop_local_inventory_20261017_v1.json'
OUTPUT_CAP = 512*1024


def sha(body):
    return hashlib.sha256(body).hexdigest()


def parse_name_table(body):
    """Exact primary name only; aliases/categories never inferred from numeric code."""
    by_name = defaultdict(set)
    versions = []
    rejected = []
    for line_no, line in enumerate(body.decode('gb18030').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('['):
            continue
        if line.startswith('ConfigVer='):
            versions.append(line.partition('=')[2])
            continue
        match = re.fullmatch(r'(\d{6})=(.+)', line)
        if not match:
            rejected.append(line_no)
            continue
        name = match[2].split('|')[0].strip()
        if not name:
            rejected.append(line_no)
            continue
        by_name[name].add(match[1])
    return {name: sorted(codes) for name, codes in by_name.items()}, versions, rejected


def bridge(boards, names, day_codes):
    result = []
    for board in boards:
        codes = names.get(board['name'], [])
        status = 'UNMAPPED' if not codes else ('EXACT_NAME_CANDIDATE' if len(codes) == 1 else 'AMBIGUOUS_NAME')
        result.append(dict(catalog_code=board['code'], name=board['name'],
                           desktop_code_candidates=codes, bridge_status=status,
                           cached_day_candidates=[code for code in codes if code in day_codes],
                           identity_verified=False, historical_membership_verified=False,
                           training_admitted=False))
    return result


def file_evidence(path):
    before = path.stat()
    body = path.read_bytes()
    after = path.stat()
    return dict(path=str(path), bytes=len(body), sha256=sha(body),
                header_hex=body[:16].hex(),
                stat_unchanged_during_read=(before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                market_data_format_verified=False)


def run():
    if OUT.exists():
        raise RuntimeError('OUTPUT_EXISTS_NO_OVERWRITE')
    scope_body = SCOPE.read_bytes()
    scope = json.loads(scope_body)
    if len(scope['boards']) != scope['board_count']:
        raise ValueError('CATALOG_COUNT_MISMATCH')
    # Bind the archived catalog's actual source, not a filename/date proxy.
    if sha(Path(scope['source']).read_bytes()) != scope['source_sha256']:
        raise ValueError('ARCHIVED_CATALOG_SOURCE_HASH_MISMATCH')
    name_body = NAME_TABLE.read_bytes()
    names, versions, rejected = parse_name_table(name_body)
    if rejected:
        raise ValueError(f'NAME_TABLE_UNPARSED_LINES:{rejected}')
    hist_root = THS/'history/newindx'
    historical = [file_evidence(p) for p in sorted(hist_root.rglob('*')) if p.is_file()]
    day_codes = {Path(item['path']).stem for item in historical if Path(item['path']).suffix == '.day'}
    real_time = [file_evidence(p) for directory in ['newindx', 'blockindex']
                 for p in sorted((THS/'realtime'/directory).glob('*')) if p.is_file()]
    rows = bridge(scope['boards'], names, day_codes)
    docs = [Path('D:/a股数据/free-stockdb-main/README.md'),
            Path('D:/a股数据/free-stockdb-main/docs/DATA_SOURCE.md'),
            Path('D:/a股数据/free-stockdb-main/调用方式/python/zb_test.py'),
            Path('D:/a股数据/stockdb/先看！这个！！使用说明.txt'),
            Path('D:/a股数据/stockdb/pybao/stock_sdk.py'),
            Path('D:/a股数据/free_stockdb_loader.py')]
    report = dict(status='LOCAL_SOURCE_CANDIDATES_NOT_PRICE_ADMISSION', http_requests=0,
                  catalog_id=scope['catalog_id'], catalog_count=len(rows),
                  name_table=dict(path=str(NAME_TABLE), bytes=len(name_body), sha256=sha(name_body),
                                  config_versions=versions, unique_primary_names=len(names),
                                  unique_codes=len({c for codes in names.values() for c in codes}),
                                  known_at=None, taxonomy_verified=False),
                  counts={status: sum(r['bridge_status'] == status for r in rows)
                          for status in ['EXACT_NAME_CANDIDATE', 'AMBIGUOUS_NAME', 'UNMAPPED']},
                  boards_with_cached_day_candidate=sum(bool(r['cached_day_candidates']) for r in rows),
                  historical_files=historical, realtime_files=real_time, boards=rows,
                  historical_price_range='UNVERIFIED', historical_minute_coverage='UNVERIFIED',
                  binary_format='UNVERIFIED_NOT_TDX', source_documents=[file_evidence(p) for p in docs],
                  inputs={str(SCOPE): sha(scope_body), str(Path(__file__)): sha(Path(__file__).read_bytes()),
                          str(HERE/'test_inventory_ths_desktop_local_v1.py'):
                          sha((HERE/'test_inventory_ths_desktop_local_v1.py').read_bytes())},
                  training_ready=False, next_fetch_authorized=False)
    body = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    from research_budget_v2 import TOTAL_CAP, bytes_used
    if len(body) > OUTPUT_CAP or bytes_used()+len(body) > TOTAL_CAP:
        raise RuntimeError('REGISTERED_OR_OUTPUT_BUDGET_STOP')
    with OUT.open('xb') as stream:
        stream.write(body)
    print(json.dumps(dict(counts=report['counts'], catalog_count=len(rows),
                          boards_with_cached_day_candidate=report['boards_with_cached_day_candidate'],
                          historical_file_count=len(historical), realtime_file_count=len(real_time),
                          name_table=report['name_table'], output_bytes=len(body)), ensure_ascii=False))


if __name__ == '__main__':
    run()
