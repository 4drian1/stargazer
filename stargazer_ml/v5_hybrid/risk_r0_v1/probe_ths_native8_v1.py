"""One approved localhost-only eight-key probe; no retries or source expansion."""
import hashlib
import json
import math
from pathlib import Path
import time
from datetime import datetime, timezone
from urllib.parse import quote
from urllib.request import build_opener, ProxyHandler, HTTPRedirectHandler, Request

import msgpack

HERE = Path(__file__).resolve().parent
PLAN = HERE/'ths_full_concept_scope_20261017_v1.json'
OUT = HERE/'ths_native8_probe_20261017_v1'
RESPONSE_CAP = 4*1024**2
DISK_CAP = 8*1024**2
RESERVE = 64*1024


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')


def digest(body):
    return hashlib.sha256(body).hexdigest()


def fixed_queries():
    return [dict(board=name, catalog_code=full, table=table, query_code=code,
                 date='20250102*')
            for name, full in [('储能', '306380.TI'), ('创新药', '308014.TI')]
            for code in [full, full.split('.')[0]] for table in ['日k', '分钟k']]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def transport(query, limit):
    key = ':'.join(query[k] for k in ['table', 'query_code', 'date'])
    url = 'http://127.0.0.1:7899/?cmd=get&t='+quote(key)
    opener = build_opener(ProxyHandler({}), NoRedirect())
    req = Request(url, headers={'Accept': 'application/x-msgpack'})
    with opener.open(req, timeout=10) as response:
        # Exactly remaining allowance. Hitting it leaves EOF unproven -> terminate.
        return response.read(limit)


def summarize(body, query):
    rows = msgpack.unpackb(body, raw=False, strict_map_key=False)
    if isinstance(rows, dict) and 'data' in rows:
        rows = rows['data']
    if not isinstance(rows, list):
        raise ValueError('UNEXPECTED_RESPONSE_SHAPE')
    records = []
    for item in rows:
        row = item[1] if isinstance(item, (list, tuple)) and len(item) == 2 else item
        if not isinstance(row, dict):
            raise ValueError('UNEXPECTED_ROW_SHAPE')
        code = str(row.get('code', ''))
        if code and code not in {query['catalog_code'], query['catalog_code'].split('.')[0]}:
            raise ValueError('ASSET_CODE_MISMATCH')
        if row.get('name') and row['name'] != query['board']:
            raise ValueError('ASSET_NAME_MISMATCH')
        if not str(row.get('date', '')).startswith('20250102'):
            raise ValueError('DATE_MISMATCH_OR_UNKNOWN')
        for field in ['open', 'high', 'low', 'close']:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('INVALID_PRICE')
        records.append(row)
    return dict(rows=len(records), fields=sorted({k for r in records for k in r}),
                identity_status='EMPTY' if not records else 'REQUIRES_SOURCE_IDENTITY_REVIEW',
                native_index_admitted=False)


def execute(out=OUT, send=transport, pause=time.sleep, global_check=None,
            disk_cap=DISK_CAP, response_cap=RESPONSE_CAP):
    plan_body = PLAN.read_bytes()
    queries = fixed_queries()
    if json.loads(plan_body)['proposed_local_probe']['requests'] != queries:
        raise ValueError('FROZEN_QUERY_PLAN_MISMATCH')
    if out.exists():
        raise RuntimeError('BATCH_EXISTS_REFETCH_REFUSED')
    if global_check is None:
        from research_budget_v2 import bytes_used, TOTAL_CAP
        global_check = lambda inc: bytes_used()+inc <= TOTAL_CAP
    used = 0

    def write(name, body, terminal=False):
        nonlocal used
        allowance = 0 if terminal else RESERVE
        if used+len(body)+allowance > disk_cap or not global_check(len(body)+allowance):
            raise RuntimeError('DISK_BUDGET_STOP')
        with (out/name).open('xb') as stream:
            stream.write(body)
        used += len(body)

    binding = dict(scope='LOCAL_THS_NATIVE8', authorization='USER_CHAT_APPROVED_20261017',
                   requests=queries, request_cap=8, retries=0, response_cap=response_cap,
                   disk_cap=disk_cap, plan_sha256=digest(plan_body),
                   code_sha256=digest(Path(__file__).read_bytes()),
                   test_sha256=digest((HERE/'test_probe_ths_native8_v1.py').read_bytes()),
                   training_authorized=False)
    if len(encode(binding))+RESERVE > disk_cap or not global_check(len(encode(binding))+RESERVE):
        raise RuntimeError('PREFLIGHT_BUDGET_STOP')
    out.mkdir(exist_ok=False)
    write('binding.json', encode(binding))
    attempts, received, entries = 0, 0, []
    status, error = 'COMPLETE', None
    try:
        for query in queries:
            remaining = response_cap-received
            if remaining <= 0 or used+remaining+2*RESERVE > disk_cap or not global_check(remaining+2*RESERVE):
                raise RuntimeError('RESPONSE_OR_DISK_HEADROOM_STOP')
            if attempts:
                pause(1.0)
            attempts += 1
            entry = dict(seq=attempts, query=query,
                         started_at=datetime.now(timezone.utc).isoformat())
            entries.append(entry)
            write(f'{attempts:02d}_attempt.json', encode(entry))
            body = send(query, remaining)
            received += len(body)
            if len(body) > remaining:
                raise RuntimeError('TRANSPORT_EXCEEDED_READ_CAP')
            # Preserve even malformed / cap-reaching responses before interpretation.
            write(f'{attempts:02d}_raw.msgpack', body)
            entry.update(bytes=len(body), sha256=digest(body))
            if len(body) == remaining:
                raise RuntimeError('RESPONSE_CAP_EOF_UNPROVEN')
            entry.update(summarize(body, query))
            write(f'{attempts:02d}_result.json', encode(entry))
    except Exception as exc:
        status, error = 'FAILED', f'{type(exc).__name__}: {exc}'
    report = dict(status=status, error=error, attempts=attempts, response_bytes=received,
                  entries=entries, training_ready=False, claim='Fixed keys/date only; not full-catalog availability',
                  disk_bytes_before_stop=used, failure_response_bytes_unknown=status == 'FAILED' and len(entries) > 0 and 'bytes' not in entries[-1])
    write('stop_record.json', encode(report), terminal=True)
    return report


if __name__ == '__main__':
    result = execute()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if result['status'] == 'COMPLETE' else 2)
