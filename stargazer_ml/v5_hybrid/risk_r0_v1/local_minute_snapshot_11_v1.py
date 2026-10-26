"""One-shot chat-approved 11-query localhost batch. No retries, aliases or fit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

import msgpack

from minute_risk_features_v1 import parse_ts
from price_verification_guarded_run import sha
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = Path(__file__).resolve().parent
INVENTORY = HERE / 'synchronized_context_inventory_20261016_v1/inventory.json'
AUTH = HERE / 'contracts/local_minute_snapshot_11_chat_20261016_v2.json'
OUT = HERE / 'local_minute_snapshot_11_20261016_v1'
RESPONSE_CAP, DISK_CAP, RESERVE = 4*1024**2, 5*1024**2, 64*1024
BASE = 'http://127.0.0.1:7899/'


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf8')


def utc():
    return datetime.now(timezone.utc).isoformat()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('REDIRECT_REFUSED')


def transport(url, cap):
    if not url.startswith(BASE + '?cmd=get&t='):
        raise ValueError('NON_CANONICAL_URL')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(url, method='GET', headers={'Accept': 'application/json, application/msgpack'})
    try:
        response = opener.open(request, timeout=10)
    except urllib.error.HTTPError as error:
        response = error  # Error bodies consume the same budget and are archived.
    with response:
        body = response.read(cap)
        return body, response.code


def decode_rows(body, query):
    try:
        obj = json.loads(body.decode('utf8'), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    except (UnicodeDecodeError, json.JSONDecodeError):
        obj = msgpack.unpackb(body, raw=False, strict_map_key=False)
    if not isinstance(obj, list):
        raise ValueError('RESPONSE_NOT_PAIR_LIST')
    rows = []
    for pair in obj:
        if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[1], dict):
            raise ValueError('RESPONSE_PAIR_SHAPE')
        key, row = pair
        if not isinstance(key, str) or not key.startswith(f"分钟k:{query['code']}:{query['date']}"):
            raise ValueError('RESPONSE_KEY_IDENTITY')
        if row.get('code') != query['code']:
            raise ValueError('ROW_CODE_IDENTITY')
        ts = row.get('date')
        parse_ts(ts)
        if ts // 1000000 != query['date'] or key != f"分钟k:{query['code']}:{ts}":
            raise ValueError('ROW_DATE_IDENTITY')
        if query['scope'] == 'SSE_PRICE_INDEX' and not (
                row.get('name') in ('上证指数', '上证综合指数', '上证综指')):
            raise ValueError('INDEX_NAME_IDENTITY')
        values = [row.get(k) for k in ('open', 'high', 'low', 'close', 'volume')]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError('INVALID_PRICE_OR_VOLUME')
        o, h, l, c, v = values
        if min(o, h, l, c) <= 0 or v < 0 or not l <= min(o, c) <= max(o, c) <= h:
            raise ValueError('INVALID_OHLC_ORDER')
        rows.append(row)
    timestamps = {}
    for r in rows:
        value = tuple(r[k] for k in ('open', 'high', 'low', 'close', 'volume'))
        if r['date'] in timestamps and timestamps[r['date']] != value:
            raise ValueError('CONFLICTING_MINUTE_ROWS')
        timestamps[r['date']] = value
    return rows


class Batch:
    def __init__(self, out, *, disk_cap=DISK_CAP, response_cap=RESPONSE_CAP,
                 cap=11, io=transport, sleeper=time.sleep, clock=time.monotonic,
                 registered=bytes_used):
        self.out, self.disk_cap, self.response_cap, self.cap = Path(out), disk_cap, response_cap, cap
        self.io, self.sleep, self.clock, self.registered = io, sleeper, clock, registered
        self.calls, self.received, self.last = 0, 0, None

    def start(self, opening):
        if self.out.exists():
            raise ValueError('BATCH_EXISTS_NO_RESTART')
        size = len(encoded(opening))
        if size + RESERVE > self.disk_cap or self.registered() + self.disk_cap > TOTAL_CAP:
            raise ValueError('OPENING_BUDGET')
        self.out.mkdir(exist_ok=False)
        self.write('opening.json', encoded(opening))

    def disk_bytes(self):
        return sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())

    def write(self, name, body, final=False):
        reserve = 0 if final else RESERVE
        if self.disk_bytes() + len(body) + reserve > self.disk_cap or self.registered() + len(body) + reserve > TOTAL_CAP:
            raise ValueError('WRITE_BUDGET')
        with (self.out / name).open('xb') as stream:
            stream.write(body)

    def execute(self, queries):
        if (self.out / 'stop_record.json').exists():
            raise ValueError('BATCH_TERMINATED')
        results, status = [], 'completed'
        for query in queries:
            body, http_status, error, rows = b'', None, None, None
            try:
                if self.calls >= self.cap or self.received >= self.response_cap:
                    raise ValueError('REQUEST_OR_RESPONSE_CAP')
                if self.last is not None:
                    self.sleep(max(0., 1. - (self.clock() - self.last)))
                number = self.calls + 1
                self.write(f'{number:02d}_attempt.json', encoded(dict(query=query, attempted_at=utc())))
                self.calls += 1
                self.last = self.clock()
                url = BASE + '?cmd=get&t=' + urllib.parse.quote(query['key'], safe='')
                remaining = self.response_cap - self.received
                body, http_status = self.io(url, remaining)
                self.received += len(body)
                if len(body) >= remaining:
                    raise ValueError('RESPONSE_LIMIT_COMPLETENESS_UNKNOWN')
                if http_status != 200:
                    raise ValueError(f'HTTP_STATUS_{http_status}')
                rows = decode_rows(body, query)
            except Exception as exc:
                error = type(exc).__name__ + ':' + str(exc)[:200]
            item = dict(query=query, captured_at=utc(), http_status=http_status, error=error,
                bytes=len(body), sha256=hashlib.sha256(body).hexdigest(),
                result='FAIL' if error else 'PRESENT_UNCERTIFIED' if rows else 'NOT_RETURNED_FOR_THIS_KEY_DATE',
                row_count=len(rows) if rows is not None else None,
                historical_available_at=None, clock_verified=False, scale_verified=False)
            try:
                # Preflight the full raw+metadata increment, not one half of the pair.
                size = len(body) + len(encoded(item))
                if self.disk_bytes() + size + RESERVE > self.disk_cap or self.registered() + size + RESERVE > TOTAL_CAP:
                    raise ValueError('EVIDENCE_PAIR_WRITE_BUDGET')
                self.write(f'{self.calls:02d}_raw.bin', body)
                self.write(f'{self.calls:02d}_meta.json', encoded(item))
            except Exception as exc:
                item['error'] = str(exc)[:200]
                item['result'] = 'FAIL_EVIDENCE_WRITE'
            results.append(item)
            if item['error']:
                status = 'failed'
                break
        stop = dict(status=status, stopped_at=utc(), calls_started=self.calls,
            response_bytes_received=self.received, requests_cap=self.cap,
            response_cap=self.response_cap, disk_cap=self.disk_cap, reserve_not_usage=RESERVE,
            results=results, fit=False, restart_allowed=False)
        self.write('stop_record.json', encoded(stop), final=True)
        return stop


def check():
    authorization = json.loads(AUTH.read_bytes())
    inputs = authorization['inputs']
    if authorization.get('scope') != 'LOCAL_MINUTE_SNAPSHOT_11' or authorization.get('status') != 'CHAT_APPROVED':
        raise ValueError('SCOPE_OR_AUTHORIZATION')
    for ref in inputs:
        if sha(Path(ref['path'])) != ref['sha256']:
            raise ValueError('FROZEN_INPUT_DRIFT')
    required = {str(p.resolve()) for p in (INVENTORY, Path(__file__), HERE / 'test_local_minute_snapshot_11_v1.py')}
    if not required <= {str(Path(r['path']).resolve()) for r in inputs}:
        raise ValueError('INPUT_BINDING_INCOMPLETE')
    spec = json.loads(INVENTORY.read_bytes())['proposed_new_snapshot']
    queries = spec['stock_queries'] + spec['index_queries']
    if len(queries) != 11 or spec['http_request_cap'] != 11 or spec['retries'] != 0:
        raise ValueError('QUERY_SCOPE_DRIFT')
    if authorization.get('inventory_sha256') != sha(INVENTORY):
        raise ValueError('INVENTORY_BINDING')
    if OUT.exists():
        raise ValueError('BATCH_EXISTS_NO_RESTART')
    if bytes_used() + DISK_CAP > TOTAL_CAP:
        raise ValueError('REGISTERED_BUDGET')
    return authorization, queries


def run(fetch=False):
    authorization, queries = check()
    if not fetch:
        print(json.dumps(dict(check='PASS', queries=len(queries), fetch=False)))
        return
    batch = Batch(OUT)
    batch.start(dict(authorization=authorization, authorization_sha256=sha(AUTH), queries=queries,
                     limits=dict(requests=11, response=RESPONSE_CAP, disk=DISK_CAP), started_at=utc()))
    result = batch.execute(queries)
    print(json.dumps(dict(status=result['status'], calls=result['calls_started'],
                         response_bytes=result['response_bytes_received'], disk_bytes=batch.disk_bytes())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fetch', action='store_true')
    run(parser.parse_args().fetch)
