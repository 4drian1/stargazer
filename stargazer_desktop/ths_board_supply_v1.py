"""Official free THS industry constituents; invoked by explicit confirm/pool work only."""
import base64
from datetime import datetime, timedelta
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
import urllib.request

import cross_process_lock_v1 as cpl
import ths_board_identity_v1 as identity

URL = 'https://q.10jqka.com.cn/thshy/detail/code/881274/'
SEED = Path(__file__).with_name('ths_industry_source_seed_v1.json.gz')
MAX_RESPONSE = 512 * 1024
SUPPORTED = 'ths:881274'


def _text(value):
    return html.unescape(re.sub('<[^>]+>', '', value)).strip()


def parse_page(raw):
    """No guessed totals: require identity, a complete non-paged table and row audit."""
    if len(raw) > MAX_RESPONSE:
        raise ValueError('THS_RESPONSE_TOO_LARGE')
    text = raw.decode('gb18030')
    if not re.search(r'<h3>\s*影视院线\s*<span>\s*881274\s*</span>', text):
        raise ValueError('THS_BOARD_IDENTITY_MISMATCH')
    inputs = re.findall(r'<input\b[^>]*>', text, re.I)
    request = next((v for v in inputs if re.search(r'\bid=[\'"]request[\'"]', v)), '')
    pager = re.search(r'<div\b[^>]*\bid=[\'"]m-page[\'"][^>]*>(.*?)</div>', text, re.S)
    if not re.search(r'\bvalue=[\'"]\[\][\'"]', request) or pager is None or _text(pager[1]):
        raise ValueError('THS_PAGINATION_NOT_PROVEN_COMPLETE')
    body = re.findall(r'<tbody\b[^>]*>(.*?)</tbody>', text, re.S | re.I)
    if len(body) != 1:
        raise ValueError('THS_CONSTITUENT_TABLE_MISSING')
    members = []
    for number, row in enumerate(re.findall(r'<tr\b[^>]*>(.*?)</tr>', body[0], re.S | re.I), 1):
        cells = re.findall(r'<td\b[^>]*>(.*?)</td>', row, re.S | re.I)
        if len(cells) < 3 or _text(cells[0]) != str(number):
            raise ValueError('THS_CONSTITUENT_ROW_SEQUENCE')
        code = _text(cells[1])
        if not re.fullmatch(r'\d{6}', code) or not re.search(
                r'stockpage\.10jqka\.com\.cn/' + code + '/', cells[1]):
            raise ValueError('THS_CONSTITUENT_CODE_INVALID')
        members.append(code)
    if not members or len(set(members)) != len(members):
        raise ValueError('THS_CONSTITUENT_EMPTY_OR_DUPLICATE')
    breadth = re.search(r'<dt>\s*涨跌家数\s*</dt>\s*<dd>(.*?)</dd>', text, re.S)
    counts = re.findall(r'<span\b[^>]*>\s*(\d+)\s*</span>', breadth[1]) if breadth else []
    if len(counts) != 2 or sum(map(int, counts)) > len(members):
        raise ValueError('THS_CONSTITUENT_COVERAGE_MISMATCH')
    return dict(members=members, total=len(members), source_url=URL,
                source_key=SUPPORTED, complete=True,
                completeness='OFFICIAL_NO_PAGINATION_AND_CONTIGUOUS_UNIQUE_ROWS',
                raw_sha256=hashlib.sha256(raw).hexdigest(),
                raw_gzip_b64=base64.b64encode(gzip.compress(raw, mtime=0)).decode('ascii'))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Transport:
    def fetch_page(self, item, page):
        if item.get('key') != SUPPORTED or page != 0:
            raise ValueError('THS_SOURCE_NOT_SUPPORTED')
        request = urllib.request.Request(URL, headers={
            'User-Agent': 'Mozilla/5.0', 'Referer': 'https://q.10jqka.com.cn/'})
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=6) as response:
            raw = response.read(MAX_RESPONSE + 1)
        return parse_page(raw)


class _SeedTransport:
    def __init__(self, response):
        self.response = response

    def fetch_page(self, item, page):
        if item.get('key') != SUPPORTED or page != 0:
            raise ValueError('THS_SOURCE_NOT_SUPPORTED')
        return self.response


def _reserve(store_dir, size):
    from runtime_disk_limit_v1 import disk_limit, disk_usage, MONITOR_RESERVE
    runtime = Path(store_dir).parent
    if disk_usage(runtime) + 2 * size + MONITOR_RESERVE > disk_limit(runtime):
        raise ValueError('THS_CONSTITUENT_DISK_BUDGET')


def _store(item, response, store_dir, now):
    _reserve(store_dir, len(json.dumps(response).encode('utf8')) + 8192)
    return identity.fetch_constituents(item, _SeedTransport(response), store_dir,
                                       now.strftime('%Y%m%d'), now=now)


def _result(item, store_dir, reason=None):
    members, provenance = identity.constituent_record(item, store_dir)
    return dict(status='READY' if members else 'CONSTITUENTS_PENDING',
                member_count=len(members or []), provenance=provenance, reason=reason)


def ensure(token, store_dir, now=None, transport=None):
    """Prepare only the exact industry. Quote loops and GET routes never call this."""
    item = identity.resolve(token)
    if not item or item['key'] != SUPPORTED:
        return None
    now = now or datetime.now(identity.ZONE)
    folder = Path(store_dir)
    try:
        with cpl.FileLock(folder / 'industry_refresh', timeout=10):
            pointer = folder / 'constituents/ths_881274_current.json'
            # Verified source bytes ship as a dated cache, not hand-written members.
            if not pointer.exists() and SEED.is_file():
                seed = json.loads(gzip.decompress(SEED.read_bytes()))
                stamp = datetime.fromisoformat(seed['fetched_at'])
                raw = gzip.decompress(base64.b64decode(seed['raw_gzip_b64']))
                if (seed.get('source_url') != URL or stamp > now or
                        hashlib.sha256(raw).hexdigest() != seed['raw_sha256']):
                    raise ValueError('THS_SEED_PROVENANCE_INVALID')
                _store(item, parse_page(raw), folder, stamp)
            current = _result(item, folder)
            if (current['provenance'] or {}).get('day') == now.strftime('%Y%m%d'):
                return current
            attempts = folder / ('industry_attempts_' + now.strftime('%Y%m%d') + '.json')
            state = json.loads(attempts.read_bytes()) if attempts.exists() else {'attempts': 0}
            if state['attempts'] >= 3:
                return _result(item, folder, 'THS_DAILY_ATTEMPTS_EXHAUSTED')
            if state.get('at') and now - datetime.fromisoformat(state['at']) < timedelta(minutes=15):
                return _result(item, folder, state.get('reason') or 'THS_RETRY_COOLDOWN')
            state.update(attempts=state['attempts'] + 1, at=now.isoformat())
            _reserve(folder, 8192)
            identity._write_json(attempts, state)  # reserve before the HTTP attempt
            try:
                response = (transport or Transport()).fetch_page(item, 0)
                _store(item, response, folder, now)
            except Exception as exc:
                state['reason'] = type(exc).__name__ + ':' + str(exc)[:180]
                identity._write_json(attempts, state)
                return _result(item, folder, state['reason'])
            return _result(item, folder)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _result(item, folder, type(exc).__name__ + ':' + str(exc)[:180])
