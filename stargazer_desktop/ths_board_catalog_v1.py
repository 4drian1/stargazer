"""Read-only identity search and the dated local THS constituent archive."""
from functools import lru_cache
import hashlib
import html
import json
from pathlib import Path
import re
import urllib.request

BASE = 'https://q.10jqka.com.cn/'
GROUPS = {'concept': 'gn', 'industry': 'thshy'}
ARCHIVE_DAY = '20261017'


def text(markup):
    return html.unescape(re.sub('<[^>]+>', '', markup)).strip()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def read_public(url):
    limit = 2 * 1024 * 1024
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Referer': BASE})
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=5) as response:
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('THS_RESPONSE_TOO_LARGE')
    return raw


def decode(raw):
    try:
        return raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        return raw.decode('gb18030')


def heading(markup):
    matches = {(text(name), code) for name, code in re.findall(
        r'<h3\b[^>]*>([^<]+)<span\b[^>]*>\s*(\d{6})\s*</span>', markup, re.I)}
    if len(matches) != 1:
        raise ValueError('THS_BOARD_HEADING_UNVERIFIED')
    return next(iter(matches))


def detail_url(kind, web_id):
    if not re.fullmatch(r'\d{6}', web_id):
        raise ValueError('THS_BOARD_WEB_ID')
    return BASE + f'{GROUPS[kind]}/detail/code/{web_id}/'


@lru_cache(maxsize=2)
def _archive(path, sha):
    import pyarrow.parquet as pq
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('BOARD_CATALOG_SHA')
    import pyarrow as pa
    rows = pq.read_table(pa.BufferReader(raw), columns=['code', 'name', 'source', 'type', 'symbols']).to_pylist()
    return {r['code']: r for r in rows if r['source'] == 'ths' and r['type'] == 'concept'
            and re.fullmatch(r'\d{6}\.TI', r['code'])}


def archive():
    from sector_rule_scan import project_root
    root = project_root() / 'stargazer_ml/v5_hybrid'
    scope = json.loads((root / f'risk_r0_v1/ths_full_concept_scope_{ARCHIVE_DAY}_v1.json').read_bytes())
    path = root / 'opportunity_v2/o5_sector_lag/_upstream/stockdb_boards_snapshot.parquet'
    return _archive(path, scope['source_sha256'])


def archived_members(item):
    codes = item.get('legacy_codes') or []
    if item['type'] != 'concept' or len(codes) != 1:
        return None
    try:
        row = archive().get(codes[0])
    except (OSError, ValueError, ImportError):
        return None
    if row and row['name'] == item['name'] and row['code'] == item['web_id'] + '.TI':
        members = row['symbols']
        if members and all(re.fullmatch(r'\d{6}', code) for code in members):
            return sorted(set(members))
    return None


def _directory(kind):
    group = GROUPS[kind]
    url = BASE + group + '/'
    markup = decode(read_public(url))
    pattern = rf'<a\b[^>]*\bhref=[\'"](?:https?:)?(?://q\.10jqka\.com\.cn)?/{group}/detail/code/(\d{{6}})/?[\'"][^>]*>(.*?)</a>'
    return {(kind, web_id, text(label)) for web_id, label in re.findall(pattern, markup, re.I | re.S)}


def verified_items(saved):
    local = dict(saved)
    seed = Path(__file__).with_name('ths_board_verified_catalog_v1.json')
    if seed.is_file():
        for item in json.loads(seed.read_bytes()):
            local.setdefault(item['key'], item)
    return local


def search(query, saved):
    """Only rows with a confirmed official name/index/web-id binding are selectable."""
    query = query.strip().casefold()
    if not query or len(query) > 64:
        raise ValueError('请输入板块名称或六位代码')
    local = verified_items(saved)
    exact = [i for i in local.values() if query in {v.casefold() for v in (
        i['name'], i['key'], i['index_code'], i['web_id'], *i.get('legacy_codes', []))}]
    if exact:
        return exact, []

    code = query if re.fullmatch(r'\d{6}', query) else None
    label = query
    notes = []
    if code and not code.startswith('30'):
        try:
            page = decode(read_public('https://basic.10jqka.com.cn/' + code + '/'))
            title = re.search(r'<title\b[^>]*>(.*?)</title>', page, re.I | re.S)
            name = re.match(r'^(.+?)[(（]\s*' + code + r'\s*[)）]', text(title[1])) if title else None
            if name:
                label = name[1].strip().casefold()
        except (OSError, ValueError):
            notes.append('正式指数代码查询暂不可用')
    try:
        candidates = {('concept', r['code'][:6], r['name']) for r in archive().values()
                      if label in r['name'].casefold() or query in (r['code'][:6], r['code'].casefold())}
    except (OSError, ValueError, ImportError):
        candidates = set()
    if not candidates or label != query or any(label != c[2].casefold() for c in candidates):
        for kind in GROUPS:
            try:
                candidates.update(c for c in _directory(kind) if label in c[2].casefold() or code == c[1])
            except (OSError, ValueError):
                notes.append('同花顺' + ('概念' if kind == 'concept' else '行业') + '目录暂不可用')
    results = [i for i in local.values() if query in i['name'].casefold()]
    order = sorted(candidates, key=lambda c: (c[2].casefold() != label, c[2], c[0], c[1]))
    if len(order) > 8:
        notes.append('匹配较多，请输入完整名称或代码缩小范围')
    for kind, web_id, expected in order[:8]:
        url = detail_url(kind, web_id)
        try:
            name, index = heading(decode(read_public(url)))
            if name != expected or (code and code not in (web_id, index)):
                raise ValueError('THS_BOARD_IDENTITY_MISMATCH')
            results.append(dict(key='ths:' + index, name=name, type=kind, source='ths',
                                web_id=web_id, index_code=index, source_url=url,
                                legacy_codes=[web_id + '.TI'] if kind == 'concept' else []))
        except (OSError, ValueError):
            notes.append(f'{expected}（网页编号 {web_id}）身份待核验')
    mappings = {}
    for item in results:
        mappings[(item['key'], item['type'], item['web_id'])] = item
    return list(mappings.values()), list(dict.fromkeys(notes))
