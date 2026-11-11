"""同花顺板块身份、成分版本与监控状态登记（Step2 dev，隔离于 _step2_dev/）。

依据 2026-11-09 同花顺官方页面逐项核对：
    影视院线  行业  指数代码 881274  网页编号 881274  旧目录兼容身份：待补（无）
    短剧游戏  概念  指数代码 886060  网页编号 309093  旧目录兼容身份：309093.TI
    光伏概念  概念  指数代码 885531  网页编号 301079  旧目录兼容身份：301079.TI

纪律：
- 只按精确别名（名称/指数代码/网页编号/旧目录身份）解析到同一身份，不做模糊匹配；
  禁止把"影视院线"替换成 VR影视、影视动漫或东财相似板块。
- 新增映射与成分版本独立保存在显式 store_dir，不覆盖冻结研究目录
  （stockdb_boards_snapshot.parquet / ths_full_concept_scope 只读），
  也不改变 Pyxis 冻结板块口径。
- 成分按内容 SHA 保存版本文件 + current 指针，逐页原始响应随版本归档并记 SHA；
  读取时校验身份、日期与成员哈希；版本永不覆盖。
- 87/599 只冻结归档成员仅作历史成员使用，输出注明快照日期（20261017），
  不冒充当日完整成分。
- 状态分三级返回：IDENTITY_CONFIRMED（身份已确认）/ CONSTITUENTS_PENDING（成分待准备）
  / MONITOR_ENABLED（监控已启用）。保存名称绝不直接宣称生效。
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

CONTRACT = 'STARGAZER_THS_BOARD_IDENTITY_V1'
ZONE = timezone(timedelta(hours=8))

# frozen_ref: 冻结归档成员（历史成员，非当日完整成员），独立保存于
# ths_board_frozen_members_v1.json（从 stockdb_boards_snapshot 提取并记录来源SHA，
# 不覆盖冻结研究目录）；可用 frozen_lookup 注入覆盖。
_IDENTITIES = [
    dict(key='ths:881274', name='影视院线', source='ths', type='industry',
         index_code='881274', web_id='881274', legacy_codes=[],
         frozen_ref=None, frozen_note='旧目录兼容身份待补；无冻结归档成员'),
    dict(key='ths:886060', name='短剧游戏', source='ths', type='concept',
         index_code='886060', web_id='309093', legacy_codes=['309093.TI'],
         frozen_ref='309093.TI',
         frozen_note='历史成员快照20261017（未认证当日完整成员），随本模块独立保存'),
    dict(key='ths:885531', name='光伏概念', source='ths', type='concept',
         index_code='885531', web_id='301079', legacy_codes=['301079.TI'],
         frozen_ref='301079.TI',
         frozen_note='历史成员快照20261017（未认证当日完整成员），随本模块独立保存'),
    dict(key='ths:886069', name='人形机器人', source='ths', type='concept',
         index_code='886069', web_id='309119', legacy_codes=['309119.TI'],
         frozen_ref='309119.TI',
         frozen_note='历史成员快照20261017（未认证当日完整成员），随本模块独立保存'),
]

_FROZEN_CACHE = None


def _frozen_doc():
    """冻结归档成员表（只读）；缺失返回空文档，不抛异常。"""
    global _FROZEN_CACHE
    if _FROZEN_CACHE is None:
        try:
            doc = json.loads((Path(__file__).resolve().parent /
                              'ths_board_frozen_members_v1.json').read_bytes())
            _FROZEN_CACHE = doc if doc.get('contract') == 'STARGAZER_THS_BOARD_FROZEN_MEMBERS_V1' else {}
        except (OSError, ValueError):
            _FROZEN_CACHE = {}
    return _FROZEN_CACHE


def _file_key(key):
    """Windows 文件名不允许 ':'；身份 key 落盘时替换。"""
    return str(key).replace(':', '_')


def _sha(value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')
    elif isinstance(value, str):
        value = value.encode('utf-8')
    return hashlib.sha256(value).hexdigest()


def identities():
    return [dict(item, legacy_codes=list(item['legacy_codes'])) for item in _IDENTITIES]


def _aliases(item):
    return {item['name'], item['index_code'], item['web_id'], *item['legacy_codes']}


def resolve(token):
    """精确解析名称/指数代码/网页编号/旧目录身份 → 同一身份；无模糊匹配。"""
    if not isinstance(token, str):
        return None
    token = token.strip()
    if not token:
        return None
    for item in _IDENTITIES:
        if token in _aliases(item):
            return dict(item, legacy_codes=list(item['legacy_codes']))
    return None


def describe(item):
    return (f"{item['name']}（同花顺{TYPE_LABEL.get(item['type'], item['type'])}，"
            f"指数代码 {item['index_code']}，网页编号 {item['web_id']}）")


TYPE_LABEL = {'industry': '行业', 'concept': '概念'}


# ---------------- 存储（显式 store_dir，调用方决定生产位置） ----------------

def _read_json(path, default):
    try:
        return json.loads(Path(path).read_bytes())
    except (OSError, ValueError):
        return default


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8'))
    tmp.replace(path)


def _registry(store_dir):
    return _read_json(Path(store_dir) / 'identities.json', {})


# ---------------- 成分（版本化 + 原始回执 + 出处） ----------------

def _verify_version(doc, pointer=None):
    """读取校验：身份、日期、成员哈希。不通过返回 None。"""
    if not isinstance(doc, dict) or doc.get('contract') != CONTRACT:
        return False
    members = doc.get('members')
    if not isinstance(members, list):
        return False
    if _sha(sorted(members)) != doc.get('members_sha256'):
        return False
    if pointer is not None and (
            pointer.get('contract') != CONTRACT or pointer.get('key') != doc.get('key') or
            pointer.get('day') != doc.get('day') or pointer.get('members_sha256') != doc.get('members_sha256')):
        return False
    return True


def constituent_record(item, store_dir, frozen_lookup=None):
    """返回 (members, provenance)；provenance 注明成分出处与快照日期。缺失返回 (None, None)。"""
    if not isinstance(item, dict):
        item = resolve(item) or {}
    key = item.get('key')
    pointer = _read_json(Path(store_dir) / 'constituents' / (_file_key(key) + '_current.json'), None)
    if pointer and pointer.get('key') == key:
        doc = _read_json(Path(store_dir) / 'constituents' / pointer.get('file', ''), None)
        if _verify_version(doc, pointer):
            provenance = dict(kind='FETCHED', day=doc['day'], fetched_at=doc.get('fetched_at'),
                              source=doc.get('source'), members_sha256=doc['members_sha256'],
                              raw_sha256=doc.get('raw_sha256'),
                              source_url=doc.get('source_url'), source_asof=doc.get('source_asof'),
                              note='成分抓取日期见day（非报价时间）；网页未声明成分生效日，原始响应已存证')
            return list(doc['members']), provenance
        return None, None  # 指针/版本损坏：不静默回退，按缺失如实报告
    ref = item.get('frozen_ref')
    codes = None
    if ref and frozen_lookup is not None:
        codes = frozen_lookup(ref)
    elif ref:
        codes = (_frozen_doc().get('members') or {}).get(ref)
    if codes:
        snapshot = str(_frozen_doc().get('snapshot_date') or 'UNKNOWN')
        provenance = dict(kind='FROZEN_ARCHIVE', snapshot_date=snapshot,
                          source='stockdb_boards_snapshot.parquet',
                          members_sha256=_sha(sorted(codes)),
                          note=f'历史成员快照{snapshot}（未认证当日完整成员），不作为当日完整成分')
        return [str(c).zfill(6) for c in codes], provenance
    return None, None


def constituent_codes(item, store_dir, frozen_lookup=None):
    members, _ = constituent_record(item, store_dir, frozen_lookup)
    return members


def fetch_constituents(item, transport, store_dir, day, now=None):
    """分页抓取成分，按内容SHA保存新版本（不覆盖旧版本），更新 current 指针，
    逐页原始响应随版本归档并记 SHA，回执追加 receipts.jsonl。
    transport 为可注入传输层（生产接同花顺免费接口；测试用模拟传输）。
    失败抛异常，状态保持待准备，不落盘半成品。
    """
    if not isinstance(item, dict) or not item.get('key'):
        raise ValueError('BOARD_IDENTITY_UNKNOWN')
    if not str(day).isdigit() or len(str(day)) != 8:
        raise ValueError('CONSTITUENT_DAY_INVALID')
    seen, page, pages, raw_pages = set(), 0, [], []
    total = None
    while True:
        response = transport.fetch_page(item, page)
        if not isinstance(response, dict) or not isinstance(response.get('members'), list):
            raise ValueError('CONSTITUENT_PAGE_INVALID')
        raw_pages.append(response)
        if total is None:
            total = response.get('total')
            if type(total) is not int or total < 0:
                raise ValueError('CONSTITUENT_TOTAL_MISSING')
        members = []
        for code in response['members']:
            code = str(code).zfill(6)
            if code in seen:
                continue
            seen.add(code)
            members.append(code)
        pages.append(dict(page=page, returned=len(response['members']), unique=len(members)))
        if len(seen) >= total:
            break
        if len(response['members']) == 0:
            raise ValueError('CONSTITUENT_PAGINATION_STUCK')
        page += 1
        if page > 200:
            raise ValueError('CONSTITUENT_PAGE_LIMIT')
    if len(seen) != total:
        raise ValueError('CONSTITUENT_INCOMPLETE')
    stamp = (now or datetime.now(ZONE)).strftime('%Y%m%d%H%M%S')
    members_sorted = sorted(seen)
    members_sha = _sha(members_sorted)
    raw_sha = _sha(raw_pages)
    version_file = '%s_%s_%s.json' % (_file_key(item['key']), members_sha[:12], day)
    value = dict(contract=CONTRACT, key=item['key'], name=item['name'], day=str(day),
                 fetched_at=stamp, total=total, members=members_sorted,
                 members_sha256=members_sha, raw_sha256=raw_sha, raw_pages=raw_pages,
                 source_url=raw_pages[0].get('source_url'), source_asof=None,
                 source='ths_free_interface', pagination=pages,
                 note='本地抓取的当日成分版本；与冻结研究目录版本相互独立；版本文件不覆盖')
    folder = Path(store_dir) / 'constituents'
    _write_json(folder / version_file, value)  # 内容相同则重写同版本，不覆盖其他版本
    pointer = dict(contract=CONTRACT, key=item['key'], day=str(day), file=version_file,
                   members_sha256=members_sha, raw_sha256=raw_sha, fetched_at=stamp)
    _write_json(folder / (_file_key(item['key']) + '_current.json'), pointer)
    receipt = dict(key=item['key'], day=str(day), fetched_at=stamp, contract=CONTRACT,
                   pages=pages, total=total, unique_members=len(seen),
                   version_file=version_file, members_sha256=members_sha, raw_sha256=raw_sha)
    receipt_path = folder / (_file_key(item['key']) + '_receipts.jsonl')
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False) + '\n')
    return value


# ---------------- 状态 ----------------

def set_monitor_enabled(item, store_dir, enabled, plan_ref=None):
    registry = _registry(store_dir)
    entry = registry.get(item['key'], {})
    entry.update(name=item['name'], enabled=bool(enabled),
                 enabled_at=datetime.now(ZONE).strftime('%Y%m%d%H%M%S'),
                 plan_ref=plan_ref or entry.get('plan_ref'))
    registry[item['key']] = entry
    _write_json(Path(store_dir) / 'identities.json', registry)


def monitor_status(token_or_item, store_dir, frozen_lookup=None):
    """三级状态：身份已确认 / 成分待准备 / 监控已启用。保存名称 ≠ 生效。"""
    item = token_or_item if isinstance(token_or_item, dict) else resolve(token_or_item)
    if not item:
        return dict(status='IDENTITY_UNKNOWN', item=None)
    enabled = bool((_registry(store_dir).get(item['key']) or {}).get('enabled'))
    members, provenance = constituent_record(item, store_dir, frozen_lookup)
    if members is None:
        return dict(status='CONSTITUENTS_PENDING', item=item, member_count=0, provenance=None)
    if not enabled:
        return dict(status='IDENTITY_CONFIRMED', item=item, member_count=len(members),
                    provenance=provenance)
    return dict(status='MONITOR_ENABLED', item=item, member_count=len(members),
                provenance=provenance)


def resolve_for_pool(token, store_dir, frozen_lookup=None):
    """监控消费者（强势池构建等）入口：解析身份 + 成分及出处。
    返回 dict(identity, members, provenance, status)；无法解析返回 None。"""
    item = resolve(token)
    if not item:
        return None
    members, provenance = constituent_record(item, store_dir, frozen_lookup)
    status = monitor_status(item, store_dir, frozen_lookup)
    return dict(identity=dict(key=item['key'], name=item['name'], source=item['source'],
                              type=item['type'], index_code=item['index_code'], web_id=item['web_id']),
                members=members, provenance=provenance, member_count=len(members) if members else 0,
                status=status['status'])


# ---------------- 盘中报价汇总（复用既有免费批量报价格式） ----------------

def _breadth(rows):
    valid = [r for r in rows if r.get('pct') is not None]
    return dict(valid_count=len(valid), total_count=len(rows),
                coverage='%d/%d' % (len(valid), len(rows)),
                up_count=sum(1 for r in valid if r['pct'] > 0),
                down_count=sum(1 for r in valid if r['pct'] < 0),
                mean_pct=(sum(r['pct'] for r in valid) / len(valid)) if valid else None)


def quote_summary(item, quotes, *, index_quote=None, members=None, pool_codes=None,
                  store_dir='.', frozen_lookup=None, now=None):
    """三个口径分别标注，互不冒充：
    index_change   同花顺指数（指数代码报价；缺项 UNKNOWN，不用成分均值冒充）
    constituents   全体成分（quote 覆盖数 valid/total，注明成分出处）
    strong_pool    冻结强势池子集（非全板块；pool_codes 为空则不存在该口径）
    """
    if not isinstance(item, dict):
        item = resolve(item) or {}
    quotes = quotes or {}
    now = now or datetime.now(ZONE)

    def _pct(q):
        if not isinstance(q, dict):
            return None
        try:
            price, prev = float(q.get('price')), float(q.get('prev'))
            stamp = datetime.strptime(str(q.get('time')), '%Y%m%d%H%M%S').replace(tzinfo=ZONE)
        except (TypeError, ValueError):
            return None
        if not (price > 0 and prev > 0) or not (0 <= (now - stamp).total_seconds() <= 120):
            return None
        return round((price / prev - 1) * 100, 2)

    index_value = _pct(index_quote)
    result = dict(identity=dict(key=item.get('key'), name=item.get('name'), source=item.get('source'),
                                type=item.get('type'), index_code=item.get('index_code'), web_id=item.get('web_id')),
                  index_change=(dict(label='同花顺指数', status='OK', pct=index_value,
                                     index_code=item.get('index_code'))
                                if index_value is not None else
                                dict(label='同花顺指数', status='UNKNOWN', pct=None,
                                     index_code=item.get('index_code'),
                                     reason='指数报价缺失、过期或不可用；不以成分均值冒充')),
                  constituents=None, strong_pool=None)
    if members is None:
        members, provenance = constituent_record(item, store_dir, frozen_lookup)
    else:
        provenance = dict(kind='CALLER_SUPPLIED', note='调用方直接提供的成员列表')
    if not members:
        result['constituents'] = dict(label='全体成分', status='CONSTITUENTS_PENDING', coverage='0/0')
        return result
    rows = [dict(code=c, pct=_pct(quotes.get(c))) for c in members]
    result['constituents'] = dict(label='全体成分', status='OK', **_breadth(rows),
                                  provenance=provenance)
    if pool_codes is not None:
        pool_rows = [dict(code=c, pct=_pct(quotes.get(c))) for c in pool_codes]
        result['strong_pool'] = dict(label='强势池（冻结池内，非全板块）', status='OK', **_breadth(pool_rows))
    return result
