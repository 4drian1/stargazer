"""仅追加/移除监控板块的车道：不依赖重新选股、个股风险准备或 DeepSeek。

修复背景：原 plan_work → plan_chat 分流后，plan_chat 开头强制
snapshot() + require_next_day_zones + plan_expired，超过观察日 09:30 时
"仅改板块"与"重新制定个股计划"走同一条校验，板块-only 请求被
"名单未就绪或超过观察日09:30"误杀。

本模块职责：
- parse：先识别完整板块引用（含括号内编号）与动作，再检查剩余内容；
  混合（含个股代码/个股名称移出/重新制定/扫描/否定句）返回 None，
  由原 plan_chat 继续校验（09:30 门保留，不删除）。同句可加可减按分句拆分；
  无法安全拆分的动作交回原流程，不静默忽略。
- preview：解析板块身份（同花顺名称/指数代码/网页编号/旧目录身份 → 同一身份），
  生成变更预览与待确认草案；草案绑定预览时的计划基线（身份摘要）；
  保留原板块，追加去重；个股、关注位、补仓位、止损位、数量偏好、已发送记录一律不动。
- confirm：先核对基线（计划已更新则要求重新预览）；有效计划只改写 boards 键，
  正式计划+助手计划+监控登记+回执事务化写入（失败回滚，不留部分成功）；
  过期/无计划只保存后续板块偏好并明确提示尚未生效，不延长旧计划、不恢复旧个股提醒。
- 有效期按生产实际契约判断：stocks[].valid_until（取最大）；仅板块计划无个股，
  回退 monitor_date+150000。不读不存在的顶层 plan.valid_until。
"""
from datetime import datetime, timedelta, timezone
import copy
import hashlib
import json
import re
from pathlib import Path

import ths_board_identity_v1 as ths

ZONE = timezone(timedelta(hours=8))
CONTRACT = 'STARGAZER_PLAN_BOARD_CHANGE_V1'

_STOCK_CODE = re.compile(r'(?<![\d.])(?:00|30|60|68)\d{4}(?![\d.])')
_ADD = r'加入|添加|加进|加到|加上|纳入|增补|新增|放入|放进'
_REMOVE = r'移出|移走|拿掉|移除|删除|剔除|去掉|不再观察|取消关注'
_NEGATED_ADD = r'(?:不要|不用|无需|别|不想|暂不|先不)\s*(?:要|用|需|想)?\s*(?:' + _ADD + ')'
_NON_BOARD_WORK = r'重新制定|重选|选股|评分|模型|多少只|几只|数量|持仓|买入|卖出|补仓|止损'
_SCAN_ONLY = r'扫描|扫一下|筛一遍|筛选|有没有|是否符合.*规则'
_SEGMENT_SPLIT = re.compile(r'[，。；;！？\n]|\s*同时\s*|\s*并且\s*|\s*而且\s*|\s*然后\s*|\s*接着\s*|\s*并\s*')


def _now14(now=None):
    return (now or datetime.now(ZONE)).strftime('%Y%m%d%H%M%S')


# ---------------- 草案记录（sha256 命名 + 摘要校验） ----------------

def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _root(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def store(root, value):
    identifier = digest(value)
    file = _root(root) / ('plan_%s.json' % identifier)
    if not file.exists():
        file.write_bytes(_encoded(value))
    return {'id': identifier, **value}


def load_draft(root, identifier):
    if not re.fullmatch('[0-9a-f]{64}', identifier or ''):
        raise ValueError('草案编号无效')
    value = json.loads((_root(root) / ('plan_%s.json' % identifier)).read_bytes())
    if digest(value) != identifier:
        raise ValueError('草案哈希不符，不允许确认')
    return value


def latest(root):
    files = sorted(_root(root).glob('plan_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    identifier = files[0].stem.split('_', 1)[1]
    return {'id': identifier, **load_draft(root, identifier)}


# ---------------- 意图识别：先取板块引用与动作，再查剩余内容 ----------------

def _board_ref_spans(message):
    """板块引用在消息中的区间（含紧邻的（编号）括号），用于剔除后检查剩余内容。"""
    spans = []
    for item in ths.identities():
        for alias in {item['name'], item['index_code'], item['web_id'], *item['legacy_codes']}:
            if not alias or alias not in message:
                continue
            start = message.index(alias)
            end = start + len(alias)
            # 吞掉紧邻括号编号：（881274）/ (886060)
            while end < len(message) and message[end] in '（(':
                close = message.find('）', end)
                close2 = message.find(')', end)
                close = close if close != -1 else close2
                if close == -1:
                    break
                end = close + 1
            spans.append((start, end))
    spans.sort()
    merged = []
    for span in spans:
        if merged and span[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], span[1]))
        else:
            merged.append(span)
    return merged


def _strip_spans(message, spans):
    for start, end in reversed(spans):
        message = message[:start] + ' ' + message[end:]
    return message


def _tokens_in(text):
    found, seen = [], set()
    for item in ths.identities():
        for alias in {item['name'], item['index_code'], item['web_id'], *item['legacy_codes']}:
            if alias and alias in text and item['key'] not in seen:
                seen.add(item['key'])
                found.append(item)
                break
    return found


def parse(message):
    """板块-only → {'add':[identity], 'remove':[identity]}；否则 None（交回原流程）。"""
    if not isinstance(message, str) or not message.strip():
        return None
    if re.search(_NEGATED_ADD, message):
        return None  # 否定句（"不要加入光伏概念板块"）不是板块变更
    if re.search(_NON_BOARD_WORK, message):
        return None
    if re.search(_SCAN_ONLY, message):
        return None
    spans = _board_ref_spans(message)
    if not spans:
        return None
    remainder = _strip_spans(message, spans)
    if _STOCK_CODE.search(remainder):
        return None  # 剩余内容有裸股票代码 → 混合
    add, remove = [], []
    for segment in _SEGMENT_SPLIT.split(message):
        if not segment.strip():
            continue
        tokens = _tokens_in(segment)
        add_v = bool(re.search(_ADD, segment))
        remove_v = bool(re.search(_REMOVE, segment))
        if tokens:
            if add_v and remove_v:
                return None  # 同句对板块同时加和减，无法安全拆分 → 交回原流程
            if add_v:
                add.extend(t for t in tokens if t not in add and t not in remove)
            elif remove_v:
                remove.extend(t for t in tokens if t not in add and t not in remove)
            else:
                return None  # 板块被提及但无明确动作
        else:
            # 无板块引用的分句：任何加/减动词的目标都无法核验（可能是个股名称），
            # 不得静默忽略 → 交回原流程
            if add_v or remove_v or re.search(_STOCK_CODE, segment):
                return None
    if not add and not remove:
        return None
    if {t['key'] for t in add} & {t['key'] for t in remove}:
        return None  # 同一板块又加又减，语义矛盾 → 交回原流程
    return {'add': add, 'remove': remove}


def is_sector_only(message):
    return parse(message) is not None


def _board_key(board):
    """板块去重键：能解析身份的按身份 key，否则按名称。"""
    if isinstance(board, dict) and board.get('identity'):
        return board['identity']
    if isinstance(board, dict) and board.get('name'):
        item = ths.resolve(board['name'])
        return item['key'] if item else 'name:' + board['name'].strip()
    return None


def _normalize_boards(boards):
    kept = []
    for board in boards or []:
        if isinstance(board, dict) and isinstance(board.get('name'), str) and board['name'].strip():
            kept.append(copy.deepcopy(board))
    return kept


# ---------------- 计划基线与有效期（生产实际契约） ----------------

_BASELINE_FIELDS = ('version', 'created_ts', 'data_date', 'monitor_date',
                    'stocks', 'boards', 'assistant_plan_id', 'mode')


def plan_baseline(formal):
    """预览/确认绑定的计划基线摘要。无正式计划返回 None。"""
    if not formal:
        return None
    return digest({k: formal.get(k) for k in _BASELINE_FIELDS})


def plan_expiry_ts(formal):
    """生产契约：有效期在 stocks[].valid_until（取最大）；仅板块计划无个股，
    回退 monitor_date+150000。顶层 valid_until 不存在，不读。"""
    if not formal:
        return ''
    stamps = []
    for stock in formal.get('stocks') or []:
        value = str((stock or {}).get('valid_until', ''))
        if re.fullmatch(r'\d{14}', value):
            stamps.append(value)
    if stamps:
        return max(stamps)
    monitor_date = str(formal.get('monitor_date', ''))
    return monitor_date + '150000' if re.fullmatch(r'\d{8}', monitor_date) else ''


def plan_effective(formal, stamp=None):
    expiry = plan_expiry_ts(formal)
    if isinstance(stamp, datetime):
        stamp = stamp.astimezone(ZONE).strftime('%Y%m%d%H%M%S')
    return bool(expiry) and expiry >= (stamp or _now14())


# ---------------- 预览 ----------------

_STATUS_TEXT = {'IDENTITY_CONFIRMED': '身份已确认', 'CONSTITUENTS_PENDING': '成分待准备',
                'MONITOR_ENABLED': '监控已启用', 'IDENTITY_UNKNOWN': '身份未核验'}


def preview(message, prior_boards, root, store_dir, *, formal=None, assistant_plan_path=None):
    """生成板块变更预览并保存待确认草案（绑定当前计划基线）；不触碰已生效计划。"""
    parsed = parse(message)
    if not parsed:
        return None
    prior = _normalize_boards(prior_boards)
    prior_keys = {_board_key(b) for b in prior}
    additions = [t for t in parsed['add'] if t['key'] not in prior_keys]
    already = [t for t in parsed['add'] if t['key'] in prior_keys]
    remove_keys = {t['key'] for t in parsed['remove']}
    removed = [b for b in prior if _board_key(b) in remove_keys]
    new_boards = [b for b in prior if _board_key(b) not in remove_keys]
    for item in additions:
        new_boards.append(dict(name=item['name'], note='',
            identity=item['key'], source=item['source'], type=item['type'],
            index_code=item['index_code'], web_id=item['web_id']))
    statuses = {}
    for item in additions + parsed['remove']:
        statuses[item['key']] = ths.monitor_status(item, store_dir)['status']
    lines = ['监控板块变更预览（仅板块，不重新选股、不需要个股风险准备、不调用 DeepSeek）：']
    lines.append('确认后监控板块：' + ('、'.join(b['name'] for b in new_boards) if new_boards else '（空）'))
    if additions:
        lines.append('追加：' + '；'.join(
            ths.describe(item) + '——' + _STATUS_TEXT[statuses[item['key']]] for item in additions))
    if removed:
        lines.append('移除：' + '、'.join(b['name'] for b in removed))
    if already:
        lines.append('已存在，不重复添加：' + '、'.join(item['name'] for item in already))
    lines.append('个股、关注位、补仓位、止损位、数量偏好与已发送提醒记录均不改变。')
    lines.append('确认前不改变现行计划。' + ('' if additions or removed else '本次无实际变更。'))
    reply = '\n'.join(lines)
    assistant_baseline = None
    if assistant_plan_path and Path(assistant_plan_path).is_file():
        try:
            assistant_baseline = digest(json.loads(Path(assistant_plan_path).read_bytes()))
        except (OSError, ValueError):
            assistant_baseline = None
    draft = dict(origin='SECTOR_BOARD_CHANGE', contract=CONTRACT,
                 message=message, boards=new_boards,
                 changes={'boards_added': [i['name'] for i in additions],
                          'boards_removed': [b['name'] for b in removed],
                          'boards_kept': [b['name'] for b in prior if b not in removed]},
                 identity_status={k: _STATUS_TEXT[v] for k, v in statuses.items()},
                 plan_baseline=plan_baseline(formal),
                 assistant_plan_baseline=assistant_baseline,
                 confirmable=bool(additions or removed),
                 created_at=datetime.now(ZONE).isoformat(), reply=reply)
    result = store(root, draft)
    return {'reply': reply, 'draft': result, 'sector_only': True}


def try_sector_change(message, *, root, store_dir, formal_plan_path, assistant_plan_path=None):
    """服务端接线入口：板块-only 返回 {'reply','draft'}；其余返回 None 走原 plan_chat。"""
    if not is_sector_only(message):
        return None
    formal = None
    prior = []
    try:
        import trade_plan_v1
        formal = trade_plan_v1.load_plan(Path(formal_plan_path))
        if formal:
            prior = formal.get('boards') or []
    except Exception:
        formal = None
    if not prior:
        previous = latest(root)
        if previous and isinstance(previous.get('boards'), list):
            prior = previous['boards']
    return preview(message, prior, root, store_dir, formal=formal,
                   assistant_plan_path=assistant_plan_path)


def pending_preferences(store_dir):
    file = Path(store_dir) / 'pending_board_preference.json'
    if not file.is_file():
        return None
    value = json.loads(file.read_bytes())
    if value.get('contract') != CONTRACT or not isinstance(value.get('boards'), list):
        raise ValueError('板块偏好记录损坏，未沿用')
    return value


# ---------------- 确认（基线核对 + 事务化写入） ----------------

def _update_boards_only(file, boards):
    file = Path(file)
    if not file.is_file():
        return False
    value = json.loads(file.read_bytes())
    if not isinstance(value, dict):
        return False
    value['boards'] = copy.deepcopy(boards)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')
    tmp = file.with_suffix(file.suffix + '.tmp')
    tmp.write_bytes(raw)
    tmp.replace(file)
    return True


def stock_alert_identity(plan, store_dir=None):
    """Reuse touch identity only for a committed, otherwise identical stock plan."""
    if store_dir is None:
        import postclose_risk_runtime
        store_dir = postclose_risk_runtime.directory() / 'board_identities'
    stock_hash = digest({k: v for k, v in plan.items() if k != 'boards'})
    for file in (Path(store_dir) / 'board_changes').glob('*.json'):
        try:
            receipt = json.loads(file.read_bytes())
            parent = receipt.get('stock_alert_identity')
            if (receipt.get('contract') == CONTRACT and
                    receipt.get('assistant_stock_hash') == stock_hash and
                    isinstance(parent, str) and re.fullmatch('[0-9a-f]{64}', parent)):
                return parent
        except (OSError, ValueError, TypeError):
            continue
    return digest(plan)


def confirm(identifier, confirmed, *, root, store_dir, formal_plan_path,
            assistant_plan_path=None, now=None):
    """核对基线后事务化写入：正式计划 → 助手计划 → 监控登记 → 回执（提交标记）。
    任一步失败回滚全部，不留部分成功。过期/无计划：仅保存后续板块偏好。"""
    if confirmed is not True:
        raise ValueError('必须人工确认板块变更')
    draft = load_draft(root, identifier)
    if draft.get('origin') != 'SECTOR_BOARD_CHANGE':
        raise ValueError('该草案不是板块变更草案，请走原计划确认流程')
    if latest(root)['id'] != identifier:
        raise ValueError('已有更新版草案，请确认最新一版')
    boards = _normalize_boards(draft.get('boards'))
    stamp = _now14(now)

    import trade_plan_v1
    formal_path = Path(formal_plan_path)
    formal = trade_plan_v1.load_plan(formal_path)
    committed = Path(store_dir) / 'board_changes' / (identifier + '.json')
    if (committed.is_file() and formal and
            (formal.get('board_change') or {}).get('draft_id') == identifier and
            _normalize_boards(formal.get('boards')) == boards):
        return {'ok': True, 'duplicate': True, 'effective': plan_effective(formal, stamp),
                'boards': [b['name'] for b in boards], 'monitor_date': formal.get('monitor_date'),
                'reply': '这次板块变更已确认，未重复修改。'}
    # 基线核对：预览后计划被更新过 → 要求重新预览，禁止把旧板块列表写进新计划
    if plan_baseline(formal) != draft.get('plan_baseline'):
        raise ValueError('计划已更新，与预览时的基线不一致；请重新预览后再确认')
    if draft.get('assistant_plan_baseline') is not None:
        if not assistant_plan_path or not Path(assistant_plan_path).is_file() or \
                digest(json.loads(Path(assistant_plan_path).read_bytes())) != draft['assistant_plan_baseline']:
            raise ValueError('助手计划已更新，与预览时的基线不一致；请重新预览后再确认')

    effective = plan_effective(formal, stamp)
    if effective and _normalize_boards(formal.get('boards')) == boards:
        return {'ok': True, 'duplicate': True, 'effective': True,
                'boards': [b['name'] for b in boards], 'monitor_date': formal.get('monitor_date')}
    if effective:
        backups = []
        for path in (formal_path, Path(assistant_plan_path) if assistant_plan_path else None):
            if path and Path(path).is_file():
                backups.append((Path(path), Path(path).read_bytes()))
        for path in (Path(store_dir) / 'identities.json',
                     Path(store_dir) / 'board_changes' / (identifier + '.json')):
            backups.append((path, path.read_bytes() if path.is_file() else None))
        try:
            new_formal = copy.deepcopy(formal)
            new_formal['boards'] = copy.deepcopy(boards)
            new_formal['board_change'] = dict(draft_id=identifier, changed_at=stamp,
                                              contract=CONTRACT,
                                              note='仅板块变更；个股与已发送记录未动')
            trade_plan_v1.save_plan(new_formal, formal_path)
            if assistant_plan_path:
                _update_boards_only(assistant_plan_path, boards)
            for board in boards:
                item = ths.resolve(board.get('name') or '') if isinstance(board, dict) else None
                if item:
                    ths.set_monitor_enabled(item, store_dir, True, plan_ref=identifier)
            # 回执最后写入，作为事务提交标记；缺回执即未生效
            receipt = dict(contract=CONTRACT, draft_id=identifier, changed_at=stamp,
                           before_boards=[b.get('name') for b in _normalize_boards(formal.get('boards'))],
                           after_boards=[b['name'] for b in boards],
                           plan_baseline=draft.get('plan_baseline'),
                           monitor_date=formal.get('monitor_date'))
            if assistant_plan_path:
                original = next((json.loads(raw) for path, raw in backups
                                 if path == Path(assistant_plan_path)), None)
                if original is not None:
                    receipt['assistant_stock_hash'] = digest(
                        {k: v for k, v in original.items() if k != 'boards'})
                    receipt['stock_alert_identity'] = stock_alert_identity(original, store_dir)
            receipt_dir = Path(store_dir) / 'board_changes'
            receipt_dir.mkdir(parents=True, exist_ok=True)
            (receipt_dir / (identifier + '.json')).write_bytes(
                json.dumps(receipt, ensure_ascii=False, sort_keys=True).encode('utf-8'))
        except Exception as exc:
            for path, raw in reversed(backups):
                if raw is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(raw)
            raise ValueError('板块变更写入失败，已回滚全部修改：%s' % type(exc).__name__) from None
        missing = [b['name'] for b in boards
                   if ths.resolve(b.get('name') or '') and
                   ths.monitor_status(b['name'], store_dir)['status'] == 'CONSTITUENTS_PENDING']
        reply = ('板块变更已生效：' + '、'.join(b['name'] for b in boards) +
                 '。个股计划与已发送提醒记录未改变。' +
                 (('注意：' + '、'.join(missing) + '成分待准备，监控状态为"成分待准备"，补齐成分前不宣称全板块监控已启用。')
                  if missing else ''))
        return {'ok': True, 'effective': True, 'duplicate': False, 'receipt': receipt,
                'boards': [b['name'] for b in boards], 'monitor_date': formal.get('monitor_date'),
                'constituents_pending': missing, 'reply': reply}
    # 过期或无有效计划：只保存后续板块偏好，不延长旧计划、不恢复旧个股提醒
    preference = dict(contract=CONTRACT, boards=boards, saved_at=stamp,
                      draft_id=identifier, effective=False,
                      note='计划已过期或不存在：仅保存后续板块偏好，尚未生效；'
                           '不延长旧计划、不恢复旧个股提醒；新计划制定时沿用本偏好。')
    pref_file = Path(store_dir) / 'pending_board_preference.json'
    pref_file.parent.mkdir(parents=True, exist_ok=True)
    pref_file.write_bytes(json.dumps(preference, ensure_ascii=False, sort_keys=True).encode('utf-8'))
    return {'ok': True, 'effective': False, 'preference_saved': True,
            'boards': [b['name'] for b in boards],
            'reply': ('当前没有生效中的计划（已过期或不存在）。板块偏好已保存：' +
                      '、'.join(b['name'] for b in boards) +
                      '；尚未生效，未改动旧计划与任何个股提醒。下次制定计划时将沿用。')}
