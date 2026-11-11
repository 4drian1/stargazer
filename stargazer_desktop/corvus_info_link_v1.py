#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""corvus_info_link_v1.py — Corvus 信息证据接线层（只读适配，Stargazer 1.0）。

把 i34 展示链（i34_desktop_integration.all_records：冻结批次+运行时导入+D2信息层）
规范化为计划上下文与盘中事实表可用的信息证据，并给出"截至某请求时刻是否已可得"的
显式判定。铁律：
- 重要性只排序、不删除；信息影响解释，不等于涨跌预测；
- D2 等待 / 权威日期未匹配 / 披露日晚于请求时刻 / 旧版本已更正 → 一律 UNKNOWN，
  绝不当成当时已知事实，更不得写成"无利空"；
- 更正版本：同 source_id 下 body_version 最高且 full_body_sha256 一致者为现行版本，
  旧版保留可追溯但标注 superseded，不得引用旧版当现行信息（引用编号须区分版本）；
- 断连/缺数据 → 显式缺失原因，不冒充空集为"没有信息"。

可得性（修复后）按完整时刻核对：
- 记录若携带可靠披露时刻（disclosed_at 或 disclosure_time+authoritative_date），与请求时刻
  逐分钟比较；披露晚于请求时刻 → 当时不可知（即便同日）。
- 记录仅有日期、无可靠时刻：若披露日早于请求日 → KNOWN；若与请求同日 → 保守按待证
  （DATE_UNMATCHED），盘中/计划均不得当作当时已知事实。
- 制定计划使用"请求发生时刻"，不使用次日观察日（monitor_date）代替。
"""
import datetime as _dt
import re

_IMP_ORDER = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2, 'UNKNOWN': 3}

_STATUS_TEXT = {
    'KNOWN_AT_ASOF': '请求时刻前已披露且D2分析完成',
    'D2_PENDING': 'D2离线分析等待中（未知，不等于无利空）',
    'DISCLOSED_AFTER_ASOF': '披露时刻晚于请求时刻（当时不可知，不得当已知事实）',
    'DATE_UNMATCHED': '权威披露日期/时刻未匹配可靠（可得时间未知，保守按待证处理）',
    'SUPERSEDED': '旧版本，已被更正版本取代（可追溯，不作为现行信息）',
}


def _int_module():
    """Lazy i34 bridge import; raises when the Corvus bundle is unavailable."""
    import i34_bridge  # noqa: F401  (already configured by server on startup)
    import i34_desktop_integration as INT
    return INT


def connected():
    """Dynamic connection state for UI slots. Never raises."""
    try:
        int_mod = _int_module()
    except Exception as exc:
        return dict(ok=False, detail='Corvus 信息模型未接入：' + type(exc).__name__, counts={})
    try:
        recs = int_mod.all_records()
    except Exception as exc:
        return dict(ok=False, detail='Corvus 记录读取失败：' + str(exc)[:80], counts={})
    waiting = sum(1 for r in recs if r.get('d2_status') == 'QUEUED_FOR_OFFLINE_D2')
    merged = sum(1 for r in recs if r.get('d2_status') == 'D2_INFO_LAYER_MERGED')
    return dict(ok=True, detail='已接入 · 本地信息层（I33 分类 + D2 规则层，SHADOW）',
                counts=dict(records=len(recs), d2_waiting=waiting, d2_merged=merged))


def _date_digits(value):
    if value is None:
        return None
    digits = re.sub(r'\D', '', str(value))
    return int(digits[:8]) if len(digits) >= 8 else None


def _ymd(dt):
    return dt.year * 10000 + dt.month * 100 + dt.day


def _asof_datetime(asof):
    """Normalize asof to a datetime: accepts datetime, 8-digit date, or 14-digit timestamp."""
    if asof is None:
        return None
    if isinstance(asof, _dt.datetime):
        return asof
    s = re.sub(r'\D', '', str(asof))
    if len(s) >= 14:
        return _dt.datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]),
                            int(s[8:10]), int(s[10:12]), int(s[12:14]))
    if len(s) >= 8:
        return _dt.datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]))
    return None


def cutoff_to_datetime(cutoff):
    """Public helper: convert an int YYYYMMDDHHMMSS cutoff into a naive datetime."""
    return _asof_datetime(cutoff)


def _disc_datetime(rec):
    """Reliable disclosure datetime if the record carries one; else None."""
    iso = rec.get('disclosed_at')
    if iso:
        for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y%m%d%H%M%S', '%Y%m%d%H%M'):
            try:
                return _dt.datetime.strptime(str(iso), fmt)
            except ValueError:
                continue
    t = rec.get('disclosure_time')
    d = rec.get('authoritative_date') or rec.get('announcement_date')
    if t and d:
        nums_t = re.findall(r'\d+', str(t))
        nums_d = re.findall(r'\d+', str(d))
        if len(nums_t) >= 2 and len(nums_d) >= 3:
            try:
                return _dt.datetime(int(nums_d[0]), int(nums_d[1]), int(nums_d[2]),
                                   int(nums_t[0]), int(nums_t[1]),
                                   int(nums_t[2]) if len(nums_t) > 2 else 0)
            except ValueError:
                return None
    return None


def _record_codes(rec):
    codes = set()
    if rec.get('code'):
        codes.add(str(rec['code']))
    for c in rec.get('stocks') or []:
        if c:
            codes.add(str(c))
    return codes


def _versioned_ref(rec):
    sid = rec.get('source_id') or rec.get('id') or (rec.get('input_sha') or '')[:16]
    return 'corvus:%s@v%s' % (sid, rec.get('body_version') or 1)


def _identity(rec):
    """Full version identity: (source_id, body_version, full body sha)."""
    return (rec.get('source_id') or rec.get('id'),
            rec.get('body_version') or 1,
            rec.get('full_body_sha256') or rec.get('input_sha') or '')


def _current_map(records):
    """source_id -> current record (highest body_version; tie-break by full body sha)."""
    best = {}
    for rec in records:
        sid = rec.get('source_id') or rec.get('id')
        if not sid:
            continue
        rank = (rec.get('body_version') or 1,
                rec.get('full_body_sha256') or rec.get('input_sha') or '')
        cur = best.get(sid)
        if cur is None or rank > (cur.get('body_version') or 1,
                                  cur.get('full_body_sha256') or cur.get('input_sha') or ''):
            best[sid] = rec
    return best


def _d2_core(rec):
    """Extract D2 rule-layer core (l0 / rule_judgments / event_statuses / unknowns)."""
    layer = rec.get('info_layer') or {}
    if layer:
        return dict(l0_status=layer.get('l0_status'),
                    l0_blocking_gates=layer.get('l0_blocking_gates'),
                    event_statuses=layer.get('event_statuses'),
                    rule_judgments=layer.get('rule_judgments'),
                    unknowns=layer.get('unknowns'))
    # Fallback: derive from top-level evidence + items_to_verify.
    ev = rec.get('evidence') or []
    l0 = next((e['value'] for e in ev if e.get('kind') == 'L0_GATE'), None)
    statuses = next((e['value'] for e in ev if e.get('kind') == 'L1S_EVENT_STATUSES'), None)
    rules = [dict(rule_id=e.get('kind'), result=e.get('value'), snippet=e.get('snippet'))
             for e in ev if str(e.get('kind', '')).startswith('RULE_')]
    return dict(l0_status=l0, l0_blocking_gates=None,
                event_statuses=statuses, rule_judgments=rules,
                unknowns=rec.get('items_to_verify'))


def records_for(codes, sectors=None):
    """All record versions matching any candidate code (or plan sector)."""
    codes = {str(c) for c in codes or []}
    sectors = {str(s) for s in sectors or []}
    out = []
    for rec in _int_module().all_records():
        rec_codes = _record_codes(rec)
        rec_sectors = {str(s) for s in rec.get('sectors') or []}
        if (codes & rec_codes) or (sectors & rec_sectors):
            out.append(rec)
    return out


def _entry(rec, asof, current_map):
    sid = rec.get('source_id') or rec.get('id')
    cur = current_map.get(sid) if sid else None
    superseded = False
    if cur is not None:
        superseded = _identity(rec) != _identity(cur)
    asof_dt = _asof_datetime(asof)
    asof_date = asof_dt.date() if asof_dt else None
    disc_dt = _disc_datetime(rec)
    disc_date = _date_digits(rec.get('authoritative_date') or rec.get('announcement_date'))
    if superseded:
        status = 'SUPERSEDED'
    elif rec.get('d2_status') != 'D2_INFO_LAYER_MERGED':
        status = 'D2_PENDING'
    elif disc_dt is None:
        if disc_date is None or asof_date is None:
            status = 'DATE_UNMATCHED'
        elif disc_date < _ymd(asof_date):
            status = 'KNOWN_AT_ASOF'
        elif disc_date == _ymd(asof_date):
            # 同日但无可靠披露时刻：保守按待证（盘中/计划均不得当已知）
            status = 'DATE_UNMATCHED'
        else:
            status = 'DISCLOSED_AFTER_ASOF'
    else:
        status = 'KNOWN_AT_ASOF' if disc_dt <= asof_dt else 'DISCLOSED_AFTER_ASOF'
    evidence = ['「%s」' % (e.get('snippet') or '')[:36] for e in (rec.get('evidence') or [])[:3]]
    d2 = _d2_core(rec)
    return dict(
        ref=_versioned_ref(rec),
        input_sha=rec.get('input_sha'),
        full_body_sha=(rec.get('full_body_sha256') or '')[:16] or None,
        body_version=rec.get('body_version') or 1,
        title=rec.get('title'), importance=rec.get('importance'),
        authoritative_date=rec.get('authoritative_date'),
        disclosure_time=rec.get('disclosure_time') or rec.get('disclosed_at'),
        codes=sorted(_record_codes(rec)), sectors=sorted(rec.get('sectors') or []),
        d2_status=rec.get('d2_status'),
        items_to_verify=(rec.get('items_to_verify') or [])[:4],
        evidence=evidence,
        d2_core=d2,
        availability=status,
        availability_reason=_STATUS_TEXT[status],
    )


def listing(limit=8):
    """Neutral workbench listing: importance-first, no as-of semantics."""
    try:
        recs = list(_int_module().all_records())
    except Exception:
        return []
    recs.sort(key=lambda r: (_IMP_ORDER.get(r.get('importance'), 9),
                             str(r.get('authoritative_date') or '')))
    return [dict(title=r.get('title'), importance=r.get('importance'),
                 date=r.get('authoritative_date'),
                 d2_status=r.get('d2_status'),
                 codes=sorted(_record_codes(r)),
                 body_version=r.get('body_version') or 1)
            for r in recs[:limit]]


def assess(records, asof):
    """Normalize + availability per record; sorted important-first, current before old."""
    current = _current_map(records)
    entries = [_entry(rec, asof, current) for rec in records]
    entries.sort(key=lambda e: (_IMP_ORDER.get(e['importance'], 9),
                                e['availability'] != 'KNOWN_AT_ASOF',
                                e['availability'] == 'SUPERSEDED',
                                str(e['authoritative_date'])))
    summary = dict(n=len(entries),
                   known_at_asof=sum(1 for e in entries if e['availability'] == 'KNOWN_AT_ASOF'),
                   waiting_d2=sum(1 for e in entries if e['availability'] == 'D2_PENDING'),
                   after_asof=sum(1 for e in entries if e['availability'] == 'DISCLOSED_AFTER_ASOF'),
                   date_unmatched=sum(1 for e in entries if e['availability'] == 'DATE_UNMATCHED'),
                   superseded=sum(1 for e in entries if e['availability'] == 'SUPERSEDED'))
    return entries, summary


def plan_information_block(codes, sectors, asof):
    """Plan-request information section: entries + honest scope note.
    asof = 请求发生时刻（datetime）；缺失时回退到当前时刻。"""
    if asof is None:
        asof = _dt.datetime.now()
    scope = ('Corvus公告信息（冻结I33重要性仅排序不删除；D2规则层结论 l0_status/rule_judgments/'
             'event_statuses/unknowns 与可得时刻已逐条标注）。空/等待/日期未匹配/旧版本只能如实说明，'
             '不当作无利空，也不得当作当时已知事实。信息可影响解释与关注点，不等于涨跌预测；'
             '引用使用给定 corvus: 前缀 ref（含版本号，可区分更正前后）。')
    try:
        recs = records_for(codes, sectors)
    except Exception as exc:
        return dict(connected=False, entries=[], summary=dict(n=0),
                    note='Corvus 信息模型未接入（%s）：信息缺口显式保留，未知不等于无利空。' % type(exc).__name__,
                    scope=scope)
    entries, summary = assess(recs, asof)
    note = ('本观察日相关公告 %d 条：请求时刻前已可得且分析完成 %d；D2等待 %d；披露晚于请求时刻 %d；'
            '日期未匹配（待证） %d；旧版已被更正 %d。' % (summary['n'], summary['known_at_asof'],
            summary['waiting_d2'], summary['after_asof'], summary['date_unmatched'],
            summary['superseded'])) if entries else \
        '本观察日的候选票、人工票与计划板块暂无已导入公告：信息缺失如实保留，不当作无利空。'
    return dict(connected=True, entries=entries, summary=summary, note=note, scope=scope)


def stock_facts(code, asof, cutoff):
    """Intraday computed_facts entries for ONE stock, as-of the trade date.
    Keys are legal citation ids (info.N); KNOWN only when truly known at the request moment.
    Each value carries the D2 rule-layer core so the intraday prompt consumes the same
    structured evidence as the independent /i34 chain."""
    facts = {}
    try:
        recs = [r for r in records_for([code]) if code in _record_codes(r)]
    except Exception:
        facts['info.corvus'] = dict(asof=cutoff, text='信息模型未接入：该股公告信息未知，不等于没有利空',
                                    value=None, status='UNKNOWN')
        return facts
    if not recs:
        facts['info.stock'] = dict(asof=cutoff,
                                   text='该股暂无已导入公告信息：未知保留，不等于无利空',
                                   value=None, status='UNKNOWN')
        return facts
    current = _current_map(recs)
    entries, _summary = assess(recs, asof)
    for i, e in enumerate(entries):
        key = 'info.%d' % i
        known = e['availability'] == 'KNOWN_AT_ASOF'
        text = '[Corvus·%s·v%s] %s（%s披露；%s）' % (
            e['importance'], e['body_version'], e['title'],
            e['authoritative_date'] or '日期未匹配', e['availability_reason'])
        if e['items_to_verify']:
            text += ' 待核查：' + '；'.join(e['items_to_verify'][:3])
        if e['evidence']:
            text += ' 证据：' + '；'.join(e['evidence'][:2])
        d2 = e.get('d2_core') or {}
        if d2.get('l0_status'):
            text += ' [L0=%s' % d2['l0_status']
            if d2.get('unknowns'):
                text += '; 待核%d项' % len(d2['unknowns'])
            text += ']'
        facts[key] = dict(asof=cutoff, text=text,
                          value=dict(source=e['ref'], input_sha=e['input_sha'],
                                     body_version=e['body_version'],
                                     availability=e['availability'],
                                     d2_core=d2),
                          status='KNOWN' if known else 'UNKNOWN')
    return facts


def stock_summary(code, asof):
    """Compact outer-facts summary for the stock (replaces the static placeholder)."""
    try:
        recs = [r for r in records_for([code]) if code in _record_codes(r)]
    except Exception as exc:
        return dict(status='NOT_CONNECTED', reason='信息模型未接入：未知不等于无利空', error=type(exc).__name__)
    if not recs:
        return dict(status='NO_INFO', reason='该股暂无已导入公告信息：未知保留，不等于无利空')
    entries, summary = assess(recs, asof)
    return dict(status='CONNECTED', n=len(entries), summary=summary,
                current=[dict(ref=e['ref'], title=e['title'], importance=e['importance'],
                              date=e['authoritative_date'], availability=e['availability'],
                              body_version=e['body_version']) for e in entries
                         if e['availability'] != 'SUPERSEDED'])
