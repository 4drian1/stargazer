"""Bounded API input; original history remains in the local business archive."""
import json

MAX_CHARS = 58000


def serialized(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def candidate_table(snapshot):
    columns = ['code', 'name', 'setup', 'reference_price', 'zone_lower', 'zone_upper', 'zone_sources', 'status']
    rows = []
    for row in snapshot['rows']:
        zone = row.get('zone') or {}
        rows.append([row.get(k) for k in columns[:4]] + [zone.get(k) for k in columns[4:7]] + [row.get('status')])
    if snapshot.get('rule_version') == 'TAROT_A_SECOND_CASH_V2':
        columns += ['rank', 'selection_score']
        for values, row in zip(rows, snapshot['rows']):
            values += [row.get('rank'), row.get('selection_score')]
    return {'columns': columns, 'rows': rows}


def bounded_messages(prefix, history):
    """Keep all user conditions verbatim and newest complete tool-call groups.

    Older assistant/tool text is evidence, not durable policy. Its omission is
    explicit; callers retain the full original archive, including tool results.
    Never sever an assistant tool-call from its responses.
    """
    groups = []
    for message in history:
        if message.get('role') == 'user' or not groups:
            groups.append([])
        groups[-1].append(message)
    recent = groups[-4:]
    older = groups[:-4]
    constraints = [m['content'] for group in older for m in group if m.get('role') == 'user']
    note = ([{'role': 'user', 'content': serialized({
        'archived_user_conditions_verbatim': constraints,
        'history_notice': '较早助手/工具输出未载入；原文保存在本地，不能声称已读取或重新解释已确认条件。'})}]
        if older else [])
    def complete_group(group):
        calls = {c['id'] for m in group for c in m.get('tool_calls', [])}
        replies = {m.get('tool_call_id') for m in group if m.get('role') == 'tool'}
        return calls == replies
    recent = [g if complete_group(g) else
              [m for m in g if m.get('role') != 'tool' and not m.get('tool_calls')]
              + [{'role': 'assistant', 'content': '上次工具链未完整结束；不声明其操作成功，请核对真实账本。'}]
              for g in recent]
    result = prefix + note + [m for g in recent for m in g]
    while len(serialized(result)) > MAX_CHARS and len(recent) > 1:
        removed = recent.pop(0)
        constraints.extend(m['content'] for m in removed if m.get('role') == 'user')
        note = [{'role': 'user', 'content': serialized({
            'archived_user_conditions_verbatim': constraints,
            'history_notice': '较早助手/工具输出已归档未载入；用户原条件保留，不自动改写。'})}]
        result = prefix + note + [m for g in recent for m in g]
    if len(serialized(result)) > MAX_CHARS:
        raise ValueError('本次事实与用户条件超出单次输入预算；原记录保留，请拆分问题或开新会话（不是API额度不足）')
    return result
