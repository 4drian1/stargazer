"""Local append-only conversations. API context is a separate bounded view."""
import json
from pathlib import Path
import uuid
import assistant_workflow as workflow


def open_session(system, fresh=False):
    with workflow.LOCK:
        files = sorted(workflow.root().glob('chat_session_*.jsonl'), key=lambda p: p.stat().st_mtime_ns)
        if files and not fresh:
            records = [json.loads(line) for line in files[-1].read_text(encoding='utf-8').splitlines()]
            # Reload current system policy, never treat an old system prompt as current policy.
            return files[-1], [system] + [m for m in records if m.get('role') != 'system']
        file = workflow.root() / f'chat_session_{uuid.uuid4().hex}.jsonl'
        append(file, [system], exclusive=True)
        return file, [system]


def append(file, messages, exclusive=False):
    if not messages:
        return
    with workflow.LOCK:
        raw = b''.join(workflow.encoded(m) + b'\n' for m in messages)
        from persistent_records_v1 import reserve_space
        reserve_space(file.parent, len(raw))
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open('xb' if exclusive else 'ab') as stream:
            stream.write(raw)


def tool_content(result, file):
    raw = json.dumps(result, ensure_ascii=False, default=str, allow_nan=False)
    if len(raw) <= 10000:
        return raw
    evidence = workflow.store('tool_evidence', {'session': file.stem, 'result': json.loads(raw)})
    if isinstance(result,dict) and result.get('kind') in ('CONFIRMED_PLAN_VIEW_V1','MINUTE_REVIEW_VIEW_V1'):
        return workflow.encoded({**result,'evidence_id':evidence['id'],
            'detail_notice':'可读名单与复盘事实已完整载入；原始证据保留本地，不需要用户重新提供。'}).decode()
    if isinstance(result, dict) and result.get('scope') == 'COMPLETED_DAILY_RULE_SCAN_NOT_INTRADAY_TRIGGER':
        summary = {k:v for k,v in result.items() if k != 'excluded'}
        summary['excluded_counts'] = {}
        for row in result.get('excluded', []):
            reason = row['reason']
            summary['excluded_counts'][reason] = summary['excluded_counts'].get(reason, 0) + 1
        summary['evidence_id'] = evidence['id']
        summary['detail_notice'] = '逐股缺失/异常明细已归档；排名、全部观察档位和成员口径仍完整载入。'
        return workflow.encoded(summary).decode()
    return workflow.encoded({'status': 'RESULT_ARCHIVED_NOT_LOADED',
                             'evidence_id': evidence['id'],
                             'note': '完整工具结果留在本地；请缩小查询，不把截断JSON当作完整数据。'}).decode()
