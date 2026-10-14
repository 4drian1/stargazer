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
        total = sum(p.stat().st_size for p in workflow.root().rglob('*') if p.is_file())
        if total + len(raw) > workflow.LIMIT:
            raise ValueError('助手原始对话达到10MiB，停止写入；风控提醒不受影响')
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open('xb' if exclusive else 'ab') as stream:
            stream.write(raw)


def tool_content(result, file):
    raw = json.dumps(result, ensure_ascii=False, default=str, allow_nan=False)
    if len(raw) <= 10000:
        return raw
    evidence = workflow.store('tool_evidence', {'session': file.stem, 'result': json.loads(raw)})
    return workflow.encoded({'status': 'RESULT_ARCHIVED_NOT_LOADED',
                             'evidence_id': evidence['id'],
                             'note': '完整工具结果留在本地；请缩小查询，不把截断JSON当作完整数据。'}).decode()
