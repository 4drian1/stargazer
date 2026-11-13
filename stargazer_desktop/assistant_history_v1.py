"""Local history management for sessions and assistant drafts."""
import json
import re

import assistant_workflow as workflow


KINDS = {'chat': ('chat_session_', '.jsonl', 32),
         'plan': ('plan_', '.json', 64),
         'review': ('review_', '.json', 64)}


def _locked_paths(root, kind, active_session):
    prefix, suffix, _ = KINDS[kind]
    live = list(root.glob(prefix + '*' + suffix))
    locked = {max(live, key=lambda p: p.stat().st_mtime_ns).name} if live else set()
    if kind == 'chat' and active_session:
        locked.add(active_session.name)
    if kind != 'chat':
        locked.update(prefix + p.stem.removeprefix('confirmed_') + suffix
                      for p in root.glob('confirmed_*.json'))
    return locked


def entries(kind, archived=False, active_session=None):
    root = workflow.root()
    folder = root / 'history' if archived else root
    if not folder.resolve().is_relative_to(root.resolve()):
        raise ValueError('历史目录不在本地记录区')
    prefix, suffix, width = KINDS[kind]
    with workflow.LOCK:
        locked = _locked_paths(root, kind, active_session)
        files = [p for p in folder.glob(prefix + '*' + suffix)
                 if re.fullmatch(r'[0-9a-f]{%d}' % width, p.stem.removeprefix(prefix))]
        return [dict(id=p.stem.removeprefix(prefix), bytes=p.stat().st_size,
                     updated_at=p.stat().st_mtime, locked=p.name in locked)
                for p in sorted(files, key=lambda x: x.stat().st_mtime_ns, reverse=True)]


def record(kind, identifier, archived, active_session=None, action=None):
    if kind not in KINDS:
        raise ValueError('历史类别无效')
    prefix, suffix, width = KINDS[kind]
    if not isinstance(identifier, str) or not re.fullmatch(r'[0-9a-f]{%d}' % width, identifier):
        raise ValueError('历史编号无效')
    root = workflow.root()
    archive = root / 'history'
    if not archive.resolve().is_relative_to(root.resolve()):
        raise ValueError('历史目录不在本地记录区')
    filename = prefix + identifier + suffix
    file = (archive if archived else root) / filename
    with workflow.LOCK:
        if not file.is_file() or file.is_symlink() or file.is_junction():
            raise ValueError('历史记录不存在')
        if action is None:
            if kind != 'chat':
                data = workflow.load_draft(kind, identifier)
                return dict(text=data.get('reply') or data.get('body') or '', messages=[])
            rows = [json.loads(line) for line in file.read_text(encoding='utf-8').splitlines()]
            return dict(text='', messages=[dict(role=r['role'], content=r['content'])
                       for r in rows if r.get('role') in ('user', 'assistant') and r.get('content')])
        if filename in _locked_paths(root, kind, active_session) and action != 'restore':
            raise ValueError('当前使用中或已经确认的记录不能清理')
        if action == 'delete' and archived and kind == 'chat':
            file.unlink()
        elif (action == 'archive' and not archived) or (action == 'restore' and archived):
            target = (root if archived else archive) / filename
            target.parent.mkdir(exist_ok=True)
            file.rename(target)
        else:
            raise ValueError('该历史记录不支持此操作')
        return {}

