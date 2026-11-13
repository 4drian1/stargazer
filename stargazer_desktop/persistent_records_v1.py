"""User-authorized durable records. No cache cap, no automatic record deletion.

NTFS directory junctions preserve old absolute evidence references. Only the
fixed categories below may leave the cache account; arbitrary links may not.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading

CONTRACT = 'STARGAZER_PERSISTENT_RECORDS_V1'
POLICY = 'persistence_policy.json'
MIN_FREE = 128 * 1024**2
_LOCK = threading.RLock()


def records_root(data):
    return Path(data) / 'local_records'


def authorized(root):
    policy = Path(root) / POLICY
    if not policy.exists():
        return False
    value = json.loads(policy.read_bytes())
    if value.get('contract') != CONTRACT or value.get('user_authorized') is not True or value.get('uncapped') is not True:
        raise ValueError('PERSISTENT_RECORD_POLICY_INVALID')
    return True


def is_record(path):
    p = Path(path).resolve()
    for parent in (p, *p.parents):
        if parent.name == 'local_records' and authorized(parent):
            return True
    return False


def reserve_space(path, extra=0):
    p = Path(path)
    while not p.exists():
        p = p.parent
    if shutil.disk_usage(p).free < max(0, extra) + MIN_FREE:
        raise ValueError('LOCAL_RECORDS_DISK_SPACE_LOW:本地记录磁盘空间不足；旧记录保留')


def file_path(app_dir, name):
    """The execution plan lives in records; the old path is a migration snapshot."""
    normalized = name.replace('\\', '/')
    if normalized not in ('data/trade_plan.json', 'dist/data/trade_plan.json'):
        return None
    root = records_root(Path(app_dir) / ('dist/data' if normalized.startswith('dist/') else 'data'))
    return root / 'trade_plan.json' if authorized(root) else None


def cache_files(base):
    """Do not follow durable-record junctions or silently exempt unknown links."""
    base = Path(base)
    if not base.exists() or is_record(base):
        return
    if base.is_symlink() or base.is_junction():
        raise ValueError('UNAUTHORIZED_CACHE_LINK:' + str(base))
    folders = [base]
    while folders:
        folder = folders.pop()
        try:
            entries = os.scandir(folder)
        except FileNotFoundError:
            continue  # A cache writer may have removed the directory.
        with entries:
            for entry in entries:
                linked = entry.is_symlink() or entry.is_junction()
                if linked:
                    if entry.is_dir() and is_record(entry.path):
                        continue
                    raise ValueError('UNAUTHORIZED_CACHE_LINK:' + entry.path)
                if entry.is_dir(follow_symlinks=False):
                    if entry.name == 'local_records' and is_record(entry.path):
                        continue
                    folders.append(entry.path)
                else:
                    # DirEntry retains file-size metadata from enumeration on Windows.
                    yield entry


def inventory(folder):
    """Byte identity before/after rename; no contents or credentials in receipt."""
    folder = Path(folder)
    values = []
    size = 0
    for p in sorted(folder.rglob('*')):
        if p.is_symlink() or p.is_junction():
            raise ValueError('RECORD_MIGRATION_NESTED_LINK')
        if not p.is_file():
            continue
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                h.update(block)
        size += p.stat().st_size
        values.append((p.relative_to(folder).as_posix(), p.stat().st_size, h.hexdigest()))
    raw = json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode('utf8')
    return dict(files=len(values), bytes=size, tree_sha256=hashlib.sha256(raw).hexdigest())


def _junction(source, target):
    def literal(p):return "'" + str(p).replace("'", "''") + "'"
    command = 'New-Item -ItemType Junction -Path ' + literal(source) + ' -Target ' + literal(target) + ' -ErrorAction Stop | Out-Null'
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                            capture_output=True, timeout=30, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise OSError('PERSISTENT_JUNCTION_CREATE_FAILED')


def move_folder(source, target, root):
    source, target, root = Path(source).absolute(), Path(target).absolute(), Path(root).resolve()
    if not target.resolve().is_relative_to(root) or not authorized(root):
        raise ValueError('PERSISTENT_TARGET_OUTSIDE_AUTHORITY')
    if source.is_junction():
        if source.resolve() != target.resolve():raise ValueError('PERSISTENT_LINK_TARGET_CHANGED')
        return dict(source=str(source), target=str(target), reused=True)
    if source.is_symlink():raise ValueError('PERSISTENT_SOURCE_LINK')
    key=hashlib.sha256(str(source).encode('utf8')).hexdigest()
    intent=root/'migration_intents'/(key+'.json')
    if not source.exists():
        if not target.exists():return None
        receipt=json.loads(intent.read_bytes()) if intent.exists() else {}
        if receipt.get('source')!=str(source) or receipt.get('target')!=str(target) or inventory(target)!=receipt.get('before'):
            raise ValueError('PERSISTENT_INTERRUPTED_TARGET_UNVERIFIED')
        _junction(source,target)
        return dict(source=str(source),target=str(target),resumed=True,**receipt['before'])
    if target.exists():raise ValueError('PERSISTENT_TARGET_EXISTS_REFUSE_MERGE')
    reserve_space(target)
    before = inventory(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    intent.parent.mkdir(parents=True,exist_ok=True)
    body=json.dumps(dict(source=str(source),target=str(target),before=before),ensure_ascii=False).encode('utf8')
    temporary=intent.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        stream.write(body);stream.flush();os.fsync(stream.fileno())
    temporary.replace(intent)
    # Same-volume rename: no duplicate multi-GB data and no rewriting receipts.
    source.rename(target)
    try:
        _junction(source, target)
        if source.resolve() != target.resolve() or inventory(source) != before:
            raise ValueError('PERSISTENT_MIGRATION_VERIFY_FAILED')
    except Exception:
        if source.is_junction():source.rmdir()
        if not source.exists():target.rename(source)
        raise
    return dict(source=str(source), target=str(target), reused=False, **before)


def prepare_capture(output, runtime):
    """A new risk run is durable from its first write, while common caches stay bounded."""
    output, runtime = Path(output), Path(runtime)
    root = records_root(runtime.parent)
    if output.parent != runtime or not output.name.startswith('run_') or not authorized(root):
        return
    with _LOCK:
        if not output.exists() and not (root/'risk_runs'/output.name).exists():
            output.mkdir(parents=True)
        move_folder(output, root/'risk_runs'/output.name, root)


def migrate(data):
    data = Path(data).absolute()
    root = records_root(data)
    if not authorized(root):raise ValueError('PERSISTENT_RECORDS_NOT_AUTHORIZED')
    runtime = data/'postclose_risk'
    mappings = [(data/'pentacles_demo', root/'pentacles_demo'),
                (data/'assistant', root/'assistant'),
                (data/'review_archive', root/'review_archive'),
                (runtime/'tdx_daily_v2', root/'selection_inputs/tdx_daily_v2'),
                (runtime/'tdx_prepared', root/'risk_inputs/tdx_prepared'),
                (runtime/'intraday_native', root/'risk_inputs/intraday_native'),
                (runtime/'strong_pullback',root/'strong_pullback'),
                (runtime/'tarot_v2',root/'risk_scores'),
                (runtime/'risk_outlook',root/'risk_outlook')]
    mappings += [(p, root/'risk_runs'/p.name) for p in sorted(runtime.glob('run_*')) if p.is_dir()]
    results = []
    with _LOCK:
        for source, target in mappings:
            if not source.exists() and not target.exists():source.mkdir(parents=True,exist_ok=True)
            result = move_folder(source, target, root)
            if result:results.append(result)
        old, new = data/'trade_plan.json', root/'trade_plan.json'
        if old.exists() and not new.exists():
            body=old.read_bytes();reserve_space(new,len(body));new.write_bytes(body)
            if new.read_bytes()!=body:raise ValueError('PLAN_MIGRATION_VERIFY_FAILED')
    return dict(contract=CONTRACT, mappings=results, cache_budget_included=False)


def status(data):
    root = records_root(data)
    existing=root
    while not existing.exists():existing=existing.parent
    return dict(directory=str(root), authorized=authorized(root), cache_budget_included=False,
                bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0,
                free_bytes=shutil.disk_usage(existing).free,
                policy='独立本地记录，无固定容量上限；不自动删除历史')
