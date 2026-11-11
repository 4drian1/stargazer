"""Bounded recovery of local transport failures; never override evidence checks."""
from datetime import datetime
from http.client import IncompleteRead
import json
from pathlib import Path
import socket
import urllib.error

MAX_RESUMES = 3
COOLDOWN_SECONDS = 60


def transient_error(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return False
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    return isinstance(reason, (ConnectionError, TimeoutError, IncompleteRead))


def retry_candidate(state):
    if state.get('status') == 'INTERRUPTED':
        return (state.get('reason', '').startswith('任务锁已释放') and state.get('stage') in (
            'SOURCE_PREFLIGHT', 'CANDIDATE_MINUTE_PREFLIGHT', 'CAPTURING_MINUTES',
            'REBUILDING_FROZEN_FEATURES', 'PACKING_AND_VALIDATING'))
    if state.get('status') not in ('FAILED', 'WAIT_SOURCE'):
        return False
    if 'source_retryable' in state:
        return state['source_retryable'] is True
    # Compatibility with pre-recovery worker receipts. Do not retry generic errors.
    text = state.get('reason', '')
    return text.startswith('URLError:') and any(s in text for s in (
        'WinError 10061', 'WinError 10060', 'WinError 10054', 'Connection refused', 'timed out'))


def source_available():
    # This proves only a listener exists; the worker still validates actual responses.
    try:
        with socket.create_connection(('127.0.0.1', 7899), timeout=.5):
            return True
    except OSError:
        return False


def _path(root, day):
    day = str(day)
    if len(day) != 8 or not day.isdigit():
        raise ValueError('RECOVERY_DATE')
    return Path(root) / ('source_recovery_' + day + '.json')


def allowance(root, day, now):
    path = _path(root, day)
    record = json.loads(path.read_bytes()) if path.exists() else {'attempts': 0}
    if type(record.get('attempts')) is not int or not 0 <= record['attempts'] <= MAX_RESUMES:
        raise ValueError('RECOVERY_LEDGER_INVALID')
    if record['attempts'] >= MAX_RESUMES:
        return False, '自动恢复已达同日3次上限；保留进度，请检查本地数据服务'
    if record.get('last_at'):
        last = datetime.strptime(str(record['last_at']), '%Y%m%d%H%M%S')
        current = datetime.strptime(str(now), '%Y%m%d%H%M%S')
        if (current - last).total_seconds() < COOLDOWN_SECONDS:
            return False, '已保留进度；恢复间隔尚未达到60秒'
    return True, ''


def reserve(root, day, now, snapshot_digest, prior_run):
    """Persist before spawning. A restart cannot reset the same-day recovery budget."""
    import msvcrt
    from pentacles_monitor import atomic_json
    root = Path(root)
    with (root / 'source_recovery.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        try:
            ok, _ = allowance(root, day, now)
            if not ok:
                return False
            path = _path(root, day)
            old = json.loads(path.read_bytes()) if path.exists() else {'attempts': 0}
            record = dict(old, attempts=old['attempts']+1, last_at=now,
                          snapshot_digest=snapshot_digest, prior_run=str(prior_run))
            atomic_json(path, record)
            return True
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
