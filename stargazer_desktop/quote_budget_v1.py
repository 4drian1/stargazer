"""Shared quote-request accounting with lane-specific limits kept separate.

This ledger counts actual HTTP attempts. It deliberately does not impose a
global cap: the existing 2,000/day plan-ticket cap remains owned by that lane
until a system-wide provider budget is explicitly established.
"""
import json
import threading
import time
from pathlib import Path

CONTRACT = 'QUOTE_REQUEST_USAGE_V1'
MAX_BYTES = 128 * 1024
_LOCK = threading.RLock()


def _read(path, day):
    path = Path(path)
    if not path.exists():
        return dict(contract=CONTRACT, day=day, total_attempts=0, by_lane={})
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('QUOTE_USAGE_FILE_TOO_LARGE')
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('contract') != CONTRACT:
        raise ValueError('QUOTE_USAGE_CONTRACT_MISMATCH')
    if value.get('day') != day:
        return dict(contract=CONTRACT, day=day, total_attempts=0, by_lane={})
    return value


def record_attempt(path, lane, *, now=None):
    """Record one real HTTP attempt; callers invoke this immediately pre-send."""
    if not isinstance(lane, str) or not lane.strip():
        raise ValueError('QUOTE_USAGE_LANE_REQUIRED')
    now = now or time.localtime()
    day = time.strftime('%Y%m%d', now)
    stamp = time.strftime('%Y%m%d%H%M%S', now)
    path = Path(path)
    with _LOCK:
        value = _read(path, day)
        value['total_attempts'] = int(value.get('total_attempts', 0)) + 1
        lanes = value.setdefault('by_lane', {})
        entry = lanes.setdefault(lane, dict(attempts=0))
        entry['attempts'] = int(entry.get('attempts', 0)) + 1
        entry['last_attempt_at'] = stamp
        value['updated_at'] = stamp
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')
        if len(raw) > MAX_BYTES:
            raise ValueError('QUOTE_USAGE_FILE_TOO_LARGE')
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_bytes(raw)
        temp.replace(path)
        return value


def snapshot(path, *, day=None):
    day = day or time.strftime('%Y%m%d')
    with _LOCK:
        return _read(path, day)
