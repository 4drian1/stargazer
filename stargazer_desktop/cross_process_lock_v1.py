"""Cross-process advisory lock + lease reservation/recovery.

Problem this solves (Step2 稳定性修复 — 跨进程预约/恢复保护):

``strong_touch_lane_v1`` and ``strong_sector_watch_v1`` previously relied on a
per-process ``threading.RLock`` plus a per-process ``PROCESS_ID = uuid4()`` to
decide whether the on-disk state/alerts files "belonged" to the current
process.  That assumption is false the moment two processes share the same disk
files — e.g. the desktop app launched twice, or ``strong_sector_watch_v1.invoke``
spawning a child interpreter that touches the same ``pools/``/``minutes/``/
``alerts.json``/``touch/<day>.json`` files.  Every read by the "other" process
reset in-flight delivery/indicator status, and ``alerts.json`` writes could
interleave and corrupt.

This module provides the missing cross-process layer:

* :class:`FileLock` — a portable *blocking* advisory lock on a ``.lock`` sibling
  file (``fcntl.flock`` on POSIX, ``msvcrt.locking`` on Windows).  Holding the
  lock serialises the read-modify-write of the shared file across processes.
* :class:`OwnerLease` — a heartbeat-based reservation on a ``<file>.owner`` meta
  file.  Exactly one *live* process may own a shared state file at a time; a
  process that finds a dead or stale owner takes it over and performs recovery
  (re-enqueue of PENDING/WAIT_REVERIFY, no resend of already-sent events).  A
  live competing owner means the current process must stay read-only on that
  file so it cannot clobber the owner's work.

These are deliberately dependency-free (stdlib only) so they can be imported by
every monitoring module without pulling in the heavy desktop stack.
"""
from __future__ import annotations

import ctypes
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["FileLock", "OwnerLease", "pid_alive", "OWNER_CORRUPT"]

# Sentinel returned by OwnerLease.read() when the owner file exists but cannot be
# parsed.  Crucially this is NOT the same as "no owner": a corrupt owner file must
# never be treated as "nobody owns this" (which would let a second process seize
# control and corrupt the state).  It forces read-only / refuse behaviour.
OWNER_CORRUPT = object()


def pid_alive(pid: int) -> bool:
    """Best-effort liveness check for an arbitrary process id (cross-platform)."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        # OpenProcess with SYNCHRONIZE (0x00100000) + PROCESS_QUERY_LIMITED_INFORMATION
        # (0x1000).  If the handle opens, the process exists.  We do NOT need
        # PROCESS_TERMINATE — opening for sync/query is enough and safe.
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        SYNCHRONIZE = 0x00100000
        PROCESS_QUERY_LIMITED = 0x1000
        handle = kernel32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED, 0, pid)
        if handle:
            try:
                status = kernel32.WaitForSingleObject(handle, 0)
                if status == 258:
                    return True
                if status == 0:
                    return False
                raise OSError(ctypes.get_last_error(), 'PROCESS_STATUS_UNKNOWN')
            finally:
                kernel32.CloseHandle(handle)
        error = ctypes.get_last_error()
        if error == 87:
            return False
        if error == 5:
            raise PermissionError(error, 'PROCESS_STATUS_UNKNOWN')
        raise OSError(error, 'PROCESS_STATUS_UNKNOWN')
    # POSIX: signal 0 performs the existence/permission check without signalling.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class FileLock:
    """Blocking cross-process advisory lock on a ``.lock`` sibling file.

    Usage::

        with FileLock(state_file):
            value = json.loads(state_file.read_bytes())  # safe RMW
            state_file.write_bytes(...)

    The lock is held for the duration of the ``with`` block and released
    (handle closed, on Windows the byte-range unlock is implicit on close) on
    exit, even on exception.  Multiple threads inside the *same* process are
    still serialised by the additional ``threading.Lock`` per path so a process
    never blocks on its own lock.
    """

    _local = threading.Lock()
    _per_path: Dict[str, threading.Lock] = {}
    _open_handles: Dict[str, int] = {}

    def __init__(self, path: Any, *, timeout: float = 30.0):
        self._path = Path(path)
        self._lock_file = self._path.with_suffix(self._path.suffix + ".lock")
        self._timeout = timeout
        self._fh = None
        self._thread_lock = self._thread_lock_for(str(self._lock_file))

    @classmethod
    def _thread_lock_for(cls, key: str) -> threading.Lock:
        with cls._local:
            lock = cls._per_path.get(key)
            if lock is None:
                lock = threading.Lock()
                cls._per_path[key] = lock
            return lock

    def _acquire(self) -> None:
        # Intra-process serialisation first: never deadlock against our own threads.
        self._thread_lock.acquire()
        self._lock_file.parent.mkdir(parents=True, exist_ok=True)
        # Open in append/create mode; we never read/write this file's content,
        # only hold an exclusive OS lock on it.
        fd = os.open(str(self._lock_file), os.O_CREAT | os.O_RDWR, 0o644)
        self._fh = fd
        deadline = time.monotonic() + self._timeout
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    return
                except OSError:
                    if time.monotonic() >= deadline:
                        os.close(fd)
                        self._fh = None
                        self._thread_lock.release()
                        raise TimeoutError(
                            f"FileLock 获取超时: {self._lock_file}"
                        )
                    time.sleep(0.02)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return
                except OSError:
                    if time.monotonic() >= deadline:
                        os.close(fd)
                        self._fh = None
                        self._thread_lock.release()
                        raise TimeoutError(
                            f"FileLock 获取超时: {self._lock_file}"
                        )
                    time.sleep(0.02)

    def _release(self) -> None:
        try:
            if self._fh is not None:
                if os.name == "nt":
                    import msvcrt

                    try:
                        msvcrt.locking(self._fh, msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                else:
                    import fcntl

                    try:
                        fcntl.flock(self._fh, fcntl.LOCK_UN)
                    except OSError:
                        pass
                try:
                    os.close(self._fh)
                except OSError:
                    pass
                self._fh = None
        finally:
            self._thread_lock.release()

    def __enter__(self) -> "FileLock":
        self._acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._release()


class OwnerLease:
    """Cross-process reservation for a shared state file.

    Exactly one process owns a given state file.  The owner writes
    ``{pid, nonce, heartbeat, ...}`` to ``<file>.owner``.  Takeover is allowed
    ONLY when the recorded owner pid is *confirmed dead* — a live owner, an
    unconfirmable-dead owner (permission error etc.), or a corrupt owner file is
    never seized.  A per-claim random ``nonce`` prevents a PID-reuse impostor
    (a new, unrelated process that happened to reuse the dead owner's pid) from
    writing to a record it did not create.

    Heartbeat age no longer triggers takeover: a live owner keeps ownership
    regardless of how long it has been since its last heartbeat refresh.  This is
    deliberate — the stability spec forbids taking over a live or merely
    heartbeat-expired owner.
    """

    def __init__(self, path: Any, *, ttl: float = 300.0):
        self._path = Path(path)
        self._owner_file = self._path.with_suffix(self._path.suffix + ".owner")
        self._ttl = float(ttl)
        self._token = os.getpid()
        # Per-process claim identity.  None until this process has claimed.  Used so
        # a reused pid cannot masquerade as the owner of a record it did not make.
        self._nonce: Optional[str] = None

    @property
    def token(self) -> int:
        return self._token

    @property
    def nonce(self) -> Optional[str]:
        return self._nonce

    def _read(self):
        if not self._owner_file.exists():
            return None
        try:
            data = json.loads(self._owner_file.read_bytes())
        except (OSError, ValueError):
            # File exists but is unparseable -> do NOT treat as "no owner".
            return OWNER_CORRUPT
        if not isinstance(data, dict) or "pid" not in data or "nonce" not in data:
            return OWNER_CORRUPT
        return data

    def read(self):
        return self._read()

    def is_confirmed_dead(self, owner) -> bool:
        """True only when the owner pid is CONFIRMED dead (not alive, not unknown)."""
        if not isinstance(owner, dict):
            # OWNER_CORRUPT or malformed -> never treat as confirmed dead.
            return False
        pid = owner.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            return False
        try:
            return not pid_alive(pid)
        except Exception:
            # Cannot confirm death (e.g. permission denied) -> do not seize.
            return False

    def is_stale(self, owner) -> bool:
        """Safe to take over?  Only when there is NO owner, or the owner pid is
        CONFIRMED dead.  A live owner, an unconfirmable-dead owner, or a corrupt
        owner file is NOT stale."""
        if owner is None:
            return True
        if owner is OWNER_CORRUPT:
            return False
        return self.is_confirmed_dead(owner)

    def i_am_owner(self, owner) -> bool:
        """Am *I* the owner?  Requires both pid and nonce to match (anti-PID-reuse)."""
        if owner is None or owner is OWNER_CORRUPT:
            return False
        return (owner.get("pid") == self._token
                and isinstance(owner.get("nonce"), str)
                and owner.get("nonce") == self._nonce)

    def claim(self, *, nonce: Optional[str] = None) -> Dict[str, Any]:
        """Take over (or refresh) the lease.  Caller must already hold FileLock."""
        use_nonce = nonce if nonce else secrets.token_hex(8)
        self._nonce = use_nonce
        host = ""
        try:
            host = os.environ.get("COMPUTERNAME") or os.uname().nodename  # type: ignore[attr-defined]
        except Exception:
            host = ""
        record = dict(pid=self._token, nonce=use_nonce, heartbeat=time.time(),
                      ttl=self._ttl, host=host)
        self._owner_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._owner_file.with_suffix(".owner.tmp")
        tmp.write_bytes(json.dumps(record, ensure_ascii=False).encode("utf-8"))
        tmp.replace(self._owner_file)
        return record

    def refresh(self) -> None:
        """Bump heartbeat if we still own the lease (idempotent on staleness)."""
        owner = self._read()
        if self.i_am_owner(owner):
            owner["heartbeat"] = time.time()  # type: ignore[index]
            try:
                tmp = self._owner_file.with_suffix(".owner.tmp")
                tmp.write_bytes(json.dumps(owner, ensure_ascii=False).encode("utf-8"))
                tmp.replace(self._owner_file)
            except OSError:
                pass

    def release(self) -> None:
        """Best-effort release: only delete if we still own it."""
        owner = self._read()
        if self.i_am_owner(owner):
            try:
                self._owner_file.unlink()
            except OSError:
                pass
