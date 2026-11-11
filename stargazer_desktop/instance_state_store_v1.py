"""Cross-process-safe state store for the strong-touch lane.

Encapsulates the read-modify-write of ``touch/<day>.json`` under
:class:`cross_process_lock_v1.FileLock` plus :class:`cross_process_lock_v1.OwnerLease`
so that two processes sharing the same disk file cannot corrupt each other:

* Exactly one *live* process owns a given day's state file (lease).  The owner is
  the one that performs writes and recovery.
* A second process that finds a *live* competing owner is forced read-only on
  that file — it may read health/quotes but must not mutate delivery/indicator
  state or re-send alerts.
* When a process finds a dead or stale owner (crash / heartbeat TTL exceeded),
  it takes over and performs bounded recovery: PENDING/WAIT_REVERIFY events are
  dequeued so they can be re-enqueued, and PROCESSING/SENDING events are moved to
  a safe state (WAIT_REVERIFY if interrupted before transport, UNKNOWN
  otherwise).  Already-sent events keep their terminal status, so takeover never
  re-sends a push.

This module is the single place the contract constants ``STRONG_TOUCH_LANE_V1``
and the recovery rules live for the on-disk state; the lane file delegates to it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import cross_process_lock_v1 as cpl

MAX_STATE_BYTES = 4 * 1024 * 1024
DEFAULT_TTL = 300.0  # seconds; > any single poll/indicator operation

# Per-process record of which state files THIS process has claimed, as
# (pid, nonce).  Used so a reused pid cannot masquerade as the owner of a record
# it did not create, and so the owner can recognise its own lease on re-read.
_OWNER_IDENTITY: Dict[str, Tuple[int, str]] = {}


def _recover(value: Dict[str, Any]) -> bool:
    """Reset in-flight delivery/indicator status for a taken-over state file.

    Only ever invoked when *this* process legitimately takes over a dead owner, so
    it can never run against a live competing owner.  Recovery distinguishes:
      * PENDING / WAIT_REVERIFY  -> dequeued for safe re-enqueue (send not started).
      * PROCESSING + PRE_SEND_VERIFY phase -> dequeued for safe re-enqueue (the
        verify/send had not actually begun; re-verify then send once).
      * SENDING / TRANSPORT (send started, receipt not written) -> UNKNOWN, which
        is terminal and must NOT auto-resend (delivery outcome is uncertain).
    Already-SENT events keep their terminal status, so takeover never re-sends.
    """
    dirty = False
    for event in value.get("events", {}).values():
        status = event.get("delivery_status")
        if status in ("PENDING", "WAIT_REVERIFY"):
            event["delivery_queued"] = False
            dirty = True
        elif status in ("PROCESSING", "SENDING"):
            if status == "PROCESSING" and event.get("delivery_phase") == "PRE_SEND_VERIFY":
                event.update(delivery_status="WAIT_REVERIFY", delivery_queued=False,
                             restart_reason="PRE_SEND_INTERRUPTED_SAFE_RETRY")
            else:
                event.update(delivery_status="UNKNOWN", delivery_queued=False,
                             restart_reason="DELIVERY_UNCERTAIN_NO_RESEND")
            dirty = True
        if event.get("indicator_status") in ("PENDING", "PROCESSING"):
            event.update(indicator_status="PENDING", indicator_queued=False)
            dirty = True
    return dirty


def _identity_matches(meta, fid: str) -> bool:
    my_id = _OWNER_IDENTITY.get(fid)
    return (my_id is not None and isinstance(meta, dict)
            and meta.get("pid") == my_id[0] and meta.get("nonce") == my_id[1])


def read_state(
    file: Any,
    *,
    contract: str,
    day: Optional[int] = None,
    pool_id: Optional[str] = None,
    codes: Optional[Any] = None,
    build_default: Callable[[Optional[int], Optional[str]], Dict[str, Any]],
    ttl: float = DEFAULT_TTL,
    timeout: float = 30.0,
) -> Tuple[Dict[str, Any], bool]:
    """Read (and possibly take over) a state file.

    Returns ``(value, is_owner)``.  ``is_owner`` is True only when *this* process
    holds the lease (it just claimed a dead owner, or it already owns it).  A live
    competing owner, an unconfirmable-dead owner, or a corrupt owner file yields
    ``is_owner=False`` and a strictly read-only value (no recovery, no mutation).
    """
    file = Path(file)
    fid = str(file)
    with cpl.FileLock(file, timeout=timeout):
        if not file.exists():
            if day is None:
                raise FileNotFoundError(str(file))
            value = build_default(day, pool_id)
            owner = cpl.OwnerLease(file, ttl=ttl)
            rec = owner.claim()
            _OWNER_IDENTITY[fid] = (rec["pid"], rec["nonce"])
            value["owner"] = rec
            return value, True
        if file.stat().st_size > MAX_STATE_BYTES:
            raise ValueError("STRONG_TOUCH_STATE_TOO_LARGE")
        value = json.loads(file.read_text(encoding="utf-8"))
        if value.get("contract") != contract:
            raise ValueError("STRONG_TOUCH_STATE_CONTRACT_MISMATCH")
        if day is not None and int(value.get("day") or 0) != int(day):
            value = build_default(day, pool_id)
            owner = cpl.OwnerLease(file, ttl=ttl)
            rec = owner.claim()
            _OWNER_IDENTITY[fid] = (rec["pid"], rec["nonce"])
            value["owner"] = rec
            return value, True
        if pool_id is not None and value.get("pool_id") != pool_id:
            # Same-day per-symbol quote history is kept; events stay bound to
            # their original frozen pool (cancelled by the worker).
            value["pool_id"] = pool_id
        if codes is not None:
            allowed = set(map(str, codes))
            value["symbols"] = {
                k: v for k, v in value.get("symbols", {}).items() if k in allowed
            }
        value.setdefault("symbols", {})
        value.setdefault("events", {})
        value.setdefault("health", {})

        lease = cpl.OwnerLease(file, ttl=ttl)
        # The .owner meta file is the authoritative cross-process reservation;
        # value["owner"] (inside the JSON) is only informational.
        meta = lease.read()
        if lease.is_stale(meta):
            # Owner pid CONFIRMED dead (or no owner) -> take over, recover, refresh
            # heartbeat.  A live/unconfirmable-dead owner or a corrupt owner file
            # is NEVER seized here.
            value["_restart_recovery_pending"] = True
            _recover(value)
            rec = lease.claim()
            _OWNER_IDENTITY[fid] = (rec["pid"], rec["nonce"])
            value["owner"] = rec
            return value, True
        if _identity_matches(meta, fid):
            # We already own it (pid+nonce match) -> refresh heartbeat, keep owner.
            lease.refresh()
            value["owner"] = meta
            return value, True
        # A *different* live process owns this file: read-only, no recovery.
        value.pop("_restart_recovery_pending", None)
        value["owner"] = meta
        return value, False


def is_owner(file: Any, *, ttl: float = DEFAULT_TTL) -> bool:
    """Cheap check (no full read) of whether THIS process owns the state file.

    Used by the delivery/indicator workers to gate side-effecting operations:
    a non-owner must never enqueue, broadcast, compute indicators, or send.
    """
    file = Path(file)
    fid = str(file)
    lease = cpl.OwnerLease(file, ttl=ttl)
    meta = lease.read()
    return _identity_matches(meta, fid)


def write_state(
    file: Any,
    value: Dict[str, Any],
    *,
    ttl: float = DEFAULT_TTL,
    timeout: float = 30.0,
) -> None:
    """Atomically write a state file.  Fail-closed if we do not own the lease.

    Ownership is re-verified against ``<file>.owner`` (pid+nonce); a competing
    live owner, an unconfirmable-dead owner, or a corrupt owner file means our
    write would clobber theirs, so we refuse rather than silently corrupt.
    """
    file = Path(file)
    fid = str(file)
    value.pop("_restart_recovery_pending", None)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    if len(raw) > MAX_STATE_BYTES:
        raise ValueError("STRONG_TOUCH_STATE_TOO_LARGE")
    with cpl.FileLock(file, timeout=timeout):
        lease = cpl.OwnerLease(file, ttl=ttl)
        owner = lease.read()
        if not _identity_matches(owner, fid):
            # We are not the owner — refuse to clobber another process's state.
            raise RuntimeError("STRONG_TOUCH_STATE_NOT_OWNED")
        lease.refresh()
        file.parent.mkdir(parents=True, exist_ok=True)
        temp = file.with_suffix(file.suffix + ".tmp")
        temp.write_bytes(raw)
        temp.replace(file)


def release_ownership(file: Any, *, ttl: float = DEFAULT_TTL) -> None:
    """Best-effort release of a state-file lease (e.g. on clean shutdown)."""
    cpl.OwnerLease(file, ttl=ttl).release()
    _OWNER_IDENTITY.pop(str(Path(file)), None)
