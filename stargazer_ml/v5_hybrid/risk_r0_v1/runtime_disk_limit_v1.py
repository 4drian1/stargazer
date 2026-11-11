"""Runtime disk-budget gate for desktop review / minute-prep / strong-pool.

This module is DELIBERATELY lightweight: stdlib only, NO import of
``joint_shadow_snapshot_v1`` / ``research_budget_v2`` / any risk kernel. The
desktop callers import ``disk_limit`` from HERE so they never transitively pull
the research-storage contract loader (which lives in ``research_budget_v2`` and
is intentionally absent from a frozen desktop build). Do not add heavy imports
here -- that is exactly the bug this file was split out to remove.

See live_postclose_source_v1.disk_limit for the historical inline copy; that
module now re-exports this one so the research entry keeps a single source of
truth while the desktop stays clear of the chain.
"""
from pathlib import Path
import json

# User-approved 1 GiB for this runtime on 2026-10-21. The policy file may raise
# this to the shared 4 GiB postclose cache cap, but only via an explicit,
# user-authorized policy -- never silently.
DISK_CAP = 1024**3
MONITOR_RESERVE = 32 * 1024**2


def disk_usage(root):
    """Count rebuildable cache; authorized durable records have their own account."""
    root = Path(root)
    archive = root.parent / 'review_archive/strong_sector'
    from persistent_records_v1 import cache_files
    return sum(p.stat().st_size for base in (root, archive) for p in cache_files(base))


def headroom(root):
    return disk_limit(root) - disk_usage(root)


def disk_limit(root):
    """Return the authorized disk cap (bytes) for ``root``'s runtime cache.

    Falls back to ``DISK_CAP`` when no policy file is present. Raises
    ``UNAUTHORIZED_RUNTIME_BUDGET`` if the stored policy is not one of the two
    explicitly approved caps (1 GiB / 4 GiB) together with
    ``user_authorized == True`` -- the gate is fail-closed, not defaulting.
    """
    policy = Path(root) / 'runtime_budget_v2.json'
    if not policy.exists():
        return DISK_CAP
    value = json.loads(policy.read_bytes())
    cap = value.get('disk_bytes')
    if cap not in (1024**3, 4 * 1024**3) or value.get('user_authorized') is not True:
        raise ValueError('UNAUTHORIZED_RUNTIME_BUDGET')
    return cap
