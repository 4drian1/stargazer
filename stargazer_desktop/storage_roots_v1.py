"""Unified storage zones for the desktop runtime (#18.199).

Everything reads/writes runtime data through this module — no more ad-hoc
``dist/data`` path joining. Three independent zones:

  assets/   verified historical foundations, resumable indicator state, daily
            adapt assets, shared risk context. Separate budget; ordinary cache
            cleaning must never delete assets.
  records/  selection snapshots, confirmed plans, risk scores, real pushes,
            replay records. Archived by day with source references kept.
  cache/    temporary responses, intermediates, rebuildable arrays. Governed by
            the existing runtime disk budget.

Zone budgets are fail-closed policy files (explicit bytes + user_authorized);
nothing auto-expands and no research/Corvus quota is borrowed.
"""
import json
import os
from pathlib import Path

DEFAULT_ROOT_NAME = 'stargazer_data'
ZONES = ('assets', 'records', 'cache')
DEFAULT_BUDGETS = {
    'assets': None,          # NOT authorized by default: proposal only (#18.199-补)
    'records': None,         # NOT authorized by default: proposal only
    'cache': 4 * 1024**3,    # already user-authorized 4 GiB runtime cap
}
ALLOWED_BUDGETS = {
    'assets': (1024**3, 2 * 1024**3, 3 * 1024**3),
    'records': (256 * 1024**2, 512 * 1024**2, 1024**3),
    'cache': (1024**3, 4 * 1024**3),
}


def data_root() -> Path:
    """Configurable D-drive data root; default stays the legacy location so
    existing installs keep working until an authorized migration moves it."""
    value = os.environ.get('STARGAZER_DATA_ROOT')
    if value:
        return Path(value)
    import paths
    return Path(paths.data_file('data', seed_from_bundle=False))


def zone_root(zone: str) -> Path:
    if zone not in ZONES:
        raise ValueError('UNKNOWN_ZONE:' + zone)
    return data_root() / DEFAULT_ROOT_NAME / zone


def assets_root() -> Path:
    return zone_root('assets')


def records_root() -> Path:
    return zone_root('records')


def cache_root() -> Path:
    return zone_root('cache')


def zone_budget(zone: str):
    """Fail-closed budget. cache keeps its already-authorized 4 GiB default.
    assets/records have NO default: an unauthorized zone returns None and
    writers must refuse new bytes — the proposal numbers are not an allowance."""
    policy = zone_root(zone) / 'zone_budget.json'
    try:
        value = json.loads(policy.read_bytes())
    except (OSError, ValueError):
        return DEFAULT_BUDGETS[zone]
    cap = value.get('bytes')
    if cap not in ALLOWED_BUDGETS[zone] or value.get('user_authorized') is not True:
        raise ValueError('UNAUTHORIZED_ZONE_BUDGET:' + zone)
    return cap


def zone_usage(zone: str) -> int:
    root = zone_root(zone)
    if not root.is_dir():
        return 0
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())


def zone_headroom(zone: str):
    """None means the zone has no authorized budget: refuse new writes while
    existing verified assets stay readable."""
    budget = zone_budget(zone)
    if budget is None:
        return None
    return budget - zone_usage(zone)


def ensure_zone(zone: str) -> Path:
    root = zone_root(zone)
    root.mkdir(parents=True, exist_ok=True)
    return root


class ZoneWriter:
    """Single-writer guard: one writer per asset folder at a time; consumers
    only read folders that carry a complete marker."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.lock = self.folder.with_name(self.folder.name + '.writing.lock')
        self.marker = self.folder / 'COMPLETE.json'

    def __enter__(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        try:
            self.lock.open('xb').close()
        except FileExistsError:
            raise ValueError('ASSET_BUSY:' + self.folder.name)
        return self

    def __exit__(self, *exc):
        try:
            if exc[0] is None and self.marker.is_file():
                self.lock.unlink()
            # On failure the lock stays: the half-written folder is never
            # consumed, and the next verified rebuild resumes from scratch.
        except OSError:
            pass
        return False


def complete(folder: Path) -> bool:
    """Only marker-complete, not-currently-written assets may be consumed."""
    folder = Path(folder)
    return (folder / 'COMPLETE.json').is_file() and not \
        (folder.parent / (folder.name + '.writing.lock')).exists()
