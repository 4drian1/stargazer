"""Bounded local producer. No downloads, large writes, or historical arrival fabrication."""
from datetime import datetime
from collections.abc import Mapping
from zoneinfo import ZoneInfo
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

from joint_shadow_compute_v2 import ComputeUnavailable
from joint_shadow_context_v1 import RiskSnapshot
from tdx_index_acceptance_v1 import read_binary

HERE = Path(__file__).resolve().parent
INDEX_PATHS = {'000001.SH': Path('D:/a股数据/沪指/sh999999.lc5'),
               '399006.SZ': Path('D:/a股数据/创业板指/sz399006.lc5')}
MAX_READ = 32 * 1024**3  # read traffic, NOT a disk-write allowance
MAX_WORKING_ESTIMATE = 1024**3
MAX_SINGLE_FILE = 32 * 1024**2


class _StockPrefixes(Mapping):
    """Second-pass verified reads, one stock at a time; never retain full frames."""
    def __init__(self, entries, read, cutoff):
        self.entries, self.read, self.cutoff = entries, read, cutoff

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __contains__(self, code):
        return code in self.entries

    def __getitem__(self, code):
        r = self.entries[code]
        frame = pd.read_parquet(io.BytesIO(self.read(r['path'], r['sha256'])))
        return frame[frame.date <= self.cutoff].copy()


def now_stamp():
    return int(datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d%H%M%S'))


def produce(cutoff, *, historical=False, engineering_boards=None, clock=now_stamp):
    """Read accepted local cache into a snapshot; forward mode never narrows boards.

    Historical mode is explicit research. Forward timestamps record this reader's
    actual byte observation, NOT first upstream arrival or an old file's mtime.
    Full-universe cost is checked before reading stock files; missing sources stay
    in membership, never silently replaced by a smaller comparison universe.
    """
    cutoff = int(cutoff)
    pd.to_datetime(str(cutoff), format='%Y%m%d%H%M%S', errors='raise')
    if engineering_boards and not historical:
        raise ComputeUnavailable('FORWARD_UNIVERSE_NARROWING_FORBIDDEN')
    if not historical and cutoff // 1000000 != clock() // 1000000:
        raise ComputeUnavailable('FORWARD_DAY_NOT_TODAY')
    if not historical and cutoff > clock():
        raise ComputeUnavailable('FUTURE_CUTOFF')
    used, sources = 0, {}
    counters = {'bytes_read': 0, 'read_cap': MAX_READ}
    def read(path, expected=None):
        nonlocal used
        path = Path(path)
        if str(path).startswith('\\\\'):
            raise ComputeUnavailable('LOCAL_FILES_ONLY')
        if used + path.stat().st_size > MAX_READ:
            raise ComputeUnavailable('LOCAL_READ_BUDGET_EXCEEDED')
        if path.stat().st_size > MAX_SINGLE_FILE:
            raise ComputeUnavailable('LOCAL_SINGLE_FILE_BUDGET_EXCEEDED')
        with path.open('rb') as stream:
            body = stream.read(min(MAX_SINGLE_FILE, MAX_READ - used) + 1)
        used += len(body)
        counters['bytes_read'] = used
        if used > MAX_READ or len(body) > MAX_SINGLE_FILE:
            raise ComputeUnavailable('LOCAL_READ_BUDGET_EXCEEDED')
        sha = hashlib.sha256(body).hexdigest()
        if expected is not None and sha != expected:
            raise ComputeUnavailable('LOCAL_SOURCE_DRIFT:' + path.name)
        sources[str(path)] = sha
        return body
    scope = json.loads(read(HERE / 'ths_full_concept_scope_20261017_v1.json'))
    catalog = pd.read_parquet(io.BytesIO(read(scope['source'], scope['source_sha256'])))
    catalog = catalog[(catalog.source == 'ths') & (catalog.type == 'concept')]
    if engineering_boards:
        catalog = catalog[catalog.code.isin(engineering_boards)]
        if set(catalog.code) != set(engineering_boards):
            raise ComputeUnavailable('ENGINEERING_BOARD_UNKNOWN')
    elif set(catalog.code) != {r['code'] for r in scope['boards']}:
        raise ComputeUnavailable('COMPARISON_UNIVERSE_DRIFT')
    boards = {r.code: list(r.symbols) for r in catalog.sort_values('code').itertuples()}
    indices, receipts = {}, {}
    def receipt(identity, path, clock_kind):
        receipts[identity] = dict(reference=str(path) + '#sha256=' + sources[str(path)],
            price_basis='RAW', clock=clock_kind, available_at=None if historical else clock(),
            availability_basis=('HISTORICAL_RECONSTRUCTION_ASSUMPTION' if historical
                                else 'LOCAL_READER_OBSERVED_BYTES_NOT_UPSTREAM_FIRST_ARRIVAL'))
    for symbol, path in INDEX_PATHS.items():
        frame = pd.DataFrame(read_binary(read(path)))
        prefix = frame[frame.date <= cutoff]
        if prefix.empty or int(prefix.date.max()) < cutoff:
            latest = None if prefix.empty else int(prefix.date.max())
            raise ComputeUnavailable('INDEX_STALE_AT_CUTOFF:' + symbol + ':latest=' + str(latest))
        indices[symbol] = prefix
        receipt(symbol, path, 'END_5M')
    # The same accepted index calendar, not a single stock's visible dates.
    days = sorted(int(d) for d in set(indices['000001.SH'].date // 1000000) if d >= 20250102)
    if len(days) < 2 or days[-1] != cutoff // 1000000:
        raise ComputeUnavailable('PREFIX_CALENDAR_INCOMPLETE')
    ledger = json.loads(read(HERE / 'relative_stall_5m_20261018_v1/raw_input_ledger.json'))
    by_code = {r['code']: r for r in ledger}
    codes = sorted({c for m in boards.values() for c in m})
    available = [c for c in codes if c in by_code]
    estimate = len(codes) * (64 + 16*24) * 8 + 256 * 1024**2
    if estimate > MAX_WORKING_ESTIMATE:
        raise ComputeUnavailable('WORKING_SET_BUDGET_REQUIRES_INCREMENTAL_CACHE')
    if used + 2*sum(Path(by_code[c]['path']).stat().st_size for c in available) > MAX_READ:
        raise ComputeUnavailable('LOCAL_READ_BUDGET_REQUIRES_INCREMENTAL_CACHE')
    # Pass 1 verifies the exact bytes and records when this reader actually saw
    # them. Pass 2 streams one verified stock at a time through the feature kernel.
    # Thus real arrival receipts precede observed_at without inventing timestamps.
    for code in available:
        entry = by_code[code]
        read(entry['path'], entry['sha256'])
        receipt(code, entry['path'], 'END_1M')
    stocks = _StockPrefixes({c: by_code[c] for c in available}, read, cutoff)
    observed = None if historical else clock()
    snapshot = RiskSnapshot(cutoff, observed, days, stocks, indices, boards, receipts,
        observed, 'SNAPSHOT_RECONSTRUCTION_RESEARCH' if historical else 'FORWARD_SNAPSHOT',
        observed_at=observed)
    audit = dict(source_hashes=sources, read_counters=counters, estimated_working_bytes=estimate,
        board_count=len(boards), declared_stock_count=len(codes), bound_stock_count=len(stocks),
        missing_stock_codes=[c for c in codes if c not in by_code],
        engineering_subset=bool(engineering_boards), historical=historical,
        source_policy='FROZEN_ACCEPTED_CACHE_NO_UNBOUND_NEW_SOURCE', observed_at=observed)
    return snapshot, audit
