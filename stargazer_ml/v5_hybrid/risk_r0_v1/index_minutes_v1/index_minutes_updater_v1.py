"""Index 5-minute mirror updater for Tongdaxin (TDX) .lc5 exports — index_minutes_v1/.

Hot-folder design (zero external network, read-only on source binaries):
  1. scan inbox dirs for *.lc5 (default: ./inbox and the user's TDX export root
     D:/a股数据 when present; more via --inbox);
  2. parse + validate every candidate (48 bars/day, exact 09:35..15:00 END grid,
     OHLC consistency, finite positive prices, monotonic unique timestamps);
  3. incremental append to the local mirror parquet (existing rows are never
     modified; overlap must match byte-identically, else the run REFUSES to write);
  4. write freshness.json (last_date / sha256 / per-source hashes) and report.json.

Known index identities (validated 2026-10-16, see tdx_index_acceptance_20261016_v1):
  sh999999.lc5  -> 000001.SH (上证综指)
  sz399006.lc5  -> 399006.SZ (创业板指)
Time labels are TDX convention 5-min bar END times (ASSUMED_END, not certified).
Raw volume/amount columns are kept for audit only — volume unit UNVERIFIED, do
not build volume rules on them (see TDX_INDEX_ACCEPTANCE_RESULT_20261016.md).

UI / evidence-pack freshness contract:
  get_index_freshness() -> {'last_date', 'stale', 'max_lag_days', ...}
  stale means mirror last_date < (today's trading day - 1 trading day).
  When stale: market_state fields must be treated as MISSING and hard-gate the
  evidence pack as EVIDENCE_INSUFFICIENT — never silently reuse stale values.

Usage:
  python index_minutes_updater_v1.py              # one-shot scan + update
  python index_minutes_updater_v1.py --watch 300  # poll every 300s (hot folder)
  python index_minutes_updater_v1.py --inbox DIR [--inbox DIR2]
  python -c "from index_minutes_updater_v1 import get_index_freshness as g; print(g())"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
MIRROR = HERE                      # mirror parquets live beside this script
INBOX_LOCAL = HERE / 'inbox'       # user drop folder
TDX_EXPORT_ROOT = Path(r'D:/a股数据')  # user's standing TDX export dir (if present)

# filename stem -> (symbol, display). Only identities already accepted on
# 2026-10-16 are bound here; anything else is reported as unrecognized, skipped.
KNOWN = {
    'sh999999': ('000001.SH', '上证综指'),
    'sz399006': ('399006.SZ', '创业板指'),
}
PARQUET_NAME = {'000001.SH': 'sse_5min.parquet', '399006.SZ': 'cyb_5min.parquet'}
STOCKDB_BASE = 'http://127.0.0.1:7899'   # local service only; used as trading-day ref
CALENDAR_PROBE_CODE = '600519'           # liquid stock: its 日k dates == trading days
BARS_PER_DAY = 48
END_OF_DAY = 150000  # 15:00:00 as HHMMSS

# Runtime fingerprint: bumped when the sync contract changes. This version is
# prefix-aware: explicit caller-supplied observation time, completed-only writes,
# in_progress tails recorded (never written), release gate, and cache re-judge.
RUNTIME_FINGERPRINT = 'index_minutes_updater_v1.prefix_aware.versioned_public_day.20261030'


def get_runtime_fingerprint() -> str:
    return RUNTIME_FINGERPRINT


def parse_lc5(body: bytes) -> list[dict]:
    """TDX .lc5 32-byte records: <HHfffffiI (date-packed, minute, OHLC f32,
    amount f32, volume int32, reserved). Same layout verified byte-by-byte in
    tdx_index_acceptance_v1.py (route #18.45, 2026-10-16)."""
    if len(body) % 32:
        raise ValueError('BINARY_RECORD_SIZE')
    rows = []
    for packed, minute, o, h, l, c, amount, volume, _reserved in struct.iter_unpack('<HHfffffiI', body):
        day = (packed // 2048 + 2004) * 10000 + (packed % 2048 // 100) * 100 + packed % 2048 % 100
        if not 0 <= minute < 1440:
            raise ValueError('BINARY_CLOCK')
        datetime.strptime(str(day), '%Y%m%d')  # raises on impossible dates
        if not (0 <= packed % 2048 // 100 <= 12):
            raise ValueError('BINARY_MONTH')
        ts = day * 1000000 + (minute // 60) * 10000 + minute % 60 * 100
        rows.append(dict(ts=ts, day=day, open=o, high=h, low=l, close=c,
                         amount=amount, raw_volume=volume))
    return rows


def grid(day: int, step: int = 5) -> list[int]:
    """TDX native 5m END grid: 09:35..11:30, 13:05..15:00 (48 slots)."""
    d = datetime.strptime(str(day), '%Y%m%d')
    return [int((d.replace(hour=hour) + timedelta(minutes=start + i)).strftime('%Y%m%d%H%M%S'))
            for hour, start in ((9, 30), (13, 0)) for i in range(step, 121, step)]


def validate(rows: list[dict]) -> dict:
    """Full per-day validation. Returns report; raises ValueError on hard fail."""
    if not rows:
        raise ValueError('EMPTY_SOURCE')
    report = dict(rows=len(rows), days=0, first_day=None, last_day=None,
                  per_day={}, issues=[])
    by_day: dict[int, list[dict]] = {}
    prev_ts = None
    for r in rows:
        ts = r['ts']
        if prev_ts is not None and ts <= prev_ts:
            raise ValueError('TIMESTAMP_NOT_STRICTLY_INCREASING')
        prev_ts = ts
        by_day.setdefault(r['day'], []).append(r)
    report['days'] = len(by_day)
    report['first_day'] = min(by_day)
    report['last_day'] = max(by_day)
    for day in sorted(by_day):
        recs = by_day[day]
        ts_list = [r['ts'] for r in recs]
        expected = grid(day, 5)
        entry = dict(bars=len(recs), grid_exact=(sorted(ts_list) == expected))
        if len(recs) != BARS_PER_DAY:
            entry['issue'] = 'BAR_COUNT_NOT_48'
            report['issues'].append((day, entry['issue'], len(recs)))
        if not entry['grid_exact']:
            entry['issue'] = 'GRID_MISMATCH'
            report['issues'].append((day, 'GRID_MISMATCH', None))
        bad_ohlc = bad_price = neg_va = 0
        for r in recs:
            o, h, l, c = r['open'], r['high'], r['low'], r['close']
            vals = (o, h, l, c, r['amount'])
            if any(not isinstance(v, (int, float)) or v != v or v in (float('inf'), float('-inf')) for v in vals):
                bad_price += 1
                continue
            if min(o, c, 1) <= 0 or r['amount'] < 0 or r['raw_volume'] < 0:
                bad_price += 1
                continue
            if not (l <= min(o, c) <= max(o, c) <= h):
                bad_ohlc += 1
        entry['bad_ohlc'] = bad_ohlc
        entry['bad_price'] = bad_price
        if bad_ohlc:
            report['issues'].append((day, 'OHLC_VIOLATION', bad_ohlc))
        if bad_price:
            report['issues'].append((day, 'INVALID_PRICE_OR_AMOUNT', bad_price))
        report['per_day'][day] = entry
    return report


def validate_prefix(rows, day, cutoff, as_of=None, force_full_day=False):
    """Prefix-aware per-day validator (port of the isolated research validator).

    Distinguishes a historical complete 48-bar day from a current-day continuous
    COMPLETE prefix; only COMPLETED bars (END <= as_of) count. in_progress tails
    (END > as_of) are reported but never treated as complete. The 48-bar whole-day
    requirement applies ONLY to historical / post-close; the adapter's
    date.max()==cutoff means a complete K-line exists at the requested cutoff.

    Returns a dict: verdict + gaps / beyond_records / in_progress / last_complete_ts.
    """
    if as_of is None:
        as_of = cutoff
    as_of_date = as_of // 1000000
    cutoff_day = cutoff // 1000000
    out = dict(verdict=None, reason='', day=day, gaps=[], beyond_records=[],
               in_progress=[], last_complete_ts=None, meets_cutoff=False, fully_closed=False)
    if day > as_of_date:
        out.update(verdict='REJECT_FUTURE_DATE', reason='examined day after observation date')
        return out
    # structural checks on INPUT order (do NOT sort first)
    raw = [int(r['ts']) for r in rows if r.get('day') == day]
    if len(raw) != len(set(raw)):
        out.update(verdict='REJECT_DUPLICATE', reason='duplicate ts within day', duplicates=True)
        return out
    if any(raw[i] >= raw[i + 1] for i in range(len(raw) - 1)):
        out.update(verdict='REJECT_ORDER', reason='out-of-order ts within day (input order)',
                   out_of_order=True)
        return out
    ts_list = sorted(raw)
    g = grid(day)
    gset = set(g)
    off_grid = [t for t in ts_list if t not in gset]
    if off_grid:
        out.update(verdict='REJECT_OFF_GRID', reason='bar ts not on 48-slot END grid', gaps=off_grid)
        return out
    present = set(ts_list)
    k = 0
    while k < len(g) and g[k] in present:
        k += 1
    missing = []
    j = k
    while j < len(g) and g[j] not in present:
        missing.append(g[j])
        j += 1
    has_beyond = j < len(g) and g[j] in present
    if has_beyond:
        beyond_records = sorted(t for t in ts_list if t not in set(g[:k]))
        out.update(verdict='REJECT_GAP', reason='gap / non-contiguous prefix',
                   gaps=missing, beyond_records=beyond_records)
        return out
    complete = [t for t in ts_list if t <= as_of]
    in_progress = [t for t in ts_list if t > as_of]
    out['in_progress'] = in_progress
    out['last_complete_ts'] = max(complete) if complete else None
    post_close = force_full_day or day < cutoff_day or (as_of % 1000000) >= END_OF_DAY
    if post_close:
        if len(complete) == len(g) and set(complete) == gset:
            out.update(verdict='POSTCLOSE_OK', fully_closed=True,
                       reason='48 completed bars + 15:00 last')
        else:
            out.update(verdict='POSTCLOSE_FAIL',
                       reason='requires 48 COMPLETED bars (END<=as_of) + 15:00 last')
        return out
    if not complete:
        out.update(verdict='NO_PREFIX', reason='no complete K-line yet today')
        return out
    out['meets_cutoff'] = out['last_complete_ts'] >= cutoff
    out.update(verdict='OK_CURRENT_PREFIX' if out['meets_cutoff'] else 'WAIT_FRESHNESS',
               reason='continuous complete prefix' + ('' if out['meets_cutoff'] else '; behind cutoff'))
    return out


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def find_candidates(inboxes: list[Path]) -> dict[str, list[Path]]:
    """inbox -> {symbol: [paths]} for recognized .lc5 stems.

    An inbox may be a directory (scanned recursively for *.lc5) or a single
    .lc5 file (used by the desktop entry to bind exactly the two index sources
    without enumerating an entire fzline folder).
    """
    found: dict[str, list[Path]] = {}
    for box in inboxes:
        if box.is_dir():
            items = sorted(box.rglob('*.lc5'))
        elif box.is_file() and box.suffix.lower() == '.lc5':
            items = [box]
        else:
            items = []
        for p in items:
            symbol = KNOWN.get(p.stem.lower())
            if symbol is None:
                continue  # unrecognized index files are ignored (reported via --verbose)
            found.setdefault(symbol[0], []).append(p)
    return found


def load_mirror(symbol: str, mirror_dir=None) -> tuple[pa.Table | None, dict]:
    base = Path(mirror_dir) if mirror_dir else MIRROR
    path = base / PARQUET_NAME[symbol]
    if not path.exists():
        return None, {}
    table = pq.read_table(path)
    meta_path = path.with_suffix('.meta.json')
    meta = json.loads(meta_path.read_bytes()) if meta_path.exists() else {}
    return table, meta


def merge_and_write(symbol: str, incoming: list[dict], source_info: list[dict], report: dict,
                    mirror_dir=None, day_release=False, pending_tail_end=None,
                    as_of=None, current_day=None, last_complete_ts=None,
                    time_semantics='TDX_5MIN_BAR_END_ASSUMED_NOT_CERTIFIED', public_revision=None):
    """Incremental append: existing rows untouched; overlapping rows must match.
    Only COMPLETED bars (caller-filtered to ts <= as_of) are written. in_progress
    tails are recorded in the sidecar (pending_tail_end) and never written here, so a
    later tail change cannot trigger an OVERLAP_CONFLICT on a previously-mirrored bar.
    """
    base = Path(mirror_dir) if mirror_dir else MIRROR
    path = base / PARQUET_NAME[symbol]
    table, prior_meta = load_mirror(symbol, mirror_dir=base)
    old_rows = []
    if table is not None:
        cols = table.to_pydict()
        old_rows = [dict(zip(cols, vals)) for vals in zip(*cols.values())]
    old_ts = {r['ts']: r for r in old_rows}
    conflict = None
    revisions=[]
    history_days=(public_revision or {}).get('history_days',[])
    if history_days:
        owned=[r['day'] for r in old_rows if r.get('amount') is None and r.get('raw_volume') is None]
        for d in history_days:
            old=[r for r in old_rows if r['day']==d];new=[r for r in incoming if r['day']==d]
            if (not owned or not min(owned)<=d<current_day or len(old)>=48 or
                any(r.get('amount') is not None or r.get('raw_volume') is not None for r in old) or
                [r['ts'] for r in new]!=grid(d)):
                raise ValueError('PUBLIC_HISTORY_COMPLETION_SCOPE')
    if public_revision is not None:
        valid=(prior_meta.get('source_contract')=='SINA_INDEX_5M_PRICE_ONLY_V1'
            and current_day==int(as_of)//1000000 and public_revision.get('day')==current_day
            and len(source_info)==1 and source_info[0].get('provider')=='SINA_PUBLIC'
            and all(r['day'] in [current_day,*history_days] and r['ts']<=as_of and r.get('amount') is None and r.get('raw_volume') is None for r in incoming))
        if not valid:raise ValueError('PUBLIC_DAY_REVISION_SCOPE')
        if path.exists() and sha256(path.read_bytes())!=prior_meta.get('sha256'):
            raise ValueError('PUBLIC_DAY_REVISION_OLD_HASH')
        for r in incoming:
            prev=old_ts.get(r['ts'])
            if prev is not None and any(prev[k]!=r[k] for k in ('open','high','low','close')):
                revisions.append(dict(previous=prev,current=r))
                old_ts[r['ts']]=r
        old_rows=[old_ts[r['ts']] for r in old_rows]
    new_rows = []
    for r in incoming:
        prev = old_ts.get(r['ts'])
        if prev is not None:
            same = all(prev[k] == r[k] for k in ('day', 'open', 'high', 'low', 'close', 'amount', 'raw_volume'))
            if not same:
                conflict = r['ts']
                break
        else:
            new_rows.append(r)
    if conflict is not None:
        raise ValueError(f'OVERLAP_CONFLICT_AT_{conflict}: source disagrees with mirror; refusing to write')
    merged = sorted(old_rows + new_rows, key=lambda r: r['ts'])
    payload = pa.Table.from_pylist(merged) if merged else pa.table(
        {k: pa.array([], type=pa.int64()) for k in ('ts', 'day')} |
        {k: pa.array([], type=pa.float64()) for k in ('open', 'high', 'low', 'close', 'amount')} |
        {'raw_volume': pa.array([], type=pa.int64())})
    sink = pa.BufferOutputStream()
    pq.write_table(payload, sink, compression='zstd')
    body = sink.getvalue().to_pybytes()
    revision_ref=None
    if revisions or history_days:
        import gzip
        folder=Path(public_revision['archive_dir'])
        if folder.resolve()!=Path(source_info[0]['raw_archive']).resolve().parent:
            raise ValueError('PUBLIC_DAY_REVISION_ARCHIVE')
        proof=dict(as_of=as_of,symbol=symbol,day=current_day,previous_sha=prior_meta['sha256'],
                   next_sha=sha256(body),source=source_info[0],revisions=revisions,completed_history_days=history_days,added_timestamps=[r['ts'] for r in new_rows])
        receipt=folder/(symbol.replace('.','_')+'_revision_'+prior_meta['sha256'][:12]+'_'+sha256(body)[:12]+'.json.gz')
        if not receipt.exists():
            with receipt.open('xb') as stream:stream.write(gzip.compress(json.dumps(proof,ensure_ascii=False).encode(),mtime=0))
        revision_ref=str(receipt)
    tmp = path.with_suffix('.parquet.tmp')
    tmp.write_bytes(body)
    tmp.replace(path)  # atomic on same volume
    days = sorted({r['day'] for r in merged})
    sidecar = dict(symbol=symbol, rows=len(merged), days=len(days),
                   first_date=days[0] if days else None, last_date=days[-1] if days else None,
                   sha256=sha256(body), parquet_bytes=len(body),
                   time_semantics=time_semantics,
                   volume_verified=False,
                   runtime_fingerprint=get_runtime_fingerprint(),
                   day_release=day_release, pending_tail_end=pending_tail_end,
                   as_of=as_of, current_day=current_day, last_complete_ts=last_complete_ts,
                   sources=source_info, public_revision_receipt=revision_ref)
    path.with_suffix('.meta.json').write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding='utf-8')
    return dict(symbol=symbol, appended=len(new_rows), total_rows=len(merged),
                first_date=sidecar['first_date'], last_date=sidecar['last_date'],
                sha256=sidecar['sha256'])


def trading_days_via_stockdb(today: datetime, back: int = 20) -> list[int] | None:
    """Trading-day chain from the LOCAL stockdb service (日k dates of a liquid
    stock == trading days). Returns sorted YYYYMMDD list ending at the latest
    trading day <= today, or None when the service is unreachable."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    days = []
    d = today
    tried = 0
    while len(days) < back and tried < 45:  # bound the walk (holidays/long weekends)
        if d.weekday() < 5:
            tried += 1
            key = int(d.strftime('%Y%m%d'))
            url = (STOCKDB_BASE + '/?cmd=get&t='
                   + urllib.parse.quote(f'日k:{CALENDAR_PROBE_CODE}:{key}'))
            try:
                with opener.open(urllib.request.Request(url, headers={'Accept': 'application/json'}),
                                 timeout=3) as resp:
                    raw = resp.read()
                if raw.strip():
                    days.append(key)
            except Exception:
                return None  # service down: caller falls back to heuristic
        d -= timedelta(days=1)
    return sorted(days, reverse=True)  # newest first


def trading_days_weekday_heuristic(today: datetime, back: int = 20) -> list[int]:
    """Fallback: weekdays only (holidays NOT excluded — flagged in calendar_source)."""
    days, d = [], today
    while len(days) < back:
        if d.weekday() < 5:
            days.append(int(d.strftime('%Y%m%d')))
        d -= timedelta(days=1)
    return days


def get_index_freshness(mirror_dir: str | Path | None = None) -> dict:
    """UI/status contract. stale = mirror last_date < (today's trading day - 1).

    max_lag_days = trading days of lag between last_date and today's trading day.
    Downstream rule when stale=True: treat market_state as MISSING and let the
    evidence-pack hard gate record EVIDENCE_INSUFFICIENT — never reuse stale bars.
    """
    base = Path(mirror_dir) if mirror_dir else MIRROR
    today = datetime.now()
    chain = trading_days_via_stockdb(today)
    calendar_source = 'stockdb_local_daily'
    if chain is None:
        chain = trading_days_weekday_heuristic(today)
        calendar_source = 'weekday_heuristic_holidays_not_excluded'
    today_td = chain[0]
    required_latest = chain[1] if len(chain) > 1 else None  # today-1 trading day
    symbols = {}
    last_dates = []
    for symbol, name in KNOWN.values():
        meta_path = base / (PARQUET_NAME[symbol].replace('.parquet', '.meta.json'))
        meta = json.loads(meta_path.read_bytes()) if meta_path.exists() else {}
        last = meta.get('last_date')
        last_dates.append(last)
        symbols[symbol] = dict(name=name,
                               last_date=last, rows=meta.get('rows'),
                               sha256=meta.get('sha256'), volume_verified=False)
    last_date = min(d for d in last_dates if d) if any(last_dates) else None
    if last_date is None:
        stale, lag = True, None
    else:
        lag = sum(1 for d in chain if d > last_date)
        stale = required_latest is not None and last_date < required_latest
    return dict(last_date=last_date, stale=stale, max_lag_days=lag,
                today_trading_day=today_td, required_latest=required_latest,
                calendar_source=calendar_source,
                checked_at=datetime.now().isoformat(timespec='seconds'),
                symbols=symbols,
                rule='stale <=> last_date < today_trading_day - 1 trading day')


def run_update(inboxes, as_of, verbose=False, mirror_dir=None) -> dict:
    """Prefix-aware mirror update.

    as_of: caller-supplied Shanghai wall-clock YYYYMMDD HHMMSS (observation time).
    Only COMPLETED bars (END <= as_of) are written; in_progress tails are recorded in
    the sidecar (pending_tail_end) and never written, so a later tail change cannot
    trigger an OVERLAP_CONFLICT on an already-mirrored bar. Release (risk evidence
    complete) requires BOTH indices' current day to be POSTCLOSE_OK (48 completed
    bars + 15:00 last) -- never just a date check. A UNCHANGED file signature does
    NOT skip re-judgement when a pending tail's END has been crossed by as_of.
    """
    base = Path(mirror_dir) if mirror_dir else MIRROR
    base.mkdir(parents=True, exist_ok=True)
    cache_path = base / 'sync_cache.json'
    cache = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
    candidates = find_candidates(inboxes)
    run_report = dict(run_at=datetime.now().isoformat(timespec='seconds'),
                      inboxes=[str(b) for b in inboxes], as_of=as_of,
                      unrecognized=[], symbols={}, status='OK')
    for box in inboxes:
        if box.is_dir():
            for p in box.rglob('*.lc5'):
                if p.stem.lower() not in KNOWN:
                    run_report['unrecognized'].append(str(p))
    as_of_date = as_of // 1000000
    release = {}
    new_cache = {}
    known_by_symbol = {sym: (stem, disp) for stem, (sym, disp) in KNOWN.items()}
    for symbol in PARQUET_NAME:
        stem, disp = known_by_symbol[symbol]
        paths = candidates.get(symbol, [])
        if not paths:
            run_report['symbols'][symbol] = dict(status='NO_SOURCE_FILE')
            release[symbol] = dict(symbol=symbol, verdict='NO_SOURCE_FILE', release=False)
            new_cache[symbol] = dict(file_sha=None, pending_tail_end=None, release=False,
                                    cache_date=as_of_date)
            continue
        parsed = []
        refused = False
        for p in paths:
            body = p.read_bytes()
            try:
                rows = parse_lc5(body)
                rep = validate(rows)  # OHLC/price/identity/negative/duplicate checks
            except ValueError as exc:
                run_report['symbols'][symbol] = dict(status='PARSE_OR_VALIDATE_ERROR', error=str(exc))
                release[symbol] = dict(symbol=symbol, verdict='PARSE_OR_VALIDATE_ERROR', release=False)
                new_cache[symbol] = dict(file_sha=sha256(body), pending_tail_end=None,
                                        release=False, cache_date=as_of_date)
                refused = True
                break
            parsed.append(dict(path=p, body=body, rows=rows, report=rep,
                               sha=sha256(body), bytes=len(body)))
        if refused:
            continue
        parsed.sort(key=lambda item: item['report']['last_day'], reverse=True)
        best = parsed[0]
        # hard quality failures from the original validator (bad price / OHLC)
        hard = [iss for iss in best['report']['issues']
                if iss[1] in ('OHLC_VIOLATION', 'INVALID_PRICE_OR_AMOUNT')]
        if hard:
            run_report['symbols'][symbol] = dict(status='REFUSED_BAD_PRICE', issues=hard)
            release[symbol] = dict(symbol=symbol, verdict='REJECT_BAD_PRICE', release=False)
            new_cache[symbol] = dict(file_sha=best['sha'], pending_tail_end=None,
                                    release=False, cache_date=as_of_date)
            continue
        prev = cache.get(symbol, {})
        pending_prev = prev.get('pending_tail_end')
        prev_cache_date = prev.get('cache_date')
        unchanged = (prev.get('file_sha') == best['sha'])
        # Skip only if file unchanged AND same observation date AND no pending tail's
        # END has been crossed by as_of. Cross-day cache entries are NEVER reused: a
        # prior day's release verdict must not carry into the new trading day.
        if unchanged and prev_cache_date == as_of_date and not (pending_prev and pending_prev <= as_of):
            # Same day + same data version: keep the previously verified close
            # qualification. "No new data this run" (data_unchanged) is separate from
            # the business release, so a second sync must NOT downgrade POSTCLOSE_OK.
            run_report['symbols'][symbol] = dict(status='UNCHANGED', pending_tail_end=pending_prev,
                                                release=prev.get('release', False),
                                                data_unchanged=True)
            release[symbol] = dict(symbol=symbol, verdict=prev.get('verdict'),
                                   release=prev.get('release', False),
                                   last_complete_ts=prev.get('last_complete_ts'),
                                   mirror_path=prev.get('mirror_path'),
                                   mirror_sha=prev.get('mirror_sha'),
                                   data_unchanged=True)
            new_cache[symbol] = prev
            continue
        rows = best['rows']
        by_day = {}
        for r in rows:
            by_day.setdefault(r['day'], []).append(r)
        completed_rows = []
        current_day = max(by_day) if by_day else None
        pending_tail_end = None
        day_release = False
        verdict = None
        refuse_reason = None
        if current_day is not None:
            if current_day > as_of_date:
                verdict = 'REJECT_FUTURE_DATE'
                refuse_reason = 'examined day after observation date'
            else:
                for d, drows in by_day.items():
                    vp = validate_prefix(drows, d, as_of, as_of)
                    if d > as_of_date:
                        verdict = 'REJECT_FUTURE_DATE'
                        refuse_reason = 'future day in source'
                        break
                    if d < as_of_date:
                        if vp['verdict'] != 'POSTCLOSE_OK':
                            continue  # historical incomplete -> do not write partial
                        completed_rows.extend(r for r in drows if r['ts'] <= as_of)
                        continue
                    # d == as_of_date (current)
                    verdict = vp['verdict']
                    if vp['verdict'] in ('OK_CURRENT_PREFIX', 'WAIT_FRESHNESS', 'POSTCLOSE_OK'):
                        completed_rows.extend(r for r in drows if r['ts'] <= as_of)
                        # Frontier = EARLIEST pending bar END. Re-judge fires as soon as
                        # as_of crosses it, so multiple pending tails advance bar-by-bar
                        # instead of waiting for the last (15:00) bar.
                        pending_tail_end = min(vp['in_progress']) if vp['in_progress'] else None
                        day_release = (vp['verdict'] == 'POSTCLOSE_OK')
                    else:
                        refuse_reason = vp['reason']
                        break
        if refuse_reason is not None:
            run_report['symbols'][symbol] = dict(status='REFUSED_' + verdict, reason=refuse_reason)
            release[symbol] = dict(symbol=symbol, verdict=verdict, release=False)
            new_cache[symbol] = dict(file_sha=best['sha'], pending_tail_end=None,
                                    release=False, cache_date=as_of_date)
            continue
        source_info = [dict(path=str(x['path']), sha256=x['sha'], bytes=x['bytes']) for x in parsed]
        try:
            result = merge_and_write(
                symbol, completed_rows, source_info, best['report'], mirror_dir=base,
                day_release=day_release, pending_tail_end=pending_tail_end, as_of=as_of,
                current_day=current_day,
                last_complete_ts=max((r['ts'] for r in completed_rows), default=None))
        except ValueError as exc:
            run_report['symbols'][symbol] = dict(status='REFUSED', error=str(exc))
            run_report['status'] = 'PARTIAL_REFUSAL'
            release[symbol] = dict(symbol=symbol,
                                   verdict='OVERLAP_CONFLICT' if 'OVERLAP' in str(exc) else 'REFUSED',
                                   release=False)
            new_cache[symbol] = dict(file_sha=best['sha'], pending_tail_end=pending_tail_end,
                                    release=False, cache_date=as_of_date)
            continue
        result['source'] = str(best['path'])
        result['source_sha256'] = best['sha']
        result['validation'] = dict(days=best['report']['days'],
                                    first_day=best['report']['first_day'],
                                    last_day=best['report']['last_day'],
                                    issues=len(best['report']['issues']))
        run_report['symbols'][symbol] = result
        parquet_path = base / PARQUET_NAME[symbol]
        mirror_sha = sha256(parquet_path.read_bytes()) if parquet_path.exists() else None
        release[symbol] = dict(symbol=symbol, verdict=verdict, release=day_release,
                               last_complete_ts=result.get('last_complete_ts'),
                               mirror_path=str(parquet_path), mirror_sha=mirror_sha)
        new_cache[symbol] = dict(file_sha=best['sha'], pending_tail_end=pending_tail_end,
                                last_complete_ts=result.get('last_complete_ts'),
                                release=day_release, cache_date=as_of_date,
                                verdict=verdict, mirror_sha=mirror_sha,
                                mirror_path=str(parquet_path))
    # cache
    cache_path.write_text(json.dumps(new_cache, ensure_ascii=False, indent=2), encoding='utf-8')
    # release gate: evidence complete only if BOTH indices released the current day
    both_released = bool(release) and all(r.get('release') for r in release.values())
    release_status = dict(as_of=as_of, as_of_date=as_of_date, symbols=release,
                         evidence_complete=bool(both_released),
                         runtime_fingerprint=get_runtime_fingerprint(),
                         generated_at=datetime.now().isoformat(timespec='seconds'))
    (base / 'release_status.json').write_text(
        json.dumps(release_status, ensure_ascii=False, indent=2), encoding='utf-8')
    run_report['release_status'] = release_status
    # freshness + report_latest
    fresh = get_index_freshness(base)
    (base / 'freshness.json').write_text(
        json.dumps(dict(last_date=fresh['last_date'], stale=fresh['stale'],
                        max_lag_days=fresh['max_lag_days'],
                        today_trading_day=fresh['today_trading_day'],
                        required_latest=fresh['required_latest'],
                        calendar_source=fresh['calendar_source'],
                        symbols={s: dict(last_date=v['last_date'], sha256=v['sha256'])
                                 for s, v in fresh['symbols'].items()},
                        updated_at=fresh['checked_at']),
                   ensure_ascii=False, indent=2), encoding='utf-8')
    run_report['freshness'] = {k: fresh[k] for k in ('last_date', 'stale', 'max_lag_days')}
    # runtime fingerprint artifact for independent read-back
    (base / 'runtime_fingerprint.json').write_text(
        json.dumps(dict(runtime_fingerprint=get_runtime_fingerprint(),
                        generated_at=datetime.now().isoformat(timespec='seconds')),
                   ensure_ascii=False, indent=2), encoding='utf-8')
    (base / 'report_latest.json').write_text(
        json.dumps(run_report, ensure_ascii=False, indent=2), encoding='utf-8')
    return run_report


def main(argv=None):
    ap = argparse.ArgumentParser(description='TDX .lc5 -> index 5-min mirror updater')
    ap.add_argument('--inbox', action='append', type=Path, default=None,
                    help='extra inbox dir to scan (repeatable)')
    ap.add_argument('--watch', type=float, default=None,
                    help='poll interval seconds; omit for one-shot run')
    ap.add_argument('--as-of', type=int, default=None,
                    help='Shanghai observation time YYYYMMDD HHMMSS (required for prefix-aware update; '
                         'defaults to host local clock if omitted)')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args(argv)
    inboxes = [INBOX_LOCAL, TDX_EXPORT_ROOT] + (args.inbox or [])
    as_of = args.as_of
    if as_of is None:
        as_of = int(datetime.now().strftime('%Y%m%d%H%M%S'))  # host local clock as Shanghai proxy
    while True:
        rep = run_update(inboxes, as_of, verbose=args.verbose)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        if args.watch is None:
            break
        time.sleep(args.watch)


if __name__ == '__main__':
    main()
