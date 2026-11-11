"""Native TDX 5-minute (.lc5) reader — v2 (route 18.175 guard fixes).

Fixes vs v1 (review 18.175 P1-A / P1-B):
  A. Bad values no longer pass as a complete day:
     - original record order is validated (v1 sorted first, hiding reversal);
     - a real calendar date is constructed (v1 only checked month 1-12 / day 1-31,
       so 20260230 passed);
     - negative/non-finite low, and any bad OHLC/amount/volume, make the day
       INELIGIBLE (v1 counted them but still admitted the day).
  B. The 5-min slot is no longer floor-divided: a minute must be an exact
     multiple of 5 AND on the 09:35/13:05 grid, so 09:36 cannot be folded into
     the 09:35 slot to fabricate a "complete" 15-min bar.

Read-only on source binaries; re-read sha256 proves read-immutability.
"""
from __future__ import annotations

import hashlib
import struct
from datetime import date as _date
from datetime import datetime, timedelta
from pathlib import Path

LC5_RECORD_SIZE = 32
GRID_STEP = 5
BARS_PER_DAY = 48
_SESSION_ANCHORS = ((9, 30), (13, 0))
_FIELDS = ('open', 'high', 'low', 'close', 'amount')
_PRICE_FIELDS = ('open', 'high', 'low', 'close')


def _valid_calendar_day(day: int) -> bool:
    """Real calendar check (v1 only range-checked month/day)."""
    y, m, d = day // 10000, (day % 10000) // 100, day % 100
    if not (1990 <= y <= 2200):
        return False
    try:
        _date(y, m, d)
    except ValueError:
        return False   # e.g. 20260230
    return True


# --------------------------------------------------------------------------- #
# Binary parse
# --------------------------------------------------------------------------- #
def parse_lc5(body: bytes) -> list[dict]:
    """TDX .lc5 32-byte records: <HHfffffiI (packed date, minute, OHLC f32,
    amount f32, volume i32, reserved u32)."""
    if len(body) % LC5_RECORD_SIZE:
        raise ValueError('BINARY_RECORD_SIZE')
    rows: list[dict] = []
    for packed, minute, o, h, l, c, amount, volume, _reserved in struct.iter_unpack(
            '<HHfffffiI', body):
        day = ((packed // 2048 + 2004) * 10000
               + (packed % 2048 // 100) * 100 + (packed % 2048) % 100)
        if not 0 <= minute < 1440:
            raise ValueError('BINARY_CLOCK')
        if not _valid_calendar_day(day):
            raise ValueError('BINARY_DATE')
        ts = day * 1000000 + (minute // 60) * 10000 + (minute % 60) * 100
        rows.append(dict(ts=ts, day=day, minute=minute,
                         open=float(o), high=float(h), low=float(l), close=float(c),
                         amount=float(amount), raw_volume=int(volume)))
    return rows


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
def grid_5min(day: int) -> list[int]:
    """Exact 48-slot END grid: 09:35..11:30 (24) and 13:05..15:00 (24)."""
    base = datetime.strptime(str(day), '%Y%m%d')
    out = []
    for hour, start in _SESSION_ANCHORS:
        for i in range(GRID_STEP, 121, GRID_STEP):
            t = base.replace(hour=hour) + timedelta(minutes=start + i)
            out.append(int(t.strftime('%Y%m%d%H%M%S')))
    return out


def grid_minutes() -> set[int]:
    """Minutes-of-day that are valid 5-min END labels."""
    out = set()
    for base, n in ((9 * 60 + 30, 24), (13 * 60, 24)):
        for i in range(1, n + 1):
            out.add(base + GRID_STEP * i)
    return out


_GRID_MINUTES = grid_minutes()


def slot_of_minute(minute: int) -> int:
    """5-min slot (0..47) or -1. Strict: must be a multiple of 5 AND on grid,
    so 09:36 (576) is rejected instead of folding into slot 0."""
    if minute % GRID_STEP != 0 or minute not in _GRID_MINUTES:
        return -1
    if 575 <= minute <= 690:
        return (minute - 575) // 5
    if 785 <= minute <= 900:
        return 24 + (minute - 785) // 5
    return -1


def row_values_ok(r: dict) -> bool:
    """Per-row value gate: all fields finite; prices > 0; OHLC ordering;
    amount/volume non-negative. Catches negative low and NaN prices."""
    try:
        vals = [float(r[f]) for f in _FIELDS]
        vol = int(r['raw_volume'])
    except (TypeError, ValueError, KeyError):
        return False
    if any(v != v or v in (float('inf'), float('-inf')) for v in vals):
        return False                      # NaN / inf
    o, h, l, c, amt = vals
    if min(o, h, l, c) <= 0 or amt < 0 or vol < 0:
        return False                      # negative low, non-positive price
    if not (l <= min(o, c) <= max(o, c) <= h):
        return False                      # OHLC violation
    return True


def row_ts_consistent(r: dict) -> bool:
    """ts / day / minute must agree with each other."""
    if r['day'] <= 0 or not _valid_calendar_day(r['day']):
        return False
    if not 0 <= r['minute'] < 1440:
        return False
    return r['ts'] == (r['day'] * 1000000 + (r['minute'] // 60) * 10000
                       + (r['minute'] % 60) * 100)


# --------------------------------------------------------------------------- #
# File read (read-immutability)
# --------------------------------------------------------------------------- #
def read_lc5_file(path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    body = p.read_bytes()
    sha_before = hashlib.sha256(body).hexdigest()
    rows = parse_lc5(body)
    if hashlib.sha256(p.read_bytes()).hexdigest() != sha_before:
        raise RuntimeError('SOURCE_MUTATED_DURING_READ')
    return dict(path=str(p), bytes=len(body), sha256=sha_before,
                read_immutable=True, mtime_ns=p.stat().st_mtime_ns, rows=rows)


# --------------------------------------------------------------------------- #
# Per-day validation (strict)
# --------------------------------------------------------------------------- #
def validate_day(rows: list[dict], day: int) -> dict:
    """Strict per-day check. `eligible` is the single gate for feature use.

    eligible = 48 bars AND original order strictly increasing AND no duplicate
    ts AND ts sequence exactly equals the 48-slot grid AND zero bad rows.
    """
    ts_seq = [r['ts'] for r in rows]          # ORIGINAL order, never sorted
    original_monotonic = all(ts_seq[i] < ts_seq[i + 1]
                             for i in range(len(ts_seq) - 1))
    duplicate_ts = len(ts_seq) != len(set(ts_seq))
    expected = grid_5min(day)
    grid_exact = (ts_seq == expected)          # exact membership AND order
    bad_ohlc = bad_price = bad_ts = 0
    for r in rows:
        if not row_ts_consistent(r):
            bad_ts += 1
            continue
        if not row_values_ok(r):
            if not (l_le_c(r)):
                bad_ohlc += 1
            else:
                bad_price += 1
    bad_rows = bad_ohlc + bad_price + bad_ts
    return dict(bars=len(rows), original_order_monotonic=original_monotonic,
                duplicate_ts=duplicate_ts, grid_exact=grid_exact,
                bad_rows=bad_rows, bad_ohlc=bad_ohlc, bad_price=bad_price,
                bad_ts=bad_ts,
                eligible=bool(len(rows) == BARS_PER_DAY and original_monotonic
                              and not duplicate_ts and grid_exact
                              and bad_rows == 0))


def validate_prefix(rows: list[dict], day: int, cutoff: int) -> dict:
    """Strict prefix gate for a day replayed UP TO `cutoff` (inclusive).

    Used when the session is not finished (live intraday replay) or, with
    cutoff = session end, it degenerates to the full 48-bar requirement.

    The kept bars (ts <= cutoff, same day, original file order) must:
      * be non-empty -- at least one completed slot exists before cutoff;
      * form EXACTLY the continuous grid prefix ending at the cutoff slot,
        i.e. ts_seq == [g for g in grid_5min(day) if g <= cutoff]; this single
        equality rejects missing slots, out-of-order rows, duplicates, off-grid
        bars and any bar past the cutoff (they all make the sequences differ);
      * have every row structurally valid (consistent ts/day/minute and sane
        OHLC/amount/volume) so negative low, NaN and OHLC violations are also
        rejected.
    """
    if not _valid_calendar_day(day):
        return dict(eligible=False, reason='BAD_DAY')
    expected = [g for g in grid_5min(day) if g <= cutoff]
    if not expected:
        return dict(eligible=False, reason='NO_SLOT_BEFORE_CUTOFF')
    kept = [r for r in rows if r['day'] == day and r['ts'] <= cutoff]
    ts_seq = [r['ts'] for r in kept]            # ORIGINAL order, never sorted
    if ts_seq != expected:                       # gap / dup / reorder / off-grid
        return dict(eligible=False, reason='PREFIX_NOT_CONTINUOUS')
    bad = 0
    for r in kept:
        if not (row_ts_consistent(r) and row_values_ok(r)):
            bad += 1
    if bad:
        return dict(eligible=False, reason='BAD_ROWS', bad_rows=bad)
    return dict(eligible=True, bars=len(kept), expected=len(expected))


def l_le_c(r: dict) -> bool:
    """True when the row's OHLC ordering holds (used to classify bad rows)."""
    try:
        o, h, l, c = (float(r[f]) for f in _PRICE_FIELDS)
    except (TypeError, ValueError, KeyError):
        return False
    finite = all(v == v and v not in (float('inf'), float('-inf'))
                 for v in (o, h, l, c))
    return bool(finite and l <= min(o, c) <= max(o, c) <= h)


def classify_file(rows: list[dict], min_history_days: int = 20) -> dict:
    """Categorize a parsed file. Uses `eligible` (never bad-value days).

    Status: COMPLETE / DOWNLOADING / INSUFFICIENT_HISTORY / MISSING_SLOT.
    Anomaly counts are returned so build_quality can report them (not 0).
    """
    by_day: dict[int, list[dict]] = {}
    for r in rows:
        by_day.setdefault(r['day'], []).append(r)
    days = sorted(by_day)
    day_reports = {d: validate_day(by_day[d], d) for d in days}
    eligible_days = [d for d in days if day_reports[d]['eligible']]
    anomaly = dict(bad_rows=0, bad_ohlc=0, bad_price=0, bad_ts=0,
                   duplicate_ts=0, out_of_order=0, off_grid=0)
    for d, rep in day_reports.items():
        for k in ('bad_rows', 'bad_ohlc', 'bad_price', 'bad_ts'):
            anomaly[k] += rep[k]
        if rep['duplicate_ts']:
            anomaly['duplicate_ts'] += 1
        if not rep['original_order_monotonic']:
            anomaly['out_of_order'] += 1
        if not rep['grid_exact']:
            anomaly['off_grid'] += rep['bars']
    last_day = days[-1] if days else None
    last_complete = bool(last_day is not None and last_day in eligible_days)
    historical_incomplete = [d for d in days[:-1] if d not in eligible_days]
    statuses = []
    if last_day is not None and not last_complete:
        statuses.append('DOWNLOADING')
    if len(eligible_days) < min_history_days:
        statuses.append('INSUFFICIENT_HISTORY')
    if historical_incomplete:
        statuses.append('MISSING_SLOT')
    if not statuses:
        statuses.append('COMPLETE')
    if 'DOWNLOADING' in statuses:
        primary = 'DOWNLOADING'
    elif 'MISSING_SLOT' in statuses:
        primary = 'MISSING_SLOT'
    elif 'INSUFFICIENT_HISTORY' in statuses:
        primary = 'INSUFFICIENT_HISTORY'
    else:
        primary = 'COMPLETE'
    feature_days = [d for d in eligible_days if d != last_day or last_complete]
    return dict(status=primary, all_statuses=statuses, days_total=len(days),
                eligible_days=eligible_days, feature_days=feature_days,
                last_day=last_day, last_complete=last_complete,
                historical_incomplete=historical_incomplete,
                anomaly=anomaly, day_reports=day_reports)


# --------------------------------------------------------------------------- #
# Optional scale cross-check against an independent daily close
# --------------------------------------------------------------------------- #
def cross_check_scale(code: str, last_close_by_day: dict,
                      daily_parts: str | None = None) -> dict:
    """Compare a parsed 5-min last-bar close with the daily parquet close on
    the most recent COMMON day (requiring an exact last-day match would make
    this guard vacuous)."""
    import os

    if daily_parts is None:
        daily_parts = os.environ.get(
            'STARGAZER_DAILY_PARTS', r'D:/a股数据/data_daily_parts')
    p = Path(daily_parts)
    if not p.is_dir():
        return dict(verdict='SKIPPED', reason='daily_parts_absent')
    try:
        import pandas as pd
        c6 = str(code).split('.')[0].zfill(6)
        for f in sorted(p.glob('part_*.parquet'), reverse=True):
            df = pd.read_parquet(f, filters=[('code', '=', c6)],
                                 columns=['date', 'close'])
            if not len(df):
                continue
            for _, row in df.sort_values('date', ascending=False).iterrows():
                d, ref = int(row['date']), float(row['close'])
                if d in last_close_by_day:
                    got = float(last_close_by_day[d])
                    ratio = got / ref if ref else float('nan')
                    if 0.9 <= ratio <= 1.1:
                        return dict(verdict='PASS', day=d, daily_close=ref,
                                    minute_close=got, ratio=ratio)
                    if abs(ratio) > 10 or abs(ratio) < 0.1:
                        return dict(verdict='SCALE_MISMATCH', day=d,
                                    daily_close=ref, minute_close=got, ratio=ratio)
                    return dict(verdict='WARN_PRICE_GAP', day=d, daily_close=ref,
                                minute_close=got, ratio=ratio)
        return dict(verdict='SKIPPED', reason='no_common_day')
    except Exception as exc:  # pragma: no cover - defensive
        return dict(verdict='SKIPPED', reason=f'{type(exc).__name__}:{exc}')
