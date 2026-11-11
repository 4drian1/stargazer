"""Fast strong-pool quote lane: bounded scans, first-observed events and health.

Price arrival is recorded before indicator work. Direction is evidence metadata,
never a gate: a first quote already in-band still creates an event. Delivery and
indicator work use independent bounded queues so neither blocks quote scanning.
"""
from datetime import datetime, timedelta
import json
import math
import queue
import threading
import time
import os
from pathlib import Path
import cross_process_lock_v1 as cpl
import instance_state_store_v1 as iss

CONTRACT = 'STRONG_TOUCH_LANE_V1'
MAX_SCAN_INTERVAL_SECONDS = 120
SYMBOLS_PER_PASS = 120
QUOTE_BATCH_SIZE = 30
QUOTE_WORKERS = 4
QUOTE_TIMEOUT_SECONDS = 6
POLL_INTERVAL_SECONDS = 12
MAX_STATE_BYTES = 4 * 1024 * 1024
MAX_EVENTS = 1024
# Cross-process identity for quote-continuity checks.  Unlike the old per-process
# uuid, this is the OS pid, which is what the lease owner token uses so a second
# live instance (or the invoke() subprocess) can never be mistaken for the owner.
OWNER_TOKEN = os.getpid()
STATE_LEASE_TTL = 300.0  # seconds; > any single poll/indicator operation

# Ownership of each state file is decided by instance_state_store_v1 (cross-process
# lease).  A non-owner must NEVER enqueue, broadcast, compute indicators, or send;
# see is_state_owner() and the gates in poll/enqueue_pending/_worker_loop.
_STATE_LOCK = threading.RLock()
_WORKER_LOCK = threading.Lock()
_DELIVERY_QUEUE = queue.Queue(maxsize=1024)
_INDICATOR_QUEUE = queue.Queue(maxsize=1024)
_SUPPLEMENT_QUEUE = queue.Queue(maxsize=MAX_EVENTS)
_SUPPLEMENT_INFLIGHT = set()  # transient queue reservations, rebuilt from event state
_WORKERS = []
_INFLIGHT = {'delivery': 0, 'indicators': 0, 'supplements': 0}


def state_file(state_dir, day):
    return Path(state_dir) / (str(day) + '.json')


def prune_state_files(state_dir, day, keep_days=30):
    """Retain a bounded rolling window of daily diagnostics, not every quote."""
    folder = Path(state_dir)
    if not folder.is_dir():
        return 0
    cutoff = datetime.strptime(str(day), '%Y%m%d').date() - timedelta(days=max(1, int(keep_days)) - 1)
    removed = 0
    for file in folder.glob('????????.json'):
        if not file.name[:8].isdigit():
            continue
        try:
            if datetime.strptime(file.name[:8], '%Y%m%d').date() < cutoff:
                file.unlink()
                removed += 1
        except ValueError:
            continue
    return removed


def _default_state(day, pool_id):
    # `owner` is populated by instance_state_store_v1 on first write; it is None
    # here so the store can claim the lease with this process's pid.
    return dict(contract=CONTRACT, day=int(day), pool_id=pool_id,
                owner=None, symbols={}, events={}, health={})


def is_state_owner(state_file):
    """True iff THIS process owns the state file's cross-process lease.

    Used to gate every side-effecting operation: a non-owner must not enqueue,
    broadcast, compute indicators, or send.  Cheap (reads the .owner meta file).
    """
    return iss.is_owner(Path(state_file), ttl=STATE_LEASE_TTL)


def _read_state_owned(file, day=None, pool_id=None, codes=None):
    """Cross-process safe read AND the only legitimate lease-claim path.

    Returns ``(value, owned)``.  ``owned`` is True only when this process holds
    the reservation (no owner / previous owner confirmed dead / already ours);
    a live competitor, an unconfirmable-dead owner or a corrupt ``.owner`` gives
    ``owned=False`` and a strictly read-only value.

    Callers MUST gate scans, writes, quote requests, enqueues and sends on
    ``owned`` rather than on ``iss.is_owner()``: the latter is a cheap in-memory
    identity check that is necessarily empty on the first call of a fresh
    process, which used to deadlock the whole lane into ``NOT_OWNER`` forever.

    On a legitimate takeover of a dead owner, ``value['_restart_recovery_pending']``
    is set so callers re-enqueue PENDING/WAIT_REVERIFY work without ever re-sending.
    """
    file = Path(file)
    return iss.read_state(
        file, contract=CONTRACT, day=day, pool_id=pool_id, codes=codes,
        build_default=_default_state, ttl=STATE_LEASE_TTL)


def _read_state(file, day=None, pool_id=None, codes=None):
    """Read-only snapshot helper; ownership is deliberately not consulted."""
    value, _owned = _read_state_owned(file, day, pool_id, codes)
    return value


def _write_state(file, value):
    """Cross-process safe, atomic write.  Refuses (no-op) if we are not the owner."""
    file = Path(file)
    if not iss.is_owner(file, ttl=STATE_LEASE_TTL):
        # A live competing process (or an unconfirmable-dead/corrupt owner) owns
        # this file; never clobber it.  Read-only pass.
        return
    iss.write_state(file, value, ttl=STATE_LEASE_TTL)


def _epoch(value):
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


# ---- 动态分级扫描参数 (任务书阶段 1b: 近带高频 / 中距提频 / 远距公平) ----

# ATR 归一化距离阈值
TIER_HIGH_ATR = 0.3       # NEAREST_BAND_ATR < 0.3 → 已在带内或极度接近, 最高优先级
TIER_MEDIUM_ATR = 1.5     # 0.3 <= NEAREST_BAND_ATR < 1.5 → 趋近中, 中优先级
# >= TIER_MEDIUM_ATR 或无有效观察带 → LOW, 公平轮询

# 每轮配额比例 (加起来=1.0, 最后一档吃剩余防止除不尽)
TIER_HIGH_RATIO = 0.60    # 60% 给 HIGH
TIER_MEDIUM_RATIO = 0.25  # 25% 给 MEDIUM
# 剩余 = 15% 给 LOW

# 防饿死: 若某票 last_scan_gap_seconds >= MAX_SCAN_INTERVAL_SECONDS, 强制提升到 HIGH
TIER_STARVE_PROTECT_SECONDS = MAX_SCAN_INTERVAL_SECONDS  # 120s


def _nearest_band_atr(price, row):
    """返回 (nearest_band_atr, band, distance_atr); 若 row 无有效 positions 或 atr_ref 不可用返回 (None, None, None).

    nearest_band_atr = min over positions of:
      if price in [lower, upper]: 0.0
      elif price < lower: (lower - price) / atr_ref
      else: (price - upper) / atr_ref
    """
    try:
        positions = row.get('positions') or []
        atr_ref = float(row.get('atr_ref') or 0)
    except (TypeError, ValueError):
        return None, None, None
    if not positions or atr_ref <= 0:
        return None, None, None
    best = None  # (dist_atr, band)
    for band in positions:
        try:
            lo = float(band.get('lower'))
            hi = float(band.get('upper'))
        except (TypeError, ValueError):
            continue
        if lo <= price <= hi:
            dist = 0.0
        elif price < lo:
            dist = (lo - price) / atr_ref
        else:
            dist = (price - hi) / atr_ref
        if best is None or dist < best[0]:
            best = (dist, band)
    if best is None:
        return None, None, None
    return best[0], best[1], best[0]  # nearest_band_atr, band, distance_atr


def _tier(row, symbol):
    """返回 'HIGH' / 'MEDIUM' / 'LOW' / 'UNKNOWN'. 纯 oldest-first 作为 fallback.

    防饿死: last_scan_gap_seconds >= TIER_STARVE_PROTECT_SECONDS 强制 HIGH.
    """
    try:
        last_gap = float(symbol.get('last_scan_gap_seconds') or 0)
    except (TypeError, ValueError):
        last_gap = 0.0
    if last_gap >= TIER_STARVE_PROTECT_SECONDS:
        return 'HIGH'
    return _tier_raw(row, symbol)


def _tier_raw(row, symbol):
    """原始 tier 判定, 排除饿死提升. 用于观测 tier_selected 原始分布."""
    price = symbol.get('last_price')
    if price is None:
        return 'UNKNOWN'
    try:
        price_f = float(price)
    except (TypeError, ValueError):
        return 'UNKNOWN'
    nearest_atr, _band, _dist = _nearest_band_atr(price_f, row)
    if nearest_atr is None:
        return 'UNKNOWN'
    if nearest_atr < TIER_HIGH_ATR:
        return 'HIGH'
    if nearest_atr < TIER_MEDIUM_ATR:
        return 'MEDIUM'
    return 'LOW'


def _select_by_tier(rows, symbols, limit):
    """分级扫描主函数. 返回 [code, ...] 长度 <= limit."""
    # 按 tier 分组 (UNKNOWN 也当作 LOW)
    buckets = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
    for row in rows:
        code = str(row['code'])
        sym = symbols.get(code) or {}
        tier = _tier(row, sym)
        if tier == 'UNKNOWN':
            tier = 'LOW'  # 无有效 positions → 公平轮询
        buckets[tier].append((row, sym, code))

    def oldest_first(items):
        """每档内 oldest-attempt-first."""
        return sorted(items, key=lambda t: (
            _epoch((t[1] or {}).get('last_attempt_epoch')),
            t[2]))

    high = oldest_first(buckets['HIGH'])
    medium = oldest_first(buckets['MEDIUM'])
    low = oldest_first(buckets['LOW'])

    n = max(1, int(limit))
    # HIGH: 60%, 但 HIGH 数量可能不够; 不够的配额喂给 MEDIUM 再给 LOW
    n_high = min(len(high), int(n * TIER_HIGH_RATIO))
    remaining = n - n_high
    n_medium = min(len(medium), int(remaining * (TIER_MEDIUM_RATIO / (1 - TIER_HIGH_RATIO)))) if remaining > 0 else 0
    remaining2 = remaining - n_medium
    n_low = min(len(low), remaining2)

    selected = (high[:n_high] + medium[:n_medium] + low[:n_low])
    return [t[2] for t in selected]


def select_scan_codes(rows, symbols, limit=SYMBOLS_PER_PASS):
    """分级扫描 (任务书阶段 1b): 近带高频 / 中距提频 / 远距公平轮询.

    防饿死: 若某票 last_scan_gap_seconds >= MAX_SCAN_INTERVAL_SECONDS, 强制 HIGH 档.
    降级回退: 若全部 tier=UNKNOWN (没有任何 row.positions + atr_ref), 回退到原来的 oldest-first.
    """
    # 检查是否所有 row 都有效 (不是全部 UNKNOWN)
    has_valid_positions = False
    for row in rows[:min(32, len(rows))]:  # 抽样前 32 只即可快速判断
        if (row.get('positions') or []) and float(row.get('atr_ref') or 0) > 0:
            has_valid_positions = True
            break
    if not has_valid_positions:
        # 纯 oldest-first fallback
        ordered = sorted(rows, key=lambda row: (
            _epoch((symbols.get(str(row['code'])) or {}).get('last_attempt_epoch')),
            str(row['code'])))
        return [str(row['code']) for row in ordered[:max(1, int(limit))]]
    return _select_by_tier(rows, symbols, limit)


def classify_direction(previous_price, price, band):
    """Classify observed quote-to-quote direction without suppressing an event."""
    lower, upper = float(band['lower']), float(band['upper'])
    if previous_price is None:
        return ('FIRST_QUOTE_IN_BAND_DIRECTION_UNKNOWN'
                if lower <= price <= upper else 'DIRECTION_UNKNOWN')
    if previous_price > upper and price <= upper:
        return 'DOWNWARD_ENTRY_OBSERVED'
    if previous_price < lower and price >= lower:
        return 'REBOUND_FROM_BELOW_OBSERVED'
    if lower <= price <= upper:
        return 'IN_BAND_DIRECTION_NOT_OBSERVED'
    return 'DIRECTION_NOT_CLASSIFIED'


def band_location(price, band):
    if band['lower'] <= price <= band['upper']:
        return 'STILL_IN_ORIGINAL_BAND'
    return 'BELOW_ORIGINAL_BAND' if price < band['lower'] else 'ABOVE_ORIGINAL_BAND'


def quote_quality(quote, day, now):
    if not isinstance(quote, dict):
        return 'QUOTE_MISSING'
    try:
        stamp = datetime.strptime(str(quote['time']), '%Y%m%d%H%M%S')
    except (KeyError, TypeError, ValueError):
        return 'QUOTE_TIME_INVALID'
    if stamp.strftime('%Y%m%d') != str(day):
        return 'QUOTE_WRONG_DAY'
    age = (now - stamp).total_seconds()
    if age < 0:
        return 'QUOTE_FROM_FUTURE'
    if age > 120:
        return 'QUOTE_STALE'
    for key in ('price', 'prev'):
        value = quote.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            return 'QUOTE_' + key.upper() + '_INVALID'
    return None


def _quote_snapshot(quote):
    keys = ('name', 'price', 'prev', 'open', 'high', 'low', 'pct', 'amount_wan', 'time')
    return {key: quote.get(key) for key in keys if key in quote}


def _cutoff(now):
    day = int(now.strftime('%Y%m%d'))
    minute = now.hour * 60 + now.minute
    cutoff = int(now.strftime('%Y%m%d%H%M%S')) if minute < 900 else day * 1000000 + 150000
    if cutoff % 1000000 == 130000:
        cutoff = day * 1000000 + 113000
    return cutoff


def _queue_depths():
    """Queue depths + worker thread counts (任务书阶段1 P1: 线程数监控)."""
    alive_threads = [t for t in _WORKERS if t.is_alive()]
    delivery_workers = sum(1 for t in alive_threads if t.name.startswith('strong-touch-delivery-'))
    indicator_workers = sum(1 for t in alive_threads if t.name.startswith('strong-touch-indicators-'))
    # queue 总积压估算: 即时 qsize + inflight (inflight 代表正在处理的事件数)
    delivery_backlog = _DELIVERY_QUEUE.qsize() + _INFLIGHT['delivery']
    indicator_backlog = _INDICATOR_QUEUE.qsize() + _INFLIGHT['indicators']
    return dict(
        delivery=_DELIVERY_QUEUE.qsize(),
        indicators=_INDICATOR_QUEUE.qsize(),
        supplements=_SUPPLEMENT_QUEUE.qsize(),
        delivery_inflight=_INFLIGHT['delivery'],
        indicators_inflight=_INFLIGHT['indicators'],
        # ---- 线程数监控 (新增) ----
        delivery_worker_alive=delivery_workers,
        indicator_worker_alive=indicator_workers,
        total_worker_alive=len(alive_threads),
        # ---- 积压估算 (新增) ----
        delivery_backlog_total=delivery_backlog,
        indicator_backlog_total=indicator_backlog,
        # ---- 队列容量 (用于监控满风险) ----
        delivery_queue_maxsize=_DELIVERY_QUEUE.maxsize,
        indicator_queue_maxsize=_INDICATOR_QUEUE.maxsize,
        delivery_fill_pct=round(delivery_backlog / _DELIVERY_QUEUE.maxsize * 100, 1)
            if _DELIVERY_QUEUE.maxsize else 0.0,
        indicator_fill_pct=round(indicator_backlog / _INDICATOR_QUEUE.maxsize * 100, 1)
            if _INDICATOR_QUEUE.maxsize else 0.0,
    )


def restore_pending_events(state_dir, plan, *, stop_event=None):
    """显式启动入口: 在 start_workers 之后、首次 poll 之前主动扫描 state 文件,
    将 PENDING / WAIT_REVERIFY 事件重新入队 (任务书阶段1 P1: 启动时主动恢复).

    原来的 enqueue_pending() 依赖首次 poll 被动触发; 本函数独立调用, 恢复更快且
    不与 poll 主循环 race. 返回恢复的 delivery/indicator 事件数.
    """
    # 扫描所有 state 文件 (不仅今日)
    files = sorted(Path(state_dir).glob('*.json'))
    today_stem = datetime.now().strftime('%Y%m%d')
    delivery_restored = 0
    indicator_restored = 0
    supplements_restored = 0
    for file in files:
        try:
            now = datetime.now()
            import queue as _q
            with _STATE_LOCK:
                if file.stem == today_stem:
                    # Today's file: claim/confirm inside the lock instead of the
                    # old in-memory is_owner() gate, which was always empty at
                    # startup and silently skipped every pending event.
                    state, owned = _read_state_owned(file)
                else:
                    # Historical files are read-only unless we already own them.
                    # Never create a fresh .owner lease for an expired day (that
                    # would leave orphan reservations the pruner cannot see).
                    owned = iss.is_owner(file, ttl=STATE_LEASE_TTL)
                    state = _read_state(file) if owned else None
                if not owned:
                    # Live competitor / unconfirmable-dead / corrupt .owner:
                    # read-only, never enqueue or send on its behalf.
                    continue
                dirty = bool(state.pop('_restart_recovery_pending', False))
                for event_id, event in state.get('events', {}).items():
                    event_day=int(event.get('day') or state.get('day') or 0)
                    today=int(now.strftime('%Y%m%d'))
                    if event_day != today:
                        if event_day < today:
                            if event.get('delivery_status') in ('PENDING','WAIT_REVERIFY'):
                                event.update(delivery_status='EXPIRED',delivery_queued=False)
                                dirty=True
                            if event.get('indicator_status')=='PENDING':
                                event.update(indicator_status='EXPIRED',indicator_queued=False)
                                dirty=True
                        continue
                    retry_after = _epoch(event.get('delivery_retry_after_epoch'))
                    if (event.get('delivery_status') in ('PENDING', 'WAIT_REVERIFY') and
                            not event.get('delivery_queued') and retry_after <= now.timestamp()):
                        try:
                            _DELIVERY_QUEUE.put_nowait((str(file), event_id, plan))
                        except _q.Full:
                            pass
                        else:
                            event['delivery_queued'] = True
                            event['send_queued_at'] = now.isoformat(timespec='seconds')
                            delivery_restored += 1
                            dirty = True
                    if event.get('indicator_status') == 'PENDING' and not event.get('indicator_queued'):
                        try:
                            _INDICATOR_QUEUE.put_nowait((str(file), event_id, plan))
                        except _q.Full:
                            pass
                        else:
                            event['indicator_queued'] = True
                            event['indicator_queued_at'] = now.isoformat(timespec='seconds')
                            indicator_restored += 1
                            dirty = True
                    supplements_restored += _enqueue_supplement(file, event_id, event, plan, now)
                if dirty:
                    _write_state(file, state)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return dict(delivery_restored=delivery_restored, indicator_restored=indicator_restored,
                supplements_restored=supplements_restored, files_scanned=len(files))


def _usage_snapshot(usage_file, day):
    if not usage_file:
        return None
    try:
        from quote_budget_v1 import snapshot
        return snapshot(usage_file, day=str(day))
    except Exception as exc:
        return dict(status='UNAVAILABLE', reason=type(exc).__name__)


def poll(plan, now, *, state_dir, quote_fn=None, usage_file=None, clock=None):
    """Scan the oldest 120 strong-pool symbols, record quote health and events."""
    from sector_watch_v1 import names
    from strong_sector_watch_v1 import frozen, candidate_events, digest

    day = int(now.strftime('%Y%m%d'))
    board_names = names(plan or {})
    if not board_names:
        return dict(status='WAIT_BOARDS', pool_count=0)
    pool = frozen(board_names, day)
    if not pool:
        return dict(status='WAIT_POOL', pool_count=0)
    rows = pool.get('rows') or []
    row_by_code = {str(row['code']): row for row in rows}
    codes = list(row_by_code)
    file = state_file(state_dir, day)
    with _STATE_LOCK:
        # Claim (or confirm) the lease FIRST, under the same cross-process lock
        # used to read .owner.  Gating on iss.is_owner() here used to be a
        # self-deadlock: on the first poll of a fresh process the in-memory
        # identity is empty, so the only path that can populate it was never
        # reached and the lane reported NOT_OWNER forever without quoting.
        state, owned = _read_state_owned(file, day, pool.get('rows_sha256'), codes)
        if not owned:
            # A live competitor, an unconfirmable-dead owner or a corrupt
            # .owner holds this file — possibly with no state JSON written yet.
            # Zero scan, zero quote, zero write, zero enqueue.
            return dict(status='NOT_OWNER', pool_count=len(rows))
        prune_state_files(state_dir, day)
        selected = select_scan_codes(rows, state['symbols'])
        attempt_epoch = now.timestamp()
        attempt_stamp = now.isoformat(timespec='seconds')
        for code in selected:
            symbol = state['symbols'].setdefault(code, dict(attempts=0))
            symbol['last_attempt_epoch'] = attempt_epoch
            symbol['last_attempt_at'] = attempt_stamp
            symbol['attempts'] = int(symbol.get('attempts', 0)) + 1
            symbol['status'] = 'FETCHING'
            symbol['last_error'] = None
        state['health'] = dict(state.get('health') or {}, status='SCANNING',
            pool_id=pool.get('rows_sha256'), pool_count=len(rows),
            last_scan_started_at=attempt_stamp, selected_count=len(selected),
            max_scan_interval_seconds=MAX_SCAN_INTERVAL_SECONDS)
        _write_state(file, state)

    request_error = None
    try:
        if quote_fn is None:
            from tools.market import quotes
            def on_attempt(_url):
                if usage_file:
                    try:
                        from quote_budget_v1 import record_attempt
                        record_attempt(usage_file, 'strong_pool')
                    except Exception:
                        pass
                return True
            live = quotes(selected, on_attempt=on_attempt, timeout=QUOTE_TIMEOUT_SECONDS,
                          batch_size=QUOTE_BATCH_SIZE, max_workers=QUOTE_WORKERS, tries=1)
        else:
            live = quote_fn(selected) or {}
    except Exception as exc:
        live = {}
        request_error = type(exc).__name__ + ':' + str(exc)[:160]

    completed_at = (clock or datetime.now)()
    if completed_at.strftime('%Y%m%d') != str(day):
        return dict(status='DAY_CHANGED_DURING_SCAN', pool_count=len(rows), scanned_count=len(selected))
    cutoff = _cutoff(completed_at)
    with _STATE_LOCK:
        state, owned = _read_state_owned(file, day, pool.get('rows_sha256'), codes)
        if not owned:
            # Ownership changed while quotes were in flight: drop the results
            # instead of writing or enqueueing into another process's state.
            return dict(status='NOT_OWNER', pool_count=len(rows), scanned_count=len(selected))
        valid_count = 0
        for code in selected:
            symbol = state['symbols'].setdefault(code, dict(attempts=0))
            quote = (live or {}).get(code) or {}
            reason = quote_quality(quote, day, completed_at)
            if reason:
                symbol.update(status=reason, last_error=reason)
                continue
            valid_count += 1
            previous_observed = _epoch(symbol.get('last_success_observed_epoch'))
            previous_price = (symbol.get('last_price')
                if symbol.get('last_success_process_id') == OWNER_TOKEN and
                previous_observed and completed_at.timestamp() - previous_observed <= MAX_SCAN_INTERVAL_SECONDS
                else None)
            quote_stamp = datetime.strptime(str(quote['time']), '%Y%m%d%H%M%S')
            quote_epoch = quote_stamp.timestamp()
            observed_epoch = completed_at.timestamp()
            symbol.update(status='OK', last_error=None, last_success_epoch=quote_epoch,
                last_success_at=quote_stamp.isoformat(timespec='seconds'),
                last_success_observed_epoch=observed_epoch,
                last_success_observed_at=completed_at.isoformat(timespec='seconds'),
                last_success_process_id=OWNER_TOKEN,
                last_quote_at=str(quote.get('time')), last_price=float(quote['price']),
                last_quote=_quote_snapshot(quote),
                last_scan_gap_seconds=(round(observed_epoch - previous_observed, 3)
                                       if previous_observed else None))
            try:
                candidates = candidate_events(row_by_code[code], quote, day, cutoff)
            except Exception as exc:
                symbol.update(status='DETECTION_ERROR', last_error=type(exc).__name__)
                continue
            if not candidates:
                continue
            selected_event = candidates[0]
            band = selected_event['band']
            event_id = digest([day, code, row_by_code[code]['pool'], band['lower'], band['upper']])
            already_for_code = any(e.get('code') == code for e in state['events'].values())
            if event_id not in state['events'] and not already_for_code:
                direction = classify_direction(previous_price, float(quote['price']), band)
                # 事件级证据链时间戳 (任务书阶段1 P0)
                quote_attempt_at = attempt_stamp  # L207 symbol 级 attempt_stamp, 提升到事件级
                quote_received_at = completed_at.isoformat(timespec='seconds')
                state['events'][event_id] = dict(
                    event_id=event_id, day=day, code=code,
                    name=row_by_code[code].get('name'), pool=row_by_code[code].get('pool'),
                    pool_id=pool.get('rows_sha256'), boards=row_by_code[code].get('boards', []),
                    band=dict(band), kind=selected_event.get('kind'),
                    arrival_basis=selected_event.get('arrival_basis'),
                    direction_class=direction, direction_previous_price=previous_price,
                    # ---- 证据链时间戳 (补全 P0 级缺口) ----
                    scan_started_at=attempt_stamp,
                    quote_attempt_at=quote_attempt_at,
                    quote_received_at=quote_received_at,
                    first_observed_touch_at=completed_at.isoformat(timespec='seconds'),
                    first_observed_quote_at=str(quote.get('time')),
                    first_observed_at=completed_at.isoformat(timespec='seconds'),
                    event_recorded_at=completed_at.isoformat(timespec='seconds'),
                    # reverify_quote_at 由 delivery worker 在 strong_sector_watch_v1.deliver_touch_event 里补
                    # ---- 原始数据 ----
                    initial_quote=_quote_snapshot(quote), indicator_cutoff=cutoff,
                    delivery_status='PENDING', indicator_status='PENDING',
                    delivery_queued=False, indicator_queued=False)
        if len(state['events']) > MAX_EVENTS:
            raise ValueError('STRONG_TOUCH_EVENT_CAP_REACHED')
        now_epoch = completed_at.timestamp()
        attempt_ages = [max(0.0, now_epoch - _epoch((state['symbols'].get(c) or {}).get('last_attempt_epoch')))
                        for c in codes if _epoch((state['symbols'].get(c) or {}).get('last_attempt_epoch'))]
        quote_ages = [max(0.0, now_epoch - _epoch((state['symbols'].get(c) or {}).get('last_success_epoch')))
                      for c in codes if _epoch((state['symbols'].get(c) or {}).get('last_success_epoch'))]
        observation_ages = [max(0.0, now_epoch - _epoch((state['symbols'].get(c) or {}).get('last_success_observed_epoch')))
                            for c in codes if _epoch((state['symbols'].get(c) or {}).get('last_success_observed_epoch'))]
        never_scanned = sum(not _epoch((state['symbols'].get(c) or {}).get('last_attempt_epoch')) for c in codes)
        never_quoted = sum(not _epoch((state['symbols'].get(c) or {}).get('last_success_epoch')) for c in codes)
        overdue_attempts = sum((not _epoch((state['symbols'].get(c) or {}).get('last_attempt_epoch')) or
                                now_epoch - _epoch((state['symbols'].get(c) or {}).get('last_attempt_epoch')) >
                                MAX_SCAN_INTERVAL_SECONDS) for c in codes)
        overdue_quotes = sum((not _epoch((state['symbols'].get(c) or {}).get('last_success_epoch')) or
                              now_epoch - _epoch((state['symbols'].get(c) or {}).get('last_success_epoch')) >
                              MAX_SCAN_INTERVAL_SECONDS) for c in codes)
        health_status = ('QUOTE_ERROR' if request_error or not valid_count else
                         'MONITORING' if valid_count == len(selected) else 'DEGRADED')
        # ---- 分级扫描统计 (任务书阶段 1b: tier 分布 + 饿死提升数) ----
        tier_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNKNOWN': 0}
        tier_selected = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        starve_promotions = 0
        for row in rows:
            code = str(row['code'])
            sym = state['symbols'].get(code) or {}
            gap = float(sym.get('last_scan_gap_seconds') or 0)
            if gap >= TIER_STARVE_PROTECT_SECONDS:
                starve_promotions += 1
            t = _tier(row, sym)
            tier_counts[t] = tier_counts.get(t, 0) + 1
        selected_set = set(selected)
        for t, count in tier_counts.items():
            if t == 'UNKNOWN':
                continue
            for row in rows:
                if str(row['code']) in selected_set:
                    sym = state['symbols'].get(str(row['code'])) or {}
                    # 重新算 tier (排除饿死提升, 看原始 tier 是什么)
                    t_raw = _tier_raw(row, sym)
                    if t_raw in ('HIGH', 'MEDIUM', 'LOW'):
                        tier_selected[t_raw] += 1
                        selected_set.remove(str(row['code']))
                        break

        health = dict(status=health_status,
            pool_id=pool.get('rows_sha256'), pool_count=len(rows), scanned_count=len(selected),
            valid_quotes=valid_count, missing_or_invalid_quotes=len(selected) - valid_count,
            never_scanned_count=never_scanned, never_quoted_count=never_quoted,
            overdue_attempt_count=overdue_attempts, overdue_quote_count=overdue_quotes,
            max_attempt_age_seconds=round(max(attempt_ages), 3) if attempt_ages else None,
            max_quote_age_seconds=round(max(quote_ages), 3) if quote_ages else None,
            max_observation_age_seconds=round(max(observation_ages), 3) if observation_ages else None,
            last_scan_started_at=attempt_stamp,
            last_scan_finished_at=completed_at.isoformat(timespec='seconds'),
            scan_duration_seconds=round((completed_at - now).total_seconds(), 3),
            max_scan_interval_seconds=MAX_SCAN_INTERVAL_SECONDS,
            request_error=request_error, event_count=len(state['events']),
            queue_depths=_queue_depths(), quote_usage=_usage_snapshot(usage_file, day),
            # ---- 分级扫描统计 (新增) ----
            tier_counts=tier_counts, tier_selected=tier_selected,
            starve_promotions_this_round=starve_promotions,
            tier_thresholds={'high_atr': TIER_HIGH_ATR, 'medium_atr': TIER_MEDIUM_ATR,
                             'high_ratio': TIER_HIGH_RATIO, 'medium_ratio': TIER_MEDIUM_RATIO,
                             'starve_protect_seconds': TIER_STARVE_PROTECT_SECONDS})
        state['health'] = health
        _write_state(file, state)
    enqueue_pending(file, plan, now=completed_at)
    health['queue_depths'] = _queue_depths()
    return health


def _mutate_event(file, event_id, fields):
    with _STATE_LOCK:
        state = _read_state(file)
        event = state.get('events', {}).get(event_id)
        if event is None:
            return None
        event.update(fields)
        _write_state(file, state)
        return dict(event)


def get_event(file, event_id):
    with _STATE_LOCK:
        state = _read_state(file)
        value = state.get('events', {}).get(event_id)
        return dict(value) if value else None


def update_event(file, event_id, **fields):
    return _mutate_event(file, event_id, fields)


def _enqueue_supplement(file, event_id, event, plan, now):
    """Join persisted indicator results and the first-send outcome under _STATE_LOCK."""
    if event.get('indicator_supplement_status') not in ('PENDING', 'WAIT_RETRY', 'SENDING'):
        return False
    if (event.get('indicator_status') in (None, 'PENDING', 'PROCESSING')
            or event.get('delivery_status') in (None, 'PENDING', 'WAIT_REVERIFY', 'PROCESSING', 'SENDING')
            or str(event.get('day')) != now.strftime('%Y%m%d')
            or _epoch(event.get('supplement_retry_after_epoch')) > now.timestamp()):
        return False
    from notifier import market_open
    if not market_open(now):
        return False
    key = (str(file), event_id)
    if key in _SUPPLEMENT_INFLIGHT:
        return False
    _SUPPLEMENT_INFLIGHT.add(key)
    try:
        _SUPPLEMENT_QUEUE.put_nowait((*key, plan))
    except queue.Full:
        _SUPPLEMENT_INFLIGHT.discard(key)
        return False
    return True


def enqueue_pending(file, plan, *, now=None):
    now = now or datetime.now()
    if not iss.is_owner(file, ttl=STATE_LEASE_TTL):
        # Non-owner must not enqueue delivery/indicator work (zero enqueue).
        return dict(delivery=0, indicators=0, supplements=0, queue_depths=_queue_depths())
    queued_delivery = queued_indicators = 0
    queued_supplements = 0
    with _STATE_LOCK:
        state = _read_state(file)
        dirty = False
        for event_id, event in state.get('events', {}).items():
            retry_after = _epoch(event.get('delivery_retry_after_epoch'))
            if (event.get('delivery_status') in ('PENDING', 'WAIT_REVERIFY') and
                    not event.get('delivery_queued') and retry_after <= now.timestamp()):
                try:
                    _DELIVERY_QUEUE.put_nowait((str(file), event_id, plan))
                except queue.Full:
                    pass
                else:
                    event['delivery_queued'] = True
                    event['send_queued_at'] = now.isoformat(timespec='seconds')
                    queued_delivery += 1
                    dirty = True
            if event.get('indicator_status') == 'PENDING' and not event.get('indicator_queued'):
                try:
                    _INDICATOR_QUEUE.put_nowait((str(file), event_id, plan))
                except queue.Full:
                    pass
                else:
                    event['indicator_queued'] = True
                    event['indicator_queued_at'] = now.isoformat(timespec='seconds')
                    queued_indicators += 1
                    dirty = True
            queued_supplements += _enqueue_supplement(file, event_id, event, plan, now)
        if dirty:
            state.setdefault('health', {})['queue_depths'] = _queue_depths()
            _write_state(file, state)
    return dict(delivery=queued_delivery, indicators=queued_indicators, supplements=queued_supplements,
                queue_depths=_queue_depths())


def _worker_loop(work_queue, handler, stop_event, lane):
    while not stop_event.is_set():
        try:
            file, event_id, plan = work_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if not iss.is_owner(file, ttl=STATE_LEASE_TTL):
            # Non-owner: drop the item without enqueue/broadcast/indicate/send.
            if lane == 'supplements':
                with _STATE_LOCK:
                    _SUPPLEMENT_INFLIGHT.discard((file, event_id))
            work_queue.task_done()
            continue
        _INFLIGHT[lane] += 1
        try:
            stamp = datetime.now().isoformat(timespec='seconds')
            if lane == 'delivery':
                update_event(file, event_id, delivery_queued=False,
                             delivery_status='PROCESSING', delivery_phase='PRE_SEND_VERIFY')
            elif lane == 'indicators':
                update_event(file, event_id, indicator_queued=False,
                             indicator_status='PROCESSING', indicator_started_at=stamp)
            handler(event_id, Path(file), plan)
        except Exception as exc:
            fields = dict(error=type(exc).__name__ + ':' + str(exc)[:160])
            if lane == 'delivery':
                current = get_event(file, event_id) or {}
                attempted = current.get('delivery_phase') == 'TRANSPORT'
                fields.update(delivery_status='UNKNOWN' if attempted else 'WAIT_REVERIFY',
                              delivery_retry_after_epoch=time.time()+POLL_INTERVAL_SECONDS,
                              delivery_finished_at=datetime.now().isoformat(timespec='seconds'))
            elif lane == 'indicators':
                fields.update(indicator_status='FAILED', indicator_finished_at=datetime.now().isoformat(timespec='seconds'),
                              indicator_supplement_status='PENDING',
                              indicator_result=dict(status='FAILED', reason=fields['error']))
            else:
                fields = dict(supplement_error=fields['error'],
                              supplement_retry_after_epoch=time.time()+POLL_INTERVAL_SECONDS)
            try:
                update_event(file, event_id, **fields)
            except Exception:
                pass
        finally:
            if lane == 'supplements':
                with _STATE_LOCK:
                    _SUPPLEMENT_INFLIGHT.discard((file, event_id))
            _INFLIGHT[lane] = max(0, _INFLIGHT[lane] - 1)
            work_queue.task_done()


def start_workers(delivery_handler, indicator_handler, stop_event, *, supplement_handler=None):
    """Start bounded delivery, indicator and supplement workers once."""
    with _WORKER_LOCK:
        alive = [thread for thread in _WORKERS if thread.is_alive()]
        _WORKERS[:] = alive
        for lane, work_queue, handler, limit in (
                ('delivery', _DELIVERY_QUEUE, delivery_handler, 2),
                ('indicators', _INDICATOR_QUEUE, indicator_handler, 2),
                ('supplements', _SUPPLEMENT_QUEUE, supplement_handler, 1)):
            if handler is None:
                continue
            count = sum(thread.name.startswith('strong-touch-' + lane + '-') for thread in alive)
            for index in range(count, limit):
                thread = threading.Thread(target=_worker_loop,
                    args=(work_queue, handler, stop_event, lane), daemon=True,
                    name='strong-touch-' + lane + '-' + str(index + 1))
                _WORKERS.append(thread)
                thread.start()


def latest_quotes(state_dir, now, *, max_age_seconds=120):
    """Latest valid per-symbol quote cache for slow board summaries/holdings."""
    file = state_file(state_dir, int(now.strftime('%Y%m%d')))
    try:
        with _STATE_LOCK:
            state = _read_state(file)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    out = {}
    for code, symbol in state.get('symbols', {}).items():
        quote = symbol.get('last_quote') or {}
        if quote_quality(quote, int(now.strftime('%Y%m%d')), now) is None:
            try:
                age = (now - datetime.strptime(str(quote['time']), '%Y%m%d%H%M%S')).total_seconds()
            except (KeyError, ValueError, TypeError):
                continue
            if age <= max_age_seconds:
                out[code] = quote
    return out


def health_snapshot(state_dir, now):
    file = state_file(state_dir, int(now.strftime('%Y%m%d')))
    try:
        with _STATE_LOCK:
            state = _read_state(file)
        return dict(state.get('health') or {}, symbols=state.get('symbols') or {},
                    events=state.get('events') or {}, queue_depths=_queue_depths())
    except (OSError, ValueError, json.JSONDecodeError):
        return dict(status='UNAVAILABLE', queue_depths=_queue_depths())
