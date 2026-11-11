"""Touch-price alert lane (#盘中监控方案): quote channel notifies an observed
downward crossing or first valid quote already in a plan band, without waiting for bars, the
three-vote combination, Pyxis or DeepSeek.

Gates run BEFORE detection: monitor enabled, confirmed plan, observation day,
and quote quality (same trading day, fresh, finite, positive). Bands come from
the REAL plan structures (zone / entry_rule.observation_bands / f618) through
one normalization shared by detection and dedupe. Each band has ONE event
identity; the golden (confirmed-signal) channel marks the same identity so the
two channels never double-notify. Events persist before any send; delivery is
recorded per event. A touch is an observed crossing, never a claim of band
fills.

证据链时间戳 (任务书阶段1):
  scan_started_at / quote_attempt_at / quote_received_at /
  first_observed_touch_at / event_recorded_at / send_queued_at /
  send_started_at / reverify_quote_at / send_finished_at /
  indicator_started_at / indicator_finished_at

方向分类 (任务书阶段1):
  DOWNWARD_ENTRY_OBSERVED / FIRST_QUOTE_IN_BAND_DIRECTION_UNKNOWN /
  REBOUND_FROM_BELOW_OBSERVED / DIRECTION_UNKNOWN
  方向是事件属性, 不是发送门槛.

变更说明 (2026-11-05 P0):
  - 方向分类由 classify_direction 统一生成 (与 strong_touch_lane_v1 共享实现)
  - 事件结构扩充完整证据链时间戳
  - 向后兼容: 旧 state 文件仍能读取, 新增字段按缺失处理

变更说明 (2026-11-06 触价策略 V2 —— 分档去重 + 真实向下穿越):
  触发方式从「价格进入区间/低于下沿即推」改为「相邻两次有效报价的真实向下穿越」：
  A. 真实向下穿越: 状态新增 last_observed_quotes (按交易日保存每票最近一次有效
     观察价 price/quote_time/observed_at, 跨轮询持久化, 重启可用, 按日清理)。
     仅当 prev_price > band.upper 且 cur_price <= band.upper 才建
     DOWNWARD_CROSS_UPPER 事件; quote_time 重复或倒退不判定、不更新基准;
     第一笔报价(含带内/带下)只建基准不推送; 从下方反弹进入带内不推送。
     方向证据链来自 last_observed_quotes, 禁止使用 quote.prev (那是昨日收盘价)。
  B. 同股同日分档去重 (维度=交易日+代码, 跨 plan_id, 次日自然重置): 已通知档位
     (delivery ∈ {SENT, UNKNOWN} 或 consumed_by=golden_channel) 存在时, 仅当
     new_band.upper < min(已通知档位 lower) 才允许下一档; FAILED 不算已送达但
     保留证据; queued / 可重试 CANCELLED 事件阻止同股其他档位; 重叠或仅略低
     的带禁止再推 (记 SUPPRESSED_OVERLAPPING_BAND / SUPPRESSED_SAME_CODE)。
  C. 同轮多档: 一轮扫描同时满足多个带只建 band.lower 最小的最深一档, 其余记
     SUPPRESSED_SAME_SCAN, 不得同轮进入发送队列。
  D. 带身份: 保留 zone/obs0/obs1/obs2/f618 原始 id 与来源, 不引入模糊合并阈值,
     不改黄金节点 (mark_notified/band_id_for_price 语义不变)。
  E. crossed 语义 = 「已观察到从上沿上方向下穿越上沿」; reverify 检查报价有效/
     时间/策略版本/是否已送达(consumed)/现价三态; message 按现价区分三态:
     下穿上沿现价仍在带内 / 已低于下沿 / 触发后已反弹离带; 快照报价永不描述成
     连续逐笔成交。
  状态兼容: 旧 notified 历史全部保留, 不重解释、不自动重发 —— legacy 条目
  (无 trigger_policy) 在 pending_sends 中被隔离并打 legacy_reason 标记; 新事件
  带 trigger_policy='V2'; suppression_events 有界 (每票每日 + 总量, 超出丢弃
  最旧); 三类 SUPPRESSED_* 计入 health_snapshot。

2026-11-07 用户裁定：首笔有效报价已在带内也立即提醒，方向未知；
使用原档位去重，发送前须仍在带内，离带等待重入，不追旧价。
"""
import json
import threading
import time
from pathlib import Path

_STATE_LOCK = threading.RLock()

CONTRACT = 'TOUCH_ALERT_LANE_V1'
QUOTE_MAX_AGE_SECONDS = 120

# ---- 触价策略 V2 常量 ----
TRIGGER_POLICY_V2 = 'V2'
TRIGGER_DOWNWARD_CROSS_UPPER = 'DOWNWARD_CROSS_UPPER'
TRIGGER_FIRST_IN_BAND = 'FIRST_QUOTE_IN_BAND'
# V2.1: 首次有效观察即破位 (开盘缺口/监控启动前已穿越) → 一次性告知事件
TRIGGER_OPEN_GAP_BELOW = 'OPEN_GAP_BELOW_BAND'
LEGACY_REASON = 'PRE_V2_ORIGINAL_SEMANTICS_ISOLATED'
SUPPRESSED_SAME_SCAN = 'SUPPRESSED_SAME_SCAN'
SUPPRESSED_SAME_CODE = 'SUPPRESSED_SAME_CODE'
SUPPRESSED_OVERLAPPING_BAND = 'SUPPRESSED_OVERLAPPING_BAND'
# suppression_events 有界: 每票每日上限 + 全量上限, 超出丢弃最旧
SUPPRESSION_PER_CODE_DAILY_LIMIT = 50
SUPPRESSION_TOTAL_LIMIT = 500

# ---- 方向分类 (与 strong_touch_lane_v1.classify_direction 实现一致, 共享同语义) ----

DIRECTION_DOWNWARD_ENTRY = 'DOWNWARD_ENTRY_OBSERVED'
DIRECTION_FIRST_QUOTE_UNKNOWN = 'FIRST_QUOTE_IN_BAND_DIRECTION_UNKNOWN'
DIRECTION_REBOUND_FROM_BELOW = 'REBOUND_FROM_BELOW_OBSERVED'
DIRECTION_UNKNOWN = 'DIRECTION_UNKNOWN'
# V2.1: 破位事件的方向证据 —— 无前置观察价, 穿越方向不可知, 仅陈述「已破位」事实
DIRECTION_GAP_BELOW_BAND = 'GAP_BELOW_BAND_OBSERVED'


def classify_direction(previous_price, price, band):
    """Return one of the four task-book direction constants. Direction is
    EVIDENCE metadata, never a send gate. A first quote already in-band still
    creates an event with DIRECTION_FIRST_QUOTE_UNKNOWN.

    previous_price: the prior known price (quote.prev from market quote), or
                    None when no prior observation is available (开盘/重启/断网恢复).
    price:          current observed price.
    band:           dict with keys 'lower', 'upper'.

    注意 (V2): 触价通道自身的下穿判定使用 last_observed_quotes 的相邻两次观察价,
    不使用 quote.prev (昨日收盘价); 本函数保持原语义供方向元数据与共享调用方使用。
    """
    try:
        lower, upper = float(band['lower']), float(band['upper'])
    except (KeyError, TypeError, ValueError):
        return DIRECTION_UNKNOWN
    if previous_price is None:
        return (DIRECTION_FIRST_QUOTE_UNKNOWN
                if lower <= price <= upper else DIRECTION_UNKNOWN)
    try:
        prev = float(previous_price)
    except (TypeError, ValueError):
        return DIRECTION_UNKNOWN
    if prev > upper and price <= upper:
        return DIRECTION_DOWNWARD_ENTRY
    if prev < lower and price >= lower:
        return DIRECTION_REBOUND_FROM_BELOW
    if lower <= price <= upper:
        return DIRECTION_FIRST_QUOTE_UNKNOWN  # 新进入带内, 方向无证据
    return DIRECTION_UNKNOWN


def _now_stamp():
    return time.strftime('%Y%m%d%H%M%S')


def _epoch_now():
    return int(time.time())


def bands_for(row):
    """Unified watch bands from the real plan shape: zone, observation_bands,
    then f618 as a single-level fallback. Identical numeric ranges merge into
    ONE identity (zone+observation_bands often repeat the same band); the
    result is shared by detection, dedupe and the golden channel.

    V2 规则 D: 保留 zone/obsN/f618 原始 id 与来源, 不做任何模糊合并。"""
    raw = []
    zone = row.get('zone') or {}
    if zone.get('zone_lower') and zone.get('zone_upper'):
        raw.append(('zone', float(zone['zone_lower']), float(zone['zone_upper'])))
    for i, band in enumerate((row.get('entry_rule') or {}).get('observation_bands') or []):
        if band.get('lower') and band.get('upper'):
            raw.append((f'obs{i}', float(band['lower']), float(band['upper'])))
    merged = {}
    for bid, lo, hi in raw:
        key = (round(lo, 4), round(hi, 4))
        if key in merged:
            merged[key]['id'] += '+' + bid
        else:
            merged[key] = dict(id=bid, lower=lo, upper=hi)
    out = list(merged.values())
    if not out:
        f618 = (row.get('entry_rule') or {}).get('f618')
        if isinstance(f618, (int, float)) and f618 > 0:
            out.append(dict(id='f618', lower=float(f618), upper=float(f618)))
    return out


def _load(state_file):
    try:
        state = json.loads(Path(state_file).read_bytes())
    except (OSError, ValueError):
        state = {}
    if state.get('contract') != CONTRACT:
        state = dict(contract=CONTRACT, notified={})
    return state


def _save(state_file, state):
    state_file = Path(state_file)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temp = state_file.with_suffix('.tmp')
    temp.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
    temp.replace(state_file)


def _quote_ok(quote, day, now_stamp):
    price = quote.get('price')
    import math
    if isinstance(price, bool) or not isinstance(price, (int, float)):
        return False
    if not math.isfinite(price) or price <= 0:
        return False
    stamp = str(quote.get('time') or '')
    if len(stamp) != 14 or not stamp.isdigit() or stamp[:8] != str(day):
        return False
    try:
        now_t = time.mktime(time.strptime(str(now_stamp), '%Y%m%d%H%M%S'))
        q_t = time.mktime(time.strptime(stamp, '%Y%m%d%H%M%S'))
    except (ValueError, OverflowError, OSError):
        return False
    # 报价不能来自未来；时效只允许 0..120s 的单向窗口
    return 0 <= now_t - q_t <= QUOTE_MAX_AGE_SECONDS


def band_id_for_price(plan, code, price):
    """The band identity actually CONTAINING this price. A price below every
    band does not claim ownership of an upper band (that swallowed lower
    bands); when nothing contains it, ownership is unknown → None and the
    caller consumes nothing."""
    for row in plan.get('rows') or []:
        if str(row.get('code') or '').zfill(6) != str(code).zfill(6):
            continue
        for band in bands_for(row):
            if band['lower'] <= price <= band['upper']:
                return band['id']
    return None


def detect(plan, quotes, state_file, *, now_stamp=None,
           scan_started_at=None, quote_attempt_at=None, quote_received_at=None):
    """Gated detection (V2): enabled/confirmed/day/quote-quality checks happen
    in the caller; here each stock is isolated so one bad row never blocks
    others. Returns NEW events only — one event per stock per scan at most
    (the deepest tier reached), dedupe persisted atomically.

    V2 触发条件 (规则 A): 相邻两次有效观察价满足
      prev_price > band.upper 且 cur_price <= band.upper
    才建 DOWNWARD_CROSS_UPPER 事件; 基准价来自状态内 last_observed_quotes
    (跨轮询持久化), 与 quote.prev (昨日收盘) 无关。

    V2.1 (规则 G): 当日首次有效观察价已在某档下沿下方 (开盘缺口 / 监控启动
    前已破位) 时, 建一条 OPEN_GAP_BELOW_BAND 一次性告知事件 (取最深破位档,
    占用该档当日身份, 受规则 B/C 同样约束); 首笔带内按方向未知提醒。

    证据链时间戳参数:
      scan_started_at:   调用方记录的本轮扫描开始时间 (可 None)
      quote_attempt_at:  调用方记录的本轮报价请求发起时间 (可 None)
      quote_received_at: 调用方记录的本轮报价完成时间 (可 None)
    三者可由调用方 (pentacles_monitor 等) 注入, 也由本函数自动补默认值.
    """
    day = str(plan.get('monitor_date') or '')
    now_stamp = now_stamp or int(_now_stamp())
    scan_started_at = scan_started_at or _now_stamp()
    quote_attempt_at = quote_attempt_at or scan_started_at
    quote_received_at = quote_received_at or scan_started_at
    with _STATE_LOCK:
        return _detect_locked(plan, quotes, state_file, day, now_stamp,
                              scan_started_at, quote_attempt_at, quote_received_at)


def _detect_locked(plan, quotes, state_file, day, now_stamp,
                   scan_started_at, quote_attempt_at, quote_received_at):
    state = _load(state_file)
    plan_id = plan.get('assistant_plan_id') or 'snapshot'
    scan_finished_at = _now_stamp()
    events = []
    suppressions = []
    dirty = False
    # ---- 旧事件隔离 (状态兼容): legacy 条目只补标记, 不重解释、不清空、不重发 ----
    for entry in state.get('notified', {}).values():
        if 'trigger_policy' not in entry and 'legacy_reason' not in entry:
            entry['legacy_reason'] = LEGACY_REASON
            dirty = True
    # 按代码聚合 (同码多行合并候选带, 保证一轮同股只可能出一条事件)
    per_code = {}
    for row in plan.get('rows') or []:
        try:
            code = str(row.get('code') or '').zfill(6)
            quote = quotes.get(code) or {}
            if not _quote_ok(quote, day, now_stamp):
                continue
            slot = per_code.setdefault(code, dict(row=row, quote=quote, bands=[]))
            slot['bands'].extend(bands_for(row))
        except (TypeError, ValueError, KeyError, AttributeError):
            continue  # per-stock isolation
    # ---- 规则 A 基准: last_observed_quotes (按交易日保存, 跨轮询持久化) ----
    day_quotes = state.setdefault('last_observed_quotes', {}).setdefault(day, {})
    for code, slot in per_code.items():
        try:
            row, quote, bands = slot['row'], slot['quote'], slot['bands']
            price = float(quote['price'])
            quote_time = str(quote.get('time') or '')
            baseline = day_quotes.get(code)
            if baseline is not None and str(baseline.get('quote_time') or '') >= quote_time:
                # 报价时间重复或倒退: 不得产生虚假下穿, 也不更新基准
                continue
            baseline_price = None
            prev_quote_time = ''
            if baseline is not None:
                prev_quote_time = str(baseline.get('quote_time') or '')
                try:
                    baseline_price = float(baseline.get('price'))
                except (TypeError, ValueError):
                    baseline_price = None
            # ---- 规则 G (V2.1): 首次有效观察即破位 → 一次性 OPEN_GAP_BELOW_BAND ----
            # 适用: 开盘缺口直接穿越观察带 / 监控启动前价格已在带下方。
            # 无前置观察价, 不构成「盘中穿越」, 但用户要求至少一次性知情;
            # 每股每日至多一条, 取已破位档位中最深一档, 事件同样占用该档当日身份。
            first_in_band = baseline_price is None and any(b['lower'] <= price <= b['upper'] for b in bands)
            if baseline_price is None and not first_in_band:
                breached = [b for b in bands if price < b['lower']]
                if breached:
                    notified_ranges, blocking_keys = _tier_state_locked(state, code, day)
                    eligible_gap = []
                    for band in breached:
                        if blocking_keys:
                            suppressions.append(dict(
                                day=day, code=code, plan_id=plan_id, band_id=band['id'],
                                lower=band['lower'], upper=band['upper'],
                                reason=SUPPRESSED_SAME_CODE,
                                detail='ALIVE_EVENT_BLOCKS_NEW_TIER',
                                blocking_event_key=blocking_keys[0],
                                at=scan_finished_at))
                            continue
                        if notified_ranges:
                            min_lower = min(lo for lo, _hi in notified_ranges)
                            if band['upper'] >= min_lower:
                                overlapped = any(band['lower'] <= hi and lo <= band['upper']
                                                 for lo, hi in notified_ranges)
                                suppressions.append(dict(
                                    day=day, code=code, plan_id=plan_id, band_id=band['id'],
                                    lower=band['lower'], upper=band['upper'],
                                    reason=(SUPPRESSED_OVERLAPPING_BAND if overlapped
                                            else SUPPRESSED_SAME_CODE),
                                    detail='NOT_DEEPER_THAN_NOTIFIED_TIERS',
                                    notified_min_lower=min_lower,
                                    at=scan_finished_at))
                                continue
                        eligible_gap.append(band)
                    # 身份已存在的档 (FAILED 证据 / golden 已消费) 静默跳过
                    eligible_gap = [b for b in eligible_gap
                                    if f'{day}:{plan_id}:{code}:{b["id"]}' not in state['notified']]
                    if eligible_gap:
                        # 同轮多档破位只建最深一档 (band.lower 最小), 其余记抑制
                        eligible_gap.sort(key=lambda b: (b['lower'], b['upper']))
                        deepest = eligible_gap[0]
                        for band in eligible_gap[1:]:
                            suppressions.append(dict(
                                day=day, code=code, plan_id=plan_id, band_id=band['id'],
                                lower=band['lower'], upper=band['upper'],
                                reason=SUPPRESSED_SAME_SCAN,
                                detail='DEEPER_BAND_SELECTED_GAP',
                                deeper_band_id=deepest['id'], at=scan_finished_at))
                        key = f'{day}:{plan_id}:{code}:{deepest["id"]}'
                        first_observed_touch_at = _now_stamp()
                        state['notified'][key] = dict(
                            price=price, lower=deepest['lower'], upper=deepest['upper'],
                            at=quote_time, event_key=key,
                            # V2.1 crossed 语义: 现价已在带下沿下方 (破位事实, 非穿越观察)
                            crossed=True,
                            trigger_policy=TRIGGER_POLICY_V2,
                            trigger_type=TRIGGER_OPEN_GAP_BELOW,
                            prev_price=None, prev_quote_time='',
                            scan_started_at=scan_started_at,
                            quote_attempt_at=quote_attempt_at,
                            quote_received_at=quote_received_at,
                            first_observed_touch_at=first_observed_touch_at,
                            first_observed_quote_at=quote_time,
                            event_recorded_at=first_observed_touch_at,
                            direction_class=DIRECTION_GAP_BELOW_BAND,
                            direction_previous_price=None,
                            direction_evidence='first_observed_quote',
                        )
                        events.append(dict(
                            event_key=key, code=code, name=row.get('name', ''),
                            price=price, lower=deepest['lower'], upper=deepest['upper'],
                            at=quote_time, first_at=quote_time,
                            plan_id=plan_id, crossed=True,
                            trigger_policy=TRIGGER_POLICY_V2,
                            trigger_type=TRIGGER_OPEN_GAP_BELOW,
                            prev_price=None,
                            scan_started_at=scan_started_at,
                            quote_attempt_at=quote_attempt_at,
                            quote_received_at=quote_received_at,
                            first_observed_touch_at=first_observed_touch_at,
                            first_observed_quote_at=quote_time,
                            event_recorded_at=first_observed_touch_at,
                            direction_class=DIRECTION_GAP_BELOW_BAND,
                            direction_previous_price=None,
                        ))
            # ---- 规则 A: 真实向下穿越 (相邻两次有效报价) ----
            crossing = []
            if first_in_band:
                crossing = [b for b in bands if b['lower'] <= price <= b['upper']]
            elif baseline_price is not None:
                for band in bands:
                    if baseline_price > band['upper'] and price <= band['upper']:
                        crossing.append(band)
            # 首笔带内也提醒，方向未知；仍使用原有档位身份与发送去重。
            day_quotes[code] = dict(price=price, quote_time=quote_time,
                                    observed_at=_now_stamp())
            dirty = True
            if not crossing:
                continue  # 反弹进入带内不推送；首笔带下在上方分支处理。
            # ---- 规则 B: 同股同日分档去重 (维度=交易日+代码, 跨 plan_id) ----
            notified_ranges, blocking_keys = _tier_state_locked(state, code, day)
            eligible = []
            for band in crossing:
                if blocking_keys:
                    # queued / 可重试 CANCELLED / 未发送 在途事件阻止其他档位
                    suppressions.append(dict(
                        day=day, code=code, plan_id=plan_id, band_id=band['id'],
                        lower=band['lower'], upper=band['upper'],
                        reason=SUPPRESSED_SAME_CODE,
                        detail='ALIVE_EVENT_BLOCKS_NEW_TIER',
                        blocking_event_key=blocking_keys[0],
                        at=scan_finished_at))
                    continue
                if notified_ranges:
                    min_lower = min(lo for lo, _hi in notified_ranges)
                    if band['upper'] >= min_lower:
                        # 不是更低的独立档位: 重叠带 / 仅略低带禁止再推
                        overlapped = any(band['lower'] <= hi and lo <= band['upper']
                                         for lo, hi in notified_ranges)
                        suppressions.append(dict(
                            day=day, code=code, plan_id=plan_id, band_id=band['id'],
                            lower=band['lower'], upper=band['upper'],
                            reason=(SUPPRESSED_OVERLAPPING_BAND if overlapped
                                    else SUPPRESSED_SAME_CODE),
                            detail='NOT_DEEPER_THAN_NOTIFIED_TIERS',
                            notified_min_lower=min_lower,
                            at=scan_finished_at))
                        continue
                eligible.append(band)
            # 身份已存在的候选 (FAILED 证据 / golden 已消费) 静默跳过, 不重建
            eligible = [b for b in eligible
                        if f'{day}:{plan_id}:{code}:{b["id"]}' not in state['notified']]
            if not eligible:
                continue
            # ---- 规则 C: 同轮多档只建当前到达的最深一档 (band.lower 最小) ----
            eligible.sort(key=lambda b: (b['lower'], b['upper']))
            deepest = eligible[0]
            for band in eligible[1:]:
                suppressions.append(dict(
                    day=day, code=code, plan_id=plan_id, band_id=band['id'],
                    lower=band['lower'], upper=band['upper'],
                    reason=SUPPRESSED_SAME_SCAN, detail='DEEPER_BAND_SELECTED',
                    deeper_band_id=deepest['id'], at=scan_finished_at))
            key = f'{day}:{plan_id}:{code}:{deepest["id"]}'
            first_observed_touch_at = _now_stamp()
            event_recorded_at = first_observed_touch_at
            state['notified'][key] = dict(
                price=price, lower=deepest['lower'], upper=deepest['upper'],
                at=quote_time, event_key=key,
                # V2 crossed 语义: 已观察到自上沿上方向下穿越上沿 (带身份事件恒为真)
                crossed=not first_in_band,
                trigger_policy=TRIGGER_POLICY_V2,
                trigger_type=TRIGGER_FIRST_IN_BAND if first_in_band else TRIGGER_DOWNWARD_CROSS_UPPER,
                prev_price=baseline_price, prev_quote_time=prev_quote_time,
                # ---- 证据链时间戳 ----
                scan_started_at=scan_started_at,
                quote_attempt_at=quote_attempt_at,
                quote_received_at=quote_received_at,
                first_observed_touch_at=first_observed_touch_at,
                first_observed_quote_at=quote_time,
                event_recorded_at=event_recorded_at,
                # 发送/指标时间戳后续在 reserve_send / record_delivery 里补
                # ---- 方向分类: 证据链来自 last_observed_quotes, 不是 quote.prev ----
                direction_class=DIRECTION_FIRST_QUOTE_UNKNOWN if first_in_band else DIRECTION_DOWNWARD_ENTRY,
                direction_previous_price=baseline_price,
                direction_evidence='first_observed_quote' if first_in_band else 'last_observed_quotes',
            )
            events.append(dict(
                event_key=key, code=code, name=row.get('name', ''),
                price=price, lower=deepest['lower'], upper=deepest['upper'],
                at=quote_time, first_at=quote_time,
                plan_id=plan_id, crossed=not first_in_band,
                trigger_policy=TRIGGER_POLICY_V2,
                trigger_type=TRIGGER_FIRST_IN_BAND if first_in_band else TRIGGER_DOWNWARD_CROSS_UPPER,
                prev_price=baseline_price,
                # ---- 证据链 ----
                scan_started_at=scan_started_at,
                quote_attempt_at=quote_attempt_at,
                quote_received_at=quote_received_at,
                first_observed_touch_at=first_observed_touch_at,
                first_observed_quote_at=quote_time,
                event_recorded_at=event_recorded_at,
                # ---- 方向 ----
                direction_class=DIRECTION_FIRST_QUOTE_UNKNOWN if first_in_band else DIRECTION_DOWNWARD_ENTRY,
                direction_previous_price=baseline_price,
            ))
        except (TypeError, ValueError, KeyError, AttributeError):
            continue  # per-stock isolation
    if suppressions:
        _record_suppressions_locked(state, suppressions)
        dirty = True
    # ---- last_observed_quotes 按日清理: 只保留当前交易日, 防膨胀 ----
    loq = state.get('last_observed_quotes')
    if loq and set(loq) - {day}:
        state['last_observed_quotes'] = {day: loq.get(day) or {}}
        dirty = True
    # 本轮整体扫描结束时间写 health (不进入逐事件)
    state.setdefault('health', {})
    state['health']['last_scan_started_at'] = scan_started_at
    state['health']['last_scan_finished_at'] = scan_finished_at
    state['health']['last_scan_event_count'] = len(events)
    if dirty:
        _save(state_file, state)
    return events  # 网络与推送均在锁外；锁只保护读-改-写


def _tier_state_locked(state, code, day):
    """同股同日的分档状态 (规则 B, 维度=交易日+代码, 跨 plan_id)。

    返回 (notified_ranges, blocking_keys):
      notified_ranges: 已送达档位区间 [(lower, upper), ...] —— delivery ∈
        {SENT, UNKNOWN} 或 consumed_by (golden_channel); 无区间记录的
        consumed 条目只锁自身身份, 无法参与深度比较。
      blocking_keys: 在途事件键 —— 无 delivery / queued / 可重试 CANCELLED,
        存在时阻止同股其他档位入队。
    FAILED 与 EXPIRED_NOT_SENT: 保留证据但不阻断、不算已送达。
    """
    prefix = f'{day}:'
    code = str(code).zfill(6)
    notified_ranges, blocking = [], []
    for key, entry in (state.get('notified') or {}).items():
        parts = key.split(':')
        if len(parts) != 4 or parts[0] != day or parts[2] != code:
            continue
        delivery = str(entry.get('delivery') or '')
        if entry.get('consumed_by') or delivery in ('SENT', 'UNKNOWN'):
            lo, hi = entry.get('lower'), entry.get('upper')
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                notified_ranges.append((float(lo), float(hi)))
            continue
        if delivery == 'FAILED' or delivery == 'EXPIRED_NOT_SENT':
            continue
        if delivery.startswith('CANCELLED_'):
            reason = delivery[len('CANCELLED_'):].split(':')[0]
            if reason in RETRYABLE_CANCEL:
                blocking.append(key)
            continue
        blocking.append(key)  # 未发送 / 排队中 / 未知状态 → 在途, 阻断新档位
    return notified_ranges, blocking


def _record_suppressions_locked(state, new_items):
    """suppression_events 有界存储: 每票每日上限 + 总量上限, 超出丢弃最旧。"""
    lst = state.setdefault('suppression_events', [])
    lst.extend(new_items)
    if len(lst) > SUPPRESSION_TOTAL_LIMIT:
        del lst[:-SUPPRESSION_TOTAL_LIMIT]  # 总量上限: 保留最新
    seen = {}
    kept = []
    for item in reversed(lst):  # 新→旧: 每票每日保留最新 N 条
        k = (item.get('day'), item.get('code'))
        seen[k] = seen.get(k, 0) + 1
        if seen[k] > SUPPRESSION_PER_CODE_DAILY_LIMIT:
            continue
        kept.append(item)
    kept.reverse()
    state['suppression_events'] = kept


def mark_notified(plan, code, state_file, band_id=None):
    """Consume the touch identity the confirmed signal ACTUALLY reached.
    band_id is REQUIRED: None consumes NOTHING (unknown ownership must never
    swallow the whole ladder — lower bands stay armed)."""
    if not band_id:
        return
    with _STATE_LOCK:
        _mark_locked(plan, code, state_file, band_id)


def _mark_locked(plan, code, state_file, band_id):
    day = str(plan.get('monitor_date') or '')
    plan_id = plan.get('assistant_plan_id') or 'snapshot'
    state = _load(state_file)
    changed = False
    for row in plan.get('rows') or []:
        if str(row.get('code') or '').zfill(6) != str(code).zfill(6):
            continue
        for band in bands_for(row):
            if band['id'] != band_id:
                continue
            key = f'{day}:{plan_id}:{code}:{band["id"]}'
            if key not in state['notified']:
                state['notified'][key] = dict(consumed_by='golden_channel', event_key=key,
                                              trigger_policy=TRIGGER_POLICY_V2)
                changed = True
    if changed:
        _save(state_file, state)


def pending_sends(plan, state_file, now_stamp, quotes):
    """Displayed-but-never-sent events eligible for delivery NOW, re-verified
    against the CURRENT plan and CURRENT quotes.

    V2: legacy 条目 (无 trigger_policy) 一律隔离 —— 不自动重发、不重解释,
    只补 legacy_reason 标记。V2 条目在报价有效且新鲜时即可交付: 下穿是已观察
    到的事实, 现价三态 (带内 / 已低于下沿 / 反弹离带) 均可发送, message 按
    现价如实描述。身份过期 (不同日/不同计划) 或报价过期的 V2 条目照旧归档为
    EXPIRED_NOT_SENT —— 旧价格绝不能以「现价」名义推送。"""
    with _STATE_LOCK:
        return _pending_locked(plan, state_file, now_stamp, quotes)


def _pending_locked(plan, state_file, now_stamp, quotes):
    day = str(plan.get('monitor_date') or '')
    plan_id = plan.get('assistant_plan_id') or 'snapshot'
    state = _load(state_file)
    out = []
    dirty = False
    for key, entry in state.get('notified', {}).items():
        if str(entry.get('trigger_policy') or '') != TRIGGER_POLICY_V2:
            # 旧事件按原语义隔离: 不自动重发, 只补标记
            if 'legacy_reason' not in entry:
                entry['legacy_reason'] = LEGACY_REASON
                dirty = True
            continue
        if entry.get('consumed_by') or ('queued' in entry):
            continue
        delivery = str(entry.get('delivery') or '')
        if delivery and not delivery.startswith('CANCELLED_'):
            continue  # SENT / UNKNOWN / FAILED / EXPIRED 终止态；可重试 CANCELLED 重新核验
        parts = key.split(':')
        if len(parts) != 4 or parts[0] != day or parts[1] != plan_id:
            state['notified'][key]['delivery'] = 'EXPIRED_NOT_SENT'
            dirty = True
            continue
        code = parts[2]
        quote = (quotes or {}).get(code) or {}
        if not _quote_ok(quote, day, now_stamp):
            state['notified'][key]['delivery'] = 'EXPIRED_NOT_SENT'
            dirty = True
            continue
        price = float(quote['price'])
        if entry.get('trigger_type') == TRIGGER_FIRST_IN_BAND and not float(entry['lower']) <= price <= float(entry['upper']):
            continue  # 首笔带内事件离带后等待重入，不追旧价。
        # V2: 现价三态均可交付 (带内 / 已低于下沿 / 触发后反弹离带), 无 waiting_reentry
        out.append(dict(event_key=key, code=code, name='',
                        price=price,
                        lower=float(entry.get('lower')), upper=float(entry.get('upper')),
                        at=str(quote.get('time') or ''),
                        first_at=entry.get('at') or str(quote.get('time') or ''),
                        plan_id=plan_id, crossed=bool(entry.get('crossed')),
                        trigger_policy=TRIGGER_POLICY_V2,
                        trigger_type=entry.get('trigger_type') or TRIGGER_DOWNWARD_CROSS_UPPER))
    if dirty:
        _save(state_file, state)
    return out


def reserve_send(event, state_file):
    """Mark event as queued for delivery; writes send_queued_at evidence timestamp."""
    with _STATE_LOCK:
        state = _load(state_file)
        entry = state['notified'].get(event['event_key'])
        if entry is not None:
            entry['queued'] = True
            entry['send_queued_at'] = _now_stamp()
            entry['send_queued_epoch'] = _epoch_now()
            _save(state_file, state)


RETRYABLE_CANCEL = ('STALE_OR_OUT_OF_BAND', 'PUSH_DISABLED',
                    'CONFIG_UNREADABLE', 'PLAN_CHANGED', 'POOL_BUSY')


def record_delivery(event, ok, state_file, cancelled=None,
                    send_started_at=None, reverify_quote_at=None):
    """Per-event delivery record — never rewrites other levels' receipts.

    新增证据链时间戳:
      send_started_at:   实际发送 worker 开始时间 (可 None, 由 worker 注入)
      reverify_quote_at: 发送前 reverify 拉取报价的时间戳 (可 None)
    """
    with _STATE_LOCK:
        _record_locked(event, ok, state_file, cancelled,
                       send_started_at, reverify_quote_at)


def _record_locked(event, ok, state_file, cancelled,
                   send_started_at, reverify_quote_at):
    state = _load(state_file)
    entry = state['notified'].get(event['event_key'])
    if entry is not None:
        if send_started_at is not None:
            entry['send_started_at'] = send_started_at
        if reverify_quote_at is not None:
            entry['reverify_quote_at'] = reverify_quote_at
        entry['send_finished_at'] = _now_stamp()
        if cancelled:
            reason = cancelled.split(':')[0]
            entry['delivery'] = 'CANCELLED_' + cancelled
            if reason in RETRYABLE_CANCEL:
                entry.pop('queued', None)  # 释放资格，等待重新具备条件
        elif ok is None:
            entry['delivery'] = 'UNKNOWN'  # 超时送达未知：不自动重发
            entry['sent_at'] = entry['send_finished_at']
        else:
            entry['delivery'] = 'SENT' if ok else 'FAILED'
            entry['sent_at'] = entry['send_finished_at']
        # 汇总健康监控
        state.setdefault('health', {})
        state['health']['last_delivery_at'] = entry['send_finished_at']
        state['health']['last_delivery_status'] = entry['delivery']
        _save(state_file, state)


def health_snapshot(state_file, *, day=None, now_stamp=None):
    """Return a health snapshot for the entire touch-alert lane, including
    per-symbol scan/quote ages, queue depths, expired counts — matches the
    task-book Phase-1 health-monitoring requirements.

    V2: 新增 SUPPRESSED_SAME_CODE / SUPPRESSED_OVERLAPPING_BAND /
    SUPPRESSED_SAME_SCAN 计数与 last_observed_quotes 基准数。"""
    state = _load(state_file)
    now_stamp = now_stamp or _now_stamp()
    day = day or now_stamp[:8]
    notified = state.get('notified', {})
    queued_events = [k for k, v in notified.items()
                     if v.get('queued') and not str(v.get('delivery') or '').startswith('CANCELLED_')]
    pending_events = [k for k, v in notified.items()
                      if not v.get('delivery') and not v.get('consumed_by')]
    sent_events = [k for k, v in notified.items() if v.get('delivery') == 'SENT']
    failed_events = [k for k, v in notified.items()
                     if v.get('delivery') in ('FAILED', 'UNKNOWN')]
    cancelled_events = [k for k, v in notified.items()
                        if str(v.get('delivery') or '').startswith('CANCELLED_')]

    # 逐票扫描健康 (从 entry 的 first_observed_touch_at / quote_received_at 推断)
    max_event_age_seconds = 0
    try:
        now_t = _epoch_now()
        for v in notified.values():
            ts = v.get('event_recorded_at') or v.get('first_observed_touch_at')
            if ts and len(ts) == 14:
                try:
                    t = time.mktime(time.strptime(ts, '%Y%m%d%H%M%S'))
                    age = now_t - int(t)
                    if age > max_event_age_seconds:
                        max_event_age_seconds = age
                except (ValueError, OverflowError, OSError):
                    pass
    except Exception:
        max_event_age_seconds = 0

    direction_counts = {}
    trigger_counts = {}
    for v in notified.values():
        d = v.get('direction_class') or 'UNKNOWN'
        direction_counts[d] = direction_counts.get(d, 0) + 1
        t = v.get('trigger_type') or 'LEGACY'
        trigger_counts[t] = trigger_counts.get(t, 0) + 1

    # ---- V2 抑制统计 ----
    suppression_items = state.get('suppression_events') or []
    suppression_counts = {}
    for s in suppression_items:
        r = str(s.get('reason') or 'UNKNOWN')
        suppression_counts[r] = suppression_counts.get(r, 0) + 1
    baselines = (state.get('last_observed_quotes') or {}).get(day) or {}

    return dict(
        contract=state.get('contract'),
        day=day,
        checked_at=now_stamp,
        # ---- 事件状态分布 ----
        total_events=len(notified),
        queued_count=len(queued_events),
        pending_count=len(pending_events),
        sent_count=len(sent_events),
        failed_count=len(failed_events),
        cancelled_count=len(cancelled_events),
        # ---- 时间戳 ----
        last_scan_started_at=state.get('health', {}).get('last_scan_started_at'),
        last_scan_finished_at=state.get('health', {}).get('last_scan_finished_at'),
        last_scan_event_count=state.get('health', {}).get('last_scan_event_count', 0),
        last_delivery_at=state.get('health', {}).get('last_delivery_at'),
        last_delivery_status=state.get('health', {}).get('last_delivery_status'),
        # ---- 年龄 ----
        max_event_age_seconds=max_event_age_seconds,
        # ---- 方向分布 ----
        direction_distribution=direction_counts,
        # ---- V2.1 触发类型分布 (含破位事件计数) ----
        trigger_type_distribution=trigger_counts,
        gap_event_count=trigger_counts.get(TRIGGER_OPEN_GAP_BELOW, 0),
        # ---- V2 抑制分布 ----
        suppression_count=len(suppression_items),
        suppression_distribution=suppression_counts,
        suppressed_same_code_count=suppression_counts.get(SUPPRESSED_SAME_CODE, 0),
        suppressed_overlapping_band_count=suppression_counts.get(SUPPRESSED_OVERLAPPING_BAND, 0),
        suppressed_same_scan_count=suppression_counts.get(SUPPRESSED_SAME_SCAN, 0),
        observed_quote_baseline_count=len(baselines),
        # ---- 队列 (touch lane 自身没有单独 queue, pentacles_monitor._TOUCH_QUEUE 是发送队列) ----
        queue_depth_proxy=len(queued_events),
    )


def message(event):
    at = event.get('at') or '未知'
    first = event.get('first_at')
    timing = f"首次到位 {first}；" if first and first != at else ''
    price = float(event['price'])
    lo, hi = float(event['lower']), float(event['upper'])
    if str(event.get('trigger_policy') or '') == TRIGGER_POLICY_V2:
        if event.get('trigger_type') == TRIGGER_FIRST_IN_BAND:
            where = (f"首次有效报价已在观察带 {lo:.4f}—{hi:.4f} 内；现价 {price:.2f}；"
                     "进入方向未知，未观察到此前穿越过程")
        elif str(event.get('trigger_type') or '') == TRIGGER_OPEN_GAP_BELOW:
            # V2.1 破位文案: 只陈述「首次有效观察已在带下方」事实, 绝不描述成盘中穿越
            base = (f"首次有效观察已低于观察带下沿 {lo:.4f}"
                    f"（开盘缺口或监控启动前已破位，未观察到盘中穿越过程，非连续逐笔成交）")
            if price < lo:
                where = f"{base}；现价 {price:.2f} 仍在下沿下方"
            elif price > hi:
                where = f"{base}；现价 {price:.2f} 已回到观察带上沿 {hi:.4f} 上方"
            else:
                where = f"{base}；现价 {price:.2f} 已回到观察带 {lo:.4f}—{hi:.4f} 内"
        else:
            # V2 三态文案: 下穿事实 + 现价位置; 快照报价绝不描述成连续逐笔成交
            base = (f"已观察到自上向下穿越观察带上沿 {hi:.4f}"
                    f"（由相邻两次快照报价判定，非连续逐笔成交）")
            if price < lo:
                where = f"{base}；现价 {price:.2f} 已低于观察带下沿 {lo:.4f}"
            elif price > hi:
                where = f"{base}；触发后已反弹离带，现价 {price:.2f} 回到上沿上方"
            else:
                where = f"{base}；现价 {price:.2f} 仍在观察带 {lo:.4f}—{hi:.4f} 内"
    elif event.get('crossed'):
        # legacy 快照报价无连续行情证据，不能确认跳空；陈述带下方事实。
        where = (f"当前已在观察带下方{lo:.4f}，现价 {price:.2f}"
                 f"（快照报价，非连续行情，不能确认跳空）")
    else:
        where = (f"现价 {price:.2f} 进入观察带 "
                 f"{lo:.4f}—{hi:.4f}")
    return (f"{event.get('name','')} {event['code']}｜{where}\n"
            f"{timing}最新报价 {at}；实时试算、未收盘；指标与三选二待核验，"
            "本提醒不等待五分钟收盘，不代表最低价已现，非买入建议，不自动下单。")


def reverify(event, quote, now_stamp, state_file=None):
    """Queue-side recheck before send.

    V2 语义 (trigger_policy='V2'):
      - 报价必须有效、同事件交易日、新鲜（同 _quote_ok 门）
      - 事件必须是 V2 策略版本
      - 提供 state_file 时核对送达状态: 已 SENT/UNKNOWN 或已被 golden 消费
        → 拒绝（不重复送达）
      - 现价三态（带内 / 已低于下沿 / 触发后反弹离带）均可发送: 下穿是已观察
        到的事实, 不因反弹失效, message 按现价如实描述
    legacy 事件保持原语义（隔离路径兜底, 兼容升级瞬间仍在途的旧队列事件）:
      crossed → price <= lower; 否则必须仍在带内。此处不改状态。"""
    parts = str(event.get('event_key') or '').split(':')
    if len(parts) != 4 or not _quote_ok(quote or {}, parts[0], now_stamp):
        return False
    if str(event.get('trigger_policy') or '') == TRIGGER_POLICY_V2:
        if state_file is not None:
            with _STATE_LOCK:
                entry = (_load(state_file).get('notified') or {}).get(event['event_key'])
                if entry is None or entry.get('consumed_by'):
                    return False
                if str(entry.get('delivery') or '') in ('SENT', 'UNKNOWN'):
                    return False
        if event.get('trigger_type') == TRIGGER_FIRST_IN_BAND:
            return float(event['lower']) <= float(quote['price']) <= float(event['upper'])
        return True
    price = float(quote['price'])
    lo, hi = float(event['lower']), float(event['upper'])
    if event.get('crossed'):
        return price <= lo
    return lo <= price <= hi
