"""Async strong-pool event delivery and indicator enrichment."""
from datetime import datetime
import threading

_ALERT_LOCK = threading.RLock()


def _same_band(left, right):
    return (abs(float(left.get('lower', 0)) - float(right.get('lower', 0))) < 1e-8 and
            abs(float(left.get('upper', 0)) - float(right.get('upper', 0))) < 1e-8 and
            left.get('landmark_ratio') == right.get('landmark_ratio'))


def deliver(event_id, state_file, plan_ref, *, get_plan, get_cfg, quote_fn, broadcast, send, active):
    """Deliver a recorded touch after fresh quote/plan/band revalidation."""
    import strong_touch_lane_v1 as lane
    import strong_sector_watch_v1 as watch
    event = lane.get_event(state_file, event_id)
    if not event:
        return
    now = datetime.now()
    current_plan = get_plan() or {}
    from golden_rules import digest as plan_digest
    if plan_digest(current_plan) != plan_digest(plan_ref) or not active(current_plan, now):
        lane.update_event(state_file, event_id, delivery_status='CANCELLED_PLAN_CHANGED')
        return
    from sector_watch_v1 import names
    pool = watch.frozen(names(current_plan), int(now.strftime('%Y%m%d')))
    if not pool or pool.get('rows_sha256') != event.get('pool_id'):
        lane.update_event(state_file, event_id, delivery_status='CANCELLED_POOL_CHANGED')
        return
    row = next((r for r in pool.get('rows', []) if str(r.get('code')) == str(event['code'])), None)
    if not row:
        lane.update_event(state_file, event_id, delivery_status='CANCELLED_SYMBOL_REMOVED')
        return
    band = next((b for b in row.get('positions', []) if _same_band(b, event.get('band') or {})), None)
    if not band:
        lane.update_event(state_file, event_id, delivery_status='CANCELLED_BAND_CHANGED')
        return

    quote = (quote_fn([event['code']]) or {}).get(event['code']) or {}
    now = datetime.now()
    from strong_pullback_policy_v1 import fresh
    if not fresh(quote, now):
        lane.update_event(state_file, event_id, delivery_status='WAIT_REVERIFY',
            delivery_retry_after_epoch=now.timestamp() + lane.POLL_INTERVAL_SECONDS,
            reverify_error='QUOTE_STALE_OR_INVALID')
        return
    latest_plan = get_plan() or {}
    if plan_digest(latest_plan) != plan_digest(plan_ref) or not active(latest_plan, datetime.now()):
        lane.update_event(state_file, event_id, delivery_status='CANCELLED_PLAN_CHANGED')
        return

    from push_layout_v1 import notice
    price = float(quote['price'])
    lower, upper = float(band['lower']), float(band['upper'])
    location = lane.band_location(price, band)
    containing = sum(float(b['lower']) <= price <= float(b['upper']) for b in row.get('positions', []))
    packet = dict(asof=int(quote['time']), indicator_asof=event['indicator_cutoff'],
        code=row['code'], name=row['name'], board='、'.join(row.get('boards') or []),
        price=price, return_value=price / float(quote['prev']) - 1,
        level_lower=lower, level_upper=upper,
        level_label='共振带' if band.get('source_waves', 0) >= 2 else '黄金位',
        priority_label='P1' if band.get('primary') else 'P2',
        wave_low=row.get('low'), wave_high=row.get('high'), wave_return=row.get('wave_return'),
        indicator_basis='价格触位先行；5/15分钟指标后台计算中', period_pairs={},
        relative_turnover=dict(status='UNKNOWN', reason='指标后台计算中'),
        morning_mark='首次观察到触位；发送前已重新核验报价和冻结观察带',
        trigger_text=('价格到达路标；完整K线收盘确认待核验' if row['pool'] == 'LANDMARK' else None))
    packet['distance'] = (price - max(lower, min(upper, price))) / price
    formatted = notice(event.get('kind') or 'STRONG_STOCK', packet)
    direction = {
        'DOWNWARD_ENTRY_OBSERVED': '观察到从上方向下进入',
        'FIRST_QUOTE_IN_BAND_DIRECTION_UNKNOWN': '首次有效报价已在带内，方向未知',
        'REBOUND_FROM_BELOW_OBSERVED': '观察到从下方反弹进入',
        'DIRECTION_UNKNOWN': '方向证据不足'}.get(event.get('direction_class'), '方向未分类')
    location_text = {'STILL_IN_ORIGINAL_BAND': '仍在原观察带内',
        'BELOW_ORIGINAL_BAND': '已在原观察带下方',
        'ABOVE_ORIGINAL_BAND': '已回到原观察带上方'}.get(location, location)
    evidence = (f"\n\n触位方向：{direction}。"
        f"\n首次观察到触位：{event.get('first_observed_touch_at')}；程序记录：{event.get('first_observed_at')}。"
        "（首次观察时间不等于市场实际首次触位时间。）"
        f"\n发送前复核：现价 {price:.2f}；{location_text}；当前落入观察带数量 {containing}。"
        "\n指标状态：后台计算中；价格触位不代表止跌、支撑成立或允许买入。")
    body = formatted['body'] + evidence
    title = f"强势池 · {row['code']} {formatted['title']}"
    key = watch.digest([int(event['day']), row['code'], row['pool'], band['lower'], band['upper']])
    alerts_file = watch.root() / 'alerts.json'
    record = dict(code=row['code'], cutoff=event['indicator_cutoff'], pool_id=pool['rows_sha256'],
        event=dict(event), packet=packet, title=title, body=body, delivery='RECORDED', wechat=False,
        first_observed_touch_at=event.get('first_observed_touch_at'),
        first_observed_at=event.get('first_observed_at'), send_queued_at=event.get('send_queued_at'),
        send_started_at=datetime.now().isoformat(timespec='seconds'),
        reverify_quote=dict(time=quote.get('time'), price=price), reverify_band_status=location,
        indicator_status='PENDING')
    with _ALERT_LOCK:
        alerts = json.loads(alerts_file.read_bytes()) if alerts_file.exists() else {}
        if any(v.get('code') == row['code'] and str(v.get('cutoff', ''))[:8] == str(event['day'])
               for v in alerts.values()):
            lane.update_event(state_file, event_id, delivery_status='DUPLICATE_SUPPRESSED')
            return
        alerts[key] = record
        watch.save(alerts_file, alerts)
    broadcast(dict(type='push', kind='strong_sector', title=title, body=body,
                   wechat=False, delivery='RECORDED'))
    cfg = get_cfg() or {}
    ok, message, delivery = False, '微信推送已关闭', 'DISABLED'
    if (cfg.get('push') or {}).get('enabled'):
        try:
            ok, message = send(cfg, title, body)
            delivery = 'SENT' if ok else 'FAILED'
        except Exception as exc:
            message, delivery = type(exc).__name__, 'FAILED'
    finished = datetime.now().isoformat(timespec='seconds')
    with _ALERT_LOCK:
        latest = json.loads(alerts_file.read_bytes()) if alerts_file.exists() else {}
        if key in latest:
            latest[key].update(wechat=bool(ok), delivery=delivery, wechat_msg=str(message),
                send_finished_at=finished, sent_at=finished if delivery == 'SENT' else None)
            watch.save(alerts_file, latest)
    lane.update_event(state_file, event_id, delivery_status=delivery, send_finished_at=finished,
        sent_at=finished if delivery == 'SENT' else None,
        reverify_quote=dict(time=quote.get('time'), price=price), reverify_band_status=location)



def compute_indicators(event_id, state_file, plan_ref):
    """Compute indicator context off the quote/delivery path; never gates touch."""
    import strong_touch_lane_v1 as lane
    import strong_sector_watch_v1 as watch
    event = lane.get_event(state_file, event_id)
    if not event:
        return
    from sector_watch_v1 import names
    pool = watch.frozen(names(plan_ref or {}), int(event['day']))
    if not pool or pool.get('rows_sha256') != event.get('pool_id'):
        lane.update_event(state_file, event_id, indicator_status='CANCELLED_POOL_CHANGED',
                          indicator_finished_at=datetime.now().isoformat(timespec='seconds'))
        return
    row = next((r for r in pool.get('rows', []) if str(r.get('code')) == str(event.get('code'))), None)
    if not row:
        lane.update_event(state_file, event_id, indicator_status='CANCELLED_SYMBOL_REMOVED',
                          indicator_finished_at=datetime.now().isoformat(timespec='seconds'))
        return
    result = watch.invoke(dict(mode='INDICATORS', row=row, cutoff=event['indicator_cutoff'],
                               quote=event.get('initial_quote') or {}))
    finished = datetime.now().isoformat(timespec='seconds')
    summary = {k: result.get(k) for k in ('status', 'reason', 'asof', 'last_time',
              'periods', 'relative_turnover', 'volume_ratio') if k in result}
    lane.update_event(state_file, event_id, indicator_status=result.get('status', 'UNKNOWN'),
                      indicator_finished_at=finished, indicator_result=summary)

