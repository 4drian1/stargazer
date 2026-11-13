"""Independent quote reminders. No minute-bar execution, orders or buy advice."""
from datetime import datetime
import json
import time
import math
from pathlib import Path
import queue
import threading
from pentacles_journal import ObservationJournal

_TOUCH_QUEUE = None
_TOUCH_WORKER = None


_TIMEOUT_POOL = None
_TIMEOUT_SENTINEL = object()
_REJECTED_SENTINEL = object()  # 池满拒收：任务未执行，可重试
_POOL_SUBMITTED = [0]
_POOL_LOCK = threading.Lock()
QUOTE_LANE_MIN_SECONDS = 10
QUOTE_LANE_MAX_SECONDS = 15
QUOTE_LANE_DEFAULT = 12
QUOTE_TIMEOUT_SECONDS = 6
SEND_TIMEOUT_SECONDS = 10
QUOTE_DAILY_CAP = 2000
_BUDGET_LOCK = threading.Lock()
_BUDGET_FILE = [None]


def _budget_gate(url):
    """请求前额度闸：加锁查余额→持久化预约→不足则拦截该次请求。"""
    file = _BUDGET_FILE[0]
    if file is None:
        return True
    day = time.strftime('%Y%m%d')
    with _BUDGET_LOCK:
        try:
            budget = json.loads(Path(file).read_bytes())
        except (OSError, ValueError):
            budget = dict(day=day, polls=0, blocked=False)
        if budget.get('day') != day:
            budget = dict(day=day, polls=0, blocked=False)
        if budget['polls'] >= QUOTE_DAILY_CAP:
            if not budget.get('blocked'):
                budget['blocked'] = True
                Path(file).write_text(json.dumps(budget), encoding='utf-8')
                try:
                    from notifier import log as _log
                    _log(f'报价通道当日请求达{QUOTE_DAILY_CAP}次上限，后续尝试拦截')
                except Exception:
                    pass
            return False
        budget['polls'] += 1
        Path(file).write_text(json.dumps(budget), encoding='utf-8')
        # This 2,000/day limit is plan-ticket-lane only. The shared ledger is
        # observational accounting across lanes; it does not impose a global cap.
        try:
            from quote_budget_v1 import record_attempt
            record_attempt(Path(file).with_name('quote_request_usage.json'), 'plan_ticket')
        except Exception:
            pass  # Accounting failure must not silently change the existing lane contract.
        return True


def _budget_count(file):
    try:
        return json.loads(Path(file).read_bytes()).get('polls', 0)
    except (OSError, ValueError):
        return 0


def _timeout_pool():
    global _TIMEOUT_POOL
    if _TIMEOUT_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _TIMEOUT_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix='bounded-io')
    return _TIMEOUT_POOL


def _with_timeout(fn, args, seconds):
    """Bounded: full pool → reject (sentinel), never queue behind slow work;
    timeout abandons the task and cancels it if it never started."""
    pool = _timeout_pool()
    with _POOL_LOCK:
        if _POOL_SUBMITTED[0] >= 4:
            return _REJECTED_SENTINEL  # 未执行，与执行后超时区分
        _POOL_SUBMITTED[0] += 1
    future = pool.submit(fn, *args)
    future.add_done_callback(lambda _f: _pool_done())
    try:
        return future.result(timeout=seconds)
    except Exception:
        future.cancel()
        return _TIMEOUT_SENTINEL


def _pool_done():
    with _POOL_LOCK:
        _POOL_SUBMITTED[0] = max(0, _POOL_SUBMITTED[0] - 1)


def _quotes_with_timeout(codes, seconds=QUOTE_TIMEOUT_SECONDS, budget_file=None):
    from tools.market import quotes
    if budget_file is not None:
        _BUDGET_FILE[0] = budget_file
    result = _with_timeout(
        lambda: quotes(codes, timeout=seconds, on_attempt=_budget_gate, batch_size=30),
        (), seconds + 1)
    return result if isinstance(result, dict) else {}


def _plan_current(plan):
    """Queue-side recheck at SEND time: the confirmed plan file must exist
    and still carry this exact plan. Missing file means there is nothing
    confirmed to alert under — cancel, never default-allow."""
    try:
        import pentacles_demo
        current = pentacles_demo.snapshot_path().with_name('assistant_plan.json')
        if not current.exists():
            return False
        from golden_rules import digest
        return digest(json.loads(current.read_bytes())) == digest(plan)
    except Exception:
        return False



def _deliver_one(cfg, title, body, state_file, event, plan_ref,
                 quote_fn=None, send_fn=None, now_stamp=None):
    """One queued touch delivery with all guards. quote_fn/send_fn injectable
    for tests; production uses the real market quote and notifier send."""
    import time as _time
    import touch_alert_lane_v1 as lane
    ok, cancelled = False, None
    # 消费前读取当前开关与当前确认计划；读取失败取消该事件，绝不回退
    # 入队时的旧开关快照。取消与接口失败分别记账。
    try:
        from agent.brain import load_config
        cfg_now = load_config()
    except Exception as exc:
        cancelled = 'CONFIG_UNREADABLE:' + type(exc).__name__
        cfg_now = None
    if cancelled is None and not (cfg_now.get('push') or {}).get('enabled'):
        cancelled = 'PUSH_DISABLED'
    if cancelled is None and not _plan_current(plan_ref):
        cancelled = 'PLAN_CHANGED'
    if cancelled is None:
        # 队列等待过久：发送前取真实报价(独立超时+请求前额度闸)，重核时效与在带/带下状态。
        try:
            if quote_fn is None:
                fetched = _quotes_with_timeout([event['code']],
                                               budget_file=Path(state_file).with_name('quote_lane_budget.json'))
                q = fetched.get(event['code']) or {}
            else:
                q = (quote_fn([event['code']]) or {}).get(event['code']) or {}
            stamp = now_stamp or int(_time.strftime('%Y%m%d%H%M%S'))
            if not lane.reverify(event, q, stamp):
                cancelled = 'STALE_OR_OUT_OF_BAND'
            else:
                # 发送时用最新报价重建正文；首次到位时间与最新报价时间分列。
                body = lane.message(dict(event, price=float(q['price']),
                                         at=str(q.get('time') or event.get('at')),
                                         first_at=event.get('first_at') or event.get('at')))
        except Exception:
            cancelled = 'REVERIFY_FAILED'
    if cancelled is None:
        if send_fn is None:
            import notifier
            send_fn = notifier.send_wechat
        sent = _with_timeout(send_fn, (cfg_now, title, body), SEND_TIMEOUT_SECONDS)
        if sent is _REJECTED_SENTINEL:
            cancelled = 'POOL_BUSY'  # 从未执行：释放资格等待重试，不记UNKNOWN
        elif sent is _TIMEOUT_SENTINEL:
            ok = None  # 已执行但超时：送达未知，保持不自动重发
        else:
            try:
                ok, _ = sent
            except Exception:
                ok = False
    lane.record_delivery(event, ok, state_file, cancelled=cancelled)
    return ok, cancelled


def _quote_lane_ok(self, plan, now):
    """Same hard gates as the shared entry, evaluated by the fast lane."""
    try:
        data = self._load()
        if not data.get('enabled') or not plan or not plan.get('plan_ready'):
            return False
        if int(now.strftime('%Y%m%d')) != int(plan.get('monitor_date') or 0):
            return False
        if not session(now):
            return False
        if not plan.get('assistant_plan_id'):
            return False
        current = Path(self.directory()) / 'assistant_plan.json'
        if not current.is_file():
            return False
        from golden_rules import digest
        return digest(json.loads(current.read_bytes())) == digest(plan)
    except Exception:
        return False


def _touch_send_queue():
    """One worker drains touch-alert sends: a slow push request never blocks
    the quote loop or other stocks. Failures are recorded, never retried."""
    global _TOUCH_QUEUE, _TOUCH_WORKER
    if _TOUCH_QUEUE is None:
        _TOUCH_QUEUE = queue.Queue()
        def work():
            while True:
                cfg, title, body, state_file, event, plan_ref = _TOUCH_QUEUE.get()
                try:
                    _deliver_one(cfg, title, body, state_file, event, plan_ref)
                except Exception:
                    pass
                _TOUCH_QUEUE.task_done()
        _TOUCH_WORKER = threading.Thread(target=work, daemon=True, name='touch-send-queue')
        _TOUCH_WORKER.start()
    return _TOUCH_QUEUE


def atomic_json(file, data):
    raw = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError('提醒状态超过 2MiB，停止写入')
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_suffix('.tmp')
    temp.write_bytes(raw)
    temp.replace(file)


def session(now):
    hm = now.hour * 100 + now.minute
    return now.weekday() < 5 and (930 <= hm < 1130 or 1300 <= hm < 1500)


def risk_session(now):
    return session(now) or (now.weekday()<5 and ((now.hour==11 and 30<=now.minute<=36) or (now.hour==15 and now.minute<=6)))


def patrol_recorded(alerts, bulletin):
    for alert in alerts.values():
        review = alert.get('review') or {}
        if review.get('decision') == 'REPORT' and (
            review.get('bulletin_key') == bulletin['key'] or
            (review.get('cutoff') == bulletin['asof'] and alert.get('plan_id') == bulletin['plan_id'])
        ):
            return True
    return False


def combination_passed(row):
    return (row.get('status')=='RULE_TRIGGERED' and (row.get('combination_policy')=='POSITION_TWO_OF_THREE_O20_CROSS1_V1' or str(row.get('combination_policy','')).startswith('PLAN_CUSTOM_V1:'))
        and sum(v is True for v in row.get('combination',{}).get('votes',{}).values())>=row.get('combination',{}).get('required',2))


def combination_name(signal):
    return '三选二' if signal.get('combination_policy')=='POSITION_TWO_OF_THREE_O20_CROSS1_V1' else '自定义组合'


def rule_condition_text(signal):
    combination=signal.get('combination') or {}
    names={'repair':'位置修复','momentum':'动能','oversold':str(combination.get('oversold_ratio',.2)*100)+'%严重超卖'}
    votes=(signal.get('combination') or {}).get('votes') or {}
    if not (signal.get('combination_policy')=='POSITION_TWO_OF_THREE_O20_CROSS1_V1' or str(signal.get('combination_policy','')).startswith('PLAN_CUSTOM_V1:')) or signal.get('status')=='DATA_INSUFFICIENT':
        votes={}
    groups=[('已满足',[label for key,label in names.items() if votes.get(key) is True]),
        ('未满足',[label for key,label in names.items() if votes.get(key) is False]),
        ('待核验',[label for key,label in names.items() if votes.get(key) is not True and votes.get(key) is not False])]
    text='；'.join(label+'：'+'、'.join(items) for label,items in groups if items)
    if votes:
        text+='（三选'+str(combination.get('required',2))+'：'+str(sum(votes.get(k) is True for k in names))+'/'+str(combination.get('required',2))+'）'
    else:
        reasons={'CURRENT_COMPLETE_5M_PREFIX_MISSING':'当前完整5分钟K线缺失',
            'HISTORY_SEED_MISSING':'历史分钟基准不足','PREVIOUS_SESSION_INCOMPLETE':'昨日完整分钟基准缺失',
            'CROSSDAY_RSI_SEED_UNKNOWN':'跨日RSI基准缺失',
            'EXECUTION_CONTRACT_CHANGED_WAIT_CURRENT_5M':'等待当前规则重新核验'}
        why=reasons.get(signal.get('reason')) or '、'.join(signal.get('pending_conditions') or []) or '等待同截止完整5分钟核验结果'
        text+='；原因：'+why
    return text+'。'


def check_quote(row, quote, now):
    """Return a quality reason; snapshot close and live previous close must agree."""
    try:
        stamp = datetime.strptime(str(quote['time']), '%Y%m%d%H%M%S')
        age = (now - stamp).total_seconds()
        if stamp.date() != now.date() or not 0 <= age <= 90:
            return 'QUOTE_STALE'
        values = [float(quote['price']), float(quote['prev']), float(row['reference_price']),
                  float(row['zone']['zone_lower']), float(row['zone']['zone_upper'])]
        if not all(math.isfinite(x) and x > 0 for x in values) or values[3] > values[4]:
            return 'INVALID_PRICE'
        if abs(values[1] - values[2]) > 0.011:
            return 'PRICE_BASIS_MISMATCH'
        return None
    except (KeyError, TypeError, ValueError):
        return 'QUOTE_OR_ZONE_UNKNOWN'


def observation_reviews(plan, quotes, now, alerts, signals=None):
    """Quote touches are notifications, never execution confirmations."""
    if int(plan['monitor_date']) != int(now.strftime('%Y%m%d')) or not session(now):
        return []
    from golden_rules import digest
    reviews = []
    for row in plan.get('rows', []):
        rule=row.get('entry_rule') or {}
        levels = rule.get('levels') or []
        limits={v:v for v in levels}
        if rule.get('policy')=='GOLDEN_RESONANCE_CROSSDAY_V4':
            from golden_watch_groups import watch_groups
            limits={b['lower']:b['zone_upper'] for b in watch_groups(rule)}
            levels=sorted(limits,reverse=True)
        if not levels:
            continue
        quote = quotes.get(row['code']) or {}
        basis = dict(row, zone=dict(zone_lower=min(levels), zone_upper=max(levels)))
        if check_quote(basis, quote, now):
            continue
        price = float(quote['price'])
        try:
            low = float(quote.get('low', price))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(low) or not 0 < low <= price:
            continue
        # Expanding another board must not repeat an existing stock's touch.
        from plan_board_change_v1 import stock_alert_identity
        identity=plan.get('board_parent_digest') or stock_alert_identity(plan)
        prefix = f"touch:{identity}:{now:%Y%m%d}:{row['code']}:"
        reached = [v for v in levels if low <= limits[v] and 'tarot-v2:'+prefix+str(v) not in alerts]
        if not reached:
            continue
        signal=(signals or {}).get(row['code']) or {}
        pending=signal.get('pending_conditions') or (
            ['风险与AI综合意见待核验'] if signal.get('status')=='RULE_TRIGGERED' else ['完整5分钟组合尚未确认'])
        detail=(combination_name(signal)+'已通过，等待风险与DeepSeek综合意见。' if combination_passed(signal) else
            '尚未确认：'+'、'.join(pending)+'。建议观察并等待。')
        if rule.get('policy')=='GOLDEN_RESONANCE_CROSSDAY_V4' or signal.get('combination_policy')=='POSITION_TWO_OF_THREE_O20_CROSS1_V1':
            detail=rule_condition_text(signal)+(combination_name(signal)+'已通过，等待风险与DeepSeek综合意见。' if combination_passed(signal) else '建议观察并等待。')
        reviews.append(dict(status='ADVICE_READY', decision='OBSERVE', code=row['code'],
            cutoff=int(quote['time']), event_id=prefix+str(min(reached)),
            covered_event_ids=[prefix+str(v) for v in reached], action='黄金观察位已到达',
            pending_conditions=pending,observed_regions=[dict(lower=v,upper=limits[v]) for v in reached],reason='观察区 '+
                '、'.join(f'{v:.2f}—{limits[v]:.2f}' for v in reached)+'；'+detail))
    return reviews


class PriceMonitor:
    def __init__(self, directory):
        self.directory = directory
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.strong_quote_thread = None
        self.strong_touch_health = {'status': 'NOT_STARTED', 'pool_count': 0}
        self.health = {'status': 'NOT_STARTED', 'rows': {}, 'message': '到价提醒尚未启动'}
        self.journal = ObservationJournal(lambda: Path(self.directory()) / 'journal')

    def _load(self):
        file = Path(self.directory()) / 'monitor.json'
        if not file.exists():
            return {'enabled': False, 'alerts': {}, 'date': None}
        if file.stat().st_size > 2 * 1024 * 1024:
            raise ValueError('提醒状态文件过大')
        data = json.loads(file.read_text(encoding='utf-8'))
        if type(data.get('enabled')) is not bool or not isinstance(data.get('alerts'), dict):
            raise ValueError('提醒状态损坏，需人工处理；不重置去重记录')
        return data

    def state(self):
        with self.lock:
            try:
                data = self._load()
                return dict(self.health, enabled=data['enabled'], alerts=data['alerts'],
                            strong_touch_lane=dict(self.strong_touch_health))
            except Exception as exc:
                return {'enabled': False, 'status': 'STATE_ERROR', 'message': str(exc), 'rows': {}}

    def enable(self, value):
        if type(value) is not bool:
            raise ValueError('enabled 必须为布尔值')
        with self.lock:
            data = self._load()
            data['enabled'] = value
            atomic_json(Path(self.directory()) / 'monitor.json', data)
            self.health.update(status='WAITING' if value else 'DISABLED',
                               message='已开启，等待有效名单/交易时段' if value else '次日到价提醒关闭')

    def tick(self, plan, cfg, now, quotes, broadcast, send, clock=None):
        self.journal.register(plan)
        try:
            self._tick(plan, cfg, now, quotes, broadcast, send, clock)
        except Exception:
            with self.lock:
                self.health.update(status='MONITOR_ERROR', message='行情/监控失败；日记不记为无触价')
            raise
        finally:
            with self.lock:
                health = dict(self.health, rows=dict(self.health.get('rows', {})))
            self.journal.record(plan, health, clock() if clock else now)

    def _tick(self, plan, cfg, now, quotes, broadcast, send, clock=None):
        # Shared gates for BOTH branches (current TAROT_A_SECOND_CASH_V2 plans
        # included): enabled, plan ready, observation day, session, and the
        # confirmed-plan identity. The touch lane runs AFTER these gates and
        # BEFORE any indicator computation — never behind golden_monitor.evaluate.
        with self.lock:
            data = self._load()
            self.health = {'status': 'DISABLED', 'message': '次日到价提醒关闭', 'rows': {},
                           'checked_at': now.isoformat(timespec='seconds')}
            if not data['enabled']:
                return
            if not plan or not plan.get('plan_ready'):
                self.health.update(status='PLAN_NOT_READY', message='请先更新日线并生成有效盘后名单')
                return
            today = int(now.strftime('%Y%m%d'))
            target = int(plan.get('monitor_date') or 0)
            if today != target:
                return
            current_file = Path(self.directory()) / 'assistant_plan.json'
            if current_file.exists():
                from golden_rules import digest
                if digest(json.loads(current_file.read_bytes())) != digest(plan):
                    self.health.update(status='PLAN_CHANGED', message='确认计划已变化，丢弃旧工作进程结果')
                    return
        # 触价检测由独立报价通道(10-15s, _start_quote_lane)负责；_tick 只做指标通道。
        if plan and plan.get('rule_version'):
            return self._tick_golden(plan, cfg, now, broadcast, send, clock, quotes)
        with self.lock:
            data = self._load()
            self.health = {'status': 'DISABLED', 'message': '次日到价提醒关闭', 'rows': {},
                           'checked_at': now.isoformat(timespec='seconds')}
            if not data['enabled']:
                return
            if not plan or not plan.get('plan_ready'):
                self.health.update(status='PLAN_NOT_READY', message='请先更新日线并生成有效盘后名单')
                return
            today = int(now.strftime('%Y%m%d'))
            target = int(plan['monitor_date'])
            if today != target:
                self.health.update(status='WAITING' if today < target else 'PLAN_EXPIRED',
                                   message=f'观察日 {target}；不使用过期名单提醒')
                return
            if not session(now):
                self.health.update(status='OUTSIDE_SESSION', message='等待交易时段；午休/收盘不推送')
                return
            rows = [row for row in plan['rows'] if row.get('zone')]
        # Slow network calls do not hold the UI/settings lock. Recheck switch before delivery.
        live = quotes([row['code'] for row in rows]) if rows else {}
        now = clock() if clock else now
        with self.lock:
            data = self._load()
            if not data['enabled']:
                return
            if int(now.strftime('%Y%m%d')) != today or not session(now):
                self.health.update(status='OUTSIDE_SESSION', message='取行情期间已离开交易时段，不提醒')
                return
            if data.get('date') != today:
                data.update(date=today, alerts={})
            self.health.update(status='MONITORING', message='报价监控中；到价只提醒人工复核')
            for row in rows:
                if clock:
                    now = clock()
                    if int(now.strftime('%Y%m%d')) != today or not session(now):
                        break
                code = row['code']
                quote = live.get(code) or {}
                quote = quote if isinstance(quote, dict) else {}
                reason = check_quote(row, quote, now)
                view = {'status': reason or 'WAIT_PRICE', 'price': None if reason else float(quote['price']),
                        'time': quote.get('time')}
                self.health['rows'][code] = view
                if reason:
                    continue
                price = float(quote['price'])
                key = f'{today}:{code}'
                alert = data['alerts'].get(key)
                if alert is None and price <= row['zone']['zone_upper']:
                    below = price < row['zone']['zone_lower']
                    title = f"星币侍从 · {code} {'跌破关注区域' if below else '触及关注价'}"
                    body = (f"{row.get('name', '')} {code}｜现价 {price:.2f}\n"
                            f"关注区域 {row['zone']['zone_lower']:.2f}–{row['zone']['zone_upper']:.2f}\n"
                            f"{'已在区域下方，请警惕破位。' if below else ''}请人工复核；不是买入建议。\n"
                            f"报价时刻 {quote['time']}｜数据日 {plan['data_date']}｜仅报价触及提醒，非成交信号。")
                    alert = {'title': title, 'body': body, 'attempts': 0, 'wechat': False,
                             'created_at': now.isoformat(timespec='seconds'), 'delivery': 'PENDING'}
                    data['alerts'][key] = alert
                    atomic_json(Path(self.directory()) / 'monitor.json', data)
                    broadcast({'type': 'push', 'kind': 'pentacles_price', 'title': title,
                               'body': body, 'wechat': False, 'wechat_msg': '到价提醒已记录',
                               'time': now.strftime('%H:%M:%S')})
                if alert:
                    view['status'] = 'ALERT_RECORDED'
                    if not cfg.get('push', {}).get('enabled'):
                        alert['delivery'] = 'WECHAT_DISABLED'
                    elif not alert['wechat'] and alert['attempts'] < 3:
                        # Persist the attempt first; restart cannot silently reset the retry budget.
                        alert['attempts'] += 1
                        atomic_json(Path(self.directory()) / 'monitor.json', data)
                        # Sending may take seconds; never block the page or the off switch.
                        self.lock.release()
                        try:
                            ok, message = send(cfg, alert['title'], alert['body'])
                        finally:
                            self.lock.acquire()
                        alert.update(wechat=bool(ok), delivery='SENT' if ok else 'FAILED',
                                     delivery_message='推送成功' if ok else '推送失败；最多尝试三次，请检查推送设置')
                        latest = self._load()
                        data['enabled'] = latest['enabled']
                        if not data['enabled']:
                            atomic_json(Path(self.directory()) / 'monitor.json', data)
                            return
            atomic_json(Path(self.directory()) / 'monitor.json', data)

    def _touch_alerts(self, plan, cfg, now, live, broadcast):
        """触价通道：检测先行落盘去重，推送交独立队列worker，逐票隔离。"""
        try:
            import touch_alert_lane_v1 as lane
            state_file = Path(self.directory()) / 'touch_alerts.json'
            events = lane.detect(plan, live, state_file,
                                 now_stamp=int(now.strftime('%Y%m%d%H%M%S')))
        except Exception as exc:
            self.health.update(status='TOUCH_LANE_ERROR', message=f'触价通道异常:{exc}')
            return
        push_enabled = bool((cfg.get('push') or {}).get('enabled'))
        for event in events:
            body = lane.message(event)
            title = f"星币侍从 · 触价提醒 {event['code']}"
            broadcast(dict(type='push', kind='touch_alert', title=title, body=body,
                           wechat=False, time=now.strftime('%H:%M:%S')))
            if push_enabled:
                # 开关关闭时不预约外部发送；已展示未发送的事件留在
                # pending_sends 里，重开后再交付。
                lane.reserve_send(event, state_file)
                _touch_send_queue().put((cfg, title, body, state_file, event, plan))
        if push_enabled:
            for event in lane.pending_sends(plan, state_file,
                                            int(now.strftime('%Y%m%d%H%M%S')), live):
                body = lane.message(event)
                lane.reserve_send(event, state_file)
                _touch_send_queue().put(
                    (cfg, f"星币侍从 · 触价提醒 {event['code']}",
                     body, state_file, event, plan))

    def _tick_golden(self, plan, cfg, now, broadcast, send, clock, quotes=None):
        from golden_rules import VERSION
        if plan.get('rule_version') == 'TAROT_A_SECOND_CASH_V2':
            return self._tick_tarot_v2(plan, cfg, now, broadcast, send, clock, quotes)
        today = int(now.strftime('%Y%m%d'))
        with self.lock:
            data = self._load()
            self.health = dict(status='WAITING', rows={}, checked_at=now.isoformat(), message='黄金规则等待有效交易时段')
            if not data['enabled'] or not plan.get('plan_ready') or plan.get('rule_version') != VERSION:
                return
            if not plan.get('assistant_plan_id'):
                self.health.update(status='WAIT_HUMAN_CONFIRMATION', message='新规则名单尚未人工确认，不启动触发提醒')
                return
            if today != int(plan['monitor_date']) or not (session(now) or (now.weekday()<5 and now.hour==15 and now.minute==0)):
                return
        import golden_monitor
        result = golden_monitor.evaluate(plan, now)
        with self.lock:
            data = self._load()  # The off switch can change while the worker is running.
            actual = clock() if clock else now
            if not data['enabled'] or int(actual.strftime('%Y%m%d')) != today:
                return
            current_file = Path(self.directory())/'assistant_plan.json'
            if current_file.exists():
                from golden_rules import digest
                if digest(json.loads(current_file.read_bytes())) != digest(plan):
                    self.health.update(status='PLAN_CHANGED', message='确认计划已变化，丢弃旧工作进程结果')
                    return
            if not (session(actual) or (actual.hour==15 and actual.minute==0)):
                return
            self.health.update(status='GOLDEN_MONITORING', message='完成5分K线核验下杀、止跌、背离；不自动买入', rows=result.get('rows',{}))
            for row in plan['rows']:
                code = row['code']
                view = self.health['rows'].get(code, {})
                if view.get('status') != 'RULE_TRIGGERED':
                    continue
                rule = row['entry_rule']
                key = f'{today}:{code}:{VERSION}:{plan.get("assistant_plan_id", "snapshot")}'
                alert = data['alerts'].get(key)
                if alert is None:
                    title = f'星币侍从 · {code} 黄金规则观察条件成立'
                    body = (f'{row.get("name", "")} {code}｜前日F618 {rule["f618"]:.4f}\n'
                            f'完成5分K线：下杀到位、止跌、有效RSI背离。时刻 {view["trigger_time"]}\n'
                            '请结合Pyxis与板块/领涨股重新人工复核；不是买入建议，不自动下单。')
                    alert = dict(title=title, body=body, attempts=0, wechat=False, delivery='PENDING', code=code)
                    data['alerts'][key] = alert
                    try:
                        import touch_alert_lane_v1 as lane
                        _node=(self.health.get('rows',{}).get(code,{}) or {}).get('node')
                        _bid=lane.band_id_for_price(plan,code,float(_node)) if isinstance(_node,(int,float)) else None
                        lane.mark_notified(plan, code, Path(self.directory())/'touch_alerts.json', band_id=_bid)
                    except Exception:
                        pass  # 共享去重失败不阻塞已确认信号通知
                    atomic_json(Path(self.directory())/'monitor.json', data)
                    broadcast(dict(type='push',kind='pentacles_golden',title=title,body=body,wechat=False,time=actual.strftime('%H:%M:%S')))
                if cfg.get('push',{}).get('enabled') and not alert['wechat'] and alert['attempts'] < 3:
                    alert['attempts'] += 1
                    atomic_json(Path(self.directory())/'monitor.json', data)
                    self.lock.release()
                    try:
                        ok, _ = send(cfg,alert['title'],alert['body'])
                    finally:
                        self.lock.acquire()
                    alert.update(wechat=bool(ok),delivery='SENT' if ok else 'FAILED')
                view['delivery'] = alert['delivery']
            latest = self._load()
            data['enabled'] = latest['enabled']
            atomic_json(Path(self.directory())/'monitor.json', data)

    def _tick_tarot_v2(self, plan, cfg, now, broadcast, send, clock, quotes=None):
        import tarot_monitor_v2 as monitor
        with self.lock:
            self.health = dict(status='WAITING', rows={}, message='等待确认计划和有效交易时段')
            if not self._load()['enabled'] or not plan.get('plan_ready'):
                return
            if not plan.get('assistant_plan_id'):
                self.health.update(status='WAIT_HUMAN_CONFIRMATION')
                return
            if int(now.strftime('%Y%m%d')) < int(plan['monitor_date']) or not risk_session(now):
                return
        from sector_watch_v1 import execution_plan,matches
        plan=execution_plan(plan,now)
        try:
            result = monitor.evaluate(plan, cfg, now)
        except Exception as exc:
            result = dict(status='MONITOR_BLOCKED' if 'BUDGET' in str(exc) else 'EVIDENCE_FAILED',
                rows={},reason='分钟或风险核验失败：'+type(exc).__name__+':'+str(exc)[:180])
        live = {}
        if quotes and session(now) and int(now.strftime('%Y%m%d')) == int(plan['monitor_date']):
            try:
                live = quotes([r['code'] for r in plan.get('rows', [])]) or {}
            except Exception:
                pass  # Quote failure must not suppress rule/risk/AI notifications.
        actual = clock() if clock else now
        from risk_outlook_v1 import patrol_bulletin, with_patrol_quotes
        bulletin = patrol_bulletin(plan, actual) if risk_session(actual) and actual.date() == now.date() else {}
        patrol_positions = []
        if bulletin and not patrol_recorded(self._load()['alerts'], bulletin):
            from services import active_positions
            patrol_positions = active_positions(dict(cfg, agent=cfg.get('agent') or {}))
            if quotes and session(actual):
                holding_codes = sorted({p['code'] for p in patrol_positions} - set(live))
                if holding_codes:
                    try:
                        live.update(quotes(holding_codes) or {})
                    except Exception:
                        pass
            bulletin = with_patrol_quotes(bulletin, actual)
            actual = clock() if clock else actual
        with self.lock:
            data = self._load()
            from golden_rules import digest
            file = Path(self.directory())/'assistant_plan.json'
            if (not data['enabled'] or not risk_session(actual) or actual.date() != now.date()
                    or not file.exists() or not matches(plan,json.loads(file.read_bytes()))):
                return
            self.health.update(status=result['status'], rows=result.get('rows', {}),
                               holdings=result.get('holdings', {}), message=result.get('reason', '模型与规则监控；最终由人工决定'))
            self.health.update(board_watch_status=plan.get('board_watch_status','SCANNING_OR_NO_FORMAL_CANDIDATES'),
                               board_watch_count=plan.get('board_watch_count',0),board_watch_warnings=plan.get('board_watch_warnings',[]))
            observed=result.get('event_cutoff') or result.get('cutoff')
            current = observed and monitor.fresh(observed, actual)
            reviews = observation_reviews(plan, live, actual, data['alerts'],result.get('rows') if current else None)
            # Keep an arrival unconsumed until its first usable per-vote result.
            # Source trouble is health state, not a repetitive all-unknown card.
            v4={r['code'] for r in plan.get('rows',[]) if (r.get('entry_rule') or {}).get('policy')=='GOLDEN_RESONANCE_CROSSDAY_V4'}
            reviews=[r for r in reviews if r['code'] not in v4 or (current and
                result.get('rows',{}).get(r['code'],{}).get('status') not in ('DATA_INSUFFICIENT','EVENT_FAILED') and
                result.get('rows',{}).get(r['code'],{}).get('combination',{}).get('votes'))]
            passed_touch_ids={}
            if current:
                # A simultaneous accepted BUY replaces the waiting wording, not the touch audit IDs.
                for notice in reviews[:]:
                    row=result.get('rows',{}).get(notice['code'],{});advice=row.get('advice') or {}
                    ready=advice.get('status')=='ADVICE_READY' and advice.get('decision') in ('BUY','DEFER') and monitor.fresh(advice.get('cutoff',0),actual)
                    pending=advice.get('status') in (None,'WAIT_RISK','AI_ATTEMPT_RESERVED','AI_FAILED','AI_LEGACY_UNVERIFIED','ADVICE_EXPIRED','ENTRY_SESSION_CLOSED','REVIEW_ALREADY_RECORDED')
                    if combination_passed(row) and ready:
                        passed_touch_ids[notice['code']]=notice['covered_event_ids']
                        if ready:row['advice']=dict(advice,covered_event_ids=notice['covered_event_ids'])
                        reviews.remove(notice)
                    elif row.get('status')=='RULE_TRIGGERED' and advice.get('status')=='ADVICE_READY' and advice.get('decision')=='BUY' and monitor.fresh(advice.get('cutoff',0),actual):
                        row['advice']=dict(advice,covered_event_ids=notice['covered_event_ids'])
                        reviews.remove(notice)
            if current:
                reviews += [v.get('advice', {}) for v in result.get('rows', {}).values()]
                from services import active_positions
                live_holding_codes = {p['code'] for p in active_positions(dict(cfg, agent=cfg.get('agent') or {}))}
                reviews += [r for r in result.get('holdings', {}).values()
                            if r.get('code') in live_holding_codes]
            if bulletin and not patrol_recorded(data['alerts'], bulletin):
                reviews.append(dict(status='ADVICE_READY', decision='REPORT', code='市场巡检',
                                    cutoff=bulletin['asof'], reason='', bulletin_key=bulletin['key'],
                                    event_id='patrol:'+bulletin['key']))
            for code,row in result.get('rows',{}).items():
                advice=row.get('advice') or {}
                waiting={'WAIT_RISK':'同截止风险证据尚不齐，继续等待，不视为安全。',
                    'AI_ATTEMPT_RESERVED':'风险与AI综合意见正在核验。',
                    'AI_FAILED':'AI分析失败，本地事实保留；未自动付费重试。',
                    'AI_LEGACY_UNVERIFIED':'旧解释未通过当前事实校验，不能作为有效建议。',
                    'ADVICE_EXPIRED':'分析返回时已过期，本次不提供有效交易建议。',
                    'ENTRY_SESSION_CLOSED':'已结束当日入场时段，不再建议买入。',
                    'RULE_PASSED':'综合研判结果待更新，继续观察。'}
                passed=combination_passed(row)
                state=advice.get('status')
                if passed and state in (None,'REVIEW_ALREADY_RECORDED'):state='RULE_PASSED'
                if current and row.get('status')=='RULE_TRIGGERED' and state in waiting and not row.get('provisional'):
                    phase='RULE_PASSED' if passed and state in ('WAIT_RISK','AI_ATTEMPT_RESERVED','RULE_PASSED') else state
                    reviews.append(dict(status='ADVICE_READY',waiting_status=state,decision='OBSERVE',code=code,cutoff=observed,
                        covered_event_ids=passed_touch_ids.get(code,[]),
                        event_id=f'{digest(plan)}:{code}:{row.get("combination_policy","")}:{row.get("trigger_time")}:{row.get("node")}:{row.get("branch")}:{phase}',
                        action=combination_name(row)+'已通过' if passed else '黄金位确认，风险核验中' if state in ('WAIT_RISK','AI_ATTEMPT_RESERVED') else '到位后暂不建议买入，查看原因',
                        reason=(combination_name(row)+'已通过；' if passed else '黄金位置价格条件已确认；')+(rule_condition_text(row) if passed else '')+waiting[state]+str(advice.get('reason') or '')+' 请人工判断。'))
            reviews.sort(key=lambda r:0 if r.get('kind')=='HOLDING' and r.get('decision')=='EXIT' else 1 if r.get('kind')=='HOLDING' else 2 if r.get('decision')!='REPORT' else 3)
            for review in reviews:
                if review.get('kind') == 'HOLDING':
                    from services import active_positions
                    if not any(p['code'] == review.get('code') for p in active_positions(dict(cfg, agent=cfg.get('agent') or {}))):
                        continue
                if review.get('status') == 'T1_WAIT':
                    review=dict(review,status='ADVICE_READY',decision='DEFER')
                if review.get('status') != 'ADVICE_READY' or review.get('decision') not in ('BUY', 'HOLD', 'EXIT','DEFER','REPORT','OBSERVE') or not monitor.fresh(review.get('cutoff',0),actual):
                    continue
                if review.get('decision')=='BUY' and result.get('rows',{}).get(review.get('code'),{}).get('status')!='RULE_TRIGGERED':
                    continue
                key = 'tarot-v2:' + (review.get('event_id') or digest(review))
                if review.get('provisional') and review.get('decision') in ('BUY','DEFER'):
                    from live_plan_lane_v1 import notice_key
                    key='tarot-v2:first-live-signal:'+notice_key(plan,review['code'],result.get('rows',{}).get(review['code'],{}))
                if key in data['alerts']:
                    continue
                code = review['code']
                passed=combination_passed(result.get('rows',{}).get(code,{}))
                action = review.get('action') or {'BUY':'建议买入','HOLD':'建议继续观察持仓','EXIT':'建议减仓或退出','DEFER':'建议观察并等待','REPORT':'半小时风险巡检','OBSERVE':'黄金位确认，风险核验中'}[review['decision']]
                if passed and review['decision']=='DEFER':action=combination_name(result.get('rows',{}).get(code,{}))+'已通过·DeepSeek暂缓'
                title = f'Stargazer · {code} {action}'
                board_names=(plan.get('board_origins') or {}).get(code)
                if board_names:title=f'Stargazer · 板块发现 {code} {action}'
                risk=review.get('risk') or {};context=risk.get('context') or {}
                if review['decision']=='BUY':
                    body=(f'{code} '+rule_condition_text(result['rows'][code])+combination_name(result['rows'][code])+'已通过。DeepSeek建议买入。' if passed else f"{code} 建议买入。")
                elif review['decision']=='OBSERVE':
                    body=review['reason']
                elif review['decision']=='DEFER' and code in result.get('rows',{}):
                    rationale=('；'.join((review.get('risk_reasons') or [])[:3]) or review.get('summary_reason') or review.get('reason') or '综合风险意见待核对').splitlines()[0]
                    body=(f'{code} '+rule_condition_text(result['rows'][code])+combination_name(result['rows'][code])+f'已通过；DeepSeek暂不建议买入。\n风险/顾虑：{rationale[:240]}\n继续观察，请人工决定。' if passed else
                        f"尚未通过：{rationale[:160]}。建议观察并等待。")
                else:
                    body=f"观察截止 {review['cutoff']}；{review['reason']}\n请人工决定；程序不下单。"
                if review['decision'] in ('REPORT','HOLD','EXIT') and all(isinstance(context.get(k),(int,float)) for k in ['amp_232','amp_266']):
                    body+=f"\nPyxis幅度证据截止 {risk.get('asof_ts',review['cutoff'])}：232 {context['amp_232']:.2%} / 266 {context['amp_266']:.2%}，非概率、非保证最大损失。"
                from push_presenter_v1 import report as render_report, review_notice
                if review['decision']=='REPORT':
                    # 强势池通道真实健康状态：NOT_OWNER/零报价不得写成「无异动」。
                    with self.lock:
                        strong_scan = dict(self.strong_touch_health or {})
                    patrol_signals = result if observed == bulletin['asof'] else {}
                    live_cutoff = result.get('event_cutoff')
                    if live_cutoff and live_cutoff == observed:
                        stamp = datetime.strptime(str(live_cutoff), '%Y%m%d%H%M%S')
                        slot = datetime.strptime(str(bulletin['asof']), '%Y%m%d%H%M%S')
                        if stamp.date() == slot.date() and 0 <= (actual-stamp).total_seconds() <= 120 and abs((stamp-slot).total_seconds()) <= 120:
                            patrol_signals = result
                    from services import active_positions
                    patrol_positions = active_positions(dict(cfg, agent=cfg.get('agent') or {}))
                    rendered=render_report(bulletin,plan,patrol_signals,live,patrol_positions,data.get('template_state'),observed_at=actual,
                                           strong_scan=strong_scan)
                    data['template_state']=rendered['next_state']
                else:
                    selected=next((r for r in plan.get('rows',[]) if r['code']==code),{})
                    rendered=review_notice(review,result.get('rows',{}).get(code,{}),selected,live.get(code,{}) if isinstance(live,dict) else {})
                title,body=rendered['title'],rendered['body']
                # Persist delivery intent first, including failed delivery; no automatic paid retry.
                alert = dict(title=title, body=body, review=review, delivery='RECORDED', wechat=False, attempts=0,
                             plan_id=plan.get('assistant_plan_id'), observed_at=actual.isoformat())
                if review['decision'] == 'REPORT':
                    alert['patrol_diagnostics'] = {k: bulletin[k] for k in (
                        'status', 'reason', 'validation_error', 'computed_facts', 'analysis', 'text'
                    ) if k in bulletin}
                    alert['patrol_diagnostics']['stock_risk_status'] = result['status']
                    if result.get('reason'):
                        alert['patrol_diagnostics']['stock_risk_reason'] = result['reason']
                data['alerts'][key] = alert
                # 共享触价身份：只消费实际到达的带，首份综合建议是独立身份，
                # 不被触价去重吞掉，更低档保持武装。
                try:
                    import touch_alert_lane_v1 as lane
                    signal = result.get('rows', {}).get(code, {}) or {}
                    node = signal.get('node') or signal.get('next_lower_node')
                    band_id = lane.band_id_for_price(plan, code, float(node)) if isinstance(node, (int, float)) else None
                    lane.mark_notified(plan, code, Path(self.directory())/'touch_alerts.json', band_id=band_id)
                except Exception:
                    pass
                for covered in review.get('covered_event_ids', []):
                    if 'tarot-v2:'+covered != key:
                        data['alerts']['tarot-v2:'+covered] = dict(covered_by=key)
                atomic_json(Path(self.directory())/'monitor.json', data)
                if cfg.get('push', {}).get('enabled'):
                    alert['attempts'] = 1
                    atomic_json(Path(self.directory())/'monitor.json', data)
                    self.lock.release()
                    try:
                        ok, _ = send(cfg, title, body)
                    except Exception as exc:
                        ok = False
                        alert['delivery_error_type'] = type(exc).__name__
                    finally:
                        self.lock.acquire()
                    alert.update(wechat=bool(ok), delivery='SENT' if ok else 'FAILED',
                                 wechat_msg='推送服务已接受' if ok else '推送服务未确认成功')
                    data['enabled'] = self._load()['enabled']
                else:
                    alert.update(delivery='DISABLED', wechat_msg='微信推送已关闭')
                atomic_json(Path(self.directory())/'monitor.json', data)
                broadcast(dict(type='push', kind='tarot_v2', title=title, body=body,
                               wechat=alert['wechat'], wechat_msg=alert['wechat_msg'],
                               delivery=alert['delivery'], time=actual.strftime('%H:%M:%S')))
                if not data['enabled']:
                    break

    def _start_quote_lane(self, get_cfg, get_plan):
        """独立报价通道：10-15秒批量报价，只做触价检测与发送入队，
        不等待指标或AI；行情读取带超时，慢请求不拖住后续轮询。"""
        import random
        def lane():
            while not self.stop.is_set():
                try:
                    self._quote_lane_pass(get_cfg, get_plan)
                except Exception:
                    pass  # 通道异常不得杀死轮询
                self.stop.wait(random.randint(QUOTE_LANE_MIN_SECONDS, QUOTE_LANE_MAX_SECONDS))
        self.quote_thread = threading.Thread(target=lane, daemon=True,
                                             name='pentacles-quote-lane')
        self.quote_thread.start()

    def _quote_lane_pass(self, get_cfg, get_plan, quote_fn=None, now=None, broadcast=None):
        """One quote-lane poll: gates → quotes(前拦截额度闸+超时) → touch detect.
        Independent of the indicator channel: an indicator stall never blocks it."""
        counter_file = Path(self.directory()) / 'quote_lane_budget.json'
        now = now or datetime.now()
        plan = get_plan() or {}
        if _quote_lane_ok(self, plan, now) and plan.get('rows'):
            before = _budget_count(counter_file)
            codes = [row['code'] for row in plan['rows']]
            if quote_fn is not None:
                live = quote_fn(codes) or {}
            else:
                live = _quotes_with_timeout(codes, budget_file=counter_file) or {}
            if _budget_count(counter_file) == before and not live:
                pass  # 额度拦截或超时：本轮无数据，静默等下一轮
            if broadcast is None:
                from notifier import broadcast as broadcast_fn
            else:
                broadcast_fn = broadcast
            self._touch_alerts(plan, get_cfg(), now, live, broadcast_fn)

    def _start_strong_touch_lane(self, get_cfg, get_plan):
        """Strong-pool quote discovery; independent from the plan-ticket lane."""
        if self.strong_quote_thread and self.strong_quote_thread.is_alive():
            return

        def run():
            from tools.market import quotes as market_quotes
            from notifier import broadcast, send_wechat
            from strong_sector_watch_v1 import (
                root as strong_root, deliver_touch_event, compute_touch_indicators,
                deliver_indicator_supplement)
            from strong_touch_lane_v1 import (
                poll, start_workers, restore_pending_events, enqueue_pending,
                POLL_INTERVAL_SECONDS, QUOTE_TIMEOUT_SECONDS,
                QUOTE_BATCH_SIZE, QUOTE_WORKERS)
            usage_file = Path(self.directory()) / 'quote_request_usage.json'
            state_dir = strong_root() / 'touch'

            def attempt_hook(lane_name):
                def record(_url):
                    try:
                        from quote_budget_v1 import record_attempt
                        record_attempt(usage_file, lane_name)
                    except Exception:
                        pass
                    return True
                return record

            def fetch_strong(codes):
                return market_quotes(codes, on_attempt=attempt_hook('strong_pool'),
                    timeout=QUOTE_TIMEOUT_SECONDS, batch_size=QUOTE_BATCH_SIZE,
                    max_workers=QUOTE_WORKERS, tries=1)

            def fetch_reverify(codes):
                return market_quotes(codes, on_attempt=attempt_hook('strong_pool_reverify'),
                    timeout=QUOTE_TIMEOUT_SECONDS, batch_size=1, max_workers=1, tries=1)

            def enabled():
                try:
                    return self._load().get('enabled') is True
                except Exception:
                    return False

            def strong_plan():
                from strong_sector_pool_prep_v1 import manual_pool
                day = int(datetime.now().strftime('%Y%m%d'))
                if manual_pool(day) is not None:
                    return dict(monitor_date=day, manual_strong_pool=True)
                return get_plan() or {}

            def deliver(event_id, file, plan_ref):
                deliver_touch_event(event_id, file, plan_ref, get_plan=strong_plan,
                    get_cfg=get_cfg, quote_fn=fetch_reverify, broadcast=broadcast,
                    send=send_wechat, active=enabled)
                enqueue_pending(file, plan_ref)

            def indicators(event_id, file, plan_ref):
                compute_touch_indicators(event_id, file, plan_ref)
                enqueue_pending(file, plan_ref)

            def supplement(event_id, file, plan_ref):
                deliver_indicator_supplement(event_id, file, plan_ref, get_plan=strong_plan,
                    get_cfg=get_cfg, quote_fn=fetch_reverify, broadcast=broadcast,
                    send=send_wechat, active=enabled)

            start_workers(deliver, indicators, self.stop, supplement_handler=supplement)
            restored = None
            while not self.stop.is_set():
                now = datetime.now()
                try:
                    plan = strong_plan()
                    manual = plan.get('manual_strong_pool') is True
                    if (enabled() and session(now) if manual else _quote_lane_ok(self, plan, now)):
                        if restored is None:
                            try:
                                restored = restore_pending_events(state_dir, plan)
                            except Exception as exc:
                                restored = dict(error=type(exc).__name__ + ':' + str(exc)[:160])
                        result = poll(plan, now, state_dir=state_dir,
                                      quote_fn=fetch_strong, usage_file=usage_file)
                        result['restore_on_start'] = restored
                        with self.lock:
                            self.strong_touch_health = dict(result, checked_at=datetime.now().isoformat(timespec='seconds'))
                    else:
                        try:
                            enabled_now = self._load().get('enabled') is True
                        except Exception:
                            enabled_now = False
                        if not enabled_now:
                            status = 'DISABLED'
                        elif not plan or not plan.get('plan_ready'):
                            status = 'PLAN_NOT_READY'
                        elif int(now.strftime('%Y%m%d')) != int(plan.get('monitor_date') or 0):
                            status = 'WAITING_FOR_MONITOR_DATE'
                        elif not session(now):
                            status = 'OUTSIDE_SESSION'
                        else:
                            status = 'PLAN_NOT_CONFIRMED'
                        with self.lock:
                            self.strong_touch_health = dict(status=status,
                                checked_at=now.isoformat(timespec='seconds'),
                                max_scan_interval_seconds=120)
                except Exception as exc:
                    with self.lock:
                        self.strong_touch_health = dict(status='LANE_ERROR',
                            error=type(exc).__name__ + ':' + str(exc)[:160],
                            checked_at=datetime.now().isoformat(timespec='seconds'))
                self.stop.wait(POLL_INTERVAL_SECONDS)

        self.strong_quote_thread = threading.Thread(
            target=run, daemon=True, name='strong-pool-quote-lane')
        self.strong_quote_thread.start()

    def start(self, get_cfg, get_plan):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            self._start_quote_lane(get_cfg, get_plan)
            self._start_strong_touch_lane(get_cfg, get_plan)
            self._indicator_inflight = False
            def loop():
                from tools.market import quotes
                from notifier import broadcast, send_wechat
                while not self.stop.is_set():
                    try:
                        from strong_sector_watch_v1 import schedule
                        confirmed=get_plan()
                        cfg=get_cfg()
                        now=datetime.now()
                        from minute_history_v1 import schedule as prepare_minutes
                        errors = {}
                        if confirmed:
                            try:
                                prepare_minutes(confirmed)
                            except Exception as exc:
                                errors['minute_history'] = type(exc).__name__ + ':' + str(exc)[:160]
                        watched=tuple(b.get('name') for b in (confirmed or {}).get('boards',[]))
                        try:
                            schedule(confirmed,cfg,now,quotes,broadcast,send_wechat,
                                lambda expected=watched:self._load()['enabled'] and tuple(b.get('name') for b in (get_plan() or {}).get('boards',[]))==expected)
                        except Exception as exc:
                            errors['strong_pool'] = type(exc).__name__ + ':' + str(exc)[:160]
                        if getattr(self, '_indicator_inflight', False):
                            pass  # 上一轮指标计算未完成：不重复启动计算任务
                        else:
                            self._indicator_inflight = True
                            try:
                                self.tick(confirmed, cfg, datetime.now(), quotes, broadcast, send_wechat, datetime.now)
                            finally:
                                self._indicator_inflight = False
                        if errors:
                            with self.lock:
                                self.health['preparation_errors'] = errors
                    except Exception as exc:
                        with self.lock:
                            self.health.update(status='MONITOR_ERROR', message=type(exc).__name__ + '：监控失败，请检查数据/网络')
                    self.stop.wait(60)
            self.thread = threading.Thread(target=loop, daemon=True, name='pentacles-price-monitor')
            self.thread.start()
