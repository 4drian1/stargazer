"""Independent quote reminders. No minute-bar execution, orders or buy advice."""
from datetime import datetime
import json
import math
from pathlib import Path
import threading
from pentacles_journal import ObservationJournal


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


def observation_reviews(plan, quotes, now, alerts):
    """Quote touches are notifications, never execution confirmations."""
    if int(plan['monitor_date']) != int(now.strftime('%Y%m%d')) or not session(now):
        return []
    from golden_rules import digest
    reviews = []
    for row in plan.get('rows', []):
        levels = (row.get('entry_rule') or {}).get('levels') or []
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
        prefix = f"touch:{digest(plan)}:{now:%Y%m%d}:{row['code']}:"
        reached = [v for v in levels if low <= v and 'tarot-v2:'+prefix+str(v) not in alerts]
        if not reached:
            continue
        reviews.append(dict(status='ADVICE_READY', decision='OBSERVE', code=row['code'],
            cutoff=int(quote['time']), event_id=prefix+str(min(reached)),
            covered_event_ids=[prefix+str(v) for v in reached], action='观察位已到达，请查看原因',
            reason=f"截至本次报价，日内最低 {low:.4f}，已到达或跌破计划观察档 "+
                '、'.join(f'{v:.4f}' for v in reached)+f"；当前报价 {price:.4f}。\n"
                '这是价格到位提醒，尚不代表止跌或买入条件成立；等待完整5分钟K线、风险及AI意见，结果另行推送。'))
    return reviews


class PriceMonitor:
    def __init__(self, directory):
        self.directory = directory
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
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
                return dict(self.health, enabled=data['enabled'], alerts=data['alerts'])
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
        result = monitor.evaluate(plan, cfg, now)
        live = {}
        if quotes and session(now) and int(now.strftime('%Y%m%d')) == int(plan['monitor_date']):
            try:
                live = quotes([r['code'] for r in plan.get('rows', [])]) or {}
            except Exception:
                pass  # Quote failure must not suppress rule/risk/AI notifications.
        actual = clock() if clock else now
        with self.lock:
            data = self._load()
            from golden_rules import digest
            file = Path(self.directory())/'assistant_plan.json'
            if (not data['enabled'] or not risk_session(actual) or actual.date() != now.date()
                    or not file.exists() or digest(json.loads(file.read_bytes())) != digest(plan)):
                return
            self.health.update(status=result['status'], rows=result.get('rows', {}),
                               holdings=result.get('holdings', {}), message=result.get('reason', '模型与规则监控；最终由人工决定'))
            observed=result.get('event_cutoff') or result.get('cutoff')
            current = observed and monitor.fresh(observed, actual)
            reviews = observation_reviews(plan, live, actual, data['alerts'])
            if current:
                reviews += [v.get('advice', {}) for v in result.get('rows', {}).values()]
                reviews += list(result.get('holdings', {}).values())
            from risk_outlook_v1 import latest
            bulletin=latest('INTRADAY')
            if bulletin.get('status') in ('READY','AI_FAILED','DATA_WAIT') and bulletin.get('plan_id')==plan.get('assistant_plan_id') and bulletin.get('plan_hash') in (None,digest(plan)) and monitor.fresh(bulletin['asof'],actual):
                reason=bulletin.get('text') or '本次AI研判失败；本地事实保留，未自动重试：'+str(bulletin.get('reason'))
                reviews.append(dict(status='ADVICE_READY',decision='REPORT',code='市场巡检',cutoff=bulletin['asof'],reason=reason,bulletin_key=bulletin['key']))
            for code,row in result.get('rows',{}).items():
                advice=row.get('advice') or {}
                waiting={'WAIT_RISK':'同截止风险证据尚不齐，继续等待，不视为安全。',
                    'AI_ATTEMPT_RESERVED':'风险与AI综合意见正在核验。',
                    'AI_FAILED':'AI分析失败，本地事实保留；未自动付费重试。',
                    'AI_LEGACY_UNVERIFIED':'旧解释未通过当前事实校验，不能作为有效建议。',
                    'ADVICE_EXPIRED':'分析返回时已过期，本次不提供有效交易建议。',
                    'ENTRY_SESSION_CLOSED':'已结束当日入场时段，不再建议买入。'}
                if current and row.get('status')=='RULE_TRIGGERED' and advice.get('status') in waiting:
                    reviews.append(dict(status='ADVICE_READY',decision='OBSERVE',code=code,cutoff=observed,
                        event_id=f'{digest(plan)}:{code}:{row.get("trigger_time")}:{row.get("node")}:{row.get("branch")}:{advice["status"]}',
                        action='黄金位确认，风险核验中' if advice['status'] in ('WAIT_RISK','AI_ATTEMPT_RESERVED') else '到位后暂不建议买入，查看原因',
                        reason='黄金位置价格条件已确认；'+waiting[advice['status']]+str(advice.get('reason') or '')+' 请人工判断。'))
            for review in reviews:
                if review.get('status') == 'T1_WAIT':
                    review=dict(review,status='ADVICE_READY',decision='DEFER')
                if review.get('status') != 'ADVICE_READY' or review.get('decision') not in ('BUY', 'HOLD', 'EXIT','DEFER','REPORT','OBSERVE') or not monitor.fresh(review.get('cutoff',0),actual):
                    continue
                key = 'tarot-v2:' + (review.get('event_id') or digest(review))
                if key in data['alerts']:
                    continue
                code = review['code']
                action = review.get('action') or {'BUY':'建议考虑买入','HOLD':'建议继续观察持仓','EXIT':'建议减仓或退出','DEFER':'到位后暂缓，查看原因','REPORT':'半小时风险巡检','OBSERVE':'黄金位确认，风险核验中'}[review['decision']]
                title = f'Stargazer · {code} {action} · 待人工决定'
                risk=review.get('risk') or {};context=risk.get('context') or {}
                body=f"观察截止 {review['cutoff']}；{review['reason']}\n请人工决定；程序不下单。"
                if all(isinstance(context.get(k),(int,float)) for k in ['amp_232','amp_266']):
                    body+=f"\nPyxis幅度证据截止 {risk.get('asof_ts',review['cutoff'])}：232 {context['amp_232']:.2%} / 266 {context['amp_266']:.2%}，非概率、非保证最大损失。"
                # Persist delivery intent first, including failed delivery; no automatic paid retry.
                alert = dict(title=title, body=body, delivery='RECORDED', wechat=False, attempts=0)
                data['alerts'][key] = alert
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

    def start(self, get_cfg, get_plan):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            def loop():
                from tools.market import quotes
                from notifier import broadcast, send_wechat
                while not self.stop.is_set():
                    try:
                        self.tick(get_plan(), get_cfg(), datetime.now(), quotes, broadcast, send_wechat, datetime.now)
                    except Exception as exc:
                        with self.lock:
                            self.health.update(status='MONITOR_ERROR', message=type(exc).__name__ + '：监控失败，请检查数据/网络')
                    self.stop.wait(60)
            self.thread = threading.Thread(target=loop, daemon=True, name='pentacles-price-monitor')
            self.thread.start()
