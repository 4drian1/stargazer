"""Independent quote reminders. No minute-bar execution, orders or buy advice."""
from datetime import datetime
import json
import math
from pathlib import Path
import threading


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


class PriceMonitor:
    def __init__(self, directory):
        self.directory = directory
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.health = {'status': 'NOT_STARTED', 'rows': {}, 'message': '到价提醒尚未启动'}

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
