"""Bounded daily observation diary. Quote samples are not minute bars or PnL labels."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import threading

MAX_BYTES = 2 * 1024 * 1024
WINDOWS = ('30m', '60m', 'H3', 'H5')


def day_string(value):
    text = str(value)
    if len(text) != 8 or not text.isdigit():
        raise ValueError('日期必须为 YYYYMMDD')
    datetime.strptime(text, '%Y%m%d')
    return text


def encoded(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False).encode('utf-8')


def clean(value):
    return str(value).replace('|', '／').replace('\n', ' ').replace('\r', ' ')


class ObservationJournal:
    def __init__(self, directory, budget=MAX_BYTES):
        self.directory = directory
        self.budget = budget
        self.lock = threading.RLock()
        self.error = None

    def _root(self):
        return Path(self.directory())

    def _bytes(self):
        return sum(p.stat().st_size for p in self._root().rglob('*') if p.is_file())

    def _write(self, changes):
        # Reserve actual bytes for all replacements AND one atomic temporary file.
        added = sum(len(raw) - (p.stat().st_size if p.exists() else 0) for p, raw in changes)
        peak = max(self._bytes(), self._bytes() + added) + max((len(raw) for _, raw in changes), default=0)
        if peak > self.budget:
            raise ValueError('观察日记达到 2MiB 硬预算；停止记录，需另批额度，不删除旧记录')
        for file, raw in changes:
            if file.exists() and file.read_bytes() == raw:
                continue
            file.parent.mkdir(parents=True, exist_ok=True)
            temp = file.with_suffix('.tmp')
            temp.write_bytes(raw)
            temp.replace(file)

    def _load(self, day):
        file = self._root() / 'days' / f'{day}.json'
        if file.exists():
            data = json.loads(file.read_text(encoding='utf-8'))
            if data.get('schema') != 'PENTACLES_OBSERVATION_DAY_V1':
                raise ValueError('日记版本不兼容，不覆盖旧件')
            return data
        return {'schema': 'PENTACLES_OBSERVATION_DAY_V1', 'day': day,
                'plans': [], 'candidates': {}, 'attempts': {}, 'last_recorded_at': None}

    def _register(self, plan):
        day = day_string(plan['monitor_date'])
        payload = dict(plan)
        payload.pop('created_at', None)  # identical reruns do not invent a new candidate population.
        plan_id = hashlib.sha256(encoded(payload)).hexdigest()
        file = self._root() / 'plans' / f'{plan_id}.json'
        data = self._load(day)
        if file.exists() and hashlib.sha256(file.read_bytes()).hexdigest() != plan_id:
            raise ValueError('不可变名单哈希不符，停止记录，不覆盖原件')
        if plan_id in data['plans'] and not file.exists():
            raise ValueError('已登记名单原件缺失，停止记录')
        if plan_id not in data['plans']:
            data['plans'].append(plan_id)
            for row in plan['rows']:
                key = f"{plan_id}:{row['code']}"
                data['candidates'][key] = {
                    'plan_id': plan_id, 'code': row['code'], 'name': row.get('name', ''),
                    'setup': row['setup'], 'zone': row.get('zone'),
                    'reference_price': row.get('reference_price'), 'samples': 0,
                    'first_quote': None, 'last_quote': None, 'sample_min': None, 'sample_max': None,
                    'first_observed_touch': None, 'quality_counts': {},
                    'after_touch_sample_min': None, 'after_touch_sample_max': None,
                    'after_touch_last_quote': None, 'after_touch_sample_return': None,
                    'last_status': 'NOT_OBSERVED' if row.get('zone') else 'NO_ZONE',
                    'exact_labels': {window: {'status': 'REQUIRES_VERIFIED_MINUTE_PATH', 'value': None}
                                     for window in WINDOWS}}
            changes = [(self._root() / 'days' / f'{day}.json', encoded(data)),
                       (self._root() / 'days' / f'{day}.md', self._render(data).encode('utf-8'))]
            if not file.exists():
                changes.insert(0, (file, encoded(payload)))
            self._write(changes)
        return day, plan_id, data

    def register(self, plan):
        if not plan or not plan.get('monitor_date'):
            return
        with self.lock:
            try:
                self._register(plan)
                self.error = None
            except Exception as exc:
                self.error = str(exc)

    def record(self, plan, health, now):
        if not plan or not plan.get('monitor_date'):
            return
        with self.lock:
            try:
                day, plan_id, data = self._register(plan)
                if now.strftime('%Y%m%d') != day:
                    return
                status = health.get('status', 'UNKNOWN')
                data['attempts'][status] = data['attempts'].get(status, 0) + 1
                data['last_recorded_at'] = now.isoformat(timespec='seconds')
                views = health.get('rows', {})
                for row in plan['rows']:
                    record = data['candidates'][f"{plan_id}:{row['code']}"]
                    view = views.get(row['code'])
                    if not view:
                        continue
                    record['last_status'] = view['status']
                    price, stamp = view.get('price'), view.get('time')
                    if view['status'] not in ('WAIT_PRICE', 'ALERT_RECORDED') or price is None:
                        counts = record['quality_counts']
                        counts[view['status']] = counts.get(view['status'], 0) + 1
                        continue
                    if record['last_quote'] and str(stamp) <= str(record['last_quote']['time']):
                        if str(stamp) < str(record['last_quote']['time']):
                            counts = record['quality_counts']
                            counts['OUT_OF_ORDER_QUOTE'] = counts.get('OUT_OF_ORDER_QUOTE', 0) + 1
                        continue
                    quote = {'time': stamp, 'price': price}
                    record['samples'] += 1
                    record['first_quote'] = record['first_quote'] or quote
                    record['last_quote'] = quote
                    record['sample_min'] = min(price, record['sample_min']) if record['sample_min'] is not None else price
                    record['sample_max'] = max(price, record['sample_max']) if record['sample_max'] is not None else price
                    zone = record['zone']
                    if zone and price <= zone['zone_upper'] and not record['first_observed_touch']:
                        record['first_observed_touch'] = dict(quote, kind='BELOW_ZONE' if price < zone['zone_lower'] else 'IN_ZONE')
                    touch = record['first_observed_touch']
                    if touch:
                        low, high = record['after_touch_sample_min'], record['after_touch_sample_max']
                        record['after_touch_sample_min'] = min(low, price) if low is not None else price
                        record['after_touch_sample_max'] = max(high, price) if high is not None else price
                        record['after_touch_last_quote'] = quote
                        record['after_touch_sample_return'] = price / touch['price'] - 1
                self._write([(self._root() / 'days' / f'{day}.json', encoded(data)),
                             (self._root() / 'days' / f'{day}.md', self._render(data).encode('utf-8'))])
                self.error = None
            except Exception as exc:
                self.error = str(exc)

    def _render(self, data):
        lines = [f"# 星币侍从观察日记 · {data['day']}", '',
                 '这是报价采样日记，不是买卖建议、完整分钟走势或策略收益报告。',
                 '仅记录程序运行期间的观测；首次观测到价≠真实首次触价，采样极值≠全天高低点。',
                 '没有观察到触价不能判为全天未触价。30m/60m/H3/H5 精确标签暂无可靠路径，保持 NULL。', '',
                 f"名单版本：{len(data['plans'])}；候选记录：{len(data['candidates'])}（同股不同版本分列）。",
                 f"最后记录：{data['last_recorded_at'] or '尚未开始当日观察'}；监控状态次数：{data['attempts']}", '',
                 '| 股票 | 形态 | 关注区域 | 首次观测到价/破位 | 首报价 → 末报价 | 采样低/高 | 到价后采样低/高/末变动 | 样本数 | 状态/质量 |',
                 '|---|---|---|---|---|---|---|---|---|']
        def price(value):
            return '—' if value is None else f'{value:.2f}'
        for row in data['candidates'].values():
            zone, touch = row['zone'], row['first_observed_touch']
            first, last = row['first_quote'], row['last_quote']
            cells = [f"{row['code']} {row['name']}", row['setup'],
                     f"{price(zone['zone_lower'])}–{price(zone['zone_upper'])}" if zone else '无区域',
                     f"{touch['time']} {price(touch['price'])} {touch['kind']}" if touch else '未观测到／不可判断',
                     f"{first['time']} {price(first['price'])} → {last['time']} {price(last['price'])}" if first else '未取得有效报价',
                     f"{price(row['sample_min'])}/{price(row['sample_max'])}",
                     f"{price(row['after_touch_sample_min'])}/{price(row['after_touch_sample_max'])}/"
                     + ('—' if row['after_touch_sample_return'] is None else f"{row['after_touch_sample_return']:+.2%}"), row['samples'],
                     f"{row['last_status']} {row['quality_counts']}"]
            lines.append('| ' + ' | '.join(clean(cell) for cell in cells) + ' |')
        lines.extend(['', '## 版本与复核', *[f'- 不可变名单 `plans/{p}.json`（保留完整候选、区域、输入哈希）。' for p in data['plans']],
                      '- 关闭、缺区域、错误报价、断流不删除候选；日记缺失不是负样本。',
                      '- 到价后采样变动不是买入收益：没有成交、手续费、滑点或持仓路径含义。',
                      '- 软件未运行期间无心跳记录，不能推断该时段走势。训练须另过分钟源/标签/样本门。'])
        return '\n'.join(lines) + '\n'

    def state(self):
        with self.lock:
            try:
                days = sorted((p.stem for p in (self._root() / 'days').glob('*.json')), reverse=True)
                return {'days': days, 'bytes': self._bytes(), 'budget_bytes': self.budget,
                        'error': self.error, 'directory': str(self._root())}
            except OSError as exc:
                return {'days': [], 'bytes': None, 'budget_bytes': self.budget,
                        'error': str(exc), 'directory': str(self._root())}

    def report(self, day=None):
        with self.lock:
            day = day_string(day) if day is not None else None
            state = self.state()
            if not state['days']:
                return dict(state, day=None, markdown='尚未归档观察名单。下一次分析或启动监控线程后自动归档。')
            day = day_string(day or state['days'][0])
            if day not in state['days']:
                raise ValueError('该日记不存在')
            data = self._load(day)
            return dict(state, day=day, markdown=self._render(data))
