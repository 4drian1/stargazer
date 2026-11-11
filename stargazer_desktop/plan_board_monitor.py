"""Observe only confirmed-plan boards; quote work never blocks holdings checks."""
from datetime import datetime, timezone, timedelta
import copy
import json
import math
from pathlib import Path
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import paths
import trade_plan_v1

ZONE = timezone(timedelta(hours=8))
LOCK = threading.Lock()
CACHE = {}
RUNNING = False


def resolve(names):
    try:
        catalog = json.loads(Path(paths.data_file('data/board_list.json', seed_from_bundle=False)).read_text(encoding='utf-8')).get('boards', {})
    except (OSError, ValueError):
        catalog = {}
    result = []
    for name in names:
        from ths_board_identity_v1 import resolve_for_pool
        import postclose_risk_runtime
        registered = resolve_for_pool(name, postclose_risk_runtime.directory() / 'board_identities')
        if registered is not None:
            result.append(dict(name=name, bk=None, identity=registered['identity'],
                               member_status=registered['status'], provenance=registered['provenance'],
                               source='ths', scope='THS_INDEX_NOT_EASTMONEY'))
            continue
        code = name if re.fullmatch(r'BK\d{4}', name) else catalog.get(name)
        if not code and name == '军工':
            code = catalog.get('国防军工')
        if not code and name != '军工':
            tokens = [name, *re.findall(r'[（(]([^（）()]+)[）)]', name)]
            hits = {value for label, value in catalog.items() if any(
                word.casefold() == label.casefold() or word.casefold() in label.casefold()
                for word in tokens)}
            code = next(iter(hits)) if len(hits) == 1 else None
        result.append({'name': name, 'bk': code if isinstance(code, str) and re.fullmatch(r'BK\d{4}', code) else None})
    return result


def quote(item, now):
    row = {**item, 'pct': None, 'status': 'UNRESOLVED'}
    if item.get('identity'):
        row.update(status='CONSTITUENTS_PENDING' if item.get('member_status') == 'CONSTITUENTS_PENDING'
                   else 'INDEX_QUOTE_UNAVAILABLE',
                   reason='同花顺指数报价未提供；成分及强势池口径另列，不借用东财同名指数')
        return row
    if not item['bk']:
        return row
    try:
        url = ('https://push2.eastmoney.com/api/qt/stock/get?secid=90.' + item['bk'] +
               '&fields=f43,f60,f58,f86&fltt=2')
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=4) as response:
            raw = response.read(65537)
        if len(raw) > 65536:
            raise ValueError('QUOTE_SIZE')
        data = json.loads(raw).get('data') or {}
        last, previous = float(data['f43']), float(data['f60'])
        stamp = datetime.fromtimestamp(int(data['f86']), ZONE)
        if not all(math.isfinite(v) and v > 0 for v in (last, previous)):
            raise ValueError('QUOTE_PRICE')
        if not 0 <= (now - stamp).total_seconds() <= 120:
            row.update(status='STALE', quote_ts=stamp.isoformat())
        else:
            row.update(status='OK', pct=(last / previous - 1) * 100,
                       quote_ts=stamp.isoformat(), source_name=data.get('f58'))
    except Exception:
        row['status'] = 'UNAVAILABLE'
    return row


def state(now=None):
    global RUNNING
    now = now or datetime.now(ZONE)
    plan = trade_plan_v1.load_plan()
    names = [b.get('name') for b in (plan or {}).get('boards', []) if isinstance(b, dict) and b.get('name')]
    if not names:
        return {'status': 'NO_BOARDS', 'rows': []}
    items = resolve(names)
    day = str(plan.get('monitor_date'))
    key = (now.strftime('%Y%m%d'), tuple(names))
    hm = now.hour * 60 + now.minute
    active = (now.strftime('%Y%m%d') >= day and now.weekday() < 5 and
              (570 <= hm <= 690 or 780 <= hm <= 900))
    with LOCK:
        cached = CACHE.get(key)
        if not active:
            return {'status': 'WAIT_SESSION',
                    'rows': [{**r, 'status': 'WAIT_SESSION' if r['bk'] else 'UNRESOLVED'} for r in items]}
        if not RUNNING and (cached is None or time.monotonic() - cached['time'] >= 60):
            RUNNING = True

            def work():
                global RUNNING
                try:
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        rows = list(pool.map(lambda item: quote(item, now), items))
                    with LOCK:
                        CACHE[key] = {'time': time.monotonic(), 'rows': rows}
                        for old in list(CACHE):
                            if old != key:
                                del CACHE[old]
                finally:
                    with LOCK:
                        RUNNING = False
            threading.Thread(target=work, daemon=True, name='plan-board-quotes').start()
        rows = copy.deepcopy(cached['rows']) if cached else [
            {**r, 'status': 'LOADING' if r['bk'] else 'UNRESOLVED'} for r in items]
        for row in rows:
            if row.get('status') == 'OK':
                try:
                    age = (now - datetime.fromisoformat(row['quote_ts'])).total_seconds()
                    if not 0 <= age <= 120:
                        row.update(status='STALE', pct=None)
                except (ValueError, KeyError):
                    row.update(status='STALE', pct=None)
    from strong_sector_watch_v1 import pool_status
    return {'status':'OBSERVING','rows':rows,'strong_pool':pool_status(names,now),'rule_scans':[]}
