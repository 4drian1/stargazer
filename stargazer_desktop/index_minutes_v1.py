"""Read the local index mirror's status without querying any network service."""
import json
import importlib.util
import os
import threading
from datetime import datetime
from pathlib import Path

import paths


MIRROR = (Path(paths.APP_DIR).parent / 'stargazer_ml' / 'v5_hybrid' /
          'risk_r0_v1' / 'index_minutes_v1')
if paths.FROZEN:
    MIRROR = Path(paths.APP_DIR).parents[1] / 'stargazer_ml' / 'v5_hybrid' / 'risk_r0_v1' / 'index_minutes_v1'

def _tdx_root_default():
    """TDX vipdoc 路径兜底：本机私有路径不入库。
    环境变量 > 本机覆盖文件 data/tdx_root.txt（已 gitignore）> 通用占位符。"""
    override = Path(paths.data_file('data/tdx_root.txt', seed_from_bundle=False))
    if override.is_file():
        text = override.read_text(encoding='utf-8').strip()
        if text:
            return text
    return os.environ.get('STARGAZER_TDX_DEFAULT') or 'D:/tdx/vipdoc'


TDX_ROOT = Path(os.environ.get('STARGAZER_TDX_ROOT') or _tdx_root_default())
SOURCE_FILES = {'000001.SH': TDX_ROOT / 'sh/fzline/sh999999.lc5',
                '399006.SZ': TDX_ROOT / 'sz/fzline/sz399006.lc5'}
_sync_lock = threading.Lock()
_worker = None
_last_success = None


def latest_complete_cutoff(now, *, for_overview=False):
    """Latest available mirror watermark for the overview, shared cutoff for other callers."""
    try:
        stamps = []
        for name in ('sse', 'cyb'):
            try:
                stamp = int(json.loads((MIRROR / f'{name}_5min.meta.json').read_bytes())['last_complete_ts'])
                if stamp <= int(now.strftime('%Y%m%d%H%M%S')):
                    stamps.append(stamp)
            except (OSError, ValueError, TypeError, KeyError):
                pass
        if not stamps or (not for_overview and len(stamps) != 2):
            return None
        cutoff = max(stamps) if for_overview else min(stamps)
        when = datetime.strptime(str(cutoff), '%Y%m%d%H%M%S')
        if for_overview or when.date() == now.date():
            return cutoff
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def _load_updater():
    spec = importlib.util.spec_from_file_location(
        'tdx_source_updater', MIRROR / 'index_minutes_updater_v1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sync_once(as_of=None, *, public=None):
    """Run the prefix-aware sync through the shared updater.

    Binds exactly the two index sources (never an old export). Calls
    run_update(as_of, mirror_dir=MIRROR) which writes only COMPLETED bars, records
    in_progress tails in the sidecar, and emits release_status.json +
    runtime_fingerprint.json consumed by the post-close launcher.

    as_of: optional explicit Shanghai observation time (YYYYMMDD HHMMSS). Defaults
    to the host clock (treated as the Shanghai proxy) when omitted.
    """
    global _last_success
    with _sync_lock:
        status = {'checked_at': datetime.now().isoformat(timespec='seconds'),
                  'status': 'ERROR', 'sources': {k: str(v) for k, v in SOURCE_FILES.items()}}
        try:
            use_public=(as_of is None) if public is None else public
            if as_of is None:
                as_of = int(datetime.now().strftime('%Y%m%d%H%M%S'))  # host clock as Shanghai proxy
            updater = _load_updater()
            inboxes = list(SOURCE_FILES.values())
            public_owned=any(json.loads(p.read_bytes()).get('source_contract')=='SINA_INDEX_5M_PRICE_ONLY_V1'
                for p in (MIRROR/'sse_5min.meta.json',MIRROR/'cyb_5min.meta.json') if p.exists())
            rep = {} if use_public and public_owned else updater.run_update(inboxes, as_of, mirror_dir=MIRROR, verbose=False)
            if use_public:
                from public_index_feed_v1 import sync_public
                from postclose_risk_runtime import directory
                rep=sync_public(updater,MIRROR,directory(),as_of,rep)
            symbols = rep.get('symbols', {})
            rel = rep.get('release_status', {}) or {}
            rel_sym = rel.get('symbols', {}) or {}
            status['status'] = 'WAIT_FEED' if any(x.get('status')=='PUBLIC_SOURCE_WAIT' for x in symbols.values()) else 'OK'
            status['as_of'] = as_of
            status['symbols'] = {
                sym: {'status': (symbols.get(sym, {}) or {}).get('status'),
                      'rows': (symbols.get(sym, {}) or {}).get('rows'),
                      'reason': (symbols.get(sym, {}) or {}).get('reason'),
                      'release': rel_sym.get(sym, {}).get('release')}
                for sym in SOURCE_FILES
            }
            status['release'] = rel
            status['runtime_fingerprint'] = rel.get('runtime_fingerprint')
            _last_success = (datetime.now().date(), as_of)
        except Exception as exc:
            status['error'] = str(exc)
        temp = MIRROR / 'source_sync.auto.tmp'
        temp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(MIRROR / 'source_sync_status.json')
        return status


def start_source_sync():
    """One daemon; startup and periodic local-only updates do not block UI."""
    global _worker
    if _worker is not None and _worker.is_alive():
        return
    def loop():
        import time
        while True:
            try:
                sync_once()
            except Exception:
                pass  # Missing/unwritable mirror remains unavailable in the status API.
            time.sleep(60)
    _worker = threading.Thread(target=loop, name='tdx-index-source-sync', daemon=True)
    _worker.start()


# ---------------------------------------------------------------------------
# 总览只读指数 K 线：只读取已落盘的已准备镜像，不联网、不现场同步、不重算指标。
# ---------------------------------------------------------------------------
KLINE_CATALOG = {
    'SSE': dict(label='沪指', code='000001.SH', tdx='sh999999', parquet='sse_5min.parquet', meta='sse_5min.meta.json'),
    'SZSE': dict(label='深证成指', code='399001.SZ', tdx='sz399001', parquet=None, meta=None),
    'STAR50': dict(label='科创50', code='000688.SH', tdx='sh000688', parquet=None, meta=None),
    'CYB': dict(label='创业板指', code='399006.SZ', tdx='sz399006', parquet='cyb_5min.parquet', meta='cyb_5min.meta.json'),
}


def _local_kline_path(spec, period):
    stem = spec['tdx']
    folder, suffix = ('lday', '.day') if period == '1d' else ('fzline', '.lc5')
    return TDX_ROOT / stem[:2] / folder / (stem + suffix)


def kline_availability():
    """所有 (symbol, period) 组合的真实准备状态，供前端渲染切换而不伪造数据。"""
    out = []
    for sym, spec in KLINE_CATALOG.items():
        prepared = bool(spec['parquet'] and (MIRROR / spec['parquet']).is_file()
                        and (MIRROR / spec['meta']).is_file())
        for period in ('5m', '1d'):
            local = _local_kline_path(spec, period).is_file()
            status = ('READY' if local or (prepared and period == '5m') else
                      'DERIVED_FROM_PREPARED_5MIN' if prepared else 'NOT_PREPARED')
            out.append(dict(symbol=sym, label=spec['label'], code=spec['code'], period=period,
                            status=status, reason='无本地 K 线' if status == 'NOT_PREPARED' else None))
    return out


def _read_prepared_5min(spec):
    import pyarrow.parquet as pq
    meta = json.loads((MIRROR / spec['meta']).read_text(encoding='utf-8'))
    table = pq.read_table(MIRROR / spec['parquet'])
    rows = table.to_pylist()
    return rows, meta


def _daily_from_5min(rows):
    """由已准备 5 分钟 K 确定性聚合日 K；不引入任何新数据源。

    ts 取该日最后一根已准备 5 分钟 K 的时间，明确为「最后一根分钟时间」，
    绝不冒充官方日线收盘时间；当日未收盘时该根是盘中不完整日线。
    """
    grouped = {}
    for r in rows:
        day = int(r['day'])
        g = grouped.get(day)
        if g is None:
            grouped[day] = dict(ts=int(r['ts']), day=day, open=r['open'], high=r['high'],
                                low=r['low'], close=r['close'], bars=1)
        else:
            g['high'] = max(g['high'], r['high'])
            g['low'] = min(g['low'], r['low'])
            g['close'] = r['close']
            g['ts'] = int(r['ts'])
            g['bars'] += 1
    return [grouped[d] for d in sorted(grouped)]


def readonly_kline(symbol='SSE', period='5m', limit=120):
    """纯读已准备镜像。返回 {ok,status,bars,source,note,availability}。"""
    sym = str(symbol or 'SSE').upper()
    per = str(period or '5m').lower()
    availability = kline_availability()
    spec = KLINE_CATALOG.get(sym)
    if spec is None or per not in ('5m', '1d'):
        return dict(ok=False, status='INVALID_REQUEST', bars=[], availability=availability,
                    reason='symbol/period 不在只读目录内')
    try:
        count = max(1, min(int(limit or 120), 500))
        local = _local_kline_path(spec, per)
        mirror = bool(spec['parquet'] and (MIRROR / spec['parquet']).is_file()
                      and (MIRROR / spec['meta']).is_file())
        if local.is_file() and (per == '1d' or not mirror):
            return _read_tdx_kline(sym, spec, per, local, count)
        if not mirror:
            return dict(ok=True, status='NOT_PREPARED', symbol=sym, label=spec['label'],
                        period=per, bars=[], availability=availability, source=None,
                        note='本地没有该指数的 K 线，可手动读取行情源')
        rows, meta = _read_prepared_5min(spec)
        bars = rows if per == '5m' else _daily_from_5min(rows)
        bars = bars[-count:]
    except Exception as exc:
        return dict(ok=False, status='READ_FAILED', symbol=sym, period=per, bars=[],
                    availability=availability, reason=type(exc).__name__ + ':' + str(exc)[:120])
    source = dict(file=spec['parquet'], sha256=meta.get('sha256'),
                  last_complete_ts=meta.get('last_complete_ts'),
                  last_date=meta.get('last_date'), rows=meta.get('rows'),
                  time_semantics=meta.get('time_semantics'),
                  volume_verified=bool(meta.get('volume_verified')))
    note = ('只读已准备镜像：5 分钟 K 为 COMPLETED bars；'
            '时间语义为混合源、结束时刻按约定假设且未认证；成交量未核验，不展示量能结论。')
    if per == '1d':
        note += ('日 K 由同一份已准备 5 分钟 K 按交易日确定性聚合，非官方日线；'
                 '每根 ts 是该日最后一根已准备分钟 K 的时间，当日未收盘则为不完整日线。')
    return dict(ok=True, status='READY' if per == '5m' else 'DERIVED_FROM_PREPARED_5MIN',
                symbol=sym, label=spec['label'], code=spec['code'], period=per,
                bars=bars, availability=availability, source=source, note=note)


def _read_tdx_kline(symbol, spec, period, path, limit):
    from hashlib import sha256
    from datetime import timedelta
    from functools import lru_cache
    from math import isfinite
    now = datetime.now()
    if period == '1d':
        from tdx_daily_source_v2 import read_day
        through = now.date() if now.hour >= 15 else now.date() - timedelta(days=1)
        frame, meta = read_day(path, int(through.strftime('%Y%m%d')))
        rows = frame[['date', 'open', 'high', 'low', 'close']].rename(columns={'date': 'ts'}).to_dict('records')
        sha = meta['sha256']
    else:
        updater = _load_updater()
        data = path.read_bytes()
        cutoff = int(now.strftime('%Y%m%d%H%M%S'))
        grid = lru_cache(maxsize=128)(updater.grid)
        rows = [bar for bar in updater.parse_lc5(data)
                if bar['ts'] <= cutoff and bar['ts'] in grid(bar['day'])
                and all(isfinite(bar[key]) and bar[key] > 0 for key in ('open','high','low','close'))
                and bar['low'] <= min(bar['open'], bar['close']) <= max(bar['open'], bar['close']) <= bar['high']]
        sha = sha256(data).hexdigest()
    return _chart_response(symbol, spec, period, rows[-limit:], path.name, sha)


def _chart_response(symbol, spec, period, rows, source, sha=None):
    return dict(ok=bool(rows), status='READY' if rows else 'NOT_PREPARED',
                symbol=symbol, label=spec['label'], code=spec['code'], period=period,
                bars=[{k: row[k] for k in ('ts', 'open', 'high', 'low', 'close')} for row in rows],
                source=dict(file=source, sha256=sha, volume_verified=False,
                            last_complete_ts=rows[-1]['ts'] if rows else None))


def fetch_chart_kline(symbol, period, limit=120):
    """Explicit display-only refresh. Index evidence and mirror files are separate."""
    from tools.market import _get
    spec = KLINE_CATALOG.get(symbol)
    if not spec or period not in ('1d', '5m'):
        raise ValueError('INVALID_REQUEST')
    count = max(1, min(int(limit), 500))
    code, exchange = spec['code'].split('.')
    market = 1 if exchange == 'SH' else 0
    if symbol == 'STAR50' and period == '5m':
        raw = _get('https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20k=/'
                   'CN_MarketDataService.getKLineData?symbol=sh000688&scale=5&ma=no'
                   f'&datalen={count + 1}', tries=1, timeout=5, encoding='utf-8')
        if raw and 'var k=(' in raw:
            try:
                data, _ = json.JSONDecoder().raw_decode(raw.split('var k=(', 1)[1])
                lines = [(bar['day'], bar['open'], bar['close'], bar['high'], bar['low']) for bar in data]
                rows = _parse_chart_bars(lines, period, count)
                if rows:
                    return _chart_response(symbol, spec, period, rows, '新浪科创50指数行情')
            except (TypeError, ValueError, KeyError):
                pass
    klt = 101 if period == '1d' else 5
    url = ('https://push2his.eastmoney.com/api/qt/stock/kline/get?'
           f'secid={market}.{code}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55'
           f'&klt={klt}&fqt=0&beg=0&end=20500101&lmt={count + 1}')
    value = json.loads(_get(url, tries=1, timeout=5) or '{}').get('data') or {}
    if value.get('code') != code or str(value.get('market')) != str(market):
        raise ValueError('INDEX_SOURCE_UNAVAILABLE')
    rows = _parse_chart_bars((line.split(',')[:5] for line in value.get('klines') or []), period, count)
    return _chart_response(symbol, spec, period, rows, '东方财富指数行情')


def _parse_chart_bars(lines, period, count):
    from math import isfinite
    now = datetime.now()
    rows = {}
    for stamp, o, c, h, l in lines:
        when = datetime.fromisoformat(stamp)
        if when > now or (period == '1d' and when.date() == now.date() and now.hour < 15):
            continue
        o, c, h, l = map(float, (o, c, h, l))
        if not all(isfinite(p) and p > 0 for p in (o, c, h, l)) or not l <= min(o, c) <= max(o, c) <= h:
            continue
        ts = int(when.strftime('%Y%m%d' if period == '1d' else '%Y%m%d%H%M%S'))
        rows[ts] = dict(ts=ts, open=o, high=h, low=l, close=c)
    return [rows[ts] for ts in sorted(rows)[-count:]]


def get_index_freshness():
    report = json.loads((MIRROR / 'freshness.json').read_text(encoding='utf-8'))
    symbols = {}
    for name in ('sse', 'cyb'):
        meta = json.loads((MIRROR / f'{name}_5min.meta.json').read_text(encoding='utf-8'))
        if not (MIRROR / f'{name}_5min.parquet').is_file():
            raise ValueError('INDEX_MIRROR_MISSING')
        symbols[name] = {'last_date': meta.get('last_date'), 'rows': meta.get('rows'),
            'last_complete_ts':meta.get('last_complete_ts'), 'last_observed_ts':meta.get('last_observed_ts')}
    dates = [item['last_date'] for item in symbols.values()]
    last_date = min(dates) if all(dates) else None
    checked = str(report.get('checked_at') or report.get('updated_at', ''))[:10]
    status_current = checked == datetime.now().strftime('%Y-%m-%d')
    sync_path = MIRROR / 'source_sync_status.json'
    sync_status = json.loads(sync_path.read_text(encoding='utf-8')) if sync_path.exists() else None
    sync_failed = sync_status is not None and sync_status.get('status') != 'OK'
    # After close, yesterday's mirror is not sufficient for today's plan.
    now = datetime.now()
    behind_close = (now.weekday() < 5 and now.hour >= 15 and
                    (last_date is None or int(last_date) < int(now.strftime('%Y%m%d'))))
    grid=[m for m in [*range(575,691,5),*range(785,901,5)] if m<=now.hour*60+now.minute]
    latest=max(grid,default=0)
    required=int(now.strftime('%Y%m%d'))*1000000+latest//60*10000+latest%60*100
    return {**report, 'last_date': last_date, 'symbols': symbols,
            'stale': bool(report.get('stale', True)) or not status_current or last_date is None or sync_failed or behind_close,
            'status_current': status_current,
            'reason': '' if status_current else 'FRESHNESS_CHECK_OUTDATED',
            'source_sync': sync_status,
            'intraday_cutoff_verified': bool(grid) and all(int(r.get('last_complete_ts') or 0)>=required for r in symbols.values())}
