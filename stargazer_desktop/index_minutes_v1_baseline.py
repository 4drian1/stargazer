"""Read the local index mirror's status without querying any network service."""
import json
import os
import importlib.util
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
    环境变量 > 模块旁 data/tdx_root.txt（已 gitignore）> 通用占位符。"""
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


def _load_updater():
    spec = importlib.util.spec_from_file_location(
        'tdx_source_updater', MIRROR / 'index_minutes_updater_v1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sync_once():
    """Read only the two bound sources; never substitute an old export."""
    global _last_success
    with _sync_lock:
        status = {'checked_at': datetime.now().isoformat(timespec='seconds'),
                  'status': 'ERROR', 'sources': {k: str(v) for k, v in SOURCE_FILES.items()}}
        try:
            signature = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns)
                              for p in SOURCE_FILES.values())
            identity = (datetime.now().date(), signature)
            if identity == _last_success:
                return {'status': 'UNCHANGED'}
            updater = _load_updater()
            prepared = []
            for symbol, source in SOURCE_FILES.items():
                before = source.stat()
                if before.st_size > 32 * 1024 * 1024:
                    raise ValueError('SOURCE_SIZE_LIMIT')
                with source.open('rb') as stream:
                    body = stream.read(32 * 1024 * 1024 + 1)
                if len(body) > 32 * 1024 * 1024:
                    raise ValueError('SOURCE_SIZE_LIMIT')
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('SOURCE_CHANGED_DURING_READ')
                rows = updater.parse_lc5(body)
                validation = updater.validate(rows)
                if validation['issues']:
                    raise ValueError('SOURCE_VALIDATION_FAILED:' + symbol)
                prepared.append((symbol, rows, [{'path': str(source),
                                 'sha256': updater.sha256(body), 'bytes': len(body)}], validation))
            current = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns)
                            for p in SOURCE_FILES.values())
            if current != signature:
                raise ValueError('SOURCE_CHANGED_DURING_READ')
            # Preflight both overlap checks before either mirror is updated.
            for symbol, rows, _, _ in prepared:
                table, _ = updater.load_mirror(symbol)
                old = {r['ts']: r for r in table.to_pylist()} if table is not None else {}
                for row in rows:
                    previous = old.get(row['ts'])
                    if previous and any(previous[k] != row[k] for k in row if k != 'ts'):
                        raise ValueError('OVERLAP_CONFLICT:' + symbol)
            status['symbols'] = {symbol: updater.merge_and_write(symbol, rows, info, validation)
                                 for symbol, rows, info, validation in prepared}
            fresh = updater.get_index_freshness()
            temp = MIRROR / 'freshness.auto.tmp'
            temp.write_text(json.dumps(fresh, ensure_ascii=False, indent=2), encoding='utf-8')
            temp.replace(MIRROR / 'freshness.json')
            status['status'] = 'OK'
        except Exception as exc:
            status['error'] = str(exc)
        temp = MIRROR / 'source_sync.auto.tmp'
        temp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(MIRROR / 'source_sync_status.json')
        if status['status'] == 'OK':
            _last_success = identity
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


def get_index_freshness():
    report = json.loads((MIRROR / 'freshness.json').read_text(encoding='utf-8'))
    symbols = {}
    for name in ('sse', 'cyb'):
        meta = json.loads((MIRROR / f'{name}_5min.meta.json').read_text(encoding='utf-8'))
        if not (MIRROR / f'{name}_5min.parquet').is_file():
            raise ValueError('INDEX_MIRROR_MISSING')
        symbols[name] = {'last_date': meta.get('last_date'), 'rows': meta.get('rows')}
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
    return {**report, 'last_date': last_date, 'symbols': symbols,
            'stale': bool(report.get('stale', True)) or not status_current or last_date is None or sync_failed or behind_close,
            'status_current': status_current,
            'reason': '' if status_current else 'FRESHNESS_CHECK_OUTDATED',
            'source_sync': sync_status,
            'intraday_cutoff_verified': False}
