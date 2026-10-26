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

TDX_ROOT = Path(os.environ.get('STARGAZER_TDX_ROOT') or
                 r'D:\chat_attachments\local-user-files\msg\attach\f8420fdf70caa64279ff160028b64197\2026-09\Rec\bb5d6414f73b90d2\F\0\new_mpv （9月）\vipdoc')
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
