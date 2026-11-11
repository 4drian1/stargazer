"""Verified local daily review records; remove only archived old strong-pool cache."""
from datetime import datetime
import base64
import gzip
import hashlib
import json
from pathlib import Path

CONTRACT = 'STRONG_SECTOR_DAILY_REVIEW_V1'


def sha(body):
    return hashlib.sha256(body).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()


def archive_directory(runtime):
    return Path(runtime).parent / 'review_archive/strong_sector'


def usage(runtime):
    """Persistent review bytes still count towards the original shared disk cap."""
    from runtime_disk_limit_v1 import disk_usage
    return disk_usage(runtime)


def _safe(root, relative):
    path = root / relative
    if path.is_symlink() or root.resolve() not in path.resolve().parents:
        raise ValueError('ARCHIVE_PATH_ESCAPE')
    return path


def build_day(runtime, day, read_day):
    root = Path(runtime) / 'strong_sector'
    files = {}
    pools = []
    for path in sorted((root / 'pools').glob(str(day) + '_*.json')):
        value = json.loads(path.read_bytes())
        if path.name.endswith('.failed.json') or value.get('status') != 'READY':
            continue
        if int(value['observation_date']) != day or sha(encoded(value['rows'])) != value['rows_sha256']:
            raise ValueError('ARCHIVE_POOL_HASH')
        pools.append(value)
        files[path.relative_to(root).as_posix()] = path.read_bytes()
    if not pools:
        return None
    alerts_path = root / 'alerts.json'
    alerts = json.loads(alerts_path.read_bytes()) if alerts_path.exists() else {}
    notices = {k: v for k, v in alerts.items() if str(v.get('cutoff', ''))[:8] == str(day)}
    rows = {}
    for pool in pools:
        for row in pool['rows']:
            code = row['code']
            if code in rows:
                continue
            events = [dict(event_id=k, **v) for k, v in notices.items() if v.get('code') == code]
            record = dict(code=code, name=row['name'], boards=row['boards'],
                reference_close=row['reference_close'], positions=row['positions'], events=events,
                observation_status='ALERT_RECORDED' if events else 'NO_RECORDED_ALERT',
                monitoring_coverage='UNKNOWN_NOT_PROOF_OF_NO_TRIGGER')
            try:
                bars, receipt = read_day(code, day)
                close = bars[-1]['close']
                record.update(price_status='KNOWN', close=close,
                    return_vs_pool_reference=close / row['reference_close'] - 1,
                    return_basis='CLOSE_VS_FROZEN_PRIOR_CLOSE_NOT_TOTAL_RETURN',
                    low=min(b['low'] for b in bars), high=max(b['high'] for b in bars),
                    source_receipt=receipt)
            except (OSError, ValueError, KeyError) as exc:
                record.update(price_status='UNKNOWN', price_reason=str(exc)[:200])
            rows[code] = record
            map_path = root / 'maps' / (row['map_sha'] + '.json.gz')
            if map_path.is_file():
                files[map_path.relative_to(root).as_posix()] = map_path.read_bytes()
    minute_root = root / 'minutes' / str(day)
    for path in minute_root.rglob('*'):
        if path.is_file() and not path.is_symlink() and path.suffix != '.tmp':
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    return dict(contract=CONTRACT, day=day, rows=list(rows.values()), pools=pools,
        recorded_alerts=notices, archived_at=datetime.now().isoformat(),
        event_semantics='Only actually recorded events; no invented missed alerts or trades',
        files={k: dict(sha256=sha(v), bytes=len(v), base64=base64.b64encode(v).decode())
               for k, v in files.items()})


def seal_and_clean(runtime, value, now, limit):
    """Archive, fsync, reread and verify every byte before any deletion."""
    runtime = Path(runtime)
    day = value['day']
    today = int(now.strftime('%Y%m%d'))
    if day >= today:
        raise ValueError('CURRENT_OR_FUTURE_DAY_PROTECTED')
    if any(row.get('price_status')!='KNOWN' for row in value['rows']):
        raise ValueError('ARCHIVE_DAY_PRICES_INCOMPLETE_CACHE_RETAINED')
    folder = archive_directory(runtime) / str(day)
    raw = encoded(value)
    compressed = gzip.compress(raw, mtime=0)
    from persistent_records_v1 import is_record,reserve_space
    persistent=is_record(folder)
    if persistent:reserve_space(folder,len(compressed)*2+65536)
    if len(compressed) > 16 * 1024**2 or (not persistent and usage(runtime) + len(compressed) * 2 + 65536 > limit):
        raise ValueError('ARCHIVE_DISK_BUDGET')
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (sha(raw) + '.json.gz')
    if not target.exists():
        import os
        with target.open('xb') as stream:
            stream.write(compressed)
            stream.flush()
            os.fsync(stream.fileno())
    loaded = json.loads(gzip.decompress(target.read_bytes()))
    if loaded != value:
        raise ValueError('ARCHIVE_VERIFY_FAILED')
    root = runtime / 'strong_sector'
    targets = []
    for relative, receipt in loaded['files'].items():
        body = base64.b64decode(receipt['base64'], validate=True)
        if sha(body) != receipt['sha256'] or len(body) != receipt['bytes']:
            raise ValueError('ARCHIVE_ENTRY_HASH')
        path = _safe(root, relative)
        if path.exists() and sha(path.read_bytes()) != receipt['sha256']:
            raise ValueError('ARCHIVE_CACHE_CHANGED')
        if relative.startswith('pools/') or relative.startswith('minutes/' + str(day) + '/'):
            targets.append(path)
    # A map is disposable only when no remaining pool refers to it.
    retiring={str(p) for p in targets}
    referenced=set()
    for pool in (root/'pools').glob('*.json'):
        if str(pool) not in retiring:
            referenced.update(r['map_sha'] for r in json.loads(pool.read_bytes()).get('rows',[]))
    for relative in loaded['files']:
        if relative.startswith('maps/') and Path(relative).name.removesuffix('.json.gz') not in referenced:
            targets.append(_safe(root,relative))
    for path in targets:
        if path.exists():
            path.unlink()
    return dict(day=day, archive=str(target), archive_sha256=sha(target.read_bytes()),
        rows=len(value['rows']), recorded_events=len(value['recorded_alerts']),
        prices_known=sum(r['price_status'] == 'KNOWN' for r in value['rows']),
        cache_files_removed=len(targets), raw_sources_removed=0)


def maintain(runtime, now, read_day, limit):
    root = Path(runtime) / 'strong_sector/pools'
    days = sorted({int(p.name[:8]) for p in root.glob('????????_*.json')
                   if p.name[:8].isdigit() and int(p.name[:8]) < int(now.strftime('%Y%m%d'))})
    results = []
    for day in days:
        value = build_day(runtime, day, read_day)
        if value is not None:
            results.append(seal_and_clean(runtime, value, now, limit))
    return dict(status='ARCHIVED', days=results)


def review(runtime, day, codes=None):
    """Load verified archived observations without rerunning signals or querying stockdb."""
    rows={}
    receipts=[]
    folder=archive_directory(runtime)/str(day)
    for path in sorted(folder.glob('*.json.gz')):
        raw=gzip.decompress(path.read_bytes())
        if sha(raw)!=path.name.removesuffix('.json.gz'):
            raise ValueError('REVIEW_ARCHIVE_HASH')
        value=json.loads(raw)
        if value.get('contract')!=CONTRACT or value.get('day')!=int(day):raise ValueError('REVIEW_ARCHIVE_IDENTITY')
        receipts.append(dict(path=str(path),sha256=sha(path.read_bytes())))
        for row in value['rows']:
            if not codes or row['code'] in codes:rows[row['code']]=row
    return dict(status='ARCHIVED_OBSERVATIONS' if receipts else 'ARCHIVE_NOT_FOUND',day=int(day),
        rows=list(rows.values()),archives=receipts,actual_trades='NOT_PROVIDED',
        note='未记录提醒不等于未触发；涨跌不代表实际买卖盈亏')
