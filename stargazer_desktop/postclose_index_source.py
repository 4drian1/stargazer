"""Freeze complete local index days independently of the live public mirror."""
import hashlib
import json
from pathlib import Path

CONTRACT = 'LOCAL_TDX_CLOSED_INDEX_BYTES_V1'
SYMBOLS = ('000001.SH', '399006.SZ')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def job_due(snapshot, now):
    day = int(snapshot['data_date'])
    today = int(now.strftime('%Y%m%d'))
    return (day == today and now.hour >= 15) or day < today <= int(snapshot['monitor_date'])


def _read(day, as_of):
    import index_minutes_v1 as index
    if day * 1000000 + 150000 > as_of:
        raise ValueError('LOCAL_CLOSE_NOT_YET_COMPLETE')
    updater = index._load_updater()
    sources = {}
    for symbol in SYMBOLS:
        path = index.SOURCE_FILES[symbol]
        if path.stem.lower() not in updater.KNOWN or updater.KNOWN[path.stem.lower()][0] != symbol:
            raise ValueError('LOCAL_INDEX_IDENTITY:' + symbol)
        if path.stat().st_size > 16 * 1024**2:
            raise ValueError('LOCAL_INDEX_FILE_SIZE:' + symbol)
        raw = path.read_bytes()
        rows = updater.parse_lc5(raw)
        report = updater.validate(rows)
        if any(issue[1] in ('OHLC_VIOLATION', 'INVALID_PRICE_OR_AMOUNT') for issue in report['issues']):
            raise ValueError('LOCAL_INDEX_BAD_PRICE:' + symbol)
        complete, excluded = [], []
        by_day = {}
        for row in rows:
            if row['day'] <= day:
                by_day.setdefault(row['day'], []).append(row)
        for date, bars in by_day.items():
            result = updater.validate_prefix(bars, date, day * 1000000 + 150000,
                                             as_of, force_full_day=True)
            if result['verdict'] == 'POSTCLOSE_OK':
                complete.extend(bars)
            else:
                excluded.append(date)
        target = [row for row in complete if row['day'] == day]
        if len(target) != 48 or target[-1]['ts'] != day * 1000000 + 150000:
            found = len(by_day.get(day, []))
            raise ValueError(f'LOCAL_INDEX_DAY_INCOMPLETE:{symbol}:{day}:bars={found}')
        sources[symbol] = dict(path=str(path), sha256=sha(raw), raw=raw,
                               rows=complete, excluded_incomplete_days=excluded)
    return sources


def inspect(day, as_of):
    try:
        sources = _read(int(day), int(as_of))
        return dict(status='READY', data_date=int(day), cutoff=int(day)*1000000+150000,
                    sources={s:{k:v for k,v in r.items() if k not in ('raw', 'rows')}
                             for s,r in sources.items()})
    except (OSError, ValueError, KeyError) as exc:
        return dict(status='WAIT_LOCAL_INDEX', data_date=int(day), reason=str(exc))


def prepare(day, as_of, root):
    """Retain exact original bytes; never overwrite the active intraday mirror."""
    sources = _read(int(day), int(as_of))
    identity = sha(json.dumps({s:r['sha256'] for s,r in sources.items()}, sort_keys=True).encode())
    folder = Path(root) / 'index_closes' / (str(day) + '_' + identity)
    target = folder / 'source.json'
    if not target.exists():
        from public_index_feed_v1 import budget
        budget(Path(root), sum(len(r['raw']) for r in sources.values()) + 8192)
        folder.mkdir(parents=True, exist_ok=True)
        entries = {}
        for symbol, record in sources.items():
            name = Path(record['path']).name
            file = folder / name
            if not file.exists():
                with file.open('xb') as stream:
                    stream.write(record['raw'])
            if sha(file.read_bytes()) != record['sha256']:
                raise ValueError('LOCAL_FROZEN_INDEX_HASH:' + symbol)
            entries[symbol] = {k:v for k,v in record.items() if k not in ('raw', 'rows')}
            entries[symbol]['captured_file'] = name
        value = dict(contract=CONTRACT, data_date=int(day), cutoff=int(day)*1000000+150000,
                     observed_at=int(as_of), availability_basis='LOCAL_READER_OBSERVED_BYTES_NOT_UPSTREAM_FIRST_ARRIVAL',
                     volume_verified=False, activity_policy='WITHHELD_NULL', sources=entries)
        with target.open('xb') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True).encode())
    digest = sha(target.read_bytes())
    load(target, digest, int(day))
    return target, digest


def load(path, expected_sha, day, read=None):
    """Read only the bound source and its complete price prefix; keep volume unknown."""
    import index_minutes_v1 as index
    import pandas as pd
    path = Path(path)
    def checked(file, digest):
        raw = read(file, digest) if read else file.read_bytes()
        if sha(raw) != digest:
            raise ValueError('LOCAL_INDEX_BINDING_HASH')
        return raw
    manifest = json.loads(checked(path, expected_sha))
    if (manifest.get('contract') != CONTRACT or manifest.get('data_date') != day
            or manifest.get('cutoff') != day*1000000+150000
            or manifest.get('observed_at', 0) < manifest['cutoff']
            or set(manifest.get('sources', {})) != set(SYMBOLS)):
        raise ValueError('LOCAL_INDEX_BINDING_CUTOFF_OR_IDENTITY')
    updater = index._load_updater()
    indices, receipts = {}, {}
    for symbol, entry in manifest['sources'].items():
        file = path.parent / entry['captured_file']
        if file.parent.resolve() != path.parent.resolve() or updater.KNOWN.get(file.stem.lower(), (None,))[0] != symbol:
            raise ValueError('LOCAL_INDEX_BINDING_PATH')
        rows = updater.parse_lc5(checked(file, entry['sha256']))
        report = updater.validate(rows)
        if any(issue[1] in ('OHLC_VIOLATION', 'INVALID_PRICE_OR_AMOUNT') for issue in report['issues']):
            raise ValueError('LOCAL_INDEX_BAD_PRICE')
        rows = [r for r in rows if r['day'] <= day and r['day'] not in entry['excluded_incomplete_days']]
        target = [r for r in rows if r['day'] == day]
        if updater.validate_prefix(target, day, manifest['cutoff'], manifest['observed_at'], force_full_day=True)['verdict'] != 'POSTCLOSE_OK':
            raise ValueError('LOCAL_INDEX_BINDING_INCOMPLETE')
        frame = pd.DataFrame(rows).rename(columns={'ts':'date'})
        frame['volume'] = float('nan')
        frame['amount'] = float('nan')
        indices[symbol] = frame[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
        receipts[symbol] = dict(reference=str(file)+'#sha256='+entry['sha256'], price_basis='RAW',
            clock='END_5M', time_semantics='END_5M_ASSUMED_NOT_CERTIFIED', available_at=manifest['observed_at'],
            availability_basis=manifest['availability_basis'], volume_verified=False,
            activity_policy='WITHHELD_NULL', source_contract=CONTRACT)
    return indices, receipts
