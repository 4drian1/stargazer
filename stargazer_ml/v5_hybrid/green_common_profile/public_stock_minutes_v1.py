"""Current-session public OHLC fallback; historical stockdb evidence stays intact."""
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
import urllib.request

CONTRACT = 'SINA_STOCK_1M_WITH_FIXED_HISTORY_V2'
MAX_BYTES = 131072


def minute_grid(day, cutoff):
    minutes = list(range(571, 691)) + list(range(781, 901))
    return [day * 1000000 + m // 60 * 10000 + m % 60 * 100
            for m in minutes if day * 1000000 + m // 60 * 10000 + m % 60 * 100 <= cutoff]


def parse(raw, code, previous_day, cutoff, previous_close):
    values = json.loads(raw)
    if not isinstance(values, list) or len(values) > 360:
        raise ValueError('PUBLIC_STOCK_SCHEMA')
    day = cutoff // 1000000
    rows, seen = [], set()
    previous = None
    for item in values:
        stamp = int(datetime.strptime(item['day'], '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M%S'))
        if stamp == previous_day * 1000000 + 150000:
            previous = float(item['close'])
        if stamp // 1000000 != day or stamp > cutoff:
            continue
        if stamp in seen or (rows and stamp < rows[-1]['date']):
            raise ValueError('PUBLIC_STOCK_DUPLICATE_OR_ORDER')
        seen.add(stamp)
        o, h, l, c = [float(item[k]) for k in ('open', 'high', 'low', 'close')]
        volume, amount = float(item['volume']), float(item['amount'])
        if (not all(math.isfinite(v) for v in (o, h, l, c, volume, amount)) or
                min(o, h, l, c) <= 0 or not l <= min(o, c) <= max(o, c) <= h or min(volume, amount) < 0):
            raise ValueError('PUBLIC_STOCK_OHLCV_INVALID')
        # Preserve shares, not lots. Refuse an incompatible volume/amount basis.
        if (volume == 0) != (amount == 0) or (volume and not l * .99 <= amount / volume <= h * 1.01):
            raise ValueError('PUBLIC_STOCK_VOLUME_BASIS_UNRESOLVED')
        rows.append(dict(date=stamp, code=code, open=o, high=h, low=l, close=c,
                         volume=volume, amount=amount))
    if previous is None or abs(previous - previous_close) > .011:
        raise ValueError('PUBLIC_STOCK_PREVIOUS_CLOSE_BASIS_MISMATCH')
    if [r['date'] for r in rows] != minute_grid(day, cutoff):
        raise ValueError('PUBLIC_STOCK_COMPLETE_PREFIX_MISSING')
    return rows


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def fetch(code):
    if len(code) != 6 or not code.isdigit():
        raise ValueError('PUBLIC_STOCK_IDENTITY')
    symbol = ('sh' if code[0] in '569' else 'sz') + code
    if code[0] in '48':
        raise ValueError('PUBLIC_STOCK_EXCHANGE_UNSUPPORTED')
    url = ('https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData?'
           'symbol=' + symbol + '&scale=1&ma=no&datalen=360')
    with urllib.request.build_opener(NoRedirect()).open(
            urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=6) as response:
        raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('PUBLIC_STOCK_RESPONSE_LIMIT')
    return raw, url


def current_minutes(folder, code, previous_day, cutoff, previous_close, local, budget, receipts):
    from golden_ladder_v3 import completed_bars
    day = cutoff // 1000000
    bars, _, _, status = completed_bars(local, code, day, cutoff)
    if status == 'COMPLETE_PREFIX_OPTIONAL_OPEN' and bars and day * 1000000 + bars[-1]['t'] == cutoff:
        return local, dict(provider='STOCKDB', contract='EXISTING_LOCAL_MINUTE_PREFIX')
    if day != int(datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')):
        raise ValueError('PUBLIC_STOCK_LIVE_DAY_ONLY')
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / f'public_{code}_{cutoff}.json.gz'
    meta = archive.with_suffix('.meta.json')
    if archive.exists() and meta.exists():
        evidence = json.loads(meta.read_bytes())
        compressed = archive.read_bytes()
        if hashlib.sha256(compressed).hexdigest() != evidence['sha256']:
            raise ValueError('PUBLIC_STOCK_CACHE_HASH')
        raw = gzip.decompress(compressed)
    else:
        ledger = folder / 'public_attempts.json'
        attempts = json.loads(ledger.read_bytes()) if ledger.exists() else {}
        if attempts.get(code, 0) >= 48:
            raise ValueError('PUBLIC_STOCK_DAILY_REQUEST_LIMIT')
        budget(MAX_BYTES + 16384)
        attempts[code] = attempts.get(code, 0) + 1
        temp = ledger.with_suffix('.tmp')
        temp.write_text(json.dumps(attempts), encoding='utf8')
        temp.replace(ledger)
        raw, url = fetch(code)  # One attempt; no immediate retry.
        compressed = gzip.compress(raw, mtime=0)
        evidence = dict(provider='SINA_PUBLIC', contract=CONTRACT, code=code, cutoff=cutoff,
            available_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
            availability_basis='LOCAL_RESPONSE_OBSERVED_NOT_UPSTREAM_FIRST_ARRIVAL',
            time_semantics='SINA_1M_END_STRUCTURE_ASSUMED_NOT_CERTIFIED',
            volume_basis='SHARES_AMOUNT_PER_SHARE_CHECKED_AGAINST_OHLC',
            history_policy='CURRENT_SESSION_ONLY_STOCKDB_HISTORY_UNCHANGED', url=url,
            sha256=hashlib.sha256(compressed).hexdigest())
        budget(len(compressed) + 4096)
        with archive.open('xb') as stream:
            stream.write(compressed)
        with meta.open('x', encoding='utf8') as stream:
            json.dump(evidence, stream, ensure_ascii=False)
    receipts.append(dict(path=str(archive), sha256=evidence['sha256'], **{
        k: evidence[k] for k in ('provider', 'contract', 'code', 'cutoff', 'available_at', 'time_semantics')}))
    current = parse(raw, code, previous_day, cutoff, previous_close)
    history = [r for r in local if int(r['date']) // 1000000 < day]
    unique = {}
    for row in history + current:
        if str(row.get('code')).zfill(6) != code:
            raise ValueError('PUBLIC_STOCK_HISTORY_IDENTITY')
        stamp = int(row['date'])
        if stamp in unique and any(unique[stamp].get(k) != row.get(k) for k in ('open','high','low','close','volume','amount')):
            raise ValueError('PUBLIC_STOCK_HISTORY_CONFLICT')
        unique[stamp] = row
    return [unique[t] for t in sorted(unique)], evidence
