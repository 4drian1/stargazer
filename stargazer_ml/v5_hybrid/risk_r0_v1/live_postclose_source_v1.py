"""Local-only daily source adapter. Frozen feature kernels remain unchanged."""
import gzip
import hashlib
import io
import json
import os
import time
from pathlib import Path
import urllib.parse
import urllib.request

import msgpack
import pandas as pd

from joint_shadow_context_v1 import RiskSnapshot
from joint_shadow_snapshot_v1 import HERE, now_stamp

READ_CAP = 48 * 1024**3  # Bounded archive replay after a recorded adapter correction; no extra HTTP scope.
DISK_CAP = 1024**3  # User approved 1 GiB for this runtime on 2026-10-21.
RESPONSE_CAP = 32 * 1024**2
REQUEST_CAP = 16000
FIELDS = ['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']


def disk_limit(root):
    policy=Path(root)/'runtime_budget_v2.json'
    if not policy.exists():return DISK_CAP
    value=json.loads(policy.read_bytes())
    cap=value.get('disk_bytes')
    if cap not in (1024**3,4*1024**3) or value.get('user_authorized') is not True:
        raise ValueError('UNAUTHORIZED_RUNTIME_BUDGET')
    return cap


def digest(body):
    return hashlib.sha256(body).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    for attempt in range(20):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt==19:
                raise
            time.sleep(.05)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('LOCAL_REDIRECT_REFUSED')


class LocalCapture:
    def __init__(self, output, budget_root=None):
        self.output = Path(output)
        self.budget_root = Path(budget_root or output)
        self.requests = self.read_bytes = 0
        self.disk_bytes = sum(p.stat().st_size for p in Path(budget_root or output).rglob('*') if p.is_file())
        self.disk_cap=disk_limit(self.budget_root)
        self.cache_day = None
        self.fresh_periods=set()
        self.reused_responses = 0
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def read(self, path, expected=None):
        path = Path(path)
        size = path.stat().st_size
        if size > RESPONSE_CAP or self.read_bytes + size > READ_CAP:
            raise ValueError('LOCAL_READ_BUDGET')
        body = path.read_bytes()
        self.read_bytes += len(body)
        if expected and digest(body) != expected:
            raise ValueError('LOCAL_SOURCE_HASH_MISMATCH:' + path.name)
        return body

    def save(self, name, body):
        if self.disk_bytes + len(body) + 2*1024**2 > self.disk_cap:
            raise ValueError('RUNTIME_DISK_BUDGET')
        path = self.output / name
        with path.open('xb') as stream:
            stream.write(body)
        self.disk_bytes += len(body)
        return {'path': str(path), 'sha256': digest(body)}

    def fetch(self, code, period):
        if self.cache_day is not None and period not in self.fresh_periods:
            name=f'raw_{code}_{period.replace("*", "all")}.msgpack.gz'
            cached=sorted(self.output.parent.glob(f'run_{self.cache_day}_*/{name}'))
            for path in cached:
                if path.parent==self.output:
                    continue
                try:
                    body=self.read(path)
                    raw=gzip.decompress(body)
                    if len(raw)>RESPONSE_CAP:
                        raise ValueError('CACHED_RESPONSE_SIZE')
                    rows=msgpack.unpackb(raw,raw=False,strict_map_key=False)
                    if not isinstance(rows,list):
                        continue
                    self.reused_responses+=1
                    return rows,dict(path=str(path),sha256=digest(body),raw_sha256=digest(raw),
                        observed_at=now_stamp(),period=period,reused=True,
                        provenance='SAME_DAY_LOCAL_RESPONSE_ARCHIVE_READBACK')
                except (ValueError,EOFError,gzip.BadGzipFile,msgpack.UnpackException):
                    continue
        journal=None
        if self.cache_day is not None:
            journal=self.budget_root/f'attempts_{self.cache_day}.jsonl'
            if journal.exists():
                if journal.stat().st_size>8*1024**2:
                    raise ValueError('ATTEMPT_JOURNAL_SIZE')
                lines=journal.read_text(encoding='utf-8').splitlines()
                if lines:
                    self.requests=max(self.requests,json.loads(lines[-1])['quota_count'])
        if self.requests >= REQUEST_CAP:
            raise ValueError('LOCAL_REQUEST_BUDGET')
        self.requests += 1
        if journal is not None:
            record=(json.dumps(dict(code=code,period=period,quota_count=self.requests,
                reserved_at=now_stamp(),kind='LOCAL_HTTP_ATTEMPT_RESERVATION'))+'\n').encode()
            if self.disk_bytes+len(record)+2*1024**2>self.disk_cap:
                raise ValueError('RUNTIME_DISK_BUDGET')
            with journal.open('ab') as stream:
                stream.write(record)
                stream.flush()
                os.fsync(stream.fileno())
            self.disk_bytes+=len(record)
        query = urllib.parse.quote(f'分钟k:{code}:{period}')
        request = urllib.request.Request('http://127.0.0.1:7899/?cmd=vals&t=' + query,
                                         headers={'Accept': 'application/msgpack'})
        cap = min(RESPONSE_CAP, READ_CAP - self.read_bytes)
        with self.opener.open(request, timeout=15) as response:
            raw = response.read(cap + 1)
        self.read_bytes += len(raw)
        if len(raw) > cap:
            raise ValueError('LOCAL_RESPONSE_BUDGET')
        ref = self.save(f'raw_{code}_{period.replace("*", "all")}.msgpack.gz',
                        gzip.compress(raw, mtime=0))
        try:
            value = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        except (ValueError, msgpack.UnpackException):
            value = json.loads(raw)
        if not isinstance(value, list):
            raise ValueError('MINUTE_RESPONSE_NOT_LIST')
        rows = [r[1] if isinstance(r, list) and len(r) == 2 and isinstance(r[1], dict) else r
                for r in value]
        if any(not isinstance(r, dict) for r in rows):
            raise ValueError('MINUTE_RESPONSE_ROW_SHAPE')
        ref.update(raw_sha256=digest(raw), observed_at=now_stamp(), period=period)
        return rows, ref


def months(start, end):
    year, month = start // 10000, start // 100 % 100
    last = end // 100
    while year*100 + month <= last:
        yield f'{year:04}{month:02}*'
        month += 1
        if month == 13:
            year, month = year+1, 1


def merge_minutes(seed, rows, code, cutoff):
    extra = pd.DataFrame(rows, columns=FIELDS)
    frame = pd.concat([seed, extra], ignore_index=True)
    if frame.empty:
        return frame
    if not (frame.code.astype(str).str.zfill(6) == code).all():
        raise ValueError('MINUTE_IDENTITY_MISMATCH')
    frame['code'] = code
    frame = frame[frame.date <= cutoff].copy()
    minute = frame.date // 100 % 10000
    frame = frame[((minute > 930) & (minute <= 1130)) |
                  ((minute > 1300) & (minute <= 1500))]
    duplicate = frame[frame.duplicated('date', keep=False)]
    if len(duplicate) and (duplicate.groupby('date')[FIELDS].nunique(dropna=False).max(axis=1) > 1).any():
        raise ValueError('MINUTE_OVERLAP_CONFLICT')
    return frame.drop_duplicates('date').sort_values('date').reset_index(drop=True)


def clean_ohlc(frame):
    """Same invalid-OHLC mask as frozen stock_features; absent bars stay absent."""
    import numpy as np
    values=frame[['open','high','low','close']].to_numpy(dtype=float)
    o,h,l,c=values.T
    bad=~np.isfinite(values).all(axis=1)|(np.minimum.reduce([o,h,l,c])<=0)|(
        l>np.minimum(o,c))|(h<np.maximum(o,c))
    return frame.loc[~bad].reset_index(drop=True),list(frame.loc[bad,'date'].astype(int))


class CapturedMinutes:
    def __init__(self, entries, capture, cutoff, progress=None):
        self.entries, self.capture, self.cutoff = entries, capture, cutoff
        self.progress,self.consumed=progress,0
        self.quarantined={}

    def __contains__(self, code):
        return code in self.entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, code):
        entry = self.entries[code]
        seed = (pd.read_parquet(io.BytesIO(self.capture.read(entry['seed']['path'], entry['seed']['sha256'])))
                if entry.get('seed') else pd.DataFrame(columns=FIELDS))
        rows = []
        for ref in entry['extensions']:
            body = gzip.decompress(self.capture.read(ref['path'], ref['sha256']))
            if digest(body) != ref['raw_sha256']:
                raise ValueError('RAW_RESPONSE_HASH_MISMATCH')
            try:
                value = msgpack.unpackb(body, raw=False, strict_map_key=False)
            except (ValueError, msgpack.UnpackException):
                value = json.loads(body)
            rows.extend(r[1] if isinstance(r, list) and len(r) == 2 else r for r in value)
        frame,bad=clean_ohlc(merge_minutes(seed, rows, code, self.cutoff))
        if bad:
            self.quarantined[code]=bad
        self.consumed+=1
        if self.progress and (self.consumed%20==0 or self.consumed==len(self.entries)):
            self.progress(stage='REBUILDING_FROZEN_FEATURES',done=self.consumed,total=len(self.entries),
                requests=self.capture.requests,read_bytes=self.capture.read_bytes,
                disk_bytes=self.capture.disk_bytes,quarantined_member_count=len(self.quarantined))
        return frame


def produce_live(day, candidates, output, progress, *, cutoff=None):
    """Capture exact local bytes first, then consume them through frozen kernels."""
    capture = LocalCapture(output, budget_root=Path(output).parent)
    capture.cache_day = day
    # Interrupted capture retains its quota ledger; do not reset attempts/reads.
    previous_runs=[p for p in Path(output).parent.glob(f'run_{day}_*') if p!=Path(output)]
    capture.requests=sum(len(list(p.glob('raw_*.msgpack.gz'))) for p in previous_runs)
    for previous_run in previous_runs:
        terminal=previous_run/'terminal.json'
        if not terminal.exists():
            continue
        previous=json.loads(terminal.read_text(encoding='utf-8'))
        if list(previous_run.glob('raw_*.msgpack.gz')) and previous.get('status') in (
                'STOPPED_FOR_AUTHORIZED_BUDGET_UPDATE','FAILED'):
            capture.requests+=1  # One interrupted/in-flight attempt is reserved conservatively.
    prior=Path(output).parent/'read_quota_carry_forward.json'
    if prior.exists():
        record=json.loads(prior.read_text())
        if Path(record['prior_stage']).name.startswith(f'run_{day}_'):
            capture.read_bytes=record['prior_read_bytes_upper_bound']
    cutoff = int(day)*1000000+150000 if cutoff is None else int(cutoff)
    if cutoff//1000000!=int(day) or cutoff>now_stamp():
        raise ValueError('SOURCE_CUTOFF_INVALID_OR_FUTURE')
    if cutoff%1000000<150000:
        capture.fresh_periods={str(day)+'*',str(day)[:6]+'*','*'}
    scope = json.loads(capture.read(HERE/'ths_full_concept_scope_20261017_v1.json'))
    catalog = pd.read_parquet(io.BytesIO(capture.read(scope['source'], scope['source_sha256'])))
    catalog = catalog[(catalog.source == 'ths') & (catalog.type == 'concept')]
    if set(catalog.code) != {r['code'] for r in scope['boards']}:
        raise ValueError('COMPARISON_UNIVERSE_DRIFT')
    boards = {r.code: list(r.symbols) for r in catalog.sort_values('code').itertuples()}
    universe = {c for members in boards.values() for c in members}
    ledger = json.loads(capture.read(HERE/'relative_stall_5m_20261018_v1/raw_input_ledger.json'))
    by_code = {r['code']: r for r in ledger}
    from live_postclose_cache_v1 import prior_entries,remaining_periods
    history=prior_entries(Path(output).parent,day)
    bound = sorted(((set(by_code)|set(history)) & universe) | (set(candidates) & universe))
    indices, receipts = {}, {}
    mirror = HERE/'index_minutes_v1'
    for symbol, name in [('000001.SH', 'sse'), ('399006.SZ', 'cyb')]:
        meta = json.loads(capture.read(mirror/f'{name}_5min.meta.json'))
        path = mirror/f'{name}_5min.parquet'
        frame = pd.read_parquet(io.BytesIO(capture.read(path, meta['sha256'])))
        frame = frame.rename(columns={'ts': 'date', 'raw_volume': 'volume'})
        frame = frame[frame.date <= cutoff]
        if frame.empty or int(frame.date.max()) != cutoff:
            raise ValueError('INDEX_MIRROR_NOT_AT_CUTOFF:' + symbol)
        indices[symbol] = frame
        receipts[symbol] = dict(reference=str(path)+'#sha256='+meta['sha256'],
            price_basis='RAW', clock='END_5M', available_at=now_stamp(),
            availability_basis='LOCAL_READER_OBSERVED_BYTES_NOT_UPSTREAM_FIRST_ARRIVAL')
    days = sorted(d for d in set((indices['000001.SH'].date//1000000).astype(int)) if d >= 20250102)
    entries, failures = {}, {}
    for i, code in enumerate(bound):
        inherited=history.get(code)
        seed = inherited.get('seed') if inherited else by_code.get(code)
        frame = pd.DataFrame(columns=FIELDS)
        if inherited:
            frame=CapturedMinutes(history,capture,cutoff)[code]
        elif seed:
            frame = pd.read_parquet(io.BytesIO(capture.read(seed['path'], seed['sha256'])))
        start = int(frame.date.max()//1000000) if len(frame) else None
        periods = (remaining_periods(start,days) if inherited and start else
                   list(months(start, int(day))) if start else ['*'])
        extra, refs = [], list(inherited['extensions']) if inherited else []
        for period in periods:
            rows, ref = capture.fetch(code, period)
            extra.extend(rows)
            refs.append(ref)
        try:
            combined = merge_minutes(frame, extra, code, cutoff)
            if combined.empty or int(combined.date.max()//1000000) < int(day):
                raise ValueError('CURRENT_DAY_MINUTES_MISSING')
            # Invalid bars are rejected, not silently counted as a flat member.
            import numpy as np
            p = combined[['open','high','low','close','volume','amount']].to_numpy(dtype=float)
            if not np.isfinite(p).all() or (p[:,:4] <= 0).any() or (p[:,4:] < 0).any():
                raise ValueError('INVALID_MINUTE_VALUE')
            entries[code] = {'seed': seed, 'extensions': refs}
            receipts[code] = dict(reference='captured:'+code+':'+digest(json.dumps(entries[code], sort_keys=True).encode()),
                price_basis='RAW', clock='END_1M', available_at=now_stamp(),
                availability_basis='LOCAL_READER_OBSERVED_BYTES_NOT_UPSTREAM_FIRST_ARRIVAL')
        except ValueError as exc:
            failures[code] = str(exc)
        if i % 20 == 0 or i+1 == len(bound):
            progress(stage='CAPTURING_MINUTES', done=i+1, total=len(bound),
                     requests=capture.requests, read_bytes=capture.read_bytes, disk_bytes=capture.disk_bytes)
    capture.save('source_manifest.json', json.dumps(dict(entries=entries, failures=failures,
        membership_source=scope, declared_stock_count=len(universe), bound_stock_count=len(entries),
        requests=capture.requests, read_bytes=capture.read_bytes, disk_bytes=capture.disk_bytes),
        ensure_ascii=False).encode())
    observed = now_stamp()
    snapshot = RiskSnapshot(cutoff, observed, days, CapturedMinutes(entries, capture, cutoff,progress),
                           indices, boards, receipts, observed, 'FORWARD_SNAPSHOT', observed_at=observed)
    return snapshot, capture, failures
