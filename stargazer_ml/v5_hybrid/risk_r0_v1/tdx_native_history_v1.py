"""Explicit native END_5M history; never fabricate 1-minute observations."""
import gzip
import hashlib
import json
import os
from pathlib import Path
from functools import lru_cache
from tdx_native_reader_v1 import read_lc5_file, validate_day, validate_prefix

CONTRACT='TDX_NATIVE_END_5M_HISTORY_V1'

def source_root():
    value=os.environ.get('STARGAZER_TDX_ROOT')
    if value:return Path(value)
    import index_minutes_v1
    return index_minutes_v1.TDX_ROOT

def configured():
    return source_root().is_dir()

def locate(code,root=None):
    root=Path(root) if root is not None else source_root()
    code=str(code).zfill(6)
    if len(code)!=6 or not code.isdigit():raise ValueError('TDX_STOCK_CODE')
    found=[root/m/'fzline'/(m+code+'.lc5') for m in ('sh','sz','bj')
           if (root/m/'fzline'/(m+code+'.lc5')).is_file()]
    if len(found)!=1:raise ValueError('TDX_STOCK_FILE_MISSING_OR_AMBIGUOUS:'+code)
    return found[0]

@lru_cache(maxsize=16)
def _read(path,mtime,size):
    return read_lc5_file(path)

def native_rows(code,days,root=None,cutoff=None):
    days=[int(d) for d in days]
    if days!=sorted(set(days)) or not days:raise ValueError('TDX_HISTORY_DAYS')
    p=locate(code,root);st=p.stat();info=_read(str(p),st.st_mtime_ns,st.st_size)
    out=[]
    for day in days:
        rows=[r for r in info['rows'] if r['day']==day]
        # cutoff is None -> a FINISHED day: enforce the exact 48-bar grid
        # (validate_day). cutoff is given -> the day is replayed only UP TO the
        # cutoff, so the kept bars must form the continuous legal grid PREFIX
        # (validate_prefix) ending at the cutoff slot. This still rejects
        # gaps/reordering/duplicates/off-grid/bad-price bars and anything past
        # the cutoff, but does not demand the whole session. With cutoff = the
        # session-end (the historical-review default) the prefix gate degrades
        # to the full 48-bar requirement, so historical days are not relaxed.
        if cutoff is None:
            if not validate_day(rows,day)['eligible']:
                raise ValueError('TDX_HISTORY_DAY_INCOMPLETE:'+str(code)+':'+str(day))
            emit = rows
        else:
            vp = validate_prefix(rows, day, cutoff)
            if not vp['eligible']:
                raise ValueError('TDX_DAY_PREFIX_INCOMPLETE:'+str(code)+':'+str(day)+':'+vp.get('reason',''))
            # Emit ONLY the bars at or before the cutoff -- validate_prefix proved
            # they form a continuous legal prefix, so dropping the post-cutoff
            # bars here is what actually excludes them from the replay.
            emit = [r for r in rows if r['ts'] <= cutoff]
        out.extend(dict(date=r['ts'],code=str(code).zfill(6),open=r['open'],high=r['high'],
            low=r['low'],close=r['close'],volume=r['raw_volume'],amount=r['amount'],
            source_frequency_minutes=5,source_clock='END_5M') for r in emit)
    prefix=hashlib.sha256(json.dumps(out,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return out,dict(contract=CONTRACT,path=str(p),source_sha256=info['sha256'],
        prefix_sha256=prefix,days=days,frequency_minutes=5,clock='END_5M',
        availability_basis='LOCAL_READER_OBSERVED_BYTES_NOT_UPSTREAM_FIRST_ARRIVAL')

def day_minutes(code,day,budget,runtime):
    rows,receipt=native_rows(code,[int(day)])
    folder=Path(runtime)/'tdx_native_history'/str(day)/str(code).zfill(6)
    target=folder/(receipt['prefix_sha256']+'.json.gz')
    if target.exists():
        value=json.loads(gzip.decompress(target.read_bytes()))
        if hashlib.sha256(json.dumps(value['rows'],sort_keys=True,allow_nan=False).encode()).hexdigest()!=receipt['prefix_sha256']:
            raise ValueError('TDX_HISTORY_CACHE_HASH')
        if value['receipt']!=receipt:
            # Full file can append future bars; identity is the requested prefix.
            if value['receipt']['prefix_sha256']!=receipt['prefix_sha256']:
                raise ValueError('TDX_HISTORY_CACHE_IDENTITY')
        return value['rows'],dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                                  contract=CONTRACT,frequency_minutes=5)
    body=gzip.compress(json.dumps(dict(rows=rows,receipt=receipt),ensure_ascii=False,
                                  allow_nan=False).encode(),mtime=0)
    budget(len(body)+4096);folder.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        with target.open('xb') as f:f.write(body)
        if hasattr(budget,'written'):budget.written(len(body))
    return rows,dict(path=str(target),sha256=hashlib.sha256(body).hexdigest(),
                     contract=CONTRACT,frequency_minutes=5)
