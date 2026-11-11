"""Local TDX event dates only. Never manufactures prices or adjustment factors."""
import gzip,hashlib,json,os,struct
from datetime import datetime
from pathlib import Path
from runtime_disk_limit_v1 import disk_limit

CONTRACT='TDX_CORPORATE_EVENT_DATES_V1'
READER_SHA256='e28fae76760b1247a960cab285aa357c2d263b3212e7cabfb3f1ebfcc127d6ba'
PRICE_EVENTS={1,11,12}
KNOWN_CATEGORIES=set(range(1,15))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def source_path():
    from tdx_native_history_v1 import source_root
    return Path(os.environ['STARGAZER_TDX_GBBQ']) if os.environ.get('STARGAZER_TDX_GBBQ') else source_root().parent/'T0002/hq_cache/gbbq'
def owner(root):
    root=Path(root).resolve()
    for p in [root,*root.parents]:
        if (p/'runtime_budget_v2.json').is_file():return p
    return root
def budget(root,extra):
    from persistent_records_v1 import is_record, reserve_space
    if is_record(root):
        reserve_space(root,extra)
        return
    root=owner(root)
    from runtime_disk_limit_v1 import disk_usage,MONITOR_RESERVE
    used=disk_usage(root)
    if used+extra+MONITOR_RESERVE>disk_limit(root):raise ValueError('RUNTIME_DISK_BUDGET')

def event_index(runtime):
    import inspect
    from pytdx.reader.gbbq_reader import GbbqReader
    source=source_path()
    if not source.is_file():raise ValueError('TDX_CORPORATE_SOURCE_MISSING')
    raw=source.read_bytes()
    if len(raw)>16*1024**2 or len(raw)<4:raise ValueError('TDX_CORPORATE_SOURCE_SIZE')
    count=struct.unpack('<I',raw[:4])[0]
    if len(raw)!=4+count*29 or count==0:raise ValueError('TDX_CORPORATE_SOURCE_LAYOUT')
    source_sha=hashlib.sha256(raw).hexdigest();reader_sha=sha(inspect.getfile(GbbqReader))
    if reader_sha!=READER_SHA256:raise ValueError('TDX_CORPORATE_DECODER_CHANGED')
    key=digest([CONTRACT,source_sha,reader_sha]);folder=owner(runtime)/'tdx_corporate_events'
    target=folder/(key+'.json.gz')
    if target.exists():
        value=json.loads(gzip.decompress(target.read_bytes()));payload=value['payload']
        if digest(payload)!=value['sha256']:raise ValueError('TDX_CORPORATE_CACHE_SHA')
        if (payload['source_sha256'],payload['reader_sha256'])!=(source_sha,reader_sha):
            raise ValueError('TDX_CORPORATE_CACHE_IDENTITY')
    else:
        frame=GbbqReader().get_df(str(source))
        if sha(source)!=source_sha:raise ValueError('TDX_CORPORATE_SOURCE_CHANGED')
        if len(frame)!=count:raise ValueError('TDX_CORPORATE_RECORD_COUNT')
        import pandas as pd
        pd.to_datetime(frame['datetime'].astype(str),format='%Y%m%d',errors='raise')
        if not frame.code.str.fullmatch(r'\d{6}').all():raise ValueError('TDX_CORPORATE_CODE')
        index={}
        for row in frame[['market','code','datetime','category']].itertuples(index=False):
            index.setdefault(str(int(row.market))+':'+row.code,[]).append([int(row.datetime),int(row.category)])
        index={k:sorted({tuple(v) for v in values}) for k,values in index.items()}
        payload=dict(contract=CONTRACT,source_path=str(source),source_sha256=source_sha,
            reader_sha256=reader_sha,record_count=count,index=index,observed_at=datetime.now().isoformat())
        packed=gzip.compress(json.dumps(dict(payload=payload,sha256=digest(payload)),separators=(',',':')).encode(),mtime=0)
        budget(runtime,len(packed)*2);folder.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            temporary=folder/(key+'.'+str(os.getpid())+'.tmp')
            try:
                with temporary.open('xb') as f:f.write(packed)
                try:os.link(temporary,target)
                except FileExistsError:pass  # Another completed identical-source index won.
            finally:
                if temporary.exists():temporary.unlink()
    if sha(source)!=source_sha:raise ValueError('TDX_CORPORATE_SOURCE_CHANGED')
    return payload,target,source.stat()

def checked_events(code,day,reference_day,runtime):
    """Explicitly checked provider snapshot; absence means no listed event here."""
    from tdx_native_history_v1 import locate
    code=str(code).zfill(6);day=int(day);reference_day=int(reference_day)
    if reference_day>=day:raise ValueError('TDX_CORPORATE_REFERENCE_DAY')
    prefix=locate(code).stem[:2]
    if prefix not in ('sh','sz'):raise ValueError('TDX_CORPORATE_MARKET_UNSUPPORTED')
    market=1 if prefix=='sh' else 0
    payload,path,stat=event_index(runtime)
    source_day=int(datetime.fromtimestamp(stat.st_mtime).strftime('%Y%m%d'))
    if source_day<reference_day:raise ValueError('TDX_CORPORATE_SOURCE_STALE')
    rows=payload['index'].get(str(market)+':'+code)
    if not rows:raise ValueError('TDX_CORPORATE_SECURITY_NOT_COVERED')
    if any(d==day and (c not in KNOWN_CATEGORIES or c==4) for d,c in rows):
        raise ValueError('TDX_CORPORATE_EVENT_TYPE_UNRESOLVED')
    events=[('tdx_event:'+code+':'+str(d),dict(date=d,category=c,source_contract=CONTRACT))
            for d,c in rows if d<=day and c in PRICE_EVENTS]
    receipt=dict(contract=CONTRACT,code=code,market=market,day=day,reference_day=reference_day,
        source_path=payload['source_path'],source_sha256=payload['source_sha256'],reader_sha256=payload['reader_sha256'],
        index_path=str(path),index_sha256=sha(path),source_local_mtime_day=source_day,
        observed_at=datetime.now().isoformat(),checked=True,source_security_records=len(rows),
        events_on_day=sum(1 for d,c in rows if d==day and c in PRICE_EVENTS),
        absence_meaning='NO_PRICE_EVENT_LISTED_IN_THIS_PROVIDER_SNAPSHOT',
        freshness_basis='LOCAL_FILE_MTIME_NOT_UPSTREAM_CERTIFICATION',
        qfq_generated=False,price_or_factor_generated=False)
    return events,receipt
