"""Local TDX daily source. Raw model/execution prices and P-fixed qfq are separate.

Only worker processes materialize data; importing this module does no IO. Current
names/ST are snapshots, never historical labels. No stockdb or network fallback.
"""
import bisect
from datetime import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
from functools import lru_cache

CONTRACT='TDX_DAILY_RAW_MODEL_QFQ_SEPARATE_V2'
PRICE_FIELDS=('open','high','low','close')
MODEL_PREFIX=re.compile(r'(?:00|30|60|68)\d{4}$')

def peak_memory():
    import ctypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in
            ('peak','working','qpeak','quota','ppeak','paged','pagefile','pagefile_peak')]
    c=Counters();c.cb=ctypes.sizeof(c);ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(c),c.cb):
        raise ctypes.WinError()
    if not 0<c.peak<=512*1024**2:raise ValueError('TDX_DAILY_RAM_LIMIT:'+str(c.peak))
    return c.peak

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def root():
    import index_minutes_v1
    return Path(os.environ.get('STARGAZER_TDX_ROOT') or index_minutes_v1.TDX_ROOT)

def configured():return root().is_dir()

def source_code_path():
    # GUI __file__ is inside _MEI; identities bind the persistent worker source.
    import paths
    app=Path(paths.APP_DIR).resolve()
    project=Path(os.environ.get('STARGAZER_PROJECT_ROOT') or
        (app.parent.parent if app.name=='dist' else app.parent))
    return project/'stargazer_desktop/tdx_daily_source_v2.py'

def calendar_file():
    import importlib.util
    if getattr(sys,'frozen',False):
        project=source_code_path().parents[1]
        for name in ('_dev_venv','rule_venv'):
            path=project/name/'Lib/site-packages/akshare/file_fold/calendar.json'
            if path.is_file():return path
        raise ValueError('TDX_WORKER_CALENDAR_MISSING')
    return Path(importlib.util.find_spec('akshare').origin).parent/'file_fold/calendar.json'

def calendar_days():return sorted(set(map(int,json.loads(calendar_file().read_bytes()))))

def runtime(project):
    # The same parent budget owns minute, risk AND daily caches.
    return Path(project)/'stargazer_desktop/dist/data/postclose_risk'

def day_path(code):
    value=str(code).upper()
    indices={'000001.SH':'sh999999','399006.SZ':'sz399006'}
    if value in indices:stem=indices[value]
    elif re.fullmatch(r'\d{6}',value):
        found=[m+value for m in ('sh','sz','bj') if (root()/m/'lday'/(m+value+'.day')).is_file()]
        if len(found)!=1:raise ValueError('TDX_DAILY_CODE_MISSING_OR_AMBIGUOUS:'+value)
        stem=found[0]
    else:raise ValueError('TDX_DAILY_CODE')
    return root()/stem[:2]/'lday'/(stem+'.day')

def read_day(path,p):
    """Unsigned daily price/volume fields, preserve order, validate before use."""
    import pandas as pd
    path=Path(path);raw=path.read_bytes();source_sha=hashlib.sha256(raw).hexdigest()
    if not raw or len(raw)%32:raise ValueError('TDX_DAY_LAYOUT')
    values=[];previous=0
    for d,o,h,l,c,a,v,_ in struct.iter_unpack('<IIIIIfII',raw):
        datetime.strptime(str(d),'%Y%m%d')
        if d<=previous:raise ValueError('TDX_DAY_ORDER_OR_DUPLICATE')
        previous=d
        # Future records are not input to historical price calculations.
        if d>p:continue
        if (not math.isfinite(a) or a<0 or min(o,h,l,c)<=0 or l>min(o,c) or h<max(o,c) or l>h):
            raise ValueError('TDX_DAY_BAD_VALUES')
        values.append((d,o/100,h/100,l/100,c/100,v,float(a)))
    if sha(path)!=source_sha:raise ValueError('TDX_DAY_SOURCE_CHANGED')
    frame=pd.DataFrame(values,columns=['date',*PRICE_FIELDS,'volume','amount'])
    return frame,dict(path=str(path),sha256=source_sha,rows=len(frame))

def names_from_tnf(path):
    """This installation's s.tnf: header50, records360, code0/name31."""
    raw=Path(path).read_bytes()
    if len(raw)<50 or (len(raw)-50)%360:raise ValueError('TDX_TNF_LAYOUT')
    out={}
    for offset in range(50,len(raw),360):
        record=raw[offset:offset+360]
        code=record[:6].decode('ascii',errors='strict')
        try:name=record[31:63].split(b'\0')[0].decode('gbk',errors='strict').strip()
        except UnicodeDecodeError:
            if MODEL_PREFIX.fullmatch(code):raise ValueError('TDX_EQUITY_NAME_ENCODING:'+code)
            continue  # Long fund names may be truncated mid-GBK in this snapshot.
        if not re.fullmatch(r'\d{6}',code):raise ValueError('TDX_TNF_CODE')
        if not name:continue
        if code in out and out[code]!=name:raise ValueError('TDX_TNF_CONFLICT')
        out[code]=name
    return out

def base_records(path):
    raw=Path(path).read_bytes()
    if len(raw)<33:raise ValueError('TDX_BASE_LAYOUT')
    count=struct.unpack_from('<I',raw,4)[0];hlen,rlen=struct.unpack_from('<HH',raw,8)
    if not count or hlen<33 or rlen<2 or len(raw)<hlen+count*rlen:raise ValueError('TDX_BASE_LAYOUT')
    fields={};offset=1
    for pos in range(32,hlen-1,32):
        desc=raw[pos:pos+32];name=desc[:11].split(b'\0')[0].decode('ascii')
        fields[name]=(offset,desc[16]);offset+=desc[16]
    if offset!=rlen or not {'SC','GPDM','GXRQ','ZGB','LTAG'}<=set(fields):raise ValueError('TDX_BASE_SCHEMA')
    out={}
    for i in range(count):
        record=raw[hlen+i*rlen:hlen+(i+1)*rlen]
        if record[:1]==b'*':continue
        if record[:1]!=b' ':raise ValueError('TDX_BASE_DELETE_FLAG')
        def cell(k):
            start,length=fields[k];return record[start:start+length].decode('ascii').strip()
        code=cell('GPDM');market={'0':'sz','1':'sh','2':'bj'}.get(cell('SC'))
        if not market or not re.fullmatch(r'\d{6}',code):continue
        try:
            stamp=int(cell('GXRQ'));datetime.strptime(str(stamp),'%Y%m%d')
            total=float(cell('ZGB'))*10000;floating=float(cell('LTAG'))*10000
        except ValueError:continue
        if not all(math.isfinite(x) and x>0 for x in (total,floating)) or floating>total*1.001:continue
        key=market+code;value=dict(snapshot_update_day=stamp,total_share=total,float_share=floating)
        if key in out and out[key]!=value:raise ValueError('TDX_BASE_CONFLICT')
        out[key]=value
    return out

def events_by_code(path):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stargazer_ml/v5_hybrid/risk_r0_v1'))
    from pytdx.reader.gbbq_reader import GbbqReader
    from tdx_corporate_events_v1 import READER_SHA256
    if sha(inspect.getfile(GbbqReader))!=READER_SHA256:raise ValueError('TDX_CORPORATE_DECODER_CHANGED')
    before=sha(path);raw=Path(path).read_bytes()
    if len(raw)<4 or len(raw)!=4+struct.unpack_from('<I',raw)[0]*29:raise ValueError('TDX_GBBQ_LAYOUT')
    table=GbbqReader().get_df(str(path))
    if len(table)!=struct.unpack_from('<I',raw)[0] or sha(path)!=before:raise ValueError('TDX_GBBQ_CHANGED')
    out={}
    for row in table.itertuples(index=False,name=None):
        market,code,day,category,*numbers=row
        if market not in (0,1,2) or not re.fullmatch(r'\d{6}',code):raise ValueError('TDX_GBBQ_IDENTITY')
        datetime.strptime(str(day),'%Y%m%d')
        if not all(math.isfinite(float(x)) for x in numbers):raise ValueError('TDX_GBBQ_BAD_VALUE')
        out.setdefault({0:'sz',1:'sh',2:'bj'}[market]+code,[]).append((int(day),int(category),*map(float,numbers)))
    return {k:sorted(set(v)) for k,v in out.items()}

def price_views(frame,events,p):
    """Category1 proportional qfq, fixed P; raw prices never overwritten.

    On suspension dates the last observed raw close is carried through *events*,
    not through missing trading bars. Unsupported price events block this view.
    """
    import numpy as np
    raw=frame.copy();raw['pre_close']=raw.close.shift(1)
    qfq=frame.copy();factors=np.ones(len(raw));applied=[]
    if raw.empty:return raw,qfq,applied
    grouped={}
    for event in events:
        d,c,*v=event
        if not int(raw.date.iloc[0])<d<=p:continue
        if c in (4,11,12,13,14) or c not in range(1,15):raise ValueError('TDX_PRICE_EVENT_UNSUPPORTED:'+str(c))
        if c==1:
            if d in grouped and grouped[d]!=v:raise ValueError('TDX_PRICE_EVENT_CONFLICT')
            grouped[d]=v
    dates=raw.date.to_numpy();previous_index=None;reference=None
    for day,v in sorted(grouped.items()):
        before=int(np.searchsorted(dates,day)-1);after=int(np.searchsorted(dates,day))
        if before<0:raise ValueError('TDX_EX_RIGHTS_PREVIOUS_CLOSE_MISSING')
        if before!=previous_index:reference=float(raw.close.iloc[before])
        dividend,rights_price,bonus,rights=v
        if min(v)<0:raise ValueError('TDX_PRICE_EVENT_NEGATIVE')
        ex=(reference*10-dividend+rights*rights_price)/(10+bonus+rights)
        if not 0<ex<reference*10:raise ValueError('TDX_EX_RIGHTS_PRICE')
        ratio=ex/reference;factors[:after]*=ratio
        if after<len(raw):raw.loc[after,'pre_close']=ex
        applied.append(dict(day=day,reference=reference,ex_reference=ex,factor=ratio,
            dividend_per10=dividend,bonus_per10=bonus,rights_per10=rights,rights_price=rights_price))
        previous_index=before;reference=ex
    for field in PRICE_FIELDS:qfq[field]=raw[field]*factors
    qfq['pre_close']=qfq.close.shift(1);qfq['qfq_factor']=factors
    return raw,qfq,applied

def normalized(frame,stem,name,base,events,p,snapshot_day):
    import numpy as np
    raw,qfq,applied=price_views(frame,events,p)
    code=stem[2:];raw['code']=code;raw['name']=name
    # Current security status is only asserted on the operational snapshot day.
    raw['is_st']=np.nan
    if name and p==snapshot_day:raw.loc[raw.date==p,'is_st']=int('ST' in name.upper() or '退' in name)
    timeline=[];seen_shares={}
    for d,c,a,b,f,t in events:
        if c not in (1,4,11,12,13,14,15) and f>0 and t>0 and f<=t*1.001:
            if d in seen_shares and max(abs(x-y) for x,y in zip(seen_shares[d],(f,t)))>0.01:
                raise ValueError('TDX_SHARE_EVENT_CONFLICT')
            seen_shares[d]=(f,t)
            timeline.append((d,f*10000,t*10000))
    timeline.sort();days=[x[0] for x in timeline]
    floating=[];total=[]
    for d in raw.date:
        i=bisect.bisect_right(days,int(d))-1
        value=timeline[i][1:] if i>=0 else (np.nan,np.nan)
        # A DBF is a current snapshot, not a historical share series.
        if d==p and p==snapshot_day and base and base['snapshot_update_day']<=p:
            value=(base['float_share'],base['total_share'])
        floating.append(value[0]);total.append(value[1])
    raw['float_share']=floating;raw['total_share']=total
    raw['float_mv']=raw.float_share*raw.close;raw['total_mv']=raw.total_share*raw.close
    raw['pct_chg']=(raw.close/raw.pre_close-1)*100
    raw['turnover']=raw.volume/raw.float_share*100
    raw['price_basis']='TDX_RAW_WITH_EVENT_REFERENCE_CLOSE'
    return raw,qfq,applied

def source_inventory(p):
    hq=root().parent/'T0002/hq_cache'
    meta=[hq/'base.dbf',hq/'gbbq',hq/'shs.tnf',hq/'szs.tnf',hq/'bjs.tnf',calendar_file(),source_code_path()]
    paths=[]
    for market in ('sh','sz','bj'):
        for f in (root()/market/'lday').glob('*.day'):
            code=f.stem[2:]
            if MODEL_PREFIX.fullmatch(code) and ((market=='sh' and code.startswith(('60','68'))) or
                    (market=='sz' and code.startswith(('00','30')))):paths.append(f)
    if not paths:raise ValueError('TDX_DAILY_SOURCE_EMPTY')
    # Every actual file has a content fingerprint; metadata alone is no cache proof.
    entries=[]
    for f in sorted([*paths,*meta]):
        entry=dict(path=str(f.resolve()),sha256=sha(f),bytes=f.stat().st_size)
        if f.suffix=='.day':entry['prefix_sha256']=day_prefix_sha(f,p)
        entries.append(entry)
    identity=dict(contract=CONTRACT,P=p,root=str(root().resolve()),inputs=entries)
    return paths,meta,identity,digest(identity)

def day_prefix_sha(path,p):
    raw=Path(path).read_bytes()
    if len(raw)%32:raise ValueError('TDX_DAY_LAYOUT')
    prefix=b''.join(raw[i:i+32] for i in range(0,len(raw),32) if struct.unpack_from('<I',raw,i)[0]<=p)
    return hashlib.sha256(prefix).hexdigest()

def frozen_directory(project,p):
    """Future appends do not invalidate a completed P's frozen metadata.

    Price corrections, added/deleted security files and calculation-code changes
    DO invalidate it. On P after close a fresh metadata download also invalidates.
    """
    now=datetime.now()
    if int(now.strftime('%Y%m%d'))==p and now.hour>=15:return None
    parent=runtime(project)/'tdx_daily_v2'/str(p)
    current=set()
    for market,prefixes in [('sh',('60','68')),('sz',('00','30'))]:
        current.update(str(f.resolve()) for f in (root()/market/'lday').glob('*.day')
            if MODEL_PREFIX.fullmatch(f.stem[2:]) and f.stem[2:].startswith(prefixes))
    for complete in sorted(parent.glob('*/COMPLETE.json')):
        if '.building.' in complete.parent.name:continue
        value=json.loads(complete.read_bytes());identity=value['identity']
        if identity.get('contract')!=CONTRACT or identity.get('P')!=p or identity.get('root')!=str(root().resolve()):continue
        inputs=identity['inputs'];days={e['path'] for e in inputs if Path(e['path']).suffix=='.day'}
        if days!=current:continue
        own=next((e for e in inputs if e['path']==str(source_code_path().resolve())),None)
        if not own or own['sha256']!=sha(source_code_path()):continue
        cal=next((e for e in inputs if e['path']==str(calendar_file().resolve())),None)
        if not cal or cal['sha256']!=sha(calendar_file()):continue
        if any(e.get('prefix_sha256')!=day_prefix_sha(e['path'],p) for e in inputs if e['path'] in days):continue
        return verify_cache(complete.parent,identity)
    return None

def verify_cache(folder,identity):
    value=json.loads((folder/'COMPLETE.json').read_bytes())
    if value['identity']!=identity:raise ValueError('TDX_DAILY_CACHE_IDENTITY')
    for entry in value['parts']:
        if sha(folder/entry['file'])!=entry['sha256']:raise ValueError('TDX_DAILY_CACHE_SHA')
    return folder

def _prune_day_versions(day_parent, keep):
    """Retention: one verified version per day. Older redundant COMPLETE
    versions of the same day are rebuildable caches and are removed; the kept
    folder and any in-flight .building dirs are never touched."""
    keep = Path(keep).resolve()
    # Never delete the only complete version when the alleged keeper is absent.
    if keep.parent != Path(day_parent).resolve() or not (keep/'COMPLETE.json').is_file():
        return
    from persistent_records_v1 import is_record
    if is_record(day_parent):
        return  # Historical selection inputs are user records, not disposable cache.
    for sibling in sorted(day_parent.glob('*')):
        if sibling.resolve() == keep or '.building.' in sibling.name:
            continue
        if not (sibling / 'COMPLETE.json').exists():
            continue
        try:
            for child in sibling.rglob('*'):
                if child.is_file():
                    child.unlink()
            for child in sorted(sibling.rglob('*'), reverse=True):
                if child.is_dir():
                    child.rmdir()
            sibling.rmdir()
        except OSError:
            continue  # a partial prune keeps the keeper usable; reader re-verifies


def _publish_daily(folder,target,identity):
    try:
        folder.rename(target)
        return target
    except FileExistsError:
        verify_cache(target,identity)
    # Only dispose of THIS writer's duplicate data, after validating both copies.
    if folder.parent.resolve()!=target.parent.resolve() or not folder.name.startswith(target.name+'.building.'+str(os.getpid())+'.'):
        raise ValueError('TDX_DUPLICATE_WRITER_IDENTITY')
    verify_cache(folder,identity)
    old=json.loads((folder/'COMPLETE.json').read_bytes())
    kept=json.loads((target/'COMPLETE.json').read_bytes())
    if old['parts']!=kept['parts']:raise ValueError('TDX_DUPLICATE_CONTENT_CHANGED')
    expected={e['file'] for e in old['parts']}|{'COMPLETE.json'}
    if {p.name for p in folder.iterdir()}!=expected or any(not p.is_file() for p in folder.iterdir()):
        raise ValueError('TDX_DUPLICATE_UNEXPECTED_FILE')
    receipts=target.parent/'build_receipts';receipts.mkdir(exist_ok=True)
    receipt=receipts/(folder.name+'.json')
    body=(folder/'COMPLETE.json').read_bytes()
    if receipt.exists():
        if receipt.read_bytes()!=body:raise ValueError('TDX_DUPLICATE_RECEIPT_CHANGED')
    else:
        with receipt.open('xb') as stream:stream.write(body)
    for name in expected:(folder/name).unlink()
    folder.rmdir()
    return target


def daily_directory(project,p,_materialize=False):
    """Idempotent one-time postclose preparation, then verified local reuse."""
    p=int(p)
    if p not in calendar_days():raise ValueError('TDX_DAILY_NOT_TRADING_DAY')
    frozen=frozen_directory(project,p)
    if frozen is not None:return frozen
    paths,meta,identity,key=source_inventory(p)
    parent=runtime(project);folder=parent/'tdx_daily_v2'/str(p)/key
    if (folder/'COMPLETE.json').exists():return verify_cache(folder,identity)
    if not _materialize:
        # The GUI only verifies finished files; heavy preparation is external.
        # Precheck BEFORE launching the worker: generation reserve + monitor
        # reserve must fit, otherwise stop cleanly instead of leaving a
        # half-written .building version behind.
        sys.path.insert(0,str(Path(project)/'stargazer_ml/v5_hybrid/risk_r0_v1'))
        from runtime_disk_limit_v1 import disk_limit, disk_usage
        parent_runtime=runtime(project)
        reserve=128*1024**2+32*1024**2
        from persistent_records_v1 import is_record,reserve_space
        persistent=is_record(parent_runtime/'tdx_daily_v2')
        if persistent:reserve_space(parent_runtime/'tdx_daily_v2',reserve)
        if not persistent and disk_limit(parent_runtime)-disk_usage(parent_runtime)<reserve:
            raise ValueError('TDX_DAILY_DISK_PRECHECK:运行缓存空间不足以生成日线版本'
                             '（需 %.0f MiB 预留）；请先清理可重建缓存，不留下半成品'%(reserve/1048576))
        import subprocess
        python=Path(project)/'_dev_venv/Scripts/python.exe'
        if not python.is_file():python=Path(project)/'rule_venv/Scripts/python.exe'
        if not python.is_file():raise ValueError('TDX_DAILY_WORKER_PYTHON_MISSING')
        proc=subprocess.run([str(python),'-B','-X','utf8',str(Path(project)/'stargazer_desktop/tdx_daily_source_v2.py'),
            '--project',str(project),'--prepare',str(p)],capture_output=True,timeout=240,
            env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if proc.returncode:raise ValueError('TDX_DAILY_PREPARE_FAILED:'+proc.stderr.decode('utf8',errors='replace')[-250:])
        verified=verify_cache(folder,identity)
        _prune_day_versions(parent_runtime/'tdx_daily_v2'/str(p),verified)
        return verified
    import pandas as pd
    sys.path.insert(0,str(Path(project)/'stargazer_ml/v5_hybrid/risk_r0_v1'))
    from tdx_corporate_events_v1 import budget
    # 128 MiB covers a version of the current one-year native equity universe.
    budget(parent/'tdx_daily_v2',128*1024**2)
    hq=root().parent/'T0002/hq_cache'
    names={}
    for m in ('sh','sz','bj'):
        names.update({m+c:n for c,n in names_from_tnf(hq/(m+'s.tnf')).items()})
    bases=base_records(hq/'base.dbf');events=events_by_code(hq/'gbbq')
    snapshot_day=int(datetime.fromtimestamp((hq/'shs.tnf').stat().st_mtime).strftime('%Y%m%d'))
    if snapshot_day<p:raise ValueError('TDX_NAMES_SNAPSHOT_STALE')
    # Operational latest P may use a later local snapshot; historical rebuilds
    # retain unknown ST rather than backfilling a current name into a past label.
    current=max(d for d in calendar_days() if d<=int(datetime.now().strftime('%Y%m%d')))
    metadata_day=p if p==current else snapshot_day
    target_folder=folder
    folder=folder.with_name(key+'.building.'+str(os.getpid())+'.'+datetime.now().strftime('%H%M%S%f'))
    folder.mkdir(parents=True);excluded=[];parts=[];summary=[];batch=[];batch_qfq=[]
    def flush():
        if not batch:return
        n=len(parts)//2
        for filename,frames in [(f'part_{n:04d}.parquet',batch),(f'qfq_{n:04d}.parquet',batch_qfq)]:
            target=folder/filename
            pd.concat(frames,ignore_index=True).to_parquet(target,index=False,compression='zstd')
            parts.append(dict(file=filename,sha256=sha(target),bytes=target.stat().st_size))
        batch.clear();batch_qfq.clear()
    expected={e['path']:e['sha256'] for e in identity['inputs']}
    for path in sorted(paths):
        try:
            frame,receipt=read_day(path,p)
            if receipt['sha256']!=expected[str(path.resolve())]:raise ValueError('TDX_DAY_SOURCE_CHANGED')
            if frame.empty:raise ValueError('NO_DAILY_PREFIX')
            raw,qfq,applied=normalized(frame,path.stem,names.get(path.stem),bases.get(path.stem),
                events.get(path.stem,[]),p,metadata_day)
            batch.append(raw);qfq['code']=path.stem[2:];batch_qfq.append(qfq)
            summary.append(dict(code=path.stem[2:],rows=len(raw),last_date=int(raw.date.iloc[-1]),
                st_known=bool(pd.notna(raw.is_st.iloc[-1])),share_known=bool(pd.notna(raw.float_share.iloc[-1])),
                price_event_count=len(applied)))
            if len(batch)>=64:flush()
        except ValueError as exc:
            if 'SOURCE_CHANGED' in str(exc):raise
            excluded.append(dict(code=path.stem[2:],reason=str(exc)))
    flush()
    # Revalidate ALL source bytes after construction, not merely successful files.
    for e in identity['inputs']:
        if sha(e['path'])!=e['sha256']:raise ValueError('TDX_DAILY_SOURCE_CHANGED')
    if not summary or not any(r['last_date']==p for r in summary):
        raise ValueError('TDX_CURRENT_DAILY_NOT_READY')
    value=dict(identity=identity,parts=parts,stocks=summary,excluded=excluded,
        peak_bytes=peak_memory(),
        available_at=datetime.now().isoformat(),availability_basis='LOCAL_READ_NOT_UPSTREAM_FIRST_ARRIVAL',
        model_price_basis='RAW_OHLC_EVENT_ADJUSTED_PRE_CLOSE',qfq_basis='P_FIXED_CATEGORY1_PROPORTIONAL_SEPARATE_VIEW',
        qfq_vendor_display_certified=False,metadata_basis='CURRENT_TDX_NAME_ST_AND_BASE_SNAPSHOT_AT_P_ONLY',
        historical_shares_basis='GBBQ_DATED_POST_EVENT_NOT_PUBLICATION_TIME_CERTIFIED',
        historical_training_eligible=False,network=0)
    raw=json.dumps(value,ensure_ascii=False,separators=(',',':')).encode()
    budget(parent/'tdx_daily_v2',len(raw));(folder/'COMPLETE.json').open('xb').write(raw)
    return _publish_daily(folder,target_folder,identity)

def read_prefix(code,p,columns=None):
    """Single stock price tool, same source and cutoff, no all-market warmup."""
    import pandas as pd
    if MODEL_PREFIX.fullmatch(str(code)):
        project=source_code_path().parents[1]
        prepared=frozen_directory(project,int(p))
        if prepared is None:
            _,_,identity,key=source_inventory(int(p))
            candidate=runtime(project)/'tdx_daily_v2'/str(p)/key
            if (candidate/'COMPLETE.json').exists():prepared=verify_cache(candidate,identity)
        if prepared is not None:
            import pyarrow.dataset as ds
            dataset=ds.dataset([str(f) for f in prepared.glob('part_*.parquet')],format='parquet')
            frame=dataset.to_table(filter=(ds.field('code')==str(code))&(ds.field('date')<=int(p))).to_pandas()
            if frame.empty:raise ValueError('TDX_SECURITY_NOT_IN_PREPARED_DAILY:'+str(code))
            frame=frame.sort_values('date').reset_index(drop=True)
            frame.attrs['source_receipt']=dict(contract=CONTRACT,prepared_manifest=str(prepared/'COMPLETE.json'),
                manifest_sha256=sha(prepared/'COMPLETE.json'),metadata_basis='P_FROZEN_SNAPSHOT',historical_training_eligible=False)
            return frame if columns is None else frame[columns]
    path=day_path(code);frame,receipt=read_day(path,int(p))
    if frame.empty:raise ValueError('TDX_DAILY_PREFIX_MISSING')
    hq=root().parent/'T0002/hq_cache';name=names_from_tnf(hq/(path.stem[:2]+'s.tnf')).get(path.stem[2:])
    bases=base_records(hq/'base.dbf');events=events_by_code(hq/'gbbq').get(path.stem,[])
    snapshot_day=int(datetime.fromtimestamp((hq/(path.stem[:2]+'s.tnf')).stat().st_mtime).strftime('%Y%m%d'))
    current=max(d for d in calendar_days() if d<=int(datetime.now().strftime('%Y%m%d')))
    raw,_,_=normalized(frame,path.stem,name,bases.get(path.stem),events,int(p),int(p) if int(p)==current and snapshot_day>=p else snapshot_day)
    raw.attrs['source_receipt']=dict(receipt,price_basis='TDX_RAW_WITH_EVENT_REFERENCE_CLOSE',
        metadata_basis='CURRENT_LOCAL_SNAPSHOT_NOT_HISTORICAL_ST_CERTIFICATE',historical_training_eligible=False)
    return raw if columns is None else raw[columns]

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--prepare',type=int,required=True);args=parser.parse_args()
    result=daily_directory(args.project,args.prepare,_materialize=True)
    print(json.dumps(dict(directory=str(result),peak_bytes=peak_memory(),fit=0,http=0)))
