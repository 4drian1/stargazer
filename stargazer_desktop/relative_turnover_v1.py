"""Optional regular-minute share turnover reference. Never a trigger or veto."""
from pathlib import Path
from datetime import datetime
from functools import lru_cache
import gzip,hashlib,json,math

CONTRACT='REGULAR_240_SAME_MINUTE_TURNOVER_REFERENCE_V1'
GRID=[m//60*10000+m%60*100 for m in [*range(571,691),*range(781,901)]]
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def finite(x):return type(x) in (int,float) and math.isfinite(x)
def unknown(reason):return dict(status='UNKNOWN',reason=reason,reference_only=True)

def specification(rows,P,calendar):
    """P-only daily metadata; missing daily records are not presumed suspended."""
    by={int(r['date']):r for r in rows if int(r['date'])<=P};history=[]
    if P not in by:return unknown('DAILY_REFERENCE_MISSING')
    for day in sorted((d for d in calendar if d<=P),reverse=True)[:40]:
        r=by.get(day)
        if r is None:return unknown('DAILY_HISTORY_MISSING')
        v,s=r.get('volume'),r.get('float_share')
        if not finite(v) or v<0:return unknown('DAILY_VOLUME_UNKNOWN')
        if v==0:continue
        if not finite(s) or s<=0:return unknown('FLOAT_SHARE_UNKNOWN')
        history.append([day,v,s])
        if len(history)==20:break
    if len(history)!=20:return unknown('HISTORY_NOT_20_NORMAL_DAYS')
    shares=by[P].get('float_share')
    if not finite(shares) or shares<=0:return unknown('REFERENCE_FLOAT_SHARE_UNKNOWN')
    return dict(status='READY',contract=CONTRACT,P=P,shares=shares,history=sorted(history))

def checked_spec(spec):
    if not isinstance(spec,dict) or spec.get('status')!='READY':raise ValueError((spec or {}).get('reason','WAIT_POSTCLOSE_REFERENCE'))
    history=spec.get('history',[])
    if (spec.get('contract')!=CONTRACT or len(history)!=20 or len({x[0] for x in history})!=20 or
        history!=sorted(history) or any(d>spec['P'] or not finite(v) or v<=0 or not finite(s) or s<=0 for d,v,s in history) or
        not finite(spec.get('shares')) or spec['shares']<=0):raise ValueError('REFERENCE_SPEC_INVALID')
    return history

def curve(spec,code,rows):
    history=checked_spec(spec);days={d for d,_,_ in history};table={}
    for r in rows:
        t=int(r['date']);d=t//1000000
        if d not in days:continue
        if str(r.get('code')).zfill(6)!=code:raise ValueError('HISTORY_IDENTITY')
        if t in table and any(table[t].get(k)!=r.get(k) for k in ('open','high','low','close','volume','amount')):
            raise ValueError('HISTORY_OVERLAP_CONFLICT')
        table[t]=r
    result=[0.]*240
    for day,reported,shares in history:
        values={t%1000000:r for t,r in table.items() if t//1000000==day}
        if set(values) not in (set(GRID),set(GRID)|{93000},set(GRID)|{130000},set(GRID)|{93000,130000}):raise ValueError('HISTORY_MINUTE_GAP')
        if any(not finite(r.get('volume')) or r['volume']<0 for r in values.values()):raise ValueError('HISTORY_VOLUME_UNKNOWN')
        total=sum(r['volume'] for r in values.values())
        if abs(total-reported)>max(100.,reported*.005):raise ValueError('HISTORY_VOLUME_MISMATCH')
        # Include opening observations for total-volume QC, not in either curve.
        cumulative=0.
        for i,t in enumerate(GRID):
            cumulative+=values[t]['volume'];result[i]+=cumulative/shares*100/20
    return result

@lru_cache(maxsize=1)
def manifest(path,mtime,size):
    if size>16*1024**2:raise ValueError('SOURCE_MANIFEST_SIZE')
    return json.loads(Path(path).read_bytes())['entries']

def local_history(base,code,days):
    """Read existing bytes only; never start stockdb, fetch, or retry a source."""
    from minute_history_v1 import CONTRACT as HISTORY_CONTRACT
    import msgpack
    values=[];refs=[];missing=[]
    def unpack(path,expected):
        path=Path(path)
        if path.stat().st_size>32*1024**2:raise ValueError('HISTORY_FILE_SIZE')
        body=path.read_bytes()
        if hashlib.sha256(body).hexdigest()!=expected:raise ValueError('HISTORY_SOURCE_SHA')
        with gzip.GzipFile(fileobj=__import__('io').BytesIO(body)) as f:raw=f.read(32*1024**2+1)
        if len(raw)>32*1024**2:raise ValueError('HISTORY_RAW_SIZE')
        rows=msgpack.unpackb(raw,raw=False,strict_map_key=False)
        if not isinstance(rows,list):raise ValueError('HISTORY_RESPONSE_SHAPE')
        refs.append(dict(path=str(path),sha256=expected))
        return [r[1] if isinstance(r,list) and len(r)==2 and isinstance(r[1],dict) else r for r in rows]
    for day in days:
        folder=Path(base)/'minute_context'/str(day)/code
        meta=folder/'source.json';file=folder/'minutes.msgpack.gz'
        if not meta.exists() or not file.exists():missing.append(day);continue
        m=json.loads(meta.read_bytes())
        if (m.get('contract'),m.get('code'),m.get('day'))!=(HISTORY_CONTRACT,code,day):raise ValueError('HISTORY_CACHE_KEY')
        values.extend(unpack(file,m['sha256']))
    if missing:
        state_file=Path(base)/'state.json'
        state=json.loads(state_file.read_bytes()) if state_file.exists() else {}
        if state.get('status')!='COMPLETE' or state.get('acceptance_status')!='PASS':raise ValueError('WAIT_VERIFIED_HISTORY')
        run=Path(state['run_dir']).resolve()
        if Path(base).resolve() not in run.parents:raise ValueError('ARCHIVE_OUTSIDE_RUNTIME')
        path=run/'source_manifest.json';stat=path.stat();entry=manifest(str(path),stat.st_mtime_ns,stat.st_size).get(code,{})
        months={str(d)[:6] for d in missing}
        for ref in entry.get('extensions',[]):
            if str(ref.get('period',''))[:6] in months:
                values.extend(r for r in unpack(ref['path'],ref['sha256']) if int(r['date'])//1000000 in missing)
    return values,refs

def cache_path(base,code,spec):return Path(base)/'minute_context/turnover_display_v1'/str(spec['P'])/(code+'_'+digest(spec)+'.json.gz')

def read_cache(base,code,spec):
    checked_spec(spec);path=cache_path(base,code,spec);proof=path.with_suffix('.sha256')
    if not path.exists() or not proof.exists():raise ValueError('WAIT_POSTCLOSE_REFERENCE')
    body=path.read_bytes()
    if len(body)>131072 or hashlib.sha256(body).hexdigest()!=proof.read_text(encoding='ascii'):raise ValueError('TURNOVER_CACHE_SHA')
    value=json.loads(gzip.decompress(body))
    if (value.get('contract'),value.get('reference_sha'),value.get('code'))!=(CONTRACT,digest(spec),code):raise ValueError('TURNOVER_CACHE_KEY')
    if len(value.get('means',[]))!=240 or any(not finite(x) or x<0 for x in value['means']):raise ValueError('TURNOVER_CACHE_CURVE')
    return value

def prepare(base,code,spec,budget):
    try:
        checked_spec(spec);path=cache_path(base,code,spec)
        if path.exists():read_cache(base,code,spec);return dict(status='READY',reference_only=True)
        rows,refs=local_history(base,code,[d for d,_,_ in spec['history']])
        value=dict(contract=CONTRACT,code=code,reference_sha=digest(spec),means=curve(spec,code,rows),sources=refs)
        body=gzip.compress(json.dumps(value,separators=(',',':'),allow_nan=False).encode(),mtime=0)
        budget(len(body)+4096);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(body)
        path.with_suffix('.sha256').write_text(hashlib.sha256(body).hexdigest(),encoding='ascii')
        if hasattr(budget,'written'):budget.written(len(body)+4096)
        return dict(status='READY',reference_only=True)
    except (OSError,ValueError,KeyError,TypeError) as exc:return unknown(str(exc)[:100])

def current(base,code,spec,minutes,asof):
    try:
        value=read_cache(base,code,spec);day=asof//1000000;table={}
        for r in minutes:
            stamp=int(r['date']);clock=stamp%1000000
            if stamp//1000000!=day or stamp>asof or clock in (93000,130000):continue
            if str(r.get('code')).zfill(6)!=code or clock not in GRID or clock in table:raise ValueError('CURRENT_IDENTITY_OR_DUPLICATE')
            v,a=r.get('volume'),r.get('amount');lo,hi=r.get('low'),r.get('high')
            if not all(finite(x) for x in (v,a,lo,hi)) or min(v,a)<0 or (v==0)!=(a==0) or (v and not lo*.99<=a/v<=hi*1.01):raise ValueError('CURRENT_SHARE_BASIS_UNKNOWN')
            table[clock]=r
        if not table:raise ValueError('WAIT_CURRENT_MINUTE')
        last=max(table)
        if sorted(table)!=[t for t in GRID if t<=last]:raise ValueError('CURRENT_MINUTE_GAP')
        evidence=day*1000000+last
        if (datetime.strptime(str(asof),'%Y%m%d%H%M%S')-datetime.strptime(str(evidence),'%Y%m%d%H%M%S')).total_seconds()>120:raise ValueError('CURRENT_MINUTE_STALE')
        mean=value['means'][GRID.index(last)]
        if mean<=0:raise ValueError('HISTORY_MEAN_ZERO')
        turnover=sum(r['volume'] for r in table.values())/spec['shares']*100
        return dict(status='KNOWN',ratio=turnover/mean,asof=evidence,reference_day=spec['P'],reference_only=True,
            contract=CONTRACT,provisional=True,turnover_pct=turnover,history_mean_pct=mean,
            note='240连续竞价分钟；不含单列开盘记录；股本取前收盘；跨源参考、不作过滤')
    except (OSError,ValueError,KeyError,TypeError) as exc:return unknown(str(exc)[:100])
