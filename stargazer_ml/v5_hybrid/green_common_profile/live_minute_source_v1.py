"""One current-session response per stock/minute, shared by all live lanes."""
from datetime import datetime,timezone,timedelta
from pathlib import Path
import gzip,hashlib,json,time
CONTRACT='LIVE_STOCK_OBSERVED_MINUTES_V1'
MAX_PER_STOCK=240

def snapshot(root,code,P,asof,previous_close,history,budget):
    from public_stock_minutes_v1 import fetch,parse
    from minute_history_v1 import cache_lock
    day=asof//1000000
    if day!=int(datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')):raise ValueError('LIVE_SOURCE_CURRENT_DAY_ONLY')
    folder=Path(root)/'minute_context'/'live'/str(day)/code;key=asof//100
    archive=folder/(str(key)+'.json.gz');meta=archive.with_suffix('.meta.json')
    with cache_lock(folder):
        if archive.exists() and meta.exists():
            receipt=json.loads(meta.read_bytes());body=archive.read_bytes()
            if hashlib.sha256(body).hexdigest()!=receipt['sha256']:raise ValueError('LIVE_CACHE_HASH')
            raw=gzip.decompress(body)
        else:
            ledger=folder/'attempts.json'
            attempts=json.loads(ledger.read_bytes()) if ledger.exists() else dict(count=0,prior=[])
            if not ledger.exists():
                # Preserve the old lane quotas when moving to the shared live store.
                for prior in (Path(root)/'golden_intraday'/str(day)/'public_attempts.json',
                              Path(root)/'strong_sector/minutes'/str(day)/'public_attempts.json'):
                    if prior.exists():
                        value=json.loads(prior.read_bytes());n=value.get(code,0)
                        attempts['count']+=n;attempts['prior'].append(dict(path=str(prior),sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),count=n))
            if attempts['count']>=MAX_PER_STOCK:raise ValueError('LIVE_STOCK_DAILY_REQUEST_LIMIT')
            if attempts.get('last_minute')==key:raise ValueError('LIVE_ATTEMPT_ALREADY_RECORDED_THIS_MINUTE')
            budget(131072+16384)
            attempts['count']+=1;attempts['last_minute']=key
            temporary=ledger.with_suffix('.tmp');temporary.write_text(json.dumps(attempts),encoding='utf8');temporary.replace(ledger)
            raw,url=fetch(code)
            body=gzip.compress(raw,mtime=0)
            receipt=dict(contract=CONTRACT,code=code,request_asof=asof,observed_at=time.time(),
                sha256=hashlib.sha256(body).hexdigest(),url=url,provisional=True)
            budget(len(body)+4096)
            with archive.open('xb') as f:f.write(body)
            with meta.open('x',encoding='utf8') as f:json.dump(receipt,f)
            if hasattr(budget,'written'):budget.written(len(body)+4096)
    values=json.loads(raw)
    if not isinstance(values,list):raise ValueError('LIVE_SOURCE_SCHEMA')
    stamps=[]
    for row in values:
        stamp=int(datetime.strptime(row['day'],'%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M%S'))
        if stamp//1000000==day and stamp<=asof:stamps.append(stamp)
    # Missing *tail* is source delay, not a fictitious gap in already available minutes.
    # Interior omissions remain rejected by the original parser's continuous grid.
    upto=max(stamps,default=day*1000000)
    if upto<=day*1000000:raise ValueError('LIVE_CURRENT_MINUTES_NOT_ARRIVED')
    lag=(datetime.strptime(str(asof),'%Y%m%d%H%M%S')-datetime.strptime(str(upto),'%Y%m%d%H%M%S')).total_seconds()
    if lag>120:raise ValueError('LIVE_CURRENT_MINUTES_STALE')
    current=parse(raw,code,P,upto,previous_close)
    raw_sha256=hashlib.sha256(raw).hexdigest()
    return history+current,dict(receipt,path=str(archive),latest_minute=upto,raw_sha256=raw_sha256),[dict(path=str(archive),sha256=receipt['sha256'],raw_sha256=raw_sha256)]
