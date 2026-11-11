"""Two public index OHLC prefixes; unverified activity never enters model input."""
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import urllib.request

SYMBOLS={'000001.SH':('sh000001','sse'),'399006.SZ':('sz399006','cyb')}
CONTRACT='SINA_INDEX_5M_PRICE_ONLY_V1'
TIME='MIXED_TDX_HISTORY_SINA_5MIN_END_ASSUMED_NOT_CERTIFIED'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf8')
    temp.replace(path)


def read(path):
    try:return json.loads(Path(path).read_bytes())
    except FileNotFoundError:return {}


def budget(root,size):
    from runtime_disk_limit_v1 import disk_limit,disk_usage
    if disk_usage(root)+size+65536>disk_limit(root):
        raise ValueError('RUNTIME_DISK_BUDGET')


def parse(raw,symbol,as_of,updater):
    if symbol not in SYMBOLS:raise ValueError('PUBLIC_INDEX_IDENTITY')
    values=json.loads(raw)
    if not isinstance(values,list) or len(values)>240:raise ValueError('PUBLIC_INDEX_SCHEMA')
    day=as_of//1000000;rows=[]
    for item in values:
        stamp=int(datetime.strptime(item['day'],'%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M%S'))
        if stamp//1000000>day:raise ValueError('PUBLIC_INDEX_FUTURE_DATE')
        if stamp//1000000!=day:continue  # Never overwrite TDX history from another provider.
        o,h,l,c=[float(item[k]) for k in ('open','high','low','close')]
        if not all(math.isfinite(x) and x>0 for x in (o,h,l,c)) or not l<=min(o,c)<=max(o,c)<=h:
            raise ValueError('PUBLIC_INDEX_OHLC')
        rows.append(dict(ts=stamp,day=day,open=o,high=h,low=l,close=c,amount=None,raw_volume=None))
    grid=updater.grid(day)
    complete=[t for t in grid if t<=as_of]
    cutoff=max(complete) if complete else day*1000000
    verdict=updater.validate_prefix(rows,day,cutoff,as_of)
    if verdict['verdict'] not in ('OK_CURRENT_PREFIX','WAIT_FRESHNESS','POSTCLOSE_OK'):
        raise ValueError('PUBLIC_INDEX_'+verdict['verdict'])
    return [r for r in rows if r['ts']<=as_of],verdict


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def fetch(symbol):
    code=SYMBOLS[symbol][0]
    url=('https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData?'
         'symbol='+code+'&scale=5&ma=no&datalen=240')
    opener=urllib.request.build_opener(NoRedirect())
    with opener.open(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=6) as response:
        raw=response.read(131073)
    if len(raw)>131072:raise ValueError('PUBLIC_INDEX_RESPONSE_LIMIT')
    return raw,url,int(datetime.now().strftime('%Y%m%d%H%M%S'))


def watermark(labels,as_of):
    day=as_of//1000000;clock=as_of%1000000
    close=clock>=150500 and len(labels)==48 and labels[-1]==day*1000000+150000
    lunch=113500<=clock<130500 and len(labels)==24 and labels[-1]==day*1000000+113000
    if close or lunch:
        return labels[-1],('POSTCLOSE' if close else 'MIDDAY')+'_SETTLING_ASSUMPTION_NOT_CERTIFIED'
    return max((x for x in labels[:-1] if x<=as_of),default=0),'FOLLOWING_SOURCE_INTERVAL_OBSERVED; LAST_INTERVAL_PROVISIONAL'


def sync_public(updater,mirror,root,as_of,local):
    """Called inside the desktop sync lock. Poll once/minute, no immediate retry."""
    moment=datetime.strptime(str(as_of),'%Y%m%d%H%M%S');hm=moment.hour*60+moment.minute
    if moment.weekday()>=5 or not 575<=hm<=920:return local
    day=as_of//1000000;folder=root/'index_public_feed'/str(day)
    poll=folder/'poll.json';previous=read(poll)
    if previous.get('minute')==as_of//100:return previous['result']
    folder.mkdir(parents=True,exist_ok=True)
    rep=dict(symbols={},release_status=dict(as_of=as_of,as_of_date=day,symbols={},evidence_complete=False,
        runtime_fingerprint=updater.get_runtime_fingerprint(),generated_at=moment.isoformat()))
    release=rep['release_status'];expected=[t for t in updater.grid(day) if t<=as_of]
    cutoff=max(expected) if expected else day*1000000
    for symbol,(_,name) in SYMBOLS.items():
        path=mirror/(name+'_5min.parquet');meta_path=path.with_suffix('.meta.json');meta=read(meta_path)
        try:
            # A fresh complete prefix needs no extra network read.
            stamp=meta.get('last_complete_ts') or 0
            if stamp//1000000!=day or stamp<cutoff:
                budget(root,2*1024**2)
                attempted=dict(symbol=symbol,requested_at=as_of,contract=CONTRACT)
                with (folder/'attempts.jsonl').open('a',encoding='utf8') as stream:
                    stream.write(json.dumps(attempted)+'\n')
                raw,url,observed=fetch(symbol)
                sha=hashlib.sha256(raw).hexdigest();archive=folder/(name+'_'+sha+'.json.gz')
                if not archive.exists():
                    with archive.open('xb') as stream:stream.write(gzip.compress(raw,mtime=0))
                # Parsing uses request observation boundary, never response time to admit a newer bar.
                rows,check=parse(raw,symbol,as_of,updater)
                if not rows:raise ValueError('PUBLIC_INDEX_NO_COMPLETE_BAR')
                if path.exists():
                    before=path.read_bytes();old_sha=hashlib.sha256(before).hexdigest()
                    if old_sha!=meta.get('sha256'):raise ValueError('PUBLIC_INDEX_OLD_MIRROR_HASH')
                    # One retained backup per pre-public source version, not one per poll.
                    if meta.get('source_contract')!=CONTRACT:
                        backup=folder/(name+'_before_public_'+old_sha+'.parquet')
                        if not backup.exists():shutil.copy2(path,backup)
                        write(backup.with_suffix('.meta.json'),meta)
                prior_sources=meta.get('historical_sources',meta.get('sources',[]))
                source=dict(provider='SINA_PUBLIC',url=url,sha256=sha,raw_archive=str(archive),
                    requested_at=as_of,available_at=observed,availability_basis='LOCAL_READER_OBSERVED_RESPONSE_NOT_UPSTREAM_FIRST_ARRIVAL',
                    identity_basis='REQUEST_URL_BOUND',volume_verified=False)
                history_days=[]
                if meta.get('source_contract')==CONTRACT:
                    from index_history_completion_v1 import completion
                    table,_=updater.load_mirror(symbol,mirror_dir=mirror)
                    historical,history_days=completion(raw,table.to_pylist() if table is not None else [],day,updater.grid)
                    rows=historical+rows
                updater.merge_and_write(symbol,rows,[source],{},mirror_dir=mirror,as_of=as_of,current_day=day,
                    day_release=check['fully_closed'],last_complete_ts=rows[-1]['ts'],
                    pending_tail_end=min(check['in_progress'],default=None),time_semantics=TIME,
                    public_revision=dict(day=day,archive_dir=str(folder),history_days=history_days) if meta.get('source_contract')==CONTRACT else None)
                meta=read(meta_path);meta.update(source_contract=CONTRACT,historical_sources=prior_sources,
                    activity_policy='PUBLIC_VOLUME_AND_AMOUNT_WITHHELD_NULL',available_at=observed)
                labels=sorted(int(datetime.strptime(x['day'],'%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M%S'))
                    for x in json.loads(raw) if x['day'].startswith(str(day)[:4]+'-'+str(day)[4:6]+'-'+str(day)[6:]))
                confirmed,basis=watermark(labels,as_of)
                meta.update(last_observed_ts=rows[-1]['ts'],last_complete_ts=confirmed,completion_basis=basis)
                write(meta_path,meta)
            else:
                import pyarrow.parquet as pq
                if hashlib.sha256(path.read_bytes()).hexdigest()!=meta.get('sha256'):
                    raise ValueError('PUBLIC_INDEX_MIRROR_HASH')
                rows=pq.read_table(path,columns=['ts','day']).to_pylist()
                check=updater.validate_prefix(rows,day,cutoff,as_of)
            complete=check['verdict']=='POSTCLOSE_OK' and int(meta.get('last_complete_ts') or 0)>=day*1000000+150000
            rep['symbols'][symbol]=dict(status=check['verdict'],rows=len(expected),source_contract=meta.get('source_contract','TDX'))
            release['symbols'][symbol]=dict(symbol=symbol,verdict=check['verdict'],release=complete,
                last_complete_ts=meta.get('last_complete_ts'),mirror_path=str(path),mirror_sha=meta['sha256'])
        except Exception as exc:
            reason=str(exc) if isinstance(exc,ValueError) else type(exc).__name__
            rep['symbols'][symbol]=dict(status='PUBLIC_SOURCE_WAIT',reason=reason)
            release['symbols'][symbol]=dict(symbol=symbol,verdict='PUBLIC_SOURCE_WAIT',release=False)
    release['evidence_complete']=all(x.get('release') for x in release['symbols'].values())
    budget(root,16384)
    write(poll,dict(minute=as_of//100,result=rep))
    write(mirror/'release_status.json',release)
    write(mirror/'runtime_fingerprint.json',dict(runtime_fingerprint=updater.get_runtime_fingerprint(),generated_at=moment.isoformat()))
    fresh=read(mirror/'freshness.json');dates=[read(mirror/(name+'_5min.meta.json')).get('last_date') for _,name in SYMBOLS.values()]
    last=min(dates) if all(dates) else None
    fresh.update(last_date=last,today_trading_day=day,updated_at=moment.isoformat())
    if last==day:fresh.update(stale=False,max_lag_days=0)
    write(mirror/'freshness.json',fresh)
    return rep
