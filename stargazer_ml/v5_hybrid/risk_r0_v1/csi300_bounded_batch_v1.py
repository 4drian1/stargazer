"""One approved CSI300 batch; exact Shanghai index symbol, no retry or ETF fallback."""
import argparse
import io
import csv
import json
from pathlib import Path
from datetime import datetime,timezone,timedelta
import urllib.parse

from independent_fetch_client import TencentDailyClient,FetchCounter,ByteCounter,_validate_raw
from price_verification_batch_44win import tencent_rows_to_map
from price_verification_guarded_run import encoded,sha
from research_budget_v2 import TOTAL_CAP,bytes_used,ADDENDUM

HERE=Path(__file__).resolve().parent
SYMBOL='sh000300'
PERIODS=[('20180102','20191231'),('20200101','20211231'),('20220101','20231231'),
         ('20240101','20251231'),('20260101','20261014')]
CAP=2*1024**2
RESERVE=64*1024
LIMITS=dict(requests=6,response_bytes=CAP,disk_bytes=CAP,retries=0)
MANIFEST=HERE/'csi300_batch_manifest_20261015.json'
AUTH=HERE/'csi300_batch_authorization_20261015.json'
OUTPUT=HERE/'independent_fetch/batch_csi300_price_index_20261015_v1'


def fetch_segment(client,start,end,output):
    param=f'{SYMBOL},day,{start[:4]}-{start[4:6]}-{start[6:]},{end[:4]}-{end[4:6]}-{end[6:]},640,'
    url=client.BASE+'?'+urllib.parse.urlencode(dict(param=param))
    client.counter.consume(1);client._throttle()
    fetched=datetime.now(timezone.utc).isoformat()
    raw=b''
    try:
        raw=client._transport(url,client.timeout_s,client.bytes.remaining)
        client.received_total+=len(raw);client.bytes.consume(len(raw))
        payload=json.loads(raw.decode('utf-8'))
        rows=_validate_raw(payload,SYMBOL)
        identity=payload['data'][SYMBOL].get('qt',{}).get(SYMBOL,[])
        if len(identity)<3 or '沪深300' not in str(identity[1]) or str(identity[2])!='000300':
            raise ValueError('INDEX_IDENTITY_NAME_MISMATCH')
        parsed=tencent_rows_to_map(rows)
    except Exception as exc:
        partial=getattr(exc,'received_bytes',b'')
        if partial:
            raw=partial;client.received_total+=len(partial)
        try:client._archive(output,SYMBOL,param,url,fetched,raw,'failed:'+type(exc).__name__)
        except Exception:
            client.last_failed_response=dict(bytes=len(raw),sha256=__import__('hashlib').sha256(raw).hexdigest(),
                                             archive_status='UNABLE_TO_ARCHIVE_WITHIN_BUDGET')
        raise
    client._archive(output,SYMBOL,param,url,fetched,raw,'ok')
    return parsed,dict(start=start,end=end,raw_rows=len(rows),returned_first=min(parsed),
                       returned_last=max(parsed),identity_name=identity[1])


def merge_required(all_prices,parsed,required):
    for d,row in parsed.items():
        if d in all_prices and all_prices[d]!=row:raise ValueError('OVERLAPPING_DATE_CONFLICT')
    all_prices.update(parsed)
    missing=[d for d in required if d not in parsed]
    if missing:raise ValueError('REQUIRED_DATES_MISSING:'+','.join(map(str,missing)))


def prepare():
    calendar=HERE/'historical_stock_day_preparation_v1/manifest.json'
    dates=json.loads(calendar.read_bytes())['calendar_dates']
    dates=[d for d in dates if 20180102<=d<=20261014]
    inputs=[calendar,Path(__file__),HERE/'test_csi300_bounded_batch_v1.py',
            HERE/'independent_fetch_client.py',HERE/'price_verification_batch_44win.py',
            HERE/'price_verification_guarded_run.py',HERE/'research_budget_v2.py',ADDENDUM]
    m=dict(schema='CSI300_BOUNDED_BATCH_V1',symbol=SYMBOL,periods=PERIODS,limits=LIMITS,
           required_dates=dates,output=str(OUTPUT),
           inputs=[dict(path=str(p.resolve()),sha256=sha(p)) for p in inputs])
    now=datetime.now(timezone.utc)
    data=encoded(m)
    auth=dict(scope='INDEPENDENT_FETCH_CSI300_V1',status='CHAT_APPROVED',
              approval_reference='用户在4GiB及沪深300单批最多6请求提案后回复：好的，请继续',
              record_type='CHAT_APPROVAL_RECORD_NOT_CRYPTOGRAPHIC_SIGNATURE',
              authorized_at=now.isoformat(),expires_at=(now+timedelta(hours=2)).isoformat(),
              manifest_sha256=__import__('hashlib').sha256(data).hexdigest(),output=str(OUTPUT),limits=LIMITS)
    if MANIFEST.exists() or AUTH.exists() or OUTPUT.exists():raise ValueError('BATCH_CONTROL_ALREADY_EXISTS')
    if bytes_used()+len(data)+len(encoded(auth))+CAP>TOTAL_CAP:raise ValueError('REGISTERED_BUDGET')
    with MANIFEST.open('xb') as f:f.write(data)
    with AUTH.open('xb') as f:f.write(encoded(auth))
    print(json.dumps(dict(prepared=True,manifest_sha256=sha(MANIFEST),authorization_sha256=sha(AUTH),
                          dates=len(dates),network_requests=0)))


def preflight(manifest=MANIFEST,authorization=AUTH,auth_sha=None):
    if not auth_sha or sha(authorization)!=auth_sha:raise ValueError('AUTHORIZATION_HASH_REQUIRED')
    m=json.loads(Path(manifest).read_bytes());a=json.loads(Path(authorization).read_bytes())
    now=datetime.now(timezone.utc)
    if (a.get('status')!='CHAT_APPROVED' or a.get('scope')!='INDEPENDENT_FETCH_CSI300_V1'
            or not a.get('approval_reference') or a.get('manifest_sha256')!=sha(manifest)
            or a.get('limits')!=LIMITS or m.get('limits')!=LIMITS or m.get('symbol')!=SYMBOL
            or m.get('periods')!=[list(p) for p in PERIODS]):raise ValueError('AUTHORIZATION_BINDING')
    if not datetime.fromisoformat(a['authorized_at'])<=now<datetime.fromisoformat(a['expires_at']):
        raise ValueError('AUTHORIZATION_EXPIRED')
    if m['output']!=str(OUTPUT) or a['output']!=str(OUTPUT):raise ValueError('OUTPUT_BINDING')
    if OUTPUT.exists():raise ValueError('BATCH_CLAIMED_NO_RESTART')
    for x in m['inputs']:
        if sha(x['path'])!=x['sha256']:raise ValueError('INPUT_DRIFT')
    if bytes_used()+CAP>TOTAL_CAP:raise ValueError('REGISTERED_BUDGET')
    return m


def execute(auth_sha):
    m=preflight(auth_sha=auth_sha)
    control_bytes=MANIFEST.stat().st_size+AUTH.stat().st_size
    disk=ByteCounter(CAP-RESERVE-control_bytes)
    client=TencentDailyClient(FetchCounter(6),ByteCounter(CAP),max_retries=0,
                              disk_counter=disk,timeout_s=10,rate_limit_s=1)
    OUTPUT.mkdir(exist_ok=False)
    opening=encoded(dict(manifest_sha256=sha(MANIFEST),authorization_sha256=sha(AUTH)))
    disk.consume(len(opening))
    with (OUTPUT/'batch_open.json').open('xb') as f:f.write(opening)
    prices={};ledger=[];error=None
    try:
        for start,end in PERIODS:
            parsed,entry=fetch_segment(client,start,end,OUTPUT)
            ledger.append(entry)
            required=[d for d in m['required_dates'] if int(start)<=d<=int(end)]
            merge_required(prices,parsed,required)
        if any(sha(x['path'])!=x['sha256'] for x in m['inputs']):raise ValueError('INPUT_DRIFT_AFTER_FETCH')
    except (Exception,KeyboardInterrupt) as exc:error=f'{type(exc).__name__}:{exc}'
    blobs={}
    if error is None:
        stream=io.StringIO(newline='');writer=csv.writer(stream);writer.writerow(['date','close'])
        for d in m['required_dates']:writer.writerow([d,prices[d]['close']])
        data=stream.getvalue().encode('utf-8');blobs['csi300_price_index.csv']=data
        blobs['csi300_price_index.csv.binding.json']=encoded(dict(symbol='SH000300',semantics='CSI300_PRICE_INDEX',
              adjustment='NONE',source='Tencent day/sh000300; provider cross-check, not official publication',
              sha256=__import__('hashlib').sha256(data).hexdigest(),rows=len(m['required_dates'])))
    report=dict(status='failed' if error else 'completed',error=error,requests=client.counter.used,
                response_bytes=client.received_total,control_bytes=control_bytes,
                evidence_bytes=disk.used,request_ledger=ledger,required_dates=len(m['required_dates']),
                missing_dates=[d for d in m['required_dates'] if d not in prices],
                failed_response=getattr(client,'last_failed_response',None),training_ready=False,fit_performed=False)
    blobs['stop_record.json']=encoded(report)
    total=sum(len(v) for v in blobs.values())
    if total>RESERVE or control_bytes+disk.used+total>CAP or bytes_used()+total>TOTAL_CAP:
        report=dict(status='failed',error='FINALIZATION_BUDGET_STOP',
                    requests=client.counter.used,training_ready=False)
        blobs={'stop_record.json':encoded(report)}
        error='FINALIZATION_BUDGET_STOP'
    for name,data in blobs.items():
        with (OUTPUT/name).open('xb') as f:f.write(data)
    print(json.dumps(report,ensure_ascii=False))
    return error is None


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--fetch',action='store_true')
    p.add_argument('--authorization-sha256');args=p.parse_args()
    if args.prepare:prepare()
    else:
        try:ok=execute(args.authorization_sha256)
        except Exception as exc:
            print(json.dumps(dict(blocked=str(exc),requests=0)));raise SystemExit(2)
        raise SystemExit(0 if ok else 2)
