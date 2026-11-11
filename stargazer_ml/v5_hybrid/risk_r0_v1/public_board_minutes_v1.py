"""Explicit RAW END_5M public inputs. Acquisition requires a caller quota gate.

Eastmoney responds per security here, not as an assumed batch endpoint. This
module does not poll, retry, change existing live lanes, or authorize requests.
"""
from datetime import datetime
import hashlib
import json
import math
from urllib.parse import urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler

from tdx_native_reader_v1 import grid_5min

CONTRACT='EASTMONEY_RAW_END_5M_V1'
MAX_BYTES=65536


def security(code):
    if len(code)!=6 or not code.isdigit():raise ValueError('EM_SECURITY_IDENTITY')
    if code.startswith(('0','3')):return '0.'+code,0
    if code.startswith('6'):return '1.'+code,1
    raise ValueError('EM_EXCHANGE_UNVERIFIED:'+code)


def parse(raw,code,previous_day,cutoff,previous_close):
    _,market=security(code)
    value=json.loads(raw);data=value.get('data')
    if value.get('rc')!=0 or not isinstance(data,dict):raise ValueError('EM_MINUTE_RESPONSE')
    if data.get('code')!=code or data.get('market')!=market:raise ValueError('EM_MINUTE_IDENTITY')
    klines=data.get('klines')
    if not isinstance(klines,list) or len(klines)>100:raise ValueError('EM_MINUTE_SHAPE')
    day=cutoff//1000000;current=[];prior=[];last=0
    for line in klines:
        parts=line.split(',')
        if len(parts)!=7:raise ValueError('EM_MINUTE_FIELDS')
        ts=int(datetime.strptime(parts[0],'%Y-%m-%d %H:%M').strftime('%Y%m%d%H%M%S'))
        if ts<=last:raise ValueError('EM_MINUTE_ORDER_OR_DUPLICATE')
        last=ts
        if ts>cutoff:continue
        o,c,h,l,lots,amount=map(float,parts[1:])
        if not all(math.isfinite(v) for v in (o,c,h,l,lots,amount)) or min(o,c,h,l)<=0 or min(lots,amount)<0 or not l<=min(o,c)<=max(o,c)<=h:
            raise ValueError('EM_MINUTE_OHLCV')
        volume=lots*100
        if (volume==0)!=(amount==0) or (volume and not l*.99<=amount/volume<=h*1.01):
            raise ValueError('EM_MINUTE_VOLUME_BASIS')
        row=dict(code=code,date=ts,open=o,high=h,low=l,close=c,volume=volume,amount=amount,
                 source_frequency_minutes=5,source_clock='END_5M')
        if ts//1000000==day:current.append(row)
        elif ts//1000000==previous_day:prior.append(row)
        else:raise ValueError('EM_MINUTE_DAY')
    if [r['date'] for r in prior]!=grid_5min(previous_day):raise ValueError('EM_PREVIOUS_SESSION_INCOMPLETE')
    if abs(prior[-1]['close']-previous_close)>.011:raise ValueError('EM_RAW_REFERENCE_MISMATCH')
    if [r['date'] for r in current]!=[ts for ts in grid_5min(day) if ts<=cutoff]:
        raise ValueError('EM_CURRENT_PREFIX_INCOMPLETE')
    return current


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def fetch(code,previous_day,day,reserve):
    """Exactly one bounded attempt, reserved BEFORE opening the connection."""
    secid,_=security(code)
    url='https://push2his.eastmoney.com/api/qt/stock/kline/get?'+urlencode(dict(
        secid=secid,klt=5,fqt=0,beg=previous_day,end=day,lmt=100,
        fields1='f1,f2,f3,f4,f5,f6',fields2='f51,f52,f53,f54,f55,f56,f57'))
    if reserve is None or not reserve(code,url,MAX_BYTES):raise ValueError('EM_REQUEST_NOT_AUTHORIZED')
    with build_opener(NoRedirect()).open(Request(url,headers={
            'User-Agent':'Mozilla/5.0','Referer':'https://quote.eastmoney.com/'}),timeout=5) as response:
        raw=response.read(MAX_BYTES+1)
    if len(raw)>MAX_BYTES:raise ValueError('EM_MINUTE_RESPONSE_LIMIT')
    return raw,dict(contract=CONTRACT,url=url,code=code,frequency_minutes=5,clock='END_5M',
        price_basis='RAW',volume_basis='LOTS_TIMES_100_AMOUNT_PER_SHARE_VALIDATED',
        sha256=hashlib.sha256(raw).hexdigest(),available_at=datetime.now().astimezone().isoformat(),
        availability_basis='LOCAL_RESPONSE_OBSERVED_NOT_UPSTREAM_FIRST_ARRIVAL')
