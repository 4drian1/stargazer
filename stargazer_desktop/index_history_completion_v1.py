"""Complete only unfinished public-source days; finalized/TDX history is immutable."""
from datetime import datetime
import json,math

def completion(raw,old_rows,day,grid):
    owned=[r['day'] for r in old_rows if r.get('amount') is None and r.get('raw_volume') is None]
    if not owned:return [],[]
    first=min(owned);groups={}
    for r in old_rows:groups.setdefault(r['day'],[]).append(r)
    parsed={}
    for item in json.loads(raw):
        stamp=int(datetime.strptime(item['day'],'%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M%S'));d=stamp//1000000
        if not first<=d<day:continue
        o,h,l,c=[float(item[k]) for k in ('open','high','low','close')]
        if not all(math.isfinite(v) and v>0 for v in (o,h,l,c)) or not l<=min(o,c)<=max(o,c)<=h:raise ValueError('INDEX_HISTORY_OHLC')
        parsed.setdefault(d,[]).append(dict(ts=stamp,day=d,open=o,high=h,low=l,close=c,amount=None,raw_volume=None))
    rows=[];days=[]
    for d,values in sorted(parsed.items()):
        previous=groups.get(d,[])
        if len(previous)>=48 or any(r.get('amount') is not None or r.get('raw_volume') is not None for r in previous):continue
        if [r['ts'] for r in values]!=grid(d):continue
        rows.extend(values);days.append(d)
    return rows,days
