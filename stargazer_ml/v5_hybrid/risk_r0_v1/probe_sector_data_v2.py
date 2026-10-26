"""Investigate membership sourcing for sector synthesis (read-only + 1 tiny list pilot).
1) stockdb per-stock industry field (for SW membership, local).
2) stockdb individual-stock 1-min history depth (how far back).
3) Pilot: East Money concept-board member list for BK0989 (储能) via free API.
No full download; only a tiny membership list probe.
"""
import os, sys, urllib.request, urllib.parse, msgpack, json, requests
os.environ.setdefault('NO_PROXY','*'); os.environ.setdefault('no_proxy','*')
os.environ.pop('HTTP_PROXY',None); os.environ.pop('HTTPS_PROXY',None)

BASE='http://127.0.0.1:7899'
def get(table, code='*', date='*'):
    t=urllib.parse.quote(f'{table}:{code}:{date}')
    with urllib.request.urlopen(f'{BASE}/?cmd=get&t={t}', timeout=60) as r:
        return msgpack.unpackb(r.read(), raw=False, strict_map_key=False)

print("===== stockdb per-stock industry field + 1-min depth =====")
# inspect one stock's daily record keys
try:
    recs = get('日k','600276','*')
    if isinstance(recs, list) and recs:
        # find a dict value
        d = None
        for k,v in recs:
            if isinstance(v, dict):
                d=v; break
        if d:
            print("600276 daily record keys:", list(d.keys()))
            ind = {k:v for k,v in d.items() if any(x in k.lower() for x in ('industr','行业','sector','sw','index_code','板块','bk'))}
            print("  industry-ish fields:", ind)
    else:
        print("600276 日k empty/list?", type(recs))
except Exception as e:
    print("600276 daily ERR:", type(e).__name__, str(e)[:80])

# 1-min earliest date for 600276
try:
    recs = get('分钟k','600276','*')  # all dates
    dates = []
    for k,v in recs:
        if isinstance(v, dict) and 'date' in v:
            dates.append(v['date'])
    print("600276 分钟k distinct dates:", len(set(dates)), "| earliest:", min(dates), "latest:", max(dates))
except Exception as e:
    print("600276 分钟k ERR:", type(e).__name__, str(e)[:80])

# full code table membership field
try:
    obj = get('股票代码','*','*')
    print("股票代码 table type:", type(obj))
    if isinstance(obj, list) and obj:
        sample = obj[0][1] if isinstance(obj[0], (list,tuple)) else obj[0]
        if isinstance(sample, dict):
            keys=list(sample.keys())
            ind=[k for k in keys if any(x in k.lower() for x in ('industr','行业','sector','sw','index_code','板块','bk','concept'))]
            print("  code-table keys:", keys[:20])
            print("  industry-ish:", ind, "| sample:", {k:sample.get(k) for k in ind[:5]})
    elif isinstance(obj, dict):
        k0=next(iter(obj)); print("  dict sample key", k0, "val", str(obj[k0])[:120])
except Exception as e:
    print("股票代码 table ERR:", type(e).__name__, str(e)[:100])

print("\n===== EM concept board member list pilot (BK0989 储能) =====")
s=requests.Session(); s.trust_env=False
s.headers.update({"User-Agent":"Mozilla/5.0","Referer":"https://quote.eastmoney.com/"})
# board member list endpoint
u="https://push2.eastmoney.com/api/qt/clist/get"
params={"pn":"1","pz":"2000","po":"1","np":"1","fltt":"2","invt":"2",
        "fs":f"b:BK0989","fields":"f12,f14,f20"}
try:
    j=s.get(u, params=params, timeout=30).json()
    if j.get('data') and j['data'].get('diff'):
        items=j['data']['diff']
        print("BK0989 members:", len(items), "| sample:", [(it.get('f12'),it.get('f14')) for it in items[:5]])
    else:
        print("BK0989 empty:", str(j)[:200])
except Exception as e:
    print("EM member ERR:", type(e).__name__, str(e)[:120])
