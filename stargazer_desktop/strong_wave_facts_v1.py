"""Causal daily price facts only; not a final admission or trading decision."""
import math


def swings(rows,w=3):
    """Report complete-window pivots and explicitly provisional edge candidates."""
    result=[]
    for i,row in enumerate(rows):
        known=rows[max(0,i-w):min(len(rows),i+w+1)]
        low=row['low']==min(v['low'] for v in known)
        high=row['high']==max(v['high'] for v in known)
        if low or high:
            result.append(dict(index=i,date=row['date'],low=low,high=high,
                complete_window=i>=w and i+w<len(rows),left_count=min(i,w),right_count=min(len(rows)-i-1,w)))
    return result


def primary(rows,asof):
    """The user's 60-day A/min -> subsequent B/max geometry, without future rows."""
    prefix=[r for r in rows if r['date']<=asof]
    if not prefix or prefix[-1]['date']!=asof:
        return dict(status='SOURCE_NOT_AT_CUTOFF',asof=asof)
    dates=[r['date'] for r in prefix]
    if dates!=sorted(set(dates)):raise ValueError('DAILY_ORDER_OR_DUPLICATE')
    for row in prefix:
        values=[row.get(k) for k in ('open','high','low','close')]
        if not all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in values):
            raise ValueError('DAILY_PRICE_UNKNOWN')
        if not row['low']<=min(row['open'],row['close'])<=max(row['open'],row['close'])<=row['high']:
            raise ValueError('DAILY_OHLC_CONFLICT')
    window=prefix[-60:]
    a=min(range(len(window)),key=lambda i:window[i]['low'])
    if a==len(window)-1:return dict(status='NO_POST_LOW_WAVE',asof=asof)
    b=max(range(a+1,len(window)),key=lambda i:window[i]['high'])
    A=window[a];B=window[b];offset=len(prefix)-len(window);absolute_b=b+offset
    pivots=swings(prefix)
    pivot=next((r for r in pivots if r['index']==absolute_b and r['high']),None)
    a_known=window[max(0,a-3):min(len(window),a+4)]
    # Edge allowance is explicitly specified for A only; B needs its full window.
    a_confirmed=A['low']==min(r['low'] for r in a_known)
    b_confirmed=bool(pivot and pivot['complete_window'])
    gain=B['high']/A['low']-1
    age=len(window)-1-b
    levels=[dict(ratio=r,price=B['high']-(B['high']-A['low'])*r) for r in (.236,.382,.5,.618,.786)]
    tops=[dict(p,date=prefix[p['index']]['date'],price=prefix[p['index']]['high'],
        inside_document_window=absolute_b-25<=p['index']<=absolute_b,
        after_b=p['index']>absolute_b,near_peak=prefix[p['index']]['high']>=B['high']*.93)
        for p in pivots if p['high'] and p['complete_window']]
    amounts=[r.get('amount') for r in window[-20:]]
    liquidity_known=len(amounts)==20 and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in amounts)
    return dict(status='PRICE_FACTS_ONLY',asof=asof,low_day=A['date'],high_day=B['date'],
        low=A['low'],high=B['high'],gain=gain,peak_age=age,a_confirmed=a_confirmed,b_confirmed=b_confirmed,
        gain_at_least_20pct=gain>=.20 or math.isclose(gain,.20,rel_tol=1e-12,abs_tol=1e-12),peak_within_30days=age<=30,
        amount20=sum(amounts)/20 if liquidity_known else None,
        liquidity_passed=sum(amounts)/20>=5e7 if liquidity_known else None,
        retracements=levels,raw_tops=tops,admission='NOT_DECIDED_PENDING_RULE_CONTRACT')
