"""Frozen price-only descending-main-peak diagnosis. No learned output."""
import math
CONTRACT='DESCENDING_MAIN_PEAKS_60_W2_G3_D3_V1'

def diagnose(rows,asof):
    if len(rows)<62:raise ValueError('DESCENDING_HISTORY_LT_62')
    if rows[-1]['date']!=asof or [r['date'] for r in rows]!=sorted({r['date'] for r in rows}):
        raise ValueError('DESCENDING_DATE_IDENTITY')
    for r in rows:
        if not all(type(r[k]) in (int,float) and math.isfinite(r[k]) and r[k]>0 for k in ('high','low')) or r['low']>r['high']:
            raise ValueError('DESCENDING_PRICE_INVALID')
    peaks=[]
    start=max(2,len(rows)-60)
    for i in range(start,len(rows)-2):
        h=rows[i]['high'];left=[r['high'] for r in rows[i-2:i]];right=[r['high'] for r in rows[i+1:i+3]]
        # A completely flat run is not a sequence of new rebound peaks.
        if h<max(left+right) or not any(v<h for v in left) or not any(v<h for v in right):continue
        p=dict(index=i,date=rows[i]['date'],price=h,confirmed_at=rows[i+2]['date'])
        if peaks:
            previous=peaks[-1];gap=i-previous['index']
            trough=min((r['low'] for r in rows[previous['index']+1:i]),default=float('inf'))
            if gap<3 or trough>.97*min(h,previous['price']):
                if h>previous['price']:peaks[-1]=p
                continue
        peaks.append(p)
    main=[]
    for p in peaks:
        while main and (p['price']>=main[-1]['price'] or math.isclose(p['price'],main[-1]['price'],rel_tol=1e-12,abs_tol=1e-12)):
            main.pop()
        main.append(p)
    def project(p):return {k:p[k] for k in ('date','price','confirmed_at')}
    return dict(contract=CONTRACT,asof=asof,window_start=rows[start]['date'],
        status='EXCLUDE_DESCENDING_TRIPLE' if len(main)>=3 else 'NO_DESCENDING_TRIPLE',
        excluded=len(main)>=3,main_peaks=[project(p) for p in main],confirmed_peaks=[project(p) for p in peaks],
        no_depth_exemption=True,scope='Additional price-structure veto; not a probability or return forecast')


def for_frame(frame):
    return diagnose(frame[["date","high","low"]].to_dict("records"),int(frame.date.iloc[-1]))
