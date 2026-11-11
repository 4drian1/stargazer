"""Approved price/volume funnel; no learned scores or buy decisions."""
from strong_wave_facts_v1 import primary,swings

RATIOS=(.236,.382,.5,.618,.786)
CONTRACT='STRONG_SECTOR_FROZEN_W3_20_30_V1'


def cluster(sources):
    groups=[]
    for source in sorted(sources,key=lambda r:r['price']):
        if not groups or source['price']/groups[-1][0]['price']-1>=.008:groups.append([])
        groups[-1].append(source)
    return [dict(lower=g[0]['price'],upper=g[-1]['price'],
        source_waves=len({tuple(r['wave']) for r in g}),sources=g,
        primary= len({tuple(r['wave']) for r in g})>=3 and g[-1]['price']/g[0]['price']-1<.01)
        for g in groups]


def build(rows,asof,atr):
    import math
    if not isinstance(atr,(float,int)) or not math.isfinite(atr) or atr<=0:raise ValueError('ATR_UNKNOWN')
    rows=[r for r in rows if r['date']<=asof]
    fact=primary(rows,asof)
    result=dict(contract=CONTRACT,asof=asof,code=rows[-1].get('code') if rows else None,
                pool='EXCLUDED',facts=fact,reason=None)
    if fact['status']!='PRICE_FACTS_ONLY':return dict(result,reason=fact['status'])
    from descending_tops_v1 import diagnose
    descending=diagnose(rows,asof)
    if descending['excluded']:return dict(result,reason='EXCLUDE_DESCENDING_TRIPLE',descending=descending)
    if any(type(r.get('volume')) not in (int,float) or not math.isfinite(r['volume']) or r['volume']<0 for r in rows[-60:]):
        return dict(result,reason='RECENT_VOLUME_UNKNOWN')
    if not all(fact[k] is True for k in ['liquidity_passed','gain_at_least_20pct','peak_within_30days','a_confirmed','b_confirmed']):
        return dict(result,reason='LIQUIDITY_WAVE_OR_CONFIRMATION_NOT_PASSED')
    low,high=fact['low'],fact['high'];B=next(i for i,r in enumerate(rows) if r['date']==fact['high_day'])
    A=next(i for i,r in enumerate(rows) if r['date']==fact['low_day'])
    pivots=swings(rows);confirmed=[p for p in pivots if p['complete_window']]
    resets=[]
    for p in confirmed:
        # Only resets inside the current A->B wave can change its top count.
        # Historical pivots remain available below for resonance sources.
        if not p['high'] or not A<=p['index']<=B:continue
        i=p['index'];v=rows[i].get('volume')
        if v is None:continue
        for j in range(i+4,B+1):
            if rows[j]['close']>rows[i]['high'] and (rows[j].get('volume') or 0)>.8*v:
                resets.append(j);break
    reset=max([i for i in resets if i<=B],default=A)
    tops=[p for p in confirmed if p['high'] and max(A,reset,B-25)<=p['index'] and rows[p['index']]['high']>=high*.93]
    unique=[]
    for p in tops:
        if unique and p['index']-unique[-1]['index']<3:
            if rows[p['index']]['high']>rows[unique[-1]['index']]['high']:unique[-1]=p
        else:unique.append(p)
    known_volume=all(type(rows[p['index']].get('volume')) in (float,int) and rows[p['index']]['volume']>0 for p in unique)
    if not known_volume:return dict(result,reason='TOP_VOLUME_UNKNOWN')
    for left,right in zip(unique,unique[1:]):
        a,b=rows[left['index']],rows[right['index']]
        if b['high']>a['high'] and b['volume']<.8*a['volume']:
            return dict(result,reason='SHRINKING_VOLUME_NEW_HIGH_VETO',tops=unique)
    if len(unique)>=3 and rows[unique[2]['index']]['volume']<.7*rows[unique[0]['index']]['volume']:
        return dict(result,reason='THIRD_TOP_VOLUME_CLIFF',tops=unique)
    collapsed=False
    for p in confirmed:
        i=p['index']
        if not p['high'] or i<=B or i<5:continue
        volumes=[r.get('volume') for r in rows[i-5:i]]
        if any(v is None for v in volumes) or rows[i].get('volume') is None:continue
        if min(r['low'] for r in rows[B:i])<=high*.7 and rows[i]['volume']>=sum(volumes)/5*1.5:
            collapsed=True
    structure='TRIPLE' if len(unique)>=3 or collapsed else 'DOUBLE' if len(unique)==2 else 'SINGLE'
    if structure=='DOUBLE' and abs(rows[unique[0]['index']]['high']-rows[unique[1]['index']]['high'])/high>.05:structure='WEAK_DOUBLE'
    shoulders=[p for p in confirmed if p['high'] and max(A,reset,B-25)<=p['index'] and high*.85<=rows[p['index']]['high']<high*.93]
    if structure=='SINGLE' and shoulders:structure='WEAK_DOUBLE'
    neckline=None
    if len(unique)>=2:
        lo,hi=unique[0]['index'],unique[1]['index']
        neckline=min(r['low'] for r in rows[lo:hi+1])
    else:
        lows=[p for p in confirmed if p['low'] and A<p['index']<B]
        if lows:neckline=rows[lows[-1]['index']]['low']
    # Adjacent confirmed low->high swings, each endpoint pair counted once.
    sources=[];last_low=None
    for p in confirmed:
        if p['low']:last_low=p
        if p['high'] and last_low and p['index']>last_low['index']:
            a,b=rows[last_low['index']],rows[p['index']]
            if b['high']>a['low']:
                sources.extend(dict(price=b['high']-(b['high']-a['low'])*r,ratio=r,
                    wave=[a['date'],b['date']]) for r in RATIOS)
            last_low=None
    primary_levels=[dict(price=high-(high-low)*r,ratio=r,wave=[fact['low_day'],fact['high_day']]) for r in RATIOS]
    if not any(s['wave']==[fact['low_day'],fact['high_day']] for s in sources):sources.extend(primary_levels)
    bands=cluster(sources)
    confluences=[b for b in bands if b['source_waves']>=2]
    choices=confluences or [dict(lower=p['price'],upper=p['price'],source_waves=1,primary=False,sources=[p]) for p in primary_levels]
    r382=high-(high-low)*.382;r618=high-(high-low)*.618;r786=high-(high-low)*.786
    positions=[b for b in choices if b['lower']<=r382 and b['upper']>=r618 or abs((b['lower']+b['upper'])/2-r786)<=.2*atr]
    extensions=[dict(lower=high-(high-low)*r,upper=high-(high-low)*r,source_waves=1,primary=False,
        requires_neckline_break=True,sources=[dict(ratio=r,wave=[fact['low_day'],fact['high_day']],price=high-(high-low)*r)])
        for r in (1.,1.272,1.618) if neckline is not None and 0<high-(high-low)*r<neckline]
    positions.extend(extensions)
    if structure=='TRIPLE':
        if rows[-1]['close']<r786:return dict(result,reason='CLOSED_BELOW_786_DELETE')
        positions=[dict(lower=high-(high-low)*r,upper=high-(high-low)*r,source_waves=1,primary=False,
                        landmark_ratio=r) for r in (.5,.786)]
    return dict(result,pool='LANDMARK' if structure=='TRIPLE' else 'MONITOR',reason=None,
        structure=structure,weak_double=structure=='WEAK_DOUBLE',collapse_rebound=collapsed,
        tops=[dict(date=rows[p['index']]['date'],price=rows[p['index']]['high'],volume=rows[p['index']]['volume']) for p in unique],
        reset_day=rows[reset]['date'],neckline=neckline,positions=positions,confluences=confluences,
        reference_close=rows[-1]['close'],atr_ref=atr,history_days=[r['date'] for r in rows[-5:]],
        high=high,low=low,wave_return=fact['gain'],retracements=primary_levels)
