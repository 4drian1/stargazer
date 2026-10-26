"""Causal price evidence; experimental classifier output is kept in its own field."""
import numpy as np
import pandas as pd
from golden_ladder_v3 import _rsi, index_map

VERSION = 'MARKET_PRICE_REPAIR_V1'


def _momentum(frame):
    a, b = _rsi(frame.close.to_numpy(), 6), _rsi(frame.close.to_numpy(), 15)
    def number(value):
        return float(value) if np.isfinite(value) else None
    return dict(evidence_time=int(frame.date.iloc[-1]), rsi6=number(a[-1]), rsi15=number(b[-1]),
                rsi_above=bool(a[-1] > b[-1]) if np.isfinite([a[-1], b[-1]]).all() else None)


def observe(frame, cutoff):
    f = frame.loc[frame.date <= cutoff, ['date','open','high','low','close']].copy()
    unknown = dict(version=VERSION, asof=cutoff, status='UNKNOWN', volume_verified=False)
    if f.empty or f.date.duplicated().any() or not f.date.is_monotonic_increasing:
        return dict(unknown, reason='NO_ORDERED_COMPLETE_PREFIX')
    day = cutoff // 1000000
    today = f[f.date // 1000000 == day]
    minute = cutoff // 100 % 10000
    grid = [h*100+m for h in [9,10,11,13,14,15] for m in range(0,60,5)
            if 935 <= h*100+m <= 1130 or 1305 <= h*100+m <= 1500]
    expected = [day*1000000 + t*100 for t in grid if t <= minute]
    if cutoff%100 or not expected or minute not in grid or today.date.astype(int).tolist() != expected:
        return dict(unknown, reason='CURRENT_PREFIX_GAP_OR_CUTOFF_MISMATCH')
    prices = f[['open','high','low','close']].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        return dict(unknown, reason='INVALID_OHLC')
    o,h,l,c = prices.T
    if ((l > np.minimum(o,c)) | (h < np.maximum(o,c))).any():
        return dict(unknown, reason='INVALID_OHLC')
    mapping = index_map(f, cutoff)
    five = _momentum(f)
    a,b = _rsi(f.close.to_numpy(),6), _rsi(f.close.to_numpy(),15)
    crosses = np.flatnonzero((a[1:] > b[1:]) & (a[:-1] <= b[:-1])) + 1
    five['cross_up_times'] = [int(f.date.iloc[i]) for i in crosses if int(f.date.iloc[i])//1000000 == day]
    two = today.iloc[-2:]
    low_i = int(np.argmin(today.low.to_numpy()))
    stable = (len(today) >= 3 and len(today)-1 >= low_i+2 and
              two.low.iloc[-1] >= two.low.iloc[0] >= today.low.iloc[low_i] and
              two.close.iloc[-1] > two.close.iloc[0] >= today.close.iloc[low_i])
    levels = []
    for item in mapping.get('levels', []):
        price = item['price']
        pierced = bool((today.low < price).any())
        recovered = len(two) == 2 and bool((two.close >= price).all())
        levels.append(dict(price=price, kind=item['kind'], ratio=item['ratio'],
            pierced= pierced, two_close_above=recovered,
            reclaimed=bool(pierced and recovered),
            below_then_close_back_times=item['below_then_close_back_times']))
    # Aggregate only three completed native bars. Never create an incomplete 15m bar.
    minutes = f.date // 100 % 10000
    elapsed = minutes // 100 * 60 + minutes % 100
    ends = ((elapsed + 14)//15*15).astype(int)
    grouped = f.assign(end=f.date//1000000*1000000 + (ends//60*100+ends%60)*100).groupby('end')
    fifteen = grouped.agg(open=('open','first'),high=('high','max'),low=('low','min'),
                         close=('close','last'),count=('close','count')).reset_index().rename(columns={'end':'date'})
    fifteen = fifteen[(fifteen['count']==3) & (fifteen.date <= cutoff)]
    confirmation = _momentum(fifteen) if len(fifteen) else dict(evidence_time=None,rsi6=None,rsi15=None,rsi_above=None)
    reclaimed = any(x['reclaimed'] for x in levels)
    if reclaimed and five['rsi_above']:
        summary = '黄金档位两根收回且RSI快线在慢线上方；量能未知，暂不判强弱'
    elif stable and five['rsi_above']:
        summary = '低点后价格企稳形态与RSI修复同现；尚须逐档核对，量能未知'
    else:
        summary = '价格位置与RSI尚未共同确认修复；逐档观察，量能未知'
    return dict(version=VERSION,asof=cutoff,evidence_time=int(today.date.iloc[-1]),status='PRICE_FACTS_ONLY',
        primary='5m',confirmation='15m',last_close=float(today.close.iloc[-1]),
        session_low=float(today.low.min()),last_two_times=two.date.astype(int).tolist(),
        last_two_closes=two.close.astype(float).tolist(),price_stable_shape=bool(stable),
        five=five,fifteen=confirmation,levels=levels,
        map={k:v for k,v in mapping.items() if k!='levels'},summary=summary,
        volume_verified=False,volume_state='UNKNOWN',time_semantics='END_5M_ASSUMED_NOT_CERTIFIED',
        research_state='NOT_ENABLED_VOLUME_UNVERIFIED',forecast=False)


def for_snapshot(snapshot):
    if getattr(snapshot,'_price_evidence_cutoff',None) != snapshot.cutoff:
        result = {}
        for symbol,name in [('000001.SH','sse'),('399006.SZ','cyb')]:
            frame = (getattr(snapshot,'index_5m',{}) or {}).get(symbol)
            result[name] = (observe(frame,snapshot.cutoff) if frame is not None else
                            dict(version=VERSION,asof=snapshot.cutoff,status='UNKNOWN',reason='NO_INDEX_SOURCE'))
            result[name]['availability_basis'] = (getattr(snapshot,'receipts',{}) or {}).get(symbol,{}).get('availability_basis','UNKNOWN')
            attach_trial(result[name],frame,name,snapshot.cutoff)
        snapshot._price_evidence, snapshot._price_evidence_cutoff = result,snapshot.cutoff
    return snapshot._price_evidence


def attach_trial(evidence,frame,name,cutoff):
    if evidence.get('status')!='PRICE_FACTS_ONLY':return
    try:
        from market_research_trial_v1 import predict
        trial=predict(frame,name,cutoff)
        evidence['research_trial']=trial
        evidence['research_state']='ENABLED_EXPERIMENTAL_NOT_ACCEPTED'
        evidence['resonance_bands']=trial['resonance_bands']
    except Exception as exc:
        evidence['research_trial']=dict(asof=cutoff,status='UNKNOWN',reason=type(exc).__name__)


if __name__=='__main__':
    import gzip
    import hashlib
    import io
    import json
    from pathlib import Path
    import sys
    cutoff=int(json.load(sys.stdin)['cutoff'])
    root=Path(__file__).resolve().parent.parent/'risk_r0_v1/index_minutes_v1'
    manifest=json.loads(gzip.decompress((root.parent/'live_monitor_fingerprints_20261026_v26_flow.json.gz').read_bytes()))
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected for path,expected in manifest.items()):
        raise ValueError('RUNTIME_FINGERPRINT_MISMATCH')
    evidence={}
    for name in ['sse','cyb']:
        path=root/(name+'_5min.parquet')
        meta=json.loads(path.with_suffix('.meta.json').read_bytes())
        if path.stat().st_size>4*1024**2:raise ValueError('INDEX_SIZE_CAP')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise ValueError('INDEX_HASH_MISMATCH')
        frame=pd.read_parquet(io.BytesIO(raw)).rename(columns={'ts':'date'})
        evidence[name]=observe(frame,cutoff)
        attach_trial(evidence[name],frame,name,cutoff)
        prefix=frame.loc[frame.date<=cutoff,['date','open','high','low','close']]
        evidence[name]['prefix_sha256']=hashlib.sha256(prefix.to_json(orient='records',double_precision=15).encode()).hexdigest()
        evidence[name]['source_verification']='MIRROR_SHA256_SIDECAR_MATCH'
        evidence[name]['availability_basis']='DERIVED_FROM_ARCHIVE_ON_READ_NOT_HISTORICAL_ARRIVAL'
    print(json.dumps(dict(cutoff=cutoff,evidence=evidence),ensure_ascii=False,allow_nan=False))
