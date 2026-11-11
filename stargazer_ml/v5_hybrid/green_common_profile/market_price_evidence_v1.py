"""Causal price evidence; experimental classifier output is kept in its own field."""
import numpy as np
import pandas as pd
from golden_ladder_v3 import _rsi, index_map

VERSION = 'MARKET_PRICE_REPAIR_V1'


def critical_position(frame,cutoff):
    """Freeze the critical node from the last completed session, never today's price."""
    day=cutoff//1000000
    past=frame[frame.date//1000000<day]
    today=frame[frame.date//1000000==day]
    if past.empty or len(today)<2:return dict(state='UNKNOWN')
    previous=past[past.date//1000000==int(past.date.iloc[-1])//1000000]
    if len(previous)!=48 or int(previous.date.iloc[-1])%1000000!=150000:return dict(state='UNKNOWN')
    p=int(past.date.iloc[-1]);mapping=index_map(past,p)
    reference=float(past.close.iloc[-1])
    below=[v['price'] for v in mapping.get('levels',[]) if v['price']<=reference]
    if not below:return dict(state='UNKNOWN')
    node=max(below);two=today.close.iloc[-2:].to_numpy();momentum=_momentum(frame)
    low_i=int(np.argmin(today.low.to_numpy()))
    stable=(len(today)-1>=low_i+2 and today.low.iloc[-1]>=today.low.iloc[-2]>=today.low.iloc[low_i]
        and today.close.iloc[-1]>today.close.iloc[-2]>=today.close.iloc[low_i])
    broken=bool((two<node).all())
    repaired=bool((two>=node).all() and ((today.low<node).any() or stable) and momentum['rsi_above'] is True)
    return dict(state='BROKEN' if broken else 'REPAIRED' if repaired else 'PENDING',
                lower=node,upper=node,reference_asof=p,asof=cutoff,last_two_closes=two.tolist())


def _momentum(frame):
    a, b = _rsi(frame.close.to_numpy(), 6), _rsi(frame.close.to_numpy(), 15)
    def number(value):
        return float(value) if np.isfinite(value) else None
    return dict(evidence_time=int(frame.date.iloc[-1]), rsi6=number(a[-1]), rsi15=number(b[-1]),
                rsi_above=bool(a[-1] > b[-1]) if np.isfinite([a[-1], b[-1]]).all() else None)


def divergence_evidence(frame, cutoff):
    """Reuse the existing completed-bar low-pair rule, without volume claims."""
    from strong_low_indicators_v1 import pair, aggregate15
    f = frame.loc[frame.date <= cutoff]
    day = cutoff // 1000000
    prior = f.loc[f.date // 1000000 < day]
    if prior.empty:
        return dict(status='UNKNOWN', asof=cutoff, reason='NO_PREVIOUS_SESSION', periods={})
    previous = int(prior.date.iloc[-1]) // 1000000
    bars = [dict(day=int(r.date)//1000000, t=int(r.date)%1000000,
                 open=float(r.open), high=float(r.high), low=float(r.low),
                 close=float(r.close), volume=None) for r in f.itertuples()]
    periods = {'5': pair(bars, previous, day, 5),
               '15': pair(aggregate15(bars), previous, day, 15)}
    return dict(status='AVAILABLE', asof=cutoff, previous_day=previous, periods=periods,
                rule='EXISTING_STRONG_LOW_PAIR_COMPLETE_BARS', volume_verified=False)


def rebound_map(frame, cutoff):
    """Descriptive 60-session high -> subsequent low map, not the execution map."""
    from golden_ladder_v3 import STOPS
    f=frame.loc[frame.date<=cutoff].copy()
    day=cutoff//1000000
    daily=[];excluded=[]
    for date,group in f.groupby(f.date//1000000,sort=True):
        expected=[t for t in STOPS if date<day or date*1000000+t<=cutoff]
        if (group.date.astype(int)%1000000).tolist()!=expected:
            excluded.append(int(date));continue
        daily.append(dict(day=int(date),open=float(group.open.iloc[0]),high=float(group.high.max()),
                          low=float(group.low.min()),close=float(group.close.iloc[-1])))
    current=next((d for d in daily if d['day']==day),None)
    if current is None:return dict(status='UNKNOWN',reason='CURRENT_PRICE_PREFIX_MISSING')
    window=daily[-60:]
    if len(window)<3:return dict(status='UNKNOWN',reason='INSUFFICIENT_MAP_HISTORY')
    # Today's high is not used as a prior decline anchor.
    peak=max(window[:-1],key=lambda d:d['high'])
    after=[d for d in window if d['day']>peak['day']]
    trough=min(after,key=lambda d:d['low'])
    high,low=peak['high'],trough['low'];span=high-low
    if span<=0:return dict(status='UNKNOWN',reason='NO_POSITIVE_DECLINE_RANGE')
    price=current['close']
    levels=[dict(ratio=r,price=low+r*span,distance=(low+r*span)/price-1,
                 above_close=low+r*span>price) for r in (.236,.382,.5,.618,.786,1.)]
    previous=window[-2]
    return dict(status='PRICE_MAP_ONLY',asof=cutoff,evidence_time=int(f.date.iloc[-1]),
        rule='RECENT_60_VALID_SESSIONS_HIGHEST_PRIOR_HIGH_THEN_LOW',window_sessions=len(window),
        excluded_incomplete_days=excluded,high=high,high_day=peak['day'],low=low,low_day=trough['day'],
        low_is_current_session=trough['day']==day,low_following_sessions=sum(d['day']>trough['day'] for d in window),
        close=price,session_open=current['open'],session_high=current['high'],session_low=current['low'],
        lower_shadow=min(current['open'],price)-current['low'],
        session_complete=cutoff%1000000==150000,return_from_previous_close=price/previous['close']-1,
        reference_day=previous['day'],rebounce_from_session_low=price/current['low']-1,
        drawdown_from_high=price/high-1,recovered_fraction=(price-low)/span,levels=levels,
        volume_verified=False,volume_reason='PRICE_ONLY_SOURCE_VOLUME_NOT_VERIFIED',
        interpretation='窗口价格地图；观察天数不等于双底确认，价格反弹不证明资金流入；不改变执行闸门')


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
        map={k:v for k,v in mapping.items() if k!='levels'},summary=summary,critical_position=critical_position(f,cutoff),
        outlook=rebound_map(f,cutoff),
        divergence=divergence_evidence(f,cutoff),
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


def attach_trials(evidence,frames,cutoff,trial_seconds=8):
    """One shared deadline; late trial writes never mutate returned facts."""
    import threading
    import time
    deadline=time.monotonic()+trial_seconds
    workers={}
    results={}
    for name,original in evidence.items():
        if original.get('status')!='PRICE_FACTS_ONLY':continue
        results[name]=dict(original)
        worker=threading.Thread(target=attach_trial,
            args=(results[name],frames[name],name,cutoff,trial_seconds),daemon=True)
        workers[name]=worker
        worker.start()
    for worker in workers.values():
        worker.join(timeout=max(0,deadline-time.monotonic()))
    for name,worker in workers.items():
        if worker.is_alive():
            evidence[name]=dict(evidence[name],research_state='TRIAL_SKIPPED_TIME_BUDGET',
                research_trial=dict(asof=cutoff,status='UNKNOWN',reason='TRIAL_TIME_BUDGET_EXCEEDED'))
        else:
            evidence[name]=dict(results[name])


def attach_trial(evidence,frame,name,cutoff,trial_seconds=8):
    if evidence.get('status')!='PRICE_FACTS_ONLY':return
    # 价格事实与试验模型解耦：trial 在守护线程中运行，有独立时间预算；
    # 超时即跳过——worker 进程退出绝不等待它（线程池会join拖住整个退出）。
    import queue as _queue
    import threading as _threading
    try:
        from market_research_trial_v1 import predict
        box = _queue.Queue()
        def run():
            try:
                box.put((True, predict(frame, name, cutoff)))
            except Exception as exc:
                box.put((False, exc))
        worker = _threading.Thread(target=run, daemon=True, name='index-trial')
        worker.start()
        try:
            ok, value = box.get(timeout=trial_seconds)
        except _queue.Empty:
            evidence['research_trial']=dict(asof=cutoff,status='UNKNOWN',reason='TRIAL_TIME_BUDGET_EXCEEDED')
            evidence['research_state']='TRIAL_SKIPPED_TIME_BUDGET'
            return
        if not ok:
            raise value
        trial = value
        evidence['research_trial']=trial
        evidence['research_state']='ENABLED_EXPERIMENTAL_NOT_ACCEPTED'
        evidence['resonance_bands']=trial['resonance_bands']
    except Exception as exc:
        evidence['research_trial']=dict(asof=cutoff,status='UNKNOWN',reason=type(exc).__name__)
        evidence['research_state']='TRIAL_SKIPPED:'+type(exc).__name__


if __name__=='__main__':
    import gzip
    import hashlib
    import io
    import json
    from pathlib import Path
    import sys
    RISK_DIR=Path(__file__).resolve().parent.parent/'risk_r0_v1'
    if str(RISK_DIR) not in sys.path:sys.path.insert(0,str(RISK_DIR))
    from joint_shadow_service_v1 import verify_fingerprints
    cutoff=int(json.load(sys.stdin)['cutoff'])
    root=RISK_DIR/'index_minutes_v1'
    expected=json.loads(gzip.decompress((RISK_DIR/'live_monitor_fingerprints_20261110_v206_flow.json.gz').read_bytes()))
    mismatch,ok=verify_fingerprints(expected)
    if not ok:
        raise ValueError('RUNTIME_FINGERPRINT_MISMATCH:'+','.join(mismatch[:5])+(' 等%d项'%len(mismatch) if len(mismatch)>5 else ''))
    evidence={}
    frames={}
    for name in ['sse','cyb']:
        path=root/(name+'_5min.parquet')
        meta=json.loads(path.with_suffix('.meta.json').read_bytes())
        if path.stat().st_size>4*1024**2:raise ValueError('INDEX_SIZE_CAP')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise ValueError('INDEX_HASH_MISMATCH')
        frames[name]=pd.read_parquet(io.BytesIO(raw)).rename(columns={'ts':'date'})
        evidence[name]=observe(frames[name],cutoff)
    # 两个指数的试验模型并行挂载(各自守护线程+预算)：worker 总退出时间
    # 不受慢模型叠加拖累，价格事实先行返回。
    attach_trials(evidence,frames,cutoff)
    for name in ['sse','cyb']:
        frame=frames[name]
        prefix=frame.loc[frame.date<=cutoff,['date','open','high','low','close']]
        evidence[name]['prefix_sha256']=hashlib.sha256(prefix.to_json(orient='records',double_precision=15).encode()).hexdigest()
        evidence[name]['source_verification']='MIRROR_SHA256_SIDECAR_MATCH'
        evidence[name]['availability_basis']='DERIVED_FROM_ARCHIVE_ON_READ_NOT_HISTORICAL_ARRIVAL'
    print(json.dumps(dict(cutoff=cutoff,evidence=evidence),ensure_ascii=False,allow_nan=False))
