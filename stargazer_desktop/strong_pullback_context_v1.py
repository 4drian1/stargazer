"""Local market-test evidence and same-slot volume for pullback notices."""
from datetime import datetime
import json
import hashlib
import io
import math
from pathlib import Path
from strong_pullback_policy_v1 import market_state


def market_context(now,quote,base,observation_day=None):
    """Previous-session fixed level. Never reconstruct a historical T0 later."""
    import pandas as pd
    import index_minutes_v1
    from golden_ladder_v3 import index_map
    from tarot_monitor_v2 import save
    from tarot_live_v2 import trading_calendar
    from tdx_native_reader_v1 import grid_5min
    day=observation_day or int(now.strftime('%Y%m%d'));asof=int(now.strftime('%Y%m%d%H%M%S'))
    file=Path(base)/'strong_pullback/market_test'/(str(day)+'.json')
    prior=sorted(p for p in file.parent.glob('????????.json') if p.stem.isdigit() and int(p.stem)<day)
    old=json.loads(file.read_bytes()) if file.is_file() else json.loads(prior[-1].read_bytes()) if prior else {}
    if observation_day and observation_day>int(now.strftime('%Y%m%d')):
        # After close the authoritative source is LOCAL TDX. A stopped daytime
        # mirror (e.g. last update 14:30) must not block tomorrow's preparation.
        from tdx_native_reader_v1 import read_lc5_file
        source=index_minutes_v1.SOURCE_FILES['000001.SH'];info=read_lc5_file(source)
        frame=pd.DataFrame(info['rows']).rename(columns={'ts':'date','raw_volume':'volume'})
        meta=dict(sha256=info['sha256'],contract='LOCAL_TDX_POSTCLOSE_MAP')
    else:
        source=index_minutes_v1.MIRROR/'sse_5min.parquet'
        if source.stat().st_size>4*1024**2:raise ValueError('PULLBACK_INDEX_SIZE_CAP')
        raw=source.read_bytes();meta=json.loads(source.with_suffix('.meta.json').read_bytes())
        if hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise ValueError('PULLBACK_INDEX_HASH_MISMATCH')
        frame=pd.read_parquet(io.BytesIO(raw)).rename(columns={'ts':'date','raw_volume':'volume'})
        meta['contract']='PUBLIC_INDEX_MIRROR'
    frame=frame[frame.date<=asof]
    if old.get('day')==day:
        level=old['level'];known=old['known_at'];t0=old.get('t0')
        level_sha=old.get('level_source_sha256',old.get('source_sha256'))
    else:
        past=frame[frame.date//1000000<day]
        if past.empty:raise ValueError('PULLBACK_TEST_PREVIOUS_SESSION_MISSING')
        p=int(past.date.iloc[-1]);last=past[past.date//1000000==p//1000000]
        calendar=list(map(int,trading_calendar()))
        if day not in calendar or calendar.index(day)==0 or calendar[calendar.index(day)-1]!=p//1000000:
            raise ValueError('PULLBACK_TEST_MARKET_PREVIOUS_DAY_MISMATCH')
        if last.date.tolist()!=grid_5min(p//1000000):raise ValueError('PULLBACK_TEST_PREVIOUS_SESSION_INCOMPLETE')
        mapping=index_map(past,p);close=float(past.close.iloc[-1])
        levels=[v['price'] for v in mapping.get('levels',[]) if v['price']<=close]
        if not levels:raise ValueError('PULLBACK_TEST_MAP_EMPTY')
        level=max(levels);known=asof;t0=old.get('t0') if old.get('level')==level else None
        level_sha=meta['sha256']
    today=frame[(frame.date//1000000==day)&(frame.date<=asof)]
    expected=[t for t in grid_5min(day) if t<=asof]
    prefix=today.date.tolist()==expected[:len(today)]
    prefix=prefix and all(all(math.isfinite(v) and v>0 for v in (r.open,r.high,r.low,r.close))
        and r.low<=min(r.open,r.close)<=max(r.open,r.close)<=r.high for r in today.itertuples())
    if not t0 and prefix:
        touched=False
        for r in today.itertuples():
            touched=touched or r.low<=level*1.005
            if touched and r.close>=level and int(r.date)>known:
                t0=int(r.date);break
    save(file,dict(day=day,level=level,known_at=known,t0=t0,asof=asof,source_sha256=meta['sha256'],
        source_contract=meta['contract'],level_source_sha256=level_sha,
        basis='PREVIOUS_SESSION_FROZEN_MAP; T0_COMPLETE_5M_ONLY',historical_t0_reconstructed=False))
    return market_state(quote,level,now,t0,list(map(int,trading_calendar())))


def volume_context(code,minutes,cutoff):
    """Actual 20-session same-5m cumulative share volume, never whole-day mean.

    Arrival does not wait for it. Before 09:35 the current ratio is unknown.
    """
    from tdx_native_history_v1 import locate
    from tdx_native_reader_v1 import read_lc5_file,validate_day,grid_5min
    from tarot_live_v2 import trading_calendar
    from relative_turnover_v1 import GRID,finite
    day=cutoff//1000000;grid=[t for t in grid_5min(day) if t<=cutoff]
    if not grid:return dict(status='UNKNOWN',reason='首根5分钟量尚未形成')
    end=grid[-1];clock=end%1000000
    history=read_lc5_file(locate(code))['rows']
    dates=[int(d) for d in trading_calendar() if int(d)<day][-20:]
    if len(dates)!=20:return dict(status='UNKNOWN',reason='不足20个历史交易日')
    totals=[]
    for d in dates:
        rows=[r for r in history if r['day']==d]
        if not validate_day(rows,d)['eligible']:return dict(status='UNKNOWN',reason='20日同时间段历史不完整')
        totals.append(sum(r['raw_volume'] for r in rows if r['ts']%1000000<=clock))
    current=[r for r in minutes if int(r['date'])//1000000==day and int(r['date'])<=end
             and int(r['date'])%1000000 not in (93000,130000)]
    if [int(r['date'])%1000000 for r in current]!=[t for t in GRID if t<=clock]:
        return dict(status='UNKNOWN',reason='当日同截止分钟量不完整')
    if any(str(r.get('code')).zfill(6)!=code or not finite(r.get('volume')) or r['volume']<0 for r in current):
        return dict(status='UNKNOWN',reason='当日成交量身份或单位异常')
    mean=sum(totals)/20
    if mean<=0:return dict(status='UNKNOWN',reason='历史同时间段均量为零')
    return dict(status='KNOWN',ratio=sum(r['volume'] for r in current)/mean,asof=end,
        basis='20交易日同一5分钟截止累计成交股数；非全天量比',provisional=True)
