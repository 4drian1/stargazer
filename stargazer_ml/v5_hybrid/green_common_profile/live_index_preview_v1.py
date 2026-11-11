"""Index price/RSI preview; completed confirmations keep their own watermark."""
from pathlib import Path
from datetime import datetime
import json,sys

def observe(mirror,asof,quotes):
    import pandas as pd
    from golden_ladder_v3 import _rsi,index_map
    from live_five_preview_v1 import fifteen
    import math
    result={};day=asof//1000000
    for name in ('sse','cyb'):
        try:
            frame=pd.read_parquet(Path(mirror)/(name+'_5min.parquet')).rename(columns={'ts':'date'})
            frame=frame[frame.date<=asof].sort_values('date')
            meta=json.loads((Path(mirror)/(name+'_5min.meta.json')).read_bytes())
            watermark=int(meta.get('last_complete_ts') or 0)
            history=frame[frame.date//1000000<day];today=frame[frame.date//1000000==day]
            if history.empty:raise ValueError('INDEX_PREVIOUS_HISTORY_MISSING')
            grid=[day*1000000+m//60*10000+m%60*100 for m in [*range(575,691,5),*range(785,901,5)]]
            expected=[t for t in grid if t<=asof]
            if today.date.astype(int).tolist()!=expected:raise ValueError('INDEX_LIVE_PREFIX_GAP_WAIT_SYNC')
            values=[dict(day=int(r.date)//1000000,t=int(r.date)%1000000,nominal_end=int(r.date),
                complete=int(r.date)<day*1000000 or int(r.date)<=watermark,
                open=float(r.open),high=float(r.high),low=float(r.low),close=float(r.close),volume=None) for r in frame.itertuples()]
            quote=quotes.get('market.'+name) or {};stamp=int(quote.get('evidence_time') or 0);price=quote.get('price')
            age=(datetime.strptime(str(asof),'%Y%m%d%H%M%S')-datetime.strptime(str(stamp),'%Y%m%d%H%M%S')).total_seconds() if len(str(stamp))==14 else float('inf')
            if stamp//1000000==day and stamp<=asof and price and 0<=age<=120 and (not values or stamp>=values[-1]['day']*1000000+values[-1]['t']):
                minute=stamp%1000000//10000*60+stamp%10000//100
                end=(minute//5+1)*5;nominal=day*1000000+end//60*10000+end%60*100
                if values and values[-1]['nominal_end']==nominal and values[-1]['day']==day:
                    values[-1].update(close=price,high=max(values[-1]['high'],price),low=min(values[-1]['low'],price),complete=False,t=stamp%1000000)
                else:values.append(dict(day=day,t=stamp%1000000,nominal_end=nominal,complete=False,open=price,high=price,low=price,close=price,volume=None))
            if not values or values[-1]['day']!=day:raise ValueError('INDEX_CURRENT_PRICE_NOT_ARRIVED')
            def momentum(bars):
                closes=[b['close'] for b in bars];a,b=_rsi(closes,6),_rsi(closes,15)
                return dict(rsi6=float(a[-1]) if math.isfinite(a[-1]) else None,rsi15=float(b[-1]) if math.isfinite(b[-1]) else None,
                    evidence_time=bars[-1]['day']*1000000+bars[-1]['t'],provisional=not bars[-1].get('complete'))
            mapping=index_map(history,int(history.date.iloc[-1]));completed=[v for v in values if v['day']==day and v['complete']]
            levels=[dict(price=x['price'],two_completed_above=len(completed)>=2 and all(v['close']>=x['price'] for v in completed[-2:])) for x in mapping.get('levels',[])]
            result[name]=dict(status='OBSERVED_PRICE_PREVIEW',asof=asof,last_price=values[-1]['close'],five=momentum(values),
                fifteen=momentum(fifteen(values)),complete_watermark=watermark,levels=levels,
                map_reference_asof=int(history.date.iloc[-1]),provisional=True,volume_verified=False)
        except (OSError,ValueError,KeyError,TypeError) as exc:result[name]=dict(status='UNKNOWN',asof=asof,reason=str(exc)[:100])
    return result

if __name__=='__main__':
    request=json.load(sys.stdin);project=Path(request['project'])
    sys.path[:0]=[str(project/'stargazer_desktop'),str(project/'stargazer_ml/v5_hybrid/green_common_profile'),str(project/'stargazer_ml/v5_hybrid/risk_r0_v1')]
    print(json.dumps(observe(request['mirror'],request['asof'],request['quotes']),ensure_ascii=False,allow_nan=False))
