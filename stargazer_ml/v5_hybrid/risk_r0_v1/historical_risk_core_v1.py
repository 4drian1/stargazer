"""Historical stock-day label/features and purging. No IO, fit or candidate inputs."""
from bisect import bisect_left
from decimal import Decimal
import math
import statistics

STOCK_FIELDS = ('ret_1d','ret_5d','ret_20d','vol_5d','vol_20d',
                'downside_vol_20d','atr14_pct','range_pct_5d',
                'drawdown_20d','drawdown_60d','gap_last')
MARKET_FIELDS = ('market_ret_5d','market_ret_20d','market_vol_20d',
                 'market_downside_vol_20d','market_drawdown_20d')
PREFIXES = ('000','001','002','003','300','301','600','601','603','605','688')


def in_scope(code):
    return isinstance(code, str) and len(code) == 6 and code.isdigit() and code.startswith(PREFIXES)


def positive(x):
    return isinstance(x, (float,int)) and not isinstance(x,bool) and math.isfinite(x) and x > 0


def valid_bar(bar):
    return (bar is not None and all(positive(bar.get(k)) for k in ('open','high','low','close'))
            and bar['low'] <= min(bar['open'],bar['close'])
            <= max(bar['open'],bar['close']) <= bar['high'])


def checked_calendar(calendar):
    if not calendar or list(calendar) != sorted(set(calendar)):
        raise ValueError('CALENDAR_NOT_SORTED_UNIQUE')


def scaled_bar(bar, factor, anchor):
    if not valid_bar(bar) or not positive(factor) or not positive(anchor):
        return None
    out = {k:bar[k]*factor/anchor for k in ('open','high','low','close')}
    return out if valid_bar(out) else None


def label_at(bars, calendar, p, factors, *, scale_verified=False, coverage_verified=False):
    checked_calendar(calendar)
    if p not in calendar:
        raise ValueError('P_NOT_IN_CALENDAR')
    i = bisect_left(calendar,p)
    window = list(calendar[i+1:i+6])
    result = dict(mae_5d=None,y_tail_5d=None,window=window,
                  label_end_date=window[-1] if len(window)==5 else None)
    if len(window) != 5:
        return dict(result,status='CENSORED_TAIL')
    if not scale_verified or not coverage_verified:
        return dict(result,status='PRICE_EVIDENCE_PENDING')
    if not valid_bar(bars.get(p)):
        return dict(result,status='INVALID_REFERENCE')
    if not positive(factors.get(p)):
        return dict(result,status='INVALID_FACTOR')
    ratios = []
    for d in window:
        if not valid_bar(bars.get(d)) or not positive(factors.get(d)):
            return dict(result,status='DATA_INSUFFICIENT')
        ratio = (Decimal(str(bars[d]['low'])) * Decimal(str(factors[d]))
                 / Decimal(str(factors[p])) / Decimal(str(bars[p]['close'])) - 1)
        if not ratio.is_finite() or not math.isfinite(float(ratio)):
            return dict(result,status='INVALID_FACTOR')
        ratios.append(ratio)
    mae = min(ratios)
    return dict(result,mae_5d=float(mae),y_tail_5d=int(mae<=Decimal('-0.08')),
                status='LABEL_COMPLETE')


def stock_features(bars, calendar, p, factors):
    """Caller must admit price evidence separately; factors only accessed on dates <= P."""
    checked_calendar(calendar)
    if p not in calendar:
        raise ValueError('P_NOT_IN_CALENDAR')
    i = bisect_left(calendar,p)
    dates = list(calendar[max(0,i-60):i+1])
    nulls = {k:None for k in STOCK_FIELDS}
    if len(dates)!=61:
        return nulls, 'INSUFFICIENT_HISTORY'
    adjusted = [scaled_bar(bars.get(d),factors.get(d),factors.get(p)) for d in dates]
    if any(b is None for b in adjusted):
        return nulls, 'HISTORY_PRICE_OR_FACTOR_MISSING'
    c = [b['close'] for b in adjusted]
    r = [c[j]/c[j-1]-1 for j in range(1,len(c))]
    tr = [max(adjusted[j]['high']-adjusted[j]['low'],
              abs(adjusted[j]['high']-c[j-1]),abs(adjusted[j]['low']-c[j-1]))
          for j in range(1,len(c))]
    out = {f'ret_{n}d':c[-1]/c[-1-n]-1 for n in (1,5,20)}
    out.update({f'vol_{n}d':statistics.pstdev(r[-n:]) for n in (5,20)})
    out.update(downside_vol_20d=math.sqrt(sum(min(v,0)**2 for v in r[-20:])/20),
               atr14_pct=statistics.mean(tr[-14:])/c[-1],
               range_pct_5d=statistics.mean((adjusted[j]['high']-adjusted[j]['low'])/c[j-1]
                                            for j in range(len(c)-5,len(c))),
               drawdown_20d=c[-1]/max(c[-20:])-1,
               drawdown_60d=c[-1]/max(c[-60:])-1,
               gap_last=adjusted[-1]['open']/c[-2]-1)
    if not all(math.isfinite(v) for v in out.values()):
        return nulls, 'NONFINITE_FEATURE'
    return out, 'CALCULATED_EVIDENCE_ADMISSION_SEPARATE'


def segment(p, label_end):
    ranges = ((20190101,20231231,20240101,'train'),
              (20240101,20240630,20240701,'select'),
              (20240701,20241231,20250101,'calibration'),
              (20250101,20251231,20260101,'test'),
              (20260101,20260930,20261001,'description'))
    for start,end,boundary,name in ranges:
        if start <= p <= end:
            return name, label_end is not None and p < label_end < boundary
    return 'OUT_OF_SCOPE',False


def market_features(closes, calendar, p):
    """Price-index close input only. Caller binds index identity/source before admission."""
    checked_calendar(calendar)
    if p not in calendar:
        raise ValueError('P_NOT_IN_CALENDAR')
    i = bisect_left(calendar,p)
    days = list(calendar[max(0,i-20):i+1])
    if len(days)!=21 or any(not positive(closes.get(d)) for d in days):
        return {k:None for k in MARKET_FIELDS}, 'INDEX_WINDOW_INCOMPLETE'
    c = [closes[d] for d in days]
    r = [c[j]/c[j-1]-1 for j in range(1,len(c))]
    out = dict(market_ret_5d=c[-1]/c[-6]-1,market_ret_20d=c[-1]/c[0]-1,
               market_vol_20d=statistics.pstdev(r),
               market_downside_vol_20d=math.sqrt(sum(min(v,0)**2 for v in r)/20),
               market_drawdown_20d=c[-1]/max(c[-20:])-1)
    if not all(math.isfinite(v) for v in out.values()):
        return {k:None for k in MARKET_FIELDS}, 'INDEX_NONFINITE_FEATURE'
    return out, 'CALCULATED_INDEX_SOURCE_ADMISSION_SEPARATE'


def sample_gate(rows):
    """Only explicitly admitted, finite-feature, matured labels count; NULL never a negative."""
    thresholds = {'train':(2000,200,250),'select':(500,50,60),
                  'calibration':(500,50,60),'test':(500,50,60)}
    result = {}
    for name,(n_min,class_min,date_min) in thresholds.items():
        selected = []
        seen = set()
        for row in rows:
            seg,within = segment(row['P'],row.get('label_end_date'))
            if seg!=name or not within or row.get('training_eligible') is not True:
                continue
            key = (row['code'],row['P'])
            if key in seen:
                raise ValueError('DUPLICATE_STOCK_DAY')
            seen.add(key)
            y = row.get('y_tail_5d')
            if type(y) is not int or y not in (0,1):
                raise ValueError('ADMITTED_LABEL_INVALID')
            features = row.get('features',{})
            if any(type(features.get(f)) not in (int,float) or not math.isfinite(features[f])
                   for f in STOCK_FIELDS+MARKET_FIELDS):
                raise ValueError('ADMITTED_FEATURE_INVALID')
            selected.append(row)
        pos = sum(r['y_tail_5d'] for r in selected)
        dates = len({r['P'] for r in selected})
        result[name] = dict(n=len(selected),positive=pos,nonpositive=len(selected)-pos,
                            dates=dates,pass_gate=len(selected)>=n_min and pos>=class_min
                            and len(selected)-pos>=class_min and dates>=date_min)
    return dict(training_ready=all(v['pass_gate'] for v in result.values()),segments=result)
