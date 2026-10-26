"""Strict mult reconstruction; continuity diagnostics never certify event completeness."""
from bisect import bisect_right
from datetime import datetime
import math


def valid_date(value):
    if type(value) is not int or len(str(value))!=8:
        raise ValueError('INVALID_EVENT_DATE')
    datetime.strptime(str(value),'%Y%m%d')
    return value


def canonical_events(records):
    """No default multiplier, no inferred effective date and no cum input to arithmetic."""
    by_day={}
    for record in records:
        day=valid_date(record['date'])
        mult=record.get('mult')
        if type(mult) not in (int,float) or not math.isfinite(mult) or mult<=0:
            raise ValueError('INVALID_EVENT_MULT')
        if day in by_day and by_day[day]!=mult:
            raise ValueError('CONFLICTING_EVENT_MULT')
        by_day[day]=mult
    return sorted(by_day.items())


def rebuild_mult_factors(records, dates):
    """Uses recorded event dates; callers must separately verify actual effective dates."""
    events=canonical_events(records)
    product=1.
    days=[];products=[]
    for day,mult in events:
        product*=mult
        if not math.isfinite(product) or product<=0: raise ValueError('INVALID_FACTOR_PRODUCT')
        days.append(day);products.append(product)
    out={}
    for d in dates:
        valid_date(d)
        i=bisect_right(days,d)-1
        out[d]=products[i] if i>=0 else 1.
    return out


def continuity_diagnostic(bars, records, tolerance=.12):
    """Exploratory engineering only. Empty checks cannot PASS; real moves may exceed tolerance."""
    if not 0<tolerance<1: raise ValueError('INVALID_DIAGNOSTIC_TOLERANCE')
    closes={}
    for b in bars:
        d=valid_date(b['date']);c=b.get('close')
        if type(c) not in (int,float) or not math.isfinite(c) or c<=0:
            raise ValueError('INVALID_DIAGNOSTIC_PRICE')
        if d in closes and closes[d]!=c: raise ValueError('CONFLICTING_CLOSE')
        closes[d]=c
    dates=sorted(closes)
    factors=rebuild_mult_factors(records,dates)
    events=canonical_events(records)
    effective=[d for d,m in events if m!=1]
    rows=[]
    for i,d in enumerate(dates):
        if i==0 or d not in effective: continue
        prev=dates[i-1]
        raw=closes[d]/closes[prev]-1
        adjusted=closes[d]*factors[d]/(closes[prev]*factors[prev])-1
        rows.append(dict(date=d,raw_return=raw,adjusted_return=adjusted,
                         inside_diagnostic_band=abs(adjusted)<tolerance))
    return dict(diagnostic_status='NOT_TESTED' if not rows else
                ('WITHIN_DIAGNOSTIC_BAND' if all(r['inside_diagnostic_band'] for r in rows)
                 else 'OUTSIDE_DIAGNOSTIC_BAND'),checks=rows,
                factor_method='PRODUCT_OF_MULT_AT_RECORDED_DATE',
                effective_dates_verified=False,scale_verdict='UNVERIFIED',
                corporate_action_coverage_verdict='UNVERIFIED',training_eligible=False)
