"""Ordinary bonus/cash event reconciliation for explicitly scoped historical research."""
from decimal import Decimal
import math


def number(value):
    if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('PARAMETER_MISSING_OR_INVALID')
    return Decimal(str(value))


def ordinary_multiplier(previous_close,cash,song_per10,zhuan_per10):
    p,c,s,z=map(number,(previous_close,cash,song_per10,zhuan_per10))
    if p<=0 or min(c,s,z)<0 or p<=c:raise ValueError('ORDINARY_FORMULA_NOT_APPLICABLE')
    return p*(1+s/10+z/10)/(p-c)


def reconcile_event(native,external,previous_close):
    if external.get('table')!='bonus' or external.get('reconcile_status')!='eligible' or external.get('progress')!='实施':
        return dict(status='UNSUPPORTED_EVENT',multiplier=None)
    try:
        parameters=external['bonus_ratio_per10']
        c=external['per_share_cash'];s=parameters['song'];z=parameters['zhuan']
        exact=ordinary_multiplier(previous_close,c,s,z)
        if (number(native.get('div'))!=number(c) or number(native.get('give'))!=number(s)/10
                or number(native.get('trans'))!=number(z)/10):
            return dict(status='CROSS_SOURCE_PARAMETER_MISMATCH',multiplier=None)
        deviation=abs(exact-number(native.get('mult')))
        if deviation>Decimal('0.000500000001'):
            return dict(status='MULT_PRECISION_MISMATCH',multiplier=None,absolute_diff=float(deviation))
        return dict(status='ORDINARY_PARAMETERS_CORROBORATED',multiplier=float(exact),
                    exact_decimal=str(exact),stored_mult=native['mult'],
                    stored_rounded_to_one=native['mult']==1 and any(number(v)>0 for v in (c,s,z)),
                    absolute_diff=float(deviation),coverage_verified=False)
    except (KeyError,ValueError,TypeError):
        return dict(status='PARAMETER_MISSING_OR_INVALID',multiplier=None)
