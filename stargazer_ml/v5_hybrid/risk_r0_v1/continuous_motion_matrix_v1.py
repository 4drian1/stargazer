"""Factorized native 5/15 continuous features; no data IO, targets or fitting."""
import numpy as np

from multiboard_risk_features_v1 import divide

METRICS = ('deceleration','giveback',*[f'{b}_relative_{m}' for b in ('sse','cyb')
                                    for m in ('return','deceleration','giveback')])


def schema(frequency, own=False):
    if frequency not in (5,15):raise ValueError('INVALID_FREQUENCY')
    scopes = ['own'] if own else ['all','positive']
    names = [f'cm_{scope}{frequency}_{m}_{stat}' for scope in scopes
             for m in METRICS for stat in ('mean','coverage')]
    if own:
        names += [f'cm_own{frequency}_{group}_{m}_{stat}' for group in ('current','captured')
                  for m in ('deceleration','giveback') for stat in ('mean','coverage')]
    else:names += [f'cm_all{frequency}_deceleration_q{q}' for q in (10,50,90)]
    return names


def path_arrays(level, complete, frequency):
    """Same close-only semantics as accepted entity paths, day-local prefix."""
    x=np.asarray(level,dtype=float);proof=np.asarray(complete,dtype=bool)
    if frequency not in (5,15) or x.shape!=proof.shape or x.shape[-1]%(240//frequency):
        raise ValueError('INVALID_NATIVE_PATH_SHAPE')
    if np.isinf(x).any() or (x[np.isfinite(x)]<=0).any():raise ValueError('INVALID_LEVEL')
    width=240//frequency
    good=proof&np.isfinite(x)
    prev=np.roll(x,1,axis=-1); pair=good&np.roll(good,1,axis=-1)
    pair[...,::width]=False
    r=divide(x,prev)-1;r=np.where(pair,r,np.nan)
    dec=np.roll(r,1,axis=-1)-r;dec[...,::(width//2)]=np.nan
    shape=(*x.shape[:-1],-1,width)
    prefix=np.logical_and.accumulate(good.reshape(shape),axis=-1)
    high=np.maximum.accumulate(np.where(good.reshape(shape),x.reshape(shape),-np.inf),axis=-1)
    with np.errstate(invalid='ignore',divide='ignore'):
        give=np.where(prefix,1-x.reshape(shape)/high,np.nan).reshape(x.shape)
    give[...,::width]=np.nan
    return dict(return_=r,deceleration=dec,giveback=give)


def board_motion(returns,members,frequency,stocks,exclude=None):
    values=returns[members];valid=np.isfinite(values)
    winner=valid&(values==np.max(np.where(valid,values,-np.inf),axis=0))
    width=240//frequency
    capture=np.logical_or.accumulate(winner.reshape(len(members),-1,width),axis=-1)
    capture=np.roll(capture,1,axis=-1);capture[...,0]=False;capture=capture.reshape(values.shape)
    if exclude in members:
        i=members.index(exclude);valid=valid.copy();winner=winner.copy();capture=capture.copy()
        valid[i]=winner[i]=capture[i]=False
    total=len(members)-int(exclude in members);n=valid.sum(axis=0)
    mean=divide(np.where(valid,values,0).sum(axis=0),n)
    output=path_arrays(1+mean,(n==total)&(total>0),frequency)
    output['day_return']=mean
    for label,mask in [('current',winner),('captured',capture)]:
        for metric in ('deceleration','giveback'):
            a=stocks[metric][members];known=mask&np.isfinite(a);count=mask.sum(axis=0)
            result=divide(np.where(known,a,0).sum(axis=0),count)
            output[label+'_'+metric]=np.where((count>0)&(known.sum(axis=0)==count),result,np.nan)
    return output


def context(returns,memberships,markets,frequency,exclude=None,own=False,stocks=None):
    if stocks is None:stocks=path_arrays(1+returns,np.isfinite(returns),frequency)
    boards=[board_motion(returns,m,frequency,stocks,exclude) for m in memberships]
    width=returns.shape[-1]
    if not boards:return {n:np.full(width,np.nan) for n in schema(frequency,own)}
    values={k:np.stack([b[k] for b in boards]) for k in boards[0]}
    metrics={k:values[k] for k in ('deceleration','giveback')}
    for benchmark,x in markets.items():
        market=path_arrays(x,np.isfinite(x),frequency)
        for metric,key in [('return','return_'),('deceleration','deceleration'),('giveback','giveback')]:
            metrics[f'{benchmark}_relative_{metric}']=values[key]-market[key]
    result={}

    def stats(name,a,selected):
        known=selected&np.isfinite(a);n=known.sum(axis=0)
        result[name+'_mean']=divide(np.where(known,a,0).sum(axis=0),n)
        result[name+'_coverage']=divide(n,selected.sum(axis=0))

    all_rows=np.ones(values['day_return'].shape,dtype=bool)
    for scope in (['own'] if own else ['all','positive']):
        selected=all_rows if scope!='positive' else np.isfinite(values['day_return'])&(values['day_return']>0)
        for metric in METRICS:stats(f'cm_{scope}{frequency}_{metric}',metrics[metric],selected)
    if own:
        for group in ('current','captured'):
            for metric in ('deceleration','giveback'):
                stats(f'cm_own{frequency}_{group}_{metric}',values[group+'_'+metric],all_rows)
    else:
        a=values['deceleration'];has=np.isfinite(a).any(axis=0)
        for q in (10,50,90):
            out=np.full(width,np.nan)
            if has.any():out[has]=np.nanquantile(a[:,has],q/100,axis=0)
            result[f'cm_all{frequency}_deceleration_q{q}']=out
    if list(result)!=schema(frequency,own):raise ValueError('CONTINUOUS_SCHEMA_DRIFT')
    return result
