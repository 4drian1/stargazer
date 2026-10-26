"""Continuous amplitude errors and equal-retention diagnostics, not probabilities."""
import numpy as np

from prepare_auxiliary_fixed_fit_v1 import TIME,TARGET
from prepare_multiboard_comparison_v1 import neutral_keep
from risk_fit_eval_v1 import block_indices
from diagnose_auxiliary_120m_v1 import continuous_concordance


def errors(y,p):
    y,p=np.asarray(y,dtype=float),np.asarray(p,dtype=float)
    if y.shape!=p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all() or ((y<0)|(y>=1)).any():
        raise ValueError('INVALID_AMPLITUDE_OR_PREDICTION')
    return dict(mse=float(np.mean((p-y)**2)),mae=float(np.mean(np.abs(p-y))),
        predicted_mean=float(p.mean()),actual_mean=float(y.mean()),bias=float((p-y).mean()),
        below_zero=int((p<0).sum()),above_one=int((p>1).sum()))


def paired_errors(y,a,b,dates,policy):
    y,a,b,dates=map(np.asarray,[y,a,b,dates])
    days,ids=np.unique(dates,return_inverse=True)
    result=dict(dates=len(days),status='NOT_TESTED_TOO_FEW_DATES')
    draws=block_indices(len(days),policy['reps'],policy['block_dates'],policy['seed']) if len(days)>=policy['block_dates'] else None
    counts=np.bincount(ids)
    for name,delta in [('mse',(a-y)**2-(b-y)**2),('mae',np.abs(a-y)-np.abs(b-y))]:
        item=dict(left_minus_right=float(delta.mean()))
        if draws is not None:
            sums=np.bincount(ids,weights=delta)
            item['ci95']=np.quantile(sums[draws].sum(1)/counts[draws].sum(1),[.025,.975]).tolist()
        result[name]=item
    if draws is not None:result.update(status='PAIRED_OBSERVED_DATE_BLOCK',policy=policy)
    return result


def retention(frame,scores,policy):
    y=frame[TARGET].to_numpy();times=frame[TIME].to_numpy()
    masks={};support=None
    for name,p in scores.items():
        keep,available=neutral_keep(frame.code.to_numpy(),times,p,policy['rejection_fraction'],policy['minimum_cross_section'])
        if support is not None and not np.array_equal(support,available):raise ValueError('UNPAIRED_RETENTION_SUPPORT')
        support=available;masks[name]=keep&available
    answer=dict(supported_rows=int(support.sum()),excluded_rows=int((~support).sum()),models={},paired_differences={})
    if not support.any():return dict(answer,status='NOT_TESTED_NO_SUPPORTED_CUTOFFS')
    cutoffs,ids=np.unique(times[support],return_inverse=True)
    n=np.bincount(ids);observed=np.bincount(ids,weights=y[support]);kept=n-np.floor(n*policy['rejection_fraction'])
    days,day_ids=np.unique(times[support]//1000000,return_inverse=True)
    random_daily=np.bincount(np.searchsorted(days,cutoffs//1000000),weights=observed*kept/n,minlength=len(days))
    totals={};denominator=None
    for name,mask in masks.items():
        counts=np.bincount(day_ids,weights=mask[support],minlength=len(days))
        if denominator is not None:np.testing.assert_array_equal(denominator,counts)
        denominator=counts
        totals[name]=np.bincount(day_ids,weights=y[support]*mask[support],minlength=len(days))
        answer['models'][name]=dict(retained_rows=int(mask.sum()),mean_observed_drop=float(y[mask].mean()))
    answer.update(status='DESCRIPTIVE_EQUAL_RETENTION',support_dates=len(days),support_cutoffs=len(cutoffs),
        random_expected_mean_drop=float(random_daily.sum()/denominator.sum()))
    totals['random_expected']=random_daily;u=policy['uncertainty']
    draws=block_indices(len(days),u['reps'],u['block_dates'],u['seed']) if len(days)>=u['block_dates'] else None
    for a,b in [('lightgbm','ridge'),('lightgbm','random_expected'),('ridge','random_expected')]:
        delta=totals[a]-totals[b]
        item=dict(left_minus_right=float(delta.sum()/denominator.sum()),status='NOT_TESTED_TOO_FEW_DATES')
        if draws is not None:item.update(status='PAIRED_OBSERVED_DATE_BLOCK',
            ci95=np.quantile(delta[draws].sum(1)/denominator[draws].sum(1),[.025,.975]).tolist())
        answer['paired_differences'][a+'_minus_'+b]=item
    return answer


def evaluate(frame,raw,train_mean,train_median,config):
    y=frame[TARGET].to_numpy();dates=frame[TIME].to_numpy()//1000000
    physical={k:np.clip(v,0,1) for k,v in raw.items()}
    scores=dict(physical,train_mean=np.full(len(frame),train_mean),train_median=np.full(len(frame),train_median))
    paired={}
    for a,b in [('lightgbm','ridge'),('lightgbm','train_mean'),('ridge','train_mean')]:
        paired[a+'_minus_'+b]=paired_errors(y,scores[a],scores[b],dates,config['evaluation']['uncertainty'])
    monthly={}
    for month in np.unique(dates//100):
        m=dates//100==month
        monthly[str(month)]=dict(rows=int(m.sum()),metrics={k:errors(y[m],v[m]) for k,v in scores.items()})
    ranks={}
    for name,values in physical.items():
        temp=frame[['code',TIME,TARGET]].copy();temp['score']=values
        ranks[name]=continuous_concordance(temp,'score',False)[0]
    return dict(rows=len(frame),physical_amplitude_metrics={k:errors(y,v) for k,v in scores.items()},
        raw_metrics={k:errors(y,v) for k,v in raw.items()},paired_errors=paired,monthly=monthly,
        same_cutoff_concordance=ranks,equal_retention=retention(frame,physical,config['evaluation']),
        probability_calibration_performed=False,pnl_evaluated=False,new_blind_test=False)
