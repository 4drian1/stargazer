"""Exploratory conditional diagnostics using existing labels, never a new fit."""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

import direct_warning_fixed_fit_v1 as fit
import repair_multiboard_observation_v1 as sidecar
from risk_fit_eval_v1 import block_indices

HERE,TIME=fit.HERE,fit.TIME
OUT=HERE/'newhigh_target_diagnosis_20261018_v1'
CAP=2*1024**2
POLICY=dict(block_dates=10,reps=1000,seed=42)
STATES=['index_not_new_high','new_high_no_stall','new_high_stall','unknown']
META=['code',TIME,'label_end_time','y_tail_1200m','mae_1200m']


def states(rate, high):
    rate,high=np.asarray(rate,dtype=float),np.asarray(high,dtype=float)
    if rate.shape!=high.shape or np.isinf(rate).any() or np.isinf(high).any():
        raise ValueError('INVALID_STATE_INPUT')
    if not np.isin(high[np.isfinite(high)],[0,1]).all() or ((rate[np.isfinite(rate)]<0)|(rate[np.isfinite(rate)]>1)).any():
        raise ValueError('INVALID_RATIO_OR_MARKET_STATE')
    if ((high==0)&(rate>0)).any():
        raise ValueError('POSITIVE_CONJUNCTION_WITHOUT_NEW_HIGH')
    known=np.isfinite(rate)&np.isfinite(high)
    out=np.full(len(rate),'unknown',dtype=object)
    out[known&(high==0)]='index_not_new_high'
    out[known&(high==1)&(rate==0)]='new_high_no_stall'
    out[known&(high==1)&(rate>0)]='new_high_stall'
    return out


def group_stats(frame,mask,score_columns):
    sub=frame.loc[mask]
    if sub.empty:
        return dict(rows=0,dates=0,cutoffs=0,stocks=0,tail_rate=None,mean_mae=None,predictions={})
    y=sub.y_tail_1200m.to_numpy()
    daily=sub.assign(day=sub[TIME]//1000000).groupby('day').y_tail_1200m.mean()
    return dict(rows=len(sub),dates=len(daily),cutoffs=int(sub[TIME].nunique()),stocks=int(sub.code.nunique()),
        positives=int(y.sum()),tail_rate=float(y.mean()),equal_date_tail_rate=float(daily.mean()),
        mean_mae=float(sub.mae_1200m.mean()),
        predictions={c:dict(mean_probability=float(sub[c].mean()),
            brier=float(np.mean((sub[c].to_numpy()-y)**2))) for c in score_columns})


def daily_counts(frame,state):
    data=pd.DataFrame(dict(day=frame[TIME].to_numpy()//1000000,state=state,
        y=frame.y_tail_1200m.to_numpy(),mae=frame.mae_1200m.to_numpy()))
    return data.groupby(['day','state'],sort=True).agg(rows=('y','size'),
        positives=('y','sum'),mae_sum=('mae','sum')).reset_index()


def paired_days(daily,policy=POLICY):
    counts=daily.pivot(index='day',columns='state',values='rows')
    positives=daily.pivot(index='day',columns='state',values='positives')
    needed=['new_high_no_stall','new_high_stall']
    if not set(needed)<=set(counts.columns):
        return dict(status='NOT_TESTED_NO_PAIRED_DATES',paired_dates=0)
    paired=(counts[needed]>0).all(axis=1)
    rates=positives.loc[paired,needed]/counts.loc[paired,needed]
    delta=(rates['new_high_stall']-rates['new_high_no_stall']).to_numpy()
    n=len(delta)
    if not n:
        return dict(status='NOT_TESTED_NO_PAIRED_DATES',paired_dates=0)
    result=dict(paired_dates=n,equal_date_tail_rate_difference=float(delta.mean()),
        same_day_matching_is_not_same_cutoff_or_causal_control=True,
        first_paired_day=int(rates.index[0]),last_paired_day=int(rates.index[-1]))
    if n>1:
        leave=(delta.sum()-delta)/(n-1)
        result['leave_one_date_out_range']=[float(leave.min()),float(leave.max())]
    if n<policy['block_dates']:
        return dict(result,status='NOT_TESTED_TOO_FEW_DATES')
    draws=block_indices(n,policy['reps'],policy['block_dates'],policy['seed'])
    return dict(result,status='EXPLORATORY_EQUAL_DATE_MOVING_BLOCK',policy=policy,
        ci95=np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist(),
        multiple_comparisons_not_adjusted=True,no_promotion_or_feature_selection=True)


def load_segment(segment,config,predictions):
    frame=fit.prep.load_arm(segment,'C1')
    predicted=predictions[predictions.segment.eq(segment)].reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[META],predicted[META])
    high=sidecar.read_context(segment)[['code',TIME,*[
        f'nh_{benchmark}{frequency}_close_new_day_high'
        for benchmark in ['sse','cyb'] for frequency in [5,15]]]]
    if high.duplicated(['code',TIME]).any():
        raise ValueError('DUPLICATE_MARKET_STATE_KEY')
    frame=frame.merge(high,on=['code',TIME],how='left',validate='one_to_one',sort=False)
    for column in predicted:
        if column.startswith(('C0_','C1_')):
            frame[column]=predicted[column].to_numpy()
    if frame.duplicated(['code',TIME]).any() or not np.isfinite(frame[['y_tail_1200m','mae_1200m']]).all().all():
        raise ValueError('INVALID_FROZEN_COHORT')
    if len(frame)!=len(predicted) or frame.y_tail_1200m.sum()!=predicted.y_tail_1200m.sum():
        raise ValueError('COHORT_OR_LABEL_DRIFT')
    return frame


def high_for(field):
    parts=field.split('_')
    frequency=parts[1][3:] if parts[1].startswith('own') else parts[1][2:]
    return f'nh_{parts[2]}{frequency}_close_new_day_high'


def analyze(frame,field,score_columns):
    state=states(frame[field],frame[high_for(field)])
    daily=daily_counts(frame,state)
    monthly={}
    months=frame[TIME].to_numpy()//100000000
    for month in np.unique(months):
        monthly[str(month)]={s:group_stats(frame,(months==month)&(state==s),score_columns)
                             for s in ['new_high_no_stall','new_high_stall']}
    return dict(groups={s:group_stats(frame,state==s,score_columns) for s in STATES},
        monthly=monthly,conditional_same_date_contrast=paired_days(daily),
        interpretation='Stall minus no-stall only when index is new-high; descriptive, not causal'),daily


def references():
    fit.prep.check()
    plan=json.loads((fit.OUT/'plan.json').read_bytes())
    refs=dict(plan['inputs'])
    for path in [fit.OUT/'predictions.parquet',fit.OUT/'report.json',fit.OUT/'artifact_manifest.json',
                 fit.OUT/'stop_record.json',HERE/'direct_warning_fixed_fit_acceptance_20261018_v1.json',
                 sidecar.OUT/'shared.parquet',sidecar.OUT/'own.parquet',
                 Path(__file__),HERE/'test_newhigh_target_diagnosis_v1.py',HERE/'audit_newhigh_target_diagnosis_v1.py']:
        refs[str(path)]=fit.previous.sha(path)
    return refs


def run():
    import contextlib
    import unittest
    log=io.StringIO()
    with contextlib.redirect_stdout(log):
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromName('test_newhigh_target_diagnosis_v1'))
    if not result.wasSuccessful():
        raise ValueError(log.getvalue())
    accepted=json.loads((HERE/'direct_warning_fixed_fit_acceptance_20261018_v1.json').read_bytes())
    if accepted['status']!='PASS_FIXED_DIRECT_WARNING_RESEARCH_REPLAY_NOT_PROMOTION':
        raise ValueError('FIT_READBACK_REQUIRED')
    refs=references()
    manifest=json.loads((fit.OUT/'artifact_manifest.json').read_bytes())
    terminal=json.loads((fit.OUT/'stop_record.json').read_bytes())
    if fit.previous.sha(fit.OUT/'artifact_manifest.json')!=terminal['manifest_sha256'] or any(
            fit.previous.sha(fit.OUT/n)!=item['sha256'] for n,item in manifest['outputs'].items()):
        raise ValueError('FIT_ARTIFACT_DRIFT')
    config=json.loads((fit.prep.OUT/'config.json').read_bytes())
    fields=[f for f in config['added_fields'] if f.endswith('_rate')]
    if len(fields)!=16:
        raise ValueError('FROZEN_16_RATIOS_REQUIRED')
    writer=fit.base.EvidenceWriter(OUT,CAP,64*1024)
    writer.json('plan.json',dict(inputs=refs,fields=fields,signal='rate>0, no threshold search',
        primary_descriptive_comparison='index_new_high=1: positive ratio versus zero ratio; unknown excluded explicitly',
        uncertainty=POLICY,fit=False,new_targets=False,http_requests=0,output_cap=CAP,
        global_market_context_repeated_across_stocks_not_independent=True))
    try:
        predictions=pd.read_parquet(fit.OUT/'predictions.parquet')
        segments={};daily_tables=[]
        scores=[c for c in predictions if c.startswith(('C0_','C1_'))]
        for segment in fit.SEGMENTS:
            print(json.dumps(dict(stage='conditional_diagnosis',segment=segment)),flush=True)
            frame=load_segment(segment,config,predictions)
            findings={}
            for field in fields:
                if field.startswith('nh_mb') and frame.groupby(TIME)[field].nunique(dropna=False).gt(1).any():
                    raise ValueError('GLOBAL_CONTEXT_DIFFERS_ACROSS_SAME_CUTOFF')
                findings[field],daily=analyze(frame,field,scores)
                daily['segment'],daily['field']=segment,field
                daily_tables.append(daily)
            segments[segment]=dict(rows=len(frame),dates=int((frame[TIME]//1000000).nunique()),fields=findings)
        sink=io.BytesIO()
        pd.concat(daily_tables,ignore_index=True).to_parquet(sink,index=False,compression='zstd')
        writer.write('daily_support.parquet',sink.getvalue())
        writer.json('report.json',dict(status='COMPLETE_EXPLORATORY_NO_FIT',tests=result.testsRun,
            test_output=log.getvalue(),segments=segments,
            time_horizon_mismatch=dict(status='NOT_TESTED_NO_ACCEPTED_SHORT_HORIZON_TARGET',
                existing_target='future 1200 trading minutes / 8 percent adverse touch',
                short_horizon_benefit_proven=False,labels_changed=False,
                reason='Existing MAE and label_end_time do not tell whether a short intraday drop occurred or when the first touch happened'),
            no_feature_or_threshold_selection=True,causal_effect_identified=False,
            multiple_comparisons='all 16 definitions disclosed; exploratory, no winner or p-value gate',
            overlapping_labels_and_snapshot_membership_bias_preserved=True,
            train_in_sample_select_used_for_early_stop_calibration_in_sample_test_reused=True,
            main_fits=0,calibration_fits=0,http_requests=0,deployment=False))
        for path,expected in refs.items():
            if fit.previous.sha(path)!=expected:
                raise ValueError('INPUT_CHANGED:'+path)
        writer.json('manifest.json',dict(outputs=writer.outputs.copy()))
        writer.json('stop_record.json',dict(status='COMPLETE_NO_FIT',manifest_sha256=fit.previous.sha(OUT/'manifest.json')),terminal=True)
        return dict(status='COMPLETE_NO_FIT',bytes=writer.used,fields=16)
    except Exception as exc:
        writer.json('stop_record.json',dict(status='FAILED_NO_RESTART',error=f'{type(exc).__name__}:{exc}'),terminal=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args();print(json.dumps(run()))
