"""Fixed descriptive intensity bands and exact-cutoff risk diagnostics; no fit."""
import argparse
import io
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import diagnose_newhigh_target_v1 as prior
from risk_fit_eval_v1 import block_indices

HERE,TIME=prior.HERE,prior.TIME
OUT=HERE/'newhigh_intensity_diagnosis_20261018_v1'
CAP,RESERVE=2*1024**2,64*1024
BANDS=['zero','(0,.25]','(.25,.50]','(.50,.75]','(.75,1]']
POLICY=dict(block_dates=10,reps=1000,seed=42)


def bands(values):
    values=np.asarray(values,dtype=float)
    if np.isinf(values).any() or ((values[np.isfinite(values)]<0)|(values[np.isfinite(values)]>1)).any():
        raise ValueError('INVALID_INTENSITY')
    result=np.full(len(values),'unknown',dtype=object)
    for i,(lo,hi) in enumerate([(0,0),(0,.25),(.25,.5),(.5,.75),(.75,1)]):
        mask=(values==0) if i==0 else (values>lo)&(values<=hi)
        result[mask]=BANDS[i]
    return result


def summarize(frame):
    if frame.empty:
        return dict(rows=0,cutoffs=0,dates=0,stocks=0,tail_rate=None,mean_mae=None)
    cutoffs=frame.groupby(TIME).y_tail_1200m.mean()
    days=cutoffs.groupby(cutoffs.index//1000000).mean()
    return dict(rows=len(frame),cutoffs=len(cutoffs),dates=len(days),stocks=int(frame.code.nunique()),
        positives=int(frame.y_tail_1200m.sum()),tail_rate=float(frame.y_tail_1200m.mean()),
        cutoff_equal_tail_rate=float(cutoffs.mean()),date_equal_cutoff_equal_tail_rate=float(days.mean()),
        mean_mae=float(frame.mae_1200m.mean()))


def daily_interval(daily,metric):
    daily=daily.sort_values('day')
    x=daily[metric].to_numpy(dtype=float)
    if not len(x):
        return dict(status='NOT_TESTED_NO_SUPPORTED_DATES',dates=0)
    out=dict(dates=len(x),equal_date_mean=float(x.mean()),first_day=int(daily.day.iloc[0]),last_day=int(daily.day.iloc[-1]))
    if len(x)<POLICY['block_dates']:
        return dict(out,status='NOT_TESTED_TOO_FEW_DATES')
    indices=block_indices(len(x),POLICY['reps'],POLICY['block_dates'],POLICY['seed'])
    return dict(out,status='EXPLORATORY_DATE_BLOCK',policy=POLICY,
        ci95=np.quantile(x[indices].mean(1),[.025,.975]).tolist(),
        multiple_comparisons_not_adjusted=True,no_promotion_or_selection=True)


def contrasts(frame,field,shared):
    """No row or cutoff without both fixed groups is imputed into the contrast."""
    work=frame[[TIME,'y_tail_1200m',field]].copy()
    work['day']=work[TIME]//1000000
    work['level']=np.where(work[field]>.5,'strong',np.where(work[field]<=.25,'weak','middle'))
    work=work[work.level.ne('middle')]
    if shared:
        # Shared market background: first give each cutoff one observation, not each stock.
        work=work.groupby(['day',TIME,'level'],as_index=False).y_tail_1200m.mean()
        means=work.groupby(['day','level']).y_tail_1200m.mean().unstack('level')
    else:
        means=work.groupby(['day',TIME,'level']).y_tail_1200m.mean().unstack('level')
    if not {'strong','weak'}<=set(means.columns):
        daily=pd.DataFrame(columns=['day','effect','supported_cutoffs'])
    else:
        paired=means.dropna(subset=['strong','weak'])
        if shared:
            daily=pd.DataFrame(dict(day=paired.index,effect=(paired.strong-paired.weak).to_numpy(),supported_cutoffs=0))
        else:
            paired=paired.assign(effect=paired.strong-paired.weak)
            daily=paired.groupby('day').agg(effect=('effect','mean'),supported_cutoffs=('effect','size')).reset_index()
    result=daily_interval(daily,'effect')
    result.update(contrast='strong>0.50 minus weak<=0.25',
        matching='SAME_DAY_DIFFERENT_CUTOFFS' if shared else 'EXACT_SAME_CUTOFF_DIFFERENT_STOCKS',
        supported_cutoffs=None if shared else int(daily.supported_cutoffs.sum()),
        causal_effect_identified=False)
    return result,daily


def cutoff_ranking(frame,field,shared):
    if shared:
        return dict(status='NOT_IDENTIFIABLE_SHARED_ACROSS_STOCKS',
            reason='Same-cutoff board distribution is identical across candidate stocks'),pd.DataFrame()
    records=[];mixed=0;unmixed=0;all_tied=0;total_pairs=0;tied_pairs=0;credits=0.
    for cutoff,group in frame.groupby(TIME,sort=True):
        x=group[field].to_numpy();y=group.y_tail_1200m.to_numpy()
        positive=x[y==1];negative=x[y==0]
        if not len(positive) or not len(negative):
            unmixed+=1
            continue
        difference=positive[:,None]-negative[None,:]
        pairs=difference.size;ties=int((difference==0).sum())
        credit=float((difference>0).sum()+.5*ties)
        mixed+=1;total_pairs+=pairs;tied_pairs+=ties;credits+=credit
        all_tied+=int(ties==pairs)
        records.append(dict(day=int(cutoff//1000000),cutoff=int(cutoff),auc=credit/pairs))
    if records:
        daily=pd.DataFrame(records).groupby('day').agg(effect=('auc','mean'),supported_cutoffs=('auc','size')).reset_index()
    else:
        daily=pd.DataFrame(columns=['day','effect','supported_cutoffs'])
    result=daily_interval(daily,'effect')
    result.update(mixed_label_cutoffs=mixed,nonmixed_cutoffs=unmixed,all_score_tied_mixed_cutoffs=all_tied,
        positive_negative_pairs=total_pairs,tied_pairs=tied_pairs,
        pooled_pair_weighted_auc=credits/total_pairs if total_pairs else None,
        primary='each mixed-label cutoff equally within date, then each date equally; ties credit 0.5',
        causal_effect_identified=False)
    return result,daily


def analyze(frame,field):
    state=prior.states(frame[field],frame[prior.high_for(field)])
    known=(state=='new_high_no_stall')|(state=='new_high_stall')
    selected=frame.loc[known]
    shared=field.startswith('nh_mb')
    if shared and selected.groupby(TIME)[field].nunique(dropna=False).gt(1).any():
        raise ValueError('SHARED_FIELD_NOT_SHARED')
    labels=bands(selected[field])
    summaries={b:summarize(selected.loc[labels==b]) for b in BANDS}
    monthly={}
    months=selected[TIME]//100000000
    for month in sorted(months.unique()):
        monthly[str(month)]={b:summarize(selected.loc[(months==month)&(labels==b)]) for b in BANDS}
    contrast,days=contrasts(selected,field,shared)
    rank,rank_days=cutoff_ranking(selected,field,shared)
    for table,kind in [(days,'contrast'),(rank_days,'rank_auc')]:
        table['kind']=kind
    effects=pd.concat([days,rank_days],ignore_index=True)
    return dict(eligible=summarize(selected),excluded_unknown_rows=int((state=='unknown').sum()),
        excluded_index_not_new_high_rows=int((state=='index_not_new_high').sum()),bands=summaries,
        monthly=monthly,contrast=contrast,within_cutoff_ranking=rank,
        interpretation='observational after known index new-high; bands are not trading thresholds'),effects


def run():
    log=io.StringIO()
    tests=unittest.TextTestRunner(stream=log,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('test_newhigh_intensity_v1'))
    if not tests.wasSuccessful():
        raise ValueError(log.getvalue())
    acceptance=HERE/'newhigh_target_diagnosis_acceptance_20261018_v1.json'
    accepted=json.loads(acceptance.read_bytes())
    if accepted['status']!='PASS_CONDITIONAL_DIAGNOSIS_NOT_CAUSAL_NOT_FIT' or prior.fit.previous.sha(prior.OUT/'report.json')!=accepted['report_sha256']:
        raise ValueError('PARENT_DIAGNOSTIC_NOT_ACCEPTED')
    refs=dict(json.loads((prior.OUT/'plan.json').read_bytes())['inputs'])
    for path in [acceptance,prior.OUT/'report.json',Path(__file__),HERE/'test_newhigh_intensity_v1.py',
                 HERE/'audit_newhigh_intensity_v1.py']:
        refs[str(path)]=prior.fit.previous.sha(path)
    for path,expected in refs.items():
        if prior.fit.previous.sha(path)!=expected:
            raise ValueError('FROZEN_INPUT_DRIFT:'+path)
    config=json.loads((prior.fit.prep.OUT/'config.json').read_bytes())
    fields=[f for f in config['added_fields'] if f.endswith('_rate')]
    if len(fields)!=16:
        raise ValueError('FROZEN_SCHEMA_CHANGED')
    writer=prior.fit.base.EvidenceWriter(OUT,CAP,RESERVE)
    writer.json('plan.json',dict(inputs=refs,fields=fields,bands=BANDS,
        weak='<=0.25',strong='>0.50',uncertainty=POLICY,output_cap=CAP,
        design='posthoc descriptive follow-up after historical results; not an independent preregistered test',
        unchanged_target='future 1200 trading minutes, adverse touch -8%',
        labels_changed=False,fit=False,http_requests=0))
    try:
        predictions=pd.read_parquet(prior.fit.OUT/'predictions.parquet')
        segments={};effects=[]
        for segment in prior.fit.SEGMENTS:
            print(json.dumps(dict(stage='intensity_diagnosis',segment=segment)),flush=True)
            frame=prior.load_segment(segment,config,predictions)
            findings={}
            for field in fields:
                findings[field],daily=analyze(frame,field)
                daily['segment'],daily['field']=segment,field
                effects.append(daily)
            segments[segment]=dict(rows=len(frame),fields=findings)
        sink=io.BytesIO()
        pd.concat(effects,ignore_index=True).to_parquet(sink,index=False,compression='zstd')
        writer.write('daily_effects.parquet',sink.getvalue())
        writer.json('report.json',dict(status='COMPLETE_INTENSITY_DIAGNOSIS_NOT_FIT',tests=tests.testsRun,
            test_output=log.getvalue(),segments=segments,causal=False,model_or_label_changes=False,
            short_horizon_benefit_proven=False,main_fits=0,calibration_fits=0,http_requests=0,
            limitations=['same-cutoff comparison still has stock composition confounding',
                'same-day shared contrast is not same-cutoff identification','overlapping five-day labels',
                'snapshot membership and restricted corporate-action evidence','old test reused',
                'all definitions reported, no multiple-comparison adjustment or threshold selection']))
        for path,expected in refs.items():
            if prior.fit.previous.sha(path)!=expected:
                raise ValueError('INPUT_CHANGED:'+path)
        writer.json('manifest.json',dict(outputs=writer.outputs.copy()))
        writer.json('stop_record.json',dict(status='COMPLETE_NO_FIT',manifest_sha256=prior.fit.previous.sha(OUT/'manifest.json')),terminal=True)
        return dict(status='COMPLETE_NO_FIT',bytes=writer.used,fields=16)
    except Exception as exc:
        writer.json('stop_record.json',dict(status='FAILED_NO_RESTART',error=f'{type(exc).__name__}:{exc}'),terminal=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args();print(json.dumps(run()))
