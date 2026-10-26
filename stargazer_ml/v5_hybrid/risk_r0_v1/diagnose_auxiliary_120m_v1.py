"""Fixed warning/continuous-short-target association; no classifiers or fit."""
import argparse
import io
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import build_auxiliary_120m_v1 as labels
import diagnose_newhigh_intensity_v1 as intensity

HERE,TIME=labels.HERE,labels.TIME
OUT=HERE/'auxiliary_120m_warning_diagnosis_20261018_v1'
TARGET='max_adverse_drop_120m'
CAP,RESERVE=1536*1024,64*1024


def stats(frame):
    if frame.empty:return dict(rows=0,cutoffs=0,dates=0,mean_drop=None,median_drop=None,q90_drop=None)
    cut=frame.groupby(TIME)[TARGET].mean()
    return dict(rows=len(frame),cutoffs=len(cut),dates=int((frame[TIME]//1000000).nunique()),
        mean_drop=float(frame[TARGET].mean()),median_drop=float(frame[TARGET].median()),
        q90_drop=float(frame[TARGET].quantile(.9)),
        date_cutoff_equal_mean_drop=float(cut.groupby(cut.index//1000000).mean().mean()))


def contrast(frame,field,shared):
    groups=[]
    for name,mask in [('strong',frame[field]>.5),('weak',frame[field]<=.25)]:
        s=frame.loc[mask].groupby(TIME)[TARGET].mean()
        if shared:s=s.groupby(s.index//1000000).mean()
        groups.append(s.rename(name))
    matched=pd.concat(groups,axis=1).dropna()
    difference=matched.strong-matched.weak
    if shared:
        daily=pd.DataFrame(dict(day=difference.index,effect=difference.to_numpy(),supported_cutoffs=0))
    else:
        average=difference.groupby(difference.index//1000000).mean()
        count=difference.groupby(difference.index//1000000).size()
        daily=pd.DataFrame(dict(day=average.index,effect=average.to_numpy(),supported_cutoffs=count.to_numpy()))
    report=intensity.daily_interval(daily,'effect')
    report.update(matching='SAME_DAY_DIFFERENT_CUTOFFS' if shared else 'EXACT_SAME_CUTOFF_DIFFERENT_STOCKS',
        supported_cutoffs=None if shared else len(matched),contrast='mean drop strong>0.50 minus weak<=0.25')
    return report,daily


def continuous_concordance(frame,field,shared):
    if shared:return dict(status='NOT_IDENTIFIABLE_SHARED_ACROSS_STOCKS'),pd.DataFrame()
    records=[];pairs=0;ties=0;credits=0.;flat_target=0
    for cutoff,g in frame.groupby(TIME,sort=True):
        x=g[field].to_numpy();y=g[TARGET].to_numpy()
        i,j=np.triu_indices(len(g),1)
        dx=x[i]-x[j];dy=y[i]-y[j];usable=dy!=0
        if not usable.any():flat_target+=1;continue
        tied=int(((dx==0)&usable).sum());n=int(usable.sum())
        credit=float(((dx*dy>0)&usable).sum()+.5*tied)
        records.append(dict(day=int(cutoff//1000000),effect=credit/n))
        pairs+=n;ties+=tied;credits+=credit
    if records:
        daily=pd.DataFrame(records).groupby('day').agg(effect=('effect','mean'),supported_cutoffs=('effect','size')).reset_index()
    else:daily=pd.DataFrame(columns=['day','effect','supported_cutoffs'])
    report=intensity.daily_interval(daily,'effect')
    report.update(target_unequal_pairs=pairs,feature_tied_pairs=ties,nonvariable_target_cutoffs=flat_target,
        supported_cutoffs=len(records),pair_weighted_concordance=credits/pairs if pairs else None,
        is_classification_accuracy=False,is_model_auc=False)
    return report,daily


def load_frame(segment,config,predictions,auxiliary):
    frame=intensity.prior.load_segment(segment,config,predictions)
    target=auxiliary[auxiliary.segment.eq(segment)].reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[['code',TIME]],target[['code',TIME]])
    if not np.isfinite(target[TARGET]).all() or (target[TARGET]<0).any():
        raise ValueError('INVALID_CONTINUOUS_AUXILIARY')
    return frame.merge(target[['code',TIME,TARGET]],on=['code',TIME],validate='one_to_one',sort=False)


def analyze(frame,field):
    state=intensity.prior.states(frame[field],frame[intensity.prior.high_for(field)])
    selected=frame.loc[(state=='new_high_stall')|(state=='new_high_no_stall')]
    shared=field.startswith('nh_mb')
    if shared and selected.groupby(TIME)[field].nunique().gt(1).any():raise ValueError('SHARED_CONTEXT_DRIFT')
    bands=intensity.bands(selected[field]);months=selected[TIME]//100000000
    c,cd=contrast(selected,field,shared);r,rd=continuous_concordance(selected,field,shared)
    for table,kind in [(cd,'drop_difference'),(rd,'concordance')]:table['kind']=kind
    return dict(eligible=stats(selected),bands={b:stats(selected.loc[bands==b]) for b in intensity.BANDS},
        monthly={str(m):{b:stats(selected.loc[(months==m)&(bands==b)]) for b in intensity.BANDS} for m in sorted(months.unique())},
        contrast=c,within_cutoff_concordance=r,unknown_rows=int((state=='unknown').sum()),
        index_not_new_high_rows=int((state=='index_not_new_high').sum())),pd.concat([cd,rd],ignore_index=True)


def run():
    log=io.StringIO()
    tests=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_auxiliary_warning_v1'))
    if not tests.wasSuccessful():raise ValueError(log.getvalue())
    acceptance=HERE/'auxiliary_120m_acceptance_20261018_v1.json'
    accepted=json.loads(acceptance.read_bytes())
    if accepted['status']!='PASS_AUXILIARY_120M_LABELS_NOT_FIT' or labels.fit.previous.sha(labels.OUT/'report.json')!=accepted['report_sha256']:
        raise ValueError('AUXILIARY_LABEL_ACCEPTANCE_REQUIRED')
    refs=dict(json.loads((labels.OUT/'plan.json').read_bytes())['inputs'])
    for p in [acceptance,labels.OUT/'labels.parquet',labels.OUT/'report.json',
              HERE/'diagnose_newhigh_intensity_v1.py',HERE/'diagnose_newhigh_target_v1.py',
              HERE/'audit_newhigh_intensity_v1.py',Path(__file__),HERE/'test_auxiliary_warning_v1.py',HERE/'audit_auxiliary_warning_v1.py']:
        refs[str(p)]=labels.fit.previous.sha(p)
    for p,expected in refs.items():
        if labels.fit.previous.sha(p)!=expected:raise ValueError('INPUT_DRIFT:'+p)
    config=json.loads((labels.fit.prep.OUT/'config.json').read_bytes())
    fields=[f for f in config['added_fields'] if f.endswith('_rate')]
    writer=labels.fit.base.EvidenceWriter(OUT,CAP,RESERVE)
    writer.json('plan.json',dict(inputs=refs,fields=fields,bands=intensity.BANDS,target=TARGET,
        strong='>0.50',weak='<=0.25',uncertainty=intensity.POLICY,output_cap=CAP,
        continuous_sorting='target-unequal pairs; concordant feature and downside direction credit1, feature ties credit0.5',
        exploratory=True,fit=False,http_requests=0))
    try:
        auxiliary=pd.read_parquet(labels.OUT/'labels.parquet')
        predictions=pd.read_parquet(labels.fit.OUT/'predictions.parquet')
        segments={};effects=[]
        for segment in labels.fit.SEGMENTS:
            print(json.dumps(dict(stage='short_warning_diagnosis',segment=segment)),flush=True)
            frame=load_frame(segment,config,predictions,auxiliary);results={}
            for field in fields:
                results[field],daily=analyze(frame,field)
                daily['segment'],daily['field']=segment,field;effects.append(daily)
            segments[segment]=dict(rows=len(frame),fields=results)
        sink=io.BytesIO();pd.concat(effects,ignore_index=True).to_parquet(sink,index=False,compression='zstd')
        writer.write('daily_effects.parquet',sink.getvalue())
        writer.json('report.json',dict(status='COMPLETE_AUXILIARY_WARNING_DIAGNOSIS_NOT_FIT',tests=tests.testsRun,
            test_output=log.getvalue(),segments=segments,target=TARGET,threshold=None,main_target_unchanged=True,
            fit=False,calibration_fit=False,http_requests=0,deployment=False,
            causal=False,new_blind_test=False,multiple_comparisons_adjusted=False,
            limitations=['same-cutoff stock composition confounding','overlapping 120-slot windows',
                'snapshot membership hindsight','restricted ordinary-event evidence','historical clock assumed',
                'old test reused and continuous concordance is not model accuracy']))
        for p,expected in refs.items():
            if labels.fit.previous.sha(p)!=expected:raise ValueError('INPUT_CHANGED:'+p)
        writer.json('manifest.json',dict(outputs=writer.outputs.copy()))
        writer.json('stop_record.json',dict(status='COMPLETE_NO_FIT',manifest_sha256=labels.fit.previous.sha(OUT/'manifest.json')),terminal=True)
        return dict(status='COMPLETE_NO_FIT',bytes=writer.used,fields=16)
    except Exception as exc:
        writer.json('stop_record.json',dict(status='FAILED_NO_RESTART',error=f'{type(exc).__name__}:{exc}'),terminal=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args();print(json.dumps(run()))
