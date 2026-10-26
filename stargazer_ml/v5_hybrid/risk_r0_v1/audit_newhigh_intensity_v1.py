"""Independent masks, exact-time aggregates and sklearn tied-score AUC replay."""
import json
import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import diagnose_newhigh_intensity_v1 as d
from audit_direct_warning_fixed_fit_v1 import write_exclusive
from research_budget_v2 import bytes_used,TOTAL_CAP


def verify_interval(record,day_values):
    values=np.asarray(day_values,dtype=float)
    assert record['dates']==len(values)
    if len(values):
        np.testing.assert_allclose(record['equal_date_mean'],values.mean(),atol=1e-14)
    if len(values)>=10:
        rng=np.random.default_rng(42)
        starts=rng.integers(0,len(values)-9,size=(1000,math.ceil(len(values)/10)))
        ids=(starts[:,:,None]+np.arange(10)).reshape(1000,-1)[:,:len(values)]
        np.testing.assert_allclose(record['ci95'],np.quantile(values[ids].mean(1),[.025,.975]),atol=1e-14)
    elif 'ci95' in record:
        raise ValueError('UNSUPPORTED_DATE_INTERVAL')


def audit():
    root=d.OUT
    terminal=json.loads((root/'stop_record.json').read_bytes())
    manifest=json.loads((root/'manifest.json').read_bytes())
    if terminal['status']!='COMPLETE_NO_FIT' or d.prior.fit.previous.sha(root/'manifest.json')!=terminal['manifest_sha256']:
        raise ValueError('INTENSITY_TERMINAL_INVALID')
    for name,item in manifest['outputs'].items():
        if d.prior.fit.previous.sha(root/name)!=item['sha256']:
            raise ValueError('OUTPUT_DRIFT:'+name)
    plan=json.loads((root/'plan.json').read_bytes())
    for path,expected in plan['inputs'].items():
        if d.prior.fit.previous.sha(path)!=expected:
            raise ValueError('INPUT_DRIFT:'+path)
    config=json.loads((d.prior.fit.prep.OUT/'config.json').read_bytes())
    report=json.loads((root/'report.json').read_bytes())
    predictions=pd.read_parquet(d.prior.fit.OUT/'predictions.parquet')
    stored=pd.read_parquet(root/'daily_effects.parquet')
    checked=0;ranking_checks=0;total_rows=0;effects_checked=0
    for segment in d.prior.fit.SEGMENTS:
        frame=d.prior.load_segment(segment,config,predictions)
        total_rows+=len(frame)
        for field in plan['fields']:
            eligible=frame[d.prior.high_for(field)].eq(1)&frame[field].notna()
            work=frame.loc[eligible].copy()
            item=report['segments'][segment]['fields'][field]
            assert item['eligible']['rows']==len(work)
            masks=[work[field].eq(0),work[field].gt(0)&work[field].le(.25),
                work[field].gt(.25)&work[field].le(.5),work[field].gt(.5)&work[field].le(.75),work[field].gt(.75)&work[field].le(1)]
            assert sum(int(m.sum()) for m in masks)==len(work)
            for label,mask in zip(plan['bands'],masks):
                group=work.loc[mask];r=item['bands'][label]
                assert r['rows']==len(group)
                assert r['cutoffs']==group[d.TIME].nunique()
                assert r['dates']==(group[d.TIME]//1000000).nunique()
                if len(group):
                    assert r['positives']==int(group.y_tail_1200m.sum())
                    np.testing.assert_allclose(r['tail_rate'],group.y_tail_1200m.mean(),atol=1e-14)
                    cutoff_rate=group.groupby(d.TIME).y_tail_1200m.mean()
                    np.testing.assert_allclose(r['date_equal_cutoff_equal_tail_rate'],
                        cutoff_rate.groupby(cutoff_rate.index//1000000).mean().mean(),atol=1e-14)
                checked+=1
            shared=field.startswith('nh_mb')
            # Reconstruct strong/weak contrasts by separate series, not producer band/contrast functions.
            strong=work[work[field]>.5].groupby(d.TIME).y_tail_1200m.mean()
            weak=work[work[field]<=.25].groupby(d.TIME).y_tail_1200m.mean()
            if shared:
                strong=strong.groupby(strong.index//1000000).mean()
                weak=weak.groupby(weak.index//1000000).mean()
                matched=strong.index.intersection(weak.index).sort_values()
                daily=strong.loc[matched]-weak.loc[matched]
                expected=pd.DataFrame({'day':daily.index,'effect':daily.to_numpy(),'supported_cutoffs':0})
            else:
                matched=strong.index.intersection(weak.index).sort_values()
                cut=strong.loc[matched]-weak.loc[matched]
                daily=cut.groupby(cut.index//1000000).mean()
                counts=cut.groupby(cut.index//1000000).size()
                expected=pd.DataFrame({'day':daily.index,'effect':daily.to_numpy(),'supported_cutoffs':counts.to_numpy()})
                assert item['contrast']['supported_cutoffs']==len(cut)
            verify_interval(item['contrast'],daily.to_numpy())
            selected=stored[stored.segment.eq(segment)&stored.field.eq(field)&stored.kind.eq('contrast')]
            pd.testing.assert_frame_equal(selected[['day','effect','supported_cutoffs']].sort_values('day').reset_index(drop=True),
                expected.sort_values('day').reset_index(drop=True),check_dtype=False,atol=1e-14,rtol=0)
            effects_checked+=len(expected)
            if shared:
                assert item['within_cutoff_ranking']['status']=='NOT_IDENTIFIABLE_SHARED_ACROSS_STOCKS'
                continue
            rankings=[]
            for cutoff,group in work.groupby(d.TIME):
                if group.y_tail_1200m.nunique()==2:
                    rankings.append((cutoff//1000000,roc_auc_score(group.y_tail_1200m,group[field])))
            rank_frame=pd.DataFrame(rankings,columns=['day','effect'])
            expected=rank_frame.groupby('day').agg(effect=('effect','mean'),supported_cutoffs=('effect','size')).reset_index()
            verify_interval(item['within_cutoff_ranking'],expected.effect.to_numpy())
            assert item['within_cutoff_ranking']['mixed_label_cutoffs']==len(rankings)
            selected=stored[stored.segment.eq(segment)&stored.field.eq(field)&stored.kind.eq('rank_auc')]
            pd.testing.assert_frame_equal(selected[['day','effect','supported_cutoffs']].sort_values('day').reset_index(drop=True),
                expected.sort_values('day').reset_index(drop=True),check_dtype=False,atol=1e-14,rtol=0)
            effects_checked+=len(expected);ranking_checks+=len(rankings)
    assert total_rows==86319
    if report['short_horizon_benefit_proven'] or report['model_or_label_changes']:
        raise ValueError('UNSUPPORTED_PROMOTION_OR_MUTATION')
    answer=dict(status='PASS_INTENSITY_AND_EXACT_CUTOFF_DIAGNOSIS_NOT_FIT',
        inputs_verified=len(plan['inputs']),bands_checked=checked,
        sklearn_cutoff_rankings_checked=ranking_checks,daily_effects_checked=effects_checked,
        source_rows=total_rows,report_sha256=d.prior.fit.previous.sha(root/'report.json'),
        bytes=sum(p.stat().st_size for p in root.iterdir()),registered_bytes=bytes_used(),cap=TOTAL_CAP,
        main_fits=0,calibration_fits=0,http_requests=0,new_labels=False)
    write_exclusive(d.HERE/'newhigh_intensity_acceptance_20261018_v1.json',answer)
    return answer


if __name__=='__main__':print(json.dumps(audit()))
