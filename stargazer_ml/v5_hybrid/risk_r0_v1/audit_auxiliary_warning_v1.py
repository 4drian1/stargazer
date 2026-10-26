"""Independent mean-downside contrasts and unordered-pair concordance audit."""
import itertools
import json

import numpy as np
import pandas as pd

import diagnose_auxiliary_120m_v1 as d
from audit_newhigh_intensity_v1 import verify_interval
from audit_direct_warning_fixed_fit_v1 import write_exclusive
from research_budget_v2 import bytes_used,TOTAL_CAP


def audit():
    root=d.OUT
    stop=json.loads((root/'stop_record.json').read_bytes())
    manifest=json.loads((root/'manifest.json').read_bytes())
    if stop['status']!='COMPLETE_NO_FIT' or d.labels.fit.previous.sha(root/'manifest.json')!=stop['manifest_sha256']:
        raise ValueError('WARNING_DIAGNOSTIC_TERMINAL_INVALID')
    for name,item in manifest['outputs'].items():
        if d.labels.fit.previous.sha(root/name)!=item['sha256']:raise ValueError('OUTPUT_DRIFT:'+name)
    plan=json.loads((root/'plan.json').read_bytes())
    for p,expected in plan['inputs'].items():
        if d.labels.fit.previous.sha(p)!=expected:raise ValueError('INPUT_DRIFT:'+p)
    config=json.loads((d.labels.fit.prep.OUT/'config.json').read_bytes())
    predictions=pd.read_parquet(d.labels.fit.OUT/'predictions.parquet')
    auxiliary=pd.read_parquet(d.labels.OUT/'labels.parquet')
    report=json.loads((root/'report.json').read_bytes())
    stored=pd.read_parquet(root/'daily_effects.parquet')
    bands_checked=0;pair_checks=0;row_count=0;daily_checks=0
    for segment in d.labels.fit.SEGMENTS:
        frame=d.load_frame(segment,config,predictions,auxiliary);row_count+=len(frame)
        for field in plan['fields']:
            eligible=frame[d.intensity.prior.high_for(field)].eq(1)&frame[field].notna()
            selected=frame.loc[eligible];shared=field.startswith('nh_mb')
            masks=[selected[field].eq(0),selected[field].gt(0)&selected[field].le(.25),
                selected[field].gt(.25)&selected[field].le(.5),selected[field].gt(.5)&selected[field].le(.75),selected[field].gt(.75)]
            item=report['segments'][segment]['fields'][field]
            for band,mask in zip(plan['bands'],masks):
                sub=selected.loc[mask];r=item['bands'][band]
                assert r['rows']==len(sub)
                assert r['cutoffs']==sub[d.TIME].nunique()
                if len(sub):np.testing.assert_allclose(r['mean_drop'],sub[d.TARGET].mean(),atol=1e-14)
                bands_checked+=1
            levels=[]
            for mask in [selected[field]>.5,selected[field]<=.25]:
                s=selected.loc[mask].groupby(d.TIME)[d.TARGET].mean()
                if shared:s=s.groupby(s.index//1000000).mean()
                levels.append(s)
            matched=levels[0].index.intersection(levels[1].index).sort_values()
            delta=levels[0].loc[matched]-levels[1].loc[matched]
            daily=delta if shared else delta.groupby(delta.index//1000000).mean()
            verify_interval(item['contrast'],daily.to_numpy())
            for kind in ['drop_difference','concordance']:
                actual=stored[stored.segment.eq(segment)&stored.field.eq(field)&stored.kind.eq(kind)].sort_values('day')
                if kind=='drop_difference':
                    expected=daily
                elif shared:
                    assert item['within_cutoff_concordance']['status']=='NOT_IDENTIFIABLE_SHARED_ACROSS_STOCKS'
                    assert actual.empty
                    continue
                else:
                    rows=[];pairs_total=0;ties_total=0
                    for cutoff,g in selected.groupby(d.TIME,sort=True):
                        n=0;credit=0.
                        for a,b in itertools.combinations(g[[field,d.TARGET]].itertuples(index=False,name=None),2):
                            if a[1]==b[1]:continue
                            n+=1;pair_checks+=1
                            if a[0]==b[0]:credit+=.5;ties_total+=1
                            elif (a[0]>b[0])==(a[1]>b[1]):credit+=1
                        if n:rows.append((cutoff//1000000,credit/n));pairs_total+=n
                    ranks=pd.DataFrame(rows,columns=['day','score'])
                    expected=ranks.groupby('day').score.mean()
                    r=item['within_cutoff_concordance']
                    assert r['target_unequal_pairs']==pairs_total and r['feature_tied_pairs']==ties_total
                    verify_interval(r,expected.to_numpy())
                np.testing.assert_array_equal(actual.day.to_numpy(dtype=np.int64),expected.index.to_numpy(dtype=np.int64))
                np.testing.assert_allclose(actual.effect,expected.to_numpy(),atol=1e-14,rtol=0)
                daily_checks+=len(actual)
    assert row_count==86319 and report['threshold'] is None and not report['fit']
    result=dict(status='PASS_CONTINUOUS_SHORT_WARNING_DIAGNOSIS_NOT_FIT',
        inputs_verified=len(plan['inputs']),bands_checked=bands_checked,unequal_outcome_pairs_checked=pair_checks,
        daily_effects_checked=daily_checks,source_rows=row_count,report_sha256=d.labels.fit.previous.sha(root/'report.json'),
        bytes=sum(p.stat().st_size for p in root.iterdir()),registered_bytes=bytes_used(),cap=TOTAL_CAP,
        main_fits=0,calibration_fits=0,http_requests=0,binary_threshold_added=False)
    write_exclusive(d.HERE/'auxiliary_120m_warning_acceptance_20261018_v1.json',result)
    return result


if __name__=='__main__':print(json.dumps(audit()))
