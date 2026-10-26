"""Independent conditional masks and source-group aggregation readback."""
import json
import math

import numpy as np
import pandas as pd

import diagnose_newhigh_target_v1 as d
from audit_direct_warning_fixed_fit_v1 import write_exclusive
from research_budget_v2 import bytes_used,TOTAL_CAP


def audit():
    root=d.OUT
    stop=json.loads((root/'stop_record.json').read_bytes())
    manifest=json.loads((root/'manifest.json').read_bytes())
    if stop['status']!='COMPLETE_NO_FIT' or d.fit.previous.sha(root/'manifest.json')!=stop['manifest_sha256']:
        raise ValueError('DIAGNOSTIC_TERMINAL_INVALID')
    for name,item in manifest['outputs'].items():
        if d.fit.previous.sha(root/name)!=item['sha256']:
            raise ValueError('DIAGNOSTIC_OUTPUT_DRIFT:'+name)
    plan=json.loads((root/'plan.json').read_bytes())
    for p,expected in plan['inputs'].items():
        if d.fit.previous.sha(p)!=expected:
            raise ValueError('DIAGNOSTIC_INPUT_DRIFT:'+p)
    report=json.loads((root/'report.json').read_bytes())
    config=json.loads((d.fit.prep.OUT/'config.json').read_bytes())
    predictions=pd.read_parquet(d.fit.OUT/'predictions.parquet')
    stored=pd.read_parquet(root/'daily_support.parquet')
    checked=0;total_rows=0;daily_checks=0
    for segment in d.fit.SEGMENTS:
        frame=d.load_segment(segment,config,predictions)
        days=frame[d.TIME]//1000000
        total_rows+=len(frame)
        for field in plan['fields']:
            rate=frame[field];high=frame[d.high_for(field)]
            known=rate.notna()&high.notna()
            masks=dict(index_not_new_high=known&high.eq(0),new_high_no_stall=known&high.eq(1)&rate.eq(0),
                new_high_stall=known&high.eq(1)&rate.gt(0),unknown=~known)
            item=report['segments'][segment]['fields'][field]
            expected=[]
            for state,mask in masks.items():
                sub=frame.loc[mask]
                actual=item['groups'][state]
                assert actual['rows']==len(sub)
                assert actual['cutoffs']==sub[d.TIME].nunique()
                assert actual['dates']==days[mask].nunique()
                if len(sub):
                    assert actual['positives']==int(sub.y_tail_1200m.sum())
                    np.testing.assert_allclose(actual['tail_rate'],sub.y_tail_1200m.mean(),atol=1e-14)
                    np.testing.assert_allclose(actual['mean_mae'],sub.mae_1200m.mean(),atol=1e-14)
                    for score,m in actual['predictions'].items():
                        np.testing.assert_allclose(m['mean_probability'],sub[score].mean(),atol=1e-14)
                    grouped=sub.assign(day=days[mask]).groupby('day')
                    for day,group in grouped:
                        expected.append(dict(day=int(day),state=state,rows=len(group),
                            positives=int(group.y_tail_1200m.sum()),mae_sum=float(group.mae_1200m.sum()),
                            segment=segment,field=field))
                checked+=1
            expected=pd.DataFrame(expected)
            actual=stored[stored.segment.eq(segment)&stored.field.eq(field)]
            keys=['day','state','segment','field']
            pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True),
                expected[list(actual)].sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=0)
            daily_checks+=len(expected)
            # Independent same-date means and bootstrap starts, not the producer's paired_days.
            a=frame.loc[masks['new_high_stall']].assign(day=days[masks['new_high_stall']]).groupby('day').y_tail_1200m.mean()
            b=frame.loc[masks['new_high_no_stall']].assign(day=days[masks['new_high_no_stall']]).groupby('day').y_tail_1200m.mean()
            common=a.index.intersection(b.index).sort_values()
            comparison=item['conditional_same_date_contrast']
            assert comparison['paired_dates']==len(common)
            if len(common):
                delta=(a.loc[common]-b.loc[common]).to_numpy()
                np.testing.assert_allclose(comparison['equal_date_tail_rate_difference'],delta.mean(),atol=1e-14)
                if len(common)>=10:
                    rng=np.random.default_rng(42)
                    starts=rng.integers(0,len(common)-9,size=(1000,math.ceil(len(common)/10)))
                    draws=np.concatenate([starts[:,:,None]+np.arange(10)],axis=2).reshape(1000,-1)[:,:len(common)]
                    np.testing.assert_allclose(comparison['ci95'],np.quantile(delta[draws].mean(1),[.025,.975]),atol=1e-14)
    if total_rows!=86319 or report['time_horizon_mismatch']['short_horizon_benefit_proven']:
        raise ValueError('POPULATION_OR_UNSUPPORTED_HORIZON_CLAIM')
    answer=dict(status='PASS_CONDITIONAL_DIAGNOSIS_NOT_CAUSAL_NOT_FIT',
        input_refs_verified=len(plan['inputs']),group_summaries_checked=checked,
        daily_rows_checked=daily_checks,source_population_rows=total_rows,
        report_sha256=d.fit.previous.sha(root/'report.json'),
        bytes=sum(p.stat().st_size for p in root.iterdir()),registered_bytes=bytes_used(),cap=TOTAL_CAP,
        main_fits=0,calibration_fits=0,http_requests=0,new_labels=False)
    write_exclusive(d.HERE/'newhigh_target_diagnosis_acceptance_20261018_v1.json',answer)
    return answer


if __name__=='__main__':
    print(json.dumps(audit()))
