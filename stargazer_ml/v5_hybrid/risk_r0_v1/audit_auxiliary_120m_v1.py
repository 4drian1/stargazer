"""Direct per-window independent auxiliary target readback, no rolling reuse."""
import json

import numpy as np
import pandas as pd

import build_auxiliary_120m_v1 as b
from audit_direct_warning_fixed_fit_v1 import write_exclusive
from research_budget_v2 import bytes_used,TOTAL_CAP


def audit():
    terminal=json.loads((b.OUT/'stop_record.json').read_bytes())
    manifest=json.loads((b.OUT/'manifest.json').read_bytes())
    if terminal['status']!='COMPLETE_NOT_FIT' or b.fit.previous.sha(b.OUT/'manifest.json')!=terminal['manifest_sha256']:
        raise ValueError('AUXILIARY_TERMINAL_INVALID')
    for name,item in manifest['outputs'].items():
        if b.fit.previous.sha(b.OUT/name)!=item['sha256']:
            raise ValueError('AUXILIARY_OUTPUT_DRIFT:'+name)
    plan=json.loads((b.OUT/'plan.json').read_bytes())
    for path,expected in plan['inputs'].items():
        if b.fit.previous.sha(path)!=expected:
            raise ValueError('AUXILIARY_INPUT_DRIFT:'+path)
    meta=b.metadata()
    labels=pd.read_parquet(b.OUT/'labels.parquet')
    parent=meta['cohort']
    pd.testing.assert_frame_equal(labels[['code',b.TIME,'segment']],parent[['code',b.TIME,'segment']])
    if len(labels)!=86319 or labels.duplicated(['code',b.TIME]).any():
        raise ValueError('AUXILIARY_POPULATION_CHANGED')
    if labels.training_eligible.any() or labels.data_complete_time.notna().any():
        raise ValueError('UNAUTHORIZED_TRAINING_OR_FABRICATED_MATURITY')
    checked=0;max_error=0.;endpoints=0
    for code,frame in labels.groupby('code',sort=True):
        grid,evidence=b.load_code(code,meta)
        days=meta['days'];clock=meta['clock']
        for row in frame.itertuples(index=False):
            t=int(np.searchsorted(clock,getattr(row,b.TIME)))
            # Direct slicing is independent of the builder's rolling/shift implementation.
            future=np.arange(t+1,t+121)
            if len(future)!=120 or future[-1]>=len(clock) or not np.isfinite(grid[future]).all():
                raise ValueError('INCOMPLETE_AUXILIARY_WINDOW')
            factor=evidence['factors']
            expected=float(np.min(grid[future,2]*factor[future//240])/(grid[t,3]*factor[t//240])-1)
            actual=row.future_min_return_120m
            np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
            np.testing.assert_allclose(row.max_adverse_drop_120m,max(0.,-expected),atol=1e-12,rtol=0)
            assert row.auxiliary_end_time==clock[t+120]
            assert row.auxiliary_end_time>getattr(row,b.TIME)
            parent_row=parent.iloc[row.parent_position]
            assert row.max_adverse_drop_120m<=max(0.,-parent_row.mae_1200m)+1e-12
            assert row.auxiliary_end_time<=parent_row.label_end_time
            involved=np.arange(t//240,(t+120)//240+1)
            assert evidence['aligned'][involved].all() and not evidence['issues'][involved].any()
            max_error=max(max_error,abs(actual-expected));checked+=1;endpoints+=1
        print(json.dumps(dict(stage='independent_auxiliary_windows',code=code,checked=checked)),flush=True)
    report=json.loads((b.OUT/'report.json').read_bytes())
    for segment,frame in labels.groupby('segment',sort=False):
        item=report['segments'][segment]
        assert item['rows']==len(frame)
        assert item['zero_downside_rows']==int(frame.max_adverse_drop_120m.eq(0).sum())
        np.testing.assert_allclose(item['mean_max_adverse_drop'],frame.max_adverse_drop_120m.mean(),atol=1e-14)
    if report['threshold'] is not None or report['fit'] or report['warning_efficacy_proven']:
        raise ValueError('AUXILIARY_SCOPE_EXPANDED')
    # Immutable parent artifacts still match their accepted manifest.
    old=json.loads((b.fit.OUT/'artifact_manifest.json').read_bytes())
    for name,item in old['outputs'].items():
        if b.fit.previous.sha(b.fit.OUT/name)!=item['sha256']:
            raise ValueError('PARENT_FIT_CHANGED')
    answer=dict(status='PASS_AUXILIARY_120M_LABELS_NOT_FIT',rows=checked,window_endpoints_checked=endpoints,
        maximum_direct_slice_error=max_error,inputs_verified=len(plan['inputs']),
        report_sha256=b.fit.previous.sha(b.OUT/'report.json'),
        output_bytes=sum(p.stat().st_size for p in b.OUT.iterdir()),registered_bytes=bytes_used(),cap=TOTAL_CAP,
        original_main_target_unchanged=True,main_fits=0,calibration_fits=0,http_requests=0,
        new_binary_threshold=False,training_authorized=False)
    write_exclusive(b.HERE/'auxiliary_120m_acceptance_20261018_v1.json',answer)
    return answer


if __name__=='__main__':print(json.dumps(audit()))
