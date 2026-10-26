"""Independent split arithmetic, frozen input hashes and exact D2 reuse check."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_postclose_temporal_validation_v1 import OUT,SPEC,CAP
from read_postclose_daily_context_repair_v1 import read_repaired_population
from finalize_postclose_cohort_v3 import corrected_masks
from postclose_repaired_training_v1 import OUT as FIT, prepare
from inventory_ths_stockdb_cache_v1 import digest
from research_budget_v2 import TOTAL_CAP,bytes_used


def run():
    report=json.loads((OUT/'report.json').read_bytes())
    for path,sha in report['inputs'].items():
        if digest(Path(path).read_bytes())!=sha: raise ValueError('PREPARATION_INPUT_DRIFT:'+path)
    for name,entry in report['output_hashes'].items():
        if digest((OUT/name).read_bytes())!=entry['sha256']: raise ValueError('OUTPUT_DRIFT:'+name)
    data,_=read_repaired_population();fields,policy,accepted_parts,_=prepare()
    base=data.label_state.eq('CONDITIONAL_NEXT_SESSION_COMPLETE')&np.logical_and.reduce(list(corrected_masks(data).values()))
    spec=json.loads(SPEC.read_bytes());checked=0
    for name,bounds in spec['folds'].items():
        assignment=pd.read_parquet(OUT/(name+'_assignment.parquet'))
        pd.testing.assert_frame_equal(data[['code','day']],assignment[['code','day']])
        for segment,(start,end) in bounds.items():
            actual=assignment.eligible&assignment.segment.eq(segment)
            expected=(data.day>=start)&(data.day<=end)&base&data.label_end_time.gt(data.decision_ts_assumed_end)&data.label_end_time.le(end*1000000+235959)
            if not np.array_equal(actual,expected):raise ValueError('SEGMENT_MASK_DRIFT')
            g=data.loc[actual]; entry=report['folds'][name]['summary'][segment]
            if (len(g),g.day.nunique(),g.code.nunique())!=(entry['rows'],entry['dates'],entry['stocks']):raise ValueError('COUNT_DRIFT')
            if not np.array_equal(assignment.loc[data.day.between(start,end),'segment'],np.repeat(segment,int(data.day.between(start,end).sum()))):raise ValueError('EVENT_MOVED_ACROSS_BOUNDARY')
            checked+=1
    d2=pd.read_parquet(OUT/'D2_assignment.parquet')
    if not np.array_equal(d2.eligible,data.research_cohort_eligible):raise ValueError('D2_COHORT_NOT_IDENTICAL')
    for segment,expected in accepted_parts.items():
        frame=data[d2.eligible&d2.segment.eq(segment)]
        pd.testing.assert_frame_equal(frame,expected)
    exposure=pd.read_parquet(OUT/'exposure_ledger.parquet')
    pd.testing.assert_frame_equal(exposure[['code','day']],data[['code','day']])
    result=dict(status='PASS_TEMPORAL_SPLIT_AND_EXPOSURE_READBACK',segment_masks_checked=checked,
        D2_exact_reuse_accepted=True,D2_reuse_directory=str(FIT),
        D2_reuse_note='Exact population, ordered fields, labels, complete frame values and frozen policy already checked; reuse accepted second-fit weights/predictions, do not refit D2',
        new_historical_fits_needed_if_authorized=['D1','D3'],
        independent_blind_validation_verified=False,actual_fit=False,
        preflight_report_sha256=digest((OUT/'report.json').read_bytes()),
        code_sha256=digest(Path(__file__).read_bytes()))
    body=json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8')
    if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())+len(body)>CAP or bytes_used()+len(body)>TOTAL_CAP:raise ValueError('ACCEPTANCE_BUDGET_STOP')
    with (OUT/'readback_acceptance.json').open('xb') as stream:stream.write(body)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':run()
