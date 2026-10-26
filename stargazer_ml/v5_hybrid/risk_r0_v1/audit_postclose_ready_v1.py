"""Independent next-session scalar replay and numeric/readiness readback."""
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from inventory_ths_stockdb_cache_v1 import HERE,digest
from build_decision_after_labels_v1 import admitted_evidence
from finalize_postclose_cohort_v3 import OUT,read_population
from postclose_training_v1 import check
from research_budget_v2 import bytes_used,TOTAL_CAP


def run():
    data,manifest,policy,parts=check()
    audit,_=admitted_evidence()
    base=json.loads((HERE/'postclose_joint_20261017_v1/manifest.json').read_bytes())
    raw_by_code={}
    for p,sha in base['inputs'].items():
        if not p.endswith('_daily.json.gz'):continue
        path=Path(p)
        if digest(path.read_bytes())!=sha:raise ValueError('DAILY_SOURCE_DRIFT')
        rows=json.loads(gzip.decompress(path.read_bytes()))
        raw_by_code[path.name[:6]]={int(k.rsplit(':',1)[1]):v for k,v in rows if v is not None}
    checked=0;ordinary_event_cases=0
    for row in data[data.label_state.eq('CONDITIONAL_NEXT_SESSION_COMPLETE')].itertuples(index=False):
        p=int(row.day);end=int(row.label_end_time)//1000000
        bars=raw_by_code[row.code]
        factor=1.
        for event in audit[row.code]['events']:
            if p<event['date']<=end and event.get('multiplier') is not None:
                assert event['status']=='ORDINARY_PARAMETERS_CORROBORATED'
                factor*=event['multiplier']
        expected=max(0.,1-bars[end]['low']*factor/bars[p]['close'])
        assert abs(expected-row.next_session_adverse_excursion)<1e-10
        assert row.label_end_time>row.decision_ts_assumed_end
        checked+=1;ordinary_event_cases+=int(factor!=1.)
    assert not data[['code','day']].duplicated().any()
    for name,frame in parts.items():
        assert not frame.proposed_purged.any()
        assert frame.research_cohort_eligible.all()
        assert (frame.decision_ts_assumed_end//1000000==frame.day).all()
    source=pd.read_parquet(HERE/'postclose_joint_20261017_v1/stock_day_population.parquet')
    for column in source:
        if column in data:
            if pd.api.types.is_numeric_dtype(source[column]):
                pd.testing.assert_series_equal(source[column],data[column])
            else:pd.testing.assert_series_equal(source[column].astype(object),data[column].astype(object))
    refs={str(OUT/'manifest.json'):digest((OUT/'manifest.json').read_bytes()),
          str(HERE/'postclose_training_v1.py'):digest((HERE/'postclose_training_v1.py').read_bytes()),
          str(HERE/'audit_postclose_ready_v1.py'):digest(Path(__file__).read_bytes()),
          str(HERE/'test_postclose_training_entry_v1.py'):digest((HERE/'test_postclose_training_entry_v1.py').read_bytes())}
    report=dict(status='PASS_ENGINEERING_READY_NOT_FIT_AUTHORIZATION',inputs=refs,
        conditional_target_scalar_replays=checked,ordinary_event_target_cases=ordinary_event_cases,
        target_value_tolerance=1e-10,population_rows=len(data),cohort_rows=sum(len(v) for v in parts.values()),
        fields=len(manifest['fields']),summary=manifest['summary'],engineering_ready=True,
        historical_representativeness_verified=False,production_gate_passed=False,
        fit_run=False,fit_authorized=False)
    body=json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2).encode()
    if bytes_used()+len(body)>TOTAL_CAP:raise ValueError('RESEARCH_BUDGET_STOP')
    with (OUT/'readback_acceptance.json').open('xb') as stream:stream.write(body)
    print(json.dumps(dict(status=report['status'],scalar_targets=checked,ordinary_event_cases=ordinary_event_cases,
                         research_bytes=bytes_used(),cap=TOTAL_CAP)))


if __name__=='__main__':run()
