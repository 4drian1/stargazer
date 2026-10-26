"""Correct leader-family availability; no sample-floor changes or performance selection."""
import io
import json
import numpy as np
import pandas as pd
from inventory_ths_stockdb_cache_v1 import HERE,digest
from prepare_postclose_research_v2 import family_masks
from research_budget_v2 import bytes_used,TOTAL_CAP

PARENT=HERE/'postclose_research_20261017_v2'
OUT=HERE/'postclose_training_ready_20261017_v3'


def corrected_masks(data):
    masks=family_masks(data)
    # User's leader evidence includes contemporaneous daily return/tracking;
    # RSI warmup is missing one optional field, not absence of the whole family.
    masks['leader_daily_observation']=data[['context_current_leader_return_loo_raw',
        'context_captured_leader_mean_return_loo_raw']].notna().any(axis=1)
    return masks


def read_population(directory=OUT):
    manifest=json.loads((directory/'manifest.json').read_bytes())
    base=manifest['population']
    from pathlib import Path
    path=Path(base['path'])
    if digest(path.read_bytes())!=base['sha256']:raise ValueError('PARENT_DATA_DRIFT')
    data=pd.read_parquet(path)
    path=directory/'eligibility.parquet'
    if digest(path.read_bytes())!=manifest['outputs'][path.name]['sha256']:raise ValueError('ELIGIBILITY_DRIFT')
    eligibility=pd.read_parquet(path)
    if not data[['code','day']].equals(eligibility[['code','day']]):raise ValueError('ELIGIBILITY_KEY_DRIFT')
    data['research_cohort_eligible']=eligibility.research_cohort_eligible.to_numpy()
    return data,manifest


def run():
    if OUT.exists():raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    parent=json.loads((PARENT/'manifest.json').read_bytes())
    for p,sha in parent['inputs'].items():
        from pathlib import Path
        if digest(Path(p).read_bytes())!=sha:raise ValueError('PARENT_INPUT_DRIFT')
    path=PARENT/'stock_day_population.parquet'
    if digest(path.read_bytes())!=parent['outputs'][path.name]['sha256']:raise ValueError('PARENT_POPULATION_DRIFT')
    data=pd.read_parquet(path)
    masks=corrected_masks(data)
    eligible=data.label_state.eq('CONDITIONAL_NEXT_SESSION_COMPLETE')&~data.proposed_purged&np.logical_and.reduce(list(masks.values()))
    spec=json.loads((HERE/'postclose_research_policy_v1.json').read_bytes())
    all_fields=parent['fields']+parent['constant_train_fields_excluded']
    train=data[eligible&data.proposed_segment_name.eq('train')]
    constants=sorted(f for f in all_fields if train[f].nunique(dropna=True)<=1)
    fields=[f for f in all_fields if f not in constants]
    summary={};gates={}
    for name,threshold in spec['sample_gates'].items():
        g=data[eligible&data.proposed_segment_name.eq(name)]
        summary[name]=dict(rows=len(g),dates=int(g.day.nunique()),stocks=int(g.code.nunique()),
            target_nonzero=int((g.next_session_adverse_excursion>0).sum()),
            target_mean=float(g.next_session_adverse_excursion.mean()))
        gates[name]={k:summary[name][k]>=v for k,v in threshold.items()}
    manifest=dict(status='ENGINEERING_COHORT_READY_FIT_NOT_AUTHORIZED',
        population=dict(path=str(path),sha256=digest(path.read_bytes())),
        inputs={str(PARENT/'manifest.json'):digest((PARENT/'manifest.json').read_bytes()),
                str(HERE/'postclose_research_policy_v1.json'):digest((HERE/'postclose_research_policy_v1.json').read_bytes()),
                str(HERE/'finalize_postclose_cohort_v3.py'):digest((HERE/'finalize_postclose_cohort_v3.py').read_bytes()),
                str(HERE/'test_postclose_training_v1.py'):digest((HERE/'test_postclose_training_v1.py').read_bytes())},
        scope_correction='Daily leader return evidence counts as leader-family observation; RSI is optional, not a new gate',
        numeric_gates_unchanged=True,fields=fields,constant_train_fields_excluded=constants,
        summary=summary,sample_gates=gates,rows=len(data),cohort_rows=int(eligible.sum()),
        missing_family_rows={k:int((~v).sum()) for k,v in masks.items()},
        engineering_ready=all(all(v.values()) for v in gates.values()),fit_authorized=False,
        full_market_representativeness=False,fit_run=False,training_ready=False)
    e=data[['code','day']].copy();e['research_cohort_eligible']=eligible.to_numpy()
    sink=io.BytesIO();e.to_parquet(sink,index=False,compression='zstd')
    body=sink.getvalue();manifest['outputs']={'eligibility.parquet':dict(bytes=len(body),sha256=digest(body))}
    mb=json.dumps(manifest,ensure_ascii=False,allow_nan=False,indent=2).encode()
    if len(body)+len(mb)>1024**2 or bytes_used()+len(body)+len(mb)>TOTAL_CAP:raise ValueError('RESEARCH_BUDGET_STOP')
    OUT.mkdir(exist_ok=False)
    for name,payload in [('eligibility.parquet',body),('manifest.json',mb)]:
        with (OUT/name).open('xb') as stream:stream.write(payload)
    actual,_=read_population()
    pd.testing.assert_frame_equal(actual.drop(columns='research_cohort_eligible'),data.drop(columns='research_cohort_eligible'))
    assert np.array_equal(actual.research_cohort_eligible,eligible)
    print(json.dumps(dict(engineering_ready=manifest['engineering_ready'],fields=len(fields),summary=summary,gates=gates)))


if __name__=='__main__':run()
