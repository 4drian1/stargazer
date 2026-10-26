"""In-memory keyed sidecar, frozen source replay; no HTTP or disk-wide table."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from build_multiboard_full_features_v1 import SOURCE,daily_references,market_arrays
from continuous_motion_matrix_v1 import context,path_arrays,schema
from intraday_relative_stall_v1 import TIME,slots
from intraday_relative_stall_v2 import minute_returns
from materialize_relative_stall_pilot_v1 import HERE,sha
import bind_prior_anchor_evidence_v1 as binding


def fields():
    return [name for f in (5,15) for own in (False,True) for name in schema(f,own)]


def bindings():
    # Consume established research evidence; do not reopen raw coverage inventory.
    report=json.loads((HERE/'multiboard_full_features_20261018_v1/report.json').read_bytes())
    refs=dict(report['inputs'])
    for name in ['ths_full_concept_scope_20261017_v1.json','relative_stall_5m_20261018_v1/raw_input_ledger.json',
                 'local_market_context_20261017_v1/market_price_features.parquet',
                 'prior_anchor_evidence_acceptance_20261018_v1.json']:
        path=HERE/name;refs[str(path)]=sha(path)
    accepted=json.loads((HERE/'prior_anchor_evidence_acceptance_20261018_v1.json').read_bytes())
    r=json.loads((binding.OUT/'report.json').read_bytes())
    if sha(binding.OUT/'report.json')!=accepted['report_sha256']:raise ValueError('ANCHOR_ACCEPTANCE_DRIFT')
    refs.update(r['inputs'])
    refs[str(binding.OUT/'report.json')]=accepted['report_sha256']
    m=json.loads((binding.OUT/'manifest.json').read_bytes())
    refs[str(binding.OUT/'manifest.json')]=sha(binding.OUT/'manifest.json')
    for name,item in m['outputs'].items():refs[str(binding.OUT/name)]=item['sha256']
    old=json.loads((SOURCE/'manifest.json').read_bytes())
    for item in old['joint_parts']:
        p=SOURCE/item['path'];refs[str(p)]=old['output_hashes'][item['path']]['sha256']
    p=SOURCE/'shared_market_and_leaders.parquet';refs[str(p)]=old['output_hashes'][p.name]['sha256']
    return refs


def arrays(days):
    spec=json.loads((HERE/'ths_full_concept_scope_20261017_v1.json').read_bytes())
    if sha(spec['source'])!=spec['source_sha256']:raise ValueError('CATALOG_DRIFT')
    boards=sorted([r for r in pq.read_table(spec['source']).to_pylist()
                   if r['source']=='ths' and r['type']=='concept'],key=lambda r:r['code'])
    codes=sorted({c for b in boards for c in b['symbols']});ci={c:i for i,c in enumerate(codes)}
    members=[[ci[c] for c in b['symbols']] for b in boards]
    if len(codes)!=5540 or len(boards)!=978:raise ValueError('CATALOG_SCOPE_DRIFT')
    manifest=json.loads((SOURCE/'manifest.json').read_bytes())
    clock=pd.read_parquet(SOURCE/'shared_market_and_leaders.parquet',columns=['cutoff_index',TIME])
    clock=clock[clock[TIME].floordiv(1000000).isin(days)].sort_values(TIME)
    if len(clock)!=len(days)*16:raise ValueError('SOURCE_CLOCK_INCOMPLETE')
    di={d:i for i,d in enumerate(days)}
    dt={r.cutoff_index:di[r.decision_ts_assumed_end//1000000] for r in clock.itertuples()}
    positions={k:i for i,k in enumerate(clock.cutoff_index)}
    reference=np.full((len(codes),len(days)),np.nan)
    r15=np.full((len(codes),len(days)*16),np.nan)
    seen=set()
    for part in manifest['joint_parts']:
        p=SOURCE/part['path']
        if sha(p)!=manifest['output_hashes'][part['path']]['sha256']:raise ValueError('DERIVED_SOURCE_DRIFT')
        frame=pd.read_parquet(p,columns=['code','cutoff_index','stock_reference_close_raw','stock_day_return_raw'])
        present=set(frame.code.unique())
        if seen&present:raise ValueError('CODE_SPLIT_ACROSS_SOURCE_SHARDS')
        seen.update(present)
        daily_references(frame,ci,dt,reference,dict(values=r15,_map=positions))
    ordinary=pd.read_parquet(binding.OUT/'anchor_admission.parquet')
    ordinary=ordinary[ordinary.research_eligible]
    if len(ordinary)!=400 or ordinary.formal_admitted.any() or ordinary.coverage_verified.any() or not ordinary.ordinary_factor_ratio.eq(1).all():
        raise ValueError('ACCEPTED_RESTRICTED_ANCHOR_SCOPE_CHANGED')
    allowed=np.zeros_like(reference,dtype=bool)
    for row in ordinary.itertuples():
        if row.current_session_day not in di:continue
        i,j=ci[row.code],di[row.current_session_day]
        if np.isfinite(reference[i,j]):raise ValueError('ANCHOR_WOULD_OVERWRITE_EXISTING_REFERENCE')
        reference[i,j]=row.reference_close_potential;allowed[i,j]=True
    excluded={}
    with binding.prior.QUARANTINE.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):excluded.setdefault(row['code'],set()).add(int(row['day']))
    ledger=json.loads((HERE/'relative_stall_5m_20261018_v1/raw_input_ledger.json').read_bytes())
    if len(ledger)!=4196:raise ValueError('BOUND_RAW_SCOPE_CHANGED')
    r5=np.full((len(codes),len(days)*48),np.nan);compared=0
    for number,item in enumerate(ledger,1):
        code=item['code'];i=ci[code];body=Path(item['path']).read_bytes()
        if hashlib.sha256(body).hexdigest()!=item['sha256']:raise ValueError('FROZEN_RAW_DRIFT:'+code)
        frame=pq.read_table(pa.BufferReader(body),columns=['code','date','open','high','low','close','volume','amount']).to_pandas()
        values,_=minute_returns(frame,code,days,dict(zip(days,reference[i])),excluded.get(code,set()))
        old=r15[i].copy();known=np.isfinite(old)
        np.testing.assert_allclose(values[15][known],old[known],atol=1e-12,rtol=0)
        compared+=int(known.sum());r5[i]=values[5]
        patch=np.repeat(allowed[i],16)&~known
        r15[i,patch]=values[15][patch]
        if number%200==0:print(json.dumps(dict(stage='native_motion_source',files=number,total=len(ledger))),flush=True)
    return codes,members,{5:r5,15:r15},dict(raw_files=len(ledger),original_15m_cells_replayed=compared,
        restricted_unit_anchor_sessions=int(allowed.sum()),new_source_admission=False)


def build(parts):
    population=pd.concat([f[['code',TIME]].assign(segment=s,row=np.arange(len(f))) for s,f in parts.items()],ignore_index=True)
    if len(population)!=86319 or population.duplicated(['code',TIME]).any():raise ValueError('POPULATION_DRIFT')
    days=sorted(set(population[TIME]//1000000));di={d:i for i,d in enumerate(days)}
    codes,members,returns,report=arrays(days);ci={c:i for i,c in enumerate(codes)}
    market=pd.read_parquet(HERE/'local_market_context_20261017_v1/market_price_features.parquet')
    output=np.full((len(population),len(fields())),np.nan);column={f:i for i,f in enumerate(fields())}
    for month in sorted({d//100 for d in days}):
        md=[d for d in days if d//100==month];a,b=di[md[0]],di[md[-1]]+1
        rows=population.index[population[TIME].floordiv(100000000).eq(month)].to_numpy()
        local=population.loc[rows]
        for frequency in (5,15):
            width=240//frequency;r=returns[frequency][:,a*width:b*width]
            clock,markets=market_arrays(market,md,frequency);pos={t:i for i,t in enumerate(clock)}
            stocks=path_arrays(1+r,np.isfinite(r),frequency)
            shared=context(r,members,markets,frequency,stocks=stocks)
            at=np.array([pos[t] for t in local[TIME]],dtype=int)
            for name,value in shared.items():output[rows,column[name]]=value[at]
            for code,sub in local.groupby('code',sort=False):
                candidate=ci[code];related=[m for m in members if candidate in m]
                own=context(r,related,markets,frequency,exclude=candidate,own=True,stocks=stocks)
                idx=sub.index.to_numpy();at=np.array([pos[t] for t in sub[TIME]],dtype=int)
                for name,value in own.items():output[idx,column[name]]=value[at]
        print(json.dumps(dict(stage='motion_month',month=month,rows=len(rows))),flush=True)
    if np.isinf(output).any():raise ValueError('NONFINITE_FEATURE_RESULT')
    profiles={};fingerprints={}
    for s,frame in parts.items():
        idx=population.index[population.segment.eq(s)].to_numpy();values=output[idx]
        pd.testing.assert_frame_equal(population.loc[idx,['code',TIME]].reset_index(drop=True),frame[['code',TIME]])
        profiles[s]={f:dict(finite=int(np.isfinite(values[:,i]).sum()),rows=len(values)) for i,f in enumerate(fields())}
        for i,f in enumerate(fields()):
            v=values[:,i].astype('<f8',copy=True);v[np.isnan(v)]=np.nan
            fingerprints[s+'|'+f]=hashlib.sha256(v.tobytes()).hexdigest()
        addition=pd.DataFrame(values,columns=fields(),index=frame.index)
        parts[s]=pd.concat([frame,addition],axis=1)
    return parts,dict(report,profiles=profiles,column_fingerprints=fingerprints,
        new_fields=fields(),rows=len(population),persisted_wide_table=False)
