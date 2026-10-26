"""Sparse admission repair; v1 diagnostic labels and the entire population remain frozen."""
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from build_decision_after_labels_v1 import OUT as BASE,SOURCE,CONTRACT,LEGACY,evidence_for
from inventory_ths_stockdb_cache_v1 import HERE,digest
from build_joint_risk_features_v1 import QUARANTINE
from decision_after_labels_v1 import minute_clock,price_grid,label_rows,STATES
from decision_after_basis_v2 import native_aligned_days
from read_decision_after_labels_v2 import OUT,read_stored_part


def run():
    if OUT.exists():raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if json.loads((BASE/'stop_record.json').read_bytes())['status']!='COMPLETE':raise ValueError('BASE_NOT_COMPLETE')
    from research_budget_v2 import bytes_used,TOTAL_CAP
    cap,reserve=16*1024**2,1024**2
    if bytes_used()+reserve>TOTAL_CAP:raise ValueError('REGISTERED_BUDGET_STOP')
    base=json.loads((BASE/'manifest.json').read_bytes());refs=base['inputs'].copy()
    for path in [BASE/'manifest.json',BASE/'readback_acceptance.json',Path(__file__),
                 HERE/'decision_after_basis_v2.py',HERE/'test_decision_after_basis_v2.py',HERE/'read_decision_after_labels_v2.py']:
        refs[str(path)]=digest(path.read_bytes())
    for name,sha in refs.items():
        if digest(Path(name).read_bytes())!=sha:raise ValueError('FROZEN_INPUT_DRIFT')
    audit=json.loads((LEGACY/'evidence_and_gate.json').read_bytes())['audit']
    sources={s['code']:s for s in base['source_files']}
    times=pq.read_table(SOURCE/'shared_market_and_leaders.parquet',columns=['decision_ts_assumed_end']).column(0).to_numpy()
    days=np.unique(times//1000000);clock=minute_clock(days)
    quarantine=defaultdict(set)
    with QUARANTINE.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):quarantine[row['code']].add(int(row['day']))
    OUT.mkdir(exist_ok=False);used=0;parts=[];quality={};states=Counter();dates=defaultdict(set)
    counts={name:dict(v,conditional_label_rows=0,conditional_unpurged_rows=0,
                       conditional_unpurged_positive=0,conditional_unpurged_nonpositive=0,
                       conditional_unpurged_dates=0) for name,v in base['proposed_split_counts'].items()}
    spec=json.loads(CONTRACT.read_bytes())
    def write(name,body,terminal=False):
        nonlocal used
        extra=0 if terminal else reserve
        if used+len(body)+extra>cap or bytes_used()+len(body)+extra>TOTAL_CAP:raise ValueError('CORRECTION_BUDGET_STOP')
        with (OUT/name).open('xb') as stream:stream.write(body)
        used+=len(body)
    status,error='COMPLETE',None
    try:
        write('plan.json',json.dumps(dict(inputs=refs,reason='NATIVE_BOUNDARY_OHLC_ADMISSION_CORRECTION',
            population_unchanged=True,target_clock_unchanged=True,raw_diagnostic_unchanged=True,
            tolerance_unchanged=.01,cap=cap,http_requests=0,fit_run=False),ensure_ascii=False,allow_nan=False).encode())
        for i,old in enumerate(base['parts']):
            body=(BASE/old['path']).read_bytes()
            if digest(body)!=old['sha256']:raise ValueError('BASE_PART_DRIFT')
            frame=pq.read_table(pa.BufferReader(body)).to_pandas();patch=[]
            for code,g in frame[frame.code.isin(audit)].groupby('code',sort=False):
                source=sources[code];raw=Path(source['path']).read_bytes()
                if digest(raw)!=source['sha256']:raise ValueError('MINUTE_INPUT_DRIFT')
                native=pq.read_table(pa.BufferReader(raw)).to_pandas()
                grid=price_grid(native,code,clock,quarantine.get(code,set()))
                evidence,old_quality=evidence_for(code,audit,grid,days,refs)
                # RAW daily bars are exactly the archived payload already bound by v1.
                import gzip
                daily=Path(next(p for p in refs if p.endswith(f'{code}_daily.json.gz')))
                bars={int(k.rsplit(':',1)[1]):v for k,v in json.loads(gzip.decompress(daily.read_bytes())) if v is not None}
                new_aligned=native_aligned_days(grid,days,bars,native)
                quality[code]=dict(old_aligned_days=old_quality['aligned_days'],aligned_days=int(new_aligned.sum()),
                    corrected_days=int((new_aligned&~evidence['aligned']).sum()),
                    still_unaligned_days=int((~new_aligned).sum()),coverage_verified=False)
                evidence['aligned']=new_aligned
                keys=g[['code','cutoff_index']].copy().reset_index(drop=True)
                keys['stock_close']=grid[keys.cutoff_index.to_numpy(dtype=int)*15+14,3]
                fresh,_=label_rows(keys,grid,days,evidence)
                for col in ['diagnostic_mae_raw_1200m','diagnostic_tail_raw_1200m']:
                    if not np.array_equal(g[col].to_numpy(),fresh[col].to_numpy(),equal_nan=True):
                        raise ValueError('BASIS_FIX_CHANGED_RAW_TARGET')
                patch.append(fresh[['code','cutoff_index','mae_1200m','y_tail_1200m','label_state']])
            part=dict(old,path='../'+BASE.name+'/'+old['path'])
            if patch:
                correction=pd.concat(patch,ignore_index=True);sink=pa.BufferOutputStream()
                pq.write_table(pa.Table.from_pandas(correction,preserve_index=False),sink,compression='zstd')
                encoded=sink.getvalue().to_pybytes();name=f'{i:04d}_basis.parquet';write(name,encoded)
                part.update(correction=name,correction_sha256=digest(encoded),correction_rows=len(correction),correction_bytes=len(encoded))
            parts.append(part)
            logical=read_stored_part(part)
            for state,n in logical.label_state.value_counts().items():states[STATES[int(state)]]+=int(n)
            for seg_index,seg in enumerate(spec['proposed_segments']):
                name=seg['name'];select=(logical.proposed_segment==seg_index).to_numpy()
                admitted=select&(logical.label_state.to_numpy()==0);keep=admitted&~logical.proposed_purged.to_numpy()
                counts[name]['conditional_label_rows']+=int(admitted.sum())
                counts[name]['conditional_unpurged_rows']+=int(keep.sum())
                counts[name]['conditional_unpurged_positive']+=int(logical.loc[keep,'y_tail_1200m'].sum())
                dates[name].update((times[logical.loc[keep,'cutoff_index'].to_numpy(dtype=int)]//1000000).tolist())
        for name,value in counts.items():
            value['conditional_unpurged_nonpositive']=value['conditional_unpurged_rows']-value['conditional_unpurged_positive']
            value['conditional_unpurged_dates']=len(dates[name])
        for name,sha in refs.items():
            if digest(Path(name).read_bytes())!=sha:raise ValueError('INPUT_CHANGED_DURING_CORRECTION')
        result=dict(base,parts=parts,inputs=refs,states=dict(states),proposed_split_counts=counts,
                    minute_daily_alignment=quality,basis_correction='NATIVE_BOUNDARY_RECORDS_AUDIT_ONLY',
                    base_manifest_sha256=digest((BASE/'manifest.json').read_bytes()),
                    population_unchanged=True,raw_diagnostic_unchanged=True,target_clock_unchanged=True)
        write('manifest.json',json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode())
    except Exception as exc:status,error='FAILED',f'{type(exc).__name__}:{exc}'
    write('stop_record.json',json.dumps(dict(status=status,error=error,parts=len(parts),bytes_before_stop=used),allow_nan=False).encode(),terminal=True)
    print(json.dumps(dict(status=status,error=error,parts=len(parts),evidence_stocks=len(quality),bytes_written=used)))
    return status


if __name__=='__main__':raise SystemExit(0 if run()=='COMPLETE' else 2)
