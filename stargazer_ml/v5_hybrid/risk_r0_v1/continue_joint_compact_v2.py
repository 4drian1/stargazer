"""Finish a budget-stopped wide build using shared-context references.

Old shards are immutable derived inputs, not re-created raw-source lineage.
No quota reset, deletion, HTTP, labels or fit.
"""
from collections import defaultdict
import csv
import json
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from build_joint_risk_features_v1 import (
    HERE, ROOTS, CATALOG, AUTH, INVENTORY, QUARANTINE, MARKET,
    digest, stock_features, parquet_bytes,
)

OLD = HERE/'joint_risk_features_20261017_v1'
OUT = HERE/'joint_risk_features_20261017_v3'
CAP, RESERVE = 512*1024**2, 1024**2


def completed_codes(parts):
    seen=set()
    for path in parts:
        codes=set(pq.read_table(path,columns=['code']).column('code').to_pylist())
        if seen & codes:
            raise ValueError('CODE_SPANS_COMPLETED_SHARDS')
        seen.update(codes)
    return seen


def calendar_days(shared):
    return np.unique(shared.decision_ts_assumed_end.to_numpy(dtype=np.int64)//1000000)


def run():
    stop=json.loads((OLD/'stop_record.json').read_bytes())
    if stop['status']!='FAILED' or stop['error']!='ValueError:ACTUAL_SERIALIZED_BUDGET_STOP':
        raise ValueError('NOT_A_BUDGET_STOPPED_BUILD')
    if OUT.exists(): raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    from research_budget_v2 import bytes_used,TOTAL_CAP
    if bytes_used()+RESERVE>TOTAL_CAP: raise ValueError('REGISTERED_BUDGET_STOP')
    OUT.mkdir(exist_ok=False);used=0;hashes={};parts=[];board_parts=[];source_files=[]
    def write(name,body,terminal=False):
        nonlocal used
        reserve=0 if terminal else RESERVE
        if used+len(body)+reserve>CAP or bytes_used()+len(body)+reserve>TOTAL_CAP:
            raise ValueError('COMPACT_BUDGET_STOP')
        path=OUT/name;path.parent.mkdir(exist_ok=True)
        with path.open('xb') as stream:stream.write(body)
        used+=len(body);hashes[name]=dict(sha256=digest(body),bytes=len(body))
    def reference(path):
        name='../'+OLD.name+'/'+path.relative_to(OLD).as_posix()
        body=path.read_bytes();hashes[name]=dict(sha256=digest(body),bytes=len(body))
        return dict(path=name,rows=pq.ParquetFile(pa.BufferReader(body)).metadata.num_rows)
    status,error='COMPLETE',None
    try:
        refs={str(p):digest(p.read_bytes()) for p in [CATALOG,AUTH,INVENTORY,QUARANTINE,MARKET,
            OLD/'stop_record.json',HERE/'build_joint_risk_features_v1.py',Path(__file__)]}
        catalog=json.loads(CATALOG.read_bytes());source=Path(catalog['source'])
        refs[str(source)]=digest(source.read_bytes())
        auth=json.loads(AUTH.read_bytes())
        if not auth['allow_archived_member_price_reconstruction'] or refs[str(source)]!=auth['catalog_source_sha256']:
            raise ValueError('APPROVED_SCOPE_DRIFT')
        old_parts=sorted((OLD/'joint_parts').glob('*.parquet'))
        done=completed_codes(old_parts)
        parts.extend(reference(p) for p in old_parts)
        board_parts.extend(reference(p) for p in sorted((OLD/'board_parts').glob('*.parquet')))
        write('shared_market_and_leaders.parquet',(OLD/'shared_market_and_leaders.parquet').read_bytes())
        shared=pq.read_table(OUT/'shared_market_and_leaders.parquet',columns=['decision_ts_assumed_end']).to_pandas()
        days=calendar_days(shared)
        inventory=json.loads(INVENTORY.read_bytes());chosen={}
        for priority in ['old','car','ext','clean']:
            for item in inventory['files']:
                if item['root']==priority:chosen[item['code']]=priority
        if not done<=set(chosen):raise ValueError('UNKNOWN_COMPLETED_CODE')
        quarantine=defaultdict(set)
        with QUARANTINE.open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):quarantine[row['code']].add(int(row['day']))
        buffer=[]
        remaining=sorted(set(chosen)-done)
        for i,code in enumerate(remaining,1):
            label=chosen[code];body=(ROOTS[label]/f'{code}.parquet').read_bytes()
            table,_,quality=stock_features(pq.read_table(pa.BufferReader(body)).to_pandas(),code,days,quarantine.get(code,set()))
            source_files.append(dict(code=code,root=label,sha256=digest(body),bytes=len(body),**quality))
            if len(table):
                table['source_cache']=label;buffer.append(table)
            if len(buffer)>=50 or i==len(remaining):
                if buffer:
                    table=pd.concat(buffer,ignore_index=True);name=f'compact_parts/{len(parts):04d}.parquet'
                    write(name,parquet_bytes(table));parts.append(dict(path=name,rows=len(table),layout='COMPACT_SHARED_CONTEXT'))
                    buffer=[]
                print(json.dumps(dict(stage='compact_continuation',processed=i,total=len(remaining),bytes_written=used)),flush=True)
        summary=dict(joint_rows=sum(p['rows'] for p in parts),board_rows=sum(p['rows'] for p in board_parts),
                     preserved_wide_parts=len(old_parts),compact_parts=len(parts)-len(old_parts),
                     codes_preserved=len(done),codes_processed=len(remaining),concepts=len(catalog['boards']))
        write('manifest.json',json.dumps(dict(status='RECONSTRUCTED_JOINT_FEATURES_NOT_TRAINING_READY',
            inputs=refs,source_files=source_files,board_parts=board_parts,joint_parts=parts,summary=summary,
            source_priority=['clean','ext','car','old'],output_hashes=hashes.copy(),training_ready=False,
            old_source_lineage='FROZEN_DERIVED_SHARDS; OLD_RAW_HASH_LEDGER_NOT_RECOVERED',
            old_evidence_preserved=True,budget_reset=False,http_requests=0,fit_run=False),
            ensure_ascii=False,allow_nan=False,separators=(',',':')).encode())
    except Exception as exc:status,error='FAILED',f'{type(exc).__name__}:{exc}'
    write('stop_record.json',json.dumps(dict(status=status,error=error,parts=len(parts),bytes_before_stop=used),allow_nan=False).encode(),terminal=True)
    print(json.dumps(dict(status=status,error=error,parts=len(parts),bytes_written=used)))
    return status


if __name__=='__main__':
    raise SystemExit(0 if run()=='COMPLETE' else 2)
