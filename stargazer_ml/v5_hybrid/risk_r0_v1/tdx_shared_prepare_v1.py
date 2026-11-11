"""Native TDX common inputs: short member paths, selective full history.

Produces the existing verified prepared format; source hashes explicitly bind
the slices actually read, not a falsely claimed hash of an unread whole file.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd

import tdx_postclose_runtime_v1 as rt
import tdx_native_kernel_v1 as native
from tdx_native_reader_v1 import parse_lc5,validate_day,read_lc5_file,_valid_calendar_day
from intraday_native_context_v1 import NativeIntradayCache


def slice_days(item,code,P,previous):
    path=Path(item['path']);before=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(item['size'],item['mtime_ns']):
        raise ValueError('SOURCE_VERSION_CHANGED:'+code)
    # Scan binary dates once, WITHOUT constructing years of Python row dicts.
    # Select every record in both sessions, so duplicates/unsorted records
    # anywhere in the file cannot be hidden by a last-96-record shortcut.
    raw=path.read_bytes()
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('SOURCE_VERSION_CHANGED:'+code)
    if len(raw)%32:raise ValueError('BINARY_RECORD_SIZE')
    data=np.frombuffer(raw,dtype=np.dtype([('packed','<u2'),('minute','<u2'),('rest','V28')]))
    packed=data['packed'].astype(np.int64)
    dates=(packed//2048+2004)*10000+(packed%2048//100)*100+packed%2048%100
    if any(not _valid_calendar_day(int(d)) for d in np.unique(dates)) or (data['minute']>=1440).any():
        raise ValueError('BINARY_DATE_OR_CLOCK')
    selected=data[np.isin(dates,[previous,P])].tobytes()
    rows=parse_lc5(selected)
    grouped={d:[r for r in rows if r['day']==d] for d in (previous,P)}
    if any(not validate_day(grouped[d],d)['eligible'] for d in grouped):
        raise ValueError('TDX_TWO_SESSION_PREFIX_INCOMPLETE:'+code)
    receipt=dict(reference=str(path)+'#'+hashlib.sha256(raw).hexdigest(),
        path=str(path),prefix_sha256=hashlib.sha256(selected).hexdigest(),
        sha256=hashlib.sha256(raw).hexdigest(),hash_scope='WHOLE_FILE_READ',clock='END_5M',frequency_minutes=5,
        price_basis='RAW',source_size=before.st_size,source_mtime_ns=before.st_mtime_ns,
        availability_basis='LOCAL_READER_OBSERVED_BYTES_NOT_UPSTREAM_FIRST_ARRIVAL')
    return grouped,receipt,selected


def prepare(folder,spec,progress):
    """No stockdb, network, window shortening, synthetic bars or model fitting."""
    start=time.perf_counter();P,E=spec['P'],spec['E'];cutoff=spec['cutoff']
    observed=native.stamp();indices={};receipts={};sources={};refs={};slices={};failures=dict(spec['missing'])
    read_bytes=[0]
    for code in ('000001.SH','399006.SZ'):
        item=spec['files'][code];info=read_lc5_file(item['path'])
        read_bytes[0]+=info['bytes']
        if (info['bytes'],info['mtime_ns'])!=(item['size'],item['mtime_ns']):raise ValueError('INDEX_SOURCE_CHANGED')
        rows=[r for r in info['rows'] if r['ts']<=cutoff];good=native.good_days(rows)
        if P not in good:raise ValueError('INDEX_P_DAY_INCOMPLETE:'+code)
        indices[code]=(rows,good)
        receipts[code]=dict(reference=item['path']+'#'+info['sha256'],clock='END_5M',
            price_basis='RAW',sha256=info['sha256'],available_at=observed)
    first=max(min(good) for _,good in indices.values());days=[d for d in spec['legal'] if first<=d<=P]
    if len(days)<2:raise ValueError('INDEX_HISTORY_TOO_SHORT')
    for code,(rows,good) in indices.items():
        if not set(days)<=good:raise ValueError('INDEX_SHARED_CALENDAR_INCOMPLETE:'+code)
        pd.DataFrame(rows).rename(columns={'ts':'date','raw_volume':'volume'}).to_parquet(folder/(code+'.parquet'),index=False)
    # Keep only lightweight references/receipts; don't retain all raw histories.
    for i,(code,item) in enumerate(spec['files'].items()):
        if code in indices:continue
        try:
            grouped,proof,selected=slice_days(item,code,P,days[-2])
            read_bytes[0]+=item['size']
            refs[code]=dict(day=days[-2],complete=True,close=grouped[days[-2]][-1]['close'])
            sources[code]=dict(proof,available_at=observed);slices[code]=selected
        except ValueError as error:
            if 'SOURCE_VERSION_CHANGED' in str(error):raise
            failures[code]=str(error)
        if i%200==0:
            native.memory_guard();progress(stage='TDX_MEMBER_SESSION_PATHS',done=i,total=len(spec['files']))
    if not set(spec['candidates'])<=set(sources):raise ValueError('REQUIRED_CANDIDATE_MINUTES_MISSING')
    def current(code):
        rows=parse_lc5(slices[code]);proof=sources[code]
        return [dict(code=code,date=r['ts'],open=r['open'],high=r['high'],low=r['low'],close=r['close'],
            volume=r['raw_volume'],amount=r['amount'],source_frequency_minutes=5,source_clock='END_5M')
            for r in rows if r['day']==P],proof
    def history(code):
        item=spec['files'][code];info=read_lc5_file(item['path'])
        read_bytes[0]+=info['bytes']
        if (info['bytes'],info['mtime_ns'])!=(item['size'],item['mtime_ns']):raise ValueError('SOURCE_VERSION_CHANGED:'+code)
        return info['rows'],dict(reference=item['path']+'#'+info['sha256'],clock='END_5M',price_basis='RAW')
    cache=NativeIntradayCache(days,cutoff,spec['boards'],spec['candidates'],refs,current,history)
    tails=[];codes=sorted(cache.receipts)
    progress(stage='TDX_CANDIDATE_LEADER_FEATURES',done=0,total=len(cache.detail_codes))
    for code in codes:
        frame,_=cache.stock(code,days,cutoff);tails.append(frame)
    progress(stage='TDX_CANDIDATE_LEADER_FEATURES',done=cache.audit['detail_computations'],total=len(cache.detail_codes))
    for code,item in spec['files'].items():
        st=Path(item['path']).stat()
        if (st.st_size,st.st_mtime_ns)!=(item['size'],item['mtime_ns']):raise ValueError('SOURCE_VERSION_CHANGED_POST:'+code)
        if code in sources and native.sha(item['path'])!=sources[code]['sha256']:
            raise ValueError('SOURCE_VERSION_CHANGED_POST:'+code)
        if code in sources:read_bytes[0]+=item['size']
    rt.check_budget(folder.parents[2],16*1024**2)
    pd.concat(tails,ignore_index=True).to_parquet(folder/'stocks.parquet',index=False)
    np.savez_compressed(folder/'returns.npz',codes=np.array(codes),
        ret5=np.array([cache.returns[5][cache.ci[c]] for c in codes]),
        ret15=np.array([cache.returns[15][cache.ci[c]] for c in codes]))
    receipts.update(sources)
    rt.write(folder,'indicators.json',{})  # Monitoring has its own prepared indicator lane.
    rt.write(folder,'source_manifest.json',dict(contract=rt.CONTRACT,entries=sources,failures=failures,
        bound_stock_count=len(sources),declared_stock_count=len(cache.codes),comparison_board_count=len(spec['boards']),
        attempted_stock_count=len(spec['bound']),source_proof_contract='NATIVE_END5M_TWO_SESSION_SLICE_V1'))
    rt.write(folder,'snapshot.json',dict(cutoff=cutoff,days=days,boards=spec['boards'],receipts=receipts,
        available_at=observed,observed_at=observed,membership_known_at=observed,
        membership_scope='POSTCLOSE_CAPTURE_OBSERVED_LATER' if observed//1000000!=P else 'FORWARD_SNAPSHOT'))
    rt.write(folder,'stock_complete.json',dict(peak_memory=native.peak_memory(),
        strategy='MEMBER_TWO_SESSION_PATHS_PLUS_SELECTIVE_FULL_HISTORY',cache_audit=cache.audit,
        disk_read_bytes_successful_files=read_bytes[0],
        read_bytes_basis='包含二次SHA复核及选择性完整历史读取；失败文件读数未计入',
        elapsed_s=round(time.perf_counter()-start,3)))
    from postclose_shared_source_v1 import implementation_key
    rt.write(folder,'shared_source_impl.json',dict(sha256=implementation_key()))


if __name__=='__main__':
    import sys
    folder=Path(sys.argv[1]).resolve()
    if folder.parents[1].name!='tdx_prepared':raise ValueError('TDX_PREPARE_TARGET')
    def progress(**values):
        print(json.dumps(values),flush=True)
        # This is the REAL runtime root (key -> day -> tdx_prepared -> root),
        # not tdx_prepared/state.json. Only update the active same-day job.
        state_path=folder.parents[2]/'state.json'
        if state_path.is_file():
            state=rt.read_json(state_path)
            if state.get('status')=='RUNNING' and int(state.get('data_date',0))==spec['P']:
                rt.atomic_json(state_path,dict(state,**values,updated_at=native.stamp()))
    spec=rt.read_json(folder/'request.json')
    prepare(folder,spec,progress)
