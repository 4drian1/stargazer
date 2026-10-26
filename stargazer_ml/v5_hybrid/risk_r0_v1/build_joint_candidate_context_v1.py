"""Candidate-excluded context sidecars; global observations stay audit-only.

Canonical logical feature table = original joint part + same-key sidecar.
No new source reads, source replacement, HTTP, fit, or invented second leader.
"""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from build_joint_risk_features_v1 import HERE, digest, parquet_bytes

SOURCE = HERE/'joint_risk_features_20261017_v3'
OUT = HERE/'joint_risk_context_loo_20261017_v3'
CAP, RESERVE = 512*1024**2, 1024**2
AUDIT_ONLY = ['own_concepts_observed_mean_return_raw','own_concepts_observed_best_return_raw',
              'own_concepts_mean_coverage','leader_observed_stock_return_raw',
              'leader_observed_board_return_raw','captured_stock_mean_return_raw',
              'captured_stock_count_today','captured_stock_missing_today']


def excluded_means(total, count, own, is_member):
    present = is_member & np.isfinite(own)
    remaining = count-present.astype(int)
    if (remaining < 0).any():
        raise ValueError('NEGATIVE_REMAINING_MEMBER_COUNT')
    adjusted = total-np.where(present,own,0)
    return np.divide(adjusted,remaining,out=np.full_like(adjusted,np.nan,dtype=float),where=remaining>0),remaining


def known_leader(value):
    return isinstance(value,str) and bool(value)


def context_bytes(frame):
    table=pa.Table.from_pandas(frame,preserve_index=False)
    sink=pa.BufferOutputStream()
    pq.write_table(table,sink,compression='zstd',compression_level=9,use_dictionary=['code'],
                   use_byte_stream_split=[f.name for f in table.schema if pa.types.is_floating(f.type)])
    return sink.getvalue().to_pybytes()


def candidate_context(frame, totals, counts, sizes, affiliations, leaders, first_capture, extras=None):
    out=frame[['code','cutoff_index']].copy().reset_index(drop=True)
    if out.duplicated().any():
        raise ValueError('DUPLICATE_CANDIDATE_CUTOFF')
    names=['context_sector_mean_return_loo_raw','context_sector_best_return_loo_raw',
           'context_sector_mean_coverage_loo','context_leader_board_return_loo_raw',
           'context_current_leader_return_loo_raw','context_captured_leader_mean_return_loo_raw',
           'context_captured_valid_count_loo']
    for name in names: out[name]=np.nan
    out['context_current_leader_is_candidate']=False
    if extras is not None:
        for name in ['context_sector_breadth_loo','context_sector_common_change_loo_raw',
                     'context_current_leader_return_change_loo_raw',
                     'context_captured_leader_common_change_loo_raw',
                     'context_current_leader_rsi14_5m_raw','context_current_leader_rsi14_15m_raw',
                     'context_current_leader_return5m_raw','context_current_leader_return15m_raw']:
            out[name]=np.nan
    for code,indexes in out.groupby('code',sort=False).groups.items():
        t=frame.loc[indexes,'cutoff_index'].to_numpy(dtype=int)
        own=frame.loc[indexes,'stock_day_return_raw'].to_numpy(dtype=float)
        related=np.asarray(affiliations[code],dtype=int)
        mean,n=excluded_means(totals[np.ix_(related,t)],counts[np.ix_(related,t)],
                               own[None,:],np.ones((len(related),len(t)),dtype=bool))
        finite=np.isfinite(mean); present=finite.sum(axis=0)
        summary=np.divide(np.where(finite,mean,0).sum(axis=0),present,
                          out=np.full(len(t),np.nan),where=present>0)
        best=np.max(np.where(finite,mean,-np.inf),axis=0);best[~np.isfinite(best)]=np.nan
        coverage=np.divide(n,sizes[related,None]-1,out=np.full_like(mean,np.nan),where=sizes[related,None]>1)
        valid_cov=np.isfinite(coverage);cov_count=valid_cov.sum(axis=0)
        cov=np.divide(np.where(valid_cov,coverage,0).sum(axis=0),cov_count,
                      out=np.full(len(t),np.nan),where=cov_count>0)
        out.loc[indexes,names[0]]=summary;out.loc[indexes,names[1]]=best;out.loc[indexes,names[2]]=cov
        selected=leaders.iloc[t]
        current=selected['leader_observed_stock'].to_numpy(dtype=object)
        self_current=current==code
        ref=selected['leader_observed_stock_return_raw'].to_numpy(dtype=float).copy()
        ref[self_current]=np.nan
        out.loc[indexes,names[4]]=ref
        out.loc[indexes,'context_current_leader_is_candidate']=self_current
        first=np.array([first_capture.get((code,int(day)),np.iinfo(np.int64).max)
                        for day in selected['day']],dtype=np.int64)
        captured=first<=t
        number=(selected['captured_stock_count_today']-selected['captured_stock_missing_today']).to_numpy(dtype=int)
        global_mean=selected['captured_stock_mean_return_raw'].to_numpy(dtype=float)
        total=np.where(number>0,global_mean*number,0)
        ref,count=excluded_means(total,number,own,captured)
        out.loc[indexes,names[5]]=ref;out.loc[indexes,names[6]]=count
        board=selected['board_index'].to_numpy(dtype=int)
        valid_board=board>=0
        board_return=np.full(len(t),np.nan)
        if valid_board.any():
            b,positions=board[valid_board],t[valid_board]
            members=np.isin(b,related)
            values,_=excluded_means(totals[b,positions],counts[b,positions],own[valid_board],members)
            board_return[valid_board]=values
        out.loc[indexes,names[3]]=board_return
        if extras is not None:
            up=extras['up_totals'][np.ix_(related,t)]-np.where(np.isfinite(own)&(own>0),1,0)
            breadth=np.divide(up,n,out=np.full_like(mean,np.nan),where=n>0)
            valid_b=np.isfinite(breadth);bn=valid_b.sum(axis=0)
            out.loc[indexes,'context_sector_breadth_loo']=np.divide(np.where(valid_b,breadth,0).sum(axis=0),bn,
                out=np.full(len(t),np.nan),where=bn>0)
            own_prev=extras['returns'][extras['code_index'][code],np.maximum(t-1,0)]
            common_self=np.isfinite(own)&np.isfinite(own_prev)&(t%8!=0)
            change=own-own_prev
            change_mean,_=excluded_means(extras['common_totals'][np.ix_(related,t)],
                extras['common_counts'][np.ix_(related,t)],change[None,:],
                np.broadcast_to(common_self,(len(related),len(t))))
            valid_m=np.isfinite(change_mean);mn=valid_m.sum(axis=0)
            out.loc[indexes,'context_sector_common_change_loo_raw']=np.divide(np.where(valid_m,change_mean,0).sum(axis=0),mn,
                out=np.full(len(t),np.nan),where=mn>0)
            for j,name in enumerate(['context_current_leader_rsi14_5m_raw','context_current_leader_rsi14_15m_raw',
                                    'context_current_leader_return5m_raw','context_current_leader_return15m_raw']):
                values=extras['current_leader_metrics'][t,j].copy();values[self_current]=np.nan
                out.loc[indexes,name]=values
            current_change=extras['current_leader_change'][t].copy();current_change[self_current]=np.nan
            out.loc[indexes,'context_current_leader_return_change_loo_raw']=current_change
            captured_common=(first<t)&common_self
            movement,_=excluded_means(extras['captured_change_total'][t],extras['captured_change_count'][t],change,captured_common)
            out.loc[indexes,'context_captured_leader_common_change_loo_raw']=movement
    return out


def attach_shared_context(original, shared):
    if 'decision_ts_assumed_end' not in original:
        original=original.merge(shared,on='cutoff_index',how='left',validate='many_to_one')
        original['index_volume_feature']=np.nan
        original['joint_training_admitted']=False
    return original


def model_part(index):
    """Read the logical joint table; no fitting or automatic LR field selection."""
    spec=json.loads((OUT/'manifest.json').read_bytes())
    source_manifest=(SOURCE/'manifest.json').read_bytes()
    if digest(source_manifest)!=spec['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST_DRIFT')
    source_name=spec['parts'][index]['source']
    source_body=(SOURCE/source_name).read_bytes()
    if digest(source_body)!=json.loads(source_manifest)['output_hashes'][source_name]['sha256']:
        raise ValueError('JOINT_PART_INTEGRITY_FAIL')
    original=pq.read_table(pa.BufferReader(source_body)).to_pandas()
    if 'decision_ts_assumed_end' not in original:
        original=attach_shared_context(original,pq.read_table(SOURCE/'shared_market_and_leaders.parquet').to_pandas())
    context_body=(OUT/spec['parts'][index]['context']).read_bytes()
    if digest(context_body)!=spec['parts'][index]['sha256']:
        raise ValueError('CONTEXT_PART_INTEGRITY_FAIL')
    context=pq.read_table(pa.BufferReader(context_body)).to_pandas()
    if not original[['code','cutoff_index']].equals(context[['code','cutoff_index']]):
        raise ValueError('CONTEXT_ROW_IDENTITY_MISMATCH')
    result=original.drop(columns=AUDIT_ONLY,errors='ignore').merge(context,on=['code','cutoff_index'],validate='one_to_one')
    result['context_sector_minus_sse_return_raw']=result['context_sector_mean_return_loo_raw']-result['sse_15m_day_return_prev_close']
    result['context_sector_minus_sse_change_raw']=result['context_sector_common_change_loo_raw']-result['sse_15m_day_return_change_adjacent']
    result['context_stock_minus_sector_return_raw']=result['stock_day_return_raw']-result['context_sector_mean_return_loo_raw']
    return result


def run():
    if OUT.exists(): raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    original_manifest=SOURCE/'manifest.json';stop=SOURCE/'stop_record.json'
    metadata=json.loads(original_manifest.read_bytes())
    if json.loads(stop.read_bytes())['status']!='COMPLETE': raise ValueError('SOURCE_BUILD_NOT_COMPLETE')
    from research_budget_v2 import TOTAL_CAP,bytes_used
    if bytes_used()+RESERVE>TOTAL_CAP: raise ValueError('REGISTERED_BUDGET_STOP')
    OUT.mkdir(exist_ok=False);used=0;parts=[]
    def write(name,body,terminal=False):
        nonlocal used
        reserve=0 if terminal else RESERVE
        if used+len(body)+reserve>CAP or bytes_used()+len(body)+reserve>TOTAL_CAP:
            raise ValueError('CONTEXT_BUDGET_STOP')
        with (OUT/name).open('xb') as stream:stream.write(body)
        used+=len(body)
    status,error='COMPLETE',None
    try:
        def read_source(relative,columns=None):
            body=(SOURCE/relative).read_bytes()
            if digest(body)!=metadata['output_hashes'][relative]['sha256']:
                raise ValueError('JOINT_INPUT_PART_HASH_MISMATCH')
            return pq.read_table(pa.BufferReader(body),columns=columns).to_pandas()
        catalog=HERE/'ths_full_concept_scope_20261017_v1.json'
        boards=json.loads(catalog.read_bytes())['boards'];board_index={b['code']:i for i,b in enumerate(boards)}
        leaders=read_source('shared_market_and_leaders.parquet').sort_values('cutoff_index').reset_index(drop=True)
        leaders['day']=leaders['decision_ts_assumed_end']//1000000
        leaders['board_index']=leaders['leader_observed_board'].map(board_index).fillna(-1).astype(int)
        counts=np.zeros((len(boards),len(leaders)),dtype=int);totals=np.zeros_like(counts,dtype=float)
        up_totals=np.zeros_like(totals);common_mean=np.full_like(totals,np.nan)
        sizes=np.zeros(len(boards),dtype=int)
        for part in metadata['board_parts']:
            frame=read_source(part['path'],columns=['board_code','cutoff_index','observed_return_raw','observed_breadth',
                                                  'common_member_return_change','present','member_count'])
            for board,group in frame.groupby('board_code',sort=False):
                i=board_index[board];positions=group.cutoff_index.to_numpy(dtype=int)
                counts[i,positions]=group.present.to_numpy(dtype=int)
                totals[i,positions]=np.where(group.present>0,group.observed_return_raw*group.present,0)
                up_totals[i,positions]=np.where(group.present>0,np.rint(group.observed_breadth*group.present),0)
                common_mean[i,positions]=group.common_member_return_change
                sizes[i]=int(group.member_count.iloc[0])
        source=Path(json.loads(catalog.read_bytes())['source'])
        if digest(source.read_bytes())!=metadata['inputs'][str(source)]:
            raise ValueError('CATALOG_CHANGED_AFTER_JOINT_BUILD')
        membership=[b for b in pq.read_table(source).to_pylist() if b['source']=='ths' and b['type']=='concept']
        affiliations=defaultdict(list)
        for b in membership:
            for code in b['symbols']:affiliations[code].append(board_index[b['code']])
        first_capture={}
        for row in leaders.itertuples():
            if known_leader(row.leader_observed_stock):
                first_capture.setdefault((row.leader_observed_stock,int(row.day)),row.cutoff_index)
        codes=sorted(affiliations);code_index={c:i for i,c in enumerate(codes)}
        stock_returns=np.full((len(codes),len(leaders)),np.nan)
        leader_set=set(leaders.leader_observed_stock.dropna());leader_metrics={}
        metric_cols=['stock_rsi14_5m_raw','stock_rsi14_15m_raw','stock_return5m_raw','stock_return15m_raw']
        for part in metadata['joint_parts']:
            frame=read_source(part['path'],columns=['code','cutoff_index','stock_day_return_raw',*metric_cols])
            for code,g in frame.groupby('code',sort=False):
                positions=g.cutoff_index.to_numpy(dtype=int)
                stock_returns[code_index[code],positions]=g.stock_day_return_raw
                if code in leader_set:
                    values=np.full((len(leaders),4),np.nan);values[positions]=g[metric_cols].to_numpy(dtype=float)
                    leader_metrics[code]=values
        common_counts=np.zeros_like(counts)
        for board,i in board_index.items():
            related=[code_index[c] for c in codes if i in affiliations[c]]
            valid=np.isfinite(stock_returns[related])
            common_counts[i,1:]=(valid[:,1:]&valid[:,:-1]).sum(axis=0)
        common_counts[:,np.arange(len(leaders))%8==0]=0
        common_totals=np.where(common_counts>0,common_mean*common_counts,0)
        change_total=np.zeros(len(leaders));change_count=np.zeros(len(leaders),dtype=int);captured=set()
        for row in leaders.itertuples():
            t=row.cutoff_index
            if t%16==0:captured=set()
            if t%8 and captured:
                indexes=np.array([code_index[c] for c in captured],dtype=int)
                valid=np.isfinite(stock_returns[indexes,t])&np.isfinite(stock_returns[indexes,t-1])
                change_count[t]=valid.sum()
                change_total[t]=(stock_returns[indexes[valid],t]-stock_returns[indexes[valid],t-1]).sum()
            if known_leader(row.leader_observed_stock):captured.add(row.leader_observed_stock)
        current_metrics=np.full((len(leaders),4),np.nan);current_change=np.full(len(leaders),np.nan)
        for row in leaders.itertuples():
            t,code=row.cutoff_index,row.leader_observed_stock
            if not known_leader(code):continue
            if code in leader_metrics:current_metrics[t]=leader_metrics[code][t]
            if t%8:current_change[t]=stock_returns[code_index[code],t]-stock_returns[code_index[code],t-1]
        extras=dict(up_totals=up_totals,common_totals=common_totals,common_counts=common_counts,
                    returns=stock_returns,code_index=code_index,leader_metrics=leader_metrics,
                    current_leader_metrics=current_metrics,current_leader_change=current_change,
                    captured_change_total=change_total,captured_change_count=change_count)
        for i,part in enumerate(metadata['joint_parts']):
            frame=read_source(part['path'],columns=['code','cutoff_index','stock_day_return_raw'])
            context=candidate_context(frame,totals,counts,sizes,affiliations,leaders,first_capture,extras)
            for column in context.select_dtypes(include='float64'):
                context[column]=context[column].astype('float32')
            body=context_bytes(context);name=f'{i:04d}.parquet';write(name,body)
            parts.append(dict(source=part['path'],context=name,rows=len(context),sha256=digest(body),bytes=len(body)))
            print(json.dumps(dict(stage='candidate_exclusion',parts=len(parts),total=len(metadata['joint_parts']),rows=len(context),bytes_written=used)),flush=True)
        report=dict(status='CANDIDATE_EXCLUDED_JOINT_CONTEXT',source_manifest_sha256=digest(original_manifest.read_bytes()),
                    source_catalog_sha256=digest(source.read_bytes()),
                    code_sha256=digest(Path(__file__).read_bytes()),
                    test_sha256=digest((HERE/'test_build_joint_candidate_context_v1.py').read_bytes()),
                    parts=parts,audit_only_source_columns=AUDIT_ONLY,rows=sum(p['rows'] for p in parts),
                    context_storage='FLOAT32_FEATURES_ONLY_NOT_PRICES_OR_LABELS',
                    input_scope='SNAPSHOT_RECONSTRUCTION_PARTIAL_OBSERVATIONS',training_ready=False,http_requests=0,fit_run=False)
        write('manifest.json',json.dumps(report,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode())
    except Exception as exc:status,error='FAILED',f'{type(exc).__name__}:{exc}'
    write('stop_record.json',json.dumps(dict(status=status,error=error,parts=len(parts),bytes_before_stop=used),allow_nan=False).encode(),terminal=True)
    print(json.dumps(dict(status=status,error=error,rows=sum(p['rows'] for p in parts),bytes_written=used)))
    return status


if __name__=='__main__':
    raise SystemExit(0 if run()=='COMPLETE' else 2)
