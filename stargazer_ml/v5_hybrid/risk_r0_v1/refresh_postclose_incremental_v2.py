"""Corrected structured-query local refresh; v1 failed batch stays immutable."""
import gzip
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

import msgpack
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from postclose_research_scorer_v1 import HERE,ResearchScorer,encode,sha,strict_json
from postclose_local_source_bridge_v1 import load_local_features,latest_day_payload,clock_now
from build_decision_after_labels_v1 import admitted_evidence
from postclose_next_session_core_v1 import stock_days
from repair_postclose_daily_context_v1 import daily_returns,board_panel,unique_leaders,excluded_board
from research_budget_v2 import bytes_used,TOTAL_CAP

OUT=HERE/'postclose_incremental_refresh_20261017_v2'
TARGET_DAYS=(20261015,20261016)
CAP=16*1024**2
RESPONSE_CAP=32*1024**2


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise RuntimeError('REDIRECT_REFUSED')


def pairs(obj,table,code=None):
    if isinstance(obj,dict) and 'date' in obj:
        return [(f'{table}:{obj.get("code",code)}:{obj["date"]}',obj)]
    if isinstance(obj,list) and all(isinstance(v,(list,tuple)) and len(v)==2 for v in obj):
        return obj
    if isinstance(obj,dict):
        return list(obj.items())
    raise ValueError('UNKNOWN_RESPONSE_SHAPE')


def structured_pairs(obj,table,code):
    result=[]
    def walk(value,path):
        if isinstance(value,dict) and ('date' in value or 'mult' in value):
            actual_code=value.get('code',code if code!='*' else None)
            date=value.get('date')
            if date is None:
                date=next((int(p) for p in reversed(path) if str(p).isdigit() and len(str(p))==8),None)
            if actual_code is None or date is None:raise ValueError('MISSING_STRUCTURED_IDENTITY')
            if code!='*' and actual_code!=code:raise ValueError('CODE_MISMATCH')
            for p in path:
                if str(p).isdigit() and len(str(p))==6 and str(p)!=actual_code:raise ValueError('CODE_MISMATCH')
            result.append((f'{table}:{actual_code}:{date}',value))
        elif isinstance(value,dict):
            for key,item in value.items():walk(item,path+[str(key)])
        elif isinstance(value,list):
            for item in value:
                if isinstance(item,(list,tuple)) and len(item)==2 and isinstance(item[0],str):
                    walk(item[1],path+item[0].split(':'))
                else:walk(item,path)
        elif value is not None:raise ValueError('UNKNOWN_STRUCTURED_RESPONSE')
    walk(obj,[])
    return result


def valid_daily(records,day):
    result={}
    for key,row in records:
        if row is None: continue
        if not isinstance(row,dict) or row.get('date')!=day:
            raise ValueError('UNEXPECTED_DAILY_DATE')
        code=row.get('code')
        if key!=f'日k:{code}:{day}': raise ValueError('DAILY_KEY_IDENTITY')
        values=[row.get(k) for k in ['open','high','low','close']]
        good=all(type(v) in (int,float) and np.isfinite(v) and v>0 for v in values)
        if good:
            o,h,l,c=values;good=l<=min(o,c) and h>=max(o,c)
        if code in result and result[code]!=row: raise ValueError('DAILY_DUPLICATE_CONFLICT')
        result[code]=row if good else None
    return result


def contexts(daily,membership,candidates):
    codes=sorted({c for b in membership for c in b['symbols']});index={c:i for i,c in enumerate(codes)}
    closes=np.array([[daily[d].get(c,{}).get('close',np.nan) if daily[d].get(c) else np.nan
                      for d in [20261014,20261015,20261016]] for c in codes],dtype=float)
    returns=daily_returns(closes)
    members=[np.array([index[c] for c in b['symbols']],dtype=int) for b in membership]
    means,counts,ups=board_panel(returns,members)
    boards,leaders=unique_leaders(returns,means,members,codes)
    rows=[]
    for code in candidates:
        s=index[code];related=[b for b,record in enumerate(membership) if code in record['symbols']]
        for d,day in [(1,20261015),(2,20261016)]:
            row=dict(code=code,day=day,context_sector_mean_return_loo_raw=np.nan,
                     context_sector_breadth_loo=np.nan,context_sector_mean_coverage_loo=np.nan,
                     context_leader_board_return_loo_raw=np.nan,context_current_leader_return_loo_raw=np.nan,
                     context_current_leader_daily_rsi14_raw=np.nan,
                     observed_leader_board=None,observed_leader_stock=None)
            if np.isfinite(returns[s,d]) and related:
                reduced,breadth,n=excluded_board(means[related,d],counts[related,d],ups[related,d],np.full(len(related),returns[s,d]))
                sizes=np.array([len(members[b])-1 for b in related])
                coverage=np.divide(n,sizes,out=np.full(len(related),np.nan),where=sizes>0)
                for field,v in [('context_sector_mean_return_loo_raw',reduced),('context_sector_breadth_loo',breadth),
                                ('context_sector_mean_coverage_loo',coverage)]:
                    row[field]=float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
                b,leader=boards[d],leaders[d]
                if b>=0:
                    row['observed_leader_board']=membership[b]['code']
                    row['context_leader_board_return_loo_raw']=float(reduced[related.index(b)]) if b in related else float(means[b,d])
                if leader>=0:
                    row['observed_leader_stock']=codes[leader]
                    if leader!=s:row['context_current_leader_return_loo_raw']=float(returns[leader,d])
            rows.append(row)
    return pd.DataFrame(rows),dict(universe=len(codes),boards=len(membership),
        observed_return_codes={str(day):int(np.isfinite(returns[:,d]).sum()) for d,day in [(1,20261015),(2,20261016)]},
        full_universe_leader_certified=False,raw_partial_snapshot_context=True)


def run():
    if OUT.exists():raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if bytes_used()+CAP>TOTAL_CAP:raise ValueError('BUDGET_RESERVATION_FAILED')
    scorer=ResearchScorer();old,receipts=load_local_features(scorer)
    audit,_=admitted_evidence();codes=scorer.manifest['research_codes'];days=np.sort(old.day.unique())
    catalog_path=HERE/'ths_full_concept_scope_20261017_v1.json';catalog=strict_json(catalog_path.read_bytes())
    member_path=Path(catalog['source']);member_body=member_path.read_bytes()
    if sha(member_body)!=catalog['source_sha256']:raise ValueError('MEMBERSHIP_HASH_DRIFT')
    membership=sorted([r for r in pq.read_table(io.BytesIO(member_body)).to_pylist() if r['source']=='ths' and r['type']=='concept'],key=lambda r:r['code'])
    if [b['code'] for b in membership]!=[b['code'] for b in catalog['boards']]:raise ValueError('BOARD_SCOPE_DRIFT')
    sources=[Path(__file__),HERE/'postclose_next_session_core_v1.py',HERE/'repair_postclose_daily_context_v1.py',
             HERE/'test_refresh_postclose_incremental_v1.py',HERE/'test_refresh_postclose_incremental_v2.py',catalog_path,member_path]
    refs={str(p):sha(p.read_bytes()) for p in sources}
    OUT.mkdir(exist_ok=False);attempts=0;response_bytes=0;entries=[]
    def write(name,body,terminal=False):
        reserve=0 if terminal else 65536
        used=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        if used+len(body)+reserve>CAP or bytes_used()+len(body)+reserve>TOTAL_CAP:raise ValueError('WRITE_BUDGET_STOP')
        with (OUT/name).open('xb') as stream:stream.write(body)
    write('plan.json',encode(dict(inputs=refs,targets=TARGET_DAYS,max_requests=46,response_cap=RESPONSE_CAP,
        output_cap=CAP,retries=0,source='http://127.0.0.1:7899',fit_authorized=False)))
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    def fetch(table,code,date):
        nonlocal attempts,response_bytes
        if attempts>=46:raise ValueError('REQUEST_LIMIT')
        remaining=min(4*1024**2,RESPONSE_CAP-response_bytes)
        if remaining<=0:raise ValueError('RESPONSE_BUDGET_STOP')
        attempts+=1;query=f'{table}:{code}:{date}';started=clock_now()
        params={'cmd':'get','t':table,'k1':'all:' if code=='*' else 'key:'+code,'k2':'all:' if date=='*' else 'key:'+date}
        with opener.open('http://127.0.0.1:7899/?'+urllib.parse.urlencode(params),timeout=10) as stream:
            body=stream.read(remaining+1)
        response_bytes+=len(body)
        if len(body)>remaining:raise ValueError('RESPONSE_LIMIT')
        name=f'raw_{attempts:03d}.msgpack.gz';compressed=gzip.compress(body,mtime=0)
        write(name,compressed)
        entries.append(dict(query=query,file=name,bytes=len(body),sha256=sha(body),stored_sha256=sha(compressed),
                            observed_started_at=started,observed_finished_at=clock_now()))
        write(f'receipt_{attempts:03d}.json',encode(entries[-1]))
        return structured_pairs(msgpack.unpackb(body,raw=False,strict_map_key=False),table,code)
    try:
        daily={}
        for d in [20261014,20261015,20261016]:
            daily[d]=valid_daily(fetch('日k','*',str(d)),d)
            if not daily[d]:raise ValueError('EMPTY_DAILY_CROSS_SECTION')
        old_plan=strict_json((HERE/'postclose_joint_20261017_v1/manifest.json').read_bytes())
        daily_refs={Path(p).name[:6]:(Path(p),h) for p,h in old_plan['inputs'].items() if p.endswith('_daily.json.gz')}
        replacements=[];issues=[];parity=0
        stock_fields=[f for f in scorer.fields if f.startswith('stock_')]
        for code in codes:
            records=[]
            if code in daily_refs:
                path,expected_hash=daily_refs[code];body=path.read_bytes()
                if sha(body)!=expected_hash:raise ValueError('OLD_DAILY_HASH_DRIFT')
                refs[str(path)]=sha(body);records=strict_json(gzip.decompress(body))
            original=stock_days(code,records,days,audit[code])
            expected=old[old.code.eq(code)].set_index('day')[stock_fields]
            np.testing.assert_allclose(original.set_index('day')[stock_fields].loc[expected.index],expected,
                                       rtol=0,atol=1e-12,equal_nan=True)
            parity+=len(expected)
            item=json.loads(json.dumps(audit[code]))
            fresh=fetch('复权',code,'*')
            for key,event in fresh:
                event_day=int(str(key).rsplit(':',1)[1])
                if event_day in TARGET_DAYS:
                    item['issues'].append(dict(date=event_day,reason='NEW_ACTION_NOT_CORROBORATED'))
                    issues.append(dict(code=code,day=event_day,reason='NEW_ACTION_NOT_CORROBORATED'))
            existing={int(str(k).rsplit(':',1)[1]):r for k,r in records if r is not None}
            for d in TARGET_DAYS:
                if d in existing and existing[d]!=daily[d].get(code):raise ValueError('APPEND_EXISTING_DAY_CONFLICT')
            anchor=existing.get(20261014);fresh_anchor=daily[20261014].get(code)
            aligned=(anchor is not None and fresh_anchor is not None and abs(anchor['close']-fresh_anchor['close'])<=.01+1e-12)
            if not aligned:
                issues.append(dict(code=code,reason='PREVIOUS_CLOSE_NOT_ALIGNED'))
                item['issues'].extend(dict(date=d,reason='PREVIOUS_CLOSE_NOT_ALIGNED') for d in TARGET_DAYS)
            extended=[(k,r) for k,r in records if int(str(k).rsplit(':',1)[1])<TARGET_DAYS[0]]
            extended.extend((f'日k:{code}:{d}',daily[d][code]) for d in TARGET_DAYS if daily[d].get(code))
            frame=stock_days(code,extended,days,item)
            replacements.append(frame[frame.day.isin(TARGET_DAYS)][['code','day',*stock_fields]])
        updated=old[old.day.isin(TARGET_DAYS)].copy().set_index(['code','day'])
        stock=pd.concat(replacements).set_index(['code','day'])
        updated.loc[:,stock_fields]=stock.loc[updated.index,stock_fields]
        context,coverage=contexts(daily,membership,codes);context=context.set_index(['code','day'])
        for field in [c for c in scorer.fields if c.startswith('context_')]:
            updated[field]=context[field].reindex(updated.index) if field in context else np.nan
        updated['context_stock_minus_sector_daily_return']=updated.stock_return1d-updated.context_sector_mean_return_loo_raw
        updated['context_stock_minus_leader_daily_return']=updated.stock_return1d-updated.context_current_leader_return_loo_raw
        updated=updated.reset_index()
        results=[]
        for day in TARGET_DAYS:
            payload,_=latest_day_payload(updated[updated.day.eq(day)],scorer)
            scored=scorer.score(payload);scored['source_day']=day;scored['forward_validation_eligible']=False
            results.append(scored)
        sink=io.BytesIO();updated.to_parquet(sink,index=False,compression='zstd');write('feature_sidecar.parquet',sink.getvalue())
        write('scores.json',encode(results));write('source_ledger.json',encode(entries))
        for p,h in refs.items():
            if sha(Path(p).read_bytes())!=h:raise ValueError('INPUT_DRIFT_DURING_REFRESH')
        scorer.verify_artifacts()
        report=dict(status='COMPLETE_INCREMENTAL_RESEARCH_REFRESH_NOT_FORWARD_VALIDATION',requests=attempts,
                    response_bytes=response_bytes,prior_failed_requests=3,total_workflow_requests=attempts+3,old_stock_rows_replayed=parity,coverage=coverage,issues=issues,
                    days={str(r['source_day']):dict(rows=len(r['rows']),scored=sum(x['status']=='RESEARCH_SCORE' for x in r['rows'])) for r in results},
                    captured_leader_reconstructed=False,leader_rsi_newly_unverified=True,
                    original_model_unchanged=True,fit_run=False,labels_published=False,
                    independent_validation_passed=False,final_source_refs=refs)
        write('report.json',encode(report));write('stop_record.json',encode(dict(status='COMPLETE',attempts=attempts,response_bytes=response_bytes)),True)
        return report
    except Exception as exc:
        write('stop_record.json',encode(dict(status='FAILED',error=str(exc),attempts=attempts,response_bytes=response_bytes)),True)
        raise


if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False,allow_nan=False))
