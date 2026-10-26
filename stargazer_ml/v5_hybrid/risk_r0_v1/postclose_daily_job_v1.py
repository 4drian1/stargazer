"""Manual daily research refresh; cached dates are reused, no scheduler/deployment."""
import argparse
import gzip
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

import msgpack
import numpy as np
import pandas as pd

from postclose_research_scorer_v1 import HERE,ResearchScorer,strict_json,encode,sha
from postclose_local_source_bridge_v1 import load_local_features,latest_day_payload,clock_now
from read_postclose_incremental_v2 import load_refreshed,ROOT as CACHE
from refresh_postclose_incremental_v2 import NoRedirect,structured_pairs,valid_daily
from repair_postclose_daily_context_v1 import daily_returns,board_panel,unique_leaders,excluded_board
from postclose_next_session_core_v1 import stock_days
from build_decision_after_labels_v1 import admitted_evidence
from research_budget_v2 import TOTAL_CAP,bytes_used

JOBS=HERE/'postclose_daily_jobs_v1'
CAP=16*1024**2


def background_fields(fields):
    return [f for f in fields if f.startswith(('sse_','cyb_','golden_'))]


def calendar_days(values):
    from datetime import datetime
    if not isinstance(values,list) or not values or any(type(d) is not int for d in values):
        raise ValueError('INVALID_CALENDAR')
    if values!=sorted(set(values)):raise ValueError('CALENDAR_NOT_SORTED_UNIQUE')
    for d in values:
        if len(str(d))!=8:raise ValueError('INVALID_CALENDAR_DAY')
        datetime.strptime(str(d),'%Y%m%d')
    return values


def read_input(path,kind):
    body=Path(path).read_bytes()
    if len(body)>4*1024**2:raise ValueError('INPUT_SIZE_LIMIT')
    data=strict_json(body)
    if data.get('kind')!=kind or not data.get('source_refs'):
        raise ValueError('UNBOUND_INPUT_SOURCE:'+kind)
    refs={str(Path(path)):sha(body)}
    for item in data['source_refs']:
        source=Path(item['path']);raw=source.read_bytes()
        if sha(raw)!=item['sha256']:raise ValueError('INPUT_SOURCE_HASH_DRIFT')
        refs[str(source)]=item['sha256']
    return data,refs


def prior_daily_archives():
    sources=[]
    for item in strict_json((CACHE/'source_ledger.json').read_bytes()):
        if item['query'].startswith('日k:'):
            sources.append((int(item['query'].rsplit(':',1)[1]),CACHE,item))
    if JOBS.exists():
        for root in sorted(JOBS.iterdir()):
            if not root.is_dir():continue
            stop=root/'stop_record.json'
            if not stop.exists():continue
            if strict_json(stop.read_bytes()).get('status')!='COMPLETE_RESEARCH_ONLY':continue
            for item in strict_json((root/'source_ledger.json').read_bytes()):
                if item['table']=='日k':sources.append((int(item['date']),root,item))
    return sources


def plan(day,calendar=None,background=None):
    scorer=ResearchScorer();cached,_=load_refreshed()
    if day in cached.day.unique():
        return dict(status='CACHED_READY_NO_FETCH',day=day,ready=True,requests=0),scorer,None
    blockers=[];bound={};refs={}
    if calendar is None:blockers.append('CALENDAR_INPUT_REQUIRED')
    else:
        data,r=read_input(calendar,'TRADING_CALENDAR');refs.update(r)
        bound['calendar']=calendar_days(data['days'])
        if day not in bound['calendar']:blockers.append('DAY_NOT_IN_BOUND_CALENDAR')
    if background is None:blockers.append('SAME_DAY_MARKET_GOLDEN_INPUT_REQUIRED')
    else:
        data,r=read_input(background,'POSTCLOSE_MARKET_GOLDEN');refs.update(r)
        if data.get('day')!=day:raise ValueError('BACKGROUND_DAY_MISMATCH_NO_FALLBACK')
        if set(data.get('features',{}))!=set(background_fields(scorer.fields)):
            raise ValueError('BACKGROUND_FIELD_SCHEMA')
        for v in data['features'].values():
            if v is not None and (type(v) not in (int,float) or not np.isfinite(v)):
                raise ValueError('INVALID_BACKGROUND_VALUE')
        if any(data['features'][f] is None for f in ['sse_daily_rsi14','cyb_daily_rsi14','golden_position']):
            blockers.append('REQUIRED_BACKGROUND_MISSING')
        bound['background']=data['features']
    if day<=20261016:blockers.append('OLD_DAY_OUTSIDE_CACHED_REFRESH_SCOPE')
    if (JOBS/str(day)).exists():blockers.append('DAY_JOB_EXISTS_NO_RESTART')
    if bytes_used()+CAP>TOTAL_CAP:blockers.append('OUTPUT_BUDGET_RESERVATION_FAILED')
    if not blockers:
        previous=[d for d in bound['calendar'] if d<day]
        if not previous:raise ValueError('PREVIOUS_TRADING_DAY_REQUIRED')
        available={d for d,_,_ in prior_daily_archives()}
        missing=[d for d in bound['calendar'] if 20261016<d<=day and d not in available]
        if len(missing)>5:blockers.append('CATCHUP_EXCEEDS_FIVE_SESSION_BOUND')
        bound.update(previous_day=previous[-1],missing_days=missing,refs=refs)
    return dict(status='READY_RESEARCH_REFRESH' if not blockers else 'BLOCKED_NO_FETCH',day=day,
                ready=not blockers,blockers=blockers,max_requests=43+len(bound.get('missing_days',[])),
                schedule_installed=False,independent_validation_passed=False),scorer,bound


def context_day(previous,current,membership,candidates,day):
    codes=sorted({c for b in membership for c in b['symbols']});index={c:i for i,c in enumerate(codes)}
    closes=np.array([[row.get(c,{}).get('close',np.nan) if row.get(c) else np.nan
                      for row in [previous,current]] for c in codes],dtype=float)
    returns=daily_returns(closes)
    members=[np.array([index[c] for c in b['symbols']],dtype=int) for b in membership]
    means,counts,ups=board_panel(returns,members);boards,leaders=unique_leaders(returns,means,members,codes)
    result=[]
    for code in candidates:
        s=index[code];related=[b for b,record in enumerate(membership) if code in record['symbols']]
        row=dict(code=code,day=day,context_sector_mean_return_loo_raw=np.nan,
                 context_sector_breadth_loo=np.nan,context_sector_mean_coverage_loo=np.nan,
                 context_leader_board_return_loo_raw=np.nan,context_current_leader_return_loo_raw=np.nan,
                 context_current_leader_daily_rsi14_raw=np.nan)
        if related and np.isfinite(returns[s,1]):
            reduced,breadth,n=excluded_board(means[related,1],counts[related,1],ups[related,1],np.full(len(related),returns[s,1]))
            sizes=np.array([len(members[b])-1 for b in related])
            coverage=np.divide(n,sizes,out=np.full(len(related),np.nan),where=sizes>0)
            for field,values in [('context_sector_mean_return_loo_raw',reduced),('context_sector_breadth_loo',breadth),('context_sector_mean_coverage_loo',coverage)]:
                row[field]=float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
            b,leader=boards[1],leaders[1]
            if b>=0:row['context_leader_board_return_loo_raw']=float(reduced[related.index(b)]) if b in related else float(means[b,1])
            if leader>=0 and leader!=s:row['context_current_leader_return_loo_raw']=float(returns[leader,1])
        result.append(row)
    return pd.DataFrame(result)


def run(day,calendar=None,background=None):
    report,scorer,bound=plan(day,calendar,background)
    if not report['ready']:return report
    if report['status']=='CACHED_READY_NO_FETCH':
        frame,_=load_refreshed();payload,_=latest_day_payload(frame[frame.day.eq(day)],scorer)
        result=scorer.score(payload)
        result.update(day=day,requests=0,cache_reused=True,forward_validation_eligible=False)
        return result
    old,receipts=load_local_features(scorer);seed_days=sorted(old.day.unique())
    if [d for d in bound['calendar'] if seed_days[0]<=d<=seed_days[-1]]!=seed_days:
        raise ValueError('CALENDAR_OVERLAP_DRIFT')
    audit,_=admitted_evidence();catalog=strict_json((HERE/'ths_full_concept_scope_20261017_v1.json').read_bytes())
    member_body=Path(catalog['source']).read_bytes()
    if sha(member_body)!=catalog['source_sha256']:raise ValueError('MEMBERSHIP_DRIFT')
    import pyarrow.parquet as pq
    membership=sorted([r for r in pq.read_table(io.BytesIO(member_body)).to_pylist() if r['source']=='ths' and r['type']=='concept'],key=lambda r:r['code'])
    if [b['code'] for b in membership]!=[b['code'] for b in catalog['boards']]:raise ValueError('BOARD_SCOPE_DRIFT')
    raw_daily={}
    for d,archive,entry in prior_daily_archives():
        if d>day:continue
        compressed=(archive/entry['file']).read_bytes();raw=gzip.decompress(compressed)
        if sha(raw)!=entry['sha256'] or sha(compressed)!=entry['stored_sha256']:raise ValueError('REUSED_ARCHIVE_DRIFT')
        parsed=valid_daily(structured_pairs(msgpack.unpackb(raw,raw=False,strict_map_key=False),'日k','*'),d)
        if d in raw_daily and raw_daily[d]!=parsed:raise ValueError('REUSED_SOURCE_CONFLICT')
        raw_daily[d]=parsed
    if bound['previous_day'] not in raw_daily and bound['previous_day'] not in bound['missing_days']:
        raise ValueError('PREVIOUS_DAY_SOURCE_UNBOUND')
    root=JOBS/str(day);root.mkdir(parents=True,exist_ok=False)
    attempts=0;response_bytes=0;ledger=[]
    def write(name,value,terminal=False,raw=False):
        body=value if raw else encode(value);reserve=0 if terminal else 65536
        used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        if used+len(body)+reserve>CAP or bytes_used()+len(body)+reserve>TOTAL_CAP:raise ValueError('DAILY_OUTPUT_BUDGET_STOP')
        with (root/name).open('xb') as stream:stream.write(body)
    refs=dict(bound['refs']);refs[str(Path(__file__))]=sha(Path(__file__).read_bytes())
    write('plan.json',dict(**report,inputs=refs,response_cap=32*1024**2,output_cap=CAP,retries=0))
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    def fetch(table,code,date):
        nonlocal attempts,response_bytes
        if attempts>=report['max_requests']:raise ValueError('DAILY_REQUEST_LIMIT')
        remaining=min(4*1024**2,32*1024**2-response_bytes)
        if remaining<=0:raise ValueError('DAILY_RESPONSE_BUDGET_STOP')
        attempts+=1;started=clock_now()
        query=dict(cmd='get',t=table,k1='all:' if code=='*' else 'key:'+code,k2='all:' if date=='*' else 'key:'+str(date))
        with opener.open('http://127.0.0.1:7899/?'+urllib.parse.urlencode(query),timeout=10) as stream:body=stream.read(remaining+1)
        response_bytes+=len(body)
        if len(body)>remaining:raise ValueError('DAILY_RESPONSE_LIMIT')
        name=f'raw_{attempts:03d}.gz';compressed=gzip.compress(body,mtime=0);write(name,compressed,raw=True)
        entry=dict(file=name,table=table,code=code,date=date,sha256=sha(body),stored_sha256=sha(compressed),bytes=len(body),observed_started_at=started,observed_finished_at=clock_now())
        write(f'receipt_{attempts:03d}.json',entry);ledger.append(entry)
        return structured_pairs(msgpack.unpackb(body,raw=False,strict_map_key=False),table,code)
    try:
        for d in bound['missing_days']:
            raw_daily[d]=valid_daily(fetch('日k','*',d),d)
            if not raw_daily[d]:raise ValueError('EMPTY_DAILY_CROSS_SECTION')
        original=strict_json((HERE/'postclose_joint_20261017_v1/manifest.json').read_bytes())
        seeds={Path(p).name[:6]:(Path(p),h) for p,h in original['inputs'].items() if p.endswith('_daily.json.gz')}
        grid=np.array([d for d in bound['calendar'] if seed_days[0]<=d<=day],dtype=np.int64)
        stock_fields=[f for f in scorer.fields if f.startswith('stock_')];rows=[];issues=[]
        for code in scorer.manifest['research_codes']:
            records=[]
            if code in seeds:
                p,h=seeds[code];body=p.read_bytes()
                if sha(body)!=h:raise ValueError('DAILY_SEED_DRIFT')
                refs[str(p)]=h;records=strict_json(gzip.decompress(body))
            item=json.loads(json.dumps(audit[code]));events=fetch('复权',code,'*')
            for key,event in events:
                d=int(key.rsplit(':',1)[1])
                if 20261014<d<=day:
                    item['issues'].append(dict(date=d,reason='UNCORROBORATED_INCREMENTAL_ACTION'))
                    issues.append(dict(code=code,day=d))
            seed={int(k.rsplit(':',1)[1]):v for k,v in records if v is not None}
            anchor=seed.get(20261014);observed=raw_daily[20261014].get(code)
            if anchor is None or observed is None or abs(anchor['close']-observed['close'])>.01+1e-12:
                item['issues'].extend(dict(date=d,reason='SEED_ANCHOR_UNBOUND') for d in grid if d>20261014)
            extended=[(k,v) for k,v in records if int(k.rsplit(':',1)[1])<=20261014]
            for d in grid[grid>20261014]:
                if d not in raw_daily:raise ValueError('INCREMENTAL_DAY_SOURCE_GAP')
                if raw_daily[d].get(code):extended.append((f'日k:{code}:{d}',raw_daily[d][code]))
            frame=stock_days(code,extended,grid,item);s=frame[frame.day.eq(day)].iloc[0]
            row=dict(code=code,day=day,**{f:np.nan for f in scorer.fields});row.update(bound['background'])
            row.update({f:s[f] for f in stock_fields});rows.append(row)
        data=pd.DataFrame(rows).set_index(['code','day'])
        ctx=context_day(raw_daily[bound['previous_day']],raw_daily[day],membership,scorer.manifest['research_codes'],day).set_index(['code','day'])
        for f in ctx:data[f]=ctx[f]
        data['context_stock_minus_sector_daily_return']=data.stock_return1d-data.context_sector_mean_return_loo_raw
        data['context_stock_minus_leader_daily_return']=data.stock_return1d-data.context_current_leader_return_loo_raw
        data=data.reset_index();payload,_=latest_day_payload(data,scorer)
        result=scorer.score(payload);result.update(day=day,requests=attempts,forward_validation_eligible=False)
        write('input.json',payload);write('scores.json',result);write('source_ledger.json',ledger)
        for p,h in refs.items():
            if sha(Path(p).read_bytes())!=h:raise ValueError('DAILY_INPUT_CHANGED')
        scorer.verify_artifacts()
        for record in ledger:
            compressed=(root/record['file']).read_bytes()
            if sha(compressed)!=record['stored_sha256'] or sha(gzip.decompress(compressed))!=record['sha256']:
                raise ValueError('DAILY_ARCHIVE_READBACK_FAILED')
        write('stop_record.json',dict(status='COMPLETE_RESEARCH_ONLY',requests=attempts,response_bytes=response_bytes,action_issues=issues,independent_validation_passed=False),True)
        return result
    except Exception as exc:
        write('stop_record.json',dict(status='FAILED',error=str(exc),requests=attempts,response_bytes=response_bytes),True)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--day',type=int,required=True);parser.add_argument('--calendar',type=Path)
    parser.add_argument('--background',type=Path);parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    try:
        result=run(args.day,args.calendar,args.background) if args.run else plan(args.day,args.calendar,args.background)[0]
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
        return 2 if result.get('ready') is False else 0
    except (ValueError,OSError,KeyError) as exc:
        print(json.dumps(dict(status='REJECTED',error=str(exc)),ensure_ascii=False));return 2


if __name__=='__main__':raise SystemExit(main())
