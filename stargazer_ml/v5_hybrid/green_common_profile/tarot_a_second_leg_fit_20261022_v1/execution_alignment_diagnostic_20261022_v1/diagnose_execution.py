"""Versioned source grids, continuous-prefix events, no fit or HTTP."""
from collections import Counter
from datetime import datetime,timedelta
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
PRIOR=PARENT/'label_alignment_diagnostic_20261022_v1'
spec=importlib.util.spec_from_file_location('prior_diagnosis',PRIOR/'diagnose.py')
prior=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
fit=prior.fit
GREEN=PARENT.parent
sys.path[:0]=[str(GREEN),str(GREEN.parent/'step9_preflight')]
import dual_path_execution_v0_1_1 as execution

CONTRACT=GREEN/'contracts/tarot_execution_alignment_diagnostic_20261022.json'
GRID=GREEN/'gcp_round_b_small_sample_v1/minute_grid.json'
CFG=json.loads(GRID.read_bytes())


def save(name,raw):
    if sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())+len(raw)+65536>8*1024**2:
        raise ValueError('THIS_DIAGNOSIS_8MIB_CAP')
    if sum(p.stat().st_size for p in PARENT.rglob('*') if p.is_file())+len(raw)+65536>32*1024**2:
        raise ValueError('SHARED_FIRST_FIT_32MIB_CAP')
    if fit.usage()+len(raw)+65536>5*1024**3:
        raise ValueError('REGISTERED_5GIB_CAP')
    with (HERE/name).open('xb') as f:
        f.write(raw)


def dump(name,obj):
    save(name,json.dumps(prior.clean(obj),ensure_ascii=False,allow_nan=False,indent=2).encode())


def table(name,frame):
    b=io.BytesIO()
    frame.to_parquet(b,index=False,compression='zstd')
    save(name,b.getvalue())


def source_grid(day):
    return CFG['grid_241'] if day<CFG['grid_1300_from'] else CFG['grid']


def inspect_day(raw,code,day):
    if raw.empty:
        return {'source_status':'DAY_MISSING','source_complete':False},None
    raw=raw.sort_values('date')
    base=day*1000000
    if raw.date.duplicated().any():
        return {'source_status':'DUPLICATE_SOURCE_KEYS','source_complete':False},None
    if not raw.code.astype(str).str.zfill(6).eq(code).all():
        return {'source_status':'IDENTITY_MISMATCH','source_complete':False},None
    if any(not prior.valid_daily(r) for r in raw.to_dict('records')):
        return {'source_status':'OHLC_INVALID','source_complete':False},None
    present=set(int(t)-base for t in raw.date)
    expected=set(source_grid(day))
    missing=sorted(expected-present)
    extra=sorted(present-expected)
    bars=prior.kernel.to_bars(raw.to_dict('records'))
    five,segments,gaps=prior.kernel.aggregate_5min(bars,day)
    complete={b['t'] for b in five}
    absent=[t for t in prior.kernel.FIVE_STOPS if t not in complete]
    first_unknown=absent[0] if absent else None
    prefix=[b for b in five if first_unknown is None or b['t']<first_unknown]
    format_only=(len(missing)==0 and len(extra)==0 and day<CFG['grid_1300_from'] and absent==[130500])
    result={'source_status':'UNEXPECTED_SOURCE_KEYS' if extra else 'SOURCE_GAP' if missing else 'SOURCE_GRID_COMPLETE',
        'source_complete':not missing and not extra,'source_grid_version':len(expected),
        'source_missing':'|'.join(map(str,missing)), 'source_extra':'|'.join(map(str,extra)),
        'frozen_five_count':len(five),'frozen_first_uncomputable':first_unknown,
        'continuous_five_prefix':len(prefix),'format_only_unsupported_1305':format_only,
        'full_frozen_five_complete':len(five)==48 and not gaps}
    return result,prefix if not extra else None


def event_clock(day,complete_hhmmss):
    available=datetime.strptime(str(day)+f'{complete_hhmmss:06d}','%Y%m%d%H%M%S')
    last_index=available-timedelta(minutes=1)
    return int(available.strftime('%Y%m%d%H%M%S')),int(last_index.strftime('%Y%m%d%H%M%S'))


def price_events(rec,prefix):
    if not prefix:
        return [],{}
    signal=prior.confirmations(prefix,[0]*len(prefix),float(rec['node']),float(rec['atr']))
    events=[]
    for branch,field in [('RSI','rsi_index'),('RECLAIM','reclaim_index')]:
        idx=signal.get(field)
        if idx is None:
            continue
        available,last=event_clock(int(rec['E']),prefix[idx]['t'])
        events.append({k:rec[k] for k in ('code','P','E','stage','node','atr','split')}
            | {'branch':branch,'confirmation_available_ts':available,'last_source_bar_index_ts':last,
               'confirmation_close':prefix[idx]['close'], 'source_grid_version':rec['source_grid_version'],
               'prefix_five_at_confirmation':idx+1, 'risk_status':'UNBOUND',
               'speed_status':'NOT_APPLICABLE' if branch=='RSI' else 'MAX_RECLAIM_DURATION_UNDEFINED',
               'policy_status':'PRICE_EVENT_ONLY_NOT_BUY_ADVICE'})
    return events,signal


def select_events(events):
    if events.empty:
        return events
    return events.sort_values(['E','code','confirmation_available_ts']).groupby(
        ['stage','source_grid_version','branch'],sort=True).head(1).sort_values(
        ['E','code','confirmation_available_ts','branch']).reset_index(drop=True)


def main():
    if (HERE/'started.json').exists():
        raise ValueError('DIAGNOSIS_ALREADY_STARTED_NO_RESCAN')
    manifest_path=GREEN/'contracts/dual_path_execution_v0_1_1_manifest.json'
    manifest=json.loads(manifest_path.read_bytes())
    verified=[]
    for f in manifest['files']:
        actual=fit.sha(fit.PROJECT/f['path'])
        if actual!=f['sha256']:
            raise ValueError('OLD_EXECUTION_MANIFEST_CHANGED:'+f['role'])
        verified.append({'path':f['path'],'sha256':actual})
    dump('started.json',{'contract':fit.sha(CONTRACT),'script':fit.sha(__file__),'grid':fit.sha(GRID),
        'execution_engine':fit.sha(Path(execution.__file__)),'old_manifest_files':verified,
        'old_diagnosis_terminal':fit.sha(PRIOR/'FINAL_TERMINAL.json'),
        'model':fit.sha(prior.MODEL_ROOT/'model.pkl'),'fit_count':0})
    daily=pd.read_parquet(PRIOR/'daily_alignment.parquet')
    old=pd.read_parquet(PRIOR/'minute_alignment.parquet')
    expected_keys=set(zip(old.code,old.P))
    source_manifest=json.loads((PRIOR/'minute_source_manifest.json').read_bytes())
    hashes={Path(x['path']).name:x['sha256'] for x in source_manifest['entries']}
    all_rows=[]
    all_events=[]
    receipts=[]
    source_bytes=0
    for code,g in daily.groupby('code',sort=True):
        target=g[[key in expected_keys for key in zip(g.code,g.P)]]
        source={}
        path=prior.MINUTES/f'{code}.parquet'
        if not target.empty and path.exists():
            source_bytes+=3*path.stat().st_size
            if source_bytes>12*1024**3:
                raise ValueError('LOCAL_READ_RESERVATION_EXCEEDED')
            digest=fit.sha(path)
            if path.name not in hashes or hashes[path.name]!=digest:
                raise ValueError('PREVIOUSLY_BOUND_MINUTE_SOURCE_CHANGED')
            raw=pq.read_table(path,columns=['code','date','open','high','low','close'],
                filters=[('date','>=',int(target.P.min())*1000000),
                         ('date','<=',int(target.E.max())*1000000+150000)]).to_pandas()
            if fit.sha(path)!=digest:
                raise ValueError('SOURCE_CHANGED_DURING_READ')
            source={int(day):frame for day,frame in raw.groupby(raw.date//1000000,sort=False)}
            receipts.append({'path':str(path),'sha256':digest,'bytes':path.stat().st_size})
        for item in g.to_dict('records'):
            rec={k:item[k] for k in ('code','P','E','stage','node','atr','split','daily_path','close')}
            rec['event_state']='UNKNOWN'
            rec['price_event_count']=0
            rec['risk_status']='UNBOUND'
            if item['daily_path']=='NO_TOUCH_ABOVE_NODE':
                rec['event_state']='DAILY_NO_TOUCH'
            elif item['daily_path']=='GAP_BELOW_NO_RECLAIM':
                rec['event_state']='INITIAL_NODE_NOT_RECLAIMED_LOWER_NODES_UNTESTED'
            elif item['P']<20250101:
                rec['event_state']='BEFORE_BOUND_MINUTE_EPOCH'
            elif not path.exists():
                rec['source_status']='CACHE_FILE_MISSING'
            else:
                day=source.get(item['E'],pd.DataFrame())
                inspected,prefix=inspect_day(day,code,int(item['E']))
                rec.update(inspected)
                before=source.get(item['P'],pd.DataFrame())
                p_price=(float(before.sort_values('date').close.iloc[-1])
                         if not before.empty and int(before.date.max())>=item['P']*1000000+145900 else None)
                p_scale=(p_price is not None and not before.date.duplicated().any()
                         and before.code.astype(str).str.zfill(6).eq(code).all()
                         and prior.close_match(p_price,float(item['close'])))
                e_scale=(not day.empty and all(prior.close_match(a,b) for a,b in [
                    (float(day.sort_values('date').open.iloc[0]),item['E_open']),
                    (float(day.high.max()),item['E_high']),(float(day.low.min()),item['E_low']),
                    (float(day.sort_values('date').close.iloc[-1]),item['E_close'])]))
                rec.update(p_price_match=p_scale,e_ohlc_match=e_scale,
                           scale_check_scope='RETROSPECTIVE_E_CLOSE_QA_NOT_SIGNAL_FEATURE')
                if prefix is not None and p_scale and e_scale:
                    found,signal=price_events(rec,prefix)
                    all_events.extend(found)
                    rec.update(price_event_count=len(found),raw_rsi_status=signal.get('rsi_status'),
                        last_continuous_complete_ts=event_clock(item['E'],prefix[-1]['t'])[0] if prefix else None)
                    if found:
                        rec['event_state']='PRICE_EVENT_RISK_UNBOUND'
                    elif signal.get('rsi_status')=='LEGACY_SIGNAL_BEFORE_ACTUAL_TOUCH':
                        rec['event_state']='LEGACY_PRETOUCH_POLICY_MISMATCH'
                    elif rec.get('full_frozen_five_complete'):
                        rec['event_state']='NO_PRICE_CONFIRMATION_OBSERVED'
                    else:
                        rec['event_state']='UNKNOWN_AFTER_CONTINUOUS_PREFIX'
                elif not p_scale or not e_scale:
                    rec['event_state']='PRICE_BASIS_NOT_RECONCILED'
            all_rows.append(rec)
    ledger=pd.DataFrame(all_rows)
    events=pd.DataFrame(all_events)
    if len(ledger)!=6050 or ledger.duplicated(['code','P']).any():
        raise ValueError('FIXED_COHORT_NOT_PRESERVED')
    selected=select_events(events)
    table('event_ledger.parquet',ledger)
    table('price_events.parquet',events)
    table('limited_event_selection.parquet',selected)
    controls=[]
    for name,mask in [('NO_TOUCH',ledger.event_state.eq('DAILY_NO_TOUCH')&(ledger.P>=20250101)),
                      ('CACHE_MISSING',ledger.get('source_status',pd.Series(index=ledger.index,dtype=str)).eq('CACHE_FILE_MISSING')),
                      ('ACTUAL_SOURCE_GAP',ledger.get('source_status',pd.Series(index=ledger.index,dtype=str)).eq('SOURCE_GAP')),
                      ('PRETOUCH_MISMATCH',ledger.event_state.eq('LEGACY_PRETOUCH_POLICY_MISMATCH'))]:
        rows=ledger[mask].sort_values(['E','code']).head(1)
        controls.extend(rows.assign(control_role=name).to_dict('records'))
    dump('controls.json',controls)
    dump('source_manifest.json',{'minute_files':receipts,'logical_read_upper_bound':source_bytes,
        'no_http':True,'prior_daily_table':fit.sha(PRIOR/'daily_alignment.parquet'),
        'unit_provenance_pending_for_execution':True})
    targets=ledger[[key in expected_keys for key in zip(ledger.code,ledger.P)]]
    summary={'fixed_cohort':len(ledger),'old_minute_targets':len(targets),
        'states':ledger.event_state.value_counts().to_dict(),
        'source_grid_status':targets.get('source_status',pd.Series(dtype=str)).value_counts().to_dict(),
        'format_only_1305_unsupported':int(targets.get('format_only_unsupported_1305',pd.Series(dtype=bool)).eq(True).sum()),
        'price_events':len(events),'price_event_unique_stock_days':int(events[['code','P']].drop_duplicates().shape[0]),
        'events_by_stage_grid_branch':events.groupby(['stage','source_grid_version','branch']).size().rename('n').reset_index().to_dict('records'),
        'selected_ordinary':len(selected),'controls':len(controls),'fit_count':0,
        'peak_working_set_bytes':fit.peak_memory()}
    dump('source_and_event_summary.json',summary)
    print(json.dumps(prior.clean(summary),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
