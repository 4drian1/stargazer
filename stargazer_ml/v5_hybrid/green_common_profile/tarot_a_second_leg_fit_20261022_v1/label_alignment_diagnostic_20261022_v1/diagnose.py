"""Frozen-cohort label alignment; existing local files only, zero fit/HTTP."""
from collections import Counter
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
spec = importlib.util.spec_from_file_location('frozen_first_fit', PARENT/'run.py')
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)
ROOT = fit.PROJECT
MODEL_ROOT = PARENT/'source_quarantine_corrected'
CONTRACT = PARENT.parent/'contracts/tarot_label_alignment_diagnostic_20261022.json'
MINUTES = ROOT/'stargazer_ml/pullback_train/minute_cache_clean'
TRIGGER = PARENT.parent/'gcp_round_b_small_sample_v1/pentacles_golden_v1/trigger_replay_v3.py'
kspec = importlib.util.spec_from_file_location('frozen_kernel', TRIGGER)
kernel = importlib.util.module_from_spec(kspec)
kspec.loader.exec_module(kernel)
READ_CAP = 12*1024**3


def clean(value):
    if isinstance(value, dict):
        return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(name, raw):
    if sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())+len(raw)+65536>8*1024**2:
        raise ValueError('DIAGNOSTIC_8MIB_CAP')
    if sum(p.stat().st_size for p in PARENT.rglob('*') if p.is_file())+len(raw)+65536>32*1024**2:
        raise ValueError('SHARED_FIRST_ROUND_32MIB_CAP')
    if fit.usage()+len(raw)+65536>5*1024**3:
        raise ValueError('REGISTERED_5GIB_CAP')
    with (HERE/name).open('xb') as f:
        f.write(raw)


def dump(name, value):
    save(name,json.dumps(clean(value),ensure_ascii=False,allow_nan=False,indent=2).encode())


def table(name, frame):
    out=io.BytesIO()
    frame.to_parquet(out,index=False,compression='zstd')
    save(name,out.getvalue())


def classify_daily(opening, high, low, node):
    if low>node:
        return 'NO_TOUCH_ABOVE_NODE'
    if high<node:
        return 'GAP_BELOW_NO_RECLAIM'
    return 'DOWNSIDE_RANGE_TOUCH' if opening>=node else 'GAP_BELOW_POSSIBLE_RECLAIM'


def valid_daily(row):
    vals=np.array([row[k] for k in ('open','high','low','close')],dtype=float)
    return bool(np.isfinite(vals).all() and min(vals)>0 and
                row['low']<=min(row['open'],row['close'])<=max(row['open'],row['close'])<=row['high'])


def close_match(a,b):
    return abs(a-b)<=max(.002,2e-4*abs(b))


def verify_minute_day(frame, code, day):
    if frame.empty:
        return None, 'DAY_MISSING'
    frame=frame.sort_values('date')
    if not frame.code.astype(str).str.zfill(6).eq(code).all():
        return None,'CODE_MISMATCH'
    for _,group in frame[frame.duplicated('date',keep=False)].groupby('date'):
        if len(group[['open','high','low','close']].drop_duplicates())!=1:
            return None,'DUPLICATE_CONFLICT'
    frame=frame.drop_duplicates('date')
    if any(not valid_daily(row) for row in frame.to_dict('records')):
        return None,'INVALID_OHLC'
    if int(frame.date.iloc[0])!=day*1000000+93000 or int(frame.date.iloc[-1])<day*1000000+145900:
        return None,'SESSION_EDGE_MISSING'
    bars=kernel.to_bars(frame.to_dict('records'))
    five,segments,gaps=kernel.aggregate_5min(bars,day)
    if len(five)!=48 or gaps:
        return None,'INCOMPLETE_5M_GRID'
    return (frame,five,segments),'OK'


def confirmations(five, segments, node, atr):
    """Keep pre-touch legacy triggers separate; do not replace the frozen engine."""
    out={}
    first_touch=next((i for i,b in enumerate(five) if b['low']<=node),None)
    first_pierce=next((i for i,b in enumerate(five) if b['low']<node),None)
    rep=kernel.replay(five,segments,node,atr)
    index=rep['trigger']
    out['raw_rsi_reason']=rep['reason']
    out['raw_rsi_path']=rep.get('path')
    out['rsi_status']='NOT_TRIGGERED'
    if five[0]['open']<node:
        out['rsi_status']='OPEN_BELOW_NOT_OLD_DOWNSIDE_PATH'
    elif index is not None:
        if first_touch is None or index<first_touch:
            out['rsi_status']='LEGACY_SIGNAL_BEFORE_ACTUAL_TOUCH'
        else:
            out['rsi_status']='CONFIRMED_AFTER_TOUCH'
            out['rsi_index']=index
            out['rsi_time']=five[index]['t']
    reclaim=None
    if first_pierce is not None:
        for i in range(max(1,first_pierce+1),len(five)):
            if five[i-1]['close']>=node and five[i]['close']>=node:
                reclaim=i
                break
    out['reclaim_status']='TWO_BAR_PATTERN_RISK_UNKNOWN' if reclaim is not None else 'NO_TWO_BAR_PATTERN'
    if reclaim is not None:
        out['reclaim_index']=reclaim
        out['reclaim_time']=five[reclaim]['t']
    out['first_touch_index']=first_touch
    out['first_pierce_index']=first_pierce
    return out


def reference_outcome(five, index, e, old_end, delayed_end, daily, cache):
    """Next5m open is a reference, not a promised execution at signal close."""
    if index+1>=len(five):
        return {'outcome_status':'NO_NEXT_5M_ENTRY_BAR'}
    price=float(five[index+1]['open'])
    required=sorted(d for d in daily if e<d<=old_end)
    if not required or old_end not in daily or any(not valid_daily(daily[d]) for d in required):
        return {'outcome_status':'DAILY_WINDOW_INVALID'}
    events=[] if cache is None else cache.get('events',[])
    if cache is None or any(e<=int(str(day).replace('-','')[:8])<=old_end for day,_ in events):
        return {'outcome_status':'CORPORATE_ACTION_UNKNOWN'}
    ret=float(daily[old_end]['close']/price-1)
    lows=[b['low'] for b in five[index+1:]]+[daily[d]['low'] for d in required]
    highs=[b['high'] for b in five[index+1:]]+[daily[d]['high'] for d in required]
    out={'outcome_status':'REFERENCE_ONLY','entry_ref':price,
         'same_end_return':ret,'same_end_y':int(ret>0),
         'same_end_cost_sensitivity':float((daily[old_end]['close']*.9982)/(price*1.0013)-1),
         'post_entry_mae':float(min(lows)/price-1),'post_entry_mfe':float(max(highs)/price-1),
         'same_day_locked_mae':float(min(b['low'] for b in five[index+1:])/price-1)}
    if delayed_end in daily and valid_daily(daily[delayed_end]) and not any(
            e<=int(str(day).replace('-','')[:8])<=delayed_end for day,_ in events):
        dr=float(daily[delayed_end]['close']/price-1)
        out.update(delayed_end_return=dr,delayed_end_y=int(dr>0))
    return out


def summarize(frame, minute):
    result={}
    frame=frame.assign(year=frame.P//10000)
    for (year,stage),g in frame.groupby(['year','stage'],sort=True):
        crossed=g[g.daily_path.isin(['DOWNSIDE_RANGE_TOUCH','GAP_BELOW_POSSIBLE_RECLAIM'])]
        result[f'{year}:{stage}']={
            'n':len(g),'old_positive':int(g.y7.sum()),'paths':g.daily_path.value_counts().to_dict(),
            'old_positive_no_touch_above':int(((g.y7==1)&g.daily_path.eq('NO_TOUCH_ABOVE_NODE')).sum()),
            'node_reference_comparable':len(crossed),
            'old_negative_node_positive':int(((crossed.y7==0)&(crossed.node_reference_y==1)).sum()),
            'old_positive_node_negative':int(((crossed.y7==1)&(crossed.node_reference_y==0)).sum())}
    signal={}
    for kind in ('rsi','reclaim'):
        col=kind+'_same_end_return'
        if col not in minute:
            signal[kind]={'comparable':0}
            continue
        good=minute[minute[kind+'_outcome_status'].eq('REFERENCE_ONLY')]
        signal[kind]={'comparable':len(good),'days':int(good.P.nunique()),
            'old_positive':int(good.y7.sum()),
            'reference_positive':int(good[kind+'_same_end_y'].sum()),
            'disagree':int((good.y7!=good[kind+'_same_end_y']).sum()),
            'old_negative_reference_positive':int(((good.y7==0)&(good[kind+'_same_end_y']==1)).sum()),
            'old_positive_reference_negative':int(((good.y7==1)&(good[kind+'_same_end_y']==0)).sum()),
            'mean_reference_return':float(good[col].mean()),
            'mean_cost_sensitivity':float(good[kind+'_same_end_cost_sensitivity'].mean()),
            'mean_post_entry_mae':float(good[kind+'_post_entry_mae'].mean()),
            'delayed_end_label_flips':int((good[kind+'_delayed_end_y'].notna() &
                 (good[kind+'_delayed_end_y']!=good[kind+'_same_end_y'])).sum())
                 if kind+'_delayed_end_y' in good else None}
    return {'daily_by_year_stage':result,'signal_label_comparison':signal}


def main():
    if (HERE/'started.json').exists():
        raise ValueError('DIAGNOSTIC_ALREADY_STARTED_NO_RESCAN')
    population=MODEL_ROOT/'predictions.parquet'
    manifest=json.loads((MODEL_ROOT/'source_manifest.json').read_bytes())
    dump('started.json',{'contract':fit.sha(CONTRACT),'script':fit.sha(__file__),
        'population':fit.sha(population),'frozen_model':fit.sha(MODEL_ROOT/'model.pkl'),
        'trigger':fit.sha(TRIGGER),'registered_bytes':fit.usage()})
    budget_reads=0
    for path,info in manifest['daily'].items():
        if fit.sha(path)!=info['sha256']:
            raise ValueError('FROZEN_DAILY_SOURCE_CHANGED')
        budget_reads+=2*info['size']
    dataset=ds.dataset(list(manifest['daily']),format='parquet')
    pop=pd.read_parquet(population).sort_values(['code','P']).reset_index(drop=True)
    calendar=np.sort(dataset.to_table(columns=['date']).column('date').unique().to_numpy())
    calidx={int(day):i for i,day in enumerate(calendar)}
    wanted={code:set() for code in pop.code.unique()}
    for item in pop.itertuples():
        i=calidx[int(item.P)]
        wanted[item.code].update(int(day) for day in calendar[i:i+9])
    daily=dataset.to_table(columns=['code','date','open','high','low','close'],
        filter=ds.field('code').isin(pop.code.unique().tolist()) & (ds.field('date')>=int(pop.P.min()))).to_pandas()
    for _,g in daily[daily.duplicated(['code','date'],keep=False)].groupby(['code','date']):
        if len(g[['open','high','low','close']].drop_duplicates())!=1:
            raise ValueError('BOUND_DAILY_CONFLICT')
    daily=daily.drop_duplicates(['code','date']).sort_values(['code','date'])
    groups={code:g[g.date.isin(wanted[code])].set_index('date')[['open','high','low','close']].to_dict('index')
            for code,g in daily.groupby('code',sort=False)}
    del daily
    fit.peak_memory()
    cache=json.loads(fit.CACHE.read_bytes())
    rows=[]
    for item in pop.itertuples():
        i=calidx[int(item.P)]
        e,delayed=int(calendar[i+1]),int(calendar[i+8])
        dmap=groups[item.code]
        record={k:getattr(item,k) for k in ('code','P','stage','node','close','y7','chg7','label_end','split','score','rank','top5')}
        record.update(E=e,delayed_end=delayed,atr=float(item.atr_pct*item.close))
        bar=dmap.get(e)
        if bar is None or not valid_daily(bar):
            record['daily_path']='DAILY_MISSING_OR_INVALID'
        else:
            record.update(E_open=bar['open'],E_high=bar['high'],E_low=bar['low'],E_close=bar['close'],
                daily_path=classify_daily(bar['open'],bar['high'],bar['low'],item.node))
            expected=float(dmap[int(item.label_end)]['close']/item.close-1)
            if abs(expected-item.chg7)>1e-12 or int(expected>0)!=item.y7:
                raise ValueError('OLD_LABEL_REPLAY_MISMATCH')
            if bar['low']<=item.node<=bar['high']:
                ret=float(dmap[int(item.label_end)]['close']/item.node-1)
                record.update(node_reference_return=ret,node_reference_y=int(ret>0))
        rows.append(record)
    frame=pd.DataFrame(rows)
    table('daily_alignment.parquet',frame)
    minute_rows=[]
    receipts=[]
    prefix_checks=[]
    for number,(code,g) in enumerate(frame.groupby('code',sort=True)):
        modern=g[g.P>=20250101]
        needed=modern[modern.daily_path.isin(['DOWNSIDE_RANGE_TOUCH','GAP_BELOW_POSSIBLE_RECLAIM'])]
        if needed.empty:
            continue
        path=MINUTES/f'{code}.parquet'
        load_status='OK'
        source_groups={}
        if not path.exists():
            load_status='CACHE_FILE_MISSING'
        elif budget_reads+3*path.stat().st_size>READ_CAP:
            load_status='READ_BUDGET_EXHAUSTED_UNKNOWN'
        else:
            size=path.stat().st_size
            budget_reads+=3*size
            receipt={'path':str(path),'bytes':size,'sha256':fit.sha(path)}
            receipts.append(receipt)
            raw=pq.read_table(path,columns=['code','date','open','high','low','close'],
                filters=[('date','>=',int(needed.P.min())*1000000),
                         ('date','<=',int(needed.E.max())*1000000+150000)]).to_pandas()
            if fit.sha(path)!=receipt['sha256']:
                raise ValueError('MINUTE_SOURCE_CHANGED_DURING_READ')
            source_groups={int(day):x for day,x in raw.groupby(raw.date//1000000,sort=False)}
        for item in needed.itertuples():
            rec={k:getattr(item,k) for k in ('code','P','E','stage','split','y7','chg7','node','label_end','score','top5')}
            rec['minute_status']=load_status
            if load_status!='OK':
                minute_rows.append(rec)
                continue
            before=source_groups.get(item.P,pd.DataFrame())
            day=source_groups.get(item.E,pd.DataFrame())
            if before.empty or int(before.date.max())<item.P*1000000+145900:
                rec['minute_status']='P_FINAL_MINUTE_MISSING'
                minute_rows.append(rec)
                continue
            previous=before.sort_values('date').iloc[-1]
            if not before.code.astype(str).str.zfill(6).eq(code).all():
                rec['minute_status']='P_CODE_MISMATCH'
                minute_rows.append(rec)
                continue
            final_group=before[before.date==previous.date]
            if final_group.close.nunique()!=1:
                rec['minute_status']='P_CLOSE_DUPLICATE_CONFLICT'
                minute_rows.append(rec)
                continue
            if not close_match(float(previous.close),float(item.close)):
                rec['minute_status']='P_PRICE_SCALE_MISMATCH'
                minute_rows.append(rec)
                continue
            verified,status=verify_minute_day(day,code,int(item.E))
            if status!='OK':
                rec['minute_status']=status
                minute_rows.append(rec)
                continue
            day,five,segments=verified
            actual={'open':float(day.open.iloc[0]),'high':float(day.high.max()),
                    'low':float(day.low.min()),'close':float(day.close.iloc[-1])}
            expected=groups[code][int(item.E)]
            if not all(close_match(actual[k],expected[k]) for k in actual):
                rec['minute_status']='E_OHLC_SCALE_MISMATCH'
                minute_rows.append(rec)
                continue
            signal=confirmations(five,segments,float(item.node),float(item.atr))
            rec.update(signal)
            for kind in ('rsi','reclaim'):
                idx=signal.get(kind+'_index')
                if idx is None:
                    continue
                outcome=reference_outcome(five,idx,int(item.E),int(item.label_end),
                    int(item.delayed_end),groups[code],cache.get(code))
                rec.update({kind+'_'+k:v for k,v in outcome.items()})
            # Deterministic first12 confirmations: prefix vs full-day exact signal.
            check_index=signal.get('rsi_index',signal.get('reclaim_index'))
            if check_index is not None and len(prefix_checks)<12:
                prefix=confirmations(five[:check_index+1],segments[:check_index+1],float(item.node),float(item.atr))
                for kind in ('rsi','reclaim'):
                    if signal.get(kind+'_index',999)<=check_index and prefix.get(kind+'_index')!=signal.get(kind+'_index'):
                        raise ValueError('SIGNAL_PREFIX_MISMATCH')
                prefix_checks.append({'code':code,'P':int(item.P),'E':int(item.E),'bar':check_index})
            minute_rows.append(rec)
        if number%150==0:
            print('minute bound stocks',number,'records',len(minute_rows),'readGiB',round(budget_reads/1024**3,2),flush=True)
            fit.peak_memory()
    minute=pd.DataFrame(minute_rows)
    table('minute_alignment.parquet',minute)
    dump('minute_source_manifest.json',{'entries':receipts,'compressed_read_upper_bound':budget_reads,
                                     'no_http':True,'no_source_copies':True})
    out=summarize(frame,minute)
    legacy_path=fit.OLD/'full_intraday_replay_v2.parquet'
    legacy=pd.read_parquet(legacy_path,columns=['code','P','F','status','trigger_t'])
    joined=frame.merge(legacy,on=['code','P'],how='inner')
    equal_node=np.isclose(joined.node,joined.F,atol=1e-6,rtol=0)
    out['legacy_replay_binding']={'file_sha256':fit.sha(legacy_path),'matched_keys':len(joined),
        'same_frozen_node':int(equal_node.sum()),
        'same_node_triggered':int((equal_node&joined.status.eq('TRIGGERED')).sum()),
        'reason':'single-candleF618 records cannot substitute for wave/cycle nodes or a different outcome clock'}
    comparable=frame[frame.node_reference_y.notna()]
    out.update(total=len(frame),daily_paths=frame.daily_path.value_counts().to_dict(),
        old_positive=int(frame.y7.sum()),
        old_positive_no_touch_above=int(((frame.y7==1)&frame.daily_path.eq('NO_TOUCH_ABOVE_NODE')).sum()),
        node_reference_n=len(comparable),
        node_reference_disagree=int((comparable.y7!=comparable.node_reference_y).sum()),
        minute_needed=len(minute),minute_status=minute.minute_status.value_counts().to_dict(),
        rsi_status=minute.get('rsi_status',pd.Series(dtype=str)).value_counts().to_dict(),
        reclaim_status=minute.get('reclaim_status',pd.Series(dtype=str)).value_counts().to_dict(),
        prefix_checks=prefix_checks,fit_count=0,policy_risk_complete_signals=0,
        risk_status='NOT_EVALUATED_DO_NOT_CALL_BUY_RECOMMENDATION',
        peak_working_set_bytes=fit.peak_memory())
    dump('summary.json',out)
    # Deterministic examples, no selection by the size of subsequent profits.
    examples={}
    for name,mask in {
        'old_win_without_touch':(frame.y7==1)&frame.daily_path.eq('NO_TOUCH_ABOVE_NODE'),
        'old_loss_but_node_reference_win':(frame.y7==0)&(frame.node_reference_y==1)
    }.items():
        examples[name]=frame[mask].sort_values(['P','code']).head(5).to_dict('records')
    for kind in ('rsi','reclaim'):
        col=kind+'_same_end_y'
        if col in minute:
            examples[kind+'_label_flip']=minute[minute[col].notna()&(minute.y7!=minute[col])].sort_values(['P','code']).head(5).to_dict('records')
    dump('examples.json',examples)
    dump('terminal.json',{'status':'LABEL_ALIGNMENT_DIAGNOSIS_COMPLETE_NO_FIT',
        'fit_count':0,'model_unchanged':fit.sha(MODEL_ROOT/'model.pkl')==json.loads((HERE/'started.json').read_bytes())['frozen_model'],
        'summary':fit.sha(HERE/'summary.json'),'rows':len(frame),'minute_rows':len(minute),
        'source_prefix_checks':len(prefix_checks),'registered_bytes':fit.usage()})
    print(json.dumps(clean({k:v for k,v in out.items() if k not in ('daily_by_year_stage','prefix_checks')}),ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        if not (HERE/'failure.json').exists():
            dump('failure.json',{'status':'FAILED_RETAINED','error':repr(exc),'fit_count':0})
        raise
