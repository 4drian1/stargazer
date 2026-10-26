"""One authorized past-matured residual calibration replay, no model refit."""
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from audit_postclose_first_fit_v1 import loss, verify_rows, day_block_interval
from diagnose_repaired_stability_v1 import summarize, rolling, TARGET
from diagnose_postclose_predictions_v1 import filter_by_day, filter_summary
from inventory_ths_stockdb_cache_v1 import HERE, digest
from postclose_repaired_training_v1 import OUT as FIT, FIRST, prepare, CAP
from research_budget_v2 import TOTAL_CAP, bytes_used

OUT=FIT/'causal_calibration_v1'
AUTH=HERE/'contracts/POSTCLOSE_CAUSAL_CALIBRATION_REPLAY_V1.json'


def replay(history, decisions, base_shift, window=60):
    if history[['code','day']].duplicated().any() or decisions[['code','day']].duplicated().any():
        raise ValueError('DUPLICATE_HISTORY_OR_DECISION')
    if not np.isfinite(base_shift) or not np.isfinite(decisions.q90_raw.to_numpy(float)).all():
        raise ValueError('NONFINITE_CALIBRATION_INPUT')
    pool=history[history.segment.isin(['calibration','test'])].copy()
    valid=np.isfinite(pool[[TARGET,'q90_raw','label_end_time']].to_numpy(float)).all(axis=1)
    valid &= pool[TARGET].between(0,1,inclusive='left').to_numpy()
    calibration=pool[pool.segment.eq('calibration')]
    qualified=pool.loc[valid].copy()
    qualified['residual']=qualified[TARGET]-qualified.q90_raw
    result=decisions.copy(); result['rolling_q90_adverse']=np.nan
    logs=[]
    for day,g in decisions.groupby('day',sort=True):
        times=g.decision_ts_assumed_end.unique()
        if len(times)!=1: raise ValueError('MULTIPLE_DECISION_TIMES_PER_DAY')
        now=int(times[0])
        past=qualified[(qualified.day<day)&(qualified.label_end_time<now)]
        dates=np.sort(past.day.unique())
        if len(dates)>=window:
            used_dates=dates[-window:]
            used=past[past.day.isin(used_dates)]
            delta=float(np.quantile(used.residual,.9)); mode='ROLLING_60_MATURE_DATES'
        else:
            if calibration.empty or not np.isfinite(calibration[[TARGET,'q90_raw','label_end_time']].to_numpy(float)).all():
                raise ValueError('BASE_CALIBRATION_NOT_MATURE_OR_VALID')
            if not ((calibration.label_end_time<now)&(calibration.day<day)).all():
                raise ValueError('BASE_CALIBRATION_NOT_MATURE_OR_VALID')
            used=calibration; used_dates=np.sort(used.day.unique())
            delta=float(base_shift); mode='FROZEN_BASE_COLD_START'
        result.loc[g.index,'rolling_q90_adverse']=np.clip(g.q90_raw.to_numpy(float)+delta,0,1)
        logs.append(dict(day=int(day),decision_ts=now,mode=mode,shift=delta,
            available_qualified_dates=len(dates),used_dates=[int(d) for d in used_dates],used_rows=len(used),
            max_label_end_used=int(used.label_end_time.max())))
    return result,logs


def run(authorize=False):
    if not authorize: raise ValueError('EXPLICIT_CALIBRATION_REPLAY_AUTHORIZATION_REQUIRED')
    if OUT.exists(): raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    auth=json.loads(AUTH.read_bytes())
    if auth['window_qualified_decision_days']!=60 or auth['alpha']!=.9:
        raise ValueError('FROZEN_METHOD_DRIFT')
    _,_,parts,_=prepare()
    acceptance=json.loads((FIT/'comparison_and_readback.json').read_bytes())
    refs={str(FIT/n):sha for n,sha in acceptance['new_fit_sha256'].items()}
    for p in [AUTH,Path(__file__),HERE/'test_causal_calibration_v1.py',
              FIT/'comparison_and_readback.json',FIRST/'predictions.parquet',
              HERE/'diagnose_repaired_stability_v1.py',HERE/'diagnose_postclose_predictions_v1.py',
              HERE/'audit_postclose_first_fit_v1.py']:
        refs[str(p)]=digest(p.read_bytes())
    for p,sha in refs.items():
        if digest(Path(p).read_bytes())!=sha: raise ValueError('FROZEN_INPUT_DRIFT:'+p)
    history=pd.read_parquet(FIT/'predictions.parquet')
    verify_rows(history,pd.concat([f.assign(segment=s) for s,f in parts.items()],ignore_index=True))
    decisions=history[history.segment.eq('test')].merge(parts['test'][['code','day','decision_ts_assumed_end','stock_return1d']],on=['code','day'],validate='one_to_one')
    base=float(json.loads((FIT/'report.json').read_bytes())['quantile_shift'])
    OUT.mkdir(exist_ok=False)
    def write(name,body,terminal=False):
        reserve=0 if terminal else 65536
        if sum(f.stat().st_size for f in FIT.rglob('*') if f.is_file())+len(body)+reserve>CAP or bytes_used()+len(body)+reserve>TOTAL_CAP:
            raise ValueError('CALIBRATION_REPLAY_BUDGET_STOP')
        with (OUT/name).open('xb') as stream:stream.write(body)
    def write_json(name,value,terminal=False):
        write(name,json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8'),terminal)
    write_json('plan.json',dict(inputs=refs,method=auth,base_shift=base))
    try:
        predictions,logs=replay(history,decisions,base)
        old=pd.read_parquet(FIRST/'predictions.parquet')
        oldkeys=set(zip(old.loc[old.segment.eq('test'),'code'],old.loc[old.segment.eq('test'),'day']))
        common=np.array([(c,d) in oldkeys for c,d in zip(predictions.code,predictions.day)])
        tail=float(np.quantile(old.loc[old.segment.eq('train'),TARGET],.9))
        dates=np.sort(predictions.day.unique()); months=sorted(predictions.day.astype(str).str[:6].unique())
        report=dict(status='COMPLETE_SINGLE_CAUSAL_CALIBRATION_REPLAY_NOT_DEPLOYMENT',model_refit=False,
            new_calibration_estimates=True,independent_blind_test=False,deployment_authorized=False,
            populations={},base_shift=base,decision_dates=len(logs),
            cold_start_dates=sum(r['mode']=='FROZEN_BASE_COLD_START' for r in logs),
            updated_dates=sum(r['mode']=='ROLLING_60_MATURE_DATES' for r in logs))
        for name,subset in dict(all=predictions,common=predictions.loc[common],added=predictions.loc[~common]).items():
            report['populations'][name]={}
            for method in ['fixed','raw','rolling']:
                current=subset.copy()
                current['q90_adverse']=subset[{'fixed':'q90_adverse','raw':'q90_raw','rolling':'rolling_q90_adverse'}[method]].to_numpy()
                report['populations'][name][method]=dict(overall=summarize(current,tail),
                    monthly={month:summarize(current[current.day.astype(str).str[:6].eq(month)],tail) for month in months},
                    rolling20=rolling(current,dates,tail))
        diff=pd.DataFrame(dict(day=predictions.day,loss_difference=loss(predictions[TARGET],predictions.rolling_q90_adverse)-loss(predictions[TARGET],predictions.q90_adverse)))
        report['rolling_minus_fixed_pinball']=dict(mean=float(diff.loss_difference.mean()),**day_block_interval(diff))
        filters={}; selected={}; clipping={}
        for method,column in [('fixed','q90_adverse'),('rolling','rolling_q90_adverse')]:
            current=predictions.copy(); current['q90_adverse']=predictions[column]
            daily,coverage=filter_by_day(current,tail); filters[method]=dict(coverage=coverage,metrics=filter_summary(daily))
            clipping[method]=int(((current.q90_adverse==0)|(current.q90_adverse==1)).sum())
            selected[method]={}
            for day,g in current.groupby('day'):
                g=g.copy();g['tie']=[digest(f'{c}|{day}|fixed-diagnostic-tie-v1'.encode()) for c in g.code]
                if len(g)>=5:
                    selected[method][int(day)]=g.sort_values(['q90_adverse','tie'],ascending=[False,True]).code.tolist()
        report['same_day_filters']=filters
        report['clipped_rows']=clipping
        report['same_day_order_changed_dates']=sum(selected['fixed'][d]!=selected['rolling'][d] for d in selected['fixed'])
        for p,sha in refs.items():
            if digest(Path(p).read_bytes())!=sha: raise ValueError('INPUT_CHANGED_DURING_REPLAY')
        sink=io.BytesIO();predictions.to_parquet(sink,index=False,compression='zstd')
        write('predictions.parquet',sink.getvalue());write_json('calibration_ledger.json',logs)
        write_json('report.json',report);write_json('stop_record.json',dict(status='COMPLETE_SINGLE_WINDOW_NO_SEARCH'),True)
        print(json.dumps({k:v for k,v in report.items() if k not in ['populations','same_day_filters']},ensure_ascii=False,indent=2))
        print(json.dumps({m:report['populations']['all'][m]['overall'] for m in ['fixed','raw','rolling']},ensure_ascii=False,indent=2))
        return report
    except Exception as exc:
        write_json('stop_record.json',dict(status='FAILED',error=f'{type(exc).__name__}:{exc}'),True)
        raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--authorize-calibration-replay',action='store_true')
    run(parser.parse_args().authorize_calibration_replay)
