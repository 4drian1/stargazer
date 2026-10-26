"""Frozen prediction diagnostics; no fit, recalibration or new predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from audit_postclose_first_fit_v1 import loss, verify_rows
from inventory_ths_stockdb_cache_v1 import HERE, digest
from postclose_repaired_training_v1 import OUT as FIT, FIRST, prepare, CAP
from research_budget_v2 import TOTAL_CAP, bytes_used

OUT=FIT/'stability_v1'
SPEC=HERE/'contracts/POSTCLOSE_REPAIRED_STABILITY_DIAGNOSTICS_V1.json'
TARGET='next_session_adverse_excursion'


def summarize(frame, tail):
    if frame.empty: return dict(rows=0,dates=0,stocks=0,coverage=None,date_weighted_coverage=None,
        raw_coverage=None,pinball=None,raw_pinball=None,mean_prediction=None,target_q90=None,
        residual_q90_diagnostic_only=None,tail_rate=None)
    y=frame[TARGET].to_numpy(float); p=frame.q90_adverse.to_numpy(float); raw=frame.q90_raw.to_numpy(float)
    hits=y<=p
    date_hits=pd.DataFrame(dict(day=frame.day.to_numpy(),hit=hits)).groupby('day').hit.mean()
    return dict(rows=len(frame),dates=int(frame.day.nunique()),stocks=int(frame.code.nunique()),
        coverage=float(hits.mean()),date_weighted_coverage=float(date_hits.mean()),
        raw_coverage=float((y<=raw).mean()),pinball=float(loss(y,p).mean()),raw_pinball=float(loss(y,raw).mean()),
        mean_prediction=float(p.mean()),target_q90=float(np.quantile(y,.9)),
        residual_q90_diagnostic_only=float(np.quantile(y-raw,.9)),tail_rate=float((y>tail).mean()))


def rolling(frame, dates, tail, window=20):
    return [dict(start=int(dates[i-window+1]),end=int(dates[i]),
                 **summarize(frame[frame.day.isin(dates[i-window+1:i+1])],tail))
            for i in range(window-1,len(dates))]


def coverage_bins(values, edges):
    return np.where(np.isfinite(values),np.searchsorted(edges,values,side='left'),-1)


def shift_profile(train, month, fields):
    result={}
    for field in fields:
        a=train[field].to_numpy(float); b=month[field].to_numpy(float)
        a=a[np.isfinite(a)]; valid=np.isfinite(b); finite=b[valid]
        std=float(a.std()) if len(a) else 0.
        result[field]=dict(train_missing=float(train[field].isna().mean()),month_missing=float((~valid).mean()),
            mean_shift_train_sd=float((finite.mean()-a.mean())/std) if len(finite) and std>0 else None,
            outside_train_range=float(((finite<a.min())|(finite>a.max())).mean()) if len(finite) and len(a) else None)
    return result


def run():
    if OUT.exists(): raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    fields,_,parts,_=prepare()
    accepted=json.loads((FIT/'comparison_and_readback.json').read_bytes())
    refs={str(FIT/name):sha for name,sha in accepted['new_fit_sha256'].items()}
    for path in [SPEC,Path(__file__),HERE/'test_repaired_stability_v1.py',FIT/'comparison_and_readback.json',
                 FIRST/'predictions.parquet',HERE/'audit_postclose_first_fit_v1.py']:
        refs[str(path)]=digest(path.read_bytes())
    for path,sha in refs.items():
        if digest(Path(path).read_bytes())!=sha: raise ValueError('FROZEN_INPUT_DRIFT:'+path)
    predictions=pd.read_parquet(FIT/'predictions.parquet')
    verify_rows(predictions,pd.concat([f.assign(segment=s) for s,f in parts.items()],ignore_index=True))
    context=['sse_lag1_price_return','sse_daily_rsi14','context_sector_mean_coverage_loo']
    features=pd.concat([f[['code','day',*context]] for f in parts.values()],ignore_index=True)
    data=predictions.merge(features,on=['code','day'],validate='one_to_one')
    train=data[data.segment.eq('train')]
    test=data[data.segment.eq('test')].copy()
    old=pd.read_parquet(FIRST/'predictions.parquet')
    tail=float(np.quantile(old.loc[old.segment.eq('train'),TARGET],.9))
    oldkeys=pd.MultiIndex.from_frame(old.loc[old.segment.eq('test'),['code','day']])
    test['common']=pd.MultiIndex.from_frame(test[['code','day']]).isin(oldkeys)
    test['month']=test.day.astype(str).str[:6]
    edges=np.quantile(train.context_sector_mean_coverage_loo.dropna(),[1/3,2/3]).tolist()
    dates=np.sort(test.day.unique())
    report=dict(status='FIXED_PREDICTION_STABILITY_DIAGNOSTICS',
        fixed_calibration_shift=json.loads((FIT/'report.json').read_bytes())['quantile_shift'],
        diagnostic_tail_threshold=tail,sector_coverage_train_edges=edges,
        monthly={},rolling={},background={},stock_month=[],fit_run=False,recalibrated=False,
        retrospective_labels_never_used_for_prediction=True,independent_blind_test=False)
    OUT.mkdir(exist_ok=False)
    def write(name,value,terminal=False):
        body=json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8')
        reserve=0 if terminal else 65536
        if sum(p.stat().st_size for p in FIT.rglob('*') if p.is_file())+len(body)+reserve>CAP or bytes_used()+len(body)+reserve>TOTAL_CAP:
            raise ValueError('STABILITY_BUDGET_STOP')
        with (OUT/name).open('xb') as stream:stream.write(body)
    write('plan.json',dict(inputs=refs,method=json.loads(SPEC.read_bytes())))
    try:
        populations=dict(all=test,common=test[test.common],added=test[~test.common])
        for name,frame in populations.items():
            report['monthly'][name]={str(month):summarize(frame[frame.month.eq(month)],tail) for month in sorted(test.month.unique())}
            report['rolling'][name]=rolling(frame,dates,tail)
        groups=dict(market_day_return=np.where(test.sse_lag1_price_return.isna(),'unknown',np.where(test.sse_lag1_price_return<0,'negative','nonnegative')),
            market_rsi=np.where(test.sse_daily_rsi14.isna(),'unknown',np.where(test.sse_daily_rsi14<30,'below30',np.where(test.sse_daily_rsi14<70,'30to70','atleast70'))),
            sector_coverage=coverage_bins(test.context_sector_mean_coverage_loo.to_numpy(float),edges))
        for name,values in groups.items():
            report['background'][name]={str(value):summarize(test.loc[values==value],tail) for value in np.unique(values)}
        for (code,month),frame in test.groupby(['code','month'],sort=True):
            report['stock_month'].append(dict(code=code,month=month,**summarize(frame,tail)))
        shifts={str(month):shift_profile(parts['train'],parts['test'][parts['test'].day.astype(str).str[:6].eq(month)],fields)
                for month in sorted(test.month.unique())}
        for path,sha in refs.items():
            if digest(Path(path).read_bytes())!=sha: raise ValueError('INPUT_CHANGED_DURING_DIAGNOSTICS')
        write('feature_shift_all131.json',shifts)
        write('report.json',report)
        write('stop_record.json',dict(status='COMPLETE_NO_FIT_NO_RECALIBRATION'),terminal=True)
        print(json.dumps(dict(status=report['status'],fixed_calibration_shift=report['fixed_calibration_shift'],
            monthly=report['monthly'],background=report['background'],rolling_windows={k:len(v) for k,v in report['rolling'].items()}),ensure_ascii=False,indent=2))
        return report
    except Exception as exc:
        write('stop_record.json',dict(status='FAILED',error=f'{type(exc).__name__}:{exc}'),terminal=True)
        raise


if __name__=='__main__':run()
