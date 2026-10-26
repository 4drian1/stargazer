"""Check every residual selection and actual label-perturbation causality."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from replay_causal_calibration_v1 import OUT,FIT,AUTH,replay,TARGET
from postclose_repaired_training_v1 import prepare,CAP
from inventory_ths_stockdb_cache_v1 import digest
from research_budget_v2 import TOTAL_CAP,bytes_used


def run():
    plan=json.loads((OUT/'plan.json').read_bytes())
    for path,sha in plan['inputs'].items():
        if digest(Path(path).read_bytes())!=sha:raise ValueError('INPUT_DRIFT:'+path)
    _,_,parts,_=prepare()
    history=pd.read_parquet(FIT/'predictions.parquet')
    predictions=pd.read_parquet(OUT/'predictions.parquet')
    original=history[history.segment.eq('test')].merge(parts['test'][['code','day','decision_ts_assumed_end','stock_return1d']],on=['code','day'],validate='one_to_one')
    pd.testing.assert_frame_equal(predictions.drop(columns='rolling_q90_adverse'),original)
    logs=json.loads((OUT/'calibration_ledger.json').read_bytes())
    report=json.loads((OUT/'report.json').read_bytes())
    base=plan['base_shift']; pool=history[history.segment.isin(['calibration','test'])]
    for row in logs:
        now=row['decision_ts']; day=row['day']
        past=pool[(pool.day<day)&(pool.label_end_time<now)]
        dates=sorted(past.day.unique())
        if len(dates)>=60:
            used=past[past.day.isin(dates[-60:])]
            expected=float(np.percentile((used[TARGET]-used.q90_raw).to_numpy(float),90))
            if row['mode']!='ROLLING_60_MATURE_DATES' or row['used_dates']!=dates[-60:]:raise ValueError('WINDOW_DRIFT')
        else:
            used=pool[pool.segment.eq('calibration')]; expected=base
            if not ((used.day<day)&(used.label_end_time<now)).all():raise ValueError('UNMATURED_BASE')
            if row['mode']!='FROZEN_BASE_COLD_START':raise ValueError('COLD_MODE_DRIFT')
        if row['used_rows']!=len(used) or row['max_label_end_used']!=int(used.label_end_time.max()):raise ValueError('LEDGER_DRIFT')
        if not row['max_label_end_used']<now:raise ValueError('LABEL_LEAKAGE')
        np.testing.assert_allclose(row['shift'],expected,rtol=0,atol=1e-14)
        today=predictions[predictions.day.eq(day)]
        np.testing.assert_allclose(today.rolling_q90_adverse,np.clip(today.q90_raw+expected,0,1),rtol=0,atol=1e-14)
    perturb=[]
    dates=sorted(predictions.day.unique())
    for index in [0,30,60,len(dates)-1]:
        day=dates[index]; dec=original[original.day<=day]
        now=int(dec.loc[dec.day.eq(day),'decision_ts_assumed_end'].iloc[0])
        truncated=history[history.day<=day]
        prefix,_=replay(truncated,dec,base)
        modified=history.copy()
        modified.loc[(modified.day>=day)|(modified.label_end_time>=now),TARGET]=.99
        changed,_=replay(modified,dec,base)
        stored=predictions[predictions.day<=day]
        pd.testing.assert_frame_equal(prefix,stored)
        pd.testing.assert_frame_equal(changed,stored)
        perturb.append(int(day))
    for method,column in [('fixed','q90_adverse'),('raw','q90_raw'),('rolling','rolling_q90_adverse')]:
        facts=report['populations']['all'][method]['overall']; y=predictions[TARGET].to_numpy();p=predictions[column].to_numpy()
        residual=y-p; value=np.maximum(.9*residual,-.1*residual).mean()
        np.testing.assert_allclose(facts['pinball'],value,rtol=0,atol=1e-14)
        np.testing.assert_allclose(facts['coverage'],(y<=p).mean(),rtol=0,atol=1e-14)
    for path,sha in plan['inputs'].items():
        if digest(Path(path).read_bytes())!=sha:raise ValueError('INPUT_CHANGED_DURING_AUDIT')
    result=dict(status='PASS_STRICT_MATURITY_AND_CAUSAL_PREFIX_REPLAY',decision_ledgers_checked=len(logs),
        actual_prefix_and_current_future_label_perturbation_days=perturb,original_predictions_unchanged=True,
        model_refit=False,three_method_overall_arithmetic_pass=True,
        hashes={n:digest((OUT/n).read_bytes()) for n in ['plan.json','report.json','predictions.parquet','calibration_ledger.json','stop_record.json']},
        audit_code_sha256=digest(Path(__file__).read_bytes()))
    body=json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8')
    if sum(f.stat().st_size for f in FIT.rglob('*') if f.is_file())+len(body)>CAP or bytes_used()+len(body)>TOTAL_CAP:raise ValueError('AUDIT_BUDGET_STOP')
    with (OUT/'readback_acceptance.json').open('xb') as stream:stream.write(body)
    print(json.dumps(result))


if __name__=='__main__':run()
