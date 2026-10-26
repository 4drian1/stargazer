"""Arithmetic/readback acceptance for fixed prediction diagnostics; no fit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from diagnose_repaired_stability_v1 import OUT,FIT,FIRST,TARGET
from postclose_repaired_training_v1 import prepare,CAP
from inventory_ths_stockdb_cache_v1 import digest
from research_budget_v2 import TOTAL_CAP,bytes_used


def run():
    plan=json.loads((OUT/'plan.json').read_bytes())
    for path,sha in plan['inputs'].items():
        if digest(Path(path).read_bytes())!=sha: raise ValueError('FROZEN_INPUT_DRIFT:'+path)
    fields,_,parts,_=prepare()
    p=pd.read_parquet(FIT/'predictions.parquet'); test=p[p.segment.eq('test')].copy()
    original=pd.read_parquet(FIRST/'predictions.parquet')
    keys=set(zip(original.loc[original.segment.eq('test'),'code'],original.loc[original.segment.eq('test'),'day']))
    test['common']=[(c,d) in keys for c,d in zip(test.code,test.day)]
    r=json.loads((OUT/'report.json').read_bytes())
    dates=sorted(test.day.unique()); checked=0
    for name,frame in dict(all=test,common=test[test.common],added=test[~test.common]).items():
        for month,facts in r['monthly'][name].items():
            g=frame[frame.day.astype(str).str.startswith(month)]
            if len(g)!=facts['rows']: raise ValueError('MONTH_ROWS_DRIFT')
            if len(g):
                hits=g[TARGET].to_numpy()<=g.q90_adverse.to_numpy()
                np.testing.assert_allclose(facts['coverage'],hits.mean(),rtol=0,atol=1e-14)
                perdate=[hits[g.day.to_numpy()==d].mean() for d in sorted(g.day.unique())]
                np.testing.assert_allclose(facts['date_weighted_coverage'],np.mean(perdate),rtol=0,atol=1e-14)
                np.testing.assert_allclose(facts['residual_q90_diagnostic_only'],np.quantile(g[TARGET]-g.q90_raw,.9),rtol=0,atol=1e-14)
            elif facts['coverage'] is not None: raise ValueError('EMPTY_GROUP_NOT_NULL')
        windows=r['rolling'][name]
        if len(windows)!=max(0,len(dates)-19): raise ValueError('WINDOW_COUNT_DRIFT')
        for i,w in enumerate(windows):
            start,end=dates[i],dates[i+19]
            if (w['start'],w['end'])!=(start,end): raise ValueError('WINDOW_END_DRIFT')
            g=frame[frame.day.between(start,end)]
            if len(g)!=w['rows']: raise ValueError('WINDOW_ROWS_DRIFT')
            if len(g): np.testing.assert_allclose(w['coverage'],(g[TARGET]<=g.q90_adverse).mean(),rtol=0,atol=1e-14)
            elif w['coverage'] is not None: raise ValueError('EMPTY_WINDOW_NOT_NULL')
            checked+=1
    shifts=json.loads((OUT/'feature_shift_all131.json').read_bytes())
    for month,profiles in shifts.items():
        if list(profiles)!=fields: raise ValueError('FEATURE_SET_DRIFT')
        g=parts['test'][parts['test'].day.astype(str).str.startswith(month)]
        for c,facts in profiles.items():
            np.testing.assert_allclose(facts['month_missing'],g[c].isna().mean(),rtol=0,atol=1e-14)
            a=parts['train'][c].dropna().to_numpy(dtype=float); b=g[c].dropna().to_numpy(dtype=float)
            if len(b):
                np.testing.assert_allclose(facts['mean_shift_train_sd'],(b.mean()-a.mean())/a.std(),rtol=0,atol=1e-12)
    comparison=json.loads((FIT/'comparison_and_readback.json').read_bytes())
    for name,sha in comparison['new_fit_sha256'].items():
        if digest((FIT/name).read_bytes())!=sha: raise ValueError('DIAGNOSTICS_CHANGED_FIT')
    result=dict(status='PASS_FIXED_PREDICTION_STABILITY_READBACK',rolling_windows_checked=checked,
        feature_month_profiles_checked=sum(len(x) for x in shifts.values()),
        inputs_unchanged=True,actual_fit=False,recalibration_applied=False,
        hashes={n:digest((OUT/n).read_bytes()) for n in ['plan.json','report.json','feature_shift_all131.json','stop_record.json']},
        audit_code_sha256=digest(Path(__file__).read_bytes()))
    body=json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8')
    if sum(f.stat().st_size for f in FIT.rglob('*') if f.is_file())+len(body)>CAP or bytes_used()+len(body)>TOTAL_CAP: raise ValueError('AUDIT_BUDGET_STOP')
    with (OUT/'readback_acceptance.json').open('xb') as stream:stream.write(body)
    print(json.dumps(result))


if __name__=='__main__':run()
