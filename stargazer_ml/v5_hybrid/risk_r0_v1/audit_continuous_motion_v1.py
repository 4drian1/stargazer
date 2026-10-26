"""Independent NumPy summary replay of the bounded engineering acceptance."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from continuous_motion_features_v1 import METRICS, entity_motion
from intraday_relative_stall_v1 import TIME
from materialize_relative_stall_pilot_v1 import sha
from research_budget_v2 import bytes_used, TOTAL_CAP
import verify_continuous_motion_v1 as verifier


def expected(source, prefix, keys, known, peak, frequency):
    f=source.assign(_known=known,_peak=peak).sort_values([*keys,TIME]).reset_index(drop=True)
    r=np.where(f._known.fillna(False),f[prefix+'interval_return'],np.nan)
    d=np.where(f._peak.fillna(False),np.maximum(0.,-f[prefix+'drawdown_from_close_day_high']),np.nan)
    t=f[TIME].to_numpy(); minute=t//10000%100*60+t//100%100
    pair=np.r_[False,(t[1:]//1000000==t[:-1]//1000000)&(minute[1:]-minute[:-1]==frequency)
                    &((minute[1:]<720)==(minute[:-1]<720))]
    for k in keys:
        a=f[k].to_numpy();pair[1:]&=a[1:]==a[:-1]
    dec=np.where(pair,np.roll(r,1)-r,np.nan)
    return f,np.c_[r,dec,d]


def stats(values,mask,report):
    assert int(mask.sum())==report['selected']
    for j,name in enumerate(METRICS):
        x=values[mask,j];known=x[np.isfinite(x)];old=report[name]
        assert old['known']==len(known) and old['unknown']==len(x)-len(known)
        assert old['coverage']==(len(known)/len(x) if len(x) else None)
        if len(known):
            np.testing.assert_allclose(old['mean'],known.mean(),atol=1e-14,rtol=0)
            np.testing.assert_allclose(old['quantiles'],np.quantile(known,[.1,.5,.9]),atol=1e-14,rtol=0)
        else:assert old['mean'] is None and old['quantiles'] is None


def run():
    out=verifier.OUT; report=json.loads((out/'report.json').read_bytes(),parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    plan=json.loads((out/'plan.json').read_bytes());manifest=json.loads((out/'manifest.json').read_bytes())
    assert json.loads((out/'stop_record.json').read_bytes())['manifest_sha256']==sha(out/'manifest.json')
    for p,h in plan['inputs'].items():assert sha(p)==h,p
    for p,h in manifest['outputs'].items():assert sha(out/p)==h['sha256'],p
    checks=0; examples={};prefix_checks=0
    for frequency,root in verifier.SOURCES.items():
        l=pd.read_parquet(root/f'leader_watch_{frequency}m.parquet')
        lf,lv=expected(l,'',['board_code','code'],l.path_prefix_complete,l.path_prefix_complete,frequency)
        for benchmark in ('sse','cyb'):
            b=pd.read_parquet(root/f'{benchmark}_board_evidence_{frequency}m.parquet')
            bf,bv=expected(b,'sector_',['board_code'],b.full_members,b.sector_path_prefix_complete,frequency)
            cols=[TIME,'market_interval_return','market_drawdown_from_close_day_high','market_path_prefix_complete']
            m=b[cols].drop_duplicates();m['code']=benchmark
            mf,mv=expected(m,'market_',['code'],m.market_path_prefix_complete,m.market_path_prefix_complete,frequency)
            lookup=pd.Index(mf[TIME])
            bv=np.c_[bv,bv-mv[lookup.get_indexer(bf[TIME])]]
            values=np.c_[lv,lv-mv[lookup.get_indexer(lf[TIME])]]
            old=report['results'][str(frequency)]['benchmarks'][benchmark]
            stats(bv,np.ones(len(bf),bool),old['all_boards'])
            stats(bv,bf.day_return.gt(0).to_numpy(),old['positive_boards'])
            current=lf.current_observed_leader.fillna(False).to_numpy(dtype=bool)
            for mask,name in [(np.ones(len(lf),bool),'all_watched_leaders'),(current,'current_observed_leaders'),(~current,'retained_noncurrent_leaders')]:
                stats(values,mask,old[name])
            masks=dict(rising_but_slowing=(bv[:,0]>0)&(bv[:,1]>0),
                rising_but_lagging=(bv[:,0]>0)&(bv[:,3]<0),strengthening=bv[:,1]<0,giveback=bv[:,2]>0)
            for name,mask in masks.items():assert int(mask.sum())==old['cases'][name]['rows']
            examples[f'{frequency}_{benchmark}']=int(((bf.day_return>0)&masks['rising_but_slowing']).sum())
            checks+=5*len(METRICS)
        # Actual path prefix/perturbation; choose first entity with observable data, never a label.
        p=verifier.canonical(l,'',l.path_prefix_complete,l.path_prefix_complete)
        for _,group in p.groupby(['board_code','code'],sort=True):
            if len(group)<6 or not group.interval_return.notna().any():continue
            n=len(group)//2; full=entity_motion(group,['board_code','code'],frequency)
            changed=group.copy(); changed.iloc[n:,changed.columns.get_loc('interval_return')]=.1
            changed.iloc[n:,changed.columns.get_loc('close_drawdown')]=-.2
            pd.testing.assert_frame_equal(full.iloc[:n],entity_motion(group.iloc[:n],['board_code','code'],frequency))
            pd.testing.assert_frame_equal(full.iloc[:n],entity_motion(changed,['board_code','code'],frequency).iloc[:n])
            assert full.iloc[:n].motion_interval_return.notna().any()
            prefix_checks+=1;break
    assert prefix_checks==2
    answer=dict(status='PASS_INDEPENDENT_NUMPY_SUMMARIES_AND_REAL_PREFIX_NOT_FIT',
        report_sha256=sha(out/'report.json'),input_refs_checked=len(plan['inputs']),
        summary_metric_groups_checked=checks,real_prefix_frequencies=prefix_checks,
        positive_day_and_interval_but_slowing_rows=examples,http_requests=0,fit=False,
        full_86319_materialization=False,formal_training_admitted=False)
    body=json.dumps(answer,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8')
    target=verifier.HERE/'continuous_motion_independent_acceptance_20261018_v1.json'
    if bytes_used()+len(body)+32*1024>TOTAL_CAP:raise ValueError('REGISTERED_BUDGET_STOP')
    with target.open('xb') as stream:stream.write(body)
    return answer


if __name__=='__main__':
    print(json.dumps(run()))
