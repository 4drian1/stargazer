"""Daily-only leader overlay and fixed complete-family research cohort."""
from pathlib import Path
import io
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from inventory_ths_stockdb_cache_v1 import HERE,digest
from build_decision_after_labels_v1 import SOURCE
from build_joint_risk_features_v1 import wilder_rsi
from research_budget_v2 import bytes_used,TOTAL_CAP

BASE=HERE/'postclose_joint_20261017_v1'
OUT=HERE/'postclose_research_20261017_v2'
POLICY=HERE/'postclose_research_policy_v1.json'
CAP,RESERVE=16*1024**2,1024**2
INTRADAY_ONLY=['context_sector_common_change_loo_raw','context_current_leader_return_change_loo_raw',
              'context_captured_leader_common_change_loo_raw','context_current_leader_rsi14_5m_raw',
              'context_current_leader_rsi14_15m_raw']


def complete_daily_close(frame,day_count):
    if frame.duplicated(['code','cutoff_index']).any():raise ValueError('DUPLICATE_SOURCE_KEY')
    data=frame.copy();data['d']=data.cutoff_index//16
    if (data.d<0).any() or (data.d>=day_count).any():raise ValueError('INVALID_DAY_INDEX')
    result={}
    for code,g in data.groupby('code',sort=False):
        counts=g.groupby('d').stock_close.count()
        last=g[g.cutoff_index%16==15]
        selected=last[last.d.map(counts).eq(16)]
        closes=np.full(day_count,np.nan)
        closes[selected.d.to_numpy(dtype=int)]=selected.stock_close.to_numpy(dtype=float)
        result[code]=closes
    return result


def family_masks(frame):
    return dict(stock_daily=frame.stock_daily_rsi14.notna(),
        market_daily=frame[['sse_daily_rsi14','cyb_daily_rsi14']].notna().all(axis=1),
        sector_daily_observation=frame.context_sector_mean_return_loo_raw.notna(),
        leader_daily_observation=frame[['context_current_leader_daily_rsi14_raw','context_captured_leader_daily_rsi14_raw']].notna().any(axis=1),
        golden_position=frame.golden_position.notna())


def run():
    if OUT.exists():raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    spec=json.loads(POLICY.read_bytes())
    refs={str(POLICY):digest(POLICY.read_bytes())}
    for path in [BASE/'manifest.json',SOURCE/'manifest.json',Path(__file__),HERE/'test_postclose_research_v2.py',
                 HERE/'build_joint_risk_features_v1.py']:
        refs[str(path)]=digest(path.read_bytes())
    parent=json.loads((BASE/'manifest.json').read_bytes());source=json.loads((SOURCE/'manifest.json').read_bytes())
    for path,sha in parent['inputs'].items():
        if digest(Path(path).read_bytes())!=sha:raise ValueError('PARENT_INPUT_DRIFT')
    body=(BASE/'stock_day_population.parquet').read_bytes()
    if digest(body)!=parent['outputs']['stock_day_population.parquet']['sha256']:raise ValueError('PARENT_OUTPUT_DRIFT')
    refs[str(BASE/'stock_day_population.parquet')]=digest(body)
    population=pd.read_parquet(io.BytesIO(body));days=np.sort(population.day.unique());n=len(days)
    daily={}
    for i,part in enumerate(source['joint_parts']):
        path=SOURCE/part['path'];body=path.read_bytes()
        if digest(body)!=source['output_hashes'][part['path']]['sha256']:raise ValueError('SOURCE_PART_DRIFT')
        refs[str(path)]=digest(body)
        frame=pq.read_table(io.BytesIO(body),columns=['code','cutoff_index','stock_close']).to_pandas()
        current=complete_daily_close(frame,n)
        if set(current)&set(daily):raise ValueError('CODE_SPANS_PARTS')
        daily.update(current)
        if i%15==0:print(json.dumps(dict(stage='daily_leader_history',parts=i+1,codes=len(daily))),flush=True)
    rsis={code:wilder_rsi(closes) for code,closes in daily.items()}
    sharedpath=SOURCE/'shared_market_and_leaders.parquet';body=sharedpath.read_bytes()
    if digest(body)!=source['output_hashes'][sharedpath.name]['sha256']:raise ValueError('LEADER_TRACE_DRIFT')
    refs[str(sharedpath)]=digest(body)
    leaders=pq.read_table(io.BytesIO(body),columns=['cutoff_index','leader_observed_stock']).to_pandas()
    captures={};current={}
    for d,g in leaders.groupby(leaders.cutoff_index//16):
        captures[d]={v for v in g.leader_observed_stock if isinstance(v,str) and v}
        last=g[g.cutoff_index%16==15]
        current[d]=last.leader_observed_stock.iloc[0] if len(last) else None
    rows=[]
    for r in population[['code','day']].itertuples(index=False):
        d=int(np.searchsorted(days,r.day));leader=current.get(d)
        cur=rsis[leader][d] if leader in rsis and leader!=r.code else np.nan
        selected=captures.get(d,set())-{r.code}
        values=[rsis[c][d] for c in selected if c in rsis and np.isfinite(rsis[c][d])]
        rows.append(dict(code=r.code,day=r.day,context_current_leader_daily_rsi14_raw=cur,
            context_captured_leader_daily_rsi14_raw=float(np.mean(values)) if values else np.nan,
            leader_daily_observed_count=len(values),leader_daily_reference_count=len(selected)))
    overlay=pd.DataFrame(rows)
    joined=population.merge(overlay,on=['code','day'],validate='one_to_one',how='left',sort=False)
    pd.testing.assert_frame_equal(joined[list(population)],population)
    masks=family_masks(joined)
    complete=np.logical_and.reduce(list(masks.values()))
    eligible=joined.label_state.eq('CONDITIONAL_NEXT_SESSION_COMPLETE')&~joined.proposed_purged&complete
    fields=[f for f in parent['fields'] if f not in INTRADAY_ONLY and 'macd_dea' not in f]
    fields+=['context_current_leader_daily_rsi14_raw','context_captured_leader_daily_rsi14_raw']
    joined['research_cohort_eligible']=eligible
    train=joined[eligible&joined.proposed_segment_name.eq('train')]
    constants=[f for f in fields if train[f].nunique(dropna=True)<=1]
    final_fields=[f for f in fields if f not in constants]
    if any(train[f].isna().all() for f in final_fields):raise ValueError('ALL_NULL_TRAIN_FEATURE')
    summary={};gates={}
    for segment,threshold in spec['sample_gates'].items():
        g=joined[eligible&joined.proposed_segment_name.eq(segment)]
        entry=dict(rows=len(g),dates=int(g.day.nunique()),stocks=int(g.code.nunique()),
            target_nonzero=int((g.next_session_adverse_excursion>0).sum()),target_mean=float(g.next_session_adverse_excursion.mean()) if len(g) else None,
            target_max=float(g.next_session_adverse_excursion.max()) if len(g) else None)
        summary[segment]=entry;gates[segment]={k:entry[k]>=v for k,v in threshold.items()}
    manifest=dict(status='POSTCLOSE_DAILY_ONLY_COHORT_PREPARED_NOT_FIT_AUTHORIZED',inputs=refs,
        rows=len(joined),fields=final_fields,field_count=len(final_fields),constant_train_fields_excluded=constants,
        intraday_fields_excluded=INTRADAY_ONLY,summary=summary,sample_gates=gates,
        missing_family_rows={k:int((~v).sum()) for k,v in masks.items()},
        stock_day_denominator_preserved=True,source_daily_leader_codes=len(rsis),
        historical_leader_prices='RAW_CONTEXT_NOT_CORPORATE_ACTION_CERTIFIED',
        exact_label_and_parent_columns_unchanged=True,fit_run=False,training_ready=False,http_requests=0)
    config=dict(spec,fields=final_fields,data_manifest=str(OUT/'manifest.json'),
        data_path=str(OUT/'stock_day_population.parquet'),status='DRAFT_READY_FOR_SCOPE_REVIEW',fit_authorized=False)
    buffer=io.BytesIO();joined.to_parquet(buffer,index=False,compression='zstd')
    bodies={'stock_day_population.parquet':buffer.getvalue(),
            'fit_config_draft.json':json.dumps(config,ensure_ascii=False,allow_nan=False,indent=2).encode()}
    manifest['outputs']={k:dict(bytes=len(v),sha256=digest(v)) for k,v in bodies.items()}
    bodies['manifest.json']=json.dumps(manifest,ensure_ascii=False,allow_nan=False,indent=2).encode()
    if sum(map(len,bodies.values()))+RESERVE>CAP or bytes_used()+sum(map(len,bodies.values()))+RESERVE>TOTAL_CAP:raise ValueError('RESEARCH_BUDGET_STOP')
    if any(digest(Path(p).read_bytes())!=sha for p,sha in refs.items()):raise ValueError('INPUT_DRIFT_DURING_PREPARATION')
    OUT.mkdir(exist_ok=False)
    for name,body in bodies.items():
        with (OUT/name).open('xb') as stream:stream.write(body)
    reread=pd.read_parquet(OUT/'stock_day_population.parquet')
    numeric=[c for c in joined if pd.api.types.is_numeric_dtype(joined[c])]
    pd.testing.assert_frame_equal(reread[numeric],joined[numeric])
    for c in joined.columns.difference(numeric):
        pd.testing.assert_series_equal(reread[c].astype(object),joined[c].astype(object))
    print(json.dumps(dict(status=manifest['status'],rows=len(joined),fields=len(final_fields),summary=summary,gates=gates,
                         bytes=sum(map(len,bodies.values())))))


if __name__=='__main__':run()
