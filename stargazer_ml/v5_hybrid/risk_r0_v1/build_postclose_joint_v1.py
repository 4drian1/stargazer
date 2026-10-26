"""Real restricted postclose population, exact-day contexts and next-session targets."""
from pathlib import Path
import gzip
import io
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from inventory_ths_stockdb_cache_v1 import HERE, digest
from build_decision_after_labels_v1 import admitted_evidence, SOURCE, LEGACY
from build_joint_candidate_context_v1 import OUT as CONTEXT
from read_decision_after_labels_v2 import OUT as LABELS
from postclose_next_session_core_v1 import stock_days
from accept_left_side_market_v2 import read_stage
from decision_after_labels_v1 import proposed_split
from materialize_local_market_context_v1 import SOURCES
from tdx_index_acceptance_v1 import read_binary
from research_budget_v2 import bytes_used, TOTAL_CAP

OUT=HERE/'postclose_joint_20261017_v1'
CAP,RESERVE=32*1024**2,1024**2
CONTRACT=HERE/'contracts/POSTCLOSE_NEXT_SESSION_RISK_V1.json'
CONTEXT_FIELDS=['context_sector_mean_return_loo_raw','context_sector_breadth_loo',
    'context_sector_mean_coverage_loo','context_sector_common_change_loo_raw',
    'context_leader_board_return_loo_raw','context_current_leader_return_loo_raw',
    'context_captured_leader_mean_return_loo_raw','context_current_leader_return_change_loo_raw',
    'context_captured_leader_common_change_loo_raw','context_current_leader_rsi14_5m_raw',
    'context_current_leader_rsi14_15m_raw']


def load_bound(path, expected, refs):
    body=path.read_bytes()
    if digest(body)!=expected:raise ValueError('FROZEN_INPUT_DRIFT:'+str(path))
    refs[str(path)]=expected
    return body


def run():
    if OUT.exists():raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    audit,acceptance=admitted_evidence()
    refs={str(acceptance):digest(acceptance.read_bytes())}
    for p in [CONTRACT,LEGACY/'evidence_and_gate.json',SOURCE/'manifest.json',CONTEXT/'manifest.json',
              LABELS/'manifest.json',LABELS/'readback_acceptance.json',
              HERE/'golden_anchor_replay_20261017_v2/replay.json',
              HERE/'market_momentum_proposal_20261017_v1/manifest.json',
              HERE/'left_side_market_20261017_v2/manifest.json',
              HERE/'postclose_next_session_core_v1.py',HERE/'test_postclose_next_session_core_v1.py',Path(__file__)]:
        refs[str(p)]=digest(p.read_bytes())
    source=json.loads((SOURCE/'manifest.json').read_bytes())
    context=json.loads((CONTEXT/'manifest.json').read_bytes())
    labels=json.loads((LABELS/'manifest.json').read_bytes())
    accepted_labels=json.loads((LABELS/'readback_acceptance.json').read_bytes())
    if accepted_labels['manifest_sha256']!=refs[str(LABELS/'manifest.json')]:raise ValueError('LABEL_ACCEPTANCE_DRIFT')
    if context['source_manifest_sha256']!=refs[str(SOURCE/'manifest.json')]:raise ValueError('CONTEXT_BINDING_DRIFT')
    sharedbody=load_bound(SOURCE/'shared_market_and_leaders.parquet',source['output_hashes']['shared_market_and_leaders.parquet']['sha256'],refs)
    times=pq.read_table(io.BytesIO(sharedbody),columns=['decision_ts_assumed_end']).column(0).to_numpy()
    days=np.unique(times//1000000)
    selected=[]
    for index,part in enumerate(labels['parts']):
        body=load_bound(LABELS/part['path'],part['sha256'],refs)
        keys=pq.read_table(io.BytesIO(body),columns=['code','cutoff_index']).to_pandas()
        take=keys.code.isin(audit)&(keys.cutoff_index%16==15)
        if not take.any():continue
        sourcepath=SOURCE/part['source']
        raw=load_bound(sourcepath,source['output_hashes'][part['source']]['sha256'],refs)
        base=pq.read_table(io.BytesIO(raw),columns=['code','cutoff_index','stock_close']).to_pandas()
        if not base[['code','cutoff_index']].equals(keys):raise ValueError('CONTEXT_POPULATION_MISMATCH')
        ctxpart=context['parts'][index]
        if ctxpart['source']!=part['source']:raise ValueError('CONTEXT_SOURCE_ORDER')
        ctxbody=load_bound(CONTEXT/ctxpart['context'],ctxpart['sha256'],refs)
        ctx=pq.read_table(io.BytesIO(ctxbody),columns=['code','cutoff_index',*CONTEXT_FIELDS]).to_pandas()
        if not ctx[['code','cutoff_index']].equals(keys):raise ValueError('CONTEXT_KEY_MISMATCH')
        result=base.loc[take].merge(ctx.loc[take],on=['code','cutoff_index'],validate='one_to_one')
        result['day']=times[result.cutoff_index.to_numpy(dtype=int)]//1000000
        selected.append(result.drop(columns='cutoff_index'))
        print(json.dumps(dict(stage='postclose_context',part=index,stock_days=len(result))),flush=True)
    context_frame=pd.concat(selected,ignore_index=True)
    if context_frame.duplicated(['code','day']).any():raise ValueError('DUPLICATE_POSTCLOSE_KEY')
    # The existing accepted label manifest already binds the archived daily payloads.
    bound_daily={Path(p).name[:6]:(Path(p),sha) for p,sha in labels['inputs'].items() if p.endswith('_daily.json.gz')}
    pieces=[];missing=[]
    for code,item in sorted(audit.items()):
        if code in bound_daily:
            path,sha=bound_daily[code]
            records=json.loads(gzip.decompress(load_bound(path,sha,refs)))
        else:records=[];missing.append(code)
        pieces.append(stock_days(code,records,days,item))
    result=pd.concat(pieces,ignore_index=True)
    result=result.merge(context_frame,on=['code','day'],how='left',validate='one_to_one')
    result['context_close_aligned']=(result.stock_close-result.reference_close_raw).abs().le(.01+1e-12)
    result.loc[~result.context_close_aligned,CONTEXT_FIELDS]=np.nan
    result=result.drop(columns='stock_close')
    # Current completed day profiles are valid for postclose only, never same-day early decisions.
    momentum_root=HERE/'market_momentum_proposal_20261017_v1'
    mm=json.loads((momentum_root/'manifest.json').read_bytes())
    body=load_bound(momentum_root/'market_momentum.parquet',mm['output']['sha256'],refs)
    mom=pd.read_parquet(io.BytesIO(body))
    left=read_stage('postclose_daily')
    leftmeta=json.loads((HERE/'left_side_market_20261017_v2/manifest.json').read_bytes())
    for entry in leftmeta['artifacts'].values():
        refs[entry['path']]=entry['sha256']
    market=pd.DataFrame(dict(day=days))
    price_manifest_path=HERE/'local_market_context_20261017_v1/manifest.json'
    pm=json.loads(price_manifest_path.read_bytes());refs[str(price_manifest_path)]=digest(price_manifest_path.read_bytes())
    for symbol,prefix in [('000001.SH','sse'),('399006.SZ','cyb')]:
        data=mom[(mom.symbol==symbol)&(mom.frequency_minutes==5)&(mom.decision_ts_assumed_end%1000000==150000)].copy()
        data['day']=data.decision_ts_assumed_end//1000000
        names=['rsi14','macd_dif_over_close','macd_dea_over_close','macd_hist_over_close',
               'bbi_slope','close_over_bbi_minus_one','price_above_bbi']
        profile=data[['day',*['provisional_daily_'+n for n in names]]].rename(columns={'provisional_daily_'+n:prefix+'_daily_'+n for n in names})
        market=market.merge(profile,on='day',how='left',validate='one_to_one')
        l=left[left.symbol==symbol][['day',*[c for c in left if c.startswith('lag')]]]
        market=market.merge(l.rename(columns={c:prefix+'_'+c for c in l if c!='day'}),on='day',how='left',validate='one_to_one')
        pricepath=SOURCES[symbol]
        # Amount input is bound by the accepted original price component manifest.
        expected=pm['inputs'][str(pricepath)]
        native=pd.DataFrame(read_binary(load_bound(pricepath,expected,refs)))
        native['day']=native.date//1000000
        amount=native.groupby('day').amount.agg(['sum','count'])
        amount.loc[amount['count']!=48,'sum']=np.nan
        amount=amount.reindex(days)['sum']
        baseline=amount.shift(1).rolling(5,min_periods=5).mean()
        market[prefix+'_amount_activity_ratio_prior5d']=(amount/baseline).where(baseline>0).to_numpy()
    replay=json.loads((HERE/'golden_anchor_replay_20261017_v2/replay.json').read_bytes())
    if pm['inputs'][str(HERE/'golden_anchor_replay_20261017_v2/replay.json')]!=refs[str(HERE/'golden_anchor_replay_20261017_v2/replay.json')]:raise ValueError('ANCHOR_BINDING_DRIFT')
    anchors={s['asof_day']:s for s in replay['states']}
    pricepath=HERE/'local_market_context_20261017_v1/market_price_features.parquet'
    source_price=pd.read_parquet(io.BytesIO(load_bound(pricepath,pm['output']['sha256'],refs)))
    closes=source_price[(source_price.symbol=='000001.SH')&(source_price.frequency_minutes==5)&(source_price.decision_ts_assumed_end%1000000==150000)].set_index('day').close
    golden=[]
    for day in days:
        state=anchors.get(int(day));row=dict(day=day)
        if state:
            lo,hi=state['low'],state['high'];close=closes.loc[day]
            row['golden_position']=(close-lo)/(hi-lo)
            for q in [.236,.382,.5,.618,.786]:row['golden_distance_'+str(q).replace('.','_')]=close/(lo+(hi-lo)*q)-1
        golden.append(row)
    market=market.merge(pd.DataFrame(golden),on='day',how='left',validate='one_to_one')
    result=result.merge(market,on='day',how='left',validate='many_to_one')
    result['context_stock_minus_sector_daily_return']=result.stock_return1d-result.context_sector_mean_return_loo_raw
    result['context_stock_minus_leader_daily_return']=result.stock_return1d-result.context_current_leader_return_loo_raw
    # Do not rename same-day context into day-to-day momentum across absent days.
    segments=json.loads((HERE/'contracts/INTRADAY_DECISION_AFTER_LABEL_V1.json').read_bytes())['proposed_segments']
    assignment,purged=proposed_split(result.decision_ts_assumed_end.to_numpy(),result.label_end_time.to_numpy(dtype=float,na_value=np.nan),segments)
    result['proposed_segment']=assignment;result['proposed_purged']=purged
    result['proposed_segment_name']=pd.Series(assignment).map({i:s['name'] for i,s in enumerate(segments)}).fillna('OUT_OF_SCOPE')
    result['reseal_state']='UNKNOWN_NO_HISTORICAL_EVENT_TRACE'
    result['plan_stage']='POSTCLOSE_NEXT_SESSION_CONDITIONAL_EVIDENCE_NOT_ORDER'
    result=result.sort_values(['day','code']).reset_index(drop=True)
    fields=[c for c in result if c.startswith(('stock_','context_','sse_','cyb_','golden_')) and c not in ['context_close_aligned']]
    if np.isinf(result[fields].to_numpy(dtype=float)).any():raise ValueError('INFINITE_FEATURE')
    admitted=result.label_state.eq('CONDITIONAL_NEXT_SESSION_COMPLETE')&~result.proposed_purged
    summary={}
    for segment,g in result[admitted].groupby('proposed_segment_name'):
        summary[segment]=dict(rows=len(g),dates=int(g.day.nunique()),stocks=int(g.code.nunique()),
            nonzero_adverse=int((g.next_session_adverse_excursion>0).sum()),
            target_mean=float(g.next_session_adverse_excursion.mean()),
            context_present_rows=int(g.context_close_aligned.sum()))
    sink=io.BytesIO();result.to_parquet(sink,index=False,compression='zstd')
    payloads={'stock_day_population.parquet':sink.getvalue()}
    manifest=dict(status='POSTCLOSE_NEXT_SESSION_JOINT_PREPARED_NOT_FIT_READY',inputs=refs,
        fields=fields,field_count=len(fields),rows=len(result),evidence_codes=len(audit),
        unbound_daily_codes=missing,conditional_unpurged_summary=summary,
        label_states=result.label_state.value_counts().to_dict(),
        scope='RESTRICTED_DAILY_EVIDENCE_CODES_ARCHIVED_MEMBERSHIP_PARTIAL_CONTEXT',
        calendar_source='EXISTING_FROZEN_MARKET_OBSERVED_CALENDAR_NOT_NEW_EXCHANGE_CERTIFICATION',
        amount_semantics='EXPLICIT_AMOUNT_ACTIVITY_NOT_SHARE_VOLUME',right_confirmation_bars=0,
        training_ready=False,fit_run=False,http_requests=0,
        outputs={n:dict(bytes=len(b),sha256=digest(b)) for n,b in payloads.items()})
    for name,sha in refs.items():
        if digest(Path(name).read_bytes())!=sha:raise ValueError('INPUT_DRIFT_DURING_BUILD')
    payloads['manifest.json']=json.dumps(manifest,ensure_ascii=False,allow_nan=False,indent=2).encode()
    if sum(map(len,payloads.values()))+RESERVE>CAP or bytes_used()+sum(map(len,payloads.values()))+RESERVE>TOTAL_CAP:
        raise ValueError('RESEARCH_BUDGET_STOP')
    OUT.mkdir(exist_ok=False)
    for name,body in payloads.items():
        with (OUT/name).open('xb') as stream:stream.write(body)
    actual=pd.read_parquet(OUT/'stock_day_population.parquet')
    pd.testing.assert_frame_equal(actual,result,check_dtype=False)
    print(json.dumps(dict(status=manifest['status'],rows=len(result),fields=len(fields),summary=summary,
                         bytes=sum(map(len,payloads.values())))))


if __name__=='__main__':run()
