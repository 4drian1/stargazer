"""Native index files -> same-day research market/golden inputs, no HTTP or fit."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from postclose_research_scorer_v1 import HERE,ResearchScorer,encode,sha,strict_json
from postclose_local_source_bridge_v1 import clock_now
from materialize_local_market_context_v1 import SOURCES
from tdx_index_acceptance_v1 import read_binary,grid
from multitimeframe_momentum_v1 import MomentumState
from materialize_multitimeframe_momentum_v1 import PARAMETERS
from left_side_candidates_v1 import candidates
from golden_anchor_rules_v2 import replay
from postclose_daily_job_v1 import background_fields
from postclose_daily_cli_v1 import require_completed_day
from research_budget_v2 import bytes_used,TOTAL_CAP

SEED=HERE/'postclose_market_background_seed_20261018_v1'
OUT=HERE/'postclose_background_inputs_v1'
CAP=2*1024**2
RULE_FILES=['build_postclose_background_v1.py','golden_anchor_rules_v2.py','golden_anchor_rules_v1.py',
            'multitimeframe_momentum_v1.py','materialize_multitimeframe_momentum_v1.py',
            'left_side_candidates_v1.py','tdx_index_acceptance_v1.py','materialize_local_market_context_v1.py']


def source_bytes(path):
    with Path(path).open('rb') as stream:body=stream.read(8*1024**2+1)
    if len(body)>8*1024**2:raise ValueError('INDEX_INPUT_SIZE_LIMIT')
    return body


def aggregate_days(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[row['date']//1000000].append(row)
    result=[]
    for day,bars in sorted(grouped.items()):
        if sorted(r['date'] for r in bars)!=grid(day,5):raise ValueError('INCOMPLETE_OR_DUPLICATE_INDEX_DAY:'+str(day))
        bars=sorted(bars,key=lambda r:r['date'])
        for r in bars:
            values=[r[k] for k in ['open','high','low','close']]
            if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values):
                raise ValueError('INVALID_INDEX_PRICE')
            o,h,l,c=values
            if not l<=min(o,c)<=max(o,c)<=h:raise ValueError('INVALID_INDEX_OHLC')
        amounts=[r['amount'] for r in bars]
        amount=sum(amounts) if all(math.isfinite(v) and v>=0 for v in amounts) else None
        result.append(dict(day=day,open=bars[0]['open'],high=max(r['high'] for r in bars),
                           low=min(r['low'] for r in bars),close=bars[-1]['close'],amount=amount,completed=True))
    return result


def merge_history(seed,current,cutoff):
    original={r['day']:r for r in seed if r['day']<=cutoff}
    current={r['day']:r for r in current if r['day']<=cutoff}
    if not (original.keys()&current.keys()):raise ValueError('NO_VERIFIED_INDEX_OVERLAP')
    for day,row in current.items():
        if day in original and row!=original[day]:raise ValueError('CONFIRMED_INDEX_HISTORY_CHANGED:'+str(day))
        original[day]=row
    return [original[d] for d in sorted(original)]


def compute_background(history,fields):
    day_grid=[r['day'] for r in history['000001.SH']]
    if day_grid!=[r['day'] for r in history['399006.SZ']]:raise ValueError('INDEX_SESSION_GRID_MISMATCH')
    output=pd.DataFrame(dict(day=day_grid))
    for symbol,prefix in [('000001.SH','sse'),('399006.SZ','cyb')]:
        bars=history[symbol];state=MomentumState(**PARAMETERS)
        indicators=pd.DataFrame([state.update(r['close']) for r in bars],dtype=float)
        close=pd.Series([r['close'] for r in bars],dtype=float)
        for name in ['rsi14','bbi_slope','close_over_bbi_minus_one','price_above_bbi']:
            output[prefix+'_daily_'+name]=indicators[name]
        for name in ['macd_dif','macd_hist']:
            output[prefix+'_daily_'+name+'_over_close']=indicators[name]/close
        comparison=indicators[['rsi14','macd_dif','macd_hist','bbi']].copy()
        comparison.insert(0,'close',close)
        left=candidates(comparison)
        for field in left:
            if prefix+'_'+field in fields:output[prefix+'_'+field]=left[field]
        amount=pd.Series([r['amount'] for r in bars],dtype=float)
        baseline=amount.shift(1).rolling(5,min_periods=5).mean()
        output[prefix+'_amount_activity_ratio_prior5d']=(amount/baseline).where(baseline>0)
    anchors={s['asof_day']:s for s in replay(history['000001.SH'])}
    position=[];distances={q:[] for q in [.236,.382,.5,.618,.786]}
    for bar in history['000001.SH']:
        state=anchors.get(bar['day'])
        position.append((bar['close']-state['low'])/(state['high']-state['low']) if state else np.nan)
        for q in distances:
            distances[q].append(bar['close']/(state['low']+(state['high']-state['low'])*q)-1 if state else np.nan)
    output['golden_position']=position
    for q,values in distances.items():output['golden_distance_'+str(q).replace('.','_')]=values
    if set(fields)-set(output):raise ValueError('BACKGROUND_CONSTRUCTION_MISSING_FIELD')
    return output[['day',*fields]]


def write_bundle(root,payloads):
    if root.exists():raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    bodies={name:encode(value) for name,value in payloads.items()}
    total=sum(map(len,bodies.values()))
    if total>CAP or bytes_used()+total>TOTAL_CAP:raise ValueError('BACKGROUND_BUDGET_STOP')
    root.mkdir(parents=True,exist_ok=False)
    for name,body in bodies.items():
        with (root/name).open('xb') as stream:stream.write(body)


def seed_history():
    if SEED.exists():
        manifest=strict_json((SEED/'manifest.json').read_bytes());body=(SEED/'daily_seed.json').read_bytes()
        if sha(body)!=manifest['daily_seed_sha256']:raise ValueError('DAILY_SEED_HASH_DRIFT')
        if set(manifest['rule_refs'])!={str(HERE/name) for name in RULE_FILES}:raise ValueError('SEED_RULE_REGISTRY_DRIFT')
        for p,h in manifest['rule_refs'].items():
            if sha(Path(p).read_bytes())!=h:raise ValueError('SEED_RULE_DRIFT')
        return strict_json(body)
    accepted=strict_json((HERE/'local_market_context_20261017_v1/manifest.json').read_bytes())
    history={};refs={}
    for symbol,path in SOURCES.items():
        body=source_bytes(path)
        if sha(body)!=accepted['inputs'][str(path)]:raise ValueError('INITIAL_INDEX_SOURCE_CHANGED_BEFORE_SEED')
        history[symbol]=aggregate_days(read_binary(body));refs[str(path)]=sha(body)
    rule_refs={str(HERE/name):sha((HERE/name).read_bytes()) for name in RULE_FILES}
    write_bundle(SEED,{'daily_seed.json':history,'manifest.json':dict(source_refs=refs,rule_refs=rule_refs,
        daily_seed_sha256=sha(encode(history)),scope='IMMUTABLE_ACCEPTED_DAILY_SEED_NOT_ARRIVAL_EVIDENCE')})
    return history


def background_for_day(day):
    scorer=ResearchScorer();fields=background_fields(scorer.fields);seed=seed_history()
    history={};refs={str(SEED/'daily_seed.json'):sha((SEED/'daily_seed.json').read_bytes())}
    observations=[]
    for symbol,path in SOURCES.items():
        started=clock_now();body=source_bytes(path);refs[str(path)]=sha(body)
        rows=[r for r in read_binary(body) if r['date']//1000000<=day]
        current=aggregate_days(rows)
        if day not in [r['day'] for r in current]:raise ValueError('TARGET_DAY_NOT_IN_SOURCE_NO_FALLBACK:'+symbol)
        history[symbol]=merge_history(seed[symbol],current,day)
        observations.append(dict(symbol=symbol,path=str(path),sha256=sha(body),
            observed_started_at=started,observed_finished_at=clock_now(),original_available_at=None))
    frame=compute_background(history,fields);row=frame[frame.day.eq(day)].iloc[0]
    features={f:None if pd.isna(row[f]) else float(row[f]) for f in fields}
    if any(features[f] is None for f in ['sse_daily_rsi14','cyb_daily_rsi14','golden_position']):
        raise ValueError('BACKGROUND_REQUIRED_GROUP_NOT_READY')
    for name in RULE_FILES:
        path=HERE/name;refs[str(path)]=sha(path.read_bytes())
    source_refs=[dict(path=p,sha256=h) for p,h in refs.items()]
    background=dict(kind='POSTCLOSE_MARKET_GOLDEN',day=day,features=features,source_refs=source_refs,
        constructed_at=clock_now(),original_available_at=None,provenance_verified=False,
        volume_semantics='AMOUNT_ACTIVITY_PROXY_NOT_VERIFIED_SHARE_VOLUME')
    calendar=dict(kind='TRADING_CALENDAR',days=frame.day.astype(int).tolist(),source_refs=source_refs,
        verification='OBSERVED_INDEX_SESSIONS_NOT_INDEPENDENT_EXCHANGE_CERTIFICATION')
    return background,calendar,frame,observations


def build(day):
    require_completed_day(day,clock_now())
    root=OUT/str(day)
    if root.exists():raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    background,calendar,_,observations=background_for_day(day)
    manifest=dict(status='COMPLETE_RESEARCH_BACKGROUND_NOT_PIT_CERTIFICATION',day=day,observations=observations,
        background_sha256=sha(encode(background)),calendar_sha256=sha(encode(calendar)),
        fields=len(background['features']),fit_run=False,http_requests=0,independent_validation_passed=False)
    for item in background['source_refs']:
        if sha(Path(item['path']).read_bytes())!=item['sha256']:raise ValueError('INPUT_CHANGED_DURING_BUILD')
    write_bundle(root,{'market_golden.json':background,'calendar.json':calendar,'manifest.json':manifest})
    if sha((root/'market_golden.json').read_bytes())!=manifest['background_sha256'] or sha((root/'calendar.json').read_bytes())!=manifest['calendar_sha256']:
        raise ValueError('BACKGROUND_WRITE_READBACK_FAILED')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--day',type=int,required=True)
    args=parser.parse_args()
    try:print(json.dumps(build(args.day),ensure_ascii=False,allow_nan=False));return 0
    except (ValueError,OSError,KeyError) as exc:
        print(json.dumps(dict(status='REJECTED',error=str(exc)),ensure_ascii=False));return 2


if __name__=='__main__':raise SystemExit(main())
