"""Native index background + bounded manual research job. No scheduler or fitting."""
import argparse
import json
from pathlib import Path

import pandas as pd

import build_postclose_background_v1 as bg
from postclose_research_scorer_v1 import HERE,ResearchScorer,strict_json,sha,encode
from postclose_daily_job_v1 import background_fields,plan,run,read_input
from postclose_daily_cli_v1 import require_completed_day
from postclose_local_source_bridge_v1 import clock_now

INPUTS=HERE/'postclose_market_daily_inputs_v1'


def checkpoint_history(day):
    history={s:[dict(r) for r in rows if r['day']<=day] for s,rows in bg.seed_history().items()}
    refs={str(bg.SEED/'daily_seed.json'):sha((bg.SEED/'daily_seed.json').read_bytes())}
    if INPUTS.exists():
        for root in sorted(INPUTS.iterdir()):
            if not root.is_dir() or not root.name.isdigit() or int(root.name)>day:continue
            manifest=strict_json((root/'manifest.json').read_bytes())
            if manifest['status']!='COMPLETE_BACKGROUND_CHECKPOINT':raise ValueError('INCOMPLETE_BACKGROUND_CHECKPOINT')
            body=(root/'daily_delta.json').read_bytes()
            if sha(body)!=manifest['daily_delta_sha256']:raise ValueError('BACKGROUND_CHECKPOINT_HASH_DRIFT')
            delta=strict_json(body);refs[str(root/'daily_delta.json')]=sha(body)
            for symbol,rows in delta.items():
                existing={r['day']:r for r in history[symbol]}
                for row in rows:
                    if row['day']>int(root.name):raise ValueError('FUTURE_DATA_IN_CHECKPOINT')
                    if row['day'] in existing and existing[row['day']]!=row:raise ValueError('CHECKPOINT_HISTORY_CONFLICT')
                    existing[row['day']]=row
                history[symbol]=[existing[d] for d in sorted(existing)]
    return history,refs


def prepare(day,write=False):
    if write:require_completed_day(day,clock_now())
    root=INPUTS/str(day)
    if write and root.exists():raise ValueError('BACKGROUND_EXISTS_NO_OVERWRITE')
    scorer=ResearchScorer();fields=background_fields(scorer.fields);history,refs=checkpoint_history(day)
    previous_days={s:{r['day'] for r in rows} for s,rows in history.items()}
    observed=[]
    for symbol,path in bg.SOURCES.items():
        started=clock_now();body=bg.source_bytes(path);refs[str(path)]=sha(body)
        rows=bg.aggregate_days([r for r in bg.read_binary(body) if r['date']//1000000<=day])
        if day not in {r['day'] for r in rows}:raise ValueError('TARGET_DAY_NOT_IN_SOURCE_NO_FALLBACK:'+symbol)
        history[symbol]=bg.merge_history(history[symbol],rows,day)
        observed.append(dict(symbol=symbol,path=str(path),sha256=sha(body),
                             observed_started_at=started,observed_finished_at=clock_now(),original_available_at=None))
    frame=bg.compute_background(history,fields);row=frame[frame.day.eq(day)].iloc[0]
    features={f:None if pd.isna(row[f]) else float(row[f]) for f in fields}
    if any(features[f] is None for f in ['sse_daily_rsi14','cyb_daily_rsi14','golden_position']):raise ValueError('BACKGROUND_GROUP_NOT_READY')
    for name in [*bg.RULE_FILES,'postclose_daily_update_v1.py']:
        p=HERE/name;refs[str(p)]=sha(p.read_bytes())
    source_refs=[dict(path=p,sha256=h) for p,h in refs.items()]
    background=dict(kind='POSTCLOSE_MARKET_GOLDEN',day=day,features=features,source_refs=source_refs,
                    constructed_at=clock_now(),original_available_at=None,provenance_verified=False,
                    amount_semantics='ACTIVITY_PROXY_NOT_VERIFIED_SHARE_VOLUME')
    calendar=dict(kind='TRADING_CALENDAR',days=frame.day.astype(int).tolist(),source_refs=source_refs,
                  verification='OBSERVED_INDEX_SESSIONS_NOT_INDEPENDENT_EXCHANGE_CERTIFICATION')
    result=dict(status='BACKGROUND_READY_RESEARCH_ONLY',day=day,fields=len(fields),
                background=str(root/'market_golden.json'),calendar=str(root/'calendar.json'),
                source_observations=observed,http_requests=0,independent_validation_passed=False)
    if write:
        delta={s:[r for r in rows if r['day'] not in previous_days[s]] for s,rows in history.items()}
        manifest=dict(status='COMPLETE_BACKGROUND_CHECKPOINT',day=day,observations=observed,
            daily_delta_sha256=sha(encode(delta)),background_sha256=sha(encode(background)),
            calendar_sha256=sha(encode(calendar)),new_days={s:len(rows) for s,rows in delta.items()},
            provenance_verified=False,fit_run=False)
        if any(sha(Path(p).read_bytes())!=h for p,h in refs.items()):raise ValueError('BACKGROUND_INPUT_CHANGED')
        bg.write_bundle(root,{'market_golden.json':background,'calendar.json':calendar,
                              'daily_delta.json':delta,'manifest.json':manifest})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--day',type=int,required=True)
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--prepare-background',action='store_true');actions.add_argument('--run',action='store_true')
    args=parser.parse_args()
    try:
        if args.prepare_background:
            result=prepare(args.day,write=True)
        elif args.run:
            require_completed_day(args.day,clock_now())
            cached=plan(args.day)[0]
            if cached.get('status')=='CACHED_READY_NO_FETCH':result=run(args.day)
            else:
                root=INPUTS/str(args.day)
                if not root.exists():prepare(args.day,write=True)
                read_input(root/'market_golden.json','POSTCLOSE_MARKET_GOLDEN')
                read_input(root/'calendar.json','TRADING_CALENDAR')
                result=run(args.day,root/'calendar.json',root/'market_golden.json')
        else:
            cached=plan(args.day)[0]
            result=cached if cached.get('status')=='CACHED_READY_NO_FETCH' else prepare(args.day)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
        return 2 if result.get('ready') is False else 0
    except (ValueError,OSError,KeyError) as exc:
        print(json.dumps(dict(status='REJECTED_NO_FETCH',error=str(exc)),ensure_ascii=False));return 2


if __name__=='__main__':raise SystemExit(main())
