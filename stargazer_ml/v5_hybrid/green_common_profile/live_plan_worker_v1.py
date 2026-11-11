"""Isolated one-stock live evaluation. No AI, notification, or trading."""
from pathlib import Path
import json,sys

def evaluate(request):
    from gcp_golden_intraday_v1 import verify_plan_sources,prefix_result
    from minute_history_v1 import DiskBudget,load_days,cache_lock,_write
    from tdx_corporate_events_v1 import checked_events
    from live_minute_source_v1 import snapshot
    from live_five_preview_v1 import signal,comparisons
    from golden_rules import digest
    plan=request['plan'];row=plan['rows'][0];code=row['code'];P=int(plan['data_date']);asof=int(request['cutoff'])
    if len(plan['rows'])!=1 or asof//1000000!=int(plan['monitor_date']):raise ValueError('LIVE_PLAN_IDENTITY')
    verify_plan_sources(plan)
    budget=DiskBudget();days=row['entry_rule'].get('history_days') or [P]
    if days!=sorted(set(days)) or not 1<=len(days)<=4 or days[-1]!=P:raise ValueError('LIVE_HISTORY_DAYS')
    history,refs=load_days(code,days,budget)
    previous=sorted((r for r in history if int(r['date'])//1000000==P),key=lambda r:int(r['date']))
    # The unchanged scale/action checks remain mandatory before any live signal.
    folder=Path(request['runtime'])/'minute_context'/'actions'/str(asof//1000000)/code
    actions,receipt=checked_events(code,asof//1000000,P,request['runtime'])
    if abs(float(previous[-1]['close'])-float(row['reference_price']))>.01:raise ValueError('PREVIOUS_CLOSE_SCALE_MISMATCH')
    for key,value in actions:
        if str(value.get('date',str(key).split(':')[-1])).startswith(str(asof//1000000)):raise ValueError('EXECUTION_DAY_CORPORATE_ACTION_UNRESOLVED')
    minutes,provider,live_refs=snapshot(request['runtime'],code,P,asof,float(previous[-1]['close']),history,budget)
    observation=folder/'oversold.json';anchor_key=digest(dict(P=P,history=refs,ratio=row['entry_rule'].get('execution_profile')))
    with cache_lock(folder):
        old=json.loads(observation.read_bytes()) if observation.exists() else {}
        carried=old.get('event') if old.get('anchor_key')==anchor_key else None
        value,_,_=signal(row,P,asof,minutes,request.get('quote'),carried)
        event=value.get('oversold_evidence')
        if event and carried is None:_write(observation,dict(anchor_key=anchor_key,event=event,source=provider,asof=asof))
    value.update(rule_hash=digest(row['entry_rule']),minute_source=provider,
        comparisons=comparisons(minutes,code,P,asof,request.get('quote')),
        raw_evidence=refs+live_refs+[receipt])
    return dict(cutoff=asof,rule_version=plan['rule_version'],plan_hash=digest(plan),rows={code:value})

if __name__=='__main__':
    request=json.load(sys.stdin);project=Path(request['project'])
    sys.path[:0]=[str(project/'stargazer_desktop'),str(project/'stargazer_ml/v5_hybrid/green_common_profile'),str(project/'stargazer_ml/v5_hybrid/risk_r0_v1')]
    import paths
    paths.APP_DIR=str(project/'stargazer_desktop/dist');paths.FROZEN=True
    try:result=evaluate(request)
    except Exception as exc:
        from golden_rules import digest
        row=request['plan']['rows'][0]
        result=dict(cutoff=request['cutoff'],plan_hash=digest(request['plan']),rule_version=request['plan']['rule_version'],
            rows={row['code']:dict(status='DATA_INSUFFICIENT',rule_hash=digest(row['entry_rule']),reason=str(exc)[:180],error_type=type(exc).__name__)})
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
