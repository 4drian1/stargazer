"""Five-minute event lane, independent of slow 15m Pyxis prewarming."""
from datetime import datetime
import json
import threading

import postclose_risk_runtime as runtime
from golden_rules import digest
from golden_monitor import MINUTE_SOURCE_CONTRACT

_lock=threading.Lock()
_thread=None


def cutoff5(now):
    if now.weekday()>=5:return None
    minute=now.hour*60+now.minute
    grid=list(range(575,691,5))+list(range(785,901,5))
    past=[m for m in grid if m<=minute]
    if not past:return None
    m=past[-1]
    return int(now.strftime('%Y%m%d'))*1000000+(m//60)*10000+m%60*100


def read_risk(plan):
    try:
        value=json.loads((runtime.directory()/'tarot_v2/latest.json').read_bytes())
        return value if value.get('plan_id')==plan.get('assistant_plan_id') and value.get('plan_hash')==digest(plan) else {}
    except (OSError,ValueError):return {}


def work(plan,cfg,now,source,target):
    from tarot_monitor_v2 import save,advise,enabled,fresh
    import golden_monitor
    from plan_evidence import archived_price_overlay
    from trade_facts_v1 import CONTRACT
    cutoff=cutoff5(now)
    result=dict(status='EVENT_CHECKING',cutoff=cutoff,plan_hash=digest(plan),plan_id=plan.get('assistant_plan_id'),fact_contract=CONTRACT,minute_source_contract=MINUTE_SOURCE_CONTRACT,rows={})
    try:
        signals=golden_monitor.evaluate(plan,now,five_minute=True)
        if signals.get('cutoff')!=cutoff:raise ValueError('FIVE_MINUTE_SIGNAL_CUTOFF')
        result['rows']={c:dict(s,advice=dict(status='WAIT_RISK')) for c,s in signals['rows'].items()}
        save(target,result)  # publish a confirmed price event before waiting on model/LLM
        save(target.parent/'latest.json',result)
        records=source.get('risk',{}).get('records',{})
        fired=[c for c,s in signals['rows'].items() if s.get('status')=='RULE_TRIGGERED']
        prices=archived_price_overlay(cutoff) if fired else {}
        for code in fired:
            signal=signals['rows'][code];record=dict(records.get(code) or {})
            context=dict(record.get('context') or {})
            if prices:context['market_price_evidence']=prices
            closes=signal.get('last_two_closes') or []
            if closes:
                context.update(amp_reference_price=context.get('stock_close'),stock_close=closes[-1],stock_price_asof=cutoff)
            record.update(context=context,mixed_cutoff_observation=cutoff)
            if not fresh(cutoff) or not enabled(plan):
                result['rows'][code]['advice']=dict(status='ADVICE_EXPIRED',reason='价格事件已过时或计划变化；不发送建议')
            else:result['rows'][code]['advice']=advise(cfg,plan,code,cutoff,record,signal=signal)
        result['status']='EVENTS_READY'
    except Exception as exc:
        reason=str(exc) if isinstance(exc,ValueError) else type(exc).__name__
        result.update(status='EVENT_FAILED',reason=reason,automatic_retry=False)
    save(target,result);save(target.parent/'latest.json',result)


def evaluate(plan,cfg,now):
    from tarot_event_live_v1 import evaluate as evaluate_live
    return evaluate_live(plan,cfg,now)


def evaluate_completed(plan,cfg,now):
    global _thread
    cutoff=cutoff5(now)
    if cutoff is None or int(plan.get('monitor_date',0))!=cutoff//1000000:return {}
    source=read_risk(plan)
    from trade_facts_v1 import CONTRACT
    target=runtime.directory()/'tarot_events'/f'{cutoff}_{digest(plan)}_{source.get("cutoff",0)}_{CONTRACT}_{MINUTE_SOURCE_CONTRACT}.json'
    with _lock:
        if target.exists():return json.loads(target.read_bytes())
        if not (_thread and _thread.is_alive()):
            from tarot_monitor_v2 import save
            save(target,dict(status='EVENT_QUEUED',cutoff=cutoff,fact_contract=CONTRACT,minute_source_contract=MINUTE_SOURCE_CONTRACT,rows={}))
            _thread=threading.Thread(target=work,args=(plan,cfg,now,source,target),daemon=True,name='tarot-5m-events')
            _thread.start()
        try:
            value=json.loads((target.parent/'latest.json').read_bytes())
            if value.get('cutoff')==cutoff and value.get('plan_hash')==digest(plan) and value.get('fact_contract')==CONTRACT and value.get('minute_source_contract')==MINUTE_SOURCE_CONTRACT:return value
        except (OSError,ValueError):pass
    return {}
