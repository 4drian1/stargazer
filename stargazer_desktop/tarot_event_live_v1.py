"""First live signal review; later close changes are local audit only."""
from datetime import datetime
import json,threading
import postclose_risk_runtime as runtime
from golden_rules import digest
_lock=threading.Lock();_thread=None

def evaluate(plan,cfg,now):
    global _thread
    from live_plan_lane_v1 import active
    if not active(now) or int(plan.get('monitor_date',0))!=int(now.strftime('%Y%m%d')):return {}
    folder=runtime.directory()/'tarot_events_live'
    key=digest(plan);target=folder/(now.strftime('%Y%m%d%H%M')+'_'+key+'.json')
    with _lock:
        if not (_thread and _thread.is_alive()) and not target.exists():
            _thread=threading.Thread(target=work,args=(plan,cfg,now,target),daemon=True,name='live-plan-events');_thread.start()
        try:
            result=json.loads((folder/'latest.json').read_bytes())
            if result.get('plan_hash')==key:return result
        except (OSError,ValueError):pass
    return {}

def work(plan,cfg,requested,target):
    from tarot_monitor_v2 import save,advise,enabled,fresh
    from live_plan_lane_v1 import evaluate as check,risk,notice_key
    from tools.market import quotes
    try:prices=quotes([r['code'] for r in plan['rows']])
    except Exception:prices={}
    now=datetime.now();cutoff=int(now.strftime('%Y%m%d%H%M%S'))
    result=dict(status='EVENT_CHECKING',cutoff=cutoff,plan_hash=digest(plan),plan_id=plan.get('assistant_plan_id'),rule_version=plan['rule_version'],rows={},provisional=True)
    publish=threading.Lock()
    def update(code,signal):
        with publish:
            result['rows'][code]=dict(signal,advice=dict(status='WAIT_SIGNAL'))
            save(target,result);save(target.parent/'latest.json',result)
        if signal.get('status')!='RULE_TRIGGERED':return
        if not fresh(cutoff) or not enabled(plan):return
        record=risk(plan,code,cutoff,signal,now)
        advice=advise(cfg,plan,code,cutoff,record,signal=signal)
        if advice.get('status')=='ADVICE_READY':advice=dict(advice,cutoff=cutoff,event_id='first-live-signal:'+notice_key(plan,code,signal),provisional=True)
        with publish:
            result['rows'][code]['advice']=advice
            save(target,result);save(target.parent/'latest.json',result)
    try:
        check(plan,now,prices,update);result['status']='EVENTS_READY'
    except Exception as exc:result.update(status='EVENT_FAILED',reason=str(exc)[:160])
    save(target,result);save(target.parent/'latest.json',result)
