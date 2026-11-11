"""Explicit offline EXE acceptance. No live transports or user-data writes."""
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


def run(receipt_path):
    import joint_risk_bridge
    import pentacles_monitor as monitor
    import tarot_monitor_v2 as risk
    import touch_alert_lane_v1 as touch
    import services
    import tarot_event_v1
    import risk_outlook_v1
    import paths
    receipt=Path(receipt_path).resolve()
    if receipt.exists():raise FileExistsError(receipt)
    root=receipt.parent/'offline_realtime_state'
    root.mkdir(parents=True,exist_ok=True)
    plan=dict(plan_ready=True,assistant_plan_id='OFFLINE_ACCEPTANCE',data_date=20261103,
        monitor_date=20261104,rule_version=risk.VERSION,rows=[dict(code='300652',name='fixture',
        entry_rule=dict(observation_bands=[dict(lower=24.,upper=24.3)]))])
    stamp=20261104093200
    quotes={'300652':dict(price=24.2,time=stamp)}
    state=root/'touch_alerts.json'
    restarted=state.exists()
    notices=touch.detect(plan,quotes,state,now_stamp=stamp)
    sent=[]
    with patch('agent.brain.load_config',return_value=dict(push=dict(enabled=True))), \
            patch.object(monitor,'_plan_current',return_value=True):
        for event in notices:
            touch.reserve_send(event,state)
            ok,cancelled=monitor._deliver_one({},'fixture','',state,event,plan,
                quote_fn=lambda codes:quotes,
                send_fn=lambda cfg,title,body:sent.append(body) or (True,''),now_stamp=stamp)
            assert ok and cancelled is None
    if restarted:
        assert notices==[] and sent==[],'RESTART_DUPLICATE'
    else:
        assert len(notices)==1 and len(sent)==1,'TOUCH_NOT_DELIVERED'
        assert '24.20' in sent[0],'OLD_QUOTE_IN_MESSAGE'
    entries=json.loads(state.read_bytes())['notified']
    assert len(entries)==1,'MISSING_PERSISTED_DELIVERY'
    assert all(v.get('delivery')=='SENT' for v in entries.values())
    assert touch.detect(plan,quotes,state,now_stamp=stamp)==[]
    # Exercise evaluate -> worker -> independent retry ledger, not just classifier.
    runtime=root/('risk_'+receipt.stem)
    folder=runtime/'tarot_v2'
    folder.mkdir(parents=True,exist_ok=True)
    old=dict(status='EVIDENCE_FAILED',flow_version='v0',plan_id=plan['assistant_plan_id'],
        plan_hash=risk.digest(plan),reason='REALTIME_SOURCE_UNAVAILABLE')
    (folder/'failed_20261104.json').write_text(json.dumps(old),encoding='utf8')
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return cls(2026,11,4,11,1,tzinfo=tz)
    class Thread:
        def __init__(self,target,args,**kwargs):self.target,self.args=target,args
        def start(self):self.target(*self.args)
        def is_alive(self):return False
    class Response:
        returncode=0
        stdout=json.dumps(dict(status='EVIDENCE_INSUFFICIENT',reason='REALTIME_SOURCE_UNAVAILABLE'))
    with patch.object(risk.runtime,'directory',return_value=runtime), \
            patch.object(risk,'datetime',Clock), \
            patch.object(risk,'RETRY_COOLDOWN_SECONDS',0), \
            patch.object(risk,'_thread',None), \
            patch.object(risk,'index_prefix_ready',return_value=True), \
            patch.object(risk,'intraday_slot_wait',return_value=False), \
            patch.object(services,'wl_load',return_value={'positions':[]}), \
            patch.object(tarot_event_v1,'evaluate',return_value={}), \
            patch.object(risk_outlook_v1,'queue'), \
            patch.object(joint_risk_bridge,'_worker_paths',return_value=(Path('python'),Path('worker.py'))), \
            patch.object(risk.subprocess,'run',return_value=Response()) as worker, \
            patch.object(risk.threading,'Thread',Thread):
        assert risk._flow_version()!='UNKNOWN','MISSING_RELEASE_ID'
        assert risk.failure_stale(old,plan),'OLD_FAILURE_BLOCKED'
        for expected in (1,2,3):
            risk.evaluate(plan,{},Clock.now())
            assert risk.retry_attempts(plan)==expected,'RETRY_COUNTER_RESET'
        blocked=risk.evaluate(plan,{},Clock.now())
        assert worker.call_count==3,'RETRY_LIMIT_BROKEN'
        assert blocked['status']=='RISK_BATCH_FAILED'
        assert len(list(folder.glob('failed_20261104_a*.json')))>=2
    result=dict(status='PASS',mode='OFFLINE_FIXTURE_FAKE_TRANSPORTS',frozen=paths.FROZEN,
        release=risk._flow_version(),restart_dedup=restarted,touch_events=len(notices),
        persisted_sent_events=len(entries),
        simulated_sends=len(sent),retry_counts=[1,2,3],fourth_attempt_blocked=True,
        http=0,paid=0,real_push=0,user_data_changed=False)
    with receipt.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
