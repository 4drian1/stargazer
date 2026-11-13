"""Selection completion prepares a draft and history; only humans enable it."""
from datetime import datetime
from copy import deepcopy
import json

def prepare_holdings(snapshot,cfg=None):
    """Positions have a separate preparation roster; watch rows are not holdings."""
    import hashlib
    from pathlib import Path
    from minute_history_v1 import schedule
    from services import active_positions
    if cfg is None:
        from agent.brain import load_config
        cfg=load_config()
    positions=active_positions(cfg)
    if not positions:return
    refs=[r for r in snapshot.get('inputs',[]) if Path(r.get('path','')).name=='calendar.json']
    if len(refs)!=1:raise ValueError('HOLDINGS_CALENDAR_REFERENCE')
    raw=Path(refs[0]['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=refs[0]['sha256']:raise ValueError('HOLDINGS_CALENDAR_SHA')
    days=sorted({int(d) for d in json.loads(raw)})
    P,E=int(snapshot['data_date']),int(snapshot['monitor_date'])
    if P not in days or days.index(P)+1>=len(days) or days[days.index(P)+1]!=E:
        raise ValueError('HOLDINGS_P_E_MISMATCH')
    history=[d for d in days if d<=P][-4:]
    rows=[dict(code=str(r['code']).zfill(6),entry_rule=dict(history_days=history)) for r in positions]
    schedule(dict(data_date=P,monitor_date=E,rows=rows,prepare_role='HOLDINGS_ONLY'))

def after_selection(snapshot):
    import assistant_workflow as workflow
    import plan_evidence
    from minute_history_v1 import schedule
    from overview_index_maps_v1 import request_postclose
    try:
        request_postclose(snapshot['data_date'])
    except (OSError, ValueError, KeyError):
        import logging
        logging.getLogger(__name__).exception('Index overview map request could not be saved')
    saved_previous=workflow.latest('plan') or {}
    previous=workflow.plan_preferences(saved_previous)
    # Preserve explicit preferences, not stale tickers or a former confirmation.
    boards=deepcopy(previous.get('boards') or [])
    if not boards and not previous.get('board_preference_id'):
        import pentacles_demo
        try:boards=json.loads(pentacles_demo.snapshot_path().with_name('assistant_plan.json').read_bytes()).get('boards',[])
        except (OSError,ValueError):pass
    schedule(snapshot)
    prepare_holdings(snapshot)
    # W2c：准备清单落盘（逐项状态+版本绑定+未完成项单列）。
    try:
        import postclose_prepared_v1 as prepared
        roster = prepared.roster_from_plan(dict(rows=snapshot.get('rows') or []))
        P = int(snapshot['data_date']); E = int(snapshot.get('monitor_date') or snapshot.get('observation_date') or P)
        targets = prepared.schedule_roster(roster, P, E)
        prepared.refresh_manifest(P, E, roster, targets)
    except Exception:
        pass  # 清单登记失败不阻断选股后的既有准备链
    if not snapshot.get('plan_ready') or workflow.plan_expired(snapshot):return
    if (previous.get('snapshot_hash')==workflow.digest(snapshot) and
            previous.get('boards')==saved_previous.get('boards') and
            previous.get('board_preference_id')==saved_previous.get('board_preference_id')):return
    requested=previous.get('requested_count')
    rows=snapshot.get('rows') or []
    selected=[r['code'] for r in rows[:requested] if r.get('entry_rule')] if type(requested) is int else [r['code'] for r in rows if r.get('entry_rule')]
    if not selected:return
    evidence=plan_evidence.for_snapshot(snapshot)
    reply='已按本次选股与既有数量、板块偏好生成次日草案，历史分钟与强势池后台准备。请查看并人工确认；未确认不启用本草案提醒。'
    draft=workflow.store('plan',dict(origin='AUTOMATIC_PREPARATION',snapshot_hash=workflow.digest(snapshot),snapshot=snapshot,
        dialogue=[],reply=reply,selected_codes=selected,requested_count=requested,awaiting_count=False,boards=boards,
        board_preference_id=previous.get('board_preference_id'),
        manual_observations=[],created_at=datetime.now().isoformat(),changes=[],stock_analyses=[],
        model_evidence={c:evidence[c] for c in selected},model_commentary={},commentary_warnings=[]))
    from assistant_runtime import gate
    gate('plan',draft)

def refreshed(draft, evidence=None):
    if draft.get('origin')!='AUTOMATIC_PREPARATION':return draft
    import plan_evidence
    current = plan_evidence.for_snapshot(draft['snapshot']) if evidence is None else evidence
    return dict(draft,model_evidence={c:current[c] for c in draft['selected_codes']})
