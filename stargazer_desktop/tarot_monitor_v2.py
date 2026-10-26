"""Confirmed-plan prewarming and bounded advice. Human decisions only; no orders."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import threading

import postclose_risk_runtime as runtime
from golden_rules import digest

VERSION = 'TAROT_A_SECOND_CASH_V2'
_lock = threading.Lock()
_thread = None
_advice_lock = threading.Lock()


def save(file, value):
    from pentacles_monitor import atomic_json
    root = runtime.directory()
    policy = root / 'runtime_budget_v2.json'
    limit = 1024**3
    if policy.exists():
        config = json.loads(policy.read_bytes())
        if config.get('user_authorized') is True and config.get('disk_bytes') == 4*1024**3:
            limit = 4*1024**3
    size = len(json.dumps(value, ensure_ascii=False).encode('utf-8'))
    if sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) + size + 65536 > limit:
        raise ValueError('RUNTIME_DISK_BUDGET')
    atomic_json(file, value)


def fresh(cutoff, now=None):
    now = now or datetime.now()
    then = datetime.strptime(str(cutoff), '%Y%m%d%H%M%S')
    return then.date() == now.date() and 0 <= (now-then).total_seconds() <= 360


def enabled(plan):
    import pentacles_demo
    directory = pentacles_demo.snapshot_path().parent
    state = json.loads((Path(directory)/'monitor.json').read_bytes())
    current = json.loads((Path(directory)/'assistant_plan.json').read_bytes())
    return state.get('enabled') is True and digest(current) == digest(plan)


def risk_ready(record, cutoff):
    from oracles_v1 import EvidencePack
    from jev_adapter_v1 import _risk_evidence_incomplete
    if record.get('status') != 'AVAILABLE' or record.get('asof_ts') != cutoff:
        return False
    context = record.get('context') or {}
    pack = EvidencePack(evidence_id='live-check', asof_ts=str(cutoff), market_date=str(cutoff)[:8],
                        query_ts=str(cutoff), candidates=[], risk_context=context, risk_missing=[])
    return str(context.get('asof_ts')) == str(cutoff) and not _risk_evidence_incomplete(pack)


def event_risk_ready(record,cutoff):
    observed=record.get('asof_ts')
    if observed==cutoff:return risk_ready(record,cutoff)
    if not isinstance(observed,int) or observed//1000000!=cutoff//1000000:return False
    age=(datetime.strptime(str(cutoff),'%Y%m%d%H%M%S')-datetime.strptime(str(observed),'%Y%m%d%H%M%S')).total_seconds()
    evidence=record.get('context',{}).get('market_price_evidence') or {}
    return (0<age<900 and record.get('mixed_cutoff_observation')==cutoff and risk_ready(record,observed)
        and all(evidence.get(n,{}).get('status')=='PRICE_FACTS_ONLY' and evidence[n].get('asof')==cutoff for n in ['sse','cyb']))


def index_prefix_ready(cutoff):
    from joint_risk_bridge import _worker_paths
    _,worker=_worker_paths()
    try:
        for name in ['sse','cyb']:
            meta=json.loads((worker.parent/'index_minutes_v1'/f'{name}_5min.meta.json').read_bytes())
            complete=int(meta.get('last_complete_ts') or 0)
            if complete//1000000!=cutoff//1000000 or complete<cutoff:return False
        return True
    except (OSError,ValueError,TypeError):return False


def advise(cfg, plan, code, cutoff, risk, signal=None, position=None,allow_ai=True):
    """Reserve before API, reuse exact attempt, and never retry failed paid calls."""
    if not event_risk_ready(risk, cutoff):
        return dict(status='WAIT_RISK', reason='同截止指数、板块或领涨证据不齐；保持未知', risk=risk)
    if position is None and (signal or {}).get('status') != 'RULE_TRIGGERED':
        return dict(status='WAIT_SIGNAL', risk=risk)
    if position is None and cutoff%1000000 >= 150000:
        return dict(status='ENTRY_SESSION_CLOSED', risk=risk)
    if position is not None and position.get('tarot_execution'):
        policy=position['tarot_execution']
        entry_day=int(position['entry_date'].replace('-',''))
        days=[d for d in risk.get('trading_days',[]) if entry_day<d<=cutoff//1000000]
        price=risk['context'].get('stock_close')
        stop=price is not None and price<=policy['stop_price']
        due=len(days)>=policy['horizon_sessions'] and cutoff%1000000>=150000
        if stop or due:
            allowed=cutoff//1000000>entry_day
            return dict(status='ADVICE_READY' if allowed else 'T1_WAIT',code=code,cutoff=cutoff,
                        kind='HOLDING',decision='EXIT' if allowed else 'DEFER',risk=risk,
                        reason='既定ATR止损触发' if stop else '既定H3持有期到期；请人工处理实际成交',
                        automatic_trading=False)
    if not allow_ai:return dict(status='OBSERVING',code=code,cutoff=cutoff,risk=risk,reason='本时点执行本地持仓风控，综合巡检在半小时节点生成')
    kind = 'HOLDING' if position is not None else 'ENTRY'
    market_evidence=risk.get('context',{}).get('market_price_evidence') or {}
    repair_version={n:dict(state=(e.get('research_trial') or {}).get('protected_state'),
        rsi=e.get('five',{}).get('rsi_above'),stable=e.get('price_stable_shape'),
        reclaimed=[x['price'] for x in e.get('levels',[]) if x.get('reclaimed')]) for n,e in market_evidence.items()}
    # A fresh event may be reviewed; identical events and failed attempts are never retried.
    key = digest(dict(day=cutoff//1000000, plan=plan.get('assistant_plan_id'), code=code, kind=kind,
                      position=position, node=(signal or {}).get('node'), branch=(signal or {}).get('branch'),
                      trigger=(signal or {}).get('trigger_time'),holding_slot=cutoff if position is not None else None,
                      risk_version=risk.get('asof_ts'),repair_version=repair_version))
    file = runtime.directory()/'tarot_v2'/'advice'/f'{key}.json'
    with _advice_lock:
        if file.exists():
            old = json.loads(file.read_bytes())
            from trade_facts_v1 import CONTRACT
            if old.get('status')=='ADVICE_READY' and old.get('fact_contract')!=CONTRACT:
                return dict(status='AI_LEGACY_UNVERIFIED',reason='旧AI自由文本未通过新版事实引用校验；原记录保留，不自动付费重试',previous=old,risk=risk)
            return old if old.get('cutoff') == cutoff else dict(status='REVIEW_ALREADY_RECORDED', previous=old, risk=risk)
        result = dict(status='AI_ATTEMPT_RESERVED', code=code, cutoff=cutoff, kind=kind,
                      risk=risk, signal=signal, position=position, automatic_trading=False)
        save(file, result)
    try:
        if not fresh(cutoff) or not enabled(plan):
            raise ValueError('PLAN_CHANGED_OR_EVIDENCE_STALE')
        import deepseek_api
        row = next((r for r in plan['rows'] if r['code'] == code), {})
        facts = dict(code=code, cutoff=cutoff, kind=kind, position=position, signal=signal,
                     selection_score=row.get('selection_score'), selection_model=row.get('selection_model'),
                     risk=risk, corvus='UNKNOWN_NOT_NO_BAD_NEWS')
        from trade_facts_v1 import fact_table,validate_and_render,PROMPT,CONTRACT
        facts['computed_facts']=fact_table(cutoff,risk,signal,row)
        from plan_evidence import prompt_risk
        facts['risk']=prompt_risk(risk)
        result['computed_facts']=facts['computed_facts']
        prompt=PROMPT
        settings = dict(cfg, deepseek=dict(cfg.get('deepseek') or {}, max_tokens=1536))
        response = deepseek_api.complete(settings, [{'role':'system','content':prompt},
                {'role':'user','content':json.dumps(facts, ensure_ascii=False, allow_nan=False)}],
                structured=True, purpose='review')
        result['raw_reply']=response.choices[0].message.content
        advice = json.loads(result['raw_reply'])
        reason=validate_and_render(advice,facts['computed_facts'],cutoff,position is not None)
        result.update(status='ADVICE_READY', decision=advice['decision'],reason=reason,
                      fact_contract=CONTRACT,computed_facts=facts['computed_facts'],
                      advice_references={k:advice[k] for k in ('supporting_ids','opposing_ids','unknown_ids')})
        if position is not None and advice['decision']=='EXIT':
            entry=str(position.get('entry_date','')).replace('-','')
            if not entry.isdigit() or int(entry)>=cutoff//1000000:
                result.update(status='T1_WAIT',decision='DEFER',reason='风险退出意见已保留，但买入日未知或属于当日；T+1可卖条件未满足，请人工核对')
        if not fresh(cutoff) or not enabled(plan):
            result.update(status='ADVICE_EXPIRED', reason='返回时计划已变化或证据已过时；保留记录，不推送交易建议')
    except Exception as exc:
        result.update(status='AI_FAILED', reason='AI未完成；本地事实保留，不自动重试', error_type=type(exc).__name__)
    save(file, result)
    return result


def work(plan, cfg, positions, cutoff, target):
    result = dict(status='PREWARMING', cutoff=cutoff, plan_hash=digest(plan),
                  plan_id=plan.get('assistant_plan_id'), positions_hash=digest(positions), rows={}, holdings={})
    try:
        from joint_risk_bridge import _worker_paths
        python, worker = _worker_paths()
        worker = worker.parent/'tarot_intraday_risk_v2.py'
        body = dict(plan=plan, positions=positions, cutoff=cutoff, runtime=str(runtime.directory()))
        process = subprocess.run([str(python), '-B', '-X', 'utf8', str(worker)],
            input=json.dumps(body, ensure_ascii=False), capture_output=True, text=True, encoding='utf-8',
            timeout=600, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if process.returncode or len(process.stdout.encode()) > 2*1024**2:
            raise ValueError('RISK_WORKER_FAILED')
        risk = json.loads(process.stdout)
        active_rows=plan['rows'] if int(plan['monitor_date'])==cutoff//1000000 else []
        if risk.get('cutoff') != cutoff or risk.get('requested_codes') != sorted({r['code'] for r in active_rows} | {p['code'] for p in positions}):
            raise ValueError('RISK_BINDING_OR_SOURCE_FAILED')
        result['risk'] = risk
        if not fresh(cutoff) or not enabled(plan):
            result.update(status='EVIDENCE_EXPIRED', reason='预核验完成但已过时，保留来源，不发建议')
        else:
            import golden_monitor
            moment = datetime.strptime(str(cutoff), '%Y%m%d%H%M%S')
            signals = golden_monitor.evaluate(plan, moment) if active_rows else dict(cutoff=cutoff,rows={})
            if signals.get('cutoff') != cutoff:
                raise ValueError('SIGNAL_RISK_CUTOFF_MISMATCH')
            for row in active_rows:
                code = row['code']; signal = signals['rows'][code]
                record = risk['records'].get(code, {})
                result['rows'][code] = dict(signal, advice=advise(cfg, plan, code, cutoff, record, signal=signal))
            from risk_outlook_v1 import half_hour,queue
            for position in positions:
                code = position['code']
                # Preserve immediate mechanical stop/T+1 checks; AI context review is half-hourly.
                if half_hour(cutoff) or position.get('tarot_execution'):
                    result['holdings'][code] = advise(cfg, plan, code, cutoff, risk['records'].get(code, {}), position=position,allow_ai=half_hour(cutoff))
            if half_hour(cutoff):
                view=dict(plan_hash=digest(plan),items=[dict(code=c,facts=r,plan_levels=(next((x for x in plan['rows'] if x['code']==c),{}).get('entry_rule') or {}).get('levels',[])) for c,r in risk['records'].items()])
                queue(cfg,view,cutoff,mode='INTRADAY',positions=positions,signals=signals.get('rows'),overlay=False,plan_id=plan.get('assistant_plan_id'))
            result['status'] = 'MONITORING'
    except Exception as exc:
        result.update(status='EVIDENCE_FAILED', reason=type(exc).__name__, automatic_retry=False)
        # Avoid restarting a failed whole-context batch on every subsequent minute/tick.
        save(target.parent/f'failed_{cutoff//1000000}.json', result)
    save(target, result)
    save(target.parent/'latest.json', result)


def status_for(plan, code, ts):
    """UI reads cannot initiate inference or paid calls."""
    try:
        try:value = json.loads((runtime.directory()/'tarot_v2/latest.json').read_bytes())
        except OSError:value={}
        query = datetime.strptime(str(ts), '%Y%m%d%H%M%S')
        try:
            event=json.loads((runtime.directory()/'tarot_events/latest.json').read_bytes())
            if event.get('plan_id')==plan.get('assistant_plan_id') and code in event.get('rows',{}) and fresh(event['cutoff'],query):value=event
        except (OSError,ValueError,KeyError):pass
        if value.get('plan_id') != plan.get('assistant_plan_id') or not fresh(value['cutoff'], query):
            raise ValueError('STALE_OR_WRONG_PLAN')
        row = value['rows'][code]
        stock = next(s for s in plan['stocks'] if s['code'] == code)
        if row.get('rule_hash') != digest(stock.get('entry_rule')):
            raise ValueError('RULE_CHANGED')
        advice = row.get('advice') or {}
        return dict(row, status=advice.get('status', row['status']), reason=advice.get('reason', row.get('reason')),
                    decision=advice.get('decision'), decision_source='TAROT_RULE_PYXIS_DEEPSEEK',
                    decision_note='人工决定；不自动交易')
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return dict(status='WAIT_MINUTE_EVIDENCE', reason='等待本计划同截止价格、风险与综合判断',
                    decision_source='TAROT_RULE_PYXIS_DEEPSEEK')


def evaluate(plan, cfg, now):
    global _thread
    from joint_risk_bridge import completed_cutoff
    from services import wl_load
    cutoff = completed_cutoff(now)
    if cutoff is None:
        return dict(status='WAITING', rows={}, reason='等待首个完整风险时点')
    positions = wl_load(cfg).get('positions', [])
    from tarot_event_v1 import evaluate as evaluate_events
    events=evaluate_events(plan,cfg,now)
    from risk_outlook_v1 import half_hour,queue
    prefix_ready=index_prefix_ready(cutoff)
    if half_hour(cutoff) and fresh(cutoff,now):
        from tarot_event_v1 import read_risk
        previous=read_risk(plan)
        records=previous.get('risk',{}).get('records',{}) if previous.get('cutoff')==cutoff else {}
        active=plan['rows'] if int(plan['monitor_date'])==cutoff//1000000 else []
        codes=sorted({r['code'] for r in active}|{p['code'] for p in positions})
        view=dict(plan_hash=digest(plan),observe_live=True,
            data_wait_reason='两指数本日完整五分钟前缀未跟进，Pyxis等待；实时报价不代替分钟确认' if not prefix_ready else '同截止风险正在准备',
            items=[dict(code=c,facts=records.get(c,{}),plan_levels=(next((r for r in plan['rows'] if r['code']==c),{}).get('entry_rule') or {}).get('levels',[])) for c in codes])
        if not prefix_ready or records:
            queue(cfg,view,cutoff,mode='INTRADAY',positions=positions,signals=(events or {}).get('rows'),overlay=prefix_ready,plan_id=plan.get('assistant_plan_id'))
    def combine(value):
        if events and events.get('cutoff'):
            value=dict(value,event_cutoff=events['cutoff'],event_status=events.get('status'),rows=events.get('rows',value.get('rows',{})))
        return value
    folder = runtime.directory()/'tarot_v2'
    target = folder/f'{cutoff}_{digest(plan)}_{digest(positions)}.json'
    with _lock:
        if target.exists():
            return combine(json.loads(target.read_bytes()))
        if not prefix_ready:
            return combine(dict(status='WAIT_INDEX_FEED',rows={},reason='等待两指数当前完整前缀；未启动风险批次，不登记整日失败'))
        failed = folder/f'failed_{cutoff//1000000}.json'
        if failed.exists():
            return combine(dict(status='RISK_BATCH_FAILED', rows={}, reason='本日风险批次失败已保留；不重复启动整批重建'))
        if _thread and _thread.is_alive():
            return combine(dict(status='PREWARMING', rows={}, reason='同截止风险正在预核验，原持仓风控继续运行'))
        save(target, dict(status='PREWARMING', rows={}, cutoff=cutoff, automatic_retry=False))
        _thread = threading.Thread(target=work, args=(plan, cfg, positions, cutoff, target),
                                   daemon=True, name='tarot-risk-prewarm')
        _thread.start()
        return combine(dict(status='PREWARMING', rows={}, reason='已提前启动风险核验，无需等触价才取数'))
