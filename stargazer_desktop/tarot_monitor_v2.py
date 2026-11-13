"""Confirmed-plan prewarming and bounded advice. Human decisions only; no orders."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
from subprocess import TimeoutExpired
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
    from runtime_disk_limit_v1 import disk_limit, disk_usage
    limit = disk_limit(root)
    size = len(json.dumps(value, ensure_ascii=False).encode('utf-8'))
    from persistent_records_v1 import is_record,reserve_space
    persistent=is_record(file)
    if persistent:reserve_space(file,size+65536)
    if not persistent and disk_usage(root) + size + 65536 > limit:
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
    from sector_watch_v1 import matches
    return state.get('enabled') is True and matches(plan,current)


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
    from live_plan_lane_v1 import CONTRACT as LIVE_CONTRACT,ready
    if record.get('observation_contract')==LIVE_CONTRACT:return ready(record,cutoff)
    observed=record.get('asof_ts')
    if observed==cutoff:return risk_ready(record,cutoff)
    if not isinstance(observed,int) or observed//1000000!=cutoff//1000000:return False
    age=(datetime.strptime(str(cutoff),'%Y%m%d%H%M%S')-datetime.strptime(str(observed),'%Y%m%d%H%M%S')).total_seconds()
    evidence=record.get('context',{}).get('market_price_evidence') or {}
    return (0<age<900 and record.get('mixed_cutoff_observation')==cutoff and risk_ready(record,observed)
        and all(evidence.get(n,{}).get('status')=='PRICE_FACTS_ONLY' and evidence[n].get('asof')==cutoff for n in ['sse','cyb']))



RECOVERABLE_TAGS = ('REALTIME_', 'LIVE_', 'TDX_', 'INDEX_MIRROR', 'TIMEOUT',
                    'SOURCE_LATE', 'WAIT_SOURCE', 'LIVE_CURRENT_MINUTES')
RETRY_COOLDOWN_SECONDS = 900


def _recoverable_failure(record):
    reason = str((record or {}).get('reason') or '')
    return any(tag in reason for tag in RECOVERABLE_TAGS)


def _failure_action(record, plan, now):
    """'proceed'：可恢复错误且冷却到期且总尝试<上限→重算（旧证据唯一归档）；
    'exhausted'：同绑定已达最大总尝试；'block'：永久错误或冷却中。
    次数取自独立侧car，与失败文件生命周期无关。"""
    if retry_attempts(plan) >= MAX_FAILURE_ATTEMPTS:
        return 'exhausted'
    if not _recoverable_failure(record):
        return 'block'
    entry = _retry_load().get(_retry_key(plan)) or {}
    try:
        failed_at = datetime.strptime(str(entry.get('last_failed_at') or ''), '%Y%m%d%H%M%S')
        if (now - failed_at).total_seconds() < RETRY_COOLDOWN_SECONDS:
            return 'block'
    except (ValueError, TypeError):
        pass
    return 'proceed'


def _retry_key(plan):
    return '%s:%s:%s' % (_flow_version(), plan.get('assistant_plan_id') or '',
                         digest(plan)[:12])


def _retry_state_file():
    return runtime.directory() / 'tarot_v2' / 'retry_state.json'


def _retry_load():
    try:
        value = json.loads(_retry_state_file().read_bytes())
    except (OSError, ValueError):
        value = {}
    return value if isinstance(value, dict) else {}


def retry_attempts(plan):
    """独立重试状态：按(发布版本,计划身份,计划哈希)累计总尝试，
    归档失败文件不影响计数。"""
    entry = _retry_load().get(_retry_key(plan)) or {}
    return int(entry.get('attempts', 0))


def retry_bump(plan):
    state = _retry_load()
    key = _retry_key(plan)
    entry = state.get(key) or {}
    entry['attempts'] = int(entry.get('attempts', 0)) + 1
    entry['last_failed_at'] = datetime.now().strftime('%Y%m%d%H%M%S')
    state[key] = entry
    file = _retry_state_file()
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_suffix('.tmp')
    temp.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
    temp.replace(file)
    return entry['attempts']


def _archive_failure(failed, attempts):
    """唯一归档名，禁止覆盖旧证据。"""
    stamp = datetime.now().strftime('%H%M%S')
    archive = failed.with_name('%s_a%d_%s.json' % (failed.stem, attempts, stamp))
    counter = 0
    while archive.exists():
        counter += 1
        archive = failed.with_name('%s_a%d_%s_%d.json' % (failed.stem, attempts, stamp, counter))
    failed.replace(archive)
    return archive


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



def _flow_version():
    """Current release manifest version the workers verify against."""
    try:
        import re
        from pathlib import Path as _P
        text = (_P(__file__).resolve().parent / 'golden_stock_analysis.py').read_text(encoding='utf-8')
        m = re.search(r'live_monitor_fingerprints_\d+_(v\d+)_flow', text)
        return m.group(1) if m else 'UNKNOWN'
    except Exception:
        return 'UNKNOWN'


def failure_stale(record, plan):
    """A same-day failure only blocks while its BINDING holds: same release
    version, same plan id and same plan hash. A newer verified release (or a
    changed plan) makes the old failure stale: the current effective cutoff
    may be recomputed. The old record is preserved untouched."""
    if not isinstance(record, dict):
        return True
    if record.get('flow_version') != _flow_version():
        return True
    if record.get('plan_id') and plan.get('assistant_plan_id') and             record['plan_id'] != plan.get('assistant_plan_id'):
        return True
    if record.get('plan_hash') and digest(plan) != record['plan_hash']:
        return True
    return False


MAX_FAILURE_ATTEMPTS = 3


def outlook_slot(cutoff, plan_id, plan_hash):
    """Slot identity shared with risk_outlook_v1.queue/generate."""
    key = digest(dict(mode='INTRADAY', cutoff=int(cutoff), plan_id=plan_id, plan_hash=plan_hash))
    return key, runtime.directory()/'risk_outlook'/(key + '.json')


def intraday_slot_wait(view, cutoff, plan_id, stale_only=False):
    """Defer a half-hour outlook slot until the index mirror has reached cutoff.

    A lagging mirror must not complete the slot without current-cutoff index
    facts, and must not pass older bars off as that evidence. While waiting, one
    WAITING_SOURCE record per slot keeps live quotes visible; a slot still
    unready after the 360s fresh window is recorded once as SOURCE_LATE.
    The report dispatcher may already have sent the slot's local facts.
    stale_only is the post-window path: an existing waiting slot may record its
    timeout, but a stale slot never starts waiting and never queues here.
    Returns True only when the caller may queue the slot normally.
    """
    key, slot = outlook_slot(cutoff, plan_id, view.get('plan_hash'))
    marker = slot.with_name(key + '.waiting.json')
    if slot.exists():
        return True
    if index_prefix_ready(cutoff):
        return True
    if stale_only and not marker.exists():
        return False  # nothing was waiting on this slot; stale slots never start
    now = datetime.now()
    age = (now - datetime.strptime(str(cutoff), '%Y%m%d%H%M%S')).total_seconds()
    late = age > 360
    if marker.exists():
        try:
            state = json.loads(marker.read_bytes())
        except (OSError, ValueError):
            state = {}
        want = 'SOURCE_LATE' if late else 'WAITING_SOURCE'
        if state.get('status') == want:
            return False  # already recorded exactly this state once for this slot
    facts = {}
    try:
        from risk_outlook_v1 import live_index_facts
        facts = live_index_facts(cutoff) or {}
    except (OSError, ValueError):
        facts = {}
    text = ('指数五分钟镜像超过新鲜窗口仍未达本截止 %d；本时间槽记为数据源延迟故障，不生成分析' % cutoff) if late else (
           '指数五分钟镜像尚未到达本截止 %d；等待期间仅展示实时报价，不以旧K线充当当前截止证据；镜像齐备后自动生成本槽分析' % cutoff)
    record = dict(status='SOURCE_LATE' if late else 'WAITING_SOURCE', key=key, asof=int(cutoff),
        mode='INTRADAY', plan_id=plan_id, plan_hash=view.get('plan_hash'), computed_facts=facts,
        automatic_trading=False, automatic_retry=not late, text=text)
    save(runtime.directory()/'risk_outlook'/'latest_intraday.json', record)
    save(marker, record)
    return False


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
    from trade_facts_v1 import CONTRACT
    market_evidence=risk.get('context',{}).get('market_price_evidence') or {}
    repair_version={n:dict(state=(e.get('research_trial') or {}).get('protected_state'),
        rsi=e.get('five',{}).get('rsi_above'),stable=e.get('price_stable_shape'),
        reclaimed=[x['price'] for x in e.get('levels',[]) if x.get('reclaimed')]) for n,e in market_evidence.items()}
    # A fresh event may be reviewed; identical events and failed attempts are never retried.
    asof_day = cutoff // 1000000
    try:
        from corvus_info_link_v1 import stock_facts, stock_summary, cutoff_to_datetime
        info_facts = stock_facts(code, cutoff_to_datetime(cutoff), cutoff)
    except Exception:
        def stock_summary(_code, _day):
            return dict(status='NOT_CONNECTED', reason='信息模型未接入：未知不等于无利空')
        info_facts = {'info.corvus': dict(asof=cutoff, text='信息模型未接入：该股公告信息未知，不等于没有利空',
                                          value=None, status='UNKNOWN')}
    # A corrected announcement (new full_body_sha) changes the info facts → re-review is
    # a fresh event; the old advice record stays untouched for traceability.
    corvus_version = digest(sorted((k, str(v)) for k, v in info_facts.items()))
    key = digest(dict(day=asof_day, plan=plan.get('assistant_plan_id'), code=code, kind=kind,
                      position=position, node=(signal or {}).get('node'), branch=(signal or {}).get('branch'),
                      trigger=(signal or {}).get('trigger_time'),holding_slot=cutoff if position is not None else None,
                      risk_version=risk.get('asof_ts'),repair_version=repair_version,fact_contract=CONTRACT,
                      corvus_version=corvus_version,
                      combination_policy=(signal or {}).get('combination_policy'),combination=(signal or {}).get('combination')))
    if (signal or {}).get('provisional'):
        # 实时试算分支：推送去重仍由 notice_key（不含信息版本）管控，保留原有推送次数规则；
        # 但重审键须包含 corvus_version，使更正公告（信息版本变化）触发重新分析而不命中旧建议。
        from live_plan_lane_v1 import notice_key
        # 注意：Windows 文件名不允许 ':'，使用安全分隔符 '--CV--' 拼接信息版本
        key='first_live_'+notice_key(plan,code,signal)+'--CV--'+corvus_version
    file = runtime.directory()/'tarot_v2'/'advice'/f'{key}.json'
    with _advice_lock:
        if file.exists():
            old = json.loads(file.read_bytes())
            from trade_facts_v1 import CONTRACT
            if old.get('status')=='ADVICE_READY' and old.get('fact_contract')!=CONTRACT:
                return dict(status='AI_LEGACY_UNVERIFIED',reason='旧AI自由文本未通过新版事实引用校验；原记录保留，不自动付费重试',previous=old,risk=risk)
            return old if old.get('cutoff') == cutoff else dict(status='REVIEW_ALREADY_RECORDED', previous=old, risk=risk)
        result = dict(status='AI_ATTEMPT_RESERVED', code=code, cutoff=cutoff, kind=kind, plan_id=plan.get('assistant_plan_id'),
                      risk=risk, signal=signal, position=position, automatic_trading=False)
        save(file, result)
    try:
        if not fresh(cutoff) or not enabled(plan):
            raise ValueError('PLAN_CHANGED_OR_EVIDENCE_STALE')
        import deepseek_api
        row = next((r for r in plan['rows'] if r['code'] == code), {})
        facts = dict(code=code, cutoff=cutoff, kind=kind, position=position, signal=signal,
                     selection_score=row.get('selection_score'), selection_model=row.get('selection_model'),
                     risk=risk, corvus=stock_summary(code, cutoff_to_datetime(cutoff)))
        from trade_facts_v1 import fact_table,validate_and_render,PROMPT,CONTRACT
        facts['computed_facts']=fact_table(cutoff,risk,signal,row)
        facts['computed_facts'].update(info_facts)  # info.* enters the legal citation table
        from plan_evidence import prompt_risk
        facts['risk']=prompt_risk(risk)
        result['computed_facts']=facts['computed_facts']
        prompt=PROMPT
        if (signal or {}).get('provisional'):
            prompt+=' 当前为实时试算、未收盘，可以按代码组合送审，但不可称收盘确认；数值随后可变。前收盘Pyxis幅度仅原截止参考，不是当前新推理。缺失当前板块或消息须保留未知，不自动视作利空或补为硬门。'
        settings = dict(cfg, deepseek=dict(cfg.get('deepseek') or {}, max_tokens=1536))
        response = deepseek_api.complete(settings, [{'role':'system','content':prompt},
                {'role':'user','content':json.dumps(facts, ensure_ascii=False, allow_nan=False)}],
                structured=True, purpose='review')
        result['raw_reply']=response.choices[0].message.content
        advice = json.loads(result['raw_reply'])
        reason=validate_and_render(advice,facts['computed_facts'],cutoff,position is not None)
        result.update(status='ADVICE_READY', decision=advice['decision'],reason=reason,
                      summary_reason=advice['rationale'],
                      risk_reasons=[facts['computed_facts'][k]['text'] for k in advice['opposing_ids']],
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


def work(plan, cfg, positions, cutoff, target, attempt=0):
    result = dict(status='PREWARMING', cutoff=cutoff, plan_hash=digest(plan), flow_version=_flow_version(),
                  plan_id=plan.get('assistant_plan_id'), positions_hash=digest(positions), rows={}, holdings={})
    try:
        from joint_risk_bridge import _worker_paths
        python, worker = _worker_paths()
        worker = worker.parent/'tarot_intraday_risk_v2.py'
        body = dict(plan=plan, positions=positions, cutoff=cutoff, runtime=str(runtime.directory()),
                    attempt=attempt)
        process = subprocess.run([str(python), '-B', '-X', 'utf8', str(worker)],
            input=json.dumps(body, ensure_ascii=False), capture_output=True, text=True, encoding='utf-8',
            timeout=600, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if process.returncode or len(process.stdout.encode()) > 2*1024**2:
            raise ValueError('RISK_WORKER_FAILED')
        risk = json.loads(process.stdout)
        if risk.get('status')=='SOURCE_PENDING':
            active=plan['rows'] if int(plan['monitor_date'])==cutoff//1000000 else []
            codes=sorted({r['code'] for r in active}|{p['code'] for p in positions})
            if risk.get('cutoff')!=cutoff or risk.get('requested_codes')!=codes:
                raise ValueError('RISK_BINDING_OR_SOURCE_FAILED')
            result.update(status='SOURCE_PENDING',reason=risk['reason'],risk=risk)
            save(target,result)
            save(target.parent/'latest.json',result)
            return
        if risk.get('status')=='EVIDENCE_INSUFFICIENT':
            raise ValueError(str(risk.get('reason') or 'RISK_CONTEXT_UNAVAILABLE'))
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
                result['rows'][code] = dict(signal, advice=dict(status='LOCAL_CLOSE_AUDIT',reason='收盘状态仅留档；首次通过建议由实时通道统一送审'))
            from risk_outlook_v1 import half_hour,queue
            for position in positions:
                code = position['code']
                # Preserve immediate mechanical stop/T+1 checks; AI context review is half-hourly.
                if half_hour(cutoff) or position.get('tarot_execution'):
                    result['holdings'][code] = advise(cfg, plan, code, cutoff, risk['records'].get(code, {}), position=position,allow_ai=half_hour(cutoff))
            if half_hour(cutoff):
                view=dict(plan_hash=digest(plan),items=[dict(code=c,facts=r,plan_levels=(next((x for x in plan['rows'] if x['code']==c),{}).get('entry_rule') or {}).get('levels',[])) for c,r in risk['records'].items()])
                if intraday_slot_wait(view,cutoff,plan.get('assistant_plan_id')):
                    queue(cfg,view,cutoff,mode='INTRADAY',positions=positions,signals=signals.get('rows'),overlay=True,plan_id=plan.get('assistant_plan_id'))
            result['status'] = 'MONITORING'
    except Exception as exc:
        reason=('RISK_WORKER_TIMEOUT:600s' if isinstance(exc,TimeoutExpired)
                else type(exc).__name__+':'+str(exc)[:240])
        result.update(status='EVIDENCE_FAILED', reason=reason, automatic_retry=False)
        # 失败绑定发布版本+计划身份+计划哈希：新版本校验通过后，旧失败不再锁当天；
        # 同版本重试有界(≤3次)，旧记录保留不覆盖。
        result['flow_version'] = _flow_version()
        result['failed_at'] = datetime.now().strftime('%Y%m%d%H%M%S')
        result['plan_id'] = plan.get('assistant_plan_id')
        result['plan_hash'] = digest(plan)
        failure_file = target.parent/f'failed_{cutoff//1000000}.json'
        result['attempts'] = retry_bump(plan)  # 独立侧car累计：归档不影响计数
        if failure_file.exists():
            _archive_failure(failure_file, result['attempts'])  # 旧证据唯一归档保留
        save(failure_file, result)
    save(target, result)
    save(target.parent/'latest.json', result)


def status_for(plan, code, ts):
    """UI reads cannot initiate inference or paid calls."""
    try:
        try:value = json.loads((runtime.directory()/'tarot_v2/latest.json').read_bytes())
        except OSError:value={}
        query = datetime.strptime(str(ts), '%Y%m%d%H%M%S')
        try:
            event=json.loads((runtime.directory()/'tarot_events_live/latest.json').read_bytes())
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
    from notifier import log
    from joint_risk_bridge import completed_cutoff
    from services import active_positions
    positions = active_positions(cfg)
    from tarot_event_v1 import evaluate as evaluate_events
    events=evaluate_events(plan,cfg,now)
    cutoff = completed_cutoff(now)
    if cutoff is None:
        return dict(status=(events or {}).get('status','WAITING'), rows=(events or {}).get('rows',{}),
            event_cutoff=(events or {}).get('cutoff'),event_status=(events or {}).get('status'),
            reason='五分钟事件独立更新；十五分钟风险确认尚未到首个时点')
    from risk_outlook_v1 import half_hour,queue
    prefix_ready=index_prefix_ready(cutoff)
    active=plan['rows'] if int(plan['monitor_date'])==cutoff//1000000 else []
    if half_hour(cutoff) and fresh(cutoff,now):
        from tarot_event_v1 import read_risk
        previous=read_risk(plan)
        records=previous.get('risk',{}).get('records',{}) if previous.get('cutoff')==cutoff else {}
        codes=sorted({r['code'] for r in active}|{p['code'] for p in positions})
        view=dict(plan_hash=digest(plan),observe_live=True,
            plan_status='CURRENT_CONFIRMED' if active else 'NO_CURRENT_PLAN',
            data_wait_reason='两指数本日完整五分钟前缀未跟进，Pyxis等待；实时报价不代替分钟确认' if not prefix_ready else '同截止风险正在准备',
            items=[dict(code=c,facts=records.get(c,{}),plan_levels=(next((r for r in plan['rows'] if r['code']==c),{}).get('entry_rule') or {}).get('levels',[])) for c in codes])
        failed_path=runtime.directory()/'tarot_v2'/f'failed_{cutoff//1000000}.json'
        failed_risk=json.loads(failed_path.read_bytes()) if failed_path.exists() else {}
        if failed_risk and failure_stale(failed_risk,plan):
            failed_risk={}
        if previous.get('status')=='SOURCE_PENDING' and previous.get('cutoff')==cutoff:
            view['data_wait_reason']='完整板块实时上下文尚未接入；个股Pyxis未计算，价格与规则独立运行'
        if failed_risk:
            view['data_wait_reason']='盘中风险取数失败；不冒充当刻风险：'+str(failed_risk.get('reason','未知原因'))
        # Market facts/outlook are independent of stock Pyxis completion. The
        # mirror-slot guard still waits for the exact cutoff; individual BUY
        # advice remains behind event_risk_ready(), never bypassed here.
        if intraday_slot_wait(view,cutoff,plan.get('assistant_plan_id')):
            queue(cfg,view,cutoff,mode='INTRADAY',positions=positions,signals=(events or {}).get('rows'),
                overlay=index_prefix_ready(cutoff),plan_id=plan.get('assistant_plan_id'))
    elif half_hour(cutoff):
        # Past the fresh window the slot can never run; only let an existing
        # waiting slot record SOURCE_LATE. Never queue AI or push from here.
        intraday_slot_wait(dict(plan_hash=digest(plan),observe_live=True),cutoff,
            plan.get('assistant_plan_id'),stale_only=True)
    def combine(value):
        from plan_rule_profiles import policy
        stale={r['code'] for r in plan.get('rows',[]) if r.get('entry_rule',{}).get('policy')=='GOLDEN_RESONANCE_CROSSDAY_V4'
            and value.get('rows',{}).get(r['code'],{}).get('combination_policy')!=policy(r['entry_rule'].get('execution_profile'))}
        if stale:
            value=dict(value,rows={c:(dict(status='WAIT_SIGNAL',reason='EXECUTION_CONTRACT_CHANGED_WAIT_CURRENT_5M',pending_conditions=['等待当前三选二规则重核'])
                if c in stale else r) for c,r in value.get('rows',{}).items()})
        if events and events.get('cutoff'):
            value=dict(value,event_cutoff=events['cutoff'],event_status=events.get('status'),rows=events.get('rows',value.get('rows',{})))
        return value
    if not active and not positions:
        return combine(dict(status='MARKET_ONLY',cutoff=cutoff,rows={},
            reason='当前没有当日候选或实际持仓；大盘巡检继续，不启动空个股风险批次'))
    folder = runtime.directory()/'tarot_v2'
    target = folder/f'{cutoff}_{digest(plan)}_{digest(positions)}.json'
    with _lock:
        if target.exists():
            cached = json.loads(target.read_bytes())
            if cached.get('status') != 'EVIDENCE_FAILED':
                return combine(cached)
            # 已存在的失败截止结果：先判是否允许恢复，不得直接返回旧结果
        if not prefix_ready:
            import index_minutes_v1
            sync=index_minutes_v1.get_index_freshness().get('source_sync') or {}
            detail=sync.get('error') or ';'.join(str(v.get('reason') or '') for v in sync.get('symbols',{}).values())
            return combine(dict(status='MONITOR_BLOCKED' if 'BUDGET' in detail else 'WAIT_INDEX_FEED',rows={},
                reason='运行缓存空间不足，指数同步和风险监控受阻：'+detail if 'BUDGET' in detail else
                    '等待两指数当前完整前缀；'+detail+'；未启动风险批次，不登记整日失败'))
        failed = folder/f'failed_{cutoff//1000000}.json'
        attempt = 0
        if failed.exists():
            record = json.loads(failed.read_bytes())
            if failure_stale(record, plan):
                log('旧版本/旧计划的当日失败已失效（记录保留）：当前版本允许重算本截止')
            else:
                action = _failure_action(record, plan, now)
                if action == 'exhausted':
                    return combine(dict(status='RISK_BATCH_FAILED', rows={}, reason='同绑定可恢复失败已达%d次总上限；证据保留，不自动重试' % MAX_FAILURE_ATTEMPTS))
                if action == 'block':
                    return combine(dict(status='RISK_BATCH_FAILED', rows={}, reason='本日风险批次失败已保留（永久错误或冷却中）；不重复启动整批重建'))
                _archive_failure(failed, retry_attempts(plan) + 1)
                attempt = retry_attempts(plan) + 1
                log(f'可恢复失败冷却到期：第{attempt}次尝试本截止（旧证据已唯一归档）')
        if _thread and _thread.is_alive():
            return combine(dict(status='PREWARMING', rows={}, reason='同截止风险正在预核验，原持仓风控继续运行'))
        save(target, dict(status='PREWARMING', rows={}, cutoff=cutoff, automatic_retry=False))
        _thread = threading.Thread(target=work, args=(plan, cfg, positions, cutoff, target, attempt),
                                   daemon=True, name='tarot-risk-prewarm')
        _thread.start()
        return combine(dict(status='PREWARMING', rows={}, reason='已提前启动风险核验，无需等触价才取数'))
