"""Conversation-local price results -> human-confirmed executable plan drafts."""
import copy
from datetime import datetime
import json
from pathlib import Path
import re

import assistant_workflow as W
import paths
from plan_rule_profiles import validate, describe, parse, policy


def session_key(session):
    return W.digest(str(session))[:24]


def remember(session, batch, result):
    if not result.get('plan_eligible') or not result.get('execution_row'):
        return None
    value = dict(session=session_key(session), batch=str(batch), result=result)
    value['analysis_id'] = W.digest(value)
    with W.LOCK:
        return W.store('analysis', value)


def recent(session):
    key = session_key(session)
    values = []
    for file in sorted(W.root().glob('analysis_*.json'), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        if not re.fullmatch(r'analysis_[0-9a-f]{64}',file.stem): continue
        raw = file.read_bytes()
        value = json.loads(raw)
        # W.store hashes the value before adding the returned id.
        if W.digest(value) != file.stem.removeprefix('analysis_'):
            raise ValueError('黄金分析记录哈希不符')
        if value['session'] == key:
            values.append(value)
    latest={}
    for v in values:
        if v['batch']==values[0]['batch']:
            latest.setdefault(v['result']['code'],v)
    return list(latest.values())


def prepare(session, codes=None, profile=None):
    values = recent(session)
    if not values:
        return dict(status='NEED_ANALYSIS', reply='当前对话没有可用于计划的黄金分析结果；请先告诉我要计算哪些股票。')
    previous=choice(session)
    if codes is None and profile is not None and previous and previous['batch']==values[0]['batch']:
        codes=previous['codes']
    if codes:
        wanted = set(codes)
        values = [v for v in values if v['result']['code'] in wanted]
        if {v['result']['code'] for v in values} != wanted:
            raise ValueError('指定股票不在当前对话最近一批黄金分析中，请先计算')
    if profile is None:
        with W.LOCK:
            W.store('analysis_choice',dict(session=session_key(session),batch=values[0]['batch'],codes=[v['result']['code'] for v in values]))
        return dict(status='CHOOSE_RULE', reply='关注价将原样采用刚计算的黄金共振加ATR区间。采用默认三选二，还是自定义规则？可说“用默认规则”，或说明三选几、上穿次数、超卖比例。尚未修改正式计划。',
                    codes=[v['result']['code'] for v in values])
    profile = validate(profile)
    dates = {v['result']['data_date'] for v in values}
    monitors = {v['result'].get('monitor_date') for v in values}
    if len(dates) != 1 or len(monitors) != 1 or None in monitors:
        raise ValueError('黄金分析数据日不一致或缺少交易日历，请重新计算')
    rows = []
    for item in values:
        r = copy.deepcopy(item['result']['execution_row'])
        if int(r['date']) != next(iter(dates)):
            raise ValueError('黄金分析日期绑定错误')
        r['entry_rule']['execution_profile'] = profile
        r['entry_rule']['combination_policy'] = policy(profile)
        r['entry_rule']['condition'] = describe(profile)
        r['analysis_id'] = item['analysis_id']
        rows.append(r)
    snap = dict(schema='PAGE_OF_PENTACLES_DEMO_V1', data_date=next(iter(dates)),
                monitor_date=next(iter(monitors)), rows=rows, plan_ready=True,
                rule_version=rows[0]['entry_rule']['version'],rule_inputs=copy.deepcopy(values[0]['result']['rule_inputs']), source='USER_GOLDEN_ANALYSIS')
    if W.plan_expired(snap):
        raise ValueError('这批黄金分析的观察日已开始或过期，请按最新收盘数据重新计算')
    latest = W.latest('plan')
    boards = copy.deepcopy(latest.get('boards', [])) if latest else []
    reply = ('已生成待确认草案，原样使用刚计算的关注区；仅作为人工指定股票，不冒充模型入围。\n'
             + '\n'.join(r['code'] + '：' + '、'.join('%g—%g' % (b['lower'], b['zone_upper'])
                         for b in r['entry_rule']['observation_bands']) for r in rows)
             + '\n' + describe(profile) + '\n点击“确认计划”后才开启；将替换原正式名单，不自动合并。风险需按新名单重新核验。')
    with W.LOCK:
        draft = W.store('plan', dict(origin='GOLDEN_ANALYSIS', session=session_key(session), snapshot=snap,
            snapshot_hash=W.digest(snap), selected_codes=[r['code'] for r in rows], boards=boards,
            manual_observations=[], model_evidence={}, model_commentary={}, reply=reply,
            dialogue=[], created_at=datetime.now().isoformat(), analysis_ids=[v['analysis_id'] for v in values]))
    return dict(status='DRAFT_READY', reply=reply, draft=draft)


def confirm(draft):
    """Called only from the existing human-confirmation endpoint under W.LOCK."""
    import trade_plan_v1
    if W.latest('plan')['id'] != draft['id'] or W.plan_expired(draft['snapshot']):
        raise ValueError('草案已被更新或观察日已开始，请重新制定')
    snap = copy.deepcopy(draft['snapshot'])
    if W.digest(snap) != draft['snapshot_hash']:
        raise ValueError('草案价格证据已变化')
    rows = snap['rows']
    if not snap.get('rule_inputs'):
        raise ValueError('黄金规则来源绑定缺失，请重新计算')
    import hashlib
    app=Path(paths.APP_DIR).resolve()
    project=next((p for p in (app,*app.parents) if (p/'stargazer_ml').is_dir()), app.parent)
    for ref in snap['rule_inputs']:
        file=Path(ref['path'])
        if not file.is_absolute(): file=project/file
        if hashlib.sha256(file.read_bytes()).hexdigest()!=ref['sha256']:
            raise ValueError('黄金计算后规则来源发生变化，请重新计算')
    for row in rows:
        p = validate(row['entry_rule'].get('execution_profile'))
        if policy(p) != row['entry_rule']['combination_policy']:
            raise ValueError('自定义规则指纹不符')
    snap.update(assistant_plan_id=draft['id'], boards=draft['boards'], manual_observations=[])
    formal = dict(version=1, created_ts=datetime.now().strftime('%Y%m%d%H%M%S'),
        data_date=str(snap['data_date']), monitor_date=str(snap['monitor_date']), boards=draft['boards'],
        assistant_plan_id=draft['id'], model_evidence={}, combined_commentary=draft['reply'],
        stocks=[dict(code=r['code'], name=r.get('name') or r['code'], buy_zone=[r['zone']['zone_lower'],r['zone']['zone_upper']],
            observe_only=False, source='USER_GOLDEN_ANALYSIS', entry_rule=r['entry_rule'],
            invalidations=['到位不等于买入；风险与AI意见单独推送，人工决定'],
            valid_until=str(snap['monitor_date'])+'150000', evidence_refs=[r['analysis_id']]) for r in rows])
    errors = trade_plan_v1.validate_plan(formal)
    if errors:
        raise ValueError('|'.join(errors))
    confirmed = W.root() / ('confirmed_' + draft['id'] + '.json')
    if not confirmed.exists():
        W.write(confirmed, dict(draft_id=draft['id'], plan=snap, confirmed_at=datetime.now().isoformat()))
    formal_file = Path(paths.data_file('data/trade_plan.json' if paths.FROZEN else 'dist/data/trade_plan.json', seed_from_bundle=False))
    trade_plan_v1.save_plan(formal, formal_file)
    from pentacles_monitor import atomic_json
    atomic_json(Path(paths.data_file('data/pentacles_demo/assistant_plan.json', seed_from_bundle=False)), snap)
    return dict(ok=True, plan_id=draft['id'], count=len(rows), monitor_date=snap['monitor_date'])


def direct(message, session):
    """Common phrases only; the Brain tools handle other paraphrases."""
    last = W.latest('plan')
    reference = bool(re.search(r'刚刚|刚才|算过|计算过|上面|这几只|这些', message))
    action = bool(re.search(r'加入|加到|放进|放到|列入|做成|制定|安排', message))
    target = bool(re.search(r'计划|明天|次日|观察', message))
    if action and (re.search(r'人工.{0,6}(?:观察|复核)|未过门|候补',message) or
                   reference and (last or {}).get('manual_review_candidates') and not re.search(r'黄金|共振|计算|算过',message)):
        return None  # Current review roster is resolved by plan_chat, not old golden-analysis history.
    if reference and action and target and not re.search(r'不要|别|不用', message):
        return prepare(session, re.findall(r'(?<!\d)\d{6}(?!\d)', message) or None)
    if recent(session) and choice(session):
        p = parse(message)
        if p is not None and (not last or last.get('session') == session_key(session) or '规则' in message):
            return prepare(session, profile=p)
    return None


def choice(session):
    for file in sorted(W.root().glob('analysis_choice_*.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True):
        value=json.loads(file.read_bytes())
        if W.digest(value)!=file.stem.removeprefix('analysis_choice_'):
            raise ValueError('待选规则上下文哈希不符')
        if value['session']==session_key(session): return value
    return None


def rule_discussion(message,session):
    return bool(choice(session) and re.search(r'规则|上穿|超卖|三选|条件|比例',message))



def _plan_stock_view(row):
    if not isinstance(row,dict):return row
    result={k:v for k,v in row.items() if k in ('code','name','buy_zone','observe_only','source','selection_qualified','conditions_status','valid_until','invalidations','evidence_refs')}
    rule=row.get('entry_rule') or {}
    result['entry_rule']={k:v for k,v in rule.items() if k in ('policy','eligible','combination_policy','execution_profile')}
    result['entry_rule']['observation_bands']=[{k:v for k,v in b.items() if k in
        ('lower','upper','zone_upper','node','node_price','source_waves','source_peak_groups')} for b in rule.get('observation_bands',[])]
    return result

def tool_view(value):
    """Full evidence remains local; send usable facts instead of a giant private row."""
    if not isinstance(value,dict): return value
    if isinstance(value.get('stocks'),list) and 'model_facts' in value:
        return {**value,'kind':'CONFIRMED_PLAN_VIEW_V1','status':'CONFIRMED_PLAN',
            'stocks':[_plan_stock_view(r) for r in value['stocks']],
            'manual_observations':[_plan_stock_view(r) for r in value.get('manual_observations') or []],
            'detail_notice':'计划名单、全部观察档位和规则已载入；历史分钟基准等底层字段仍保留本地。'}
    if value.get('execution_row'):
        result={k:v for k,v in value.items() if k not in ('execution_row','rule_inputs')}
        result['resonance_bands']=[{k:v for k,v in b.items() if k!='sources'} for b in result.get('resonance_bands',[])]
        return result
    if value.get('draft'):
        return dict(status=value['status'],reply=value['reply'],draft_id=value['draft']['id'],human_confirmation_required=True)
    if value.get('review_id'):
        return {k:v for k,v in value.items() if k not in ('context','rows')} | dict(kind='MINUTE_REVIEW_VIEW_V1',
            context={k:v for k,v in value.get('context',{}).items() if k in ('day','cutoff','archive_id','confirmed_population','plan_id')},rows=[
            {k:v for k,v in r.items() if k not in ('state_changes','first_signal')} | dict(
                first_signal={k:v for k,v in (r.get('first_signal') or {}).items() if k in ('trigger_time','node','branch','combination')})
            for r in value['rows']])
    return value
