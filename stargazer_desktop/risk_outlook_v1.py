"""Cached DeepSeek scenarios and half-hour reports; no orders or plan mutations."""
from datetime import datetime
from copy import deepcopy
import json
import math
import re
import threading

import postclose_risk_runtime as runtime
from golden_rules import digest

_lock=threading.Lock()
_threads={}
CONTRACT='RISK_OUTLOOK_V4_OBSERVATION_CLOCKS'
CONDITIONS={
    'TWO_5M_CLOSES_BELOW':'同一黄金带下沿连续两根完整五分钟收盘失守，且未收复，警惕继续下探',
    'RSI_TURNS_DOWN':'五分钟RSI6回到RSI15下方、低点继续下降，警惕修复中断',
    'BOARD_LEADER_WEAKEN':'对应板块与领涨股同步转弱，降低介入积极性',
    'TWO_5M_RECLAIM':'刺破后连续两根完整五分钟收回同一档位／整带上沿，观察修复',
    'STABLE_RSI_UP':'未破位置附近低点不降、收盘抬高且RSI6在RSI15上方，观察企稳',
    'CONFIRM_15M':'十五分钟辅助同向确认；五分钟信号优先，不必等拉升后才观察'}
PROMPT=('你是Stargazer风险研判助手。只用computed_facts，沪指为主、创业板辅助，板块与龙头分开。'
    'POSTCLOSE给次日条件情景，INTRADAY给当前风险巡检；不能将当前修复等同未来一定上涨。'
    '输出JSON：asof原整数；bias只能REPAIR_BIAS/DOWNSIDE_BIAS/MIXED/UNKNOWN；'
    'sections为market/boards/leaders/holdings/plan五项简短中文字符串；'
    'focus_ids为至多六个黄金档位或共振带事实ID；supporting_ids/opposing_ids各至多十个ID，unknown_ids至多四十个ID，三组互斥；'
    'danger_signals与repair_signals为条件枚举数组，只能使用conditions中的key，各至多三项；'
    'sections不得重写价格、日期、百分比或给出数字，数值与信号由代码渲染，可使用RSI6/RSI15。'
    'UNKNOWN/RESEARCH仅能归unknown_ids；TRIAL可作有限参考或因未验收归unknown_ids，须明确试运行且量能未核销，不能声称已验收；'
    '不能用昨日或旧截止值冒充当前。KNOWN负涨跌幅不排除当刻修复；未知不等于利空。'
    '黄金带来源数不是独立投票，单档不称三重共振。不能编造持仓、成本、消息或资金流。'
    '说明可能危险、可能修复和失效条件，允许暂缓，不预设必须买入；逐股买卖另由规则触发评判，最终人工决定。')
PROMPT += ('不口述股票数量；candidate_codes由代码提供。holding_codes为null表示未提供实际持仓，'
           '此时holdings项只写未提供持仓，不将候选当持仓；空数组表示已知没有持仓。')
PROMPT += ('KNOWN事实不得进入unknown_ids；例如RSI上穿次数为零或偏少，仍是已知数值，'
           '不代表缺失。不能确定它的意义时可以不引用，不要把已知数值改称未知。')
PROMPT += ('严格遵守reference_policy中的引用资格：unknown_only_ids中的ID（包括*.missing）'
           '只能进unknown_ids或不引用。证据不足是等待原因，不是市场走弱的事实，不能列为opposing_ids。')
PROMPT += ('QUOTE_SNAPSHOT是独立实时报价，按其evidence_time及observed_at描述当前涨跌，'
           '不是bar_cutoff时的完整K线，不能证明RSI、两根收回、板块联动或Pyxis已就绪。'
           'asof由请求绑定；已有asof必须与请求一致，不输出type=json_object来代替asof。')


def half_hour(cutoff):
    return cutoff%1000000 in (100000,103000,110000,113000,133000,140000,143000,150000)


def view_id(view):
    return digest(dict(contract=CONTRACT,data_date=view.get('data_date'),monitor_date=view.get('monitor_date'),
        items=[{k:i.get(k) for k in ('code','facts','plan_levels')} for i in view.get('items',[])]))


def parse_reply(raw):
    """Only normalize a missing final brace and misplaced known sibling fields.

    No values, IDs or decisions are supplied or changed. The original reply is retained.
    """
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('OUTLOOK_DUPLICATE_JSON_KEY')
            result[k]=v
        return result
    notes=[]
    try:value=json.loads(raw,object_pairs_hook=unique)
    except json.JSONDecodeError as exc:
        if exc.pos!=len(raw.rstrip()):raise
        value=json.loads(raw.rstrip()+'}',object_pairs_hook=unique)
        notes.append('append_one_final_object_brace')
    sections=value.get('sections') or {}
    for key in ('focus_ids','supporting_ids','opposing_ids','unknown_ids','danger_signals','repair_signals'):
        if key in sections:
            if key in value:raise ValueError('OUTLOOK_AMBIGUOUS_FIELD_LOCATION')
            value[key]=sections.pop(key);notes.append('lift_sections.'+key)
    return value,notes


def bind_reply_time(value,cutoff):
    """Bind an omitted echo to its saved request; never repair a wrong time/bias."""
    result=deepcopy(value);notes=[]
    if 'asof' not in result:
        result['asof']=cutoff
        notes.append('asof_from_bound_request_not_model_echo')
    return result,notes


def live_index_facts(cutoff,now=None):
    """One public quote request; its clock stays separate from completed bars."""
    from tools import market
    requested_now=now
    now=now or datetime.now()
    end=datetime.strptime(str(cutoff),'%Y%m%d%H%M%S')
    if end.date()!=now.date() or not 0<=(now-end).total_seconds()<=360:return {}
    rows=market.index_quotes(tries=1)
    now=requested_now or datetime.now()
    observed=int(now.strftime('%Y%m%d%H%M%S'));facts={}
    for row in rows:
        name={'000001':('sse','沪指'),'399006':('cyb','创业板')}.get(row.get('code'))
        if not name:continue
        try:
            stamp=datetime.strptime(str(row['time']),'%Y%m%d%H%M%S')
            price,pct=float(row['price']),float(row['pct'])
            if (stamp.date()!=now.date() or not 0<=(now-stamp).total_seconds()<=120 or
                    not math.isfinite(price) or price<=0 or not math.isfinite(pct)):
                continue
        except (KeyError,TypeError,ValueError):continue
        key,label=name
        facts['market.'+key]=dict(asof=int(row['time']),evidence_time=int(row['time']),
            observed_at=observed,bar_cutoff=cutoff,evidence_kind='QUOTE_SNAPSHOT',
            source='TENCENT_PUBLIC_INDEX_QUOTE',status='KNOWN',value=pct/100,
            text=f'{label}实时报价 {price:.4f}，相对昨收 {pct:+.2f}%；报价时间 {row["time"]}；快照不代表完整五分钟确认')
    return facts


def collect(view,cutoff,positions=None,signals=None,overlay=True):
    from trade_facts_v1 import fact_table
    from plan_evidence import archived_price_overlay
    prices=archived_price_overlay(cutoff) if overlay else {}
    if signals is None and view.get('plan_hash'):
        try:
            observed=json.loads((runtime.directory()/'tarot_events/latest.json').read_bytes())
            if observed.get('cutoff')==cutoff and observed.get('plan_hash')==view['plan_hash']:
                signals=observed.get('rows')
        except (OSError,ValueError):pass
    facts={};items=view.get('items') or []
    for i,item in enumerate(items):
        code=item['code'];risk=item.get('facts') or {};ctx=dict(risk.get('context') or {})
        if prices:ctx['market_price_evidence']=prices
        local=fact_table(cutoff,dict(risk,context=ctx))
        for fact in local.values():
            stamp=fact.get('evidence_time')
            if stamp:
                age=(datetime.strptime(str(cutoff),'%Y%m%d%H%M%S')-datetime.strptime(str(stamp),'%Y%m%d%H%M%S')).total_seconds()
                if age<0 or age>900:
                    fact.update(status='UNKNOWN',historical_value=fact.get('value'),value=None,text='过期或截止不符，仅历史参考：'+fact['text'])
        for key,fact in local.items():
            if key.startswith(('index.','trial.','market.','corvus.')):
                if i:continue
                # Keep only nearest available bands, not every distant historical band.
                if '.band.' in key:
                    name=key.split('.')[1];bands=prices.get(name,{}).get('resonance_bands') or ctx.get('market_price_evidence',{}).get(name,{}).get('resonance_bands',[])
                    close=(prices or ctx.get('market_price_evidence',{})).get(name,{}).get('last_close')
                    if close is None:continue
                    chosen=sorted(bands,key=lambda b:abs(b['node']-close))[:6]
                    if fact['value'] not in chosen:continue
                facts[key]=fact
            elif key.startswith(('stock.','risk.amp','board_states.','leader_states.')):
                facts[code+'.'+key]=fact
        if risk.get('status')!='AVAILABLE':
            facts[code+'.missing']=dict(asof=cutoff,status='UNKNOWN',value=None,text=code+'风险证据不齐，不作低风险判断')
        for j,level in enumerate(item.get('plan_levels') or []):
            facts[f'{code}.plan.{j}']=dict(asof=cutoff,status='KNOWN',value=level,text=f'{code}计划观察档位 {level:.4f}')
    for p in positions or []:
        code=p['code'];facts[code+'.holding']=dict(asof=cutoff,status='KNOWN',value=True,text=code+'为实际持仓；成本和可卖数量仅以已登记值为准')
        entry=p.get('entry');entry_day=p.get('entry_date')
        if isinstance(entry,(int,float)) and entry>0:
            facts[code+'.cost']=dict(asof=cutoff,status='KNOWN',value=entry,text=f'{code}登记成本 {entry:.4f}；非预测买价')
        if entry_day:
            facts[code+'.entry_day']=dict(asof=cutoff,status='KNOWN',value=entry_day,text=f'{code}登记买入日 {entry_day}；卖出须核对T+1及可卖数量')
    for code,signal in (signals or {}).items():
        node=signal.get('next_lower_node') or signal.get('node');closes=signal.get('last_two_closes') or []
        price=closes[-1] if closes else None
        relation='价格或下一档未知'
        if isinstance(price,(int,float)) and isinstance(node,(int,float)) and node>0:
            distance=(price-node)/node
            relation=f'现价 {price:.4f}；下一观察档 {node:.4f}；距离 {distance:.2%}，不是触位买入'
        facts[code+'.progress']=dict(asof=cutoff,status='KNOWN',value=signal.get('status'),text=f'{code}规则进度 {signal.get("status")}；{relation}')
    if view.get('plan_hash'):
        for item in items:
            code=item['code']
            facts.setdefault(code+'.progress',dict(asof=cutoff,status='UNKNOWN',value=None,text=code+'同截止分钟进度待证，不能判断是否快到位'))
    return facts


def qualify_references(value,facts):
    """Enforce source eligibility, without changing prose, bias, IDs or values.

    Keep raw replies and record every move. Unknown evidence cannot become a
    confirmed supporting/opposing fact. Structural/duplicate errors still fail.
    """
    result=deepcopy(value);used=set()
    for group,limit in [('supporting_ids',10),('opposing_ids',10),('unknown_ids',40)]:
        refs=result.get(group)
        if not isinstance(refs,list) or len(refs)>limit:
            raise ValueError('OUTLOOK_FACT_REFERENCE')
        for key in refs:
            if not isinstance(key,str) or key not in facts or key in used:
                raise ValueError('OUTLOOK_FACT_REFERENCE')
            used.add(key)
    notes=[]
    for group in ('supporting_ids','opposing_ids'):
        allowed=[]
        for key in result[group]:
            if facts[key]['status'] in ('UNKNOWN','RESEARCH'):
                result['unknown_ids'].append(key)
                notes.append(dict(fact_id=key,source=group,destination='unknown_ids',
                                  reason=facts[key]['status']))
            else:allowed.append(key)
        result[group]=allowed
    if len(result['unknown_ids'])>40:raise ValueError('OUTLOOK_FACT_REFERENCE')
    return result,notes


def validate(value,facts,cutoff):
    if type(value.get('asof')) is not int or value['asof']!=cutoff or value.get('bias') not in ('REPAIR_BIAS','DOWNSIDE_BIAS','MIXED','UNKNOWN'):
        raise ValueError('OUTLOOK_TIME_OR_BIAS')
    sections=value.get('sections') or {}
    for key in ('market','boards','leaders','holdings','plan'):
        text=sections.get(key)
        if not isinstance(text,str) or not text.strip() or len(text)>400 or re.search(r'\d',re.sub(r'RSI\s*(6|15)','RSI',text,flags=re.I)):
            raise ValueError('OUTLOOK_SECTION_INVALID')
    used=set()
    for group in ('supporting_ids','opposing_ids','unknown_ids'):
        refs=value.get(group)
        if not isinstance(refs,list) or len(refs)>(40 if group=='unknown_ids' else 10) or any(not isinstance(k,str) or k not in facts or k in used for k in refs) or len(set(refs))!=len(refs):
            raise ValueError('OUTLOOK_FACT_REFERENCE')
        used.update(refs)
        for k in refs:
            if facts[k]['status'] in ('UNKNOWN','RESEARCH') and group!='unknown_ids':raise ValueError('OUTLOOK_UNKNOWN_AS_FACT')
            if facts[k]['status']=='KNOWN' and group=='unknown_ids':raise ValueError('OUTLOOK_KNOWN_AS_UNKNOWN')
    if not used:raise ValueError('OUTLOOK_NO_BASIS')
    focus=value.get('focus_ids')
    if not isinstance(focus,list) or len(focus)>6 or len(set(focus))!=len(focus) or any(k not in facts or facts[k]['status']!='KNOWN' or not ('.band.' in k or '.plan.' in k or re.fullmatch(r'index\.(sse|cyb)\.\d+',k) or (k.endswith('.reclaim') and isinstance(facts[k].get('value'),list) and facts[k]['value'])) for k in focus):
        raise ValueError('OUTLOOK_POSITION_REFERENCE')
    for field,allowed in [('danger_signals',set(list(CONDITIONS)[:3])),('repair_signals',set(list(CONDITIONS)[3:]))]:
        values=value.get(field)
        if not isinstance(values,list) or len(values)>3 or len(set(values))!=len(values) or not set(values)<=allowed:raise ValueError('OUTLOOK_CONDITIONS_INVALID')
    lines=['DeepSeek条件研判（非确定预测）：'+sections['market']]
    lines += [k+'：'+sections[n] for n,k in [('boards','板块'),('leaders','领涨股'),('plan','计划')]]
    lines += ['持仓：'+sections['holdings'] if any(k.endswith('.holding') for k in facts) else '持仓：本包未提供实际持仓，不输出持仓结论。']
    lines += ['关注位置：'+facts[k]['text'] for k in focus]
    lines += ['危险条件：'+CONDITIONS[k] for k in value['danger_signals']]
    lines += ['修复条件：'+CONDITIONS[k] for k in value['repair_signals']]
    lines += ['行情快照：'+f['text'] for f in facts.values() if f.get('evidence_kind')=='QUOTE_SNAPSHOT']
    lines += ['试运行新风险：'+f['text'] for f in facts.values() if f['status']=='TRIAL']
    lines += ['计划进度：'+f['text'] for k,f in facts.items() if k.endswith('.progress')]
    for key in facts:
        if key.endswith('.holding'):
            code=key.split('.')[0]
            lines += ['持仓事实：'+facts[k]['text'] for k in (code+'.cost',code+'.risk.amp_232',code+'.board_states.0.strength',code+'.leader_states.0.strength') if k in facts]
    return '\n'.join(lines)


def generate(cfg,view,cutoff,mode='POSTCLOSE',positions=None,signals=None,overlay=True,plan_id=None):
    from tarot_monitor_v2 import save,fresh
    import deepseek_api
    slot_key=digest(dict(mode=mode,cutoff=cutoff,plan_id=plan_id,plan_hash=view.get('plan_hash')))
    if mode=='INTRADAY':
        prior=runtime.directory()/'risk_outlook'/f'{slot_key}.json'
        if prior.exists():return json.loads(prior.read_bytes())
    facts=collect(view,cutoff,positions,signals,overlay)
    if mode=='INTRADAY' and view.get('observe_live'):
        try:
            for key,fact in live_index_facts(cutoff).items():
                if facts.get(key,{}).get('status')!='KNOWN':facts[key]=fact
        except (OSError,ValueError):pass  # Failed quotes remain unknown; no retry.
        dynamic=any(f['status']=='KNOWN' and (k.startswith(('market.','index.','board_states.','leader_states.'))
                    or '.stock.price' in k or '.risk.amp_' in k or
                    (k.endswith('.progress') and f.get('value') not in (None,'DATA_INSUFFICIENT')))
                    for k,f in facts.items())
        if not dynamic:
            result=dict(status='DATA_WAIT',key=slot_key,asof=cutoff,mode=mode,plan_id=plan_id,
                plan_hash=view.get('plan_hash'),computed_facts=facts,automatic_trading=False,
                text='当前行情与同截止分钟风险证据尚未就绪；'+str(view.get('data_wait_reason') or '')+
                     '；本次未调用AI，不据未知判断大盘方向。',automatic_retry=False)
            save(runtime.directory()/'risk_outlook/latest_intraday.json',result)
            return result  # No paid slot reserved; later real evidence may proceed.
    key=slot_key if mode=='INTRADAY' else digest(dict(mode=mode,cutoff=cutoff,plan_id=plan_id,facts=facts))
    path=runtime.directory()/'risk_outlook'/f'{key}.json'
    if path.exists():return json.loads(path.read_bytes())
    result=dict(status='AI_ATTEMPT_RESERVED',key=key,asof=cutoff,mode=mode,plan_id=plan_id,
        plan_hash=view.get('plan_hash'),computed_facts=facts,automatic_trading=False,prompt_contract=CONTRACT)
    save(path,result)
    try:
        body=dict(asof=cutoff,mode=mode,computed_facts=facts,conditions=CONDITIONS,
            candidate_codes=[x['code'] for x in view.get('items',[])],
            holding_codes=None if positions is None else [x['code'] for x in positions],
            reference_policy=dict(known_ids=[k for k,f in facts.items() if f['status']=='KNOWN'],
                unknown_only_ids=[k for k,f in facts.items() if f['status'] in ('UNKNOWN','RESEARCH')],
                trial_ids=[k for k,f in facts.items() if f['status']=='TRIAL']))
        settings=dict(cfg,deepseek=dict(cfg.get('deepseek') or {},max_tokens=2000))
        result['request']=body
        save(path,result)
        response=deepseek_api.complete(settings,[dict(role='system',content=PROMPT),dict(role='user',content=json.dumps(body,ensure_ascii=False))],structured=True,purpose='review')
        result['raw_reply']=response.choices[0].message.content
        save(path,result)
        parsed,normalization=parse_reply(result['raw_reply']);result['syntax_normalization']=normalization
        parsed,clock_notes=bind_reply_time(parsed,cutoff);result['clock_binding']=clock_notes
        parsed,notes=qualify_references(parsed,facts);result['reference_qualification']=notes
        text=validate(parsed,facts,cutoff)
        text=f'依据截止 {cutoff}；'+('次日计划观察日 '+str(view.get('monitor_date','未知')) if mode=='POSTCLOSE' else '当前半小时巡检')+'\n'+text
        if mode=='INTRADAY' and view.get('data_wait_reason'):
            text+='\n分钟链路：'+view['data_wait_reason']
        result.update(status='READY',analysis=parsed,text=text)
        if mode=='INTRADAY' and not fresh(cutoff):result.update(status='EXPIRED',reason='返回时已过时，不推送当刻建议')
    except Exception as exc:
        result.update(status='AI_FAILED',reason=type(exc).__name__,automatic_retry=False)
        if isinstance(exc,ValueError) and str(exc).startswith('OUTLOOK_'):
            result['validation_error']=str(exc)
            result['reason']='AI回复未通过事实校验（'+str(exc)+'），本地事实仍可查看'
    save(path,result)
    save(runtime.directory()/'risk_outlook'/('latest_'+mode.lower()+'.json'),result)
    return result


def latest(mode='POSTCLOSE'):
    try:return json.loads((runtime.directory()/'risk_outlook'/('latest_'+mode.lower()+'.json')).read_bytes())
    except (OSError,ValueError):return dict(status='NOT_GENERATED')


def queue(cfg,view,cutoff,**kwargs):
    mode=kwargs.get('mode','POSTCLOSE')
    if mode=='INTRADAY':
        path=runtime.directory()/'risk_outlook'/f'{digest(dict(mode=mode,cutoff=cutoff,plan_id=kwargs.get("plan_id"),plan_hash=view.get("plan_hash")))}.json'
        if path.exists():return json.loads(path.read_bytes())
    with _lock:
        thread=_threads.get(mode)
        if thread and thread.is_alive():return dict(status='RUNNING')
        def run():
            try:generate(cfg,view,cutoff,**kwargs)
            except Exception as exc:
                from tarot_monitor_v2 import save
                value=dict(status='PREPARATION_FAILED',asof=cutoff,plan_id=kwargs.get('plan_id'),reason=type(exc).__name__,automatic_retry=False)
                save(runtime.directory()/'risk_outlook'/('latest_'+mode.lower()+'.json'),value)
        thread=threading.Thread(target=run,daemon=True,name='risk-outlook-'+mode)
        _threads[mode]=thread
        thread.start()
    return dict(status='QUEUED',asof=cutoff)
