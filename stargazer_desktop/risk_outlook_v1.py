"""Cached DeepSeek scenarios and half-hour reports; no orders or plan mutations.

闭环3：AI 预约（AI_ATTEMPT_RESERVED）生命周期——占位带 owner pid 与开始时间，
只区分"进行中/进程已退出/超时/已记录失败"；占位绝不冒充结果。DeepSeek 失败时
完整保留本地事实并单独显示分析失败原因。
"""
from datetime import datetime
from copy import deepcopy
import json
import math
import os
import re
import threading
import time
from subprocess import SubprocessError

import postclose_risk_runtime as runtime
from golden_rules import digest

AI_ATTEMPT_TTL=900.0

_lock=threading.Lock()
_threads={}
CONTRACT='RISK_OUTLOOK_V7_REBOUND_MAP'
CONDITIONS={
    'TWO_5M_CLOSES_BELOW':'同一黄金带下沿连续两根完整五分钟收盘失守，且未收复，警惕继续下探',
    'RSI_TURNS_DOWN':'五分钟RSI6回到RSI15下方、低点继续下降，警惕修复中断',
    'BOARD_LEADER_WEAKEN':'对应板块与领涨股同步转弱，降低介入积极性',
    'TWO_5M_RECLAIM':'刺破后连续两根完整五分钟收回同一档位／整带上沿，观察修复',
    'STABLE_RSI_UP':'未破位置附近低点不降、收盘抬高且RSI6在RSI15上方，观察企稳',
    'CONFIRM_15M':'十五分钟辅助同向确认；五分钟信号优先，不必等拉升后才观察'}
PROMPT=('你是Stargazer风险研判助手。只用computed_facts，沪指为主、深证成指、科创50与创业板辅助，板块与龙头分开。'
    'POSTCLOSE给次日条件情景，INTRADAY给当前风险巡检；不能将当前修复等同未来一定上涨。'
    '输出JSON：asof原整数；bias只能REPAIR_BIAS/DOWNSIDE_BIAS/MIXED/UNKNOWN；'
    'sections为market/boards/leaders/holdings/plan五项简短中文字符串；'
    'focus_ids为至多六个黄金档位或共振带事实ID；supporting_ids/opposing_ids各至多十个ID，unknown_ids至多四十个ID，三组互斥；'
    'danger_signals与repair_signals为条件枚举数组，只能使用conditions中的key，各至多三项；'
    'sections不得重写价格、日期、百分比或给出数字，数值与信号由代码渲染，可使用RSI6/RSI15。'
    'sections为纯文字判读，连计数也不得写入：如"跌超4家"必须写作"跌超阈值的家数"，'
    '"5只转弱"写作"多只转弱"；一切数量、计数、价位均由代码渲染。'
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
PROMPT += ('supporting_ids/opposing_ids/unknown_ids合计必须至少引用一个输入事实ID，不得全部为空。'
           '仅有指数报价也须引用market.sse或market.cyb等实际使用的ID；没有计划或持仓不免除大盘分析的依据引用。'
           'plan_status=NO_CURRENT_PLAN时只讨论大盘与真实登记持仓，不沿用旧候选、不建议给不存在的持仓加仓。')
PROMPT += ('focus_ids只能从reference_policy.focus_allowed_ids选；清单为空必须输出[]。'
           'market.sse/market.cyb实时报价不是黄金位，不能放focus_ids；应按事实放依据引用数组。')
PROMPT += ('rebound_map是独立的下跌高点到其后低点反弹地图，不是原执行地图，也不是Pyxis预测。'
           '结合锚点、反弹所处档位解释下一观察条件、守低失败条件和指数间是否同步；价位由代码渲染，sections仍不写数字。'
           '低点之后经过若干交易日不等于已确认双底；下影不证明承接资金流入。量能未核验时不能声称放量、缩量或量价共振。')


def half_hour(cutoff):
    return cutoff%1000000 in (100000,103000,110000,113000,133000,140000,143000,150000)


def outlook_input(view):
    """Do not carry yesterday's plan stocks into a newer market-only discussion."""
    day=view.get('market_data_date')
    if day and str(day)!=str(view.get('data_date')):
        return dict(data_date=day,monitor_date=view.get('market_monitor_date'),items=[],
                    plan_status='NO_CURRENT_PLAN')
    return view


def view_id(view):
    view=outlook_input(view)
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
        name={'000001':('sse','沪指'),'399001':('szse','深证成指'),'000688':('star50','科创50'),'399006':('cyb','创业板')}.get(row.get('code'))
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
            source='TENCENT_PUBLIC_INDEX_QUOTE',status='KNOWN',value=pct/100,price=price,
            day_low=row.get('day_low') if type(row.get('day_low')) in (int,float)
                and math.isfinite(row['day_low']) and 0 < row['day_low'] <= price else None,
            text=f'{label}实时报价 {price:.4f}，相对昨收 {pct:+.2f}%；报价时间 {row["time"]}；快照不代表完整五分钟确认')
        amount = row.get('amount_yi')
        if (type(amount) in (int, float) and math.isfinite(amount) and amount >= 0):
            facts['market.'+key+'.amount_yi'] = dict(
                asof=int(row['time']), evidence_time=int(row['time']), observed_at=observed,
                bar_cutoff=cutoff, evidence_kind='QUOTE_SNAPSHOT',
                source='TENCENT_PUBLIC_INDEX_QUOTE', status='KNOWN', value=amount, unit='亿元',
                text=f'{label}截至报价{row["time"]}累计成交额{amount:.1f}亿元；'
                     '成交额快照，非同截止成交量或历史同刻量比，不据此认定放量缩量')
    return facts


def collect(view,cutoff,positions=None,signals=None,overlay=True):
    from trade_facts_v1 import fact_table
    from plan_evidence import archived_price_overlay
    _overlay_error={}
    prices=archived_price_overlay(cutoff,_overlay_error) if overlay else {}
    if signals is None and view.get('plan_hash'):
        try:
            observed=json.loads((runtime.directory()/'tarot_events/latest.json').read_bytes())
            if observed.get('cutoff')==cutoff and observed.get('plan_hash')==view['plan_hash']:
                signals=observed.get('rows')
        except (OSError,ValueError):pass
    facts={};items=view.get('items') or []
    if overlay and not prices:
        facts['index.price_overlay']=dict(asof=cutoff,status='UNKNOWN',value=None,
            text='指数价格覆盖层不可用：'+(_overlay_error.get('reason') or '未知原因'))
    if not items and prices:
        # Market evidence is independent of candidate/holding membership.
        for key,fact in fact_table(cutoff,dict(context=dict(market_price_evidence=prices))).items():
            if not key.startswith(('index.','trial.','market.','corvus.')):continue
            stamp=fact.get('evidence_time')
            if stamp:
                age=(datetime.strptime(str(cutoff),'%Y%m%d%H%M%S')-datetime.strptime(str(stamp),'%Y%m%d%H%M%S')).total_seconds()
                if age<0 or age>900:
                    fact.update(status='UNKNOWN',historical_value=fact.get('value'),value=None,text='过期或截止不符，仅历史参考：'+fact['text'])
            facts[key]=fact
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
        progress=signal.get('status')
        # DATA_INSUFFICIENT 对推理不可用，标 UNKNOWN 让引用资格与模型直觉一致（同 generate 动态证据排除口径）。
        facts[code+'.progress']=dict(asof=cutoff,
            status='UNKNOWN' if progress in (None,'DATA_INSUFFICIENT') else 'KNOWN',
            value=progress,text=f'{code}规则进度 {progress}；{relation}')
    if view.get('plan_hash'):
        for item in items:
            code=item['code']
            facts.setdefault(code+'.progress',dict(asof=cutoff,status='UNKNOWN',value=None,text=code+'同截止分钟进度待证，不能判断是否快到位'))
    for name,evidence in prices.items():
        if evidence.get('status')!='PRICE_FACTS_ONLY' or evidence.get('asof')!=cutoff:
            # 源未更新/缺槽/过期：保留具体原因，不静默丢弃（反例3仍拒读，但说明拒什么）
            facts['index.'+name+'.price_state']=dict(asof=cutoff,status='UNKNOWN',value=None,
                text=name+'盘中价格证据未就绪：'+(evidence.get('reason') or '同截止完整五分钟不满足'))
            continue
        critical=evidence.get('critical_position') or {}
        state=critical.get('state','UNKNOWN')
        facts['index.'+name+'.critical_state']=dict(asof=cutoff,status='KNOWN' if state!='UNKNOWN' else 'UNKNOWN',value=state if state!='UNKNOWN' else None,
            text=name+'盘前固定关键位置状态：'+state,position=critical)
        if evidence.get('status')=='PRICE_FACTS_ONLY':
            # Three-layer map input: raw same-cutoff evidence (levels, node,
            # session facts, big-picture outlook), consumed by
            # index_map_layers_v1; never model prose.
            facts['index.'+name+'.map_input']=dict(asof=cutoff,status='KNOWN',value=evidence)
        facts['market.'+name+'.price']=dict(asof=cutoff,status='KNOWN',value=evidence['last_close'],text=name+'完整五分钟收盘价 '+str(evidence['last_close']))
        from index_outlook_v1 import facts_for
        facts.update(facts_for(name,evidence,cutoff))
        from index_map_layers_v1 import build_layers
        layers = build_layers(name, evidence, cutoff) or {}
        lower_watch = layers.get('next_support') if layers.get('band_is_current') else layers.get('band')
        if lower_watch:
            lower, upper = lower_watch['lower'], lower_watch['upper']
            facts['index.'+name+'.band.lower_watch'] = dict(
                asof=cutoff, status='KNOWN', value=lower_watch,
                text=f'{name}下方黄金观察带 {lower:.2f}—{upper:.2f}')
    facts.update(_board_quote_facts(cutoff))
    return facts


def _board_quote_snapshot():
    from strong_sector_watch_v1 import root as strong_root
    return json.loads((strong_root() / 'summary.json').read_bytes())


def overview_boards(now):
    try:
        packet = _board_quote_snapshot()
        stamp = datetime.strptime(str(packet['asof']), '%Y%m%d%H%M%S')
        if stamp.date() == now.date() and stamp <= now:
            rows = []
            for item in packet['boards']:
                if item.get('valid_count', 0) > 0 and isinstance(item.get('mean_return'), (int, float)):
                    leader = item.get('leader') or {}
                    rows.append(dict(name=item['name'], leader=leader.get('name') or leader.get('code'),
                                     change=item['mean_return'], valid_count=item['valid_count']))
            return packet['asof'], rows
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None, []


def _board_quote_facts(cutoff):
    facts = {}
    try:
        packet=_board_quote_snapshot()
        age=(datetime.strptime(str(cutoff),'%Y%m%d%H%M%S')-datetime.strptime(str(packet['asof']),'%Y%m%d%H%M%S')).total_seconds()
        if -120<=age<=360:
            for board in packet.get('boards',[]):
                facts['sector.watch.'+board['name']]=dict(asof=packet['asof'],observed_at=packet['asof'],bar_cutoff=cutoff,evidence_kind='QUOTE_SNAPSHOT',
                    status='KNOWN',value=board,text=board['name']+'冻结强势池报价观察：'+json.dumps(board,ensure_ascii=False)+'；并非全板块龙头认定')
    except (OSError,ValueError,KeyError):pass
    return facts


def board_flow_snapshot(*, tries=3, timeout=10):
    from tools.market import board_fund_flow
    try:
        inflow, outflow = board_fund_flow(3, tries=tries, timeout=timeout)
        inflow = [item for item in inflow if item[2] > 0]
        outflow = [item for item in outflow if item[2] < 0]
        if inflow or outflow:
            return dict(inflow=inflow, outflow=outflow,
                        observed_at=datetime.now().isoformat(timespec='seconds'),
                        source='东方财富板块主力净流向')
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


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


def focus_allowed_ids(facts):
    return [k for k,f in facts.items() if f['status']=='KNOWN' and
        ('.band.' in k or '.plan.' in k or re.fullmatch(r'index\.(sse|szse|star50|cyb)\.(\d+|rebound\.\d+)',k) or
         (k.endswith('.reclaim') and isinstance(f.get('value'),list) and f['value']))]


def qualify_focus(value,facts):
    """Drop only known quote highlights; never invent a position or alter analysis."""
    focus=value.get('focus_ids')
    if (not isinstance(focus,list) or len(focus)>6 or
            any(not isinstance(k,str) or k not in facts for k in focus) or len(set(focus))!=len(focus)):
        raise ValueError('OUTLOOK_POSITION_REFERENCE')
    allowed=set(focus_allowed_ids(facts));removed=[]
    for k in focus:
        if k in allowed:continue
        if (k not in ('market.sse','market.szse','market.star50','market.cyb') or facts[k]['status']!='KNOWN' or
                facts[k].get('evidence_kind')!='QUOTE_SNAPSHOT'):
            raise ValueError('OUTLOOK_POSITION_REFERENCE')
        removed.append(dict(fact_id=k,reason='QUOTE_IS_NOT_GOLDEN_POSITION'))
    result=deepcopy(value)
    result['focus_ids']=[k for k in focus if k in allowed]
    return result,removed


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
    if not isinstance(focus,list) or len(focus)>6 or any(not isinstance(k,str) for k in focus) or len(set(focus))!=len(focus) or any(k not in focus_allowed_ids(facts) for k in focus):
        raise ValueError('OUTLOOK_POSITION_REFERENCE')
    for field,allowed in [('danger_signals',set(list(CONDITIONS)[:3])),('repair_signals',set(list(CONDITIONS)[3:]))]:
        values=value.get(field)
        if not isinstance(values,list) or len(values)>3 or len(set(values))!=len(values) or not set(values)<=allowed:raise ValueError('OUTLOOK_CONDITIONS_INVALID')
    from index_outlook_v1 import render as render_map
    lines=[render_map(facts)] if render_map(facts) else []
    lines+=['DeepSeek条件研判（非确定预测）：'+sections['market']]
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


def _ai_reservation_reclaimable(stored):
    """AI_ATTEMPT_RESERVED 占位是否可收回重算（区分进行中/已退出/超时/遗留占位）。"""
    if not isinstance(stored,dict) or stored.get('status')!='AI_ATTEMPT_RESERVED':
        return False
    try:owner=int(stored.get('owner_pid') or 0)
    except (TypeError,ValueError):owner=0
    try:started=float(stored.get('started_epoch') or 0.0)
    except (TypeError,ValueError):started=0.0
    if not owner or not started:
        return True  # 遗留占位（旧版本无身份）：无法证明仍在计算 → 收回
    try:
        from cross_process_lock_v1 import pid_alive
        return not pid_alive(owner)  # 进程已退出 → 收回续算
    except Exception:
        return False  # 无法确认死亡，保留本地事实，不并发重调。


def with_local_failure(result):
    """Readable code facts on failure or pending AI; never repair or approve the AI opinion.

    闭环3：失败/进行中都完整保留本地事实（含 RSI6/15 及截止、背离、成交额、
    成交量、同刻量比），并单独显示分析失败原因；绝不把已有指标统一写成待核验。
    """
    if result.get('status') not in ('AI_FAILED','AI_ATTEMPT_RESERVED') or result.get('text'):
        return result
    facts=result.get('computed_facts') or {};request=result.get('request') or {}
    if result.get('status')=='AI_ATTEMPT_RESERVED':
        code='AI_ATTEMPT_RESERVED'
        reason='AI分析仍在进行（另一实例正在调用），未自动重试；以下为本地事实'
    else:
        code=result.get('validation_error') or result.get('reason') or 'AI_UNAVAILABLE'
        reason='AI回复未填写事实依据引用' if code=='OUTLOOK_NO_BASIS' else 'AI解释暂不可用'
    lines=[f'依据截止 {result["asof"]}；{reason}（{code}）；未自动重试。',
           '以下为程序读取的本地事实，不是AI研判或买卖建议：']
    from index_outlook_v1 import render as render_map
    if render_map(facts):lines.append(render_map(facts))
    shown=set()
    from index_outlook_v1 import NAMES
    for name,label in (*NAMES.items(), ('star50', '科创50')):
        quote=facts.get('market.'+name,{})
        lines.append(quote['text'] if quote.get('status')=='KNOWN' else label+'当前涨跌：未知')
        shown.add('market.'+name)
        repair=facts.get('index.'+name+'.repair',{})
        if repair.get('status')=='KNOWN':
            lines.append(repair['text']);shown.add('index.'+name+'.repair')
    known=[f['text'] for k,f in sorted(facts.items())
           if f.get('status')=='KNOWN' and f.get('text') and k not in shown]
    if known:
        lines.append('本地事实（程序逐条读取，非AI研判）：')
        lines+=['· '+t for t in known]
    unknown=[f['text'] for k,f in sorted(facts.items())
             if f.get('status')=='UNKNOWN' and f.get('text') and
             any(t in k for t in ('rsi','divergence','amount','volume','same_time','progress','price_state','missing'))]
    if unknown:
        lines.append('未核实项（保持未知，不冒充成功）：')
        lines+=['· '+t for t in unknown]
    if request.get('plan_status')=='NO_CURRENT_PLAN' or request.get('candidate_codes')==[]:
        lines.append('计划：本次没有当日已确认候选，不沿用旧名单；大盘巡检继续。')
    holdings=request.get('holding_codes')
    lines.append('持仓：当前登记为空。' if holdings==[] else
                 '持仓：未提供实际持仓，不输出持仓结论。' if holdings is None else
                 '持仓：实际登记'+ '、'.join(holdings)+'；AI持仓评估未完成，请核对本地风控。')
    lines.append('完整五分钟修复、板块龙头与Pyxis幅度须有同截止证据；缺失保持未知，报价不替代K线确认；成交额不解释为成交量或放量。')
    return dict(result,text='\n'.join(lines),fallback_kind='CODE_FACTS_ONLY')


def _cached_result(path):
    """Read a cached result; an AI reservation placeholder is never a result.

    占位区分：另一存活实例在算（且未超时）→ 返回带本地事实的进行中结果；
    进程已退出 / 超时 / 遗留占位 → 视为缓存未命中（None，由调用方收回重算）。
    """
    try:cached=json.loads(path.read_bytes())
    except (OSError,ValueError):return None
    if isinstance(cached,dict) and cached.get('status')=='AI_ATTEMPT_RESERVED':
        if _ai_reservation_reclaimable(cached):
            return None  # 收回：重算（新占位将携带 reclaimed_from 证据）
        return with_local_failure(cached)
    return cached


def generate(cfg,view,cutoff,mode='POSTCLOSE',positions=None,signals=None,overlay=True,plan_id=None):
    from tarot_monitor_v2 import save,fresh
    import deepseek_api
    slot_key=digest(dict(mode=mode,cutoff=cutoff,plan_id=plan_id,plan_hash=view.get('plan_hash')))
    if mode=='INTRADAY':
        prior=runtime.directory()/'risk_outlook'/f'{slot_key}.json'
        if prior.exists():
            cached=_cached_result(prior)
            if cached is not None:return cached
    live_facts = {}
    if mode == 'INTRADAY' and view.get('observe_live'):
        try:
            live_facts = live_index_facts(cutoff)
        except (OSError, ValueError):
            pass
        if live_facts:
            marker = prior.with_name(slot_key + '.local.json')
            save(marker, dict(status='LOCAL_FACTS', key=slot_key, asof=cutoff, mode=mode,
                              plan_id=plan_id, plan_hash=view.get('plan_hash'), computed_facts=live_facts))
    facts=collect(view,cutoff,positions,signals,overlay)
    board_flow = None
    if mode == 'INTRADAY' and fresh(cutoff, datetime.now()):
        board_flow = board_flow_snapshot()
        if board_flow:
            facts['sector.fund_flow'] = dict(asof=cutoff, status='KNOWN', value=board_flow,
                text='板块主力净流入与净流出榜，按报价源口径独立采样')
    if mode=='INTRADAY' and view.get('observe_live'):
        for key,fact in live_facts.items():
            if facts.get(key,{}).get('status')!='KNOWN':facts[key]=fact
        quote=facts.get('market.sse') or {}
        if quote.get('status')=='KNOWN' and quote.get('evidence_kind')=='QUOTE_SNAPSHOT':
            from strong_sector_watch_v1 import invoke
            try:
                price=quote['price']
                market=invoke(dict(mode='RULE_CONTEXT',as_of=datetime.now().isoformat(),
                    quote=dict(price=price,time=quote['evidence_time'])))
                facts['market.sse.pullback_environment']=dict(status='KNOWN' if market['state']!='UNKNOWN' else 'UNKNOWN',
                    asof=quote['evidence_time'],observed_at=quote['observed_at'],evidence_kind='QUOTE_SNAPSHOT',
                    bar_cutoff=cutoff,value=market,text='强势回调规则环境；不替代Pyxis风险幅度或计划票三选二')
            except (OSError,ValueError,KeyError,SubprocessError) as exc:
                facts['market.sse.pullback_environment']=dict(status='UNKNOWN',asof=cutoff,value=None,text=str(exc)[:120])
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
    if path.exists():
        cached=_cached_result(path)
        if cached is not None:return cached
    result=dict(status='AI_ATTEMPT_RESERVED',key=key,asof=cutoff,mode=mode,plan_id=plan_id,
        plan_hash=view.get('plan_hash'),computed_facts=facts,automatic_trading=False,prompt_contract=CONTRACT,
        owner_pid=os.getpid(),started_epoch=time.time(),board_flow=board_flow)
    if isinstance(json.loads(path.read_bytes()) if path.exists() else None,dict):
        # 收回续算：保留被收回占位的身份证据，可审计（不删除原证据）。
        prior=json.loads(path.read_bytes())
        if prior.get('request') or prior.get('transport_started'):
            result.update(status='AI_FAILED', reason='AI_DELIVERY_UNKNOWN_NO_AUTO_RETRY',
                          reclaimed_from=dict(owner_pid=prior.get('owner_pid'), key=prior.get('key')),
                          automatic_retry=False)
            save(path, result)
            return with_local_failure(result)
        result['reclaimed_from']=dict(owner_pid=prior.get('owner_pid'),
            started_epoch=prior.get('started_epoch'),key=prior.get('key'))
    save(path,result)
    try:
        body=dict(asof=cutoff,mode=mode,computed_facts=facts,conditions=CONDITIONS,
            plan_status=view.get('plan_status','UNKNOWN'),
            candidate_codes=[x['code'] for x in view.get('items',[])],
            holding_codes=None if positions is None else [x['code'] for x in positions],
            reference_policy=dict(known_ids=[k for k,f in facts.items() if f['status']=='KNOWN'],
                focus_allowed_ids=focus_allowed_ids(facts),
                minimum_total_references=1,
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
        parsed,notes=qualify_focus(parsed,facts);result['focus_qualification']=notes
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
    result=with_local_failure(result)
    save(path,result)
    save(runtime.directory()/'risk_outlook'/('latest_'+mode.lower()+'.json'),result)
    return result


def latest(mode='POSTCLOSE'):
    try:return with_local_failure(json.loads((runtime.directory()/'risk_outlook'/('latest_'+mode.lower()+'.json')).read_bytes()))
    except (OSError,ValueError):return dict(status='NOT_GENERATED')


def patrol_bulletin(plan, now):
    from joint_risk_bridge import completed_cutoff
    from tarot_monitor_v2 import fresh, outlook_slot

    cutoff = completed_cutoff(now)
    if not cutoff or not half_hour(cutoff) or not fresh(cutoff, now):
        return {}
    plan_id, plan_hash = plan.get('assistant_plan_id'), digest(plan)
    key, slot = outlook_slot(cutoff, plan_id, plan_hash)
    records = []
    for path in (slot, slot.with_name(key + '.local.json'), slot.with_name(key + '.waiting.json'),
                 slot.with_name('latest_intraday.json')):
        try:
            record = json.loads(path.read_bytes())
        except (OSError, ValueError):
            continue
        if (isinstance(record, dict) and record.get('key') == key
                and record.get('asof') == cutoff and record.get('plan_hash') == plan_hash):
            records.append(record)
    for record in records:
        if record.get('status') in ('READY', 'AI_FAILED', 'DATA_WAIT'):
            return record
    end = datetime.strptime(str(cutoff), '%Y%m%d%H%M%S')
    if (now - end).total_seconds() < 60:
        return {}
    if records:
        return records[0]
    return dict(status='DATA_WAIT', key=key, asof=cutoff, mode='INTRADAY',
                plan_id=plan_id, plan_hash=plan_hash, computed_facts={},
                reason='同截止巡检证据尚未到达')


def with_patrol_quotes(bulletin, now):
    """Use available local evidence independently of the risk/AI result."""
    facts = dict(bulletin.get('computed_facts') or {})
    cutoff = bulletin['asof']
    facts.update(_board_quote_facts(cutoff))
    if not all((facts.get(f'index.{name}.recent_five_bars') or {}).get('status') == 'KNOWN'
               for name in ('sse', 'szse', 'star50', 'cyb')):
        from plan_evidence import prepared_price_overlay
        from trade_facts_v1 import fact_table
        try:
            prices = prepared_price_overlay(cutoff)
            if prices:
                local = fact_table(cutoff, dict(context=dict(market_price_evidence=prices)))
                for key, fact in local.items():
                    if key.startswith(('index.', 'market.')) and fact.get('status') == 'KNOWN':
                        if (facts.get(key) or {}).get('status') != 'KNOWN':
                            facts[key] = fact
                from index_outlook_v1 import facts_for
                for name, evidence in prices.items():
                    if evidence.get('status') == 'PRICE_FACTS_ONLY' and evidence.get('asof') == cutoff:
                        facts.update(facts_for(name, evidence, cutoff))
                        facts.setdefault(f'index.{name}.map_input',
                                         dict(asof=cutoff, status='KNOWN', value=evidence))
        except (OSError, ValueError):
            pass
    missing = set()
    for name in ('sse', 'szse', 'star50', 'cyb'):
        market = facts.get('market.' + name) or {}
        mapping = facts.get(f'index.{name}.rebound_map') or {}
        if mapping.get('status') == 'KNOWN' and mapping.get('asof') == cutoff:
            continue
        if market.get('status') == 'KNOWN':
            if market.get('evidence_kind') != 'QUOTE_SNAPSHOT' and market.get('asof') == cutoff:
                continue
            if market.get('bar_cutoff') == cutoff:
                try:
                    stamp = datetime.strptime(str(market['asof']), '%Y%m%d%H%M%S')
                    if 0 <= (now - stamp).total_seconds() <= 120:
                        continue
                except (ValueError, KeyError):
                    pass
        missing.update(('market.' + name, 'market.' + name + '.amount_yi'))
    if missing:
        try:
            facts.update({key: fact for key, fact in live_index_facts(cutoff).items() if key in missing})
        except (OSError, ValueError):
            pass
    return dict(bulletin, computed_facts=facts)


def queue(cfg,view,cutoff,**kwargs):
    mode=kwargs.get('mode','POSTCLOSE')
    if mode=='INTRADAY':
        path=runtime.directory()/'risk_outlook'/f'{digest(dict(mode=mode,cutoff=cutoff,plan_id=kwargs.get("plan_id"),plan_hash=view.get("plan_hash")))}.json'
        if path.exists():return with_local_failure(json.loads(path.read_bytes()))
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
