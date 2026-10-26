"""Code-owned numeric statements. The LLM selects references, never rewrites prices."""
import math
import re

CONTRACT = 'CODE_FACT_REFERENCES_V4_CROSSDAY_TRIAL'


def fact_table(cutoff,risk,signal=None,selection=None):
    context=risk.get('context') or {};facts={}
    def add(key,text,value=None,status='KNOWN',**extra):
        if value is None and status=='KNOWN':value=text
        facts[key]=dict(asof=cutoff,text=text,value=value,status=status,**extra)
    def number(key,value,label,percent=False):
        if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):
            add(key,f'{label}：{value*100:.2f}%' if percent else f'{label}：{value:.4f}',value)
        else:add(key,f'{label}：未知',status='UNKNOWN')
    number('stock.price',context.get('stock_close'),'当前个股价')
    number('stock.volume_ratio',context.get('stock_volume_ratio'),'个股量比（原风险上下文口径，独立于指数量能）')
    for key in ['amp_232','amp_266']:
        number('risk.'+key,context.get(key),key+'未来120交易分钟不利下探幅度预测（非概率、非确定损失）',True)
    number('risk.reference_price',context.get('amp_reference_price',context.get('stock_close')),'Pyxis幅度预测参考价（按其原截止，不能换用新现价）')
    market=context.get('market_state') or {}
    number('market.sse',market.get('strength'),'沪指截至当前涨跌',True)
    number('market.cyb',market.get('cyb_strength'),'创业板截至当前涨跌',True)
    for group in ['board_states','leader_states']:
        for i,row in enumerate(context.get(group) or []):
            number(f'{group}.{i}.strength',row.get('strength'),f'{group}[{i}]观察强度',True)
            if 'coverage' in row:number(f'{group}.{i}.coverage',row['coverage'],f'{group}[{i}]覆盖',True)
            if 'return_change' in row:number(f'{group}.{i}.return_change',row['return_change'],f'{group}[{i}]最近区间涨幅变化',True)
    for name,mapping in (context.get('execution_index_maps') or {}).items():
        if mapping.get('asof')!=cutoff or not all(k in mapping for k in ('low_day','low','high_day','high','levels','current_close')):continue
        add('index.'+name+'.anchors',f"{name}执行地图锚点：{mapping['low_day']}低{mapping['low']:.4f}→{mapping['high_day']}高{mapping['high']:.4f}，独立于冻结Pyxis输入")
        levels=mapping['levels'];price=mapping['current_close']
        nearest=sorted([x for x in levels if x['price']<=price],key=lambda x:-x['price'])[:2]+sorted([x for x in levels if x['price']>price],key=lambda x:x['price'])[:2]
        for i,item in enumerate(nearest):
            add(f'index.{name}.{i}',f"{name} {item['kind']} 比例{item['ratio']} 档位{item['price']:.4f}；跌破后完整5分钟收回时刻{item['below_then_close_back_times']}；当前在其上方{item['last_close_above']}")
    for name,evidence in (context.get('market_price_evidence') or {}).items():
        if evidence.get('asof')!=cutoff or evidence.get('status')!='PRICE_FACTS_ONLY':
            add(f'index.{name}.repair','当前价格修复证据未知或截止不匹配',status='UNKNOWN');continue
        five=evidence['five'];fifteen=evidence['fifteen']
        number(f'index.{name}.rsi6',five.get('rsi6'),f'{name}五分钟RSI6')
        number(f'index.{name}.rsi15',five.get('rsi15'),f'{name}五分钟RSI15')
        add(f'index.{name}.rsi_relation',f'{name}五分钟RSI6高于RSI15：{five.get("rsi_above")}',five.get('rsi_above'),
            'KNOWN' if five.get('rsi_above') is not None else 'UNKNOWN')
        add(f'index.{name}.crosses',f'{name}五分钟上穿时刻：{five["cross_up_times"]}',five['cross_up_times'])
        add(f'index.{name}.stable',f'{name}日内低点后价格企稳形态：{evidence["price_stable_shape"]}；不含缩量资格',evidence['price_stable_shape'])
        reclaimed=[x for x in evidence['levels'] if x['reclaimed']]
        add(f'index.{name}.reclaim',f'{name}已刺破且末两根完整五分钟收在同档上方的档位：'+str([x['price'] for x in reclaimed]),
            [x['price'] for x in reclaimed])
        add(f'index.{name}.confirmation15',f'{name}十五分钟辅助确认截止{fifteen["evidence_time"]}；RSI6高于RSI15：{fifteen["rsi_above"]}',
            fifteen.get('rsi_above'),'KNOWN' if fifteen.get('rsi_above') is not None else 'UNKNOWN',
            evidence_time=fifteen['evidence_time'])
        add(f'index.{name}.repair',evidence['summary'])
        add(f'index.{name}.volume','指数成交量口径未核销；缩量与强弱修复资格未知',status='UNKNOWN')
        bands=evidence.get('resonance_bands') or []
        price=evidence.get('last_close')
        nearest=sorted(bands,key=lambda b:abs(b['node']-price))[:6] if isinstance(price,(int,float)) else []
        for i,band in enumerate(bands):
            if band not in nearest:continue
            add(f'index.{name}.band.{i}',f'{name}黄金带 {band["lower"]:.4f}—{band["upper"]:.4f}；来源波段{band["source_waves"]}，高点组{band["peak_groups"]}（非独立票数）',band)
        trial=evidence.get('research_trial') or {}
        if trial.get('asof')==cutoff and trial.get('status')=='EXPERIMENTAL_MODEL_OUTPUT':
            for field in ['raw_state','protected_state']:
                add(f'trial.{name}.{field}',f'{name}试运行39维模型{field}：{trial[field]}；量能假设未核销，原验收失败，仅供综合参考',trial[field],status='TRIAL')
    add('corvus.unknown','信息层未知，不等于没有利空',status='UNKNOWN')
    add('risk.partial','风险上下文按已披露来源覆盖计算；缺失不补造',status='UNKNOWN')
    add('risk.qualification','幅度预测值可用；不是保证的最大损失，也不代表研究修复状态已验收',status='UNKNOWN')
    add('execution.timing','确认后才能讨论下一有效交易时段的人工介入；不假定同根收盘价成交')
    if selection is not None:
        number('selection.score',selection.get('selection_score'),'前日冻结Tarot候选评分（非概率）')
    for key,value in (context.get('market_repair_research') or {}).items():
        add('research.'+key,str(value),status='RESEARCH')
    if signal:
        if signal.get('policy')=='GOLDEN_RESONANCE_CROSSDAY_V4':
            band=signal.get('band') or {};div=signal.get('divergence') or {}
            number('signal.band_lower',band.get('lower'),'本次原始共振带下沿')
            number('signal.band_upper',band.get('upper'),'本次原始共振带上沿（破位收回须守住此处）')
            number('signal.anchor_price',div.get('anchor_low'),'前交易日最低价首现根价格')
            number('signal.anchor_rsi6',div.get('anchor_rsi'),'跨日锚点RSI6')
            number('signal.low_rsi6',div.get('event_rsi'),'本次低点根收盘RSI6')
            add('signal.two_crosses','本次触位及最新新低之后上穿时刻：'+str(signal.get('event_cross_times',[])))
        number('signal.node',signal.get('node'),'本次观察黄金位')
        number('signal.low',signal.get('session_low'),'截至当前已观察最低价')
        low,node=signal.get('session_low'),signal.get('node')
        if isinstance(low,(int,float)) and isinstance(node,(int,float)):
            add('signal.node_relation','最低价'+('低于' if low<node else '等于' if low==node else '高于')+'本次观察黄金位')
        add('signal.state','价格条件状态：'+str(signal.get('status')),signal.get('status'))
        add('signal.branch','已核验路径：'+str(signal.get('branch')),signal.get('branch'))
        add('signal.pierced_levels','此前刺破档位：'+str(signal.get('pierced_levels',[])),signal.get('pierced_levels',[]))
        add('signal.requirement',('未破企稳路径已确认，不额外强制收回更高失守档位' if signal.get('branch')=='STOP_ABOVE' else
            '收回路径须两根完整五分钟收在本次同一原带上沿之上' if signal.get('branch')=='RECLAIM' and signal.get('policy')=='GOLDEN_RESONANCE_CROSSDAY_V4' else
            '收回路径须两根完整五分钟收在本次同一档位之上' if signal.get('branch')=='RECLAIM' else '沿既定规则核验，不由AI重写触发条件'))
        if signal.get('trigger_time'):add('signal.time','价格确认发生时刻：'+str(signal['trigger_time']))
        bars=signal.get('confirmation_bars') or []
        if len(bars)==3 and all(b.get('volume') is not None for b in bars):
            add('signal.volume',f"下探根成交量{bars[0]['volume']:.0f}，随后两根均量{(bars[1]['volume']+bars[2]['volume'])/2:.0f}；仅描述量收敛，不推断资金流")
        rsi=signal.get('rsi') or {}
        number('rsi.fast',rsi.get('rsi6'),'个股5分钟RSI6')
        number('rsi.slow',rsi.get('rsi15'),'个股5分钟RSI15')
        add('rsi.crosses','本次前缀已确认上穿时刻：'+str(rsi.get('cross_up_times',[])))
        add('rsi.relation','当前RSI6高于RSI15：'+str(rsi.get('above'))+'；不等同底背离',rsi.get('above'),
            'KNOWN' if rsi.get('above') is not None else 'UNKNOWN')
        divergence=signal.get('divergence')
        add('signal.divergence','既定口径底背离：'+('未知' if divergence is None else str(bool(divergence))),
            None if divergence is None else bool(divergence),'UNKNOWN' if divergence is None else 'KNOWN')
    risk_time=risk.get('asof_ts',cutoff)
    for key,fact in facts.items():
        if key.startswith(('risk.amp','risk.reference_price','market.','board_states.','leader_states.','stock.price','stock.volume_ratio')):
            observed=context.get('stock_price_asof',risk_time) if key=='stock.price' else risk_time
            fact['evidence_time']=observed
            if str(observed)!=str(cutoff):fact['text']=f'证据截止{observed}（非当前新推理）：'+fact['text']
    return facts


PROMPT=('你是Stargazer综合判断层。只依据computed_facts选择同截止证据，不自行计算或改写价格、时间、RSI关系。'
    '输出JSON：asof（原样整数）、decision（ENTRY仅BUY/DEFER；HOLDING仅HOLD/EXIT/DEFER）、'
    'supporting_ids、opposing_ids、unknown_ids（三个字符串数组，每组最多八项，只能引用computed_facts中的ID，互不重复，至少引用一条），'
    'observed_rule_status与observed_branch原样复制signal.state及signal.branch的value；无signal时两项均null。'
    'rationale为简短中文权衡，不复述任何数字、价格、代码或自创门槛；数值由程序展示，可使用RSI6/RSI15名称。'
    '用不同事实分别表达支持和反对；同一ID绝不可跨数组重复。按代码已核验的组合条件解释，不追加或省略硬门；未破企稳与破位收回是二选一的不同位置路径。'
    'RSI高不自动否决、低不自动买；交叉不是底背离。Pyxis幅度不是概率或确定损失，不能与Tarot分数相减。'
    'status为UNKNOWN或RESEARCH只能入unknown_ids，不能作正反；KNOWN且value非null不归未知，预测数值与效力未知分开。'
    '待证量能或信息未知可以降低把握，不自动否决，也不追加为必需入场条件；既定风险齐全门由代码判断。'
    '以沪指为主要环境证据、创业板为补充，五分钟判断当刻、十五分钟辅助确认，允许涨跌幅仍负但价格在修复。'
    '指数RSI高不等于个股追高；量能未核销不称缩量/放量确认，不把价格修复称为已确认强修复。'
    'TRIAL是用户授权试运行的模型意见，可以有限引用支持或反对，也可因未验收保留于unknown_ids；必须与KNOWN事实分开，不可称已验收事实或覆盖已知风险。'
    '只有指数量能未核销，不可把KNOWN的个股量比也说成未核销。rationale仅解释权衡，具体事实与未知由引用渲染，勿扩大未知范围。'
    '收回路径的曾跌破、逐档失守记录本身不是新的否决门；不得把同一收回事实又作反对。'
    '个股量比低于一不自动作利空：缩量止跌与拉升缺量须按阶段区分，不额外要求放量反转阳线。'
    '已由代码确认的STOP_ABOVE不要再强加RECLAIM或底背离；高档未收回可以讨论，不改变既定规则。'
    '信息未知不等于无利空，原始缺失保留。BUY仅供人工考虑，无自动交易。')


def validate_and_render(advice,facts,cutoff,positioned=False):
    allowed={'HOLD','EXIT','DEFER'} if positioned else {'BUY','DEFER'}
    if (not isinstance(advice,dict) or type(advice.get('asof')) is not int or advice['asof']!=cutoff or
            advice.get('decision') not in allowed):raise ValueError('AI_FACT_CONTRACT_INVALID')
    if (advice.get('observed_rule_status')!=facts.get('signal.state',{}).get('value') or
            advice.get('observed_branch')!=facts.get('signal.branch',{}).get('value') or
            'observed_rule_status' not in advice or 'observed_branch' not in advice):
        raise ValueError('AI_RULE_PATH_MISMATCH')
    rationale=advice.get('rationale')
    if not isinstance(rationale,str) or not rationale.strip() or len(rationale)>600:
        raise ValueError('AI_RATIONALE_INVALID')
    if re.search(r'\d',re.sub(r'RSI\s*(?:6|15)','RSI',rationale,flags=re.I)):
        raise ValueError('AI_RATIONALE_NUMERIC_REWRITE')
    if facts.get('stock.volume_ratio',{}).get('status')=='KNOWN' and re.search(r'指数及个股量能.{0,6}(未知|未核销)|个股量比.{0,6}(未知|未核销)',rationale):
        raise ValueError('AI_VOLUME_SCOPE_MISMATCH')
    groups=[];used=set()
    for field,title in [('supporting_ids','支持'),('opposing_ids','反对'),('unknown_ids','待核')]:
        refs=advice.get(field)
        if not isinstance(refs,list) or len(refs)>8 or len(refs)!=len(set(x for x in refs if isinstance(x,str))):
            raise ValueError('AI_FACT_REFERENCE_INVALID')
        if any(key not in facts or facts[key]['asof']!=cutoff or key in used for key in refs):
            raise ValueError('AI_FACT_REFERENCE_INVALID')
        used.update(refs)
        for key in refs:
            fact=facts[key]
            if fact.get('status') in ('UNKNOWN','RESEARCH') and field!='unknown_ids':
                raise ValueError('AI_UNKNOWN_AS_CONFIRMED')
            if fact.get('status')=='KNOWN' and fact.get('value') is not None and field=='unknown_ids':
                raise ValueError('AI_KNOWN_VALUE_AS_UNKNOWN')
        if refs:groups.append(title+'：'+'；'.join(facts[key]['text'] for key in refs))
    if not used:raise ValueError('AI_FACT_REFERENCE_EMPTY')
    label={'BUY':'建议人工考虑介入','DEFER':'暂缓','HOLD':'继续持有观察','EXIT':'建议人工考虑退出'}[advice['decision']]
    return label+'。'+'。'.join(groups)+'。AI权衡：'+rationale.strip()
