"""Presentation only. No market reads, trading rules, AI calls or notifications."""
import math

CONTRACT='PUSH_LAYOUT_V1'


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def price(value):
    return ('%.4f' % value).rstrip('0').rstrip('.') if finite(value) and value>0 else '未知'


def percent(value):
    # All return/distance inputs are fractions, never pre-multiplied percentages.
    return '%+.1f%%' % (value*100) if finite(value) else '未知'


def stamp(value):
    text=str(value)
    if len(text)!=14 or not text.isdigit():raise ValueError('PUSH_ASOF_REQUIRED')
    from datetime import datetime
    return datetime.strptime(text,'%Y%m%d%H%M%S').strftime('%H:%M')


def identity(row):
    return ' '.join(str(row.get(k) or '') for k in ('name','code')).strip() or '对象未知'


def advice(packet):
    # Caller must supply a suggestion already checked by the existing risk/AI pipeline.
    if packet.get('advice_status')=='ADVICE_READY' and packet.get('advice_verified') is True:
        return ' '.join((packet.get('advice_text') or '已完成研判，建议文本未提供；请人工查看。').splitlines())
    return '风险模型与DeepSeek研判尚未完成；请观察并等待。'


def level_text(row):
    label=row.get('level_label') or '黄金位'
    lower,upper=row.get('level_lower'),row.get('level_upper')
    location=price(lower) if lower==upper or upper is None else price(lower)+'—'+price(upper)
    return label+location


def progress(row):
    labels={'repair':'位置修复','momentum':'动能','oversold':'严重超卖'}
    votes=row.get('votes') or {}
    if not votes:return row.get('progress_text') or '完整5分钟组合待核验'
    return ' '.join(label+('✓' if votes.get(k) is True else '✗' if votes.get(k) is False else '待核验')
                    for k,label in labels.items())+' | 三选'+str(row.get('required_votes',2))


def plan_line(row):
    states={'WAIT_POSITION':'⏳未到','WAIT_SIGNAL':'🚫观察','RULE_TRIGGERED':'⚡组合通过',
            'RECLAIM_PENDING':'⚡收回中','BROKEN':'🚫破位','UNKNOWN':'待核验'}
    return '%s | %s | %s 距离%s | %s' % (identity(row),price(row.get('price')),
        level_text(row),percent(row.get('distance')),
        (row.get('status_text') or states.get(row.get('status'),'待核验')))


def holdings_line(row):
    line = '%s | 成本%s 现价%s 浮盈%s | 止盈%s 止损%s | %s' % (
        identity(row),price(row.get('cost')),price(row.get('price')),percent(row.get('pnl')),
        price(row.get('take_profit')),price(row.get('stop_loss')),row.get('state_text') or '待研判')
    if row.get('price') is None and row.get('quote_origin'):
        line += ' | 报价来源：' + str(row['quote_origin'])
    return line


def half_hour(packet,previous=None):
    cutoff=packet['asof'];time=stamp(cutoff)
    rows=packet.get('plans',[])
    # A new date/plan is a new population; suppress unchanged state only within it.
    same=previous and previous.get('date')==str(cutoff)[:8] and previous.get('plan_id')==packet.get('plan_id')
    old=(previous or {}).get('states',{}) if same else {}
    states={r['code']:{k:r.get(k) for k in ('status','votes','required_votes','level_lower','level_upper')} for r in rows}
    changes=[r for r in rows if old.get(r['code'])!=states[r['code']]]
    header='【巡检 '+time+'】指标截止 '+time
    if packet.get('quote_asof'):header+=' | 报价 '+stamp(packet['quote_asof'])
    lines=[header,'','▍计划票']
    lines.extend([plan_line(r) for r in changes] or ['本时段无状态变化' if rows else '无当日计划票'])
    lines+=['','▍大盘闸门']
    market=[]
    for key,label in [('sse','沪指'),('cyb','创业板')]:
        r=packet.get(key) or {}
        market.append(label+' '+price(r.get('price'))+'('+percent(r.get('return'))+') '+(r.get('state_text') or '状态待核验'))
    lines.append(' | '.join(market)+' | 量: '+(packet.get('volume_text') or '未知'))
    lines.append('背离/滞涨: '+(packet.get('divergence_text') or '待核验')+' | 闸门: '+(packet.get('gate_text') or '待定义'))
    if packet.get('pullback_environment_text'):lines.append('强势回调环境｜'+packet['pullback_environment_text'])
    if packet.get('lane_status_text'):
        lines.append('环节状态: '+packet['lane_status_text'])
    if packet.get('ai_failure_text'):
        lines.append('DeepSeek 失败说明: '+packet['ai_failure_text'])
    if packet.get('volume_scope_text'):
        lines.append('成交额/量比口径: '+packet['volume_scope_text'])
    lines.append('风险模型的分析：'+(packet.get('risk_analysis') or '未提供同截止研判'))
    lines+=['','▍板块扫描']
    lines.extend([str(r['name'])+' | 龙头:'+(r.get('leader_text') or '待核验')+' | 异动:'+(r.get('change_text') or '待核验')
                  for r in packet.get('boards',[])] or ['无已配置板块'])
    lines+=['','▍盘中异动（本时段新触发）']
    events=[str(r['text']) for r in packet.get('new_events',[])]
    scan_text=str(packet.get('scan_status_text') or '')
    if events:
        lines.extend(events)
        lines.append(scan_text or '扫描状态待核验')
    else:
        # 只在确认「本时段已完成扫描」后才允许说无新增异动；
        # 未扫描/零报价/状态陈旧时必须如实显示未完成及具体原因。
        lines.append(scan_text + '；本时段无新增异动' if ('已完成扫描' in scan_text)
                     else (scan_text if scan_text else '本时段未完成扫描（扫描状态未提供）'))
    lines+=['','▍持仓']
    holdings=packet.get('holdings')
    lines.extend([holdings_line(r) for r in holdings] if holdings else ['无已登记持仓' if holdings==[] else '实际持仓未提供'])
    lines+=['','▍下一动作',advice(packet)]
    return dict(contract=CONTRACT,title='巡检 '+time,body='\n'.join(lines),
                next_state=dict(date=str(cutoff)[:8],plan_id=packet.get('plan_id'),states=states))


def notice(kind,packet):
    titles={'PLAN_TOUCH':'计划票 · 到位','PLAN_TRIGGER':'计划票 · 扳机','PLAN_BREAK':'计划票 · 破位',
        'STRONG_STOCK':'强势股异动','DELETE':'强势股 · 删除确认','LANDMARK':'强势股 · 路标破位',
        'HOLDING_STOP':'持仓警报 · 止损触发','HOLDING_PROFIT':'持仓提示 · 止盈到达',
        'HOLDING_PLUNGE':'持仓警报 · 异动下杀','HOLDING_TIME':'持仓提示 · 时间止损',
        'HOLDING_MARKET':'持仓警报 · 大盘联动'}
    if kind not in titles:raise ValueError('UNKNOWN_PUSH_KIND')
    if kind in ('STRONG_STOCK','PLAN_TRIGGER'):
        from push_compact_v1 import strong,trigger
        return strong(packet) if kind=='STRONG_STOCK' else trigger(packet)
    time=stamp(packet['asof']);title=titles[kind]
    # Undefined priority codes are never guessed from a stock's return or rank.
    priority=packet.get('priority_label')
    if kind=='STRONG_STOCK' and priority:title+=' · '+priority
    lines=['【'+title+'】'+time,identity(packet)]
    if kind=='STRONG_STOCK':
        lines[-1]+=' | '+(packet.get('board') or '板块待核验')
        lines+=['现价 '+price(packet.get('price'))+' 跌幅 '+percent(packet.get('return')),
            '触及: '+level_text(packet)+' ('+percent(packet.get('distance'))+')',
            '波段: '+price(packet.get('wave_low'))+'→'+price(packet.get('wave_high'))+' ('+percent(packet.get('wave_return'))+')',
            '低点对照（'+(packet.get('indicator_basis') or '周期/截止待核验')+'）:',
            '昨日 '+str(packet.get('previous_low_time') or '未知')+' 价格'+price(packet.get('previous_low'))+
                ' DIF '+str(packet.get('previous_dif') if finite(packet.get('previous_dif')) else '未知')+
                ' RSI6 '+str(packet.get('previous_rsi6') if finite(packet.get('previous_rsi6')) else '未知'),
            '今日 '+str(packet.get('current_low_time') or '未知')+' 价格'+price(packet.get('current_low'))+
                ' DIF '+str(packet.get('current_dif') if finite(packet.get('current_dif')) else '未知')+
                ' RSI6 '+str(packet.get('current_rsi6') if finite(packet.get('current_rsi6')) else '未知'),
            '量能: '+(packet.get('volume_text') or '待核验'),'早盘标记: '+(packet.get('morning_mark') or '未记录')]
        if 'period_pairs' in packet:
            start=next(i for i,s in enumerate(lines) if s.startswith('低点对照'))
            lines=lines[:start]
            for period in ['5','15']:
                pair=(packet.get('period_pairs') or {}).get(period) or {}
                lines.append(period+'分钟低点对照：'+('待核验 '+str(pair.get('reason') or '') if not pair.get('current') else '实时试算、未收盘' if pair.get('provisional') else '完整K线'))
                for key,label in [('previous','昨日'),('current','今日')]:
                    r=pair.get(key) or {}
                    def value(k):return ('%.4f'%r[k]).rstrip('0').rstrip('.') if finite(r.get(k)) else '未知'
                    lines.append(label+' '+str(r.get('time') or '未知')+' 低点'+price(r.get('price'))+
                        ' | DIF '+value('dif')+' DEA '+value('dea')+' | RSI6 '+value('rsi6')+' RSI15 '+value('rsi15'))
                ratio=pair.get('low_volume_ratio')
                lines.append('低点根量：昨日'+str((pair.get('previous') or {}).get('volume','未知'))+
                    ' / 今日'+str((pair.get('current') or {}).get('volume','未知'))+
                    (' | 今日/昨日 %.2f倍'%ratio if finite(ratio) else ' | 量比待核验')+('（当前根未收盘，不据此确认缩量）' if pair.get('provisional') else ''))
                def check(k):return '✓' if pair.get(k) is True else '✗' if pair.get(k) is False else '待核验'
                lines.append('低点对照：DIF背离'+check('dif_divergence')+' | RSI背离'+check('rsi_divergence'))
            lines.append('早盘标记: '+(packet.get('morning_mark') or '未记录'))
        reference=packet.get('relative_turnover') or {}
        if reference.get('status')=='KNOWN' and finite(reference.get('ratio')):
            lines.append('相对换手：%.2f倍（20日同分钟，连续竞价参考）｜截至%s'%(reference['ratio'],stamp(reference['asof'])))
        else:
            reason=reference.get('reason') or 'WAIT_POSTCLOSE_REFERENCE'
            label='历史分钟量未对齐' if reason=='HISTORY_VOLUME_MISMATCH' else '分钟范围未齐' if reason in ('HISTORY_MINUTE_GAP','CURRENT_MINUTE_GAP') else '参考暂未核实'
            lines.append('相对换手：'+label+'；不影响到位提醒')
    elif kind.startswith('PLAN_'):
        lines+=['现价 '+price(packet.get('price'))+' | '+level_text(packet), '验收进度: '+progress(packet)]
    elif kind.startswith('HOLDING_'):
        lines+=[holdings_line(packet),'触发: '+(packet.get('trigger_text') or '证据待核验')]
    else:lines+=['失效路标: '+level_text(packet),'确认: '+(packet.get('trigger_text') or '收盘确认待核验')]
    if kind in ('STRONG_STOCK','LANDMARK','DELETE') and packet.get('advice_verified') is not True:
        lines+=['仅价格、指标与规则事实，请人工判断；未生成买卖建议。']
    else:lines+=['建议: '+advice(packet)]
    lines+=['程序未下单。']
    return dict(contract=CONTRACT,title=title,body='\n'.join(lines))
