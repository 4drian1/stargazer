"""Adapt verified runtime facts to the user's layout; never creates trading advice."""
import copy
from push_layout_v1 import half_hour,notice,finite


def _risk_lane_wait(plan=None, cutoff=None):
    """Classify why plan-stock risk rows are absent: 数据失败 / 计算失败 / 正常未知.
    Reads the persisted risk lane state; returns None when the lane is healthy.
    Price quotes are composed independently and are never gated on this."""
    try:
        import json as _json
        import postclose_risk_runtime
        latest = _json.loads((postclose_risk_runtime.directory()/'tarot_v2'/'latest.json').read_bytes())
    except (OSError, ValueError, ImportError):
        return '等待当日风险任务结果' if cutoff is not None else None
    # Old failures belong to their own release/day, not the current patrol.
    if cutoff is not None:
        from tarot_monitor_v2 import _flow_version
        if latest.get('flow_version') != _flow_version():
            return '等待当前版本风险结果（旧版本记录已保留）'
        recorded = latest.get('cutoff')
        if type(recorded) is not int or recorded // 1000000 != cutoff // 1000000:
            return '等待当日风险结果（历史记录已保留）'
        if recorded > cutoff:
            return '等待同截止风险结果'
        if recorded < cutoff and latest.get('status') != 'EVIDENCE_FAILED':
            return '等待本截止风险结果（较早结果已保留）'
        if plan is not None:
            from golden_rules import digest
            if latest.get('plan_id') != plan.get('assistant_plan_id') or latest.get('plan_hash') != digest(plan):
                return '等待当前计划风险结果'
    status = latest.get('status')
    reason = str(latest.get('reason') or '')
    if status == 'SOURCE_PENDING':
        return '供给未完成：完整板块实时上下文未接入；个股Pyxis未计算'
    if status == 'EVIDENCE_FAILED':
        if 'TimeoutExpired' in reason or 'RISK_WORKER_TIMEOUT' in reason:
            return '计算超时：个股Pyxis任务超出时限；价格与规则独立运行'
        if any(tag in reason for tag in ('REALTIME_', 'LIVE_', 'TDX_',
                                         'INDEX_MIRROR', 'SUPPLY_PENDING', 'SOURCE_LATE')):
            return '数据失败：' + reason[:90]
        return '计算失败：' + (reason[:90] or status)
    if status in ('WAITING_SOURCE', 'SOURCE_LATE'):
        return '正常未知：' + reason[:60]
    if cutoff is not None and status != 'MONITORING':
        return '等待风险任务：' + str(status or '状态未记录')
    return None


def gate(states):
    a,b=states.get('sse'),states.get('cyb')
    missing=[label for label,state in (('沪指',a),('创业板',b))
             if state not in ('BROKEN','REPAIRED','PENDING')]
    if missing:return '⚪灰（关键位置证据不齐：缺'+'、'.join(missing)+'关键位置状态）'
    if a==b=='BROKEN':return '🔴红（双指数失守未修复；非自动卖出指令）'
    if a=='REPAIRED' and b!='BROKEN':return '🟢绿（沪指修复；不等于低风险）'
    return '🟡黄（单边失守或修复待确认）'


def review_notice(review,signal,row,quote):
    from trade_facts_v1 import CONTRACT
    from pentacles_monitor import rule_condition_text,combination_passed
    risk=review.get('risk') or {};ctx=risk.get('context') or {}
    combination=signal.get('combination') or {}
    band=signal.get('band') or {}
    regions=review.get('observed_regions') or []
    price=ctx.get('stock_close')
    if not finite(price):price=quote.get('price')
    if not band and regions:
        band=min(regions,key=lambda b:abs((b['lower']+b['upper'])/2-price)) if finite(price) else regions[0]
    if not band:band=dict(lower=signal.get('node'),upper=signal.get('node'))
    is_holding=review.get('kind')=='HOLDING' or review.get('position') is not None
    decision=review.get('decision')
    kind='HOLDING_STOP' if is_holding and decision=='EXIT' else 'HOLDING_PLUNGE' if is_holding else 'PLAN_TRIGGER' if signal.get('status')=='RULE_TRIGGERED' else 'PLAN_TOUCH'
    reason=review.get('summary_reason') or review.get('reason') or ''
    if is_holding and '时间' in reason or is_holding and '持有期' in reason:kind='HOLDING_TIME'
    if is_holding and '止盈' in reason:kind='HOLDING_PROFIT'
    if not is_holding and review.get('event_type')=='PLAN_BREAK':kind='PLAN_BREAK'
    verified=(review.get('status')=='ADVICE_READY' and review.get('fact_contract')==CONTRACT
              and isinstance(review.get('advice_references'),dict) and risk.get('status')=='AVAILABLE')
    actions={'BUY':'DeepSeek建议买入','DEFER':'DeepSeek暂不建议买入','EXIT':'DeepSeek建议退出','HOLD':'DeepSeek建议持有'}
    text=actions.get(decision,'DeepSeek建议待查看')+'；'+reason
    if review.get('risk_reasons'):text+='；风险：'+'；'.join(review['risk_reasons'][:3])
    packet=dict(asof=review['cutoff'],name=row.get('name') or ctx.get('stock_name'),code=review['code'],
        price=price,level_lower=band.get('lower'),level_upper=band.get('upper'),level_label='黄金观察区',
        votes=combination.get('votes'),required_votes=combination.get('required',2),progress_text=rule_condition_text(signal),
        advice_status=review.get('status') if verified else review.get('waiting_status',review.get('status')),advice_verified=verified,advice_text=text,
        decision=decision,summary_reason=review.get('summary_reason') or (reason if verified else None),
        wait_reason=reason,risk_reasons=review.get('risk_reasons') or [],
        pending_facts=[(review.get('computed_facts') or {})[key]['text'] for key in (review.get('advice_references') or {}).get('unknown_ids',[]) if (review.get('computed_facts') or {}).get(key,{}).get('status') in ('UNKNOWN','RESEARCH','TRIAL')],
        provisional=bool(signal.get('provisional')),period_pairs=(signal.get('comparisons') or {}).get('periods'),
        cost=(review.get('position') or {}).get('entry'),trigger_text=reason,
        stop_loss=((review.get('position') or {}).get('tarot_execution') or {}).get('stop_price'))
    if finite(price) and finite(packet.get('cost')) and packet['cost']>0:packet['pnl']=price/packet['cost']-1
    rendered=notice(kind,packet)
    if signal.get('provisional'):
        rendered['title']+=' · 实时试算'
        if kind!='PLAN_TRIGGER':rendered['body']='实时试算、未收盘；指标可变。\n'+rendered['body']
    if regions:
        rendered['body']=rendered['body'].replace('验收进度:', '本次到位：'+'、'.join('%.2f—%.2f'%(b['lower'],b['upper']) for b in regions)+'\n验收进度:')
    if reason and not verified:
        rendered['body']=rendered['body'].replace('建议:', '研判/等待说明（非已核验买卖建议）：'+reason+'\n建议:')
    if verified and decision=='BUY':rendered['title']+=' · 建议买入'
    elif verified and decision=='DEFER':rendered['title']+=' · 暂缓'
    # Keep technical confirmation distinct from a deferred/failed AI opinion.
    if signal.get('status')=='RULE_TRIGGERED':
        label=('三选二已通过' if combination.get('required',2)==2 else '自定义组合已通过') if combination_passed(signal) else '原规则已通过'
        rendered['body']=rendered['body'].replace('验收进度:',label+'；验收进度:')
    return rendered


def report(bulletin,plan,result,quotes,positions,previous=None,sector_events=None,observed_at=None):
    facts=bulletin.get('computed_facts') or {};asof=bulletin['asof']
    analysis=bulletin.get('analysis') or {};sections=analysis.get('sections') or {}
    def value(k):
        f=facts.get(k) or {}
        return f.get('value') if f.get('status')=='KNOWN' else None
    packet=dict(asof=asof,plan_id=plan.get('assistant_plan_id'),plans=[],boards=[],holdings=None if positions is None else [],new_events=sector_events or [])
    quote_times=[int(q['time']) for q in quotes.values() if str(q.get('time','')).isdigit() and str(q['time'])[:8]==str(asof)[:8]]
    quote_times += [f['asof'] for f in facts.values() if f.get('evidence_kind')=='QUOTE_SNAPSHOT' and type(f.get('asof')) is int and f['asof']//1000000==asof//1000000]
    if quote_times:packet['quote_asof']=max(quote_times)
    wait_text = _risk_lane_wait(plan, asof)
    for row in plan.get('rows',[]) if int(plan.get('monitor_date',0))==asof//1000000 else []:
        code=row['code'];signal=(result.get('rows') or {}).get(code) or {};q=quotes.get(code) or {}
        p=q.get('price');node=signal.get('node') or signal.get('next_lower_node')
        if not finite(node):
            levels=(row.get('entry_rule') or {}).get('levels') or []
            node=min(levels,key=lambda n:abs(n-p)) if levels and finite(p) else None
        packet['plans'].append(dict(code=code,name=row.get('name'),price=p,level_lower=node,level_upper=node,
            level_label='黄金位',distance=(p-node)/node if finite(p) and finite(node) and node>0 else None,
            status=signal.get('status','UNKNOWN'),votes=(signal.get('combination') or {}).get('votes'),
            status_text=None if signal else (wait_text or '正常未知：同截止分钟与风险证据未就绪')))
    states={}
    for name in ['sse','cyb']:
        quote=facts.get('market.'+name) or {};price=value('market.'+name+'.price')
        # Quote text is not parsed to invent a numeric price. Typed value must exist.
        if quote.get('evidence_kind')=='QUOTE_SNAPSHOT' or not finite(price):price=quote.get('price')
        packet[name]=dict(price=price,return_value=None,state_text=value('index.'+name+'.repair'))
        packet[name]['return']=value('market.'+name)
        states[name]=value('index.'+name+'.critical_state')
    packet['gate_text']=gate(states)
    pullback=value('market.sse.pullback_environment')
    if isinstance(pullback,dict):
        from strong_pullback_policy_v1 import market_line
        packet['pullback_environment_text']=market_line(pullback)
    overlay_fact=facts.get('index.price_overlay')
    if overlay_fact and overlay_fact.get('status')=='UNKNOWN':
        packet['gate_text']+='；'+overlay_fact['text']
    packet['risk_analysis']=sections.get('market') or '本次AI研判尚未完成；保留本地事实'
    from pentacles_monitor import combination_passed
    wait_text = _risk_lane_wait(plan, asof)
    signals = result.get('rows') or {}
    if signals:
        passed = any(combination_passed(s) for s in signals.values())
        rules_text = '三选二通过' if passed else '已运行未通过'
    else:
        rules_text = '未运行'
    if bulletin.get('status') == 'READY':
        ai_text = '已完成'
    elif bulletin.get('status') in ('WAITING_SOURCE', 'SOURCE_LATE', 'DATA_WAIT'):
        ai_text = '等待数据：' + str(bulletin.get('status'))
    else:
        ai_text = '未完成'
    packet['lane_status_text'] = ('规则：' + rules_text + '｜Pyxis：' +
                                  (wait_text or '已供给') + '｜DeepSeek：' + ai_text)
    amounts = []
    for name, label in (('sse', '沪指'), ('cyb', '创业板')):
        fact = facts.get('market.'+name+'.amount_yi') or {}
        amount = value('market.'+name+'.amount_yi')
        if finite(amount) and not isinstance(amount, bool) and amount >= 0 and fact.get('evidence_kind') == 'QUOTE_SNAPSHOT':
            amounts.append(f'{label}{amount:.1f}亿元（报价{fact.get("evidence_time")}累计成交额）')
    packet['volume_text'] = ('；'.join(amounts) + '；' if amounts else '') + (
        '指数成交量及同刻量比待核验：历史量额口径未通过校验；不据此判断放量缩量')
    divergence = []
    for name, label in (('sse', '沪指'), ('cyb', '创业板')):
        text = value('index.'+name+'.divergence_summary')
        if text:
            divergence.append(label + text)
    packet['divergence_text'] = value('market.divergence_stall') or (
        '；'.join(divergence) + '；滞涨判定待核验' if divergence else
        'RSI/DIF跨日低点背离证据待核验；滞涨判定待核验')
    indicator_lines = []
    for name, label in (('sse', '沪指'), ('cyb', '创业板')):
        cells = []
        for key in ('rsi6', 'rsi15'):
            fact = facts.get('index.' + name + '.' + key) or {}
            number = value('index.' + name + '.' + key)
            cells.append(key.upper() + (' %.2f' % number if finite(number) else ' 未知'))
        fact = facts.get('index.' + name + '.rsi6') or {}
        indicator_lines.append(label + '5分钟 ' + '/'.join(cells) +
                               '（指标截止 ' + str(fact.get('evidence_time') or fact.get('asof') or asof) + '）')
    packet['divergence_text'] += '；' + '；'.join(indicator_lines)
    for board in plan.get('boards',[]):
        observed=value('sector.watch.'+board['name']) or {}
        leader=observed.get('leader') or {}
        leader_text=('池内领涨候选 '+str(leader.get('name') or leader.get('code'))+' %+.1f%%'%(leader['return_value']*100)) if leader else '待核验'
        changes=('池内%d只跌超4%%；报价%d/%d'%(observed['drop4_count'],observed['valid_count'],observed['pool_count'])) if observed else '待核验'
        packet['boards'].append(dict(name=board['name'],leader_text=leader_text,change_text=changes))
    for pos in positions or []:
        code=pos['code'];price=value(code+'.stock.price');cost=pos.get('entry');policy=pos.get('tarot_execution') or {}
        packet['holdings'].append(dict(code=code,name=pos.get('name'),price=price,cost=cost,
            pnl=price/cost-1 if finite(price) and finite(cost) and cost>0 else None,
            stop_loss=policy.get('stop_price'),take_profit=pos.get('take_profit'),state_text='；'.join(sections.get('holdings','').splitlines()) or '待研判'))
    packet.update(advice_status='ADVICE_READY' if bulletin.get('status')=='READY' else None,
        advice_verified=bulletin.get('status')=='READY' and bool(bulletin.get('analysis')),
        advice_text=sections.get('plan') or sections.get('market'))
    if sector_events is None:
        import json
        from strong_sector_watch_v1 import root
        try:
            alerts=json.loads((root()/'alerts.json').read_bytes())
            end=int(observed_at.strftime('%Y%m%d%H%M%S')) if observed_at is not None else asof
            start=(previous or {}).get('observed_asof',asof//1000000*1000000)
            board_names={b['name'] for b in plan.get('boards',[])}
            fresh=[r for r in alerts.values() if start<r.get('packet',{}).get('asof',0)<=end
                and r['cutoff']//1000000==asof//1000000 and board_names&set(r.get('packet',{}).get('board','').split('、'))]
            fresh.sort(key=lambda r:r['packet']['asof'])
            packet['new_events']=[dict(text=str(r['packet'].get('priority_label',''))+': '+r['code']+' '+r['title']+' '+str(r['packet']['asof'])[-6:-2]) for r in fresh[:6]]
            if len(fresh)>6:packet['new_events'].append(dict(text='另有'+str(len(fresh)-6)+'条，详见异动记录'))
        except (OSError,ValueError,KeyError):pass
    rendered=half_hour(packet,previous)
    from index_map_layers_v1 import render as render_map
    map_text=render_map(facts)
    if map_text:rendered['body']+='\n\n'+map_text
    rendered['next_state']['observed_asof']=int(observed_at.strftime('%Y%m%d%H%M%S')) if observed_at is not None else asof
    return rendered
