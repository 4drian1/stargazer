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
    # 灰灯是「指数关键位置证据不齐」，不是行情灯，也不是已判定破位/修复。
    if missing:
        return ('⚪灰（指数关键位置证据不齐：缺' + '、'.join(missing) +
                '关键位置状态；证据未就绪，不等于安全或失守）')
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


def quote_coverage_text(observed):
    """把「报价 0/N」写成可判读的覆盖说明，而不是让人误读成 0 只异动票。

    * 无观测 / 池数缺失 -> 待核验
    * valid=0           -> 明确「未取得有效报价」，不等于池内无异动
    * 0<valid<pool      -> 部分覆盖，未覆盖票不参与异动判断
    * valid=pool        -> 全覆盖
    """
    if not observed:
        return '待核验'
    try:
        valid = int(observed.get('valid_count') or 0)
        pool = int(observed.get('pool_count') or 0)
    except (TypeError, ValueError):
        return '报价覆盖待核验（计数不可用）'
    if pool <= 0:
        return '报价覆盖待核验（池内数量未提供）'
    if valid <= 0:
        return (f'本时段未完成扫描（有效报价 0/{pool}，未取得有效报价；'
                f'不代表池内无异动）')
    if valid < pool:
        return (f'有效报价 {valid}/{pool}（部分覆盖；未覆盖的 {pool - valid} 只'
                f'不参与本轮异动判断）')
    return f'有效报价 {pool}/{pool}（全覆盖）'


# 只有这些状态才代表本时段确实完成了扫描并拿到了有效报价。
_SCANNED_OK = {'MONITORING', 'DEGRADED'}
_SCAN_REASONS = {
    'NOT_STARTED': '强势池扫描线程尚未启动',
    'NOT_OWNER': '写入权未取得（NOT_OWNER），本轮零报价请求',
    'LANE_ERROR': '强势池通道异常',
    'QUOTE_ERROR': '报价请求失败，未取得有效报价',
    'WAIT_POOL': '冻结强势池未就绪',
    'WAIT_BOARDS': '板块名单未就绪',
    'DISABLED': '强势池通道已停用',
    'PLAN_NOT_READY': '当日计划未就绪',
    'WAITING_FOR_MONITOR_DATE': '等待计划交易日',
    'OUTSIDE_SESSION': '非交易时段',
    'PLAN_NOT_CONFIRMED': '计划未确认',
    'DAY_CHANGED_DURING_SCAN': '扫描跨日，结果已作废',
}


def scan_status_text(strong_scan, asof=None, max_age_seconds=900):
    """把强势池通道健康状态写成「本时段是否完成扫描」的明确结论。

    上午漏扫事故：通道因 NOT_OWNER 零报价，但巡检只显示「本时段无」，
    被误读成「无异动」。这里明确区分：
      * 未扫描 / 零报价 / 状态陈旧  -> 「本时段未完成扫描」+ 具体原因
      * 已扫描且有有效报价          -> 「本时段已完成扫描」，此时才允许说无新增异动
    """
    if not isinstance(strong_scan, dict) or not strong_scan:
        return '本时段未完成扫描（强势池扫描状态未提供，不能据此判断有无异动）'
    status = str(strong_scan.get('status') or 'UNKNOWN')
    valid = strong_scan.get('valid_quotes')
    pool = strong_scan.get('pool_count')
    checked_at = str(strong_scan.get('checked_at') or '')

    # 陈旧状态不能冒充本时段扫描结果。
    stale = False
    if asof and checked_at:
        try:
            import datetime as _dt
            stamp = _dt.datetime.fromisoformat(checked_at.replace('Z', '+00:00'))
            expected = _dt.datetime.strptime(str(asof)[:12], '%Y%m%d%H%M')
            if stamp.tzinfo is not None:
                expected = expected.replace(tzinfo=stamp.tzinfo)
            stale = abs((stamp - expected).total_seconds()) > max_age_seconds
        except (ValueError, TypeError):
            stale = False
    reason = _SCAN_REASONS.get(status)
    error = strong_scan.get('error') or strong_scan.get('request_error')
    detail = '；'.join(x for x in (reason, ('错误：' + str(error)[:120]) if error else None) if x)

    if stale:
        return (f'本时段未完成扫描（最近一次扫描 {checked_at} 已过时，'
                f'不得用旧状态判断本时段有无异动）')
    if status not in _SCANNED_OK:
        return '本时段未完成扫描（' + (detail or '通道状态：' + status) + '）'
    try:
        valid_i, pool_i = int(valid or 0), int(pool or 0)
    except (TypeError, ValueError):
        valid_i, pool_i = 0, 0
    if valid_i <= 0:
        return (f'本时段未完成扫描（有效报价 0/{pool_i or "?"}；'
                f'{detail or "未取得有效报价"}，不代表池内无异动）')
    if pool_i and valid_i < pool_i:
        return (f'本时段已完成扫描（有效报价 {valid_i}/{pool_i}，部分覆盖；'
                f'未覆盖的 {pool_i - valid_i} 只不参与异动判断）')
    return f'本时段已完成扫描（有效报价 {valid_i}/{pool_i or valid_i}，全覆盖）'


def market_summary(facts, cutoff, observed_at=None):
    from datetime import datetime

    def verified(key):
        fact = facts.get(key) or {}
        if fact.get('status') == 'KNOWN' and fact.get('asof') == cutoff and fact.get('evidence_time') == cutoff:
            return fact.get('value')
        return None

    from index_outlook_v1 import NAMES

    moves, volumes, divergences = [], [], []
    volume_known = 0
    missing_divergence = []
    direction_known = False
    for name, label in NAMES.items():
        quote = facts.get('market.' + name) or {}
        change = quote.get('value')
        quote_valid = quote.get('status') == 'KNOWN' and quote.get('asof') == cutoff
        if quote.get('evidence_kind') == 'QUOTE_SNAPSHOT' and quote.get('bar_cutoff') == cutoff:
            try:
                stamp = datetime.strptime(str(quote['asof']), '%Y%m%d%H%M%S')
                end = observed_at or datetime.strptime(str(cutoff), '%Y%m%d%H%M%S')
                quote_valid = quote.get('status') == 'KNOWN' and stamp.date() == end.date() and (
                    observed_at is None or 0 <= (end.replace(tzinfo=None) - stamp).total_seconds() <= 120)
            except (KeyError, ValueError):
                quote_valid = False
        if not quote_valid:
            change = None
        if not finite(change):
            mapping = facts.get(f'index.{name}.rebound_map') or {}
            if mapping.get('status') == 'KNOWN' and mapping.get('asof') == cutoff:
                change = (mapping.get('value') or {}).get('return_from_previous_close')
        data = verified(f'index.{name}.recent_five_bars') or {}
        bars = data.get('bars') or []
        consecutive = False
        if len(bars) == 3 and bars[-1].get('time') == cutoff:
            try:
                times = [datetime.strptime(str(bar['time']), '%Y%m%d%H%M%S') for bar in bars]
                consecutive = times[0].date() == times[-1].date() and all(
                    (b-a).total_seconds() == 300 for a, b in zip(times, times[1:]))
            except (ValueError, KeyError):
                pass
        closes = [bar.get('close') for bar in bars] if consecutive else []
        if len(closes) == 3 and all(finite(price) and price > 0 for price in closes) and (
                closes[0] > closes[1] > closes[2] or closes[0] < closes[1] < closes[2]):
            moves.append(label + '近三根已收盘五分钟K线' + ('走低' if closes[0] > closes[2] else '走高'))
            direction_known = True
        elif finite(change):
            moves.append(label + ('持平' if change == 0 else
                         ('小幅' if abs(change) < 0.01 else '') + ('上涨' if change > 0 else '下跌')))
            direction_known = True
        else:
            moves.append(label + '走势待核验')
        values = [bar.get('volume') for bar in bars] if consecutive and data.get('volume_verified') is True else []
        if len(values) == 3 and all(finite(v) and v > 0 for v in values):
            volume_known += 1
            trend = ('持续缩量' if values[0] > values[1] > values[2] else
                     '持续放量' if values[0] < values[1] < values[2] else '未呈持续增减')
            volumes.append(label + '近三根五分钟K线' + trend)
        before = len(divergences)
        negative_five = False
        for period in ('5', '15'):
            rsi, dif = [verified(f'index.{name}.divergence.{period}.{indicator}_divergence')
                        for indicator in ('rsi', 'dif')]
            triggered = [indicator for indicator, state in (('RSI', rsi), ('DIF', dif)) if state is True]
            if triggered:
                divergences.append(f'{label}{period}分钟' + '/'.join(triggered) + '底背离')
            if period == '5':
                negative_five = rsi is False and dif is False
        if len(divergences) == before:
            if negative_five:
                divergences.append(f'{label}5分钟未见RSI/DIF底背离')
            else:
                missing_divergence.append(label)
    if not volume_known and not divergences:
        details = ['量能与背离待核验']
    else:
        details = volumes + (['其余指数量能待核验' if volume_known else '量能待核验']
                             if volume_known < len(NAMES) else []) + divergences
        if len(missing_divergence) == len(NAMES):
            details.append('背离待核验')
        elif missing_divergence:
            details.append('、'.join(missing_divergence) + '背离待核验')
    observation = '观察下一根已收盘K线能否延续' if direction_known else '继续观察'
    return '，'.join(moves) + '。' + '，'.join(details + [observation]) + '。'


def report(bulletin,plan,result,quotes,positions,previous=None,sector_events=None,observed_at=None,
           strong_scan=None):
    facts=bulletin.get('computed_facts') or {};asof=bulletin['asof']
    analysis=bulletin.get('analysis') or {};sections=analysis.get('sections') or {}
    def value(k):
        f=facts.get(k) or {}
        return f.get('value') if f.get('status')=='KNOWN' else None
    packet=dict(asof=asof,plan_id=plan.get('assistant_plan_id'),plans=[],boards=[],holdings=None if positions is None else [],new_events=sector_events or [],
                scan_status_text=scan_status_text(strong_scan, asof))
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
        indicators = signal.get('comparisons') or {}
        indicator_text = None
        if indicators.get('status') == 'AVAILABLE' and indicators.get('asof') == result.get('event_cutoff'):
            from push_layout_v1 import stamp
            current = ((indicators.get('periods') or {}).get('5') or {}).get('current') or {}
            numbers = ['%s %.2f' % (label, current[key]) for key, label in
                       (('rsi6', 'RSI6'), ('rsi15', 'RSI15'), ('dif', 'DIF')) if finite(current.get(key))]
            if numbers:
                indicator_text = '5分钟低点试算 ' + stamp(indicators['asof']) + ' ' + '/'.join(numbers)
        packet['plans'].append(dict(code=code,name=row.get('name'),price=p,level_lower=node,level_upper=node,
            level_label='黄金位',distance=(p-node)/node if finite(p) and finite(node) and node>0 else None,
            status=signal.get('status','UNKNOWN'),votes=(signal.get('combination') or {}).get('votes'),
            indicator_text=indicator_text,
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
    packet['market_summary'] = market_summary(facts, asof, observed_at)
    packet['market_gate'] = packet['gate_text'].split('（', 1)[0] if all(
        state in ('BROKEN', 'REPAIRED', 'PENDING') for state in states.values()) else '待核验'
    from pentacles_monitor import combination_passed
    wait_text = _risk_lane_wait(plan, asof)
    signals = result.get('rows') or {}
    if signals:
        passed = any(combination_passed(s) for s in signals.values())
        rules_text = '三选二通过' if passed else '已运行未通过'
    else:
        # 无同截止规则结果时一律「待核验」：不得显示成"未运行"让人误以为
        # 已核验过且无信号，也不得当成规则通过/不通过的结论。
        rules_text = '待核验（本轮无同截止规则结果）'
    if bulletin.get('status') == 'READY':
        ai_text = '已完成'
    elif bulletin.get('status') == 'AI_FAILED' or bulletin.get('validation_error'):
        # OUTLOOK_SECTION_INVALID 等事实校验失败必须如实显示，不能说成"未完成"
        # 或伪装成研判完成；本地事实保留且不自动付费重试。
        err = str(bulletin.get('validation_error') or '')
        ai_text = '回复未通过事实校验' + ('（' + err + '）' if err else '')
        packet['ai_failure_text'] = (ai_text + '；本地事实仍可查看，未自动付费重试')
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
    packet['volume_scope_text'] = (
        '以上为报价时点累计成交额快照（单位亿元，非同截止成交量）；'
        '同刻量比需历史同刻成交量与已完成K线，当前口径未通过校验，标待核验，'
        '不得用累计成交额比值或当前根未收盘量比冒充放量/缩量结论')
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
        if observed.get('valid_count'):
            drop4 = observed.get('drop4_count')
            changes = (('池内%s只跌超4%%；' % (drop4 if finite(drop4) else '未知数量')
                       if 'drop4_count' in observed else '异动统计待核验；')
                       + quote_coverage_text(observed))
        else:
            changes = '待核验'
        packet['boards'].append(dict(name=board['name'],leader_text=leader_text,change_text=changes))
    active_codes = {pos['code'] for pos in positions or []}
    analyzed_codes = (bulletin.get('request') or {}).get('holding_codes')
    holding_text = (sections.get('holdings', '')
                    if isinstance(analyzed_codes, list) and set(analyzed_codes) == active_codes else '')
    for pos in positions or []:
        code=pos['code'];price=value(code+'.stock.price');cost=pos.get('entry');policy=pos.get('tarot_execution') or {}
        # Independent same-code quote, subject to the same-day freshness gate.
        from strong_touch_lane_v1 import quote_quality
        quote = quotes.get(code) or {}
        if observed_at is not None and quote_quality(quote, asof//1000000, observed_at) is None:
            price = quote['price']
            quote_origin = '持仓实时报价 ' + str(quote['time'])[-6:-4] + ':' + str(quote['time'])[-4:-2]
        elif finite(price):
            quote_origin = '持仓独立事实字段'
        else:
            price = None
            quote_origin = '未取得持仓独立报价'
        packet['holdings'].append(dict(code=code,name=pos.get('name'),price=price,cost=cost,
            quote_origin=quote_origin,
            pnl=price/cost-1 if finite(price) and finite(cost) and cost>0 else None,
            stop_loss=policy.get('stop_price'),take_profit=pos.get('take_profit'),state_text='；'.join(holding_text.splitlines()) or '待研判'))
    packet.update(advice_status='ADVICE_READY' if bulletin.get('status')=='READY' else None,
        advice_verified=bulletin.get('status')=='READY' and bool(bulletin.get('analysis')),
        advice_text=sections.get('plan'))
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
