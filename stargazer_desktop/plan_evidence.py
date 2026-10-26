"""Code-side facts for planning; never borrow another date's risk score."""
import json
import math
import shadow_ui
import paths
from pathlib import Path
import subprocess

_price_cache={}

NAMES = {'selection': 'Tarot·星币侍从 1.0', 'risk': 'Pyxis 1.0', 'information': 'Corvus'}


def prompt_risk(risk):
    """Keep decision facts, not repeated full audit-column names, in API input."""
    context=risk.get('context') or {}
    keys=('version','status','amp_232','amp_266','amp_reference_price','horizon','stock_close','stock_volume_ratio','market_state','board_states',
          'leader_states','membership_scope','bound_stock_count','declared_stock_count',
          'comparison_board_count','cross_source_close_check','trigger_state',
          'all_board_signals','signal_note','market_price_evidence','market_repair_research')
    compact={k:context[k] for k in keys if k in context}
    if 'market_price_evidence' in compact:
        compact['market_price_evidence']={name:{k:row[k] for k in ('asof','status','summary','volume_state','research_trial') if k in row}
            for name,row in compact['market_price_evidence'].items()}
        for row in compact['market_price_evidence'].values():
            if 'research_trial' in row:row['research_trial']={k:v for k,v in row['research_trial'].items() if k!='resonance_bands'}
    missing=context.get('feature_missing') or {}
    compact['feature_missing_counts']={k:len(v) if isinstance(v,list) else v for k,v in missing.items()}
    compact['full_feature_missing_detail']='未载入提示；完整字段清单保存在模型证据包，不得声称已逐项读取'
    return {**{k:risk[k] for k in ('status','ref','asof_ts','pack_hash','scope') if k in risk},
            'context':compact}


def archived_price_overlay(cutoff):
    """Read-only overlay for accepted older packs; preserve their hash and amplitudes."""
    from joint_risk_bridge import _worker_paths
    python,worker=_worker_paths()
    module=worker.parent.parent/'green_common_profile/market_price_evidence_v1.py'
    mirror=worker.parent/'index_minutes_v1'
    key=None
    try:
        signature=tuple((mirror/(name+'_5min.parquet')).stat().st_mtime_ns for name in ['sse','cyb'])
        key=(int(cutoff),signature)
        if key in _price_cache:return _price_cache[key]
        output=subprocess.run([str(python),'-B','-X','utf8',str(module)],
            input=json.dumps(dict(cutoff=int(cutoff))),capture_output=True,text=True,encoding='utf8',timeout=120,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if output.returncode or len(output.stdout.encode('utf8'))>128*1024:raise ValueError('PRICE_WORKER_FAILED')
        result=json.loads(output.stdout)
        if not isinstance(result,dict) or result.get('cutoff')!=int(cutoff) or set(result.get('evidence',{}))!={'sse','cyb'}:
            raise ValueError('PRICE_WORKER_BINDING')
        evidence=result['evidence']
        if any(not isinstance(row,dict) or row.get('asof')!=int(cutoff) for row in evidence.values()):raise ValueError('PRICE_WORKER_CUTOFF')
        if len(_price_cache)>=2:_price_cache.clear()
        _price_cache[key]=evidence
        return evidence
    except (OSError,ValueError,subprocess.SubprocessError):
        if key is not None:
            if len(_price_cache)>=2:_price_cache.clear()
            _price_cache[key]={}
        return {}


def for_snapshot(snapshot):
    day = str(snapshot.get('data_date', ''))
    result = {}
    for row in snapshot.get('rows', []):
        code = row['code']
        selection = {key: row.get(key) for key in (
            'code', 'name', 'setup', 'rank', 'reference_price', 'zone', 'distance_atr', 'rsi14', 'status', 'entry_rule',
            'personal_score','selection_score','selection_model')}
        selection['ref'] = f'tarot:{day}:{code}'
        risk = {'status': 'MISSING', 'ref': f'pyxis:{day}:{code}',
                'reason': '当前名单数据日没有同日风险证据；不能用历史分数代替'}
        import postclose_risk_runtime
        pack, runtime_reason = postclose_risk_runtime.find_pack(code, snapshot)
        if runtime_reason:
            risk['reason'] = runtime_reason
        if pack is None and runtime_reason is None:
            pack, _ = shadow_ui.find_pack(code, day)
        if pack is not None:
            context = pack.risk_context or {}
            asof = str(context.get('asof_ts', ''))
            if (len(asof) == 14 and asof.isdigit() and asof[:8] == day
                    and asof == day + '150000' and str(pack.asof_ts) == asof):
                selected = next((c for c in pack.candidates if str(c.get('code')) == code), {})
                if row.get('selection_model') is None:
                    selection['personal_score'] = selected.get('p_close_score')
                if (row.get('entry_rule') or {}).get('policy') in ('GOLDEN_LADDER_STOP_V3','GOLDEN_RESONANCE_CROSSDAY_V4'):
                    selection['risk_pack_original_node'] = selected.get('f618')
                    selection['risk_pack_original_distance_atr'] = selected.get('plan_dist_atr')
                else:
                    selection['golden_f618'] = selected.get('f618')
                    selection['golden_distance_atr'] = selected.get('plan_dist_atr')
                from jev_adapter_v1 import _risk_evidence_incomplete
                reasons = _risk_evidence_incomplete(pack)
                if not reasons:
                    # Annotation is separate from the signed pack, never written back.
                    context=dict(context)
                    if (row.get('entry_rule') or {}).get('policy') in ('GOLDEN_LADDER_STOP_V3','GOLDEN_RESONANCE_CROSSDAY_V4'):
                        context['score_origin_observation'] = {'node': selected.get('f618'),
                            'distance_atr': selected.get('plan_dist_atr'),
                            'note': '原评分包的观察位置，仅为历史引用；当前计划阶梯另列，幅度未因换图重算'}
                    if not context.get('market_price_evidence'):
                        context['market_price_evidence']=archived_price_overlay(asof)
                    risk = {'status': 'AVAILABLE', 'ref': risk['ref'], 'asof_ts': asof,
                            'context': context, 'pack_hash': pack.bind().pack_hash,
                            'scope': 'P日收盘行情截止；采集时间另见原证据，不代表收盘前已可得；未来120交易分钟幅度，不覆盖整个次日'}
                else:
                    risk['reason'] = ' / '.join(reasons)
        result[code] = {'selection': selection, 'risk': risk}
    return result


def status(snapshot):
    import postclose_risk_runtime
    evidence = for_snapshot(snapshot)
    available = sum(item['risk']['status'] == 'AVAILABLE' for item in evidence.values())
    runtime=postclose_risk_runtime.state()
    progress=''
    if str(runtime.get('data_date'))==str(snapshot.get('data_date')) and runtime.get('status')=='RUNNING':
        progress=(' · 新任务进行中，匹配的已验收证据继续可用' if available and available==len(evidence) else
                  f" · 后台重建 {runtime.get('done',0)}/{runtime.get('total','?')}")
    elif str(runtime.get('data_date'))==str(snapshot.get('data_date')) and runtime.get('status')=='FAILED':
        progress=(' · 另一次重建未完成，匹配的已验收证据继续可用' if available and available==len(evidence) else
                  ' · 重建失败：'+str(runtime.get('reason','未知')))
    return {'names': NAMES, 'risk_available': available, 'candidate_count': len(evidence),
            'selection': ('已接入 · 一段/二段共同模型排序，默认10只 · 研究效果未验收' if snapshot.get('default_candidate_count') == 10 else
                '已接入 · 共性＋周期黄金位≤0.2ATR' if snapshot.get('rule_version') else '已接入 · 共性规则候选'),
            'risk': f'模型接口已接入 · 本名单同日风险证据 {available}/{len(evidence)}'+progress,
            'information': '未接入'}


def risk_lines(risk):
    """Read-only evidence rendering, not a new risk classification or decision."""
    if risk.get('status') != 'AVAILABLE':
        return ['风险证据缺失：' + risk.get('reason', '未绑定同日输入')]
    context = risk.get('context') or {}
    def number(value, percent=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return '未知'
        return f'{value * 100:.2f}%' if percent else f'{value:.2f}'
    lines = [f"截止 {risk.get('asof_ts', '未知')}；模型已评分。",
             f"未来120交易分钟不利下探预测幅度：232主版 {number(context.get('amp_232'), True)}；266对照版 {number(context.get('amp_266'), True)}。不是概率，也不是整个次日涨跌。"]
    market = context.get('market_state') or {}
    board = context.get('board_states') or {}
    leader = context.get('leader_states') or {}
    if isinstance(market, dict):
        lines.append(f"大盘观测：沪指当日变化 {number(market.get('strength'), True)}；创业板 {number(market.get('cyb_strength'), True)}。")
        inputs = market.get('model_inputs') or {}
        if inputs:
            lines.append(f"沪指指标：5分钟RSI {number(inputs.get('sse_5m_rsi14'))}；15分钟RSI {number(inputs.get('sse_15m_rsi14'))}。指标观测不等于买卖信号。")
    for item in board if isinstance(board, list) else [board]:
        if isinstance(item, dict) and item:
            lines.append(f"所属板块篮子观测：变化 {number(item.get('strength'), True)}；成员覆盖 {number(item.get('coverage'), True)}（部分成员篮子，不冒充完整板块）。")
    for item in leader if isinstance(leader, list) else [leader]:
        if isinstance(item, dict) and item:
            change = number(item.get('return_change'), True)
            lines.append(f"领涨股观测：当日变化 {number(item.get('strength'), True)}；最近区间涨幅变化 {change[:-1] + '个百分点' if change.endswith('%') else change}。")
    missing = context.get('feature_missing') or {}
    if isinstance(missing, dict):
        counts = [f'{k}缺{len(v)}列' for k, v in missing.items() if isinstance(v, list)]
        if counts:
            lines.append('输入披露：' + '；'.join(counts) + '；缺失不等于风险为零。')
    lines.append('次日下杀、止跌、背离触发另行观察；尚未计算不等于本次风险评分缺失。Corvus未接入。')
    lines.extend(price_repair_lines(context,risk.get('asof_ts')))
    return lines


def price_repair_lines(context,cutoff):
    result=[]
    for name,label in [('sse','沪指（主要环境）'),('cyb','创业板（补充）')]:
        row=(context.get('market_price_evidence') or {}).get(name) or {}
        if str(row.get('asof'))!=str(cutoff) or row.get('status')!='PRICE_FACTS_ONLY':
            continue
        five=row['five'];confirmation=row['fifteen']
        def value(x):return '未知' if x is None else f'{x:.2f}'
        result.append(f"{label}：{row['summary']}。五分钟RSI6/RSI15={value(five.get('rsi6'))}/{value(five.get('rsi15'))}；上穿时刻{five.get('cross_up_times',[])}。")
        reclaimed=[x['price'] for x in row['levels'] if x['reclaimed']]
        result.append(f"同档两根完整五分钟收回档位：{reclaimed}；十五分钟辅助截止{confirmation['evidence_time']}，快线在慢线上方：{confirmation['rsi_above']}。指数成交量未知，不判已确认强弱；这是当刻事实，不是次日预测。")
        if row.get('availability_basis')=='DERIVED_FROM_ARCHIVE_ON_READ_NOT_HISTORICAL_ARRIVAL':
            result.append('以上为归档后补算的同截止价格证据，不证明当时盘中已收到；原风险包与评分未改。')
    return result


def market_risk_summary(codes, evidence):
    """One dated market discussion from the selected stocks' bound Pyxis packs."""
    if not codes:
        return ['本次没有模型候选，大盘风险证据未知。']
    risks = [evidence.get(code, {}).get('risk') or {} for code in codes]
    available = [risk for risk in risks if risk.get('status') == 'AVAILABLE']
    if len(available) != len(codes):
        return [f'同日Pyxis风险证据仅有{len(available)}/{len(codes)}只；大盘风险暂列未知，不用缺失冒充低风险。']
    dates = {risk.get('asof_ts') for risk in available}
    states = [risk.get('context', {}).get('market_state') or {} for risk in available]
    if len(dates) != 1 or len({json.dumps(state, sort_keys=True, ensure_ascii=False)
                                for state in states}) != 1:
        return ['候选的Pyxis大盘证据截止或内容不一致；大盘风险暂列未知。']
    market = states[0]
    repair_contexts=[risk.get('context',{}).get('market_price_evidence') for risk in available]
    if repair_contexts[0] and all(row==repair_contexts[0] for row in repair_contexts):
        repair=price_repair_lines(available[0]['context'],next(iter(dates)))
        if repair:return [f'截止{next(iter(dates))}，P收盘环境仅供次日计划参考。',*repair,
                          '个股仍须既定触位/企稳或收回及同截止风险；实际买卖由人工决定。']
    inputs = market.get('model_inputs') or {}
    def finite(value):
        return (float(value) if isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) else None)
    sse = finite(market.get('strength'))
    cyb = finite(market.get('cyb_strength'))
    m5 = finite(inputs.get('sse_5m_momentum_change'))
    m15 = finite(inputs.get('sse_15m_momentum_change'))
    position = finite(market.get('golden_position'))
    facts = (f"截止{next(iter(dates))}；沪指当日{sse*100:+.2f}%" if sse is not None else
             f"截止{next(iter(dates))}；沪指当日变化未知")
    facts += f"，创业板{cyb*100:+.2f}%" if cyb is not None else '，创业板变化未知'
    facts += (f"；沪指5/15分钟动能变化{m5:+.4f}/{m15:+.4f}"
              if m5 is not None and m15 is not None else '；沪指5/15分钟动能证据不足')
    if position is not None and 0 <= position <= 1:
        facts += f"；指数周期区间相对位置{position*100:.2f}%"
    facts += '。'
    if sse is not None and cyb is not None and m5 is not None and m15 is not None:
        if sse < 0 and cyb < 0 and m5 < 0 and m15 < 0:
            reading = '两指数收跌且沪指5/15分钟动能均走弱；计划讨论宜减少主动介入，等待触价后的止跌与人工复核。'
        elif m5 > 0 and m15 > 0:
            reading = '沪指5/15分钟动能均改善；可加强观察，但仍须个股触发及板块、风险证据配合。'
        else:
            reading = '指数方向与短周期动能未形成同向结论；保持审慎观察。'
    else:
        reading = '方向证据不全，暂不判断是否适合增加交易。'
    return [facts, reading,
            '无量跌破黄金位、放量收回、连续抬高低点需要逐日/盘中量价路径；当前绑定包未核验这些组合，状态未知。以上为P收盘环境，不代替E日实时风险。']


def presentation(draft):
    """Derive display facts without rewriting any immutable draft or AI prose."""
    rows = {r['code']: r for r in draft.get('snapshot', {}).get('rows', [])}
    codes = draft.get('selected_codes', [])
    return {'draft': f"观察日 {draft.get('snapshot', {}).get('monitor_date', '未知')}；候选 {len(codes)} 只：" +
                    '、'.join(f"{c} {rows.get(c, {}).get('name', '')}" for c in codes),
            'market_risk_summary': market_risk_summary(codes, draft.get('model_evidence') or {}),
            'manual_observations': draft.get('manual_observations', []),
            'risks': [{'code': c, 'name': rows.get(c, {}).get('name', ''),
                       'lines': risk_lines(draft.get('model_evidence', {}).get(c, {}).get('risk', {}))}
                      for c in codes],
            'confirmation': '人工确认前仅供讨论，不启动本草案提醒、不自动下单。可继续修改数量和板块；触区只提示复核，买入条件仍须下杀、止跌和有效背离。'}


def model_view(slot):
    if slot not in NAMES:
        raise ValueError('UNKNOWN_MODEL')
    import trade_plan_v1
    plan = trade_plan_v1.load_plan()
    if slot == 'information':
        return {'name': NAMES[slot], 'status': 'NOT_CONNECTED', 'items': [], 'reason': '信息模型尚未接入，未知不代表无利空'}
    status = 'CONFIRMED_PLAN'
    source_note = '已确认计划绑定证据，不自动替换为新草案。'
    if not plan:
        import assistant_workflow
        draft = assistant_workflow.latest('plan')
        if not draft or draft.get('awaiting_count'):
            return {'name': NAMES[slot], 'status': 'NO_PLAN', 'items': [], 'reason': '尚无可查看的计划草案'}
        readiness = assistant_workflow.plan_readiness(draft, assistant_workflow.snapshot())
        status = 'UNCONFIRMED_DRAFT' if readiness['current'] else 'STALE_DRAFT'
        source_note = ('未确认草案，只读查看，不启用提醒。' if readiness['current'] else
                       '旧草案证据已变化，仅供历史查看；请按最新证据重新制定。')
        plan = {**draft, 'data_date': draft['snapshot'].get('data_date'),
                'monitor_date': draft['snapshot'].get('monitor_date'),
                'manual_observations': draft.get('manual_observations', []),
                'stocks': [{'code': c, 'name': next((r.get('name', '') for r in draft['snapshot']['rows'] if r['code'] == c), '')}
                           for c in draft['selected_codes']] +
                          [{'code': r['code'], 'name': r['name']} for r in draft.get('manual_observations', [])]}
    evidence = dict(plan.get('model_evidence') or {})
    for row in plan.get('manual_observations', []):
        evidence[row['code']] = {
            'selection': {'status': 'USER_OBSERVATION', 'ref': f"human:{plan.get('data_date')}:{row['code']}",
                          'user_requests': row['user_requests'], 'reason': '人工添加，不是Tarot模型入选；原话条件未结构化、身份未核验。'},
            'risk': {'status': 'MISSING', 'reason': '人工观察股尚无同规格Pyxis评分；未知不等于安全，不借用其他股票分数。'}}
    comments = dict(plan.get('model_commentary') or {})
    supplemental = []
    if slot == 'risk' and status == 'CONFIRMED_PLAN':
        file = Path(paths.data_file('data/pentacles_demo/assistant_plan.json', seed_from_bundle=False))
        try:
            confirmed = json.loads(file.read_bytes())
            if (confirmed.get('assistant_plan_id') == plan.get('assistant_plan_id')
                    and str(confirmed.get('data_date')) == str(plan.get('data_date'))
                    and str(confirmed.get('monitor_date')) == str(plan.get('monitor_date'))):
                current = for_snapshot(confirmed)
                for code, item in current.items():
                    if (code in evidence and evidence[code].get('risk', {}).get('status') != 'AVAILABLE'
                            and item['risk']['status'] == 'AVAILABLE'):
                        evidence[code] = {**evidence[code], 'risk': {**item['risk'],
                            'supplemental': True, 'confirmed_fact': evidence[code].get('risk')}}
                        comments[code] = {**comments.get(code, {}), 'risk':
                            '确认时此项缺失；现补充已验收的同日风险事实，未重算幅度，未改计划或原讨论。'}
                        supplemental.append(code)
        except (OSError, ValueError, KeyError, TypeError):
            pass  # Keep the original missing evidence when the binding cannot be proved.
    if supplemental:
        source_note += ' 已补充同一确认计划原缺失的同日风险事实；原确认记录未改。'
    manual_codes = {row['code'] for row in plan.get('manual_observations', [])}
    market_codes = [row['code'] for row in plan.get('stocks', [])
                    if row['code'] not in manual_codes] if slot == 'risk' else []
    view = {'name': NAMES[slot], 'status': status, 'source_note': source_note,
            'data_date': plan.get('data_date'), 'monitor_date': plan.get('monitor_date'),
            'combined': ('【确认时历史讨论：其中风险缺失表述已过时，请以本页补充事实为准；未重新调用AI】\n' if supplemental else '') + plan.get('combined_commentary', ''),
            'market_risk_summary': market_risk_summary(market_codes, evidence) if slot == 'risk' else [],
            'items': [{'code': s['code'], 'name': s.get('name'),
                       'facts': evidence.get(s['code'], {}).get(slot, {'status': 'MISSING'}),
                       'risk_lines': risk_lines(evidence.get(s['code'], {}).get('risk', {})) if slot == 'risk' else [],
                       'commentary': comments.get(s['code'], {}).get(slot, '') or
                           ('按已冻结共性条件进入候选；具体形态、位置与指标见事实。' if slot == 'selection'
                            else evidence.get(s['code'], {}).get('risk', {}).get('reason', '以下为同日风险快照，不能据此保证明日涨跌。')),
                       'commentary_kind': 'AI解释，非新增模型事实',
                       'plan_levels':(s.get('entry_rule') or {}).get('levels',[])} for s in plan.get('stocks', [])]}
    if slot=='risk':
        from risk_outlook_v1 import latest,view_id
        outlook=latest()
        view['outlook']=outlook if outlook.get('plan_id')==view_id(view) else {'status':'NOT_GENERATED'}
    return view
