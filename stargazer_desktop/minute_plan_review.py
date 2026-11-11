"""Read-only causal replay. No orders, alerts, fits or retrospective AI calls."""
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def requested(message):
    if re.search(r'不要复盘|别复盘|不用复盘|先不复盘', message):
        return False
    return bool(re.search(r'复盘|回看|回顾|触发了?吗|拦住|拦截|避开|买点|选的.*怎么样|那几只.*(?:表现|赚|亏|信号)', message))


def resolve(message, now=None, day=None, codes=None, scope=None, archive_id=None):
    """Choose a dated immutable population; never substitute a current list for history."""
    import assistant_workflow as W
    import paths
    now = now or datetime.now(W.PLAN_ZONE)
    text = message
    selection_day = None
    if day is None:
        match = re.search(r'20\d{6}', text)
        if match:
            day = int(match[0])
        else:
            dates=list(re.finditer(r'(?:(\d{1,2})月)?(\d{1,2})[号日]',text))
            if len(dates)>1:
                return dict(status='CLARIFY',reply='提到了多个日期，请确认要复盘哪天的行情；选股日与观察日分别记录。')
            if dates:
                token=dates[0]
                date=now.replace(month=int(token[1] or now.month),day=int(token[2])).date()
                day=int(date.strftime('%Y%m%d'))
                if re.match(r'\s*(?:选|挑|制定|定的|定好)',text[token.end():]):
                    selection_day=day
            selected_yesterday = bool(re.search(r'(?:昨天|昨晚|昨日).*(?:选|定|计划|名单).*(?:今天|今日)|(?:昨天|昨晚).*(?:选的|挑的|定的计划)',text))
            offset = 0 if '今天' in text or '今日' in text or selected_yesterday else 2 if '前天' in text else 1 if '昨天' in text or '昨日' in text else 0
            if day is None:
                date = now.date() - timedelta(days=offset)
                day = int(date.strftime('%Y%m%d'))
                # Default branch only: when no date was named and today is a
                # weekend there is no plan observed on Saturday/Sunday, so roll
                # back to the most recent weekday (Friday) instead of concluding
                # "no list exists". An explicit date from the message is left
                # untouched.
                if date.weekday() >= 5:
                    date = date - timedelta(days=1 if date.weekday() == 5 else 2)
                    day = int(date.strftime('%Y%m%d'))
    datetime.strptime(str(day), '%Y%m%d')
    if day > int(now.strftime('%Y%m%d')):
        return dict(status='CLARIFY', reply='这个观察日尚未发生，不能复盘未来走势。')
    if scope=='strong' or scope is None and re.search(r'强势|异动',text):
        return dict(status='STRONG_ARCHIVE',day=day,codes=codes or re.findall(r'(?<!\d)\d{6}(?!\d)',text))
    if scope is None:
        manual=bool(re.search(r'未过门|人工复核|人工名单|没入围|候补', text))
        scope='all' if manual and '计划' in text else 'manual' if manual else 'plan' if '计划' in text else 'all'
    if scope not in ('all', 'plan', 'manual'):
        raise ValueError('复盘范围无效')
    codes = codes or re.findall(r'(?<!\d)\d{6}(?!\d)', text)
    plans = []
    directory = Path(paths.data_file('data/pentacles_demo/journal/plans', seed_from_bundle=False))
    for file in directory.glob('*.json'):
        raw = file.read_bytes()
        if hashlib.sha256(raw).hexdigest() != file.stem:
            raise ValueError('复盘名单归档哈希不符')
        value = json.loads(raw)
        if int(value.get('data_date' if selection_day else 'monitor_date',0)) == (selection_day or day):
            plans.append((file.stem, value, False))
    # An explicitly confirmed plan is the execution population; manual rows stay separate.
    confirmed = []
    for file in W.root().glob('confirmed_*.json'):
        value = json.loads(file.read_bytes())
        plan = value.get('plan') or {}
        if int(plan.get('data_date' if selection_day else 'monitor_date',0)) == (selection_day or day):
            confirmed.append((file.stem, plan, True))
    if scope == 'plan' and confirmed:
        plans = confirmed
    elif not plans:
        plans = confirmed
        try:
            snap = W.snapshot()
            if not plans and int(snap.get('data_date' if selection_day else 'monitor_date',0)) == (selection_day or day):
                plans = [(W.digest(snap), snap, False)]
        except (OSError, ValueError):
            pass
    if archive_id:
        plans = [p for p in plans if p[0] == archive_id]
    # Duplicate byte-equivalent populations need not make the user choose.
    unique = {}
    for key, plan, live in plans:
        unique[W.digest(dict(rows=plan.get('rows'), manual=plan.get('manual_review_rows')))] = (key, plan, live)
    plans = list(unique.values())
    if len(plans) != 1:
        return dict(status='CLARIFY', day=day, choices=[dict(archive_id=k, codes=[r['code'] for r in p.get('rows', [])]) for k,p,_ in plans],
            reply='该日有多版名单，请选归档版本。' if plans else '未找到该观察日的名单；请告诉我股票或使用哪天选出的名单，不用今天名单回填。')
    key, plan, actual = plans[0]
    day=int(plan['monitor_date'])
    if day>int(now.strftime('%Y%m%d')):
        return dict(status='CLARIFY',reply='该次选股对应的观察日尚未发生，不能用之后行情回填。')
    rows = []
    if scope in ('all','plan'):
        rows.extend(dict(r, review_origin='CONFIRMED_PLAN' if actual else 'UNCONFIRMED_SELECTION') for r in plan.get('rows', []))
    if scope in ('all','manual'):
        rows.extend(dict(r, review_origin='MANUAL_REVIEW_HYPOTHETICAL') for r in plan.get('manual_review_rows', []))
    if not codes:
        names = [r['code'] for r in rows if r.get('name') and r['name'] in text]
        codes = names
    if codes:
        rows = [r for r in rows if r['code'] in codes]
        if {r['code'] for r in rows} != set(codes):
            return dict(status='CLARIFY', reply='部分股票不在这份历史名单中，请说明对应名单或先计算其历史黄金图。')
    if not rows or len(rows) > 10:
        return dict(status='CLARIFY', reply='此次请指定1至10只股票；没有人工复核股票时不拿正式候选代替。')
    cutoff = day * 1000000 + 150000
    if day == int(now.strftime('%Y%m%d')):
        from datetime import time
        hm = now.hour * 60 + now.minute
        slots = list(range(575,691,5)) + list(range(785,901,5))
        past = [t for t in slots if t <= hm]
        if not past:
            return dict(status='CLARIFY', reply='今天尚无已完成五分钟K线，开盘后才能复盘当日过程。')
        cutoff = day * 1000000 + past[-1] // 60 * 10000 + past[-1] % 60 * 100
    return dict(status='READY', day=day, cutoff=cutoff, archive_id=key, confirmed_population=actual,
                plan_id=plan.get('assistant_plan_id'), rows=rows, source_hash=W.digest(plan))


def replay_row(row, day, cutoff, minutes, evaluator, adjustments):
    """Each evaluation sees only its own prefix. A signal fill uses the NEXT minute."""
    from golden_ladder_v3 import STOPS
    code = row['code']
    own = sorted([r for r in minutes if str(r.get('code')).zfill(6) == code and int(r['date']) <= cutoff], key=lambda r: r['date'])
    today = [r for r in own if int(r['date']) // 1000000 == day]
    if not today:
        return dict(code=code, status='DATA_MISSING', reason='TARGET_DAY_MINUTES_MISSING')
    bands = (row.get('entry_rule') or {}).get('observation_bands') or []
    touches = [dict(time=int(m['date']), lower=b['lower'], upper=b['zone_upper']) for b in bands for m in today
               if m['low'] <= b['zone_upper'] and m['high'] >= b['lower']]
    first_touch = min(touches, key=lambda v:v['time']) if touches else None
    gaps_through = [dict(time=int(m['date']), lower=b['lower']) for b in bands for m in today
                    if m['high'] < b['lower'] and int(m['date']) % 1000000 == 93000]
    states = []
    signals = []
    last = None
    for stop in STOPS:
        end = day * 1000000 + stop
        if end > cutoff:
            break
        prefix = [m for m in own if int(m['date']) <= end]
        value = evaluator(row, day, end, prefix, row['reference_price'], adjustments)
        if value.get('status') == 'RULE_TRIGGERED' and value.get('trigger_time') not in {s['trigger_time'] for s in signals}:
            signals.append(value)
        compact = dict(cutoff=end, status=value.get('status'), reason=value.get('reason'),
                       combination=value.get('combination'), pending_conditions=value.get('pending_conditions'))
        identity = json.dumps({k:v for k,v in compact.items() if k != 'cutoff'}, sort_keys=True)
        if identity != last:
            states.append(compact)
        last = identity
    first = signals[0] if signals else None
    after = [m for m in today if first and int(m['date']) > first['trigger_time']]
    reference = row.get('reference_price')
    close = float(today[-1]['close'])
    day_move = dict(close=close, reference=reference,
        change=(close / reference - 1) if isinstance(reference, (int, float)) and reference > 0 else None,
        high=float(max(m['high'] for m in today)), low=float(min(m['low'] for m in today)))
    followup = None
    if first_touch:
        touched = next(m for m in today if int(m['date']) == first_touch['time'])
        later = [m for m in today if int(m['date']) > first_touch['time']]
        base = float(touched['close'])
        followup = dict(reference_price=base, reference_basis='TOUCH_5M_CLOSE_NOT_FILL',
                        change_to_cutoff=close/base-1, later_bars=len(later),
                        max_up=max(m['high'] for m in later)/base-1 if later else None,
                        max_down=min(m['low'] for m in later)/base-1 if later else None)
    simulation = None
    if after:
        fill = float(after[0]['open'])
        simulation = dict(entry_time=int(after[0]['date']), entry_price=fill, fill_assumption='NEXT_5MIN_KLINE_OPEN_SIMULATION',
            maximum_favorable_excursion=max(m['high'] for m in after)/fill-1,
            maximum_adverse_excursion=min(m['low'] for m in after)/fill-1,
            mark_to_market_return=after[-1]['close']/fill-1, realized_return=None, net_return=None,
            note='模拟持仓浮盈亏；以真实5分钟K线的下一根K线开盘价作为入场模拟，非次分钟成交；未验证成交量/涨跌停成交可行性，不含费用；T+1不当日卖出，不把最高价当可实现收益')
    return dict(code=code, name=row.get('name',''), status='REPLAYED', origin=row.get('review_origin'),
        touch=first_touch, opening_gap_through=gaps_through[:1], first_signal=first,
        signal_count=len(signals), state_changes=states, simulation=simulation, day_move=day_move,
        touch_followup=followup,
        coverage=dict(first_minute=int(today[0]['date']), last_minute=int(today[-1]['date'])),
        actual_monitoring='NOT_PROVEN_BY_REPLAY')


def compute(body):
    """Worker: native local minutes and checked local corporate-event dates."""
    project = Path(body['project'])
    sys.path[:0] = [str(project/'stargazer_desktop'),str(project/'stargazer_ml/v5_hybrid/green_common_profile'),
                    str(project/'stargazer_ml/v5_hybrid/risk_r0_v1')]
    from gcp_golden_intraday_v1 import prefix_result
    from tdx_corporate_events_v1 import checked_events
    capture = {}
    context = body['context']
    results = []
    for row in context['rows']:
        try:
            rule = row.get('entry_rule') or {}
            if rule.get('policy') != 'GOLDEN_RESONANCE_CROSSDAY_V4' or not rule.get('eligible'):
                raise ValueError('HISTORICAL_EXECUTABLE_RULE_MISSING')
            from plan_rule_profiles import policy
            if rule.get('combination_policy')!=policy(rule.get('execution_profile')):
                raise ValueError('HISTORICAL_EXECUTION_CONTRACT_UNSUPPORTED')
            days = sorted(set(rule.get('history_days',[])))
            if not days or len(days)>4 or max(days)>=context['day']:
                raise ValueError('HISTORY_DAY_BINDING_INVALID')
            minutes = []
            from tdx_native_history_v1 import native_rows
            # History days (completed trading days strictly before the review
            # day) are finished sessions: enforce the exact 48-bar grid.
            for d in days:
                rows, receipt = native_rows(row['code'], [d])
                capture['分钟k:'+row['code']+':'+str(d)+'*']=dict(pairs=[(None,v) for v in rows],source_receipt=receipt)
                for v in rows:
                    if str(v['code']).zfill(6)!=row['code'] or int(v['date'])//1000000!=d:
                        raise ValueError('MINUTE_SOURCE_IDENTITY_OR_DAY_MISMATCH')
                    minutes.append(dict(v,code=row['code']))
            # The review day is validated UP TO the cutoff carried in context:
            # a finished (historical) review day has cutoff = session end, so the
            # prefix gate still demands all 48 bars; a live intraday replay has
            # cutoff = the latest completed slot, so only the continuous legal
            # prefix up to that slot is required. Either way gaps, reordering,
            # duplicates, off-grid bars, bad prices and post-cutoff bars are
            # rejected.
            rows, receipt = native_rows(row['code'], [context['day']],
                                        cutoff=context['cutoff'])
            capture['分钟k:'+row['code']+':'+str(context['day'])+'*']=dict(pairs=[(None,v) for v in rows],source_receipt=receipt)
            for v in rows:
                if str(v['code']).zfill(6)!=row['code'] or int(v['date'])//1000000!=context['day']:
                    raise ValueError('MINUTE_SOURCE_IDENTITY_OR_DAY_MISMATCH')
                minutes.append(dict(v,code=row['code']))
            actions,receipt=checked_events(row['code'],context['day'],max(days),
                body.get('runtime',project/'stargazer_desktop/dist/data/postclose_risk'))
            capture['公司行为事件:'+row['code']]=dict(pairs=actions,source_receipt=receipt)
            result = replay_row(row,context['day'],context['cutoff'],minutes,prefix_result,actions)
            result['corporate_event_receipt']=receipt
        except (OSError,ValueError,KeyError,TypeError) as error:
            result = dict(code=row['code'], status='DATA_OR_RULE_GAP', reason=str(error)[:180])
        results.append(result)
    from golden_resonance_v4 import __file__ as kernel_source
    return dict(status='COMPLETE', context=context, rows=results, capture=capture, http_attempts=0,
                engine_sha256=hashlib.sha256(Path(kernel_source).read_bytes()).hexdigest(),
                paid_calls=0, orders=0, alerts=0)



def save_review_evidence(file, value):
    """Raw replay evidence is not a2MiB alert-state document."""
    import gzip
    import assistant_workflow as W
    import postclose_risk_runtime as runtime
    from runtime_disk_limit_v1 import disk_limit,disk_usage
    root=runtime.directory()
    file=Path(file)
    if root.resolve() not in file.resolve().parents:
        raise ValueError('REVIEW_EVIDENCE_PATH')
    raw=json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf8')
    if len(raw)>16*1024**2:
        raise ValueError('REVIEW_EVIDENCE_TOO_LARGE')
    packed=gzip.compress(raw,mtime=0)
    with W.LOCK:
        if file.exists():
            if gzip.decompress(file.read_bytes())!=raw:
                raise ValueError('REVIEW_EVIDENCE_CONFLICT')
            return
        if disk_usage(root)+len(packed)*2+65536>disk_limit(root):
            raise ValueError('RUNTIME_DISK_BUDGET')
        file.parent.mkdir(parents=True,exist_ok=True)
        temporary=file.with_suffix(file.suffix+'.tmp')
        with temporary.open('xb') as stream:stream.write(packed)
        temporary.replace(file)

def review(message='', day=None, codes=None, scope=None, archive_id=None, cfg=None):
    import paths
    import assistant_workflow as W
    context = resolve(message, day=day, codes=codes, scope=scope, archive_id=archive_id)
    if context['status']=='STRONG_ARCHIVE':
        from strong_sector_archive_v1 import review as archived_review
        import postclose_risk_runtime as runtime
        value=archived_review(runtime.directory(),context['day'],context['codes'])
        lines=['强势股历史归档｜'+str(context['day'])]
        for row in value['rows']:
            change=('收盘相对计划前收%+.2f%%'%(row['return_vs_pool_reference']*100)
                    if row['price_status']=='KNOWN' else '收盘行情待核验')
            events=['%s 价%s 区间%s—%s（%s）'%(e.get('packet',{}).get('asof',e.get('cutoff')),
                e.get('packet',{}).get('price'),e.get('packet',{}).get('level_lower'),
                e.get('packet',{}).get('level_upper'),e.get('delivery','UNKNOWN')) for e in row['events']]
            lines.append(row['code']+' '+row['name']+'｜'+change+'｜'+('；'.join(events) or '未记录提醒；不代表全天未触发'))
        value['reply']='\n'.join(lines+[value['note']])
        # Load readable observations into chat; large indicator/map evidence stays in the archive.
        value['rows']=[{k:r[k] for k in ('code','name','boards','price_status','observation_status',
            'monitoring_coverage','close','return_vs_pool_reference') if k in r} | dict(events=[
                dict(time=e.get('packet',{}).get('asof',e.get('cutoff')),price=e.get('packet',{}).get('price'),
                    lower=e.get('packet',{}).get('level_lower'),upper=e.get('packet',{}).get('level_upper'),
                    delivery=e.get('delivery','UNKNOWN')) for e in r['events']]) for r in value['rows']]
        value['kind']='MINUTE_REVIEW_VIEW_V1'
        return value
    if context['status'] != 'READY':
        return context
    app = Path(paths.APP_DIR).resolve()
    project = next((p for p in (app,*app.parents) if (p/'stargazer_ml').is_dir()), app.parent)
    python = Path(os.environ.get('STARGAZER_PYTHON',str(project/'_dev_venv/Scripts/python.exe')))
    if not python.exists():
        python = project/'rule_venv/Scripts/python.exe'
    worker = Path(paths.resource('minute_plan_review.py'))
    import postclose_risk_runtime as runtime
    done = subprocess.run([str(python),'-B','-X','utf8',str(worker)], input=json.dumps(dict(project=str(project),context=context,runtime=str(runtime.directory()))),
        capture_output=True,text=True,encoding='utf-8',timeout=330,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if done.returncode or len(done.stdout.encode())>16*1024*1024:
        raise ValueError('分钟复盘工作进程失败；没有自动重试或修改正式计划')
    result = json.loads(done.stdout)
    # Retain exact input bytes and outcomes under the existing runtime budget.

    import postclose_risk_runtime as runtime
    identifier = W.digest(result)
    file = runtime.directory()/'assistant_reviews'/(identifier+'.json.gz')
    if not file.exists():
        save_review_evidence(file,result)
    capture=result.pop('capture',{})
    result['evidence_file'] = str(file)
    result['review_id'] = identifier
    result['historical_ai'] = historical_advice(context, runtime.directory())
    for record in result['historical_ai']['records']:
        if record.get('decision')!='DEFER' or record.get('matching_plan') is not True:
            continue
        table='分钟k:'+record['code']+':'+str(context['day'])+'*'
        minutes=[v for _,v in capture.get(table,{}).get('pairs',[]) if int(v['date'])<=context['cutoff']]
        known=[m for m in minutes if int(m['date'])<=record['cutoff']]
        later=[m for m in minutes if int(m['date'])>record['cutoff']]
        if known and later:
            ref=max(known,key=lambda m:m['date'])['close']
            record['subsequent_excursion']=dict(reference_price=ref,
                maximum_downside=min(m['low'] for m in later)/ref-1,
                maximum_upside=max(m['high'] for m in later)/ref-1,
                note='暂缓之后的双向价格变化；非实际避损或错失收益，不能据此直接认定决策正确')
    result['push_receipts']=push_receipts(context)
    from pentacles_review_data import actual_trades
    result['actual_trades'] = actual_trades(cfg or {},{r['code'] for r in context['rows']},context['day'])
    result['reply'] = render(result)
    assessment=runtime.directory()/'assistant_reviews'/(W.digest(result)+'_assessment.json.gz')
    if not assessment.exists(): save_review_evidence(assessment,result)
    result['assessment_file']=str(assessment)
    return result


def push_receipts(context):
    import paths
    file=Path(paths.data_file('data/pentacles_demo/monitor.json',seed_from_bundle=False))
    if not file.exists(): return []
    if file.stat().st_size>2*1024*1024: raise ValueError('MONITOR_RECEIPT_SIZE')
    result=[]
    for key,alert in json.loads(file.read_bytes()).get('alerts',{}).items():
        review=alert.get('review') or {}
        if review.get('code') not in {r['code'] for r in context['rows']} or review.get('cutoff',0)//1000000!=context['day']:
            continue
        result.append(dict(code=review['code'],cutoff=review['cutoff'],observed_at=alert.get('observed_at'),
            delivery=alert.get('delivery'),wechat=alert.get('wechat'),
            matching_plan=alert.get('plan_id')==context.get('plan_id') if alert.get('plan_id') else 'UNPROVEN'))
    return result


def historical_advice(context, root):
    records=[]
    for file in (root/'tarot_v2/advice').glob('*.json'):
        if file.stat().st_size>512*1024:
            continue
        value=json.loads(file.read_bytes())
        if value.get('code') not in {r['code'] for r in context['rows']} or value.get('cutoff',0)//1000000!=context['day']:
            continue
        # Old advice lacking plan binding is evidence of a response, not proof for this plan.
        records.append(dict(code=value['code'],cutoff=value['cutoff'],status=value.get('status'),
            decision=value.get('decision'),reason=value.get('reason'),path=str(file),
            matching_plan='UNPROVEN' if not value.get('plan_id') else value['plan_id']==context.get('plan_id')))
    return dict(status='RECORDED_RESPONSES_ONLY' if records else 'NO_HISTORICAL_AI_RECORD', records=records,
                interception_success='NOT_INFERRED_FROM_LATER_PRICE')


def _votes(combination):
    votes=(combination or {}).get('votes') or {}
    if not votes:return ''
    return '（'+' '.join(label+('✓' if votes.get(key) is True else '✗' if votes.get(key) is False else '待')
                         for key,label in [('repair','位置'),('momentum','动能'),('oversold','超卖')])+'）'


def render(result):
    lines=['五分钟复盘｜观察日%s｜截止%s；历史规则回放，使用历史名单与当时价格，事后重放不等于真实执行，也不等于盘中未收盘试算。' % (result['context']['day'],result['context']['cutoff'])]
    for r in result['rows']:
        if r['status']!='REPLAYED':
            lines.append(r['code']+'：未完成，'+r.get('reason','未知'))
            continue
        touch=r.get('touch');signal=r.get('first_signal');sim=r.get('simulation')
        text=r['code']+' '+r.get('name','')+'：'+('触区'+str(touch['time']) if touch else '没有已核验的触区记录')
        states=r.get('state_changes') or []
        if signal:
            text+='；规则触发'+str(signal['trigger_time'])+_votes(signal.get('combination'))
        elif states:
            last=states[-1]
            reasons={'WAIT_POSITION_AND_TWO_OF_THREE':'位置或三选二未满足',
                     'NO_QUALIFIED_LOW_RESONANCE_BAND':'原计划无合格低位共振观察区'}
            reason=last.get('reason') or last.get('status') or '原因未登记'
            text+='；未确认买入组合：'+reasons.get(reason,reason)+_votes(last.get('combination'))
            scored=[s for s in states if (s.get('combination') or {}).get('votes')]
            if scored:
                best=max(scored,key=lambda s:sum(v is True for v in s['combination']['votes'].values()))
                text+='；盘中最多满足于'+str(best['cutoff'])+_votes(best['combination'])+'；截止状态'+_votes(last.get('combination'))
                first_pass=[]
                for key,label in [('repair','位置修复'),('momentum','动能'),('oversold','严重超卖')]:
                    passed=next((s for s in scored if s['combination']['votes'].get(key) is True),None)
                    if passed:first_pass.append(label+' '+str(passed['cutoff']))
                if first_pass:text+='；各项首次成立：'+'、'.join(first_pass)+'（不同时间不合并计票）'
            if last.get('pending_conditions'):
                text+='；待'+'、'.join(str(c) for c in last['pending_conditions'])
        else:
            text+='；未确认买入组合'
        move=r.get('day_move')
        if move and move.get('change') is not None:
            text+='；'+('收盘' if result['context']['cutoff']%1000000==150000 else '截止价')+'%g（相对参考%+.2f%%，区间%g—%g）'%(move['close'],move['change']*100,move['low'],move['high'])
        followup=r.get('touch_followup')
        if followup:
            text+='；触区根收盘%g→截止%+.2f%%（比较基准，非成交价）'%(followup['reference_price'],followup['change_to_cutoff']*100)
            if followup['later_bars']:
                text+='；其后最高偏离%+.2f%%／最低偏离%+.2f%%'%(followup['max_up']*100,followup['max_down']*100)
        if sim:
            text+='；模拟下一根5分钟K线开盘价%g，截止浮盈亏%.2f%%，最大下探%.2f%%、最大上行%.2f%%（非已实现利润）' % (sim['entry_price'],sim['mark_to_market_return']*100,sim['maximum_adverse_excursion']*100,sim['maximum_favorable_excursion']*100)
        if r.get('origin')=='MANUAL_REVIEW_HYPOTHETICAL':
            text+='；未过门股票，仅假设回放，未自动加入监控'
        lines.append(text)
    lines.append('历史AI回复与真实成交另列；没有对应记录，不宣称成功拦截、真实买入或获利。')
    return '\n'.join(lines)


if __name__=='__main__':
    print(json.dumps(compute(json.load(sys.stdin)),ensure_ascii=False,allow_nan=False))
