"""Thin process boundary for completed-minute rules; quote-only checks never buy."""
from datetime import datetime
import json
import math
import re
from pathlib import Path
import subprocess

import paths
import postclose_risk_runtime as runtime
from golden_rules import VERSION, digest

COMBINATION_POLICY='POSITION_TWO_OF_THREE_O20_CROSS1_V1'
MINUTE_SOURCE_CONTRACT='SINA_STOCK_1M_WITH_FIXED_HISTORY_V2'


def evaluate(plan, now, *, five_minute=False):
    from joint_risk_bridge import _worker_paths
    minute = now.hour*60+now.minute
    end = minute-minute%5
    valid = list(range(575,691,5))+list(range(785,901,5))
    past = [m for m in valid if m <= end]
    if not past:
        return {'rows': {}, 'reason': 'NO_COMPLETED_5M_BAR'}
    cutoff = int(now.strftime('%Y%m%d'))*1000000+(past[-1]//60)*10000+(past[-1]%60)*100
    if plan.get('rule_version')=='TAROT_A_SECOND_CASH_V2' and not five_minute:
        from joint_risk_bridge import completed_cutoff
        cutoff=completed_cutoff(now)
        if cutoff is None:return {'rows':{},'reason':'NO_COMPLETED_RISK_15M_BAR'}
    python, worker = _worker_paths()
    worker = worker.parent.parent/'green_common_profile/gcp_golden_intraday_v1.py'
    body = dict(plan=plan, cutoff=cutoff, runtime=str(runtime.directory()))
    output = subprocess.run([str(python),'-B','-X','utf8',str(worker)],
        input=json.dumps(body,ensure_ascii=False),capture_output=True,text=True,encoding='utf-8',timeout=45,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if output.returncode or len(output.stdout.encode()) > 512*1024:
        detail=(output.stderr or '').strip().splitlines()
        reason=detail[-1] if detail else 'NO_ERROR_DETAIL'
        if not re.fullmatch(r'ValueError: [A-Z0-9_]+',reason):reason='WORKER_EXIT_'+str(output.returncode)
        raise ValueError('GOLDEN_WORKER_FAILED:'+reason)
    value = json.loads(output.stdout)
    if value.get('plan_hash') != digest(plan) or value.get('cutoff') != cutoff or value.get('rule_version') != plan.get('rule_version'):
        raise ValueError('GOLDEN_WORKER_BINDING_MISMATCH')
    validate_result(value, plan)
    value['plan_id'] = plan.get('assistant_plan_id')
    target = runtime.directory()/'golden_intraday/state.json'
    from pentacles_monitor import atomic_json
    atomic_json(target, value)
    return value


def validate_result(value, plan):
    if set(value.get('rows',{})) != {r['code'] for r in plan['rows']}:
        raise ValueError('GOLDEN_RESULT_POPULATION_MISMATCH')
    for row in plan['rows']:
        item = value['rows'][row['code']]
        if item.get('rule_hash') != digest(row['entry_rule']):
            raise ValueError('GOLDEN_RULE_HASH_MISMATCH')
        if item.get('status') == 'RULE_TRIGGERED':
            if plan.get('rule_version')=='TAROT_A_SECOND_CASH_V2':
                if row['entry_rule'].get('policy')=='GOLDEN_RESONANCE_CROSSDAY_V4':
                    validate_combination(item,row,value['cutoff'])
                    continue
                new_policy=row['entry_rule'].get('policy')=='GOLDEN_LADDER_STOP_V3'
                trigger=item.get('trigger_time')
                valid_time=(isinstance(trigger,int) and trigger//1000000==value['cutoff']//1000000 and trigger<=value['cutoff']) if new_policy else trigger==value['cutoff']
                if (not valid_time or item.get('price_asof')!=value['cutoff'] or
                        item.get('node') not in row['entry_rule']['levels']):
                    raise ValueError('V2_TRIGGER_BINDING_INVALID')
                if new_policy:
                    bars=item.get('confirmation_bars') or []
                    if item.get('policy')!='GOLDEN_LADDER_STOP_V3' or item.get('branch') not in ('STOP_ABOVE','RECLAIM') or len(bars)<2:
                        raise ValueError('V3_CONFIRMATION_EVIDENCE_INVALID')
                    if day_time:=bars[-1].get('t'):
                        if trigger%1000000!=day_time:raise ValueError('V3_EVENT_TIME_MISMATCH')
                    if item['branch']=='STOP_ABOVE':
                        if (len(bars)!=3 or bars[0]['low']<item['node'] or
                            not bars[2]['low']>=bars[1]['low']>=bars[0]['low'] or
                            not bars[2]['close']>bars[1]['close']>=bars[0]['close'] or
                            any(b.get('volume') is None for b in bars) or
                            (bars[1]['volume']+bars[2]['volume'])/2>bars[0]['volume']):
                            raise ValueError('V3_STOP_EVIDENCE_INVALID')
                    elif not all(b['close']>=item['node'] for b in bars[-2:]):
                        raise ValueError('TWO_COMPLETED_RECLAIM_REQUIRED')
                    continue
                if item.get('branch')=='RECLAIM':
                    closes=item.get('last_two_closes') or []
                    if len(closes)!=2 or not all(c>=item['node'] for c in closes):
                        raise ValueError('TWO_COMPLETED_RECLAIM_REQUIRED')
                elif item.get('branch')=='RSI':
                    flags,div=item.get('flags') or {},item.get('divergence') or {}
                    if (not all(flags.get(k) is True for k in ('approach','stop_seen','divergence_seen'))
                            or not div or not div.get('anchor_t',0)<div.get('event_t',0)<=value['cutoff']%1000000):
                        raise ValueError('V2_RSI_EVIDENCE_INVALID')
                else:raise ValueError('V2_TRIGGER_BRANCH_INVALID')
                continue
            flags, div = item.get('flags') or {}, item.get('divergence') or {}
            t = item.get('trigger_time')
            if (not all(flags.get(k) is True for k in ('approach','stop_seen','divergence_seen'))
                    or not isinstance(t,int) or not value['cutoff']//1000000==t//1000000 or t>value['cutoff']
                    or not div or not div.get('anchor_t',0)<div.get('event_t',0)<=t%1000000):
                raise ValueError('GOLDEN_TRIGGER_EVIDENCE_INVALID')


def validate_v4(item,row,cutoff):
    rule=row['entry_rule'];trigger=item.get('trigger_time');band=item.get('band') or {}
    def high(a,b):
        return isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b) and a>b and not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    choices=rule.get('observation_bands') or []
    div=item.get('divergence') or {};anchor=item.get('cross_day_anchor') or {}
    bars=item.get('confirmation_bars') or [];crosses=item.get('event_cross_times') or []
    low_time=div.get('event_time');touch=item.get('touch_time');rsi=item.get('rsi') or {}
    if (item.get('policy')!='GOLDEN_RESONANCE_CROSSDAY_V4' or band not in choices or
            item.get('node')!=band.get('lower') or item.get('node') not in rule['levels'] or
            not isinstance(trigger,int) or trigger//1000000!=cutoff//1000000 or trigger>cutoff or item.get('price_asof')!=cutoff or
            not isinstance(low_time,int) or not isinstance(touch,int) or low_time//1000000!=cutoff//1000000 or touch//1000000!=cutoff//1000000 or
            len(crosses)<2 or crosses!=sorted(set(crosses)) or not all(max(low_time,touch)<=t<=trigger for t in crosses) or
            not high(rsi.get('rsi6'),rsi.get('rsi15')) or
            not isinstance(anchor.get('time'),int) or anchor['time']//1000000!=rule['p_date'] or
            anchor.get('low')!=div.get('anchor_low') or anchor.get('rsi6')!=div.get('anchor_rsi') or
            not high(div.get('anchor_low'),div.get('event_low')) or not high(div.get('event_rsi'),div.get('anchor_rsi')) or
            len(bars)<2 or bars[-1].get('t')!=trigger%1000000):
        raise ValueError('V4_COMBINATION_EVIDENCE_INVALID')
    if item.get('branch')=='RECLAIM':
        if not all(b['close']>=band['upper'] for b in bars[-2:]):raise ValueError('V4_WHOLE_BAND_RECLAIM_REQUIRED')
    elif item.get('branch')=='STOP_ABOVE':
        if (len(bars)!=3 or bars[0]['low']<band['lower'] or bars[0]['low']>band['zone_upper'] or
                not bars[2]['low']>=bars[1]['low']>=bars[0]['low'] or
                not bars[2]['close']>bars[1]['close']>=bars[0]['close'] or
                any(b.get('volume') is None for b in bars) or (bars[1]['volume']+bars[2]['volume'])/2>bars[0]['volume']):
            raise ValueError('V4_STOP_EVIDENCE_INVALID')
    else:raise ValueError('V4_BRANCH_INVALID')


def validate_combination(item,row,cutoff):
    from plan_rule_profiles import validate, policy
    profile=validate(row['entry_rule'].get('execution_profile'))
    required=profile['required_votes'];cross_required=profile['momentum_crosses_required'];ratio=profile['oversold_ratio']
    rule=row['entry_rule'];band=item.get('band') or {};c=item.get('combination') or {};votes=c.get('votes') or {}
    live=item.get('provisional') is True and item.get('evaluation_basis')=='LIVE_OBSERVED_TWO_OF_THREE_V1'
    if item.get('provisional') and not live:raise ValueError('UNRECOGNIZED_PROVISIONAL_CONTRACT')
    trigger=item.get('trigger_time');touch=item.get('touch_time');div=item.get('divergence') or {};anchor=item.get('cross_day_anchor') or {}
    low_time=div.get('event_time');crosses=item.get('event_cross_times') or [];bars=item.get('confirmation_bars') or [];rsi=item.get('rsi') or {}
    def number(v):return type(v) in (int,float) and math.isfinite(v)
    def high(a,b):return number(a) and number(b) and a>b and not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    def time(v):return type(v) is int and v//1000000==cutoff//1000000 and v<=trigger
    if (item.get('policy')!='GOLDEN_RESONANCE_CROSSDAY_V4' or item.get('combination_policy')!=policy(profile) or
        band not in (rule.get('observation_bands') or []) or item.get('node')!=band.get('lower') or item['node'] not in rule['levels'] or
        type(trigger) is not int or trigger//1000000!=cutoff//1000000 or trigger>cutoff or item.get('price_asof')!=cutoff or
        not time(low_time) or not time(touch) or item.get('session_low')!=div.get('event_low') or
        type(anchor.get('time')) is not int or anchor['time']//1000000!=rule['p_date'] or
        anchor.get('low')!=div.get('anchor_low') or anchor.get('rsi6')!=div.get('anchor_rsi') or
        not number(anchor.get('rsi6')) or not 0<=anchor['rsi6']<=100 or
        set(votes)!={'repair','momentum','oversold'} or any(v is not None and type(v) is not bool for v in votes.values()) or
        c.get('count')!=sum(v is True for v in votes.values()) or c['count']<required or
        c.get('required')!=required or c.get('momentum_crosses_required')!=cross_required or c.get('oversold_ratio')!=ratio or
        crosses!=sorted(set(crosses)) or any(not time(t) or t<max(low_time,touch) for t in crosses) or
        len(bars)<(2 if required>1 or votes['repair'] else 1) or
        (not live and bars[-1].get('t')!=trigger%1000000) or
        (live and (item.get('observed_asof')!=cutoff or item.get('current_bar_complete') is not False or
            any(not isinstance(b.get('t'),int) or b['t']>trigger%1000000 for b in bars))) or
        (live and votes['repair'] and any(b.get('complete') is not True for b in bars))):
        raise ValueError('TWO_OF_THREE_EVIDENCE_INVALID')
    divergence=high(div.get('anchor_low'),div.get('event_low')) and high(div.get('event_rsi'),div.get('anchor_rsi'))
    expected_m=bool(divergence and high(rsi.get('rsi6'),rsi.get('rsi15')) and len(crosses)>=cross_required) if number(div.get('event_rsi')) else None
    if votes['momentum']!=expected_m:raise ValueError('ONE_CROSS_MOMENTUM_EVIDENCE_INVALID')
    event=item.get('oversold_evidence')
    if votes['oversold']:
        if (not isinstance(event,dict) or not time(event.get('time')) or event.get('anchor_rsi6')!=anchor['rsi6'] or
            event.get('ratio')!=ratio or not number(event.get('rsi6')) or not 0<=event['rsi6']<anchor['rsi6']*ratio):
            raise ValueError('OVERSOLD20_EVIDENCE_INVALID')
    elif event is not None:raise ValueError('OVERSOLD_VOTE_MISMATCH')
    if not votes['repair']:
        if item.get('branch')!='MOMENTUM_OVERSOLD' or sum(votes[k] is True for k in ('momentum','oversold'))<required:
            raise ValueError('POSITION_OPTIONAL_BRANCH_INVALID')
    elif item.get('branch')=='RECLAIM':
        if not all(b['close']>=band['upper'] for b in bars[-2:]):raise ValueError('WHOLE_BAND_RECLAIM_REQUIRED')
    elif item.get('branch')=='STOP_ABOVE':
        if (len(bars)!=3 or not band['lower']<=bars[0]['low']<=band['zone_upper'] or
            not bars[2]['low']>=bars[1]['low']>=bars[0]['low'] or not bars[2]['close']>bars[1]['close']>=bars[0]['close'] or
            any(not number(b.get('volume')) or b['volume']<0 for b in bars) or (bars[1]['volume']+bars[2]['volume'])/2>bars[0]['volume']):
            raise ValueError('STOP_EVIDENCE_INVALID')
    else:raise ValueError('REPAIR_BRANCH_INVALID')


def status_for(formal_plan, code, ts):
    """Read only: UI queries cannot start capture or resurrect a future signal."""
    try:
        file = runtime.directory()/'golden_intraday/state.json'
        live=runtime.directory()/'tarot_events_live/latest.json'
        if live.exists():
            candidate=json.loads(live.read_bytes())
            if candidate.get('plan_id')==formal_plan.get('assistant_plan_id'):file=live
        if file.stat().st_size > 512*1024:
            raise ValueError('STATE_SIZE')
        value = json.loads(file.read_bytes())
        stock = next(s for s in formal_plan['stocks'] if s['code']==code)
        if value.get('rule_version') != (stock.get('entry_rule') or {}).get('version'):
            raise ValueError('STATE_RULE_VERSION')
        query = datetime.strptime(str(ts),'%Y%m%d%H%M%S')
        cutoff = datetime.strptime(str(value['cutoff']),'%Y%m%d%H%M%S')
        if (not value.get('plan_id') or value['plan_id'] != formal_plan.get('assistant_plan_id')
                or cutoff.date() != query.date() or not 0 <= (query-cutoff).total_seconds() <= 360):
            raise ValueError('STATE_STALE_OR_WRONG_PLAN')
        row = value['rows'][code]
        stock = next(s for s in formal_plan['stocks'] if s['code']==code)
        if row.get('rule_hash') != digest(stock.get('entry_rule')):
            raise ValueError('STATE_RULE_CHANGED')
        if row.get('trigger_time') and int(row['trigger_time']) > int(ts):
            raise ValueError('FUTURE_TRIGGER')
        if (stock.get('entry_rule') or {}).get('policy')=='GOLDEN_RESONANCE_CROSSDAY_V4' and row.get('combination_policy')!=__import__('plan_rule_profiles').policy((stock.get('entry_rule') or {}).get('execution_profile')):
            raise ValueError('STATE_EXECUTION_CONTRACT_CHANGED')
        return row
    except (OSError, ValueError, KeyError, TypeError):
        return {'status':'WAIT_MINUTE_EVIDENCE','reason':'等待本计划已完成5分钟K线的下杀、止跌、背离核验；报价触价不构成信号'}
