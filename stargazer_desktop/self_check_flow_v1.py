"""Explicit offline acceptance only: real workflow IO, historical rules, fake transports."""
import copy,gzip,json,tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock,patch

def exercise(project,review_file):
    from datetime import datetime
    class HistoricalPreparationClock(datetime):
        @classmethod
        def now(cls,tz=None):
            return cls(2026,10,31,20,0,tzinfo=tz)
    import paths,assistant_workflow as workflow,postclose_prepare_v1 as prep
    import minute_history_v1 as history,tarot_monitor_v2 as live
    from pentacles_monitor import PriceMonitor
    from trade_facts_v1 import CONTRACT
    current=json.loads((project/'stargazer_desktop/dist/data/postclose_risk/tdx_daily_acceptance/v86/SELECTION_P30.json').read_bytes())
    replay=json.loads(gzip.decompress(Path(review_file).read_bytes()))
    result=dict(mode='OFFLINE_HISTORICAL_REPLAY_AND_TRANSPORT_SIMULATION',actual_http=0,paid=0,push=0,orders=0)
    # Real native daily history, four whole sessions used for next-session indicators.
    prepared={}
    for row in current['rows']:
        days=row['entry_rule']['history_days']
        records,receipts=history.load_days(row['code'],days,lambda n:None)
        if len(records)!=48*len(days):raise AssertionError('PREPARATION_INCOMPLETE:'+row['code'])
        prepared[row['code']]=dict(days=days,bars=len(records),receipts=len(receipts))
    result['native_preparation']=prepared
    # Real draft persistence/confirmation, in an isolated directory. Gate and
    # background schedulers are captured, never start monitors or replace user data.
    temporary=project/'stargazer_desktop/dist/data/postclose_risk/tmp'
    with tempfile.TemporaryDirectory(dir=temporary) as name:
        isolated=Path(name)
        def local_file(relative,**kw):return str(isolated/relative)
        evidence={r['code']:dict(risk=dict(status='UNKNOWN',reason='TEST_NO_RISK_FABRICATED')) for r in current['rows']}
        with patch.object(paths,'data_file',side_effect=local_file),patch.object(paths,'FROZEN',True), \
             patch.object(workflow,'datetime',HistoricalPreparationClock), \
             patch.object(workflow,'snapshot',return_value=current),patch('plan_evidence.for_snapshot',return_value=evidence), \
             patch.object(history,'schedule') as schedule,patch.object(prep,'prepare_holdings') as holdings, \
             patch('strong_sector_watch_v1.schedule') as pool,patch('assistant_runtime.gate') as gate:
            prep.after_selection(current)
            draft=workflow.latest('plan')
            if not draft or not draft['selected_codes']:raise AssertionError('NO_AUTOMATIC_DRAFT')
            if schedule.call_count!=1 or holdings.call_count!=1 or pool.call_count!=1 or gate.call_count!=1:
                raise AssertionError('PREPARATION_DISPATCH_INCOMPLETE')
            if (isolated/'data/trade_plan.json').exists():raise AssertionError('AUTO_CONFIRM_FORBIDDEN')
            try:workflow.confirm_plan(draft['id'],False)
            except ValueError:pass
            else:raise AssertionError('HUMAN_CONFIRM_REQUIRED')
            confirmed=workflow.confirm_plan(draft['id'],True)
            written=json.loads((isolated/'data/pentacles_demo/assistant_plan.json').read_bytes())
            if written['data_date']!=20261030 or written['monitor_date']!=20261102:raise AssertionError('P_E_DRIFT')
            if len(written['rows'])!=confirmed['count']:raise AssertionError('CONFIRMED_ROSTER_DRIFT')
            result['confirmation']=dict(P=written['data_date'],E=written['monitor_date'],count=confirmed['count'],
                explicit_human_required=True,user_plan_replaced=False,automatic_preparation_dispatch=True)
        # Use a real historical rule signal, including its actual 2/3 votes.
        historical=next(r for r in replay['rows'] if r['code']=='002585')
        signal=historical['first_signal'];cutoff=signal['price_asof']
        if signal['status']!='RULE_TRIGGERED' or sum(v is True for v in signal['combination']['votes'].values())<2:
            raise AssertionError('HISTORICAL_SIGNAL_NOT_CONFIRMED')
        original=next(r for r in replay['context']['rows'] if r['code']=='002585')
        plan=dict(assistant_plan_id='OFFLINE_FLOW_ONLY',plan_ready=True,monitor_date=20261030,
            rule_version=live.VERSION,rows=[original])
        folder=isolated/'monitor';folder.mkdir()
        (folder/'assistant_plan.json').write_text(json.dumps(plan,ensure_ascii=False),encoding='utf8')
        monitor=PriceMonitor(lambda:folder);monitor.enable(True)
        stamp=datetime.strptime(str(cutoff),'%Y%m%d%H%M%S')
        quote=dict(time=str(cutoff),prev=original['reference_price'],price=signal['last_two_closes'][-1],low=signal['session_low'])
        send=Mock(return_value=(True,'TEST_TRANSPORT_ONLY'));broadcast=Mock()
        def tick(value):
            with patch.object(live,'evaluate',return_value=value),patch('risk_outlook_v1.latest',return_value={}):
                monitor._tick(plan,dict(push=dict(enabled=True)),stamp,lambda codes:{'002585':quote},broadcast,send)
        # Arrival independently reaches the notifier before a qualifying combination.
        waiting=copy.deepcopy(signal);waiting.update(status='WAIT_RULE',advice={})
        waiting['combination']['votes']=dict(repair=False,momentum=False,oversold=True)
        tick(dict(status='MONITORING',event_cutoff=cutoff,rows={'002585':waiting}))
        if send.call_count!=1:raise AssertionError('ARRIVAL_NOT_SENT')
        for decision in ['DEFER','BUY']:
            row=copy.deepcopy(signal)
            row['advice']=dict(status='ADVICE_READY',code='002585',cutoff=cutoff,decision=decision,
                fact_contract=CONTRACT,advice_references=dict(supporting_ids=[],opposing_ids=[],unknown_ids=[]),
                risk=dict(status='AVAILABLE',context=dict(stock_close=quote['price'])),
                reason='测试替身：板块风险仍在，暂缓' if decision=='DEFER' else '测试替身：综合通过，人工决定')
            value=dict(status='MONITORING',event_cutoff=cutoff,rows={'002585':row})
            tick(value);tick(value)
        if send.call_count!=3:raise AssertionError('ADVICE_OR_DEDUP_FAILED')
        if '板块风险' not in send.call_args_list[1].args[2] or '建议买入' not in send.call_args_list[2].args[1]:
            raise AssertionError('DECISION_OR_REASON_LOST')
        monitor=PriceMonitor(lambda:folder);tick(value)
        if send.call_count!=3:raise AssertionError('RESTART_RESENT_ALERT')
        result['intraday']=dict(code='002585',historical_trigger=cutoff,votes=signal['combination']['votes'],
            arrival=1,defer=1,buy=1,restart_duplicate=0,transport='MOCK_NOT_SENT',ai='MOCK_NOT_A_MODEL_RESULT',
            provisional_real_feed='NOT_FORWARD_TESTED_ON_WEEKEND')
        monitor.enable(False);tick(value)
        if send.call_count!=3:raise AssertionError('DISABLED_STILL_SENT')
    result['review']=dict(rows=len(replay['rows']),replayed=sum(r['status']=='REPLAYED' for r in replay['rows']),
        actual_trades=False,actual_ai_interception='NOT_ESTABLISHED')
    if result['review']['replayed']!=10:raise AssertionError('REVIEW_INCOMPLETE')
    result['passed']=True
    return result
