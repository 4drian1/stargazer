"""Independent live stock workers and explicitly timed risk observations."""
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
import json,subprocess,threading,math
from golden_rules import digest
CONTRACT='LIVE_OBSERVATIONS_WITH_PCLOSE_REFERENCE_V1'
_reference={}
_index={}
_lock=threading.Lock()

def active(now):
    m=now.hour*60+now.minute
    return now.weekday()<5 and (570<=m<=690 or 780<=m<900)

def one(plan,row,now,quote=None):
    from joint_risk_bridge import _worker_paths
    import postclose_risk_runtime as runtime
    from golden_monitor import validate_result
    python,worker=_worker_paths();project=worker.parents[3]
    sub=dict(plan,rows=[row]);cutoff=int(now.strftime('%Y%m%d%H%M%S'))
    body=dict(project=str(project),runtime=str(runtime.directory()),plan=sub,cutoff=cutoff,quote=quote)
    output=subprocess.run([str(python),'-B','-X','utf8',str(worker.parent.parent/'green_common_profile/live_plan_worker_v1.py')],
        input=json.dumps(body,ensure_ascii=False),capture_output=True,text=True,encoding='utf8',timeout=30,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if output.returncode or len(output.stdout.encode())>512*1024:raise ValueError('LIVE_WORKER_FAILED')
    value=json.loads(output.stdout)
    if (value.get('cutoff'),value.get('plan_hash'),value.get('rule_version'))!=(cutoff,digest(sub),plan['rule_version']):raise ValueError('LIVE_WORKER_IDENTITY')
    validate_result(value,sub)
    return value['rows'][row['code']]

def evaluate(plan,now,quotes=None,on_row=None):
    result=dict(cutoff=int(now.strftime('%Y%m%d%H%M%S')),plan_hash=digest(plan),rule_version=plan['rule_version'],rows={})
    def run(row):
        code=row['code']
        try:signal=one(plan,row,now,(quotes or {}).get(code))
        except Exception as exc:signal=dict(status='DATA_INSUFFICIENT',rule_hash=digest(row['entry_rule']),reason=str(exc)[:160],error_type=type(exc).__name__)
        if on_row:on_row(code,signal)
        return code,signal
    # Independent deadlines: one source/stock cannot block publication of another.
    with ThreadPoolExecutor(max_workers=4,thread_name_prefix='live-stock') as pool:
        for job in as_completed([pool.submit(run,r) for r in plan['rows']]):
            code,signal=job.result();result['rows'][code]=signal
    return result

def risk(plan,code,asof,signal,now):
    import pentacles_demo,postclose_risk_runtime as runtime
    key=(int(plan['data_date']),code)
    with _lock:old=_reference.get(key)
    if old is None:
        try:
            snap=json.loads(pentacles_demo.snapshot_path().read_bytes())
            if int(snap['data_date'])!=key[0]:raise ValueError('REFERENCE_DAY_CHANGED')
            pack,reason=runtime.find_pack(code,snap)
            if pack is None and reason and 'CANDIDATE_SNAPSHOT_CHANGED' in reason:
                # 确认计划快照无投影父哈希；按冻结原始选股快照绑定并逐股校验（人工新增/输入变化→显式缺失）。
                pack,reason=runtime.find_pack_for_plan(code,snap)
            old=dict(context=deepcopy(pack.risk_context),asof_ts=int(pack.asof_ts),pack_hash=pack.bind().pack_hash) if pack else dict(context={},reason=reason)
        except (OSError,ValueError,KeyError,TypeError) as exc:old=dict(context={},reason=str(exc))
        if old.get('pack_hash'):
            with _lock:_reference[key]=old
    context=dict(old.get('context') or {})
    # P-close market/sector numbers must never masquerade as current observations.
    for field in ('market_state','board_states','leader_states','market_price_evidence','execution_index_maps','market_repair_research','stock_volume_ratio'):
        context.pop(field,None)
    from risk_outlook_v1 import live_index_facts
    minute=asof//100
    with _lock:indices=_index.get(minute)
    if indices is None:
        indices=live_index_facts(asof,now)
        with _lock:
            _index.clear();_index[minute]=indices
    context['market_state']=dict(strength=(indices.get('market.sse') or {}).get('value'),cyb_strength=(indices.get('market.cyb') or {}).get('value'))
    context['live_index_quotes']=indices
    from joint_risk_bridge import _worker_paths
    python,worker=_worker_paths();project=worker.parents[3]
    try:
        proc=subprocess.run([str(python),'-B','-X','utf8',str(worker.parent.parent/'green_common_profile/live_index_preview_v1.py')],
            input=json.dumps(dict(project=str(project),mirror=str(worker.parent/'index_minutes_v1'),asof=asof,quotes=indices)),
            capture_output=True,text=True,encoding='utf8',timeout=12,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if proc.returncode or len(proc.stdout.encode())>128*1024:raise ValueError('INDEX_PREVIEW_WORKER')
        context['live_market_preview']=json.loads(proc.stdout)
    except (ValueError,OSError,subprocess.SubprocessError):context['live_market_preview']={}
    try:
        summary=json.loads((runtime.directory()/'strong_sector/summary.json').read_bytes())
        age=(now-datetime.strptime(str(summary['asof']),'%Y%m%d%H%M%S')).total_seconds()
        if summary.get('pool_day')==asof//1000000 and 0<=age<=120:context['live_sector_observations']=summary
    except (OSError,ValueError,KeyError,TypeError):pass
    context['amp_reference_asof']=old.get('asof_ts')
    context['amp_reference_price']=context.get('stock_close')
    context['stock_close']=(signal.get('comparisons') or {}).get('last_close')
    context['stock_price_asof']=asof
    context['live_signal']=dict(asof=asof,rsi=signal.get('rsi'),combination=signal.get('combination'),comparisons=signal.get('comparisons'))
    return dict(status='AVAILABLE',asof_ts=asof,context=context,observation_contract=CONTRACT,
        model_reference=dict(asof=old.get('asof_ts'),pack_hash=old.get('pack_hash'),missing_reason=old.get('reason')),
        scope='当前实时技术/报价观察＋明确标时的前收盘Pyxis参考；不是同截止新Pyxis推理',provisional=True)

def ready(record,asof):
    c=record.get('context') or {};signal=c.get('live_signal') or {}
    return (record.get('observation_contract')==CONTRACT and record.get('asof_ts')==asof and record.get('provisional') is True
        and signal.get('asof')==asof and type(c.get('stock_close')) in (int,float) and math.isfinite(c['stock_close']) and c['stock_close']>0)

def notice_key(plan,code,signal):
    return digest(dict(day=plan['monitor_date'],plan=plan.get('assistant_plan_id') or digest(plan),code=code,band=signal.get('band'),policy=signal.get('combination_policy')))
