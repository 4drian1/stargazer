"""Read-only price maps for any stock; selection admission is a separate fact."""
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

FINGERPRINT='live_monitor_fingerprints_20261110_v206_flow.json.gz'


def check_code(code):
    if not isinstance(code,str) or not re.fullmatch(r'(?:00|30|60|68)\d{4}',code):
        raise ValueError('需要六位A股代码')


def summarize_map(mapping,close,atr):
    if not math.isfinite(atr) or atr<=0:raise ValueError('ATR_NOT_AVAILABLE')
    bands=[]
    for b in mapping['bands']:
        sources=b['sources'];identities={(s['low_day'],s['high_day']) for s in sources}
        bands.append(dict(node=b['node'],lower=b['lower'],upper=b['upper'],
            source_waves=len(identities),shared_peak_groups=b['shared_peak_groups'],
            sources=[{k:s[k] for k in ('low_day','high_day','low','high','ratio','kind','known_at',
                                      'source_state','anchor_low_breached_later') if k in s}
                     for s in sources[:4]],sources_truncated=len(sources)>4))
    bands.sort(key=lambda b:b['node'],reverse=True)
    new=mapping['policy']=='GOLDEN_RESONANCE_CROSSDAY_V4'
    zones=[]
    for b in ([] if new else bands[:5]):
        node=b['node']
        if not 0<node<=close:raise ValueError('NODE_ABOVE_CLOSE')
        zones.append(dict(node=node,upper_unclipped=node+.2*atr,
            zone_lower=node,zone_upper=min(close,node+.2*atr),
            source_waves=b['source_waves'],shared_peak_groups=b['shared_peak_groups']))
    if new:
        zones=[dict(node=b['lower'],zone_lower=b['lower'],zone_upper=b['zone_upper'],
            source_waves=b['source_waves'],shared_peak_groups=b['shared_peak_groups'])
            for b in mapping['bands'] if b['observable']]
    if not zones and not new:raise ValueError('NO_CONFIRMED_POSITION')
    return dict(policy=mapping['policy'],map_strategy=mapping.get('map_strategy','CURRENT_DOMINANT_CONFIRMED_WAVE'),node_zones=zones,
        resonance_bands=([b for b in bands if b['source_waves']>=2] if new else [b for b in bands if b['source_waves']>=2][:5]),
        total_bands=len(bands),resonance_band_count=sum(b['source_waves']>=2 for b in bands),
        displayed_limit=None if new else 5,primary_wave=mapping['primary_wave'],
        temporary_waves=mapping['temporary_waves'][-3:],temporary_wave_count=len(mapping['temporary_waves']),
        temporary_waves_truncated=len(mapping['temporary_waves'])>3,
        wave_scope='EXISTING_PRICE_MAP_RECENT_PEAK60_PRIOR_LOW60_NOT_ALL_HISTORY',
        cluster_max_span_fraction=mapping['cluster_max_span_fraction'],
        source_count_note='不同低点/高点端点对计来源；共享高点相关，不是独立资金票数',
        zone_note=('V4至少三端点来源；观察区上沿=min(原带上沿+0.2ATR,昨收下跌超过3%的最高分价)，多档全列；共振参考不等于入场条件' if new else
            '节点至节点+0.2ATR，盘后上沿取不高于P收盘价；共振带另列，均非买入触发'))


def compute(request):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import pandas as pd
    code=request['code'];check_code(code)
    project=Path(os.environ.get('STARGAZER_PROJECT_ROOT',Path(__file__).resolve().parents[1]))
    risk=project/'stargazer_ml/v5_hybrid/risk_r0_v1';green=risk.parent/'green_common_profile'
    sys.path[:0]=[str(risk),str(green)]
    from joint_shadow_service_v1 import verify_fingerprints
    manifest=json.loads(gzip.decompress((risk/FINGERPRINT).read_bytes()))
    mismatch,ok=verify_fingerprints(manifest)
    if not ok:raise ValueError('RUNTIME_FINGERPRINT_MISMATCH')
    import joint_shadow_compute_v2 as source
    import tarot_live_v2 as tarot
    from golden_resonance_v4 import build_map as price_map
    now=datetime.fromisoformat(request['as_of']) if request.get('as_of') else datetime.now(ZoneInfo('Asia/Shanghai'))
    if now.tzinfo is None:raise ValueError('OBSERVATION_TIMEZONE_REQUIRED')
    now=now.astimezone(ZoneInfo('Asia/Shanghai'));day=int(now.strftime('%Y%m%d'))
    from tdx_daily_source_v2 import calendar_days
    calendar=calendar_days()
    target=max(d for d in calendar if d<=day and (d<day or now.hour>=15))
    if request.get('data_date') is not None:
        requested=int(request['data_date'])
        if requested not in calendar or requested>target:raise ValueError('COMPLETED_TRADING_DATE_REQUIRED')
        target=requested
    frame=source.read_daily_prefix(code,target)
    if frame.empty:raise ValueError('DAILY_SOURCE_MISSING')
    actual=int(frame.date.max());g=source.daily_prefix(frame,code,actual,require_metadata=False)
    core=tarot.frozen();atr=float(core.indicators(g).atr.iloc[-1])
    close=float(g.close.iloc[-1])
    try:mapping=price_map(g,atr)
    except ValueError as error:
        if str(error)!='NO_CAUSAL_WAVE_MAP':raise
        prefix=g[['code','date','open','high','low','close']].to_json(orient='records',double_precision=15)
        return dict(code=code,status='NO_CONFIRMED_MAP',reason='现行最近显著波段规则未找到合格的已确认端点组合；不能虚构黄金节点或共振带',
            data_date=actual,requested_data_date=target,as_of=now.isoformat(),reference_close=close,atr_ref=atr,
            atr_basis='TAROT_FROZEN_TRUE_RANGE14_LAST5_MEDIAN',computed_now=True,
            node_zones=[],resonance_bands=[],plan_instruction=None,plan_eligible=False,
            source=dict(kind='TDX_LOCAL_RAW_DAILY',rows=len(g),through=actual,
                        price_prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest()),
            selection_score=None,risk_status='NOT_COMPUTED',plan_changed=False,orders_sent=False)
    result=summarize_map(mapping,close,atr)
    # Current wave references remain provisional; never enter confirmed bands/ladder.
    from golden_ladder_v3 import wave_levels
    result['temporary_references']=[]
    for wave in [w for w in mapping['temporary_waves'] if w['low_confirmed'] and not w['high_confirmed']][-2:]:
        points=[v for v in wave_levels(wave['low'],wave['high']) if v['kind']=='RETRACEMENT' and v['price']<=close]
        result['temporary_references'].append(dict(wave=wave,status='TEMPORARY_HIGH_NOT_CONFIRMED',
            formal_eligible=False,node_zones=[dict(node=v['price'],ratio=v['ratio'],zone_lower=v['price'],
                zone_upper=min(close,v['price']+.2*atr)) for v in sorted(points,key=lambda v:v['price'],reverse=True)]))
    prefix=g[['code','date','open','high','low','close']].to_json(orient='records',double_precision=15)
    nearest=next(iter(result['node_zones']),None)
    result.update(code=code,status='AVAILABLE' if actual==target else 'STALE_REFERENCE_ONLY',
        data_date=actual,requested_data_date=target,as_of=now.isoformat(),reference_close=close,atr_ref=atr,
        atr_basis='TAROT_FROZEN_TRUE_RANGE14_LAST5_MEDIAN',
        source=dict(kind='TDX_LOCAL_RAW_DAILY',rows=len(g),
                    through=actual,price_prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest()),
        map_basis='EXISTING_GOLDEN_PRICE_RULE_NOT_LEARNED_SCORE',computed_now=True,
        selection_admission='NOT_EVALUATED_BY_THIS_PRICE_TOOL',selection_score=None,risk_status='NOT_COMPUTED',
        plan_changed=False,orders_sent=False,plan_eligible=actual==target and nearest is not None,
        plan_instruction=(f'请把{code}加入次日计划 关注区间{nearest["zone_lower"]:.4f}-{nearest["zone_upper"]:.4f}'
                          if actual==target and nearest is not None else None))
    if result['plan_eligible']:
        observable=[{k:b[k] for k in ('lower','upper','zone_upper','source_waves','shared_peak_groups')}
                    for b in mapping['bands'] if b['observable']]
        from golden_watch_groups import watch_groups
        rule=dict(version='TAROT_A_SECOND_CASH_V2',eligible=True,policy=mapping['policy'],
            levels=mapping['levels'],f618=mapping['levels'][0],atr_ref=atr,close_p=close,
            p_date=actual,history_days=mapping['history_days'],observation_map=mapping,
            observation_bands=observable,combination_policy='POSITION_TWO_OF_THREE_O20_CROSS1_V1')
        rule['quote_observation_bands']=watch_groups(rule)
        result['execution_row']=dict(code=code,name=str(g.iloc[-1].get('name',code)),date=actual,
            setup='USER_GOLDEN_ANALYSIS',stage='USER',reference_price=close,atr_ref=atr,
            zone=dict(zone_lower=nearest['zone_lower'],zone_upper=nearest['zone_upper']),
            entry_rule=rule,selection_score=None,selection_admission='USER_REQUEST_NOT_MODEL_ADMISSION')
        future=[d for d in calendar if d>actual]
        result['monitor_date']=min(future) if future else None
        frozen_manifest=json.loads(gzip.decompress(tarot.MANIFEST.read_bytes()))
        result['rule_inputs']=[dict(path=k,sha256=v) for k,v in frozen_manifest['frozen_inputs'].items()]
    if nearest is None:result.update(status='REFERENCE_ONLY_NO_V4_ZONE',reason='历史档位可读，但暂无满足V4下杀幅度和三来源的入场观察区')
    return result


def stock_golden_levels(code):
    """Called by the assistant; local subprocess reads one stock, never publishes."""
    import paths
    check_code(code)
    app=Path(paths.APP_DIR).resolve()
    project=app.parent.parent if app.name=='dist' else app.parent
    project=Path(os.environ.get('STARGAZER_PROJECT_ROOT',project))
    python=Path(os.environ.get('STARGAZER_PYTHON') or project/'_dev_venv/Scripts/python.exe')
    if not python.is_file():python=project/'rule_venv/Scripts/python.exe'
    worker=Path(paths.resource('golden_stock_analysis.py'))
    if not python.is_file() or not worker.is_file():return dict(code=code,status='UNAVAILABLE',reason='LOCAL_WORKER_MISSING')
    try:
        done=subprocess.run([str(python),'-B','-X','utf8',str(worker)],input=json.dumps(dict(code=code)),
            capture_output=True,text=True,encoding='utf-8',timeout=60,
            env={**os.environ,'STARGAZER_PROJECT_ROOT':str(project),'OMP_NUM_THREADS':'1'},
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        result=json.loads(done.stdout)
        if done.returncode:raise ValueError('WORKER_EXIT')
        return result
    except (OSError,ValueError,subprocess.TimeoutExpired):
        return dict(code=code,status='UNAVAILABLE',reason='LOCAL_GOLDEN_WORKER_UNAVAILABLE',plan_changed=False)


if __name__=='__main__':
    try:value=compute(json.load(sys.stdin))
    except Exception as error:value=dict(status='UNAVAILABLE',reason=type(error).__name__+':'+str(error)[:160],plan_changed=False)
    print(json.dumps(value,ensure_ascii=False,allow_nan=False))
