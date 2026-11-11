"""Authorized live A/SECOND adapter of the frozen 20261023 cash-return head. No fit."""
import argparse
from datetime import datetime
import importlib.util
import gzip
import json
import os
import pickle
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace
from functools import lru_cache

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[2]/'stargazer_desktop')]
from gcp_demo_pools import process_latest_code
from gcp_demo_snapshot import calendar_file,plan_dates
from golden_rules import digest

VERSION='TAROT_A_SECOND_CASH_V2'
FIT=HERE/'tarot_a_second_leg_fit_20261022_v1'
MODEL=FIT/'initial_node_cash_fit_20261023_v1/model.pkl'
MANIFEST=HERE/'tarot_live_manifest_v201_20261111_rel.json.gz'


def _project_root():
    """仓库根：STARGAZER_PROJECT_ROOT 优先，否则按本文件位置推导（支持非 D:\\stargazer 根）。"""
    return Path(os.environ.get('STARGAZER_PROJECT_ROOT',
                               Path(__file__).resolve().parents[3]))


def verified_manifest(core):
    manifest=json.loads(gzip.decompress(MANIFEST.read_bytes()))
    root=_project_root()
    for file,sha in manifest['frozen_inputs'].items():
        p=Path(file)
        if not p.is_absolute():
            p=root/p
        if core.sha(p)!=sha:
            raise ValueError('TAROT_FROZEN_INPUT_CHANGED:'+Path(file).name)
    return manifest


@lru_cache(maxsize=1)
def trading_calendar():
    return sorted(map(int,json.loads(calendar_file().read_bytes())))


def frozen():
    spec=importlib.util.spec_from_file_location('tarot_frozen_domain_20261022',FIT/'run.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def latest_second(g,core):
    # Only t in [P-7,P-5] can produce P in the frozen t+5..t+7 loop.
    last=len(g)-1
    numpy=SimpleNamespace(flatnonzero=lambda flags:np.flatnonzero(flags & (np.arange(len(flags))>=last-7)))
    namespace=dict(core.second_candidates.__globals__,np=numpy)
    signals=FunctionType(core.second_candidates.__code__,namespace)(g,max_p=int(g.date.iloc[-1]))
    return {k:v for k,v in signals.items() if k==last}


def latest_record(g,core,old_a=None):
    """Only the P-date cap changes; shape/feature/position kernels remain frozen."""
    from descending_tops_v1 import for_frame
    if for_frame(g)['excluded']:return None
    p=int(g.date.iloc[-1]);last=len(g)-1
    expected=[day for day in trading_calendar() if day<=p][-70:]
    if len(expected)!=70 or not np.array_equal(g.date.iloc[-70:].to_numpy(dtype=int),expected):
        return None
    if old_a is None:
        a,_=process_latest_code(g);old_a=a.to_dict('records')
    namespace=dict(core.candidates.__globals__,P_END=p)
    namespace['second_candidates']=lambda frame:latest_second(frame,core)
    function=FunctionType(core.candidates.__code__,namespace)
    rows,_=function(g,old_a)
    return next((r for r in rows if r['P']==p and r['position_ready']),None)


def row_for(g,rec,core,score,model_sha,policy_v3=True):
    last=g.iloc[-1];atr=float(core.indicators(g).atr.iloc[-1])
    if rec['stage']=='A':
        anchor=int(g.index[g.date==rec['anchor_date']][0])
        wave=core.new_conditions(g,anchor,atr)['wave']
        high,low=wave['peak_high'],wave['pivot_low']
        levels=[round(float(high-r*(high-low)),4) for r in core.RATIOS]
    else:
        high,low=core.cycle_history(g)[-1]
        levels=[float(high-r*(high-low)) for r in core.RATIOS]
    close=float(last.close)
    levels=sorted({v for v in levels if 0<v<=close+1e-10},reverse=True)
    if not levels or abs(levels[0]-rec['node'])>1e-8:
        raise ValueError('FROZEN_NODE_LADDER_MISMATCH')
    node=levels[0]
    from golden_resonance_v4 import build_map
    observation_map=build_map(g,atr) if policy_v3 else None
    model_reference_node=node
    if observation_map:levels=observation_map['levels'];node=levels[0] if levels else model_reference_node
    rule=dict(version=VERSION,eligible=True,f618=node,atr_ref=atr,levels=levels,
        anchor_hi=float(high),anchor_lo=float(low),close_p=close,stage=rec['stage'],
        condition='RSI_STOP_OR_TWO_COMPLETED_5M_RECLAIM_WITH_SAME_CUTOFF_RISK',
        scope='P日冻结阶梯；触位观察，破位继续下一档；旧档保留用于收复确认')
    if observation_map:
        observable=[b for b in observation_map['bands'] if b['observable']]
        rule.update(policy=observation_map['policy'],observation_map=observation_map,
            observation_bands=[{k:b[k] for k in ('lower','upper','zone_upper','source_waves','shared_peak_groups')} for b in observable],
            p_date=int(last.date),history_days=observation_map['history_days'],
            model_reference_node=model_reference_node,
            anchor_hi=observation_map['primary_wave']['high'],anchor_lo=observation_map['primary_wave']['low'],
            combination_policy='POSITION_TWO_OF_THREE_O20_CROSS1_V1',
            condition='POSITION_REQUIRED_REPAIR_MOMENTUM_ONE_CROSS_OVERSOLD20_ANY_TWO_WITH_RISK')
        from golden_watch_groups import watch_groups
        rule['quote_observation_bands']=watch_groups(rule)
    value=dict(code=str(last.code),name=str(last.get('name','')),date=int(last.date),
        setup='A_SDP' if rec['stage']=='A' else 'SECOND_PULLBACK',stage=rec['stage'],
        reference_price=close,atr_ref=atr,rsi14=rec['rsi14'],distance_atr=(close-node)/atr,
        zone=dict(zone_lower=node,zone_upper=min(close,node+.2*atr),zone_sources=['P_FROZEN_GOLDEN_LADDER']),
        entry_rule=rule,personal_score=float(score),selection_score=float(score),
        selection_model=dict(version=VERSION,sha256=model_sha,effect_gate='NOT_PASSED_USER_AUTHORIZED_ENABLE',
            semantics='H3_INITIAL_NODE_CANDIDATE_CASH_RETURN_ESTIMATE_NOT_PROBABILITY'),
        feature_values={k:(float(rec[k]) if np.isfinite(rec[k]) else None) for k in core.FEATURES},
        status='GOLDEN_PLAN_READY',signal='触黄金位观察；完成5分止跌背离或双根收复后核验同刻风险，人工决定')
    if observation_map:
        value['zone']=(dict(zone_lower=observable[0]['lower'],zone_upper=observable[0]['zone_upper'],
            zone_sources=['V4_LOW_RESONANCE_PLUS_ATR']) if observable else None)
        value['observation_map_status']='V4_TRIAL_EXECUTION_NOT_EXISTING_MODEL_TRAINING_TARGET'
        value['signal']='位置到达必要；位置修复、动能（跨日背离＋一次上穿＋RSI快线在上）、曾20%相对超卖，三选二；技术通过与DeepSeek风险意见分别推送，人工决定'
        if not observable:value.update(status='WAIT_RESONANCE_ZONE',distance_atr=None)
    return value


def daily_groups(parts,fields,p):
    """Merge every shard/increment for a code, with bounded prefix-batch RAM."""
    inventory=[];prefixes=set()
    for path in parts:
        codes=pd.read_parquet(path,columns=['code']).code.astype(str).str.zfill(6)
        local=set(codes.str[:3]);prefixes.update(local);inventory.append((path,local))
    current=False
    for prefix in sorted(prefixes):
        frames=[pd.read_parquet(path,columns=fields,filters=[('date','<=',p),
                ('code','>=',prefix+'000'),('code','<=',prefix+'999')])
                for path,local in inventory if prefix in local]
        daily=pd.concat(frames,ignore_index=True)
        if daily.empty:continue
        for _,duplicate in daily[daily.duplicated(['code','date'],keep=False)].groupby(['code','date']):
            if len(duplicate.drop_duplicates())!=1:raise ValueError('CONFLICTING_DAILY_SOURCE')
        daily=daily.drop_duplicates(['code','date']).sort_values(['code','date'])
        daily['code']=daily.code.astype(str).str.zfill(6)
        current=current or int(daily.date.max())==p
        for code,g in daily.groupby('code',sort=False):yield code,g
    if not current:raise ValueError('CURRENT_COMPLETED_DAILY_MISSING')


def build(daily_dir,now=None,through=None,policy_v3=False):
    now=now or datetime.now()
    calendar=sorted(map(int,json.loads(calendar_file().read_bytes())))
    p,e=plan_dates(calendar,now) if through is None else (int(through),next(d for d in calendar if d>through))
    core=frozen()
    manifest=verified_manifest(core)
    from tdx_daily_source_v2 import daily_directory
    daily_dir=daily_directory(_project_root(),p)
    with MODEL.open('rb') as source:model=pickle.load(source)
    parts=sorted(Path(daily_dir).glob('part_*.parquet'))
    fields=['code','name',*core.NUMERIC]
    rows=[];failures=[];manual=[];structure_excluded=0
    for code,g in daily_groups(parts,fields,p):
        if len(g)<70 or int(g.date.iloc[-1])!=p or not str(code).startswith(('00','30','60','68')):continue
        g=g.reset_index(drop=True)
        # Invalid historical prices cannot quietly become ordinary model features.
        if (not np.isfinite(g[['open','high','low','close','volume']].to_numpy(dtype=float)).all()
                or g[['open','high','low','close']].le(0).any().any()
                or (g.low>g[['open','close']].min(axis=1)).any()
                or (g.high<g[['open','close']].max(axis=1)).any()):
            failures.append(dict(code=str(code),reason='INVALID_CAUSAL_DAILY_VALUES'));continue
        last=g.iloc[-1]
        if not np.isfinite(last[['pre_close','is_st','float_mv','amount']].to_numpy(dtype=float)).all():
            failures.append(dict(code=str(code),reason='TDX_CURRENT_METADATA_UNKNOWN'));continue
        from descending_tops_v1 import for_frame
        descending=for_frame(g)
        if descending['excluded']:
            structure_excluded+=1;continue
        # Both frozen A and SECOND require a down P; outer gates are unchanged.
        if not (last.close<last.pre_close and last.is_st==0 and last.float_mv>0
                and last.amount>=5e7 and last.volume>0):continue
        a,_=process_latest_code(g)
        rec=latest_record(g,core,a.to_dict('records'))
        review_only = rec is None
        if review_only:
            from tarot_manual_review_v1 import latest_record as near_miss
            rec=near_miss(g,core,a.to_dict('records'),trading_calendar())
        if rec is None:continue
        features=pd.DataFrame([{k:rec[k] for k in core.FEATURES}])
        if np.isinf(features.to_numpy(dtype=float)).any():
            failures.append(dict(code=str(code),reason='INFINITE_FEATURE'));continue
        score=float(model.predict(features)[0])
        if not np.isfinite(score):raise ValueError('NONFINITE_MODEL_SCORE')
        try:
            row=row_for(g,rec,core,score,
                        manifest['frozen_inputs'][
                            MODEL.resolve().relative_to(_project_root().resolve()).as_posix()],
                        policy_v3=policy_v3)
            if review_only:
                row.update(status='MANUAL_REVIEW_ONLY',failed_gates=rec['failed_gates'],
                    review_scope='未过正式门；模型分数超出已验证适用范围，仅供人工复核；不自动触发',
                    automatic_trigger=False)
                row['selection_model']['admission_scope']='OUTSIDE_FROZEN_ADMISSION_RESEARCH_REFERENCE'
                manual.append(row)
            else:rows.append(row)
        except ValueError as error:
            if str(error)!='NO_CAUSAL_WAVE_MAP':raise
            failures.append(dict(code=str(code),reason=str(error)))
    rows.sort(key=lambda r:(-r['selection_score'],r['code']))
    for rank,row in enumerate(rows,1):row['rank']=rank
    manual.sort(key=lambda r:(-r['selection_score'],r['code']))
    for rank,row in enumerate(manual,1):row['rank']=rank
    return dict(schema='PAGE_OF_PENTACLES_DEMO_V1',strategy=VERSION,rule_version=VERSION,
        created_at=now.astimezone().isoformat(),data_date=p,expected_data_date=p,monitor_date=e,plan_ready=True,
        rows=rows[:10],candidate_population_count=len(rows),excluded_rows=failures,default_candidate_count=10,
        structure_excluded_count=structure_excluded,structure_exclusion_contract='DESCENDING_MAIN_PEAKS_60_W2_G3_D3_V1',
        manual_review_rows=manual[:max(0,10-len(rows[:10]))],manual_review_population_count=len(manual),
        manual_review_scope='单列人工复核，未过门不进入正式候选或自动买入触发，不足不填异常数据',
        ranking='一段/二段共同冻结模型分数降序；同分代码；无B母池、无二段加分',
        model_status='用户授权启用研究权重；统计效果门未通过，分数不是概率',
        source='TDX_LOCAL_COMPLETED_DAILY_RAW_MODEL_V2',
        daily_source=dict(contract='TDX_DAILY_RAW_MODEL_QFQ_SEPARATE_V2',directory=str(daily_dir),
            model_price_basis='RAW_OHLC_EVENT_ADJUSTED_PRE_CLOSE',history_training_eligible=False,
            qfq_view='SEPARATE_NOT_SILENTLY_SUBSTITUTED_IN_FROZEN_MODEL'),
        inputs=[dict(path=str(path),sha256=core.sha(path)) for path in [*parts,daily_dir/'COMPLETE.json',calendar_file()]],
        rule_inputs=[dict(path=file,sha256=sha) for file,sha in manifest['frozen_inputs'].items()])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--daily-dir',type=Path,required=True)
    parser.add_argument('--through',type=int)
    parser.add_argument('--golden-flow-v3',action='store_true')
    args=parser.parse_args()
    # Updates belong to the existing explicit post-close job, never an import or replay.
    print(json.dumps(build(args.daily_dir,through=args.through,policy_v3=args.golden_flow_v3),ensure_ascii=False,allow_nan=False))
