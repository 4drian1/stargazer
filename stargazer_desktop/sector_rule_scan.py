"""Read-only sector A/SECOND scan. Parent cache and isolated local worker; no HTTP."""
from datetime import datetime
from zoneinfo import ZoneInfo
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading

FINGERPRINT='live_monitor_fingerprints_20261110_v206_flow.json.gz'
ZONE=ZoneInfo('Asia/Shanghai')
CACHE={}
LOCK=threading.RLock()
SCAN_LOCK=threading.Lock()
RUNNING=False
GATE_REASONS={'VOLUME_GATE_NOT_PASSED':'量能结构未过门',
    'PANIC_VOLUME_BELOW_1_25X':'放量阴线不足前八日均量的1.25倍',
    'FIRST_LEG_VOLUME_NOT_DECREASING':'首段三日下跌成交量未逐日递减',
    'RISE_NOT_ABOVE_15PCT':'前段上涨未超过15%',
    'RETRACE_OUTSIDE_23_6_TO_78_6':'未进入23.6%至78.6%回撤区间'}


def sha(file):
    with Path(file).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def project_root():
    import paths
    app=Path(paths.APP_DIR).resolve()
    return Path(os.environ.get('STARGAZER_PROJECT_ROOT',app.parent.parent if app.name=='dist' else app.parent))


def source_context(project, as_of):
    """Use completed calendar days, never a partial daily bar or tomorrow's data."""
    now=datetime.fromisoformat(as_of)
    if now.tzinfo is None:raise ValueError('OBSERVATION_TIMEZONE_REQUIRED')
    now=now.astimezone(ZONE);today=int(now.strftime('%Y%m%d'))
    calendar=Path(importlib.util.find_spec('akshare').origin).parent/'file_fold/calendar.json'
    days=sorted(map(int,json.loads(calendar.read_bytes())))
    if today>days[-1]:raise ValueError('TRADING_CALENDAR_EXPIRED')
    p=max(d for d in days if d<today or (d==today and now.hour>=15))
    e=next(d for d in days if d>p)
    from tdx_daily_source_v2 import daily_directory
    folder=daily_directory(project,p)
    parts=sorted(folder.glob('part_*.parquet'))
    if not parts:raise ValueError('LOCAL_DAILY_SOURCE_MISSING')
    risk=project/'stargazer_ml/v5_hybrid/risk_r0_v1'
    scope=risk/'ths_full_concept_scope_20261017_v1.json'
    meta=json.loads(scope.read_bytes())
    catalog=project/'stargazer_ml/v5_hybrid/opportunity_v2/o5_sector_lag/_upstream/stockdb_boards_snapshot.parquet'
    if not catalog.is_file():catalog=Path(meta['source'])
    def identity(path):
        st=path.stat();return [str(path),st.st_size,st.st_mtime_ns]
    identity_value=[p,e,*[identity(f) for f in [*parts,catalog,scope]],
                    ['math_deps', math_dependency_shas(project, risk/FINGERPRINT)]]
    key=hashlib.sha256(json.dumps(identity_value).encode()).hexdigest()
    return dict(P=p,E=e,parts=parts,catalog=catalog,scope=meta,key=key)


# 计算缓存身份只绑实际数学依赖的内容 SHA；发布清单继续严格校验完整性，
# 但其文件本身（及仅改 UI/推送/文案文件的发布）不进入计算身份。
_MATH_DEP_DESKTOP = frozenset({
    'sector_rule_scan.py', 'strong_sector_watch_v1.py', 'strong_sector_rules_v1.py',
    'strong_sector_archive_v1.py', 'strong_low_indicators_v1.py',
    'strong_pullback_policy_v1.py', 'strong_pullback_context_v1.py',
    'relative_turnover_v1.py', 'tdx_native_history_v1.py',
    'golden_rules.py', 'golden_monitor.py',
    'tarot_live_v2.py',
})
# 发布引用行（每次发布切换版本字符串）不参与数学身份：归一化后计算 SHA。
_RELEASE_REF_LINE = re.compile(rb"(?m)^(FINGERPRINT|MANIFEST)\s*=\s*[^\n#]*$")


def _math_file_sha(path):
    """数学依赖文件的内容 SHA；FINGERPRINT/MANIFEST 顶层引用行归一化，
    发布切换不影响身份，其余任何内容变化都影响。"""
    raw = re.sub(_RELEASE_REF_LINE, rb'\1 = "<release-reference>"',
                 Path(path).read_bytes())
    return hashlib.sha256(raw).hexdigest()


def math_dependency_shas(project, manifest_file):
    """从发布清单取数学依赖（stargazer_ml/** 及桌面数学白名单文件）的内容 SHA。
    缺文件即失败，不跳过。"""
    import gzip
    root = Path(project)
    try:
        manifest = json.loads(gzip.decompress(Path(manifest_file).read_bytes()))
    except (OSError, ValueError):
        raise ValueError('MATH_DEPENDENCY_MANIFEST_MISSING')
    entries = (manifest.get('frozen_inputs') or {}) if 'frozen_inputs' in manifest else manifest
    deps = {}
    for rel in sorted(entries):
        if rel.startswith('stargazer_ml/') or (
                rel.startswith('stargazer_desktop/') and Path(rel).name in _MATH_DEP_DESKTOP):
            file = root / rel
            if not file.is_file():
                raise ValueError('MATH_DEPENDENCY_MISSING:' + rel)
            deps[rel] = _math_file_sha(file)
    if not deps:
        raise ValueError('MATH_DEPENDENCY_EMPTY')
    return deps


def resolve_board(catalog, name):
    name=name.strip();folded=name.casefold()
    exact=catalog[(catalog.name.str.casefold()==folded)|(catalog.code.str.casefold()==folded)]
    hits=exact if not exact.empty else catalog[catalog.name.str.casefold().str.contains(folded,regex=False)]
    if len(hits)!=1:
        return None,dict(status='AMBIGUOUS_BOARD' if len(hits)>1 else 'BOARD_NOT_FOUND',
            board_requested=name,choices=[dict(name=r.name,code=r.code,source=r.source,type=r.type) for r in hits.itertuples(index=False)],
            reason='请指定板块全名或目录代码，不将歧义当作没有合格股票')
    return hits.iloc[0],None


def evaluate_stock(g, live, core, model, model_sha):
    """Same gates and features as the live selector; manual review never promotes admission."""
    import numpy as np
    import pandas as pd
    from tarot_manual_review_v1 import latest_record as near_miss
    from descending_tops_v1 import for_frame
    if for_frame(g)['excluded']:return None,'EXCLUDE_DESCENDING_TRIPLE'
    if (not np.isfinite(g[['open','high','low','close','volume']].to_numpy(dtype=float)).all()
            or g[['open','high','low','close']].le(0).any().any()
            or (g.low>g[['open','close']].min(axis=1)).any()
            or (g.high<g[['open','close']].max(axis=1)).any()):
        return None,'INVALID_CAUSAL_DAILY_VALUES'
    last=g.iloc[-1]
    try:required_known=np.isfinite([float(last[k]) for k in ('pre_close','amount','is_st','float_mv')]).all()
    except (TypeError,ValueError):required_known=False
    if not required_known:return None,'MISSING_FINAL_DAILY_FIELDS'
    if not (last.close<last.pre_close and last.is_st==0 and last.float_mv>0
            and last.amount>=5e7 and last.volume>0):return None,None
    a,_=live.process_latest_code(g)
    record=live.latest_record(g,core,a.to_dict('records'));review=record is None
    if review:record=near_miss(g,core,a.to_dict('records'),live.trading_calendar())
    if record is None:return None,None
    features=pd.DataFrame([{k:record[k] for k in core.FEATURES}])
    if np.isinf(features.to_numpy(dtype=float)).any():return None,'INFINITE_FEATURE'
    score=float(model.predict(features)[0])
    if not np.isfinite(score):return None,'NONFINITE_MODEL_SCORE'
    try:row=live.row_for(g,record,core,score,model_sha,policy_v3=True)
    except ValueError as exc:
        if str(exc)!='NO_CAUSAL_WAVE_MAP':raise
        return None,'NO_CAUSAL_WAVE_MAP'
    return dict(code=row['code'],name=row['name'],stage=row['stage'],reference_close=row['reference_price'],
        selection_score=score,status='MANUAL_REVIEW_ONLY' if review else row['status'],
        formally_admitted=not review,failed_gates=record.get('failed_gates',[]),
        failed_gate_reasons=[GATE_REASONS.get(k,k) for k in record.get('failed_gates',[])],
        score_scope='OUTSIDE_FROZEN_ADMISSION_RESEARCH_REFERENCE' if review else 'CURRENT_FROZEN_A_SECOND',
        observation_bands=row['entry_rule'].get('observation_bands',[]),
        atr_ref=row['atr_ref'],position_status='AVAILABLE' if row['zone'] else 'NO_V4_OBSERVATION_ZONE',
        automatic_trigger=False,risk_status='NOT_COMPUTED',_execution_row=row),None


def ranked_candidates(formal, manual, top):
    key=lambda r:(-r['selection_score'],r['code'])
    formal.sort(key=key);manual.sort(key=key)
    chosen=copy.deepcopy((formal or manual)[:top])
    for rank,row in enumerate(chosen,1):row['rank']=rank
    return chosen


def checked_chat_reply(text, results):
    """Keep this tool's counts code-owned; a bad explanation does not break chat."""
    ready=[r for r in results if r.get('status')=='READY']
    if not ready:return text
    fields=('member_count','loaded_member_count','evaluated_count','total_catalog_members',
            'unsupported_market_count','formal_count','manual_review_count','returned_count',
            'requested_count','shortfall','scored_count')
    counts={int(r[k]) for r in ready for k in fields if k in r}
    counts.update(len(r.get('excluded',[])) for r in ready)
    invalid=any(int(n) not in counts for n in re.findall(r'(\d+)\s*(?:只|名成员)',text))
    scored={r.get('scored_count',r['formal_count']+r['manual_review_count']) for r in ready}
    invalid=invalid or any(int(n) not in scored for n in re.findall(r'实际评分\s*(\d+)',text))
    if not invalid:return text
    return render_scan_results(ready, 'AI解读的数量口径与扫描事实不一致，本次未采纳该解释；以下为本地实算结果。')


def render_scan_results(ready, notice):
    """Deterministic tool facts, also available when AI interpretation fails."""
    lines=[notice]
    for r in ready:
        lines.extend([f"{r['board']}｜日线{r['data_date']}｜观察日{r['observation_date']}",
            f"正式入围{r['formal_count']}只；可复核{r['manual_review_count']}只；产出可用模型分数{r.get('scored_count',r['formal_count']+r['manual_review_count'])}只。",
            f"本次展示{r['returned_count']}/{r['requested_count']}只；归档成员{r.get('total_catalog_members',r['member_count'])}，模型支持范围{r['member_count']}，有效日线{r['evaluated_count']}。",
            '成员来自本地归档，未认证为当日完整实时成员。'])
        for row in r['rows']:
            zones='、'.join(f"{b['lower']:.4f}–{b['zone_upper']:.4f}" for b in row['observation_bands']) or '暂无V4观察区'
            lines.append(f"{row['rank']}. {row['code']} {row['name']}｜分数{row['selection_score']:.6f}｜观察位：{zones}")
            if not row['formally_admitted']:lines.append('未过门：'+'；'.join(row['failed_gate_reasons']))
    lines.append('分数不是胜率；Pyxis未计算，本次不自动加入计划、不构成买入触发；未自动重试。')
    return '\n'.join(lines)


def compute(request):
    import pandas as pd
    import pickle
    project=Path(request['project']);as_of=request['as_of']
    sys.path.insert(0,str(project/'stargazer_desktop'))
    context=source_context(project,as_of)
    if sha(context['catalog'])!=context['scope']['source_sha256']:raise ValueError('MEMBERSHIP_SOURCE_CHANGED')
    catalog=pd.read_parquet(context['catalog'])
    board,error=resolve_board(catalog,request['board_name'])
    if error:return error
    all_members={str(c).zfill(6) for c in board.symbols}
    members=sorted(c for c in all_members if re.fullmatch(r'(?:00|30|60|68)\d{4}',c))
    green=project/'stargazer_ml/v5_hybrid/green_common_profile'
    sys.path.insert(0,str(green))
    import tarot_live_v2 as live
    core=live.frozen();live.verified_manifest(core)
    # The runtime manifest also binds the worker, model and parent application sources.
    manifest=json.loads(gzip.decompress((green.parent/'risk_r0_v1'/FINGERPRINT).read_bytes()))
    if any(sha(project/p)!=expected for p,expected in manifest.items()):raise ValueError('RUNTIME_FINGERPRINT_MISMATCH')
    model_sha=core.sha(live.MODEL)
    with live.MODEL.open('rb') as f:model=pickle.load(f)
    formal=[];manual=[];execution=[];excluded=[];seen=set();evaluated=0
    # Predicate pushdown reads named members only, in bounded batches; no market-wide scoring.
    for start in range(0,len(members),32):
        batch=members[start:start+32]
        frames=[pd.read_parquet(p,columns=['code','name',*core.NUMERIC],
            filters=[('code','in',batch),('date','<=',context['P'])]) for p in context['parts']]
        daily=pd.concat(frames,ignore_index=True)
        for code,g in daily.groupby('code',sort=False):
            code=str(code).zfill(6);seen.add(code)
            duplicates=g[g.duplicated('date',keep=False)]
            if any(len(d.drop_duplicates())!=1 for _,d in duplicates.groupby('date')):
                excluded.append(dict(code=code,reason='CONFLICTING_DAILY_SOURCE'));continue
            g=g.drop_duplicates('date').sort_values('date').reset_index(drop=True)
            if len(g)<70 or int(g.date.iloc[-1])!=context['P']:
                excluded.append(dict(code=code,reason='INCOMPLETE_CURRENT_DAILY_HISTORY'));continue
            row,reason=evaluate_stock(g,live,core,model,model_sha)
            if reason:excluded.append(dict(code=code,reason=reason))
            else:
                evaluated+=1
                if row:
                    full=row.pop('_execution_row')
                    if row['formally_admitted']:execution.append(full)
                    (formal if row['formally_admitted'] else manual).append(row)
    excluded.extend(dict(code=c,reason='DAILY_SOURCE_MISSING') for c in members if c not in seen)
    if context['key']!=source_context(project,as_of)['key']:raise ValueError('SOURCE_CHANGED_DURING_SCAN')
    chosen=ranked_candidates(formal,manual,request['top'])
    result=dict(status='READY' if evaluated else 'DATA_UNAVAILABLE',board=board['name'],board_code=board['code'],board_source=board['source'],
        data_date=context['P'],observation_date=context['E'],as_of=as_of,source_key=context['key'],
        member_count=len(members),loaded_member_count=len(seen),evaluated_count=evaluated,excluded=excluded,
        total_catalog_members=len(all_members),unsupported_market_count=len(all_members)-len(members),
        market_scope='冻结模型仅支持00/30/60/68代码范围，其余成员未评分，不能当作规则不通过',
        formal_count=len(formal),manual_review_count=len(manual),
        scored_count=len(formal)+len(manual),daily_validated_count=evaluated,
        coverage_note='evaluated_count/daily_validated_count仅为日线数据核验数；scored_count为产出可用模型分数的数量，不是全部成员都已打分',
        mode='FORMAL_CANDIDATES' if formal else 'MANUAL_REVIEW_ONLY',rows=chosen,
        returned_count=len(chosen),requested_count=request['top'],shortfall=max(0,request['top']-len(chosen)),
        membership=dict(sha256=context['scope']['source_sha256'],inspected_at=context['scope']['inspected_at_utc'],
            current_refresh_verified=False,historical_membership_verified=False,
            note='本地归档成员；未认证为当日完整实时成员，不作为历史回测成员证书'),
        model=dict(version=live.VERSION,sha256=model_sha,semantics='H3候选资金收益估计，非概率；无二段加分'),
        scope='COMPLETED_DAILY_RULE_SCAN_NOT_INTRADAY_TRIGGER',
        position_note='V4三来源低位共振加ATR的多档观察区；无档不造价，触位后仍须组合信号及同截止风险核验',
        plan_changed=False,orders_sent=False)
    if request.get('for_monitor'):
        result.update(monitor_rows=execution,
            monitor_rule_inputs=[dict(path=f,sha256=h) for f,h in live.verified_manifest(core)['frozen_inputs'].items()])
    return result


def scan_sector_rules(board_name, top=5, as_of=None, *, for_monitor=False):
    import paths
    if not isinstance(board_name,str) or not board_name.strip() or len(board_name)>80:raise ValueError('板块名称无效')
    if type(top) is not int or not 1<=top<=10:raise ValueError('展示数量须为1至10')
    aliases={'pcb':'PCB概念','cpo':'共封装光学(CPO)','煤炭':'煤炭概念'}
    if for_monitor:aliases['存储']='存储芯片'
    board_name=aliases.get(board_name.casefold(),board_name)
    def view(result):
        result=copy.deepcopy(result)
        if not for_monitor:
            result.pop('monitor_rows',None);result.pop('monitor_rule_inputs',None)
        return result
    as_of=as_of or datetime.now(ZONE).isoformat();project=project_root()
    try:context=source_context(project,as_of)
    except (OSError,ValueError,StopIteration) as exc:
        return dict(status='UNAVAILABLE',board_requested=board_name,reason=str(exc),rows=[])
    key=(context['key'],board_name,top)
    with SCAN_LOCK:
        with LOCK:
            if key in CACHE:return view(CACHE[key])
        python=project/'_dev_venv/Scripts/python.exe'
        if not python.is_file():python=project/'rule_venv/Scripts/python.exe'
        worker=Path(paths.resource('sector_rule_scan.py'))
        if not python.is_file() or not worker.is_file():return dict(status='UNAVAILABLE',reason='LOCAL_WORKER_MISSING',rows=[])
        request=dict(project=str(project),board_name=board_name,top=top,as_of=as_of,for_monitor=True)
        try:
            done=subprocess.run([str(python),'-B','-X','utf8',str(worker)],input=json.dumps(request),
                capture_output=True,text=True,encoding='utf8',timeout=240,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'},
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            result=json.loads(done.stdout)
            if done.returncode:raise ValueError('WORKER_EXIT')
        except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
            result=dict(status='UNAVAILABLE',board_requested=board_name,reason=type(exc).__name__,rows=[])
        # Same-day source failures remain visible instead of a polling retry loop.
        with LOCK:
            CACHE[key]=result
            for old in list(CACHE):
                if old[0]!=key[0]:del CACHE[old]
        return view(result)


def monitor_state(names, now, *, include_monitors=False):
    """Schedule scans apart from quote/holdings threads; reuse chat's source cache."""
    global RUNNING
    # These resolve established user shorthand; ambiguous other names remain
    # explicit choices. Storage-chip preference was specified by the user.
    aliases={'pcb':'PCB概念','cpo':'共封装光学(CPO)','存储':'存储芯片','煤炭':'煤炭概念'}
    resolved={n:aliases.get(n.casefold(),n) for n in names}
    try:context=source_context(project_root(),now.isoformat())
    except (OSError,ValueError,StopIteration) as exc:return [dict(board_requested=n,status='UNAVAILABLE',reason=str(exc),rows=[]) for n in names]
    with LOCK:
        missing=[n for n in names if (context['key'],resolved[n],5) not in CACHE]
        rows=[copy.deepcopy(CACHE.get((context['key'],resolved[n],5),dict(status='LOADING',rows=[])))|{'board_requested':n} for n in names]
        if missing and not RUNNING:
            RUNNING=True
            def work():
                global RUNNING
                try:
                    for n in missing:scan_sector_rules(resolved[n],as_of=now.isoformat(),for_monitor=True)
                finally:
                    with LOCK:RUNNING=False
            threading.Thread(target=work,daemon=True,name='confirmed-board-rule-scan').start()
        if not include_monitors:
            for row in rows:
                row['monitor_candidate_count']=len(row.pop('monitor_rows',[]))
                row.pop('monitor_rule_inputs',None)
        return rows


if __name__=='__main__':
    try:result=compute(json.load(sys.stdin))
    except Exception as exc:result=dict(status='UNAVAILABLE',reason=type(exc).__name__+':'+str(exc)[:180],rows=[])
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
