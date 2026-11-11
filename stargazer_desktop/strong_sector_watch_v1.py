"""Nightly frozen board pools and independent observation alerts. Never orders."""
from datetime import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import math
import contextlib
import secrets
import time
import cross_process_lock_v1 as cpl

CONTRACT='STRONG_SECTOR_PULLBACK_OBSERVATION_20261104_V1'
LOCK=threading.RLock()
THREADS={}
HEALTH={}


def root():
    import postclose_risk_runtime
    return postclose_risk_runtime.directory()/'strong_sector'


def encoded(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()
def digest(value):return hashlib.sha256(encoded(value)).hexdigest()


def save(file,value):
    from tarot_monitor_v2 import save as bounded_save
    bounded_save(file,value)


def archive(file,value):
    payload=encoded(value)
    if file.exists():
        if gzip.decompress(file.read_bytes())!=payload:raise ValueError('STRONG_ARCHIVE_CACHE_CHANGED')
        return
    from runtime_disk_limit_v1 import disk_limit,disk_usage,MONITOR_RESERVE
    import postclose_risk_runtime
    raw=gzip.compress(payload,mtime=0);base=postclose_risk_runtime.directory()
    if len(raw)>16*1024**2 or disk_usage(base)+len(raw)*2+MONITOR_RESERVE>disk_limit(base):
        raise ValueError('STRONG_ARCHIVE_BUDGET')
    with cpl.FileLock(file.parent / '.publish'):
        if file.exists():
            if gzip.decompress(file.read_bytes())!=payload:raise ValueError('STRONG_ARCHIVE_CACHE_CHANGED')
            return
        part=file.with_name(file.name+'.'+secrets.token_hex(8)+'.part')
        try:
            with part.open('xb') as f:f.write(raw)
            part.replace(file)
        finally:
            part.unlink(missing_ok=True)


def constituent_versions(names):
    from ths_board_identity_v1 import resolve_for_pool
    versions = {}
    for name in names:
        record = resolve_for_pool(name, root().parent / 'board_identities')
        if record is not None:
            versions[name] = (record.get('provenance') or {}).get('members_sha256') or 'PENDING'
    return versions


def frozen(names,day):
    matches=[]
    for file in (root()/'pools').glob(str(day)+'_*.json'):
        value=json.loads(file.read_bytes())
        if value.get('contract')!=CONTRACT or value.get('boards')!=sorted(set(names)):continue
        if value.get('constituent_versions', {}) != constituent_versions(names):continue
        if str(value['built_at'])>=str(day)+'093000':continue
        if digest(value['rows'])!=value['rows_sha256']:raise ValueError('STRONG_POOL_HASH')
        matches.append(value)
    return max(matches,key=lambda v:v['built_at']) if matches else None


def pool_status(names,now):
    value=frozen(names,int(now.strftime('%Y%m%d')))
    return dict(status='FROZEN' if value else 'WAIT_POSTCLOSE_POOL',count=len(value['rows']) if value else 0,
                health=dict(HEALTH),scope='Strong stocks only; separate from Tarot candidates and rankings')


def worker(request):
    project=Path(request['project']);sys.path[:0]=[str(project/'stargazer_desktop'),
        str(project/'stargazer_ml/v5_hybrid/green_common_profile'),str(project/'stargazer_ml/v5_hybrid/risk_r0_v1')]
    if request['mode']=='ARCHIVE':
        from strong_sector_archive_v1 import maintain
        from tdx_native_history_v1 import native_rows
        from runtime_disk_limit_v1 import disk_limit
        import postclose_risk_runtime
        runtime=postclose_risk_runtime.directory()
        return maintain(runtime,datetime.fromisoformat(request['as_of']),
            lambda code,day:native_rows(code,[day]),disk_limit(runtime))
    if request['mode']=='RULE_CONTEXT':
        import tarot_live_v2 as tarot
        tarot.verified_manifest(tarot.frozen())
        from strong_pullback_context_v1 import market_context
        return market_context(datetime.fromisoformat(request['as_of']),request['quote'],root().parent)
    return build_pool(request) if request['mode']=='POOL' else fetch_indicators(request)


def build_pool(request):
    import pandas as pd
    from sector_rule_scan import source_context,resolve_board,sha
    from strong_sector_rules_v1 import build
    import tarot_live_v2 as tarot
    project=Path(request['project']);ctx=source_context(project,request['as_of'])
    if sha(ctx['catalog'])!=ctx['scope']['source_sha256']:raise ValueError('BOARD_CATALOG_SHA')
    catalog=pd.read_parquet(ctx['catalog']);members={};warnings=[];versions={}
    aliases={'pcb':'PCB概念','cpo':'共封装光学(CPO)','存储':'存储芯片','煤炭':'煤炭概念'}
    for requested in sorted(set(request['boards'])):
        from ths_board_supply_v1 import ensure as prepare_constituents
        prepared = prepare_constituents(requested, root().parent / 'board_identities')
        if prepared and prepared.get('reason'):
            warnings.append(dict(status='CONSTITUENT_REFRESH_FAILED', board_requested=requested,
                                 reason=prepared['reason'], provenance=prepared.get('provenance')))
        from ths_board_identity_v1 import resolve_for_pool
        registered = resolve_for_pool(requested, root().parent / 'board_identities')
        if registered is not None:
            versions[requested] = (registered.get('provenance') or {}).get('members_sha256') or 'PENDING'
            if registered['members'] is None:
                warnings.append(dict(status='CONSTITUENTS_PENDING', board_requested=requested,
                                     reason='身份已确认，成分待准备，未冒充监控启用'))
                continue
            symbols = registered['members']
            warnings.append(dict(status='CONSTITUENT_PROVENANCE', board_requested=requested,
                                 provenance=registered['provenance']))
        else:
            board,error=resolve_board(catalog,aliases.get(requested.casefold(),requested))
            if error:warnings.append(error);continue
            symbols = board.symbols
        for code in symbols:
            import re
            code=str(code).zfill(6)
            if re.fullmatch(r'(?:00|30|60|68)\d{4}',code):members.setdefault(code,[]).append(requested)
    core=tarot.frozen();tarot.verified_manifest(core)
    rows=[];excluded=[];seen=set();evaluated=0
    for offset in range(0,len(members),32):
        codes=sorted(members)[offset:offset+32]
        frames=[pd.read_parquet(p,columns=['code','name',*core.NUMERIC,'float_share'],filters=[('code','in',codes),('date','<=',ctx['P'])]) for p in ctx['parts']]
        data=pd.concat(frames,ignore_index=True)
        for code,g in data.groupby('code',sort=False):
            code=str(code).zfill(6);seen.add(code)
            try:
                for _,dupes in g[g.duplicated('date',keep=False)].groupby('date'):
                    if len(dupes.drop_duplicates())!=1:raise ValueError('DAILY_CONFLICT')
                g=g.drop_duplicates('date').sort_values('date').reset_index(drop=True)
                if int(g.date.iloc[-1])!=ctx['P']:raise ValueError('CURRENT_DAILY_MISSING')
                atr=float(core.indicators(g).atr.iloc[-1]);value=build(g.to_dict('records'),ctx['P'],atr)
                evaluated+=1
                if value['pool']=='EXCLUDED':excluded.append(dict(code=code,reason=value['reason']));continue
                name=g.name.iloc[-1]
                if value.get('structure')=='TRIPLE' or (isinstance(name,str) and any(s in name.upper() for s in ('ST','退'))):
                    excluded.append(dict(code=code,reason='PULLBACK_TRIPLE_OR_KNOWN_ST_DELIST'));continue
                from strong_pullback_policy_v1 import build_map
                rule_map=build_map(g.to_dict('records'),ctx['P'])
                if not rule_map:excluded.append(dict(code=code,reason='NO_CAUSAL_PULLBACK_MAP'));continue
                value['pullback_map']=rule_map
                full_sha=digest(value);archive(root()/'maps'/(full_sha+'.json.gz'),value)
                item={k:value[k] for k in ['pool','structure','reference_close','atr_ref','high','low','wave_return','history_days','neckline']}
                item.update(code=code,name=str(g.name.iloc[-1]),boards=members[code],map_sha=full_sha,
                    positions=[dict({k:v for k,v in b.items() if k!='sources'},
                        node_lower=b['lower'],node_upper=b['upper'],
                        lower=max(.01,b['lower']-.2*atr),upper=b['upper']+.2*atr,
                        tolerance_basis='0.2_PREVIOUS_DAILY_ATR') for b in rule_map],
                    map_known_before_open=True,
                    risk_clear=True if isinstance(name,str) and name.strip() else None)
                from relative_turnover_v1 import specification
                item['turnover_reference']=specification(g.to_dict('records'),ctx['P'],tarot.trading_calendar())
                rows.append(item)
            except (ValueError,KeyError,TypeError) as exc:excluded.append(dict(code=code,reason=str(exc)))
    excluded.extend(dict(code=c,reason='DAILY_MISSING') for c in set(members)-seen)
    if versions != constituent_versions(request['boards']):raise ValueError('BOARD_MEMBERS_CHANGED_DURING_PREPARE')
    if ctx['key']!=source_context(project,request['as_of'])['key']:raise ValueError('SOURCE_CHANGED_DURING_PREPARE')
    if len(rows)>512:raise ValueError('STRONG_POOL_OVER_512_NO_SILENT_TRUNCATION')
    from zoneinfo import ZoneInfo
    ready_at=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d%H%M%S')
    from strong_pullback_context_v1 import market_context
    try:market_test=market_context(datetime.now(ZoneInfo('Asia/Shanghai')).replace(tzinfo=None),{},root().parent,ctx['E'])
    except (OSError,ValueError,KeyError) as exc:
        market_test=dict(state='UNKNOWN',level=None,t0=None,reason=str(exc)[:160])
    return dict(contract=CONTRACT,status='READY' if evaluated else 'DATA_UNAVAILABLE',data_date=ctx['P'],observation_date=ctx['E'],
        boards=sorted(set(request['boards'])),source_key=ctx['key'],built_at=ready_at,started_at=request['built_at'],rows=rows,
        rows_sha256=digest(rows),excluded=excluded,warnings=warnings,member_count=len(members),evaluated_count=evaluated,
        constituent_versions=versions,
        market_test=market_test,membership=dict(sha256=ctx['scope']['source_sha256'],current_verified=False,
            note='Local archived members, not certified current membership'),ranked_by_model=False)


def opening_preview(minutes,code,P,cutoff,quote):
    from strong_pullback_policy_v1 import fresh,finite
    from live_five_preview_v1 import comparisons
    stamp=datetime.strptime(str(cutoff),'%Y%m%d%H%M%S')
    quote=quote or {}
    if not 93000<=cutoff%1000000<93500:raise ValueError('OPENING_PREVIEW_FIRST_WINDOW_ONLY')
    if not fresh(quote,stamp):raise ValueError('OPENING_PREVIEW_QUOTE_INVALID')
    if not all(finite(quote.get(k)) and quote[k]>0 for k in ('open','high','low')):
        raise ValueError('OPENING_PREVIEW_OHLC_UNKNOWN')
    if not quote['low']<=min(quote['open'],quote['price'])<=max(quote['open'],quote['price'])<=quote['high']:
        raise ValueError('OPENING_PREVIEW_OHLC_INVALID')
    value=comparisons(minutes,code,P,cutoff,quote)
    value.update(provider=dict(contract='OPENING_QUOTE_OBSERVATION_NOT_MINUTE_RECORD',
        quote=quote,available_at=cutoff),volume_ratio=dict(status='UNKNOWN',reason='首根5分钟成交量尚未形成'))
    return value


# Reservation lifecycle state (Step2 闭环1): per-process set of attempts currently
# computing, plus the reservation TTL (= invoke timeout, 900s).  These make the
# four ATTEMPT_RESERVED states distinguishable: computing / owner-exited /
# timed-out / recorded-failure.
_ACTIVE_ATTEMPTS = set()
_ATTEMPT_TTL = 900.0


def fetch_indicators(request):
    """Reservation-aware entry for indicator context computation.

    The exact-cutoff attempt is reserved BEFORE any compute.  A pre-existing
    file is interpreted by status, never blindly returned:
      * real result (success or recorded DATA_INSUFFICIENT) -> returned as
        evidence, never recomputed and never mistaken for "already done";
      * ATTEMPT_RESERVED owned by another LIVE pid within TTL -> returned
        as-is (another worker is computing; never double-start);
      * ATTEMPT_RESERVED whose owner pid is confirmed dead, older than TTL
        (covers pid reuse), a same-process abandoned attempt, or a legacy
        placeholder without owner info -> reclaimed and recomputed (restart
        continuation), carrying the prior reservation as reclaimed_from.
    """
    row = request['row'];code = row['code'];cutoff = request['cutoff']
    day = cutoff // 1000000
    folder = root() / 'minutes' / str(day);folder.mkdir(parents=True, exist_ok=True)
    target = folder / (code + '_' + str(cutoff) + '_context_v2.json')
    with cpl.FileLock(target):
        prior = None
        if target.exists():
            try:existing = json.loads(target.read_bytes())
            except (OSError, ValueError):existing = None
            if isinstance(existing, dict) and existing.get('status') != 'ATTEMPT_RESERVED':
                return existing  # evidence preserved: success or recorded failure
            try:owner = int((existing or {}).get('owner_pid') or 0)
            except (TypeError, ValueError):owner = 0
            try:started = float((existing or {}).get('started_epoch') or 0.0)
            except (TypeError, ValueError):started = 0.0
            age = time.time() - started
            if owner and (owner != os.getpid() or (code, cutoff) in _ACTIVE_ATTEMPTS):
                try:
                    if cpl.pid_alive(owner):
                        return existing  # Age alone never authorizes concurrent recomputation.
                except OSError:
                    return existing  # Uncertain liveness is not proof of death.
            prior = dict(owner_pid=owner, started_epoch=started,
                         attempt_id=(existing or {}).get('attempt_id'),
                         reclaimed_reason=('OWNER_EXITED' if owner and not cpl.pid_alive(owner)
                                           else 'TIMEOUT_OR_STALE' if started else 'LEGACY_PLACEHOLDER'))
        attempt_id = secrets.token_hex(8)
        save(target, dict(status='ATTEMPT_RESERVED', code=code, asof=cutoff, periods={},
                          owner_pid=os.getpid(), started_epoch=time.time(),
                          attempt_id=attempt_id, reclaimed_from=prior))
        _ACTIVE_ATTEMPTS.add((code, cutoff))
    try:
        # 回收留痕写独立旁证文件：结果保存会覆盖预约占位，旁证不受影响、可审计。
        if prior is not None:
            try:
                save(Path(str(target) + '.reclaim.json'),
                     dict(attempt_id=attempt_id, code=code, asof=cutoff,
                          reclaimed_from=prior, reclaimed_at=time.time()))
            except OSError:
                pass
        value = _fetch_indicators_impl(request)
        try:
            if isinstance(value, dict) and value.get('asof') == cutoff:
                stored = json.loads(target.read_bytes())
                if isinstance(stored, dict) and stored.get('status') == value.get('status'):
                    stored.setdefault('attempt_id', attempt_id)
                    save(target, stored)
        except (OSError, ValueError):
            pass
        return value
    finally:
        _ACTIVE_ATTEMPTS.discard((code, cutoff))


def _fetch_indicators_impl(request):
    import base64
    import urllib.request,urllib.parse
    import msgpack
    from gcp_golden_intraday_v1 import pairs,NoRedirect
    from public_stock_minutes_v1 import current_minutes
    from strong_low_indicators_v1 import compare
    from runtime_disk_limit_v1 import disk_limit
    import postclose_risk_runtime
    row=request['row'];code=row['code'];cutoff=request['cutoff'];day=cutoff//1000000
    folder=root()/'minutes'/str(day);folder.mkdir(parents=True,exist_ok=True)
    target=folder/(code+'_'+str(cutoff)+'_context_v2.json')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    minutes=[];receipts=[]
    try:
        from minute_history_v1 import DiskBudget,load_days
        budget=DiskBudget()
        minutes,history_receipts=load_days(code,row['history_days'],budget)
        receipts.extend(history_receipts)
        from live_minute_source_v1 import snapshot
        from live_five_preview_v1 import comparisons
        if 93000<=cutoff%1000000<93100:
            value=opening_preview(minutes,code,row['history_days'][-1],cutoff,request.get('quote'))
            value['source_receipts']=receipts;save(target,value);return value
        try:
            minutes,provider,live_refs=snapshot(postclose_risk_runtime.directory(),code,row['history_days'][-1],cutoff,row['reference_close'],minutes,budget)
        except ValueError as error:
            if str(error)!='LIVE_CURRENT_MINUTES_NOT_ARRIVED' or not 93000<=cutoff%1000000<93500:raise
            value=opening_preview(minutes,code,row['history_days'][-1],cutoff,request.get('quote'))
            value['source_receipts']=receipts;save(target,value);return value
        receipts.extend(live_refs)
        value=comparisons(minutes,code,row['history_days'][-1],cutoff,request.get('quote'))
        value.update(provider=provider,source_receipts=receipts)
        from relative_turnover_v1 import current
        value['relative_turnover']=current(postclose_risk_runtime.directory(),code,row.get('turnover_reference'),minutes,cutoff)
        from strong_pullback_context_v1 import volume_context
        try:value['volume_ratio']=volume_context(code,minutes,cutoff)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            value['volume_ratio']=dict(status='UNKNOWN',reason=str(exc)[:120])
    except (OSError,ValueError,KeyError,TypeError) as exc:value=dict(status='DATA_INSUFFICIENT',asof=cutoff,periods={},reason=str(exc)[:160])
    save(target,value);return value


def invoke(request):
    import paths
    from sector_rule_scan import project_root
    project=project_root();python=project/'_dev_venv/Scripts/python.exe'
    if not python.is_file():python=project/'rule_venv/Scripts/python.exe'
    body=dict(request,project=str(project))
    try:
        output=subprocess.run([str(python),'-B','-X','utf8',paths.resource('strong_sector_watch_v1.py')],
            input=encoded(body).decode(),capture_output=True,text=True,encoding='utf8',timeout=900 if request['mode']=='POOL' else 60,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
    except subprocess.TimeoutExpired:
        raise ValueError('STRONG_WORKER_TIMEOUT:mode=%s' % request.get('mode'))
    if output.returncode or len(output.stdout.encode())>2*1024**2:
        # 保留退出码、受限长度 stderr 与超限原因：不得把所有失败都摊平成 STRONG_WORKER_FAILED
        reason='EXIT_CODE_%s' % output.returncode if output.returncode else 'STDOUT_LIMIT_2MiB'
        stderr=(output.stderr or '')[-400:].replace('\n',' | ')
        raise ValueError('STRONG_WORKER_FAILED:%s:stderr_tail=%s' % (reason,stderr))
    return json.loads(output.stdout)


def candidate_events(row,quote,day,cutoff,indicator=None):
    """Only quote arrival gates; indicators describe completed bars, never force BUY."""
    import math
    if str(quote.get('time',''))[:8]!=str(day):return []
    p,prev=quote.get('price'),quote.get('prev')
    if not all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in (p,prev)):return []
    if abs(prev-row['reference_close'])>.011:return []
    pairs=(indicator or {}).get('periods') or {}
    current=(pairs.get('5') or {}).get('current') or {}
    if row['pool']=='LANDMARK':
        events=[]
        for b in row['positions']:
            ratio=b.get('landmark_ratio')
            if ratio==.786 and cutoff%1000000!=150000:continue
            if ratio==.5 and row['reference_close']<b['lower']:continue
            close=(indicator or {}).get('last_close',p)
            if indicator is not None and (indicator.get('status')!='AVAILABLE' or indicator.get('last_time')!=cutoff):continue
            if close<b['lower']:events.append(dict(kind='DELETE' if ratio==.786 else 'LANDMARK',band=b))
        return events
    def over4(price):
        decline=1-price/prev
        return decline>=.035-1e-12
    events=[]
    for b in row['positions']:
        if b.get('requires_neckline_break'):
            if indicator is None:
                if p>=row['neckline']:continue
            elif indicator.get('last_time')!=cutoff or indicator.get('last_close',float('inf'))>=row['neckline']:continue
        distance=max(b['lower']-p,p-b['upper'],0)
        low=quote.get('low',p)
        if not isinstance(low,(float,int)) or not math.isfinite(low) or not 0<low<=p:continue
        mode=None
        tolerance=0 if b.get('tolerance_basis') else .2*row['atr_ref']
        if over4(p) and distance<=tolerance+1e-10:mode='NOW_NEAR'
        if mode:events.append(dict(kind='STRONG_STOCK',band=b,arrival_basis=mode))
    def node_distance(event):
        b=event['band']
        lo=b.get('node_lower',b['lower']);hi=b.get('node_upper',b['upper'])
        return abs(p-(lo+hi)/2)
    return sorted(events,key=lambda e:(e.get('arrival_basis')!='NOW_NEAR',not e['band']['primary'],-e['band']['source_waves'],node_distance(e)))


def observe(plan,cfg,now,quotes,broadcast,send,active):
    from sector_watch_v1 import names
    board_names=names(plan);day=int(now.strftime('%Y%m%d'));pool=frozen(board_names,day)
    if not pool:return
    minute=now.hour*60+now.minute
    if not (570<=minute<=690 or 780<=minute<=906):return
    from strong_pullback_positions_v1 import load as load_positions, observe as observe_positions
    actual_positions=load_positions(root().parent)['positions']
    from strong_touch_lane_v1 import latest_quotes
    state_dir=root()/'touch'
    snapshot=latest_quotes(state_dir,now)
    extra=sorted({c for c,p in actual_positions.items() if p['quantity']>0}|{'sh000001'})
    missing=[c for c in extra if c not in snapshot]
    if missing:
        try:snapshot.update(quotes(missing))
        except Exception:pass
    # A batch response can be newer than the time at which collection began.
    # Validate against actual receipt time, retaining the strict no-future gate.
    now=datetime.now()
    if int(now.strftime('%Y%m%d'))!=day:return
    minute=now.hour*60+now.minute
    if not (570<=minute<=690 or 780<=minute<=906):return
    end=min(minute,900)
    cutoff=int(now.strftime('%Y%m%d%H%M%S')) if minute<900 else day*1000000+150000
    if cutoff%1000000==130000:cutoff=day*1000000+113000
    market_rule=None
    board_summary=[]
    for name in board_names:
        members=[r for r in pool['rows'] if name in r['boards']]
        valid=[]
        for r in members:
            q=snapshot.get(r['code']) or {}
            try:
                ts=datetime.strptime(str(q['time']),'%Y%m%d%H%M%S')
                if 0<=(now-ts).total_seconds()<=120 and all(type(q.get(k)) in (int,float) and math.isfinite(q[k]) and q[k]>0 for k in ('price','prev')):
                    valid.append(dict(code=r['code'],name=r['name'],return_value=q['price']/q['prev']-1,time=q['time']))
            except (KeyError,ValueError,TypeError):continue
        leader=max(valid,key=lambda x:x['return_value']) if valid else None
        board_summary.append(dict(name=name,leader=leader,drop4_count=sum(v['return_value']<-.04 for v in valid),
            mean_return=sum(v['return_value'] for v in valid)/len(valid) if valid else None,
            up_count=sum(v['return_value']>0 for v in valid),
            valid_count=len(valid),pool_count=len(members),scope='FROZEN_STRONG_POOL_NOT_WHOLE_BOARD'))
    save(root()/'summary.json',dict(asof=int(now.strftime('%Y%m%d%H%M%S')),boards=board_summary,pool_day=day))
    HEALTH.update(status='MONITORING',checked_at=now.isoformat(),pool_count=len(pool['rows']),
                  valid_quotes=sum(b['valid_count'] for b in board_summary))
    HEALTH.pop('reason',None)
    if any(p['quantity']>0 for p in actual_positions.values()):
        try:
            market_rule=invoke(dict(mode='RULE_CONTEXT',as_of=now.isoformat(),quote=snapshot.get('sh000001') or {}))
            if market_rule.get('state') not in ('RED','PROBE','ABOVE','UNKNOWN'):raise ValueError('PULLBACK_CONTEXT_NOT_READY')
        except (OSError,ValueError,KeyError,subprocess.SubprocessError):
            market_rule=dict(state='UNKNOWN',level=None,t0=None,t_plus=None)
        eligible_rows={r['code']:r for r in pool['rows'] if r['pool']=='MONITOR' and r.get('risk_clear') is True and r.get('structure')!='TRIPLE'}
        observe_positions(root().parent,cfg,now,snapshot,market_rule,
            lambda r,q:invoke(dict(mode='INDICATORS',row=eligible_rows.get(r['code'],r),cutoff=cutoff,quote=q)),
            broadcast,send,active,set(eligible_rows),refresh=quotes)

def schedule(plan,cfg,now,quotes,broadcast,send,active=lambda:True,*,prepare_only=False):
    from sector_watch_v1 import names
    if not active():return
    outside=prepare_only or now.hour>=15 or now.hour<9 or now.hour==9 and now.minute<30
    boards=names(plan or {})
    if outside:
        from plan_board_change_v1 import preparation_boards
        from zoneinfo import ZoneInfo
        observed=now.replace(tzinfo=ZoneInfo('Asia/Shanghai')) if now.tzinfo is None else now.astimezone(ZoneInfo('Asia/Shanghai'))
        boards=names({'boards':preparation_boards(plan,root().parent/'board_identities',observed)})
    if not boards:return
    with LOCK:
        # Archive historical sessions independently; slow review IO never blocks live quotes.
        today=now.strftime('%Y%m%d')
        old_pools=any(p.name[:8].isdigit() and p.name[:8]<today and not p.name.endswith('.failed.json')
                      for p in (root()/'pools').glob('????????_*.json'))
        if old_pools and HEALTH.get('archive_checked_day')!=today and now.timestamp()>=HEALTH.get('archive_retry_after',0) and not (THREADS.get('ARCHIVE') and THREADS['ARCHIVE'].is_alive()):
            def archive_work():
                try:
                    value=invoke(dict(mode='ARCHIVE',as_of=now.isoformat()))
                    if value.get('status')!='ARCHIVED':raise ValueError(value.get('reason','ARCHIVE_FAILED'))
                    HEALTH.update(archive_status='ARCHIVED',archive_checked_day=today,archive_days=len(value['days']))
                except Exception as exc:
                    HEALTH.update(archive_status='FAILED_CACHE_RETAINED',archive_reason=str(exc)[:180],
                                  archive_retry_after=now.timestamp()+300)
            archived=threading.Thread(target=archive_work,daemon=True,name='strong-sector-archive')
            THREADS['ARCHIVE']=archived;archived.start()
        key='POOL' if outside else 'OBSERVE'
        if THREADS.get(key) and THREADS[key].is_alive():return
        def work():
            try:
                if outside:
                    if not prepare_only and now.hour==15 and now.minute<=6 and names(plan or {}):
                        observe(plan,cfg,now,quotes,broadcast,send,active)
                    from ths_board_supply_v1 import ensure as prepare_constituents
                    for name in boards:
                        prepared=prepare_constituents(name,root().parent/'board_identities')
                        if prepared and prepared.get('reason'):
                            HEALTH.update(constituent_refresh_reason=prepared['reason'])
                    from sector_rule_scan import source_context,project_root
                    context=source_context(project_root(),observed.isoformat())
                    parts=[sorted(set(boards)),context['key'],'TURNOVER_REFERENCE_V1',CONTRACT]
                    versions=constituent_versions(boards)
                    if versions:parts.append({'ths_members':versions})
                    import strong_sector_pool_prep_v1 as pool_prep
                    pool_view=pool_prep.submit(boards, now=observed, context=dict(P=context['P'],E=context['E'],key=context['key']))
                    if pool_view.get('status') in ('READY','REUSED') and pool_view.get('pool'):
                        from minute_history_v1 import schedule as prepare_minutes
                        prepare_minutes(pool_view['pool'])
                        HEALTH.update(status='FROZEN',count=len(pool_view['pool']['rows']),
                                      observation_date=pool_view['pool'].get('observation_date'))
                    elif pool_view.get('status') in ('FAILED','WAIT_DAILY','RETRY_WAIT'):
                        HEALTH.update(status='POOL_'+pool_view['status'],
                                      reason=str(pool_view.get('reason'))[:160])
                else:observe(plan,cfg,now,quotes,broadcast,send,active)
            except Exception as exc:
                HEALTH.update(status='FAILED',reason=type(exc).__name__+':'+str(exc)[:160])
        t=threading.Thread(target=work,daemon=True,name='strong-sector-'+key.lower());THREADS[key]=t;t.start()


if __name__=='__main__':
    try:result=worker(json.load(sys.stdin))
    except Exception as error:result=dict(status='UNAVAILABLE',reason=type(error).__name__+':'+str(error)[:180])
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


ALERT_LOCK = threading.RLock()


@contextlib.contextmanager
def _alerts_lock():
    """Serialize alerts.json access across processes AND threads.

    The old code only held a per-process ``threading.RLock``; a second app
    instance (or any concurrent writer) could interleave the read-check-write and
    either corrupt the JSON or let two processes both pass the duplicate check
    and double-send.  We wrap the per-thread lock with a cross-process
    advisory file lock on ``alerts.json`` so the read-modify-write is atomic
    across processes too.
    """
    with cpl.FileLock(root() / 'alerts.json'):
        with ALERT_LOCK:
            yield


def _attempt_summary(rec):
    """Compact, evidence-preserving receipt for one reservation attempt."""
    return dict(attempt_id=rec.get('attempt_id'), owner=rec.get('owner'),
                delivery=rec.get('delivery'), wechat=bool(rec.get('wechat')),
                wechat_msg=str(rec.get('wechat_msg') or ''),
                sent_at=rec.get('sent_at'), send_started_at=rec.get('send_started_at'),
                send_finished_at=rec.get('send_finished_at'), error=rec.get('error'))


def _release_reservation(alerts_file, key, attempt_id, *, reason='RELEASED'):
    """Identity-verified cancel/delete of OUR OWN reservation.

    Never removes or mutates another attempt's reservation.  Evidence is kept (the
    attempt is archived into ``attempts`` and the key is marked ``reason`` rather
    than deleted) so a later re-delivery carries forward the full history.
    """
    with _alerts_lock():
        alerts = json.loads(alerts_file.read_bytes()) if alerts_file.exists() else {}
        cur = alerts.get(key)
        if cur is None:
            return
        if cur.get('attempt_id') != attempt_id:
            # Not our attempt — do not touch another caller's reservation.
            return
        cur['delivery'] = reason
        cur.setdefault('attempts', []).append(_attempt_summary(cur))
        save(alerts_file, alerts)


def _same_band(left, right):
    return (abs(float(left.get('lower', 0)) - float(right.get('lower', 0))) < 1e-8 and
            abs(float(left.get('upper', 0)) - float(right.get('upper', 0))) < 1e-8 and
            left.get('landmark_ratio') == right.get('landmark_ratio'))


def deliver_touch_event(event_id, state_file, plan_ref, *, get_plan, get_cfg,
                        quote_fn, broadcast, send, active):
    """Send a first-observed price event; indicators never gate the touch notice.

    任务书阶段1 P0: reverify_quote_at / send_started_at / send_finished_at
    事件级证据链时间戳由本函数写入 state event, 与 strong_touch_lane_v1 的
    poll() 事件创建时间戳 (scan_started_at / quote_attempt_at / quote_received_at
    / first_observed_touch_at / event_recorded_at) 形成完整链路.
    """
    import strong_touch_lane_v1 as lane
    event = lane.get_event(state_file, event_id)
    if not event:
        return
    # Re-verify reservation identity immediately: a non-owner (or a process that
    # has lost the state lease) must never enqueue/broadcast/send.
    if not lane.is_state_owner(state_file):
        return
    now = datetime.now()
    try:
        cfg = get_cfg() or {}
    except Exception as exc:
        lane.update_event(state_file,event_id,delivery_status='WAIT_REVERIFY',
                          delivery_retry_after_epoch=now.timestamp()+lane.POLL_INTERVAL_SECONDS,
                          reverify_error='CONFIG_UNREADABLE:'+type(exc).__name__)
        return
    if not (cfg.get('push') or {}).get('enabled'):
        lane.update_event(state_file,event_id,delivery_status='WAIT_REVERIFY',
                          delivery_retry_after_epoch=now.timestamp()+lane.POLL_INTERVAL_SECONDS,
                          reverify_error='PUSH_DISABLED_NO_ATTEMPT')
        return
    current_plan = get_plan() or {}
    from golden_rules import digest as plan_digest
    if plan_digest(current_plan) != plan_digest(plan_ref) or not active():
        lane.update_event(state_file, event_id, delivery_status='CANCELLED_PLAN_CHANGED',
                          delivery_finished_at=now.isoformat(timespec='seconds'),
                          send_finished_at=now.isoformat(timespec='seconds'))
        return
    from sector_watch_v1 import names
    pool = frozen(names(current_plan), int(now.strftime('%Y%m%d')))
    if not pool or pool.get('rows_sha256') != event.get('pool_id'):
        lane.update_event(state_file, event_id, delivery_status='CANCELLED_POOL_CHANGED',
                          delivery_finished_at=now.isoformat(timespec='seconds'),
                          send_finished_at=now.isoformat(timespec='seconds'))
        return
    row = next((r for r in pool.get('rows', []) if str(r.get('code')) == str(event.get('code'))), None)
    if not row:
        lane.update_event(state_file, event_id, delivery_status='CANCELLED_SYMBOL_REMOVED',
                          delivery_finished_at=now.isoformat(timespec='seconds'),
                          send_finished_at=now.isoformat(timespec='seconds'))
        return
    band = next((b for b in row.get('positions', []) if _same_band(b, event.get('band') or {})), None)
    if not band:
        lane.update_event(state_file, event_id, delivery_status='CANCELLED_BAND_CHANGED',
                          delivery_finished_at=now.isoformat(timespec='seconds'),
                          send_finished_at=now.isoformat(timespec='seconds'))
        return

    # ---- reverify_quote_at 时间戳 ----
    reverify_started_at = datetime.now().isoformat(timespec='seconds')
    quote = (quote_fn([event['code']]) or {}).get(event['code']) or {}
    reverify_quote_at = datetime.now().isoformat(timespec='seconds')
    now = datetime.now()
    from strong_pullback_policy_v1 import fresh
    if not fresh(quote, now):
        lane.update_event(state_file, event_id, delivery_status='WAIT_REVERIFY',
            delivery_retry_after_epoch=now.timestamp() + lane.POLL_INTERVAL_SECONDS,
            reverify_error='QUOTE_STALE_OR_INVALID',
            reverify_started_at=reverify_started_at,
            reverify_quote_at=reverify_quote_at)
        return
    latest_plan = get_plan() or {}
    if plan_digest(latest_plan) != plan_digest(plan_ref) or not active():
        lane.update_event(state_file, event_id, delivery_status='CANCELLED_PLAN_CHANGED',
                          delivery_finished_at=datetime.now().isoformat(timespec='seconds'),
                          send_finished_at=datetime.now().isoformat(timespec='seconds'),
                          reverify_started_at=reverify_started_at,
                          reverify_quote_at=reverify_quote_at)
        return

    from push_layout_v1 import notice
    price = float(quote['price'])
    lower, upper = float(band['lower']), float(band['upper'])
    location = lane.band_location(price, band)
    if row['pool'] != 'LANDMARK' and location != 'STILL_IN_ORIGINAL_BAND':
        lane.update_event(state_file,event_id,delivery_status='WAIT_REVERIFY',
                          delivery_retry_after_epoch=now.timestamp()+lane.POLL_INTERVAL_SECONDS,
                          reverify_error='OUT_OF_BAND_WAIT_REENTRY',
                          reverify_quote=dict(time=quote.get('time'),price=price))
        return
    containing = [b for b in row.get('positions', [])
                  if float(b['lower']) <= price <= float(b['upper'])]
    packet = dict(asof=int(quote['time']), indicator_asof=event['indicator_cutoff'],
        code=row['code'], name=row['name'], board='、'.join(row.get('boards') or []),
        price=price, return_value=price / float(quote['prev']) - 1,
        level_lower=lower, level_upper=upper,
        level_label='共振带' if band.get('source_waves', 0) >= 2 else '黄金位',
        priority_label='P1' if band.get('primary') else 'P2',
        wave_low=row.get('low'), wave_high=row.get('high'), wave_return=row.get('wave_return'),
        indicator_basis='价格触位先行；5/15分钟指标后台计算中', period_pairs={}, indicators_pending=True,
        relative_turnover=dict(status='UNKNOWN', reason='指标后台计算中；不影响价格触位事实'),
        morning_mark='首次观察到触位；发送前已重新核验报价和冻结观察带',
        trigger_text=('价格到达路标；完整K线收盘确认待核验' if row['pool'] == 'LANDMARK' else None))
    packet['distance'] = (price - max(lower, min(upper, price))) / price
    formatted = notice(event.get('kind') or 'STRONG_STOCK', packet)
    direction_labels = {
        'DOWNWARD_ENTRY_OBSERVED': '观察到从上方向下进入',
        'FIRST_QUOTE_IN_BAND_DIRECTION_UNKNOWN': '首次有效报价已在带内，方向未知',
        'REBOUND_FROM_BELOW_OBSERVED': '观察到从下方反弹进入',
        'DIRECTION_UNKNOWN': '方向证据不足',
        'IN_BAND_DIRECTION_NOT_OBSERVED': '报价在带内，未观察到新进入方向',
        'DIRECTION_NOT_CLASSIFIED': '方向未分类'}
    location_labels = {
        'STILL_IN_ORIGINAL_BAND': '仍在原观察带内',
        'BELOW_ORIGINAL_BAND': '已在原观察带下方',
        'ABOVE_ORIGINAL_BAND': '已回到原观察带上方'}
    evidence = (f"\n\n触位方向：{direction_labels.get(event.get('direction_class'), '方向未知')}。"
        f"\n首次观察到触位：{event.get('first_observed_touch_at')}；报价时间：{event.get('first_observed_quote_at')}。"
        "（程序首次观察时间不等于市场实际首次触位时间。）"
        f"\n发送前复核：现价 {price:.2f}；{location_labels.get(location, location)}；"
        f"当前落入观察带数量 {len(containing)}。"
        "\n指标状态：后台计算中；价格触位不代表止跌、支撑成立或允许买入。")
    body = formatted['body'] + evidence
    title = f"强势池 · {row['code']} {formatted['title']}"
    key = digest([int(event['day']), row['code'], row['pool'], band['lower'], band['upper']])
    alerts_file = root() / 'alerts.json'
    send_started_at = datetime.now().isoformat(timespec='seconds')
    attempt_id = secrets.token_hex(8)
    my_pid = os.getpid()
    record = dict(code=row['code'], cutoff=event['indicator_cutoff'], pool_id=pool['rows_sha256'],
        event=dict(event), packet=packet, title=title, body=body, delivery='PRE_SEND_RESERVED',
        wechat=False, attempt_id=attempt_id, owner=dict(pid=my_pid),
        first_observed_touch_at=event.get('first_observed_touch_at'),
        first_observed_quote_at=event.get('first_observed_quote_at'),
        send_queued_at=event.get('send_queued_at'),
        send_started_at=send_started_at,
        reverify_quote=dict(time=quote.get('time'), price=price),
        reverify_quote_at=reverify_quote_at,
        reverify_band_status=location, indicator_status='PENDING', attempts=[])

    with _alerts_lock():
        alerts = json.loads(alerts_file.read_bytes()) if alerts_file.exists() else {}
        # (1) already SENT for this stock/day -> never auto-resend (SENT/UNKNOWN no resend).
        if any(str(v.get('cutoff', ''))[:8] == str(event['day']) and v.get('code') == row['code']
               and v.get('delivery') == 'SENT' for v in alerts.values()):
            # 事件已是终态（SENT/UNKNOWN）则保持不动，不用 DUPLICATE_SUPPRESSED 覆盖终态回执。
            if event.get('delivery_status') not in ('SENT', 'UNKNOWN'):
                lane.update_event(state_file, event_id, delivery_status='DUPLICATE_SUPPRESSED',
                                  delivery_finished_at=datetime.now().isoformat(timespec='seconds'),
                                  send_finished_at=datetime.now().isoformat(timespec='seconds'),
                                  reverify_started_at=reverify_started_at,
                                  reverify_quote_at=reverify_quote_at)
            return
        # (2) existing active reservation for THIS event (same key)?
        existing = alerts.get(key)
        if existing is not None and existing.get('delivery') in ('PRE_SEND_RESERVED', 'SENDING'):
            ex_pid = (existing.get('owner') or {}).get('pid')
            if ex_pid == my_pid:
                # our own (re)attempt / restart -> reclaim, carry forward evidence
                record['attempts'] = existing.get('attempts', []) + [_attempt_summary(existing)]
            elif existing.get('delivery') == 'SENDING':
                # another live owner is mid-send -> outcome uncertain; do NOT send.
                lane.update_event(state_file, event_id, delivery_status='UNKNOWN',
                                  delivery_finished_at=datetime.now().isoformat(timespec='seconds'),
                                  send_finished_at=datetime.now().isoformat(timespec='seconds'),
                                  reverify_started_at=reverify_started_at,
                                  reverify_quote_at=reverify_quote_at)
                return
            else:  # PRE_SEND_RESERVED by a different LIVE owner -> block this 2nd caller
                if cpl.pid_alive(ex_pid):
                    lane.update_event(state_file, event_id, delivery_status='DUPLICATE_SUPPRESSED',
                                      delivery_finished_at=datetime.now().isoformat(timespec='seconds'),
                                      send_finished_at=datetime.now().isoformat(timespec='seconds'),
                                      reverify_started_at=reverify_started_at,
                                      reverify_quote_at=reverify_quote_at)
                    return
                # dead owner -> take over (reclaim, carry forward evidence)
                record['attempts'] = existing.get('attempts', []) + [_attempt_summary(existing)]
        # (3) broad suppression: another key for same stock/day with a LIVE reserved/sending.
        for k2, v2 in alerts.items():
            if k2 == key:
                continue
            if (str(v2.get('cutoff', ''))[:8] == str(event['day']) and v2.get('code') == row['code']
                    and v2.get('delivery') in ('PRE_SEND_RESERVED', 'SENDING')
                    and cpl.pid_alive((v2.get('owner') or {}).get('pid'))):
                lane.update_event(state_file, event_id, delivery_status='DUPLICATE_SUPPRESSED',
                                  delivery_finished_at=datetime.now().isoformat(timespec='seconds'),
                                  send_finished_at=datetime.now().isoformat(timespec='seconds'),
                                  reverify_started_at=reverify_started_at,
                                  reverify_quote_at=reverify_quote_at)
                return
        # (4) reserve (carry forward any prior evidence on this key).
        prior_rec = alerts.get(key)
        if prior_rec is not None:
            record['attempts'] = (prior_rec.get('attempts', []) or []) + [_attempt_summary(prior_rec)]
        alerts[key] = record
        try:
            save(alerts_file, alerts)
        except Exception as exc:
            # Reservation write failed -> must NOT continue to send.
            lane.update_event(state_file, event_id, delivery_status='WAIT_REVERIFY',
                              delivery_retry_after_epoch=now.timestamp() + lane.POLL_INTERVAL_SECONDS,
                              reverify_error='RESERVATION_WRITE_FAILED:' + type(exc).__name__)
            return

    broadcast(dict(type='push', kind='strong_sector', title=title, body=body,
                   wechat=False, delivery='RECORDED'))
    # Re-read at the actual send boundary; no old configuration snapshot.
    try:cfg = get_cfg() or {}
    except Exception:
        lane.update_event(state_file,event_id,delivery_status='WAIT_REVERIFY',
                          delivery_retry_after_epoch=now.timestamp()+lane.POLL_INTERVAL_SECONDS,
                          reverify_error='CONFIG_UNREADABLE_NO_ATTEMPT')
        _release_reservation(alerts_file, key, attempt_id)
        return
    ok, message, delivery = False, '微信推送已关闭', 'DISABLED'
    if (cfg.get('push') or {}).get('enabled'):
        attempted = False
        try:
            with _alerts_lock():
                latest=json.loads(alerts_file.read_bytes())
                cur = latest.get(key)
                # identity-verified transition: only our own active reservation may go SENDING
                if cur is None or cur.get('attempt_id') != attempt_id:
                    return  # taken over / removed by someone else -> do not send
                cur.update(delivery='SENDING', send_started_at=send_started_at)
                save(alerts_file, latest)
            lane.update_event(state_file,event_id,delivery_status='SENDING',delivery_phase='TRANSPORT',
                              send_started_at=send_started_at)
            attempted = True
            ok, message = send(cfg, title, body)
            delivery = 'SENT' if ok else 'FAILED'
        except Exception as exc:
            if not attempted:raise
            message, delivery = type(exc).__name__, 'UNKNOWN'
    else:
        lane.update_event(state_file,event_id,delivery_status='WAIT_REVERIFY',
                          delivery_retry_after_epoch=now.timestamp()+lane.POLL_INTERVAL_SECONDS,
                          reverify_error='PUSH_DISABLED_NO_ATTEMPT')
        _release_reservation(alerts_file, key, attempt_id)
        return
    finished = datetime.now().isoformat(timespec='seconds')
    # Write the receipt FIRST (under lock); only then update the lane event.
    try:
        with _alerts_lock():
            latest = json.loads(alerts_file.read_bytes()) if alerts_file.exists() else {}
            cur = latest.get(key)
            if cur is None or cur.get('attempt_id') != attempt_id:
                # someone else owns it now -> record risk, do not claim success
                lane.update_event(state_file, event_id, delivery_status=delivery if delivery in ('SENT', 'FAILED') else 'UNKNOWN',
                                  send_finished_at=finished,
                                  reverify_quote=dict(time=quote.get('time'), price=price),
                                  reverify_band_status=location,
                                  send_started_at=send_started_at,
                                  reverify_started_at=reverify_started_at,
                                  reverify_quote_at=reverify_quote_at)
                return
            receipt = dict(attempt_id=attempt_id, owner=dict(pid=my_pid), delivery=delivery,
                          wechat=bool(ok), wechat_msg=str(message), sent_at=finished,
                          send_started_at=send_started_at, send_finished_at=finished)
            cur.update(wechat=bool(ok), delivery=delivery, wechat_msg=str(message),
                       sent_at=finished, send_finished_at=finished)
            cur.setdefault('attempts', []).append(receipt)
            save(alerts_file, latest)
    except Exception as exc:
        # Receipt write failed -> do NOT confirm success; record risk explicitly.
        lane.update_event(state_file, event_id, delivery_status='UNKNOWN',
                          delivery_finished_at=finished, sent_at=None,
                          reverify_quote=dict(time=quote.get('time'), price=price),
                          reverify_band_status=location,
                          send_started_at=send_started_at,
                          reverify_started_at=reverify_started_at,
                          reverify_quote_at=reverify_quote_at,
                          reverify_error='RECEIPT_WRITE_FAILED:' + type(exc).__name__)
        return
    lane.update_event(state_file, event_id, delivery_status=delivery, send_finished_at=finished,
        sent_at=finished if delivery == 'SENT' else None,
        reverify_quote=dict(time=quote.get('time'), price=price),
        reverify_band_status=location,
        send_started_at=send_started_at,
        reverify_started_at=reverify_started_at,
        reverify_quote_at=reverify_quote_at)


def compute_touch_indicators(event_id, state_file, plan_ref):
    """Run slow indicator context in its own worker; it cannot delay quote polling.

    任务书阶段1: indicator_started_at / indicator_finished_at 事件级证据链.
    """
    import strong_touch_lane_v1 as lane
    event = lane.get_event(state_file, event_id)
    if not event:
        return
    if not lane.is_state_owner(state_file):
        # Non-owner must not compute or write indicator context.
        return
    indicator_started_at = datetime.now().isoformat(timespec='seconds')
    from sector_watch_v1 import names
    pool = frozen(names(plan_ref or {}), int(event['day']))
    if not pool or pool.get('rows_sha256') != event.get('pool_id'):
        lane.update_event(state_file, event_id, indicator_status='CANCELLED_POOL_CHANGED',
                          indicator_finished_at=datetime.now().isoformat(timespec='seconds'),
                          indicator_started_at=indicator_started_at)
        return
    row = next((r for r in pool.get('rows', []) if str(r.get('code')) == str(event.get('code'))), None)
    if not row:
        lane.update_event(state_file, event_id, indicator_status='CANCELLED_SYMBOL_REMOVED',
                          indicator_finished_at=datetime.now().isoformat(timespec='seconds'),
                          indicator_started_at=indicator_started_at)
        return
    try:
        result = invoke(dict(mode='INDICATORS', row=row, cutoff=event['indicator_cutoff'],
                             quote=event.get('initial_quote') or {}))
    except Exception as exc:
        # 计算失败必须留痕（不吞掉本地事实，也绝不冒充成功）；补充阶段如实报告原因。
        finished = datetime.now().isoformat(timespec='seconds')
        lane.update_event(state_file, event_id, indicator_status='FAILED',
                          indicator_finished_at=finished, indicator_started_at=indicator_started_at,
                          indicator_supplement_status='PENDING',
                          indicator_result=dict(status='FAILED', reason=type(exc).__name__ + ':' + str(exc)[:160]))
        return
    finished = datetime.now().isoformat(timespec='seconds')
    summary = {k: result.get(k) for k in
               ('status', 'reason', 'asof', 'last_time', 'periods', 'relative_turnover', 'volume_ratio')
               if k in result}
    lane.update_event(state_file, event_id, indicator_status=result.get('status', 'UNKNOWN'),
                      indicator_finished_at=finished, indicator_started_at=indicator_started_at,
                      indicator_supplement_status='PENDING',
                      indicator_result=summary)


# ---- 指标补充（闭环2） -------------------------------------------------------
SUPPLEMENT_TERMINAL = frozenset({
    'SENT', 'UNKNOWN', 'SUPPRESSED_DUPLICATE',
    'SKIPPED_PLAN_CHANGED', 'SKIPPED_POOL_CHANGED', 'SKIPPED_BAND_CHANGED',
    'SKIPPED_SYMBOL_REMOVED', 'SKIPPED_QUOTE_STALE', 'SKIPPED_FIRST_NOT_SENT',
    'SKIPPED_PUSH_DISABLED'})


def _supplement_message(event, row, quote):
    """补充消息：只陈述事实——首次触价时间 / 指标截止 / 当前报价时间 / 指标摘要或失败原因。

    绝无操作建议；未核实的口径明确写未知，不把成交额说成成交量。
    """
    code = str(event.get('code'))
    lines = ['【触价补充 · ' + code + '】',
             '首次触价时间: ' + str(event.get('first_observed_touch_at') or '未知'),
             '指标截止: ' + str(event.get('indicator_cutoff') or '未知'),
             '当前报价时间: ' + str(quote.get('time') or '未知')]
    indicator = event.get('indicator_result') or {}
    status = str(indicator.get('status') or event.get('indicator_status') or 'UNKNOWN')
    if status == 'AVAILABLE':
        from push_compact_v1 import indicator_lines
        periods = indicator.get('periods') or {}
        if periods:
            lines.extend(indicator_lines(dict(period_pairs=periods), include_times=True))
        else:
            lines.append('5/15分钟低点指标：未提供')
        rel = indicator.get('relative_turnover')
        if isinstance(rel, dict):
            ratio = rel.get('ratio')
            if isinstance(ratio, (int, float)) and math.isfinite(ratio):
                lines.append('同刻量比: %.2f' % float(ratio))
            else:
                lines.append('同刻量比: 未知（%s）' % str(rel.get('reason') or '未核实'))
        vr = indicator.get('volume_ratio')
        if isinstance(vr, dict):
            ratio = vr.get('ratio')
            if isinstance(ratio, (int, float)) and math.isfinite(ratio):
                lines.append('量能比率: %.2f' % float(ratio))
            else:
                lines.append('量能比率: 未知（%s）' % str(vr.get('reason') or '未核实'))
    else:
        lines.append('指标状态: ' + status +
                     (('（' + str(indicator.get('reason') or '') + '）') if indicator.get('reason') else ''))
    lines.append('仅价格、指标与规则事实，请人工判断；未生成买卖建议。')
    return '触价补充 · ' + code, '\n'.join(lines)


def deliver_indicator_supplement(event_id, state_file, plan_ref, *, get_plan, get_cfg,
                                 quote_fn, broadcast, send, active):
    """指标完成后，对已 SENT 的首条提醒做一次同事件身份的补充（恰好一次，持久化）。

    终态（重启不重复）：SENT / UNKNOWN / SUPPRESSED_DUPLICATE / SKIPPED_*。
    - 首条仍在排队、复核或发送中 → 等待；首条终态未 SENT → SKIPPED_FIRST_NOT_SENT；
    - 计划/池/带变化或报价失效 → SKIPPED_*，不补发过时提示；
    - PRE_SEND_RESERVED 被另一存活 pid 持有 → SUPPRESSED_DUPLICATE 不重复启动；
      死亡 pid 的预约 → 收回续发；SENDING 被他主持有 → UNKNOWN 不重发；
    - 预约写盘失败 → WAIT_RETRY（可续），不发送；
    - 回执先落盘，写失败 → UNKNOWN；传输异常 → UNKNOWN 终态不重发。
    """
    import strong_touch_lane_v1 as lane
    event = lane.get_event(state_file, event_id)
    if not event or not lane.is_state_owner(state_file) or not active():
        return
    if event.get('indicator_status') in (None, 'PENDING', 'PROCESSING'):
        return  # 指标尚未完成
    current = str(event.get('indicator_supplement_status') or '')
    if current in SUPPLEMENT_TERMINAL:
        return  # 终态：重启不重复
    primary = event.get('delivery_status')
    if primary in ('PENDING', 'WAIT_REVERIFY', 'PROCESSING', 'SENDING'):
        return
    now = datetime.now()
    alerts_file = root() / 'alerts.json'
    with _alerts_lock():
        saved = json.loads(alerts_file.read_bytes()) if alerts_file.exists() else {}
        first = saved.get(event_id) or {}
        previous = first.get('supplement') or {}
        prior_status = previous.get('status')
        if prior_status in ('SENT', 'UNKNOWN', 'SENDING') or current == 'SENDING':
            # A restart can occur between the durable receipt and event-state update.
            lane.update_event(state_file, event_id,
                              indicator_supplement_status='SENT' if prior_status == 'SENT' else 'UNKNOWN',
                              supplement_finished_at=now.isoformat(timespec='seconds'))
            return
        if primary == 'UNKNOWN' and first.get('delivery') == 'SENT':
            lane.update_event(state_file, event_id, delivery_status='SENT',
                              sent_at=first.get('sent_at'))
            primary = 'SENT'
    if primary != 'SENT':
        lane.update_event(state_file, event_id, indicator_supplement_status='SKIPPED_FIRST_NOT_SENT',
                          supplement_finished_at=now.isoformat(timespec='seconds'))
        return
    from notifier import market_open
    if str(event.get('day')) != now.strftime('%Y%m%d') or not market_open(now):
        return
    try:
        cfg = get_cfg() or {}
    except Exception:
        lane.update_event(state_file, event_id, indicator_supplement_status='WAIT_RETRY',
                          supplement_retry_after_epoch=now.timestamp() + lane.POLL_INTERVAL_SECONDS)
        return
    if not (cfg.get('push') or {}).get('enabled'):
        lane.update_event(state_file, event_id, indicator_supplement_status='SKIPPED_PUSH_DISABLED',
                          supplement_finished_at=now.isoformat(timespec='seconds'))
        return
    current_plan = get_plan() or {}
    from golden_rules import digest as plan_digest
    if plan_digest(current_plan) != plan_digest(plan_ref):
        lane.update_event(state_file, event_id, indicator_supplement_status='SKIPPED_PLAN_CHANGED',
                          supplement_finished_at=now.isoformat(timespec='seconds'))
        return
    from sector_watch_v1 import names
    pool = frozen(names(current_plan), int(event['day']))
    if not pool or pool.get('rows_sha256') != event.get('pool_id'):
        lane.update_event(state_file, event_id, indicator_supplement_status='SKIPPED_POOL_CHANGED',
                          supplement_finished_at=now.isoformat(timespec='seconds'))
        return
    row = next((r for r in pool.get('rows', []) if str(r.get('code')) == str(event.get('code'))), None)
    if not row:
        lane.update_event(state_file, event_id, indicator_supplement_status='SKIPPED_SYMBOL_REMOVED',
                          supplement_finished_at=now.isoformat(timespec='seconds'))
        return
    band = next((b for b in row.get('positions', []) if _same_band(b, event.get('band') or {})), None)
    if not band:
        lane.update_event(state_file, event_id, indicator_supplement_status='SKIPPED_BAND_CHANGED',
                          supplement_finished_at=now.isoformat(timespec='seconds'))
        return
    reverify_started_at = datetime.now().isoformat(timespec='seconds')
    quote = (quote_fn([event['code']]) or {}).get(event['code']) or {}
    reverify_quote_at = datetime.now().isoformat(timespec='seconds')
    from strong_pullback_policy_v1 import fresh
    if not fresh(quote, datetime.now()) or str(quote.get('time', ''))[:8] != str(event['day']):
        lane.update_event(state_file, event_id, indicator_supplement_status='SKIPPED_QUOTE_STALE',
                          supplement_finished_at=datetime.now().isoformat(timespec='seconds'),
                          supplement_reverify_quote_at=reverify_quote_at)
        return

    key = digest([int(event['day']), row['code'], row['pool'], band['lower'], band['upper']])
    attempt_id = secrets.token_hex(8)
    my_pid = os.getpid()
    title, body = _supplement_message(event, row, quote)
    reservation = dict(status='PRE_SEND_RESERVED', attempt_id=attempt_id, owner=dict(pid=my_pid),
                       reserved_at=reverify_started_at, title=title, body=body,
                       quote=dict(time=quote.get('time'), price=quote.get('price')))

    def _finish(status, **extra):
        lane.update_event(state_file, event_id, indicator_supplement_status=status,
                          supplement_finished_at=datetime.now().isoformat(timespec='seconds'),
                          supplement_attempt_id=attempt_id, **extra)

    with _alerts_lock():
        alerts = json.loads(alerts_file.read_bytes()) if alerts_file.exists() else {}
        rec = alerts.get(key)
        if rec is None or rec.get('delivery') != 'SENT':
            # 首条回执不在或已非 SENT（被外部改动）——绝不盲目补充。
            _finish('SKIPPED_FIRST_NOT_SENT')
            return
        sup = rec.get('supplement') or {}
        if sup.get('status') in ('SENT', 'UNKNOWN', 'SENDING'):
            _finish('SENT' if sup['status'] == 'SENT' else 'UNKNOWN')
            return
        if sup.get('status') == 'PRE_SEND_RESERVED':
            ex_pid = int((sup.get('owner') or {}).get('pid') or 0)
            if ex_pid and ex_pid != my_pid and cpl.pid_alive(ex_pid):
                _finish('SUPPRESSED_DUPLICATE')
                return
            # 死亡 pid / 超时 → 收回续发（携带先前尝试证据）
            reservation['prior_attempt'] = dict(attempt_id=sup.get('attempt_id'),
                                                owner=sup.get('owner'), status=sup.get('status'))
        rec['supplement'] = reservation
        try:
            save(alerts_file, alerts)
        except Exception as exc:
            lane.update_event(state_file, event_id, indicator_supplement_status='WAIT_RETRY',
                              supplement_retry_after_epoch=now.timestamp() + lane.POLL_INTERVAL_SECONDS,
                              supplement_error='RESERVATION_WRITE_FAILED:' + type(exc).__name__)
            return

    broadcast(dict(type='push', kind='strong_sector', title=title, body=body,
                   wechat=False, delivery='RECORDED_SUPPLEMENT'))
    # 发送边界身份核对：预约必须仍是本次 attempt。
    with _alerts_lock():
        latest = json.loads(alerts_file.read_bytes()) if alerts_file.exists() else {}
        cur = (latest.get(key) or {}).get('supplement') or {}
        if cur.get('attempt_id') != attempt_id or cur.get('status') != 'PRE_SEND_RESERVED':
            _finish('SUPPRESSED_DUPLICATE')
            return
        cur['status'] = 'SENDING'
        cur['send_started_at'] = datetime.now().isoformat(timespec='seconds')
        save(alerts_file, latest)
    lane.update_event(state_file, event_id, indicator_supplement_status='SENDING',
                      supplement_started_at=datetime.now().isoformat(timespec='seconds'))
    ok, message = False, '发送异常'
    delivery = 'UNKNOWN'
    try:
        ok, message = send(cfg, title, body)
        delivery = 'SENT' if ok else 'UNKNOWN'
    except Exception as exc:
        message, delivery = type(exc).__name__, 'UNKNOWN'
    finished = datetime.now().isoformat(timespec='seconds')
    # 回执先落盘（失败 → UNKNOWN，明确风险留痕）。
    try:
        with _alerts_lock():
            latest = json.loads(alerts_file.read_bytes()) if alerts_file.exists() else {}
            rec = latest.get(key) or {}
            sup = rec.get('supplement') or {}
            if sup.get('attempt_id') != attempt_id:
                lane.update_event(state_file, event_id, indicator_supplement_status='UNKNOWN',
                                  supplement_finished_at=finished,
                                  supplement_error='RECEIPT_IDENTITY_LOST')
                return
            sup.update(status=delivery, wechat=bool(ok), wechat_msg=str(message),
                       send_finished_at=finished)
            rec.setdefault('attempts', []).append(dict(
                attempt_id=attempt_id, kind='INDICATOR_SUPPLEMENT', owner=dict(pid=my_pid),
                delivery=delivery, wechat=bool(ok), wechat_msg=str(message),
                send_finished_at=finished))
            save(alerts_file, latest)
    except Exception as exc:
        lane.update_event(state_file, event_id, indicator_supplement_status='UNKNOWN',
                          supplement_finished_at=finished,
                          supplement_error='RECEIPT_WRITE_FAILED:' + type(exc).__name__)
        return
    _finish(delivery, supplement_sent_at=finished if delivery == 'SENT' else None,
            supplement_reverify_quote_at=reverify_quote_at,
            supplement_reverify_started_at=reverify_started_at)
