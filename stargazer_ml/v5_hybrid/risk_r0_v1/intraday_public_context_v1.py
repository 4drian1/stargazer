"""One free, real END_5M board snapshot per cutoff; TDX remains history only.

No stockdb, synthetic minutes, narrowed membership, paid API or model fitting.
The previous accepted session supplies references, never today's returns.

Step2 dev copy（闭环1：LIVE_REQUIRED_PREFIX_MISSING 逐股定位与恢复）：
- 同胞模块锚定到 risk_r0_v1 权威目录（本副本只改收集/诊断/恢复逻辑，
  数学与契约身份 intraday_native_context_v1.py 仍按权威文件计算，math_key 与生产一致）。
- acquire 返回逐票诊断（请求截止/实际数据截止/尝试次数/来源URL/失败原因），
  无论成败都落盘 prefix_diagnostics.json——失败轮次的逐票证据不再丢失。
- 新增 _collect_required 恢复循环：只补实际缺失票（已归档原始响应零重复请求），
  单票失败不拖死其他票；总时间预算与 LIVE_DAILY_ATTEMPTS 上限不放宽；
  预算耗尽仍缺 → 携逐票诊断抛 LIVE_REQUIRED_PREFIX_MISSING。
- 不缩短指标历史、不借昨日结果：历史仍只来自经校验的上一完整会话。
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import gzip
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request

import numpy as np
import pandas as pd

# 同胞模块（tdx_postclose_runtime_v1 / intraday_native_context_v1 / …）仍以
# risk_r0_v1 权威目录为准；本 dev 副本不复制数学层，避免双源漂移。
# 只有在常规导入失败（即本副本脱离权威目录运行）时才追加 sys.path，
# 避免在生产环境中重排 sys.path 造成模块遮蔽。
_SIBLING_HOME = Path(os.environ.get(
    'RISK_R0_V1_HOME', r'D:\stargazer\stargazer_ml\v5_hybrid\risk_r0_v1'))
try:
    import tdx_postclose_runtime_v1 as rt
    from intraday_native_context_v1 import NativeIntradayCache, prefix_rows
    from joint_shadow_context_v1 import RiskSnapshot
except ImportError:  # dev 副本脱离权威目录时
    if str(_SIBLING_HOME) not in sys.path:
        sys.path.insert(0, str(_SIBLING_HOME))
    import tdx_postclose_runtime_v1 as rt
    from intraday_native_context_v1 import NativeIntradayCache, prefix_rows
    from joint_shadow_context_v1 import RiskSnapshot

CONTRACT='PUBLIC_NATIVE_5M_SHARED_BOARD_V1'
MAX_BYTES=32768
WORKERS=8
COLLECTION_BUDGET_S=210.0   # 与原 LIVE_COLLECTION_DEADLINE 一致，不放宽
RETRY_BACKOFF_S=2.0


def sha(raw):return hashlib.sha256(raw).hexdigest()


def symbol(code):
    if code in ('000001.SH','399006.SZ'):
        return 'sh000001' if code=='000001.SH' else 'sz399006'
    if len(code)!=6 or not code.isdigit():raise ValueError('LIVE_CODE')
    return ('bj' if code.startswith(('4','8','920')) else 'sh' if code[0] in '569' else 'sz')+code


def fetch(code):
    from public_stock_minutes_v1 import NoRedirect
    url=('https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData?symbol='
         +symbol(code)+'&scale=5&ma=no&datalen=100')
    # Direct public quotes; no inherited development proxy or redirect to a
    # differently contracted service. Each call is one request, no hidden retry.
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    with opener.open(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=5) as response:
        raw=response.read(MAX_BYTES+1)
    if len(raw)>MAX_BYTES:raise ValueError('LIVE_RESPONSE_LIMIT')
    return raw,url


def parse(raw,code,P,cutoff,reference):
    values=json.loads(raw)
    if not isinstance(values,list) or len(values)>100:raise ValueError('LIVE_5M_SCHEMA')
    rows=[];previous=[]
    for item in values:
        ts=int(datetime.strptime(item['day'],'%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M%S'))
        if ts==P*1000000+150000:previous.append(float(item['close']))
        if ts//1000000!=cutoff//1000000 or ts>cutoff:continue
        row=dict(date=ts,code=code,source_clock='END_5M',source_frequency_minutes=5,
            **{k:float(item[k]) for k in ('open','high','low','close','volume','amount')})
        v,a=row['volume'],row['amount']
        if not np.isfinite([v,a]).all() or min(v,a)<0 or ((v==0)!=(a==0)):
            raise ValueError('LIVE_5M_ACTIVITY')
        if '.' not in code and v and not row['low']*.99<=a/v<=row['high']*1.01:
            raise ValueError('LIVE_5M_VOLUME_UNIT')
        rows.append(row)
    if len(previous)!=1 or abs(previous[0]-reference)>.011:
        raise ValueError('LIVE_5M_PREVIOUS_CLOSE_BASIS')
    prefix_rows(rows,code,cutoff//1000000,cutoff)
    return rows


def previous_source(runtime,day,codes):
    markers=sorted((Path(runtime)/'tdx_prepared').glob('*/*/complete.json'),key=lambda p:(p.parent.parent.name,p.stat().st_mtime_ns),reverse=True)
    for marker in markers:
        folder=marker.parent;done=rt.read_json(marker)
        try:
            for name in ('snapshot.json','request.json','stocks.parquet'):
                if rt.native.sha(folder/name)!=done['files'][name]:raise ValueError('LIVE_REFERENCE_SHA')
            meta=rt.read_json(folder/'snapshot.json');spec=rt.read_json(folder/'request.json')
            legal=spec['legal'];P=spec['P']
            if day not in legal or legal.index(day)==0 or legal[legal.index(day)-1]!=P:continue
            if meta['cutoff']!=P*1000000+150000:continue
            frame=pd.read_parquet(folder/'stocks.parquet')
            last=frame[frame.cutoff_index==(len(meta['days'])-1)*16+15]
            refs={str(r.code):dict(day=P,complete=True,close=float(r.stock_close)) for r in last.itertuples()
                  if np.isfinite(r.stock_close) and r.stock_close>0}
            if not set(codes)<=set(refs):continue
            return folder,spec,meta,refs
        except (OSError,ValueError,KeyError):continue
    raise ValueError('LIVE_PREVIOUS_PREPARED_SESSION_MISSING')


def acquire(folder,refs,P,cutoff,runtime,transport=fetch):
    """Bounded fanout, actual attempts reserved before HTTP, resumable raw proof.

    16 scheduled 15m cutoffs, at most 20 attempts/code/day including recovery.
    No network call is made for a verified archived response. Disk reservations
    use the one authorized runtime account, not an unmetered new asset zone.

    Step2: returns a 4th element — per-code diagnostics (requested cutoff,
    actual last data ts, attempts consumed, source url, failure reason) — and
    persists them to ``folder/prefix_diagnostics.json`` even on partial failure,
    so a failed round no longer loses per-stock evidence.
    """
    folder.mkdir(parents=True,exist_ok=True)
    ledger=folder.parent.parent/'attempts.json';lock=threading.Lock()
    counts=rt.read_json(ledger) if ledger.exists() else {}
    remaining=[rt.unified_cache_headroom(runtime)]
    failures={};receipts={};diagnostics={};started=time.monotonic()
    def one(code):
        path=folder/(symbol(code)+'.json.gz');meta=path.with_suffix('.meta.json')
        last_ts=None;url=None
        if path.exists() and meta.exists():
            proof=rt.read_json(meta);body=path.read_bytes()
            if sha(body)!=proof['sha256']:raise ValueError('LIVE_RAW_SHA')
            raw=gzip.decompress(body)
            try:
                rows=parse(raw,code,P,cutoff,refs[code]['close'])
                last_ts=max(int(r['date']) for r in rows)
                return proof,last_ts,url
            except (ValueError,KeyError,TypeError):
                # A source which was late can recover on the next worker run.
                # Keep the exact earlier response; never overwrite bad evidence.
                with lock:
                    suffix='.rejected_'+str(time.time_ns())
                    path.rename(path.with_name(path.name+suffix));meta.rename(meta.with_name(meta.name+suffix))
        if not path.exists():
            if path.exists() or meta.exists():raise ValueError('LIVE_ARCHIVE_INCOMPLETE')
            with lock:
                if time.monotonic()-started>COLLECTION_BUDGET_S:raise ValueError('LIVE_COLLECTION_DEADLINE')
                if counts.get(code,0)>=20:raise ValueError('LIVE_DAILY_ATTEMPTS')
                if remaining[0]<MAX_BYTES+8192:raise ValueError('RUNTIME_DISK_BUDGET')
                remaining[0]-=MAX_BYTES+8192
                counts[code]=counts.get(code,0)+1;rt.atomic_json(ledger,counts)
            raw,url=transport(code)
            body=gzip.compress(raw,mtime=0)
            proof=dict(contract=CONTRACT,code=code,cutoff=cutoff,clock='END_5M',frequency_minutes=5,
                price_basis='RAW',available_at=rt.native.stamp(),url=url,sha256=sha(body),raw_sha256=sha(raw),
                volume_basis='PRICE_ONLY_NOT_A_VOLUME_CONTRACT' if '.' in code else 'SHARES_AMOUNT_PER_SHARE_CHECKED',
                reference=str(path),availability_basis='LOCAL_RESPONSE_OBSERVED_NOT_UPSTREAM_FIRST_ARRIVAL')
            with path.open('xb') as stream:stream.write(body)
            rt.atomic_json(meta,proof)
        rows=parse(raw,code,P,cutoff,refs[code]['close'])
        return proof,max(int(r['date']) for r in rows),url
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures={pool.submit(one,c):c for c in refs}
        for future in as_completed(futures):
            code=futures[future]
            try:
                proof,last_ts,_url=future.result()
                receipts[code]=proof
                diagnostics[code]=dict(status='OK',code=code,requested_cutoff=cutoff,
                    data_last_ts=last_ts,attempts=int(counts.get(code,0)),
                    source_url=proof.get('url'),sha256=proof.get('sha256'))
            except (OSError,ValueError,KeyError,TypeError) as error:
                failures[code]=type(error).__name__+':'+str(error)[:120]
                diagnostics[code]=dict(status='FAILED',code=code,requested_cutoff=cutoff,
                    attempts=int(counts.get(code,0)),failure=failures[code])
    audit=dict(elapsed_s=round(time.monotonic()-started,3),attempts=sum(counts.values()),
        declared=len(refs),available=len(receipts),failed=len(failures),workers=WORKERS)
    try:rt.atomic_json(folder/'prefix_diagnostics.json',
        dict(generated_at=rt.native.stamp(),cutoff=cutoff,audit=audit,diagnostics=diagnostics))
    except OSError:pass
    return receipts,failures,audit,diagnostics


def _collect_required(folder_raw,all_refs,P,cutoff,runtime,transport=fetch,*,
                      budget_s=COLLECTION_BUDGET_S,backoff_s=RETRY_BACKOFF_S):
    """Recovery loop: only actually-missing codes are re-fetched.

    Already-archived raw responses are reused with zero repeat HTTP; a failing
    stock never blocks the others' receipts.  Total wall budget equals the
    original single-round deadline (no loosening) and the per-code daily attempt
    cap still applies inside acquire.  Exhaustion of budget with codes still
    missing raises LIVE_REQUIRED_PREFIX_MISSING carrying per-code diagnosis.
    """
    deadline=time.monotonic()+float(budget_s)
    receipts={};failures={};diagnostics={};rounds=[];total_attempts=0
    missing=list(all_refs)
    while missing:
        subset={c:all_refs[c] for c in missing}
        r,f,audit,diag=acquire(folder_raw,subset,P,cutoff,runtime,transport=transport)
        receipts.update(r);diagnostics.update(diag);total_attempts+=int(audit.get('attempts',0))
        rounds.append(dict(fetched=sorted(r),failed=sorted(f),audit=audit))
        failures=f
        missing=[c for c in missing if c not in receipts]
        if not missing:break
        if time.monotonic()>=deadline:
            raise ValueError('LIVE_REQUIRED_PREFIX_MISSING:'+','.join(missing))
        time.sleep(float(backoff_s))
    return receipts,failures,diagnostics,rounds,total_attempts


def _persist_produce_diagnostics(folder,codes,spec,rounds):
    """Persist per-stock diagnostics at produce level (success or final failure).

    Adds, per planned stock, the history reference actually bound for indicators
    (path of the validated previous-session lc5 source) so the record shows what
    history coverage the recovery relies on — never a shortened history.
    """
    per_stock={}
    for code in codes:
        info=spec.get('files',{}).get(code) or {}
        per_stock[code]=dict(history_reference=info.get('path'),
            history_days_bound=True if info.get('path') else None)
    try:
        rt.atomic_json(folder/'prefix_diagnostics.json',dict(
            generated_at=rt.native.stamp(),rounds=rounds,per_stock=per_stock,
            note='history is bound to the validated previous session; never shortened to pretend success'))
    except OSError:
        pass


def produce(runtime,codes,cutoff,output,progress):
    from minute_history_v1 import cache_lock
    from tdx_native_reader_v1 import read_lc5_file
    from tdx_market_schedule_v1 import enable
    enable();runtime=Path(runtime);day=cutoff//1000000
    previous,spec,old,refs=previous_source(runtime,day,codes)
    # Membership, source file identities and common calendar remain those of
    # the accepted P snapshot. It is NOT reduced to the current plan list.
    days=old['days']+[day];P=days[-2]
    folder=runtime/'intraday_native'/str(day)/(str(cutoff)+'_'+rt.native.sha(previous/'snapshot.json')[:16])
    folder.mkdir(parents=True,exist_ok=True)
    # One daily collector owns both the attempt ledger and cutoff manifests.
    with cache_lock(folder.parent):
        marker=folder/'ready.json'
        if marker.exists():
            ready=rt.read_json(marker)
            for name,h in ready['files'].items():
                if rt.native.sha(folder/name)!=h:raise ValueError('LIVE_COMMON_SHA:'+name)
            receipts=ready['receipts'];failures=ready['failures'];audit=dict(ready['audit'],reused=True)
        else:
            progress(stage='LIVE_SHARED_MEMBER_5M',done=0,total=len(refs))
            # Indices also use real public 5m bars, never yesterday's market state.
            index_refs={}
            for code in ('000001.SH','399006.SZ'):
                raw=pd.read_parquet(previous/(code+'.parquet'))
                if rt.native.sha(previous/(code+'.parquet'))!=rt.read_json(previous/'complete.json')['files'][code+'.parquet']:
                    raise ValueError('LIVE_INDEX_HISTORY_SHA')
                index_refs[code]=dict(day=P,complete=True,close=float(raw[raw.date==P*1000000+150000].iloc[0]['close']))
            all_refs=dict(refs);all_refs.update(index_refs)
            try:
                receipts,failures,diagnostics,rounds,total_attempts=_collect_required(
                    folder/'raw',all_refs,P,cutoff,runtime)
            except ValueError as error:
                if not str(error).startswith('LIVE_REQUIRED_PREFIX_MISSING'):raise
                _persist_produce_diagnostics(folder,codes,spec,rounds)
                raise
            audit=dict(rounds[-1]['audit']) if rounds else dict(attempts=total_attempts)
            audit.update(recovery_rounds=len(rounds),attempts_total=total_attempts)
            if not set(codes+['000001.SH','399006.SZ'])<=set(receipts):
                raise ValueError('LIVE_REQUIRED_PREFIX_MISSING:'+','.join(
                    c for c in codes+['000001.SH','399006.SZ'] if c not in receipts))
            _persist_produce_diagnostics(folder,codes,spec,rounds)
            # Only a successful immutable collection is shared. Recovered raw
            # responses can be reused after a failed round; no old result is READY.
            rt.atomic_json(marker,dict(files={},receipts=receipts,failures=failures,audit=audit))
        def current(code):
            proof=receipts[code];body=Path(proof['reference']).read_bytes()
            if sha(body)!=proof['sha256']:raise ValueError('LIVE_RAW_SHA')
            raw=gzip.decompress(body)
            reference=refs.get(code)
            if reference is None:
                frame=pd.read_parquet(previous/(code+'.parquet'))
                reference=dict(close=float(frame[frame.date==P*1000000+150000].iloc[0]['close']))
            return parse(raw,code,P,cutoff,reference['close']),proof
        def history(code):
            info=read_lc5_file(spec['files'][code]['path'])
            return [r for r in info['rows'] if r['day']<day],dict(reference=spec['files'][code]['path']+'#'+info['sha256'],
                clock='END_5M',price_basis='RAW')
        context_marker=folder/'context_ready.json'
        context_key=rt.digest([CONTRACT,rt.native.sha(Path(__file__).with_name('intraday_native_context_v1.py')),
            rt.native_math_version()])
        if context_marker.exists():
            proof_meta=rt.read_json(context_marker)
            if proof_meta['math_key']!=context_key:raise ValueError('LIVE_COMMON_MATH_CHANGED')
            for name,h in proof_meta['files'].items():
                if rt.native.sha(folder/name)!=h:raise ValueError('LIVE_CONTEXT_SHA')
            for path,version in proof_meta['raw_versions'].items():
                st=Path(path).stat()
                if [st.st_size,st.st_mtime_ns]!=version:raise ValueError('LIVE_RAW_VERSION_CHANGED')
            with np.load(folder/'context.npz',allow_pickle=False) as z:
                returns={f:z['ret'+str(f)].copy() for f in (5,15)}
            cache=NativeIntradayCache.restore(rt.read_json(folder/'context.json'),returns,codes,current,history)
        else:
            cache=NativeIntradayCache(days,cutoff,old['boards'],codes,refs,current,history)
            rt.check_budget(runtime,16*1024**2)
            rt.atomic_json(folder/'context.json',cache.common_metadata())
            np.savez_compressed(folder/'context.npz',ret5=cache.returns[5],ret15=cache.returns[15])
            versions={p['reference']:[Path(p['reference']).stat().st_size,Path(p['reference']).stat().st_mtime_ns]
                      for p in receipts.values()}
            rt.atomic_json(context_marker,dict(math_key=context_key,raw_versions=versions,
                files={n:rt.native.sha(folder/n) for n in ('context.json','context.npz')}))
        if not set(codes)<=set(cache.receipts):raise ValueError('LIVE_CANDIDATE_PREFIX_MISSING')
        # Persist only selective technical rows. Later plan changes reuse all
        # raw/context inputs and already-computed observed leaders.
        detail=folder/'details.parquet';proof=detail.with_suffix('.sha256')
        if detail.exists():
            if not proof.exists() or rt.native.sha(detail)!=proof.read_text():raise ValueError('LIVE_DETAIL_SHA')
            cached=pd.read_parquet(detail)
            cache.detail_rows={str(c):g for c,g in cached.groupby('code')}
        for c in cache.detail_codes:
            if c in cache.receipts:cache.stock(c,days,cutoff)
        rt.check_budget(runtime,4*1024**2)
        if cache.detail_rows:
            temp=detail.with_suffix('.tmp');pd.concat(cache.detail_rows.values(),ignore_index=True).to_parquet(temp,index=False)
            temp.replace(detail);rt.atomic_json(folder/'detail_audit.json',cache.audit)
            proof.write_text(rt.native.sha(detail))
        indices={}
        for code in ('000001.SH','399006.SZ'):
            historical=pd.read_parquet(previous/(code+'.parquet'));now,_=current(code)
            indices[code]=pd.concat([historical[historical.date<day*1000000],pd.DataFrame(now)],ignore_index=True)
        observed=rt.native.stamp()
        snap=RiskSnapshot(cutoff,observed,days,rt.StockMap({c:None for c in cache.receipts}),indices,old['boards'],
            receipts,old.get('membership_known_at'),'FORWARD_SNAPSHOT',observed_at=observed)
        from joint_shadow_context_v1 import _market
        market=folder/'market.parquet';shared=folder/'shared.parquet';states=folder/'states.json'
        market_marker=folder/'market_ready.json'
        if market_marker.exists():
            for name,h in rt.read_json(market_marker).items():
                if rt.native.sha(folder/name)!=h:raise ValueError('LIVE_MARKET_SHA')
            snap._tdx_market_context=(pd.read_parquet(market),rt.read_json(states),pd.read_parquet(shared))
        else:
            m,s,v=_market(snap);m.to_parquet(market,index=False);v.to_parquet(shared,index=False);rt.atomic_json(states,s)
            rt.atomic_json(market_marker,{p.name:rt.native.sha(p) for p in (market,shared,states)})
            snap._tdx_market_context=(m,s,v)
        snap._source_contract=CONTRACT
        progress(stage='LIVE_SHARED_READY',done=len(cache.receipts),total=len(cache.codes))
        from live_postclose_source_v1 import LocalCapture
        capture=LocalCapture(output,budget_root=runtime)
        capture.save('source_manifest.json',json.dumps(dict(contract=CONTRACT,cutoff=cutoff,
            previous=str(previous),board_count=len(old['boards']),declared=len(cache.codes),
            supplied=len(cache.receipts),audit=audit,cache_audit=cache.audit,failures=cache.failures,
            receipts=receipts),ensure_ascii=False).encode())
        return snap,cache,capture,failures
