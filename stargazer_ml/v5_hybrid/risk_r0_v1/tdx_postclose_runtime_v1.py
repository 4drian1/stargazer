"""Native TDX postclose worker/cache. Completed artifacts only; no stockdb IO."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(PROJECT/'stargazer_desktop'),str(HERE.parent/'green_common_profile')]
import json,gzip,hashlib,subprocess,io
import numpy as np
import pandas as pd
import tdx_native_kernel_v1 as native
from tdx_native_history_v1 import source_root,configured
from live_postclose_source_v1 import LocalCapture,disk_limit,atomic_json
from joint_shadow_context_v1 import RiskSnapshot
CONTRACT='TDX_NATIVE_POSTCLOSE_V1'

_LC5_MEMO = {}
_LC5_MEMO_BYTES = [0]
_LC5_MEMO_CAP = 256 * 1024 * 1024  # byte-bounded read memo (#18.199-补)
_LC5_DISK_BYTES = [0]  # actual disk reads (memo misses), vs logical bytes_read


def unified_cache_headroom(runtime):
    """Single merged ledger (#18.199-补5): the authorized cache cap minus the
    PATH-DEDUPLICATED sum of the new cache zone, the FULL legacy runtime area
    and the original shared archive. Per-stock cache writes and check_budget()
    both consume this one function; until the independent asset zones and the
    migration complete, nothing in the old area leaves the account."""
    import storage_roots_v1 as zones
    from runtime_disk_limit_v1 import disk_usage, MONITOR_RESERVE, disk_limit
    runtime = Path(runtime).resolve()
    budget = disk_limit(budget_root(runtime))  # THE authorized cap (fail-closed)
    zone_dir = zones.zone_root('cache').resolve()
    zone_bytes = zones.zone_usage('cache')
    legacy = disk_usage(runtime)  # legacy runtime area + shared archive
    try:
        nested = zone_dir.is_relative_to(runtime)
    except AttributeError:
        nested = str(zone_dir).startswith(str(runtime) + os.sep)
    if nested:
        legacy -= zone_bytes  # counted once only
    return budget - legacy - zone_bytes - MONITOR_RESERVE


def read_lc5_memo(path):
    """One parse per run per source version: re-parse only when the file
    version changes. Bounded by parsed-row count — raw bytes badly understate
    the parsed footprint, so the cap is on rows, not file size."""
    p = Path(path); st = p.stat()
    key = (str(p), st.st_mtime_ns, st.st_size)
    if key in _LC5_MEMO:
        return _LC5_MEMO[key]
    info = native.read_lc5_file(str(p))
    _LC5_DISK_BYTES[0] += info['bytes']
    while _LC5_MEMO and _LC5_MEMO_BYTES[0] + len(info['rows']) > 400_000:
        _, evicted = _LC5_MEMO.popitem()
        _LC5_MEMO_BYTES[0] -= len(evicted['rows'])
    _LC5_MEMO[key] = info
    _LC5_MEMO_BYTES[0] += len(info['rows'])
    return info


def history_digest(code, rows, P, legal):
    """Identity of the resumable history foundation: verified bars through the
    trading day BEFORE P (legal = the shared trading calendar). A historical
    revision inside the coverage changes it (correct invalidation); appending
    the new P day does not. raw_volume is the native field — volume would
    silently hash as 0 and hide volume revisions."""
    legal = [int(d) for d in legal]
    P = int(P)
    if P not in legal or legal.index(P) == 0:
        raise ValueError('HISTORY_DIGEST_NO_PREVIOUS_DAY')
    coverage_day = legal[legal.index(P) - 1]
    hist = [(r['day'], r['ts'], r['open'], r['high'], r['low'], r['close'],
             r.get('raw_volume', r.get('volume', 0)))
            for r in rows if r['day'] <= coverage_day]
    return digest([code, coverage_day, hist])


def today_digest(code, rows, P, cutoff, legal, days, math_ver, capture_ver, native_ver):
    """Result-of-the-day identity: P-day bars, the cutoff AND the actual
    shared day list are IN the key. A different calendar starting point
    produces a different ``days`` span and must not collide with this key.
    Mutating a same-day price invalidates the day's cached result even though
    the history foundation key did not change."""
    base = history_digest(code, rows, P, legal)
    today = [(r['ts'], r['open'], r['high'], r['low'], r['close'],
              r.get('raw_volume', r.get('volume', 0)))
             for r in rows if r['day'] == int(P)]
    return digest([base, int(cutoff), [int(d) for d in days], today,
                   math_ver, capture_ver, native_ver])


def foundation_payload(code, rows, P, legal):
    """Carry-forward state for exact seeded continuation (next increment):
    the last covered day's bars, which seed RSI/EMA recurrences exactly."""
    legal = [int(d) for d in legal]
    coverage_day = legal[legal.index(int(P)) - 1]
    last = [dict(r) for r in rows if r['day'] == coverage_day]
    return dict(coverage_last_day=coverage_day, last_day_bars=last)

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def read_json(p):return json.loads(Path(p).read_bytes())
def budget_root(root):
    root=Path(root).resolve()
    for candidate in [root,*root.parents]:
        if (candidate/'runtime_budget_v2.json').is_file():return candidate
    return root
def check_budget(root,extra):
    root=budget_root(root)
    if unified_cache_headroom(root)<extra:raise ValueError('RUNTIME_DISK_BUDGET')
def write(folder,name,value):
    body=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
    check_budget(folder.parents[2],len(body))
    with (folder/name).open('xb') as f:f.write(body)

def fingerprint():
    return digest({name:native.sha(HERE/name) for name in
        ['tdx_postclose_runtime_v1.py','tdx_native_kernel_v1.py','tdx_native_reader_v1.py',
         'tdx_native_arrays_v1.py','tdx_native_indicators_v1.py','tdx_market_schedule_v1.py',
         'build_joint_risk_features_v1.py','joint_shadow_context_v1.py']})


_NATIVE_MATH_FILES = ('tdx_native_kernel_v1.py', 'tdx_native_reader_v1.py',
                      'tdx_native_arrays_v1.py', 'tdx_native_indicators_v1.py',
                      'tdx_market_schedule_v1.py')


def native_math_version():
    """Per-stock cache identity binds ONLY the native computation files.
    The orchestrator (tdx_postclose_runtime_v1.py: budgets, storage zones,
    dispatch) must NOT invalidate stock math — editing budget/storage/display
    code re-runs nothing (#18.199-补7)."""
    return digest({name: native.sha(HERE / name) for name in _NATIVE_MATH_FILES})

def settings(selection,runtime):
    from accept_live_postclose_v1 import plan_calendar
    legal,_=plan_calendar(selection)
    P,E=int(selection['data_date']),int(selection['monitor_date'])
    scope=read_json(HERE/'ths_full_concept_scope_20261017_v1.json')
    if native.sha(scope['source'])!=scope['source_sha256']:raise ValueError('TDX_BOARD_CATALOG_CHANGED')
    cat=pd.read_parquet(scope['source'],columns=['code','symbols'],filters=[('source','==','ths'),('type','==','concept')])
    boards={r.code:list(r.symbols) for r in cat.sort_values('code').itertuples()}
    if set(boards)!={b['code'] for b in scope['boards']}:raise ValueError('COMPARISON_UNIVERSE_DRIFT')
    universe={c for m in boards.values() for c in m}
    seed=read_json(HERE/'relative_stall_5m_20261018_v1/raw_input_ledger.json')
    from live_postclose_cache_v1 import prior_entries
    history_root=budget_root(runtime)
    history=prior_entries(history_root,P)
    for terminal in history_root.glob('run_*/terminal.json'):
        record=read_json(terminal)
        if (record.get('status')=='COMPLETE' and record.get('source_contract')==CONTRACT
                and 0<int(record.get('data_date',0))<P):
            history.update(read_json(terminal.parent/'source_manifest.json')['entries'])
    candidates=sorted({str(r['code']).zfill(6) for r in selection['rows']})
    bound=sorted((({r['code'] for r in seed}|set(history))&universe)|(set(candidates)&universe))
    root=source_root();index={}
    for market in ('sh','sz','bj'):
        for p in (root/market/'fzline').glob('*.lc5'):index.setdefault(p.stem[2:],[]).append(p)
    files={};missing={}
    for code in bound:
        paths=index.get(code,[])
        if len(paths)!=1:missing[code]='TDX_FILE_MISSING_OR_AMBIGUOUS';continue
        p=paths[0];st=p.stat();files[code]=dict(path=str(p),size=st.st_size,mtime_ns=st.st_mtime_ns)
    for symbol,stem in [('000001.SH','sh999999'),('399006.SZ','sz399006')]:
        p=root/stem[:2]/'fzline'/(stem+'.lc5');st=p.stat()
        files[symbol]=dict(path=str(p),size=st.st_size,mtime_ns=st.st_mtime_ns)
    return dict(contract=CONTRACT,P=P,E=E,cutoff=P*1000000+150000,legal=legal,boards=boards,
        bound=bound,candidates=candidates,files=files,missing=missing,kernel=fingerprint(),
        root=str(root),catalog_sha256=scope['source_sha256'])

WINDOW_DAYS = 40  # frozen by the offline equivalence acceptance (#18.199-补7)


def stock_windowed(code, days_full, cutoff, info, window=WINDOW_DAYS):
    """Exact-by-construction cross-day continuation: run the frozen kernel on
    the last ``window`` days only and keep the final day's rows/returns/
    indicators. Stateful indicators (RSI/EMA/MACD) converge well within the
    window; the equivalence acceptance compares this output against the full
    history computation column-by-column on real data before the window size
    is trusted."""
    import tdx_native_arrays_v1 as arrays_mod
    import tdx_native_indicators_v1 as indicators_mod
    P = cutoff // 1000000
    window_days = [int(d) for d in days_full if int(d) <= P][-window:]
    raw = [r for r in info['rows'] if r['ts'] <= cutoff]
    by_day = {}
    for row in raw:
        by_day.setdefault(row['day'], []).append(row)
    allowed = native.good_days(raw)
    arrays, rejected = arrays_mod.build_minute_arrays(by_day, window_days, allowed_days=allowed)
    quality = native.build_quality(arrays, rejected)
    rows, ret15, _ = arrays_mod.run_features(arrays, code, window_days, quality)
    tail = rows[rows.cutoff_index // 16 == len(window_days) - 1].copy()
    tail['cutoff_index'] = tail['cutoff_index'] + (len(days_full) - len(window_days)) * 16
    daily = arrays[15]['close'].reshape(-1, 16)
    closes = np.where(np.isfinite(daily).all(axis=1), daily[:, -1], np.nan)
    prev = np.r_[np.nan, closes[:-1]]
    refs = np.repeat(prev, 48)
    ret5 = np.divide(arrays[5]['close'], refs,
                     out=np.full(len(refs), np.nan), where=np.isfinite(refs) & (refs > 0)) - 1
    ret = {5: np.where(np.isfinite(ret5) & (ret5 > -1), ret5, np.nan),
           15: np.where(np.isfinite(ret15) & (ret15 > -1), ret15, np.nan)}
    indicators = None
    try:
        indicators = indicators_mod.compute_yesterday_indicators(
            arrays, window_days, calendar=list(window_days) + [int(cutoff // 1000000) + 1],
            data_day=P, monitor_day=int(cutoff // 1000000) + 1)
    except (ValueError, KeyError, TypeError):
        indicators = None
    return tail, ret, indicators


def resolve_stock(code, spec, cache, roots, versions, remaining, *, info=None, timing=None):
    """One stock through the cache: hit if the cached result still satisfies
    the CURRENT run's requirements; otherwise compute this stock only and
    persist. A plain member cached without candidate indicators stays valid;
    when that code joins the candidate set the hit is rejected and ONLY this
    stock is recomputed with indicators — never the whole market.

    Returns (status, bytes_written). Raises on budget exhaustion."""
    import time as _time
    P, E = spec['P'], spec['E']
    days = cache.days
    if info is None:
        t0 = _time.perf_counter()
        info = read_lc5_memo(spec['files'][code]['path'] if code in spec.get('files', {}) else cache.sources[code]['path'])
        if timing is not None:
            timing['read_parse'] += _time.perf_counter() - t0
            timing['bytes_read'] += info['bytes']
    if (info['bytes'], info['mtime_ns']) != (spec['files'][code]['size'], spec['files'][code]['mtime_ns']):
        raise ValueError('SOURCE_VERSION_CHANGED:' + code)
    prefix = [r for r in info['rows'] if r['ts'] <= spec['cutoff']]
    key = today_digest(code, prefix, P, spec['cutoff'], spec['legal'], days,
                       versions['math'], versions['capture'], versions['native'])
    needs_indicators = code in cache.prep

    def _load(v):
        cache.tails.append(pd.DataFrame(v['rows']))
        cache.return_codes.append(code)
        cache.returns5.append(np.array(v['ret5'], dtype=float))
        cache.returns15.append(np.array(v['ret15'], dtype=float))
        if v.get('indicators') is not None:
            cache.indicators[code] = v['indicators']

    stale = None
    for root in roots:
        target = root / code / (key + '.json.gz')
        if target.exists():
            wrapped = json.loads(gzip.decompress(target.read_bytes()))
            v = wrapped['payload']
            if digest(v) != wrapped['sha256']:
                raise ValueError('TDX_STOCK_CACHE_HASH:' + code)
            # Current-requirement validation: a hit generated when this code
            # was a plain member carries no candidate indicators; that is a
            # miss for the current run, and only this stock is backfilled.
            if needs_indicators and v.get('indicators') is None:
                stale = target
                break
            _load(v)
            return 'hit', 0
    # Cross-day continuation: if the previous trading day's verified entry
    # exists, only the window covering the new day runs through the frozen
    # kernel (exactness proven by the offline equivalence acceptance).
    hid = history_digest(code, prefix, P, spec['legal'])
    windowed = None
    prev_day = spec['legal'][spec['legal'].index(int(P)) - 1]
    for root in roots:
        for entry in sorted((root / code).glob('*.json.gz')):
            try:
                wrapped = json.loads(gzip.decompress(entry.read_bytes()))
                v0 = wrapped['payload']
                if digest(v0) != wrapped['sha256']:
                    continue
                ident = v0.get('identity') or {}
                foundation = v0.get('foundation') or {}
                if ident.get('history') == hid and \
                        foundation.get('coverage_last_day') == prev_day:
                    windowed = stock_windowed(code, cache.days, spec['cutoff'], info)
                    break
            except (OSError, ValueError, KeyError):
                continue
        if windowed is not None:
            break
    t1 = _time.perf_counter()
    if windowed is not None:
        tail, ret, indicators = windowed
        if timing is not None:
            timing['continued'] = timing.get('continued', 0) + 1
        cache.tails.append(tail)
        cache.return_codes.append(code)
        cache.returns5.append(ret[5][-48:].copy())
        cache.returns15.append(ret[15][-16:].copy())
        if indicators is not None:
            cache.indicators[code] = indicators
    else:
        rows, ret = cache.stock(code, days, spec['cutoff'], info=info)
        if timing is not None:
            timing['features_full'] += _time.perf_counter() - t1
    def finite(value):
        return None if isinstance(value, float) and not np.isfinite(value) else value
    v = dict(rows=[{k: finite(x) for k, x in r.items()} for r in cache.tails[-1].to_dict('records')],
             ret5=[finite(x) for x in ret[5][-48:].tolist()],
             ret15=[finite(x) for x in ret[15][-16:].tolist()],
             indicators=cache.indicators.get(code),
             foundation=foundation_payload(code, prefix, P, spec['legal']),
             identity=dict(history=history_digest(code, prefix, P, spec['legal']),
                           math=versions['math'], capture=versions['capture'],
                           native=versions['native']))
    body = gzip.compress(json.dumps(dict(payload=v, sha256=digest(v)), allow_nan=False).encode(), mtime=0)
    if len(body) > remaining:
        raise ValueError('RUNTIME_DISK_BUDGET')
    t2 = _time.perf_counter()
    target = roots[0] / code / (key + '.json.gz')
    target.parent.mkdir(parents=True, exist_ok=True)
    if stale is not None and stale.exists() and stale != target:
        stale.unlink()  # superseded member entry: replaced by the backfilled one
    if not target.exists():
        with target.open('xb') as f:
            f.write(body)
    elif stale is not None:
        # Same path, stale content (member-era payload): overwrite in place.
        temp = target.with_suffix('.json.gz.tmp')
        temp.write_bytes(body)
        temp.replace(target)
    if timing is not None:
        timing['io_write'] += _time.perf_counter() - t2
    return 'miss', len(body)


def _write_progress(folder, spec, *, done, total, hits, misses, final=False):
    """Real progress on disk: the UI reads state.json done/total instead of
    guessing. Written every 100 stocks and once at the end; candidates finish
    first so the plan is never hostage to the full member universe."""
    body = dict(contract='TDX_STOCK_PROGRESS_V1', P=spec['P'], done=done, total=total,
                hits=hits, misses=misses, candidates=sorted(spec['candidates']),
                final=final, updated_at=native.stamp())
    path = Path(folder) / 'stock_progress.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(body, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)
    # The UI counter (app.js riskSlotSummary) reads state.json done/total —
    # keep it in step with the real loop, never left at 0/total.
    state_path = Path(folder).parents[1] / 'state.json'
    try:
        state = json.loads(state_path.read_bytes())
        if state.get('status') == 'RUNNING' and int(state.get('data_date', 0)) == int(spec['P']):
            state['done'] = done
            state['total'] = total
            temp = state_path.with_suffix('.tmp')
            temp.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
            temp.replace(state_path)
    except (OSError, ValueError):
        pass


def stock_phase(folder):
    spec=read_json(folder/'request.json');P,E=spec['P'],spec['E'];native.P=P;native.E=E;native.CUTOFF=spec['cutoff']
    sources={};failures=dict(spec['missing']);indices={};receipts={};prefixes={}
    observed=native.stamp()
    timing = dict(read_parse=0.0, features_full=0.0, io_write=0.0,
                  bytes_read=0, rebuilt_reason='FULL_HISTORY_FALLBACK')
    import time as _time
    for index,(code,item) in enumerate(sorted(spec['files'].items())):
        t0=_time.perf_counter()
        info=read_lc5_memo(item['path'])
        timing['read_parse'] += _time.perf_counter() - t0
        timing['bytes_read'] += info['bytes']
        if index % 200 == 0:
            native.memory_guard()  # first bulk-read loop peak check too
        if (info['bytes'],info['mtime_ns'])!=(item['size'],item['mtime_ns']):raise ValueError('SOURCE_VERSION_CHANGED:'+code)
        prefix=[r for r in info['rows'] if r['ts']<=spec['cutoff']]
        good=native.good_days(prefix)
        if P not in good:
            if code in ('000001.SH','399006.SZ'):raise ValueError('INDEX_P_DAY_INCOMPLETE:'+code)
            failures[code]='P_DAY_INCOMPLETE';continue
        ref=dict(path=item['path'],sha256=info['sha256'],bytes=info['bytes'],available_at=observed,
            clock='END_5M',price_basis='RAW',availability_basis='LOCAL_READER_OBSERVED_BYTES_NOT_UPSTREAM_FIRST_ARRIVAL')
        if code in ('000001.SH','399006.SZ'):
            indices[code]=(prefix,good);receipts[code]=dict(ref,reference=item['path']+'#'+info['sha256'])
        else:
            sources[code]=ref;prefixes[code]=digest(prefix)
    if set(spec['candidates'])-set(sources):raise ValueError('REQUIRED_CANDIDATE_MINUTES_MISSING')
    first=max(min(v[1]) for v in indices.values())
    days=[d for d in spec['legal'] if first<=d<=P]
    if len(days)<2:raise ValueError('INDEX_HISTORY_TOO_SHORT')
    for code,(rows,good) in indices.items():
        if not set(days)<=good:raise ValueError('INDEX_SHARED_CALENDAR_INCOMPLETE:'+code)
        pd.DataFrame(rows).rename(columns={'ts':'date','raw_volume':'volume'}).to_parquet(folder/(code+'.parquet'),index=False)
    cache=native.NativeCache(sources,days,set(spec['candidates']))
    runtime=folder.parents[2]
    from cache_layer_keys_v1 import math_key, capture_key
    math_ver, capture_ver = math_key(), capture_key()
    native_ver = native_math_version()  # computation deps only, NOT the orchestrator
    # Rebuildable per-code intermediates live in the cache zone and are
    # accounted against the cache zone at the ACTUAL write location; the
    # legacy runtime cache still counts so nothing escapes the account.
    import storage_roots_v1 as zones
    stock_cache_root = zones.cache_root() / 'tdx_stock_cache'
    legacy_stock_cache = runtime / 'tdx_stock_cache'
    # Unified ledger (#18.199-补5): new cache zone + FULL legacy runtime area
    # + original shared archive, path-deduplicated, one authorized cap. The
    # previous formula only counted the legacy tdx_stock_cache subdirectory
    # and under-reported the old area by ~2.27 GiB.
    remaining = unified_cache_headroom(runtime)
    hits=misses=0
    versions = dict(math=math_ver, capture=capture_ver, native=native_ver)
    roots = (stock_cache_root, legacy_stock_cache)
    import time as _time
    candidates_first = set(spec['candidates'])
    ordered = sorted(sources, key=lambda c: (c not in candidates_first, c))
    for index, code in enumerate(ordered):
        t0 = _time.perf_counter()
        info = read_lc5_memo(sources[code]['path'])
        timing['read_parse'] += _time.perf_counter() - t0  # external info bypasses nothing
        timing['bytes_read'] += info['bytes']
        status, delta = resolve_stock(code, spec, cache, roots, versions,
                                      remaining, info=info, timing=timing)
        remaining -= delta
        if status == 'hit': hits += 1
        else: misses += 1
        if index % 100 == 0:
            _write_progress(folder, spec, done=index + 1, total=len(ordered),
                            hits=hits, misses=misses)
            native.memory_guard()  # batch peak check in the compute loop
    _write_progress(folder, spec, done=len(ordered), total=len(ordered),
                    hits=hits, misses=misses, final=True)
    check_budget(folder.parents[2],16*1024**2)
    pd.concat(cache.tails,ignore_index=True).to_parquet(folder/'stocks.parquet',index=False)
    np.savez_compressed(folder/'returns.npz',codes=np.array(cache.return_codes),ret5=np.array(cache.returns5),ret15=np.array(cache.returns15))
    receipts.update({c:dict(s,reference=s['path']+'#'+s['sha256']) for c,s in sources.items()})
    write(folder,'indicators.json',cache.indicators)
    write(folder,'source_manifest.json',dict(contract=CONTRACT,entries=sources,failures=failures,
        bound_stock_count=len(sources),declared_stock_count=len({c for m in spec['boards'].values() for c in m}),
        comparison_board_count=len(spec['boards']),attempted_stock_count=len(spec['bound'])))
    write(folder,'snapshot.json',dict(cutoff=spec['cutoff'],days=days,boards=spec['boards'],receipts=receipts,
        observed_at=observed,available_at=observed,membership_known_at=observed,
        membership_scope='POSTCLOSE_CAPTURE_OBSERVED_LATER' if observed//1000000!=P else 'FORWARD_SNAPSHOT'))
    native.memory_guard()
    write(folder,'stock_complete.json',dict(peak_memory=native.peak_memory(),hits=hits,misses=misses,
        timing=dict(read_parse=round(timing['read_parse'],3),features_full=round(timing['features_full'],3),
                    io_write=round(timing['io_write'],3),
                    bytes_read_logical=timing['bytes_read'],  # includes memo hits
                    disk_read_bytes=_LC5_DISK_BYTES[0]),      # actual cold reads only
        rebuild=dict(count=misses,reason=timing['rebuilt_reason'],
                     note='指标状态精确增量未实现：保留完整历史计算作为回退，先据此计时决定需持久化的状态')))

def market_phase(folder):
    import tdx_market_schedule_v1
    tdx_market_schedule_v1.enable()
    from joint_shadow_context_v1 import _market
    meta=read_json(folder/'snapshot.json')
    indices={c:pd.read_parquet(folder/(c+'.parquet')) for c in ('000001.SH','399006.SZ')}
    snapshot=RiskSnapshot(meta['cutoff'],meta['available_at'],meta['days'],{},indices,meta['boards'],
        meta['receipts'],meta['membership_known_at'],meta['membership_scope'],observed_at=meta['observed_at'])
    market,states,shared=_market(snapshot)
    native.memory_guard();check_budget(folder.parents[2],16*1024**2)
    market.to_parquet(folder/'market.parquet',index=False);shared.to_parquet(folder/'shared.parquet',index=False)
    write(folder,'states.json',states)
    files={p.name:native.sha(p) for p in folder.iterdir() if p.is_file() and p.name!='complete.json'}
    write(folder,'complete.json',dict(contract=CONTRACT,files=files,peak_memory=native.peak_memory()))

class StockMap(dict):
    quarantined={}

def loaded(folder,output,runtime):
    done=read_json(folder/'complete.json')
    if read_json(folder/'request.json')['kernel']!=fingerprint():raise ValueError('TDX_PREPARED_KERNEL_CHANGED')
    for name,h in done['files'].items():
        if native.sha(folder/name)!=h:raise ValueError('TDX_PREPARED_CACHE_HASH:'+name)
    meta=read_json(folder/'snapshot.json');manifest=read_json(folder/'source_manifest.json')
    frame=pd.read_parquet(folder/'stocks.parquet');rows={str(c):g for c,g in frame.groupby('code')}
    with np.load(folder/'returns.npz',allow_pickle=False) as z:
        codes=z['codes'].tolist();arrays={f:{c:z['ret'+str(f)][i].copy() for i,c in enumerate(codes)} for f in (5,15)}
    indices={c:pd.read_parquet(folder/(c+'.parquet')) for c in ('000001.SH','399006.SZ')}
    snap=RiskSnapshot(meta['cutoff'],meta['available_at'],meta['days'],StockMap({c:None for c in manifest['entries']}),
        indices,meta['boards'],meta['receipts'],meta['membership_known_at'],meta['membership_scope'],observed_at=meta['observed_at'])
    snap._prepared_context=(rows,arrays)
    snap._tdx_market_context=(pd.read_parquet(folder/'market.parquet'),read_json(folder/'states.json'),pd.read_parquet(folder/'shared.parquet'))
    snap._source_contract=CONTRACT
    snap._source_identity=digest(read_json(folder/'request.json'))
    capture=LocalCapture(output,budget_root=budget_root(runtime))
    capture.save('source_manifest.json',(folder/'source_manifest.json').read_bytes())
    return snap,capture,manifest['failures']

def produce(day,codes,output,progress,*,cutoff=None,index_source=None):
    if cutoff is not None and cutoff!=int(day)*1000000+150000:raise ValueError('TDX_POSTCLOSE_ONLY')
    output=Path(output);runtime=output.parent;selection=read_json(output/'input_snapshot.json')
    spec=settings(selection,runtime)
    if int(day)!=spec['P'] or sorted(codes)!=spec['candidates']:raise ValueError('TDX_SELECTION_IDENTITY')
    key=digest(spec);folder=runtime/'tdx_prepared'/str(day)/key
    marker=folder/'complete.json'
    if not marker.exists():
        check_budget(runtime,48*1024**2);folder.mkdir(parents=True,exist_ok=True)
        if any(folder.iterdir()):raise ValueError('TDX_PREPARATION_INCOMPLETE_RETAINED')
        write(folder,'request.json',spec)
        for phase in ('stock','market'):
            progress(stage='TDX_PREPARE_'+phase.upper(),done=0,total=len(spec['bound']))
            result=subprocess.run([sys.executable,'-B','-X','utf8',str(Path(__file__)),phase,str(folder)],
                stdout=sys.stdout,stderr=sys.stderr,timeout=7200,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if result.returncode:raise ValueError('TDX_PREPARE_PHASE_FAILED:'+phase)
    progress(stage='REUSING_PREPARED_CONTEXT',done=len(codes),total=len(codes))
    return loaded(folder,output,runtime)

def ready(selection):
    if not configured():return False
    from tdx_native_history_v1 import native_rows
    try:
        for row in selection['rows']:native_rows(str(row['code']).zfill(6),[int(selection['data_date'])])
        return True
    except (OSError,ValueError,KeyError):return False

produce.native_tdx=True

if __name__=='__main__':
    # Readiness / identity pre-check used by the frozen GUI. Runs in a SEPARATE
    # python process so the GUI never imports the native kernel / risk chain; it
    # only reads this script's JSON stdout. Read-only: no artifacts written.
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        selection = read_json(sys.argv[2])
        runtime = sys.argv[3] if len(sys.argv) > 3 else None
        out = dict(contract=CONTRACT)
        try:
            if not configured():
                out['ready'] = False
                out['reason'] = '通达信根目录未配置'
            else:
                out['ready'] = ready(selection)
                if out['ready']:
                    out['identity'] = digest(settings(selection, runtime))
                else:
                    # ready() returned False without raising: supply a concrete reason
                    # so the GUI, which reads ['reason'] directly, never hits KeyError.
                    out['reason'] = '通达信名单日分钟文件未齐或未通过校验'
        except (OSError, ValueError, KeyError) as exc:
            out['ready'] = False
            out['reason'] = str(exc)
        print(json.dumps(out, ensure_ascii=False))
        sys.exit(0)
    phase,location=sys.argv[1:];folder=Path(location).resolve()
    if phase not in ('stock','market'):raise ValueError('TDX_PREPARE_PHASE')
    # Worker writes only beneath its explicitly provided runtime/tdx_prepared/day.
    if folder.parents[1].name!='tdx_prepared':raise ValueError('TDX_PREPARE_TARGET')
    try:(stock_phase if phase=='stock' else market_phase)(folder)
    except Exception as exc:
        write(folder,'failed_'+phase+'.json',dict(error=type(exc).__name__+':'+str(exc)))
        raise
