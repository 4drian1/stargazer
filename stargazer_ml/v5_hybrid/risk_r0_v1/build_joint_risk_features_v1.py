"""Stream existing stock history into a factorized, same-cutoff joint risk table.

All THS concepts remain present. Observed partial baskets/ranks are explicitly
not certified complete-universe ranks. No HTTP, labels, fit or trading actions.
"""
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from inventory_ths_stockdb_cache_v1 import HERE, ROOTS, digest

OUT = HERE/'joint_risk_features_20261017_v1'
MARKET = HERE/'local_market_context_20261017_v1/market_price_features.parquet'
CATALOG = HERE/'ths_full_concept_scope_20261017_v1.json'
AUTH = HERE/'contracts/ths_snapshot_reconstruction_chat_20261017.json'
INVENTORY = HERE/'ths_stockdb_cache_inventory_20261017_v1.json'
QUARANTINE = HERE.parents[1]/'pullback_train/quarantine_2026.csv'
CAP, RESERVE = 3*1024**3, 1024**2


def wilder_rsi(values, period=14):
    """Vectorized within contiguous valid stretches; missing bars reset warmup."""
    x = np.asarray(values, dtype=float)
    out = np.full(len(x), np.nan)
    valid = np.isfinite(x)
    edges = np.diff(np.r_[False, valid, False].astype(int))
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if end-start <= period:
            continue
        change = np.diff(x[start:end])
        gain, loss = np.maximum(change, 0), np.maximum(-change, 0)
        g = np.r_[gain[:period].mean(), gain[period:]]
        l = np.r_[loss[:period].mean(), loss[period:]]
        g = pd.Series(g).ewm(alpha=1/period, adjust=False).mean().to_numpy()
        l = pd.Series(l).ewm(alpha=1/period, adjust=False).mean().to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            rsi = 100-100/(1+g/l)
        rsi[(g == 0) & (l == 0)] = 50
        rsi[(g > 0) & (l == 0)] = 100
        out[start+period:end] = rsi
    return out


def lag(x, size=1):
    return np.r_[np.full(size, np.nan), x[:-size]]


def ratio(x, reference):
    with np.errstate(divide='ignore', invalid='ignore'):
        out = x/reference
    out[~np.isfinite(out) | (reference <= 0)] = np.nan
    return out


def aggregate_minutes(frame, code, days, quarantine_days):
    """END reconstruction only. No price adjustment/unknown slots are fabricated."""
    df = frame.copy()
    if not (df['code'].astype(str) == code).all():
        raise ValueError('SOURCE_CODE_MISMATCH')
    fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
    conflicts = set()
    dup = df[df.duplicated('date', keep=False)]
    if len(dup):
        bad = dup.groupby('date')[fields].nunique(dropna=False).max(axis=1)
        conflicts = set(bad[bad > 1].index)
    df = df.drop_duplicates('date').sort_values('date')
    timestamps = df['date'].to_numpy(dtype=np.int64)
    day = timestamps//1000000
    minute = (timestamps//10000 % 100)*60+(timestamps//100 % 100)
    am, pm = (minute > 570) & (minute <= 690), (minute > 780) & (minute <= 900)
    numbers = df[fields].to_numpy(dtype=float)
    o, h, l, c, v, a = numbers.T
    valid = (np.isfinite(numbers).all(axis=1) & (np.minimum.reduce([o,h,l,c]) > 0)
             & (l <= np.minimum(o,c)) & (h >= np.maximum(o,c)) & (v >= 0) & (a >= 0)
             & (timestamps % 100 == 0) & ((timestamps//100 % 100) < 60) & (am | pm)
             & ~np.isin(timestamps, list(conflicts)) & ~np.isin(day, list(quarantine_days)))
    index = np.searchsorted(days, day)
    in_calendar = index < len(days)
    in_calendar[in_calendar] &= days[index[in_calendar]] == day[in_calendar]
    valid &= in_calendar
    slot = np.where(am, minute-571, 120+minute-781)
    data = pd.DataFrame(numbers[valid], columns=fields)
    positions = index[valid]*240+slot[valid]
    arrays = {}
    for step in (5, 15):
        data['key'] = positions//step
        grouped = data.groupby('key', sort=True).agg(
            open=('open','first'), high=('high','max'), low=('low','min'),
            close=('close','last'), volume=('volume','sum'), count=('close','count'))
        grouped = grouped[grouped['count'] == step]
        length = len(days)*(240//step)
        arrays[step] = {}
        for field in ['open','high','low','close','volume']:
            x = np.full(length, np.nan)
            x[grouped.index.to_numpy(dtype=int)] = grouped[field].to_numpy()
            arrays[step][field] = x
    quality = dict(conflicting_timestamps=len(conflicts),
                   invalid_or_excluded_rows=int((~valid).sum()), quarantine_days=len(quarantine_days),
                   feature_rows=int(np.isfinite(arrays[15]['close']).sum()))
    return arrays, quality


def stock_features(frame, code, days, quarantine_days, returns_only=False):
    arrays, quality = aggregate_minutes(frame, code, days, quarantine_days)
    return features_from_arrays(arrays, code, days, quality, returns_only)


def features_from_arrays(arrays, code, days, quality, returns_only=False):
    """Same complete-history indicator kernel for raw and cached aggregates."""
    x5, x15 = arrays[5], arrays[15]
    close, volume = x15['close'], x15['volume']
    day_close = close.reshape(-1,16)[:,-1].copy()
    day_close[~np.isfinite(close.reshape(-1,16)).all(axis=1)] = np.nan
    reference = np.repeat(lag(day_close),16)
    returns = ratio(close, reference)-1
    if returns_only:
        return None, returns, quality
    pick5 = np.arange(len(close))*3+2
    p5, p15 = ratio(x5['close'],lag(x5['close']))-1, ratio(close,lag(close))-1
    p5[np.arange(len(p5)) % 24 == 0] = np.nan
    p15[np.arange(len(p15)) % 8 == 0] = np.nan
    r5, r15 = wilder_rsi(x5['close']), wilder_rsi(close)
    rows = pd.DataFrame(dict(cutoff_index=np.arange(len(close)),
        stock_close=close, stock_reference_close_raw=reference, stock_day_return_raw=returns,
        stock_return5m_raw=p5[pick5], stock_return15m_raw=p15,
        stock_momentum_change5m_raw=(p5-lag(p5))[pick5],
        stock_momentum_change15m_raw=p15-lag(p15),
        stock_rsi14_5m_raw=r5[pick5], stock_rsi14_15m_raw=r15,
        stock_rsi_delta5m_raw=(r5-lag(r5))[pick5], stock_rsi_delta15m_raw=r15-lag(r15),
        stock_volume_ratio_previous_session_raw=ratio(volume,lag(volume,16)),
        stock_intrabar_return15m_raw=ratio(close,x15['open'])-1))
    rows = rows[np.isfinite(rows['stock_close'])].copy().reset_index(drop=True)
    rows.insert(0,'code',code)
    rows['stock_divergence_confirmed'] = None
    rows['stock_corporate_action_scale_verified'] = False
    return rows, returns, quality


def basket_features(returns, member_indexes):
    """No zero contribution for missing members, common-member momentum explicit."""
    values = returns[member_indexes]
    valid = np.isfinite(values)
    n = valid.sum(axis=0)
    total = np.where(valid,values,0).sum(axis=0)
    mean = np.divide(total,n,out=np.full(total.shape,np.nan),where=n>0)
    up = np.divide(((values > 0) & valid).sum(axis=0),n,out=np.full(total.shape,np.nan),where=n>0)
    common = valid[:,1:] & valid[:,:-1]
    counts = common.sum(axis=0)
    delta = np.where(common,values[:,1:]-values[:,:-1],0).sum(axis=0)
    momentum = np.r_[np.nan,np.divide(delta,counts,out=np.full(delta.shape,np.nan),where=counts>0)]
    momentum[np.arange(len(momentum)) % 8 == 0] = np.nan
    return dict(observed_return_raw=mean, observed_breadth=up, present=n,
                coverage=n/len(member_indexes), common_member_return_change=momentum)


def observed_leaders(board_returns, board_counts, sizes, members, stock_returns, board_codes, stock_codes):
    """Partial observations named observed; strict leader left unknown without full coverage."""
    traces, captured_boards, captured_stocks = [], set(), set()
    for t in range(board_returns.shape[1]):
        if t % 16 == 0:
            captured_boards, captured_stocks = set(), set()
        available = np.flatnonzero(np.isfinite(board_returns[:,t]))
        board = stock = None
        if len(available):
            best = np.max(board_returns[available,t])
            winners = available[board_returns[available,t] == best]
            if len(winners) == 1:
                board = int(winners[0]); captured_boards.add(board)
                indexes = np.asarray(members[board],dtype=int)
                eligible = np.asarray([j for j in indexes if np.isfinite(stock_returns[j,t]) and
                    stock_codes[j].startswith(('000','001','002','003','300','301','600','601','603','605','688','920','83'))],dtype=int)
                if len(eligible):
                    top = np.max(stock_returns[eligible,t])
                    winners = eligible[stock_returns[eligible,t] == top]
                    if len(winners) == 1:
                        stock = int(winners[0]); captured_stocks.add(stock)
        full = bool(np.all(board_counts[:,t] == sizes))
        retained = [stock_returns[s,t] for s in captured_stocks if np.isfinite(stock_returns[s,t])]
        traces.append(dict(cutoff_index=t,
            leader_observed_board=board_codes[board] if board is not None else None,
            leader_observed_stock=stock_codes[stock] if stock is not None else None,
            leader_observed_board_return_raw=board_returns[board,t] if board is not None else np.nan,
            leader_observed_stock_return_raw=stock_returns[stock,t] if stock is not None else np.nan,
            leader_full_universe_observed=full,
            leader_strict_board=board_codes[board] if full and board is not None else None,
            captured_board_count_today=len(captured_boards), captured_stock_count_today=len(captured_stocks),
            captured_stock_mean_return_raw=float(np.mean(retained)) if retained else np.nan,
            captured_stock_missing_today=len(captured_stocks)-len(retained)))
    return pd.DataFrame(traces)


def parquet_bytes(frame):
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), sink, compression='zstd')
    return sink.getvalue().to_pybytes()


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    from research_budget_v2 import bytes_used, TOTAL_CAP
    used, opening = 0, bytes_used()
    output_hashes = {}
    if opening+RESERVE >= TOTAL_CAP:
        raise ValueError('REGISTERED_BUDGET_STOP')
    OUT.mkdir(exist_ok=False)

    def write(relative, body, terminal=False):
        nonlocal used
        reserve = 0 if terminal else RESERVE
        if used+len(body)+reserve > CAP or bytes_used()+len(body)+reserve > TOTAL_CAP:
            raise ValueError('ACTUAL_SERIALIZED_BUDGET_STOP')
        path = OUT/relative
        path.parent.mkdir(exist_ok=True)
        with path.open('xb') as stream:
            stream.write(body)
        used += len(body)
        output_hashes[relative] = dict(bytes=len(body),sha256=digest(body))

    refs, source_files, errors, joint_parts = {}, [], [], []
    status, error = 'COMPLETE', None
    try:
        market_manifest = MARKET.parent/'manifest.json'
        for path in [MARKET,market_manifest,CATALOG,AUTH,INVENTORY,QUARANTINE,Path(__file__),
                     HERE/'test_build_joint_risk_features_v1.py']:
            refs[str(path)] = digest(path.read_bytes())
        if refs[str(MARKET)] != json.loads(market_manifest.read_bytes())['output']['sha256']:
            raise ValueError('MATERIALIZED_MARKET_INPUT_DRIFT')
        auth, catalog, inventory = [json.loads(p.read_bytes()) for p in [AUTH,CATALOG,INVENTORY]]
        if not auth['allow_archived_member_price_reconstruction']:
            raise ValueError('MEMBERSHIP_SCOPE_NOT_APPROVED')
        source = Path(catalog['source'])
        body = source.read_bytes(); refs[str(source)] = digest(body)
        if refs[str(source)] != auth['catalog_source_sha256']:
            raise ValueError('APPROVED_CATALOG_DRIFT')
        boards = sorted([b for b in pq.read_table(pa.BufferReader(body)).to_pylist()
                         if b['source']=='ths' and b['type']=='concept'],key=lambda b:b['code'])
        codes = sorted({c for b in boards for c in b['symbols']}); code_index = {c:i for i,c in enumerate(codes)}
        members = [[code_index[c] for c in b['symbols']] for b in boards]
        market = pq.read_table(MARKET).to_pandas()
        sse = market[(market['symbol']=='000001.SH') & (market['frequency_minutes']==15)].sort_values('decision_ts_assumed_end')
        cutoffs = sse['decision_ts_assumed_end'].to_numpy(dtype=np.int64)
        days = np.unique(cutoffs//1000000)
        base = pd.DataFrame(dict(cutoff_index=np.arange(len(cutoffs)),decision_ts_assumed_end=cutoffs))
        market_fields = ['rsi14','rsi14_delta1','return_adjacent_bar','momentum_change','day_return_prev_close',
                         'day_return_change_adjacent','drawdown_from_observed_day_high',
                         'daily_return5_asof_prior_day','daily_return20_asof_prior_day',
                         'daily_rsi14_asof_prior_day','anchor_low','anchor_high','anchor_position',
                         'anchor_asof_day','reference_close_day','session_fraction']
        market_fields += [c for c in market.columns if c.startswith(('golden_distance_','golden_level_','time_fraction_distance_'))]
        for symbol,prefix in [('000001.SH','sse'),('399006.SZ','cyb')]:
            for freq in [5,15]:
                panel=market[(market['symbol']==symbol)&(market['frequency_minutes']==freq)].set_index('decision_ts_assumed_end')
                panel=panel.reindex(cutoffs)[market_fields].add_prefix(f'{prefix}_{freq}m_')
                for column in panel:
                    base[column]=panel[column].to_numpy()
        quarantined=defaultdict(set)
        with QUARANTINE.open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream): quarantined[row['code']].add(int(row['day']))
        chosen={}
        for priority in ['old','car','ext','clean']:
            for item in inventory['files']:
                if item['root']==priority: chosen[item['code']]=priority
        returns=np.full((len(codes),len(cutoffs)),np.nan)
        processed=0
        for code in codes:
            if code not in chosen: continue
            label=chosen[code]; path=ROOTS[label]/f'{code}.parquet'
            body=path.read_bytes()
            source_files.append(dict(code=code,root=label,sha256=digest(body),bytes=len(body)))
            try:
                frame=pq.read_table(pa.BufferReader(body)).to_pandas()
                _, r, quality=stock_features(frame,code,days,quarantined.get(code,set()),returns_only=True)
                source_files[-1].update(quality)
                returns[code_index[code]]=r
            except Exception as exc:
                source_files[-1]['excluded']=True
                errors.append(dict(code=code,source=label,error=f'{type(exc).__name__}:{exc}'))
            processed+=1
            if processed%50==0 or processed==len(chosen):
                print(json.dumps(dict(stage='stock',processed=processed,total=len(chosen),errors=len(errors),bytes_written=used)),flush=True)
        if not sum(r.get('feature_rows',0) for r in source_files):
            raise ValueError('NO_STOCK_FEATURE_ROWS')
        board_returns=np.full((len(boards),len(cutoffs)),np.nan)
        board_counts=np.zeros((len(boards),len(cutoffs)),dtype=np.int32)
        board_coverage=np.zeros_like(board_returns)
        board_buffer=[]; board_parts=[]
        for i,board in enumerate(boards):
            fields=basket_features(returns,members[i])
            board_returns[i]=fields['observed_return_raw']; board_counts[i]=fields['present']; board_coverage[i]=fields['coverage']
            table=pd.DataFrame(fields);table.insert(0,'cutoff_index',np.arange(len(cutoffs)))
            table.insert(0,'board_code',board['code']); table['member_count']=len(members[i])
            table['partial']=table['present']!=len(members[i]);table['price_scale_verified']=False
            board_buffer.append(table)
            if (i+1)%50==0 or i+1==len(boards):
                table=pd.concat(board_buffer,ignore_index=True);name=f'board_parts/{len(board_parts):04d}.parquet'
                write(name,parquet_bytes(table));board_parts.append(dict(path=name,rows=len(table)));board_buffer=[]
        sizes=np.array([len(x) for x in members])
        leaders=observed_leaders(board_returns,board_counts,sizes,members,returns,[b['code'] for b in boards],codes)
        base=base.merge(leaders,on='cutoff_index',how='left',validate='one_to_one')
        write('shared_market_and_leaders.parquet',parquet_bytes(base))
        affiliations=defaultdict(list)
        for i,indexes in enumerate(members):
            for code_i in indexes: affiliations[codes[code_i]].append(i)
        buffer=[]
        for number, source_info in enumerate(source_files,1):
            if source_info.get('excluded'): continue
            code,label=source_info['code'],source_info['root']
            body=(ROOTS[label]/f'{code}.parquet').read_bytes()
            if digest(body)!=source_info['sha256']:
                raise ValueError('SOURCE_CHANGED_BETWEEN_CONSTRUCTION_PASSES')
            table,_,_=stock_features(pq.read_table(pa.BufferReader(body)).to_pandas(),code,days,quarantined.get(code,set()))
            table['source_cache']=label
            if not len(table): continue
            own_mean=np.full(len(table),np.nan);own_best=np.full(len(table),np.nan);own_cov=np.full(len(table),np.nan)
            for code, indexes in table.groupby('code',sort=False).groups.items():
                positions=table.loc[indexes,'cutoff_index'].to_numpy(dtype=int)
                related=affiliations[code]; values=board_returns[np.ix_(related,positions)]
                valid=np.isfinite(values); n=valid.sum(axis=0)
                own_mean[indexes]=np.divide(np.where(valid,values,0).sum(axis=0),n,out=np.full(len(positions),np.nan),where=n>0)
                maxima=np.max(np.where(valid,values,-np.inf),axis=0);maxima[~np.isfinite(maxima)]=np.nan
                own_best[indexes]=maxima;own_cov[indexes]=board_coverage[np.ix_(related,positions)].mean(axis=0)
            table['own_concepts_observed_mean_return_raw']=own_mean
            table['own_concepts_observed_best_return_raw']=own_best
            table['own_concepts_mean_coverage']=own_cov
            table=table.merge(base,on='cutoff_index',how='left',validate='many_to_one')
            table['index_volume_feature']=np.nan
            table['joint_training_admitted']=False
            buffer.append(table)
            if len(buffer)>=50 or number==len(source_files):
                table=pd.concat(buffer,ignore_index=True);name=f'joint_parts/{len(joint_parts):04d}.parquet'
                write(name,parquet_bytes(table));joint_parts.append(dict(path=name,rows=len(table)));buffer=[]
                print(json.dumps(dict(stage='join',processed=number,total=len(source_files),parts=len(joint_parts),bytes_written=used)),flush=True)
        if buffer:
            table=pd.concat(buffer,ignore_index=True);name=f'joint_parts/{len(joint_parts):04d}.parquet'
            write(name,parquet_bytes(table));joint_parts.append(dict(path=name,rows=len(table)))
        summary=dict(stock_codes_with_source=len(chosen),catalog_stocks=len(codes),concepts=len(boards),
                     decisions=len(cutoffs),start=int(days[0]),end=int(days[-1]),
                     stock_rows=sum(r.get('feature_rows',0) for r in source_files),board_rows=sum(p['rows'] for p in board_parts),
                     joint_rows=sum(p['rows'] for p in joint_parts),
                     strict_leader_cutoffs=int(leaders['leader_full_universe_observed'].sum()),
                     observed_leader_cutoffs=int(leaders['leader_observed_board'].notna().sum()))
        write('manifest.json',json.dumps(dict(status='RECONSTRUCTED_JOINT_FEATURES_NOT_TRAINING_READY',inputs=refs,
            source_files=source_files,source_priority=['clean','ext','car','old'],clock='ASSUMED_END',
            membership_bias_disclosed=True,construction='TWO_PASS_NO_DUPLICATE_STOCK_STAGING',board_parts=board_parts,joint_parts=joint_parts,
            errors=errors,summary=summary,output_hashes=output_hashes.copy(),training_ready=False,fit_run=False,http_requests=0,
            remaining=['Price/corporate-action admission','Partial-vs-strict leader research scope',
                       'Unfrozen divergence/score rules','Index volume semantics','Decision-after labels and folds']),
            ensure_ascii=False,allow_nan=False,separators=(',',':')).encode())
    except Exception as exc:
        status,error='FAILED',f'{type(exc).__name__}:{exc}'
        summary=dict(sources_recorded=len(source_files),joint_parts=len(joint_parts))
    write('stop_record.json',json.dumps(dict(status=status,error=error,bytes_before_stop=used,summary=summary,
        http_requests=0,fit_run=False),ensure_ascii=False,allow_nan=False).encode(),terminal=True)
    print(json.dumps(dict(status=status,error=error,summary=summary,bytes_written=used),ensure_ascii=False))
    return status


if __name__=='__main__':
    raise SystemExit(0 if run()=='COMPLETE' else 2)
