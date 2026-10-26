"""One authorized Tarot domain-extension fit; immutable research artifacts only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import pickle
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
sys.path[:0] = [str(PROJECT), str(HERE.parent), str(PROJECT / 'stargazer_desktop')]
from stargazer_desktop.candidate_rule_audit_20261022.replay_a_combined_v1 import new_conditions

CONTRACT = HERE.parent / 'contracts/tarot_a_second_leg_first_fit_20261022.json'
OLD = HERE.parent / 'round_a_v0_1_impl_fix'
DAILY = Path('D:/a股数据/data_daily_parts')
CACHE = HERE.parent.parent / 'opportunity_v2/gate1c/feb2026_adjustment_cache_v1/adj_cache.json'
NUMERIC = ['date', 'open', 'high', 'low', 'close', 'pre_close', 'volume',
           'amount', 'is_st', 'float_mv']
FEATURES = ['rsi14', 'atr_pct', 'ret5', 'ret10', 'ret20', 'ret60',
            'ma20_dist_atr', 'ma20_slope_atr', 'boll_width_atr', 'drawdown60',
            'volume_level60', 'volume_log_slope3', 'volume_mad3', 'down_streak',
            'anchor_volume_ratio', 'anchor_age', 'prior_rise', 'wave_retrace',
            'node_gap_atr', 'cycle_wave_pos', 'cycle618_dist_atr',
            'first_end_volume_ratio', 'end_volume_ratio', 'relative_first_end_close']
RATIOS = np.array([.236, .382, .5, .618, .786])
P_END = 20260928


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def config():
    return json.loads(CONTRACT.read_bytes())


def usage():
    roots = [Path(p) for p in config()['roots']]
    return sum(p.stat().st_size for root in roots for p in root.rglob('*') if p.is_file())


def peak_memory():
    import ctypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)] + [
            (name,ctypes.c_size_t) for name in ('peak','working','qpeak','quota','ppeak','paged','pagefile','pagefile_peak')]
    kernel=ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    handle=kernel.GetCurrentProcess()
    counters=Counters()
    counters.cb=ctypes.sizeof(counters)
    api=ctypes.windll.psapi.GetProcessMemoryInfo
    api.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    if not api(handle,ctypes.byref(counters),counters.cb):
        raise OSError('MEMORY_COUNTER_UNAVAILABLE')
    if counters.peak>config()['memory_limit_bytes']:
        raise ValueError('PEAK_MEMORY_CAP')
    return counters.peak


def save(name, body):
    cfg = config()
    used = sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
    if used + len(body) + 65536 > cfg['run_cap_bytes'] or usage() + len(body) + 65536 > cfg['total_cap_bytes']:
        raise ValueError('REGISTERED_STORAGE_CAP')
    with (HERE / name).open('xb') as f:
        f.write(body)


def dump(name, obj):
    save(name, json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8'))


def parquet(name, frame):
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, compression='zstd')
    save(name, buffer.getvalue())


def top_structure(g, anchor, p):
    """Frozen V2 30-day/5%/3-day/3% definition, all confirmations <=P."""
    h, low = g.high.to_numpy(), g.low.to_numpy()
    start = max(2, anchor - 29)
    reference = max(h[start:anchor + 1])
    peaks = []
    for i in range(start, anchor + 1):
        if i + 2 > p or h[i] < .95 * reference or h[i] < max(h[i-2:i+3]):
            continue
        if peaks:
            prev = peaks[-1]
            trough = min(low[prev+1:i]) if i > prev + 1 else h[i]
            if i - prev < 3 or trough > .97 * min(h[i], h[prev]):
                continue
        peaks.append(i)
    return peaks


def second_candidates(g, max_p=P_END):
    """Anchor-driven equivalent of V3, with authorized higher-low extension."""
    c, v = g.close.to_numpy(), g.volume.to_numpy()
    low, high = g.low.to_numpy(), g.high.to_numpy()
    down = c < g.pre_close.to_numpy()
    mean8 = g.volume.rolling(8).mean().shift(1).to_numpy()
    panic = np.flatnonzero((c < g.open.to_numpy()) & (v > 1.25 * mean8))
    signals = {}
    for t in panic:
        if t < 8 or t + 5 >= len(g):
            continue
        if not down[t+1:t+4].all() or not (v[t] > v[t+1] > v[t+2] > v[t+3]):
            continue
        pivot = next((j for j in range(t-3, max(1, t-60)-1, -1)
                      if j >= 2 and low[j] <= min(low[j-2:j+3])), None)
        if pivot is None:
            continue
        peak = max(high[pivot:t+1])
        if low[pivot] <= 0 or peak / low[pivot] - 1 <= .15:
            continue
        for p in range(t+5, min(t+8, len(g))):
            if not 20190101 <= int(g.date.iloc[p]) <= max_p or p in signals:
                continue
            rebounds = [j for j in range(t+4, p) if c[j] > g.pre_close.iloc[j]]
            if not rebounds or not down[rebounds[-1]+1:p+1].all():
                continue
            longest = current = 0
            for flag in down[t:p+1]:
                current = current + 1 if flag else 0
                longest = max(longest, current)
            peaks = top_structure(g, t, p)
            signals[p] = dict(stage='SECOND', anchor=t, pivot=pivot, peak=float(peak),
                              prior_rise=float(peak/low[pivot]-1),
                              top_count=len(peaks), longest_down=longest,
                              old_v3_price_gate=bool(c[p] < c[t+3]),
                              first_end=t+3)
    return signals


def indicators(g):
    c, h, low, v, pc = (g[k].astype(float) for k in ('close', 'high', 'low', 'volume', 'pre_close'))
    tr = pd.concat([h-low, (h-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().rolling(5).median()
    ma = c.rolling(20).mean()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    down = c < pc
    streak = down.groupby((~down).cumsum()).cumsum()
    windows = np.lib.stride_tricks.sliding_window_view(v.to_numpy(), 3)
    med = np.median(windows, axis=1)
    mad = np.r_[np.nan, np.nan, np.median(abs(windows-med[:, None]), axis=1)/(med+1e-12)]
    out = pd.DataFrame({'rsi14': 100*up/(up+dn), 'atr_pct': atr/c,
        'ma20_dist_atr': (c-ma)/atr, 'ma20_slope_atr': ma.diff(5)/atr,
        'boll_width_atr': 4*c.rolling(20).std()/atr,
        'drawdown60': c/c.rolling(60).max()-1,
        'volume_level60': v.rolling(3).median()/v.shift(3).rolling(60).median(),
        'volume_log_slope3': np.log(v.where(v>0)).diff(2)/2,
        'volume_mad3': mad, 'down_streak': streak, 'atr': atr})
    for days in (5, 10, 20, 60):
        out[f'ret{days}'] = c.pct_change(days, fill_method=None)
    return out


def cycle_history(g):
    """Same recurrence as desktop golden_rules.cycle_anchors, one prefix pass."""
    c, o, high, low = (g[k].to_numpy() for k in ('close', 'open', 'high', 'low'))
    hi = lo = None
    out = np.full((len(g), 2), np.nan)
    for i in range(len(g)):
        if i == 6:
            hi, lo = max(high[:7]), min(low[:7])
        if hi is None:
            continue
        if i >= 3 and c[i-3] > o[i-3] and c[i-3] > c[i-2] > c[i-1] > c[i]:
            if any((c[j]-c[j+1])/c[j] >= .01-1e-12 for j in range(i-3, i)):
                hi = min(hi, c[i-3])
        if i >= 7 and all(c[j+1] > c[j] for j in range(i-7, i)):
            lo = max(lo, c[i-7])
        hi, lo = max(hi, c[i]), min(lo, c[i])
        out[i] = hi, lo
    return out


def candidates(g, old_a):
    feats = indicators(g)
    signals = second_candidates(g)
    idx = {int(d): i for i, d in enumerate(g.date)}
    for r in old_a:
        p, t = idx.get(int(r['date'])), idx.get(int(r['div_date']))
        if p is None or t is None or not 20190101 <= int(r['date']) <= P_END:
            continue
        new = new_conditions(g.iloc[:p+1], t, float(feats.atr.iloc[p]))
        if not new['new_gate']:
            continue
        wave = new['wave']
        signals[p] = dict(stage='A', anchor=t, pivot=idx[wave['pivot_date']],
                          peak=wave['peak_high'], prior_rise=wave['rise'],
                          top_count=len(top_structure(g, t, p)), longest_down=None,
                          old_v3_price_gate=None, first_end=min(t+3, p),
                          node=wave['observation_node'])
    if not signals:
        return [], {}
    cyc = cycle_history(g)
    rows, counts = [], Counter()
    for p, s in sorted(signals.items()):
        row = g.iloc[p]
        counts['shape_'+s['stage']] += 1
        if not (p >= 69 and row.is_st == 0 and row.float_mv > 0 and row.amount >= 5e7 and row.volume > 0):
            counts['outer_excluded'] += 1
            continue
        if s['top_count'] >= 3 or s['top_count'] == 0:
            counts['triple_excluded' if s['top_count'] >= 3 else 'top_unknown'] += 1
            continue
        if s['stage'] == 'SECOND' and s['longest_down'] >= 6:
            counts['six_down_excluded'] += 1
            continue
        hi, lo = cyc[p]
        levels = hi - RATIOS*(hi-lo)
        below = levels[levels <= row.close+1e-10]
        node = s.get('node') if s['stage'] == 'A' else (float(max(below)) if len(below) and hi>lo else None)
        t, first_end = s['anchor'], s['first_end']
        atr = float(feats.atr.iloc[p])
        rec = {k: float(feats[k].iloc[p]) for k in FEATURES if k in feats}
        rec.update(code=str(g.code.iloc[0]), P=int(row.date), stage=s['stage'],
                   anchor_date=int(g.date.iloc[t]), top_count=s['top_count'],
                   old_v3_price_gate=s['old_v3_price_gate'], node=node,
                   node_source='WAVE' if s['stage']=='A' else 'CYCLE', close=float(row.close),
                   position_ready=bool(node is not None and node>0 and node<=row.close and atr>0),
                   anchor_volume_ratio=float(g.volume.iloc[t]/g.volume.iloc[t-8:t].mean()),
                   anchor_age=p-t, prior_rise=s['prior_rise'],
                   wave_retrace=float((s['peak']-row.close)/(s['peak']-g.low.iloc[s['pivot']])),
                   node_gap_atr=(float((row.close-node)/atr) if node is not None else np.nan),
                   cycle_wave_pos=float((row.close-lo)/(hi-lo)) if hi>lo else np.nan,
                   cycle618_dist_atr=float((row.close-(hi-.618*(hi-lo)))/atr),
                   first_end_volume_ratio=float(g.volume.iloc[first_end]/g.volume.iloc[t]),
                   end_volume_ratio=float(row.volume/g.volume.iloc[t]),
                   relative_first_end_close=float(row.close/g.close.iloc[first_end]-1))
        rows.append(rec)
    return rows, counts


def label_row(rec, g, calendar, cache):
    p = rec['P']
    dates = g.date.to_numpy(dtype=int)
    pos = int(np.searchsorted(dates, p))
    ci = int(np.searchsorted(calendar, p))
    expected = calendar[ci:ci+8]
    reasons = []
    if not rec['position_ready']:
        reasons.append('NO_VALID_DOWNWARD_NODE')
    if len(expected) < 8 or expected[-1] > P_END or not np.array_equal(dates[pos:pos+8], expected):
        reasons.append('LABEL_WINDOW_INCOMPLETE')
    if ci < 69 or pos < 69 or not np.array_equal(dates[pos-69:pos+1], calendar[ci-69:ci+1]):
        reasons.append('FEATURE_CALENDAR_GAP')
    source = cache.get(rec['code'])
    rec['adj_cache_status'] = 'MISSING' if source is None else ('EVENTS' if source.get('events') else 'EMPTY')
    if source is None:
        reasons.append('CACHE_MISSING')
    if len(expected) == 8:
        rec['label_end'] = int(expected[-1])
        if source:
            if any(p < int(str(day).replace('-', '')[:8]) <= expected[-1] for day, _ in source.get('events', [])):
                reasons.append('CORP_ACTION_IN_WINDOW')
    if 'LABEL_WINDOW_INCOMPLETE' not in reasons:
        rec['chg7'] = float(g.close.iloc[pos+7]/g.close.iloc[pos]-1)
        rec['mae7'] = float(min(g.low.iloc[pos+1:pos+8])/g.close.iloc[pos]-1)
        rec['y7'] = int(rec['chg7']>0)
    rec['admission'] = 'OK' if not reasons else 'ISOLATED'
    rec['reasons'] = '|'.join(reasons)
    return rec


def prepare():
    if (HERE/'prepare_started.json').exists():
        raise ValueError('PREPARE_ALREADY_STARTED_NO_AUTOMATIC_RESTART')
    dump('prepare_started.json', {'contract_sha256': sha(CONTRACT), 'script_sha256': sha(__file__),
                                 'usage_before': usage(), 'status': 'STARTED'})
    groups = defaultdict(list)
    paths = sorted(DAILY.glob('part_*.parquet'))
    manifest = {str(p): {'sha256': sha(p), 'size': p.stat().st_size} for p in paths}
    dump('source_manifest.json', {'daily': manifest, 'contract': sha(CONTRACT),
         'old_A': sha(OLD/'pool_a_sdp_members.parquet'), 'corporate_cache': sha(CACHE),
         'legacy_model': sha(OLD/'fold_model_FINAL_B2.pkl'),
         'source_note': 'retrospective qfq snapshot; actual historical arrival unknown; adjustment cache not exhaustive'})
    memory = 0
    for i, path in enumerate(paths):
        frame = pq.read_table(path, columns=['code']+NUMERIC).to_pandas()
        frame = frame[(frame.date<=20261021) & frame.code.str.match(r'^(60|68|00|30)\d{4}$')]
        for code, sub in frame.groupby('code', sort=False):
            values = sub[NUMERIC].to_numpy(dtype=float)
            groups[code].append(values)
            memory += values.nbytes
        if memory > 1400*1024**2:
            raise ValueError('MEMORY_RESERVATION_EXCEEDED')
        peak_memory()
        print('read',i+1,'/',len(paths),'array_MiB',round(memory/1024**2),flush=True)
    del frame
    calendar = np.unique(np.concatenate([a[:,0].astype(np.int64) for parts in groups.values() for a in parts]))
    peak_memory()
    old = pd.read_parquet(OLD/'pool_a_sdp_members.parquet')
    bycode = {str(c): x.to_dict('records') for c,x in old.groupby('code')}
    cache = json.loads(CACHE.read_bytes())
    rows, audit, counts = [], [], Counter()
    for n, code in enumerate(list(groups)):
        values = np.concatenate(groups.pop(code))
        values = values[np.argsort(values[:,0], kind='stable')]
        dates, first, sizes = np.unique(values[:,0], return_index=True, return_counts=True)
        conflict = any(not np.array_equal(values[j:j+k], np.tile(values[j], (k,1)), equal_nan=True)
                       for j,k in zip(first[sizes>1], sizes[sizes>1]))
        if conflict:
            audit.append({'code':code,'reason':'DAILY_DUP_CONFLICT'})
            continue
        g = pd.DataFrame(values[first], columns=NUMERIC)
        if len(g)<70:
            continue
        g['date'] = g.date.astype(int)
        g['code'] = code
        valid = (np.isfinite(g[['open','high','low','close','pre_close','volume']]).all(axis=1)
                 & (g[['open','high','low','close','pre_close']]>0).all(axis=1)
                 & (g.volume>=0) & (g.low<=g[['open','close']].min(axis=1))
                 & (g.high>=g[['open','close']].max(axis=1)))
        if not valid.all():
            audit.append({'code':code,'reason':'INVALID_DAILY_VALUES','count':int((~valid).sum())})
            continue
        records, disposition = candidates(g, bycode.get(code, []))
        counts.update(disposition)
        for rec in records:
            rows.append(label_row(rec, g, calendar, cache))
        # Real prefix and future-perturbation audit on deterministic candidate stocks.
        if records and n%97==0:
            cutoff = records[len(records)//2]['P']
            subset = g[g.date<=cutoff].copy()
            prefix, _ = candidates(subset, bycode.get(code, []))
            expected_rows = [r for r in records if r['P']<=cutoff]
            cols = ['code','P','stage']+FEATURES+['node']
            def comparison(rs):
                return pd.DataFrame(rs).reindex(columns=cols).sort_values('P').reset_index(drop=True)
            pd.testing.assert_frame_equal(comparison(prefix), comparison(expected_rows), check_dtype=False, atol=1e-12, rtol=1e-12)
            counts['real_prefix_audits'] += 1
        if n%200==0:
            print('stocks',n,'candidates',len(rows),'isolated_codes',len(audit),flush=True)
    frame = pd.DataFrame(rows)
    if frame.duplicated(['code','P']).any():
        raise ValueError('DUPLICATE_CANDIDATE_KEYS')
    for feature in FEATURES:
        frame[feature] = frame[feature].replace([np.inf,-np.inf], np.nan)
    if any(sha(p)!=info['sha256'] for p,info in manifest.items()):
        raise ValueError('DAILY_SOURCE_CHANGED_DURING_READ')
    parquet('population.parquet', frame)
    dump('preparation.json', {'counts':dict(counts),'rows':len(frame),
        'by_stage':frame.stage.value_counts().to_dict(),
        'by_stage_year':frame.assign(year=frame.P//10000).groupby(['year','stage']).size().rename('n').reset_index().to_dict('records'),
        'admission':frame.admission.value_counts().to_dict(),
        'isolation':frame.reasons.value_counts().to_dict(), 'rejected_codes':audit,
        'calendar_min':int(calendar.min()),'calendar_max':int(calendar.max()),
        'population_sha256':sha(HERE/'population.parquet'), 'usage':usage(),
        'peak_working_set_bytes':peak_memory()})
    print('PREPARE_COMPLETE',len(frame),frame.admission.value_counts().to_dict(),flush=True)


def split(frame):
    year = frame.P//10000
    end = frame.label_end
    return {'train':(year>=2019)&(year<=2023)&(end<20240101),
            'validation_2024':(year==2024)&(end<20250101),
            'test_2025':(year==2025)&(end<20260101),
            'research_2026':(year==2026)&(end<=P_END)}


def metrics(y, p):
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    if len(y)==0:
        return {'n':0}
    p = np.clip(p,1e-6,1-1e-6)
    return {'n':len(y),'positive_rate':float(np.mean(y)),
            'brier':float(brier_score_loss(y,p)),
            'logloss':float(log_loss(y,p,labels=[0,1])),
            'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None}


def moving_block_ci(values):
    x = np.array(values,dtype=float)
    if len(x)<20:
        return {'n_dates':len(x),'mean':float(x.mean()) if len(x) else None,'lo':None,'hi':None}
    rng = np.random.default_rng(20261022)
    block = min(20,len(x))
    estimates = []
    for _ in range(1000):
        starts = rng.integers(0,len(x)-block+1,size=int(np.ceil(len(x)/block)))
        draw = np.concatenate([x[s:s+block] for s in starts])[:len(x)]
        estimates.append(draw.mean())
    return {'n_dates':len(x),'mean':float(x.mean()),'lo':float(np.quantile(estimates,.025)),
            'hi':float(np.quantile(estimates,.975)), 'block_dates':block}


def fit():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning
    if (HERE/'fit_started.json').exists():
        raise ValueError('FIT_ALREADY_STARTED_NO_AUTOMATIC_RETRAIN')
    prep = json.loads((HERE/'preparation.json').read_bytes())
    started = json.loads((HERE/'prepare_started.json').read_bytes())
    if started['contract_sha256']!=sha(CONTRACT) or started['script_sha256']!=sha(__file__):
        raise ValueError('PREPARED_CONTRACT_OR_CODE_CHANGED')
    if sha(HERE/'population.parquet') != prep['population_sha256']:
        raise ValueError('POPULATION_CHANGED')
    frame = pd.read_parquet(HERE/'population.parquet')
    frame = frame[frame.admission.eq('OK')].sort_values(['P','code']).copy()
    masks = split(frame)
    train = frame[masks['train']]
    if len(train)<200 or any(len(train[train.stage==s])<50 or train[train.stage==s].y7.nunique()<2 for s in ('A','SECOND')):
        raise ValueError('INSUFFICIENT_TRAINING_POPULATION')
    med = train[FEATURES].median()
    scaler = StandardScaler().fit(train[FEATURES].fillna(med).fillna(0))
    clf = LogisticRegression(C=.1,max_iter=2000,random_state=20261022)
    dump('fit_started.json', {'contract':sha(CONTRACT),'script':sha(__file__),
        'population':prep['population_sha256'],'training_rows':len(train),
        'training_key_hash':hashlib.sha256(pd.util.hash_pandas_object(train[['code','P','label_end']],index=False).values.tobytes()).hexdigest(),
        'status':'STARTED_ONE_FIT'})
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        clf.fit(scaler.transform(train[FEATURES].fillna(med).fillna(0)),train.y7.astype(int))
    model = {'clf':clf,'scaler':scaler,'median':med,'features':FEATURES,'contract':sha(CONTRACT),
             'scope':'A_WAVE_AND_SECOND_HIGHER_LOW_RESEARCH','train_cutoff':20231231}
    save('model.pkl',pickle.dumps(model))
    reloaded = pickle.loads((HERE/'model.pkl').read_bytes())
    x = scaler.transform(frame[FEATURES].fillna(med).fillna(0))
    scores = clf.predict_proba(x)[:,1]
    reproduced = reloaded['clf'].predict_proba(reloaded['scaler'].transform(
        frame[FEATURES].fillna(reloaded['median']).fillna(0)))[:,1]
    if not np.array_equal(scores,reproduced):
        raise ValueError('SAVED_MODEL_SCORE_MISMATCH')
    frame['score'] = scores
    frame['split'] = 'PURGED_BOUNDARY'
    for name,mask in masks.items():
        frame.loc[mask,'split'] = name
    prevalence = train.groupby('stage').y7.mean().to_dict()
    frame['reference_score'] = frame.stage.map(prevalence)
    frame['rank'] = frame.groupby('P').score.rank(method='first',ascending=False).astype(int)
    frame['top5'] = frame['rank']<=5
    parquet('predictions.parquet',frame)
    results, date_tables = {}, []
    for name in masks:
        part = frame[frame.split==name]
        details = {}
        for stage in ('ALL','A','SECOND'):
            sub = part if stage=='ALL' else part[part.stage==stage]
            met = metrics(sub.y7,sub.score)
            base = metrics(sub.y7,sub.reference_score)
            bins = sub.assign(bin=np.minimum((sub.score*10).astype(int),9)).groupby('bin').agg(
                n=('y7','size'),predicted=('score','mean'),observed=('y7','mean')).reset_index()
            details[stage] = {'model':met,'stage_constant':base,'days':int(sub.P.nunique()),
                 'events':int(sub[['code','anchor_date','stage']].drop_duplicates().shape[0]),
                 'calibration_bins':bins.to_dict('records'),
                 'research_group_pass':bool(len(sub)>=50 and sub.P.nunique()>=30 and
                     met.get('auc') is not None and met['auc']>.5 and met['brier']<base['brier'])}
        dates = []
        for p,g in part.groupby('P',sort=True):
            top = g.sort_values(['score','code'],ascending=[False,True]).head(5)
            dates.append({'P':int(p),'split':name,'candidates':len(g),'selected':len(top),
                'top_return':float(top.chg7.mean()),'candidate_return':float(g.chg7.mean()),
                'uplift':float(top.chg7.mean()-g.chg7.mean()),
                'mae_delta':float(top.mae7.mean()-g.mae7.mean()),
                'second_share':float(top.stage.eq('SECOND').mean())})
        dt = pd.DataFrame(dates)
        date_tables.extend(dates)
        selective = dt[dt.candidates>5] if len(dt) else dt
        uplift = moving_block_ci(selective.uplift if len(selective) else [])
        mae = float(selective.mae_delta.mean()) if len(selective) else None
        results[name] = {'groups':details,'eligible_dates':len(dt),'selective_dates':len(selective),
                        'top5_uplift':uplift,'mae_delta':mae,
                        'mean_selected_second_share':float(dt.second_share.mean()) if len(dt) else None,
                        'research_rank_pass':bool(uplift['lo'] is not None and uplift['lo']>0 and mae>=0)}
    parquet('daily_ranking_metrics.parquet',pd.DataFrame(date_tables))
    coefficients = {name:float(value) for name,value in zip(FEATURES,clf.coef_[0])}
    report = {'status':'FIRST_FIT_COMPLETE_RESEARCH_NOT_DEPLOYED', 'fit_count':1,
        'model_sha256':sha(HERE/'model.pkl'),'model_readback_exact':True,
        'n_iter':clf.n_iter_.tolist(),'features':FEATURES,'training_all_missing':med[med.isna()].index.tolist(),
        'training_rows':len(train),'train_stage_prevalence':prevalence,'results':results,
        'purged_rows':int(frame.split.eq('PURGED_BOUNDARY').sum()),
        'standardized_coefficients':coefficients,'intercept':clf.intercept_.tolist(),
        'production_ready':False,'reason':'historical qfq/cache limits; no prospective or intraday joint Pyxis acceptance',
        'usage_bytes':usage()}
    dump('results.json',report)
    lines = ['# Tarot 一段/二段首轮扩域训练结果','',
        '状态：实际完成一次固定 LR 拟合；研究结果，未部署。二段允许高于首段末收的回踩；B池不参加训练，二段无人工加分。',
        '目标为P收至第七交易日收盘上涨标签，非触位买入后的真实收益。训练2019–2023；2024仅验证；2025测试；2026截至9/28事后研究。跨界标签已剔除，已看历史不称全新盲测。','',
        '|时段|分支|数量|AUC|Brier|分支常数Brier|分组门|','|---|---|---:|---:|---:|---:|---|']
    for name,res in results.items():
        for stage,d in res['groups'].items():
            m,b=d['model'],d['stage_constant']
            lines.append(f"|{name}|{stage}|{m['n']}|{m.get('auc')}|{m.get('brier')}|{b.get('brier')}|{d['research_group_pass']}|")
        lines += ['',f"{name}：真正发生前五筛选的日期 {res['selective_dates']}；七日收益相对同日候选均值增量及20日期块95%区间：{res['top5_uplift']}；下探改善差 {res['mae_delta']}；排序门 {res['research_rank_pass']}。",'']
    lines += ['源为本机既有复权日线；公司行为缓存并非穷举，缺缓存/标签窗内事件/日历缺口隔离。源码及全部源哈希见source_manifest.json；人口含隔离行。',
              'model.pkl读回后全部评分逐值相等；训练期中位数/标准化仅从训练期拟合。predictions.parquet与daily_ranking_metrics.parquet可复核。',
              '尚未完成盘中黄金节点触发、成交成本、同截止Pyxis联合验证和前瞻验证，不能把本轮排名作为正式买入信号。未按结果续跑或调参。']
    save('REPORT.md','\n'.join(lines).encode('utf-8'))
    dump('terminal.json',{'status':report['status'],'fit_count':1,'usage_bytes':usage(),
         'files':{p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ('.parquet','.pkl','.json','.md','.py')}})
    brief = {'status':report['status'],'training_rows':len(train),
             'evaluation':{k:{'rank_pass':v['research_rank_pass'],
                 'groups':{s:d['model'] for s,d in v['groups'].items()}}
                 for k,v in results.items()}}
    print(json.dumps(brief,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','fit'])
    args=parser.parse_args()
    try:
        (prepare if args.action=='prepare' else fit)()
    except Exception as exc:
        failure=HERE/f'{args.action}_failure.json'
        if not failure.exists():
            dump(failure.name,{'status':'FAILED_RETAINED','error':repr(exc),'action':args.action})
        raise
