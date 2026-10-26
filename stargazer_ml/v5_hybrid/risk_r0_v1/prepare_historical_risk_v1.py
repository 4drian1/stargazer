"""Zero-network full-history structural inventory. Does not certify scale or create labels."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from historical_risk_core_v1 import in_scope, sample_gate, segment
from price_verification_guarded_run import encoded, sha

HERE = Path(__file__).resolve().parent
HYBRID = HERE.parent
DAILY = Path('D:/a股数据/data_daily_parts')
ROOTS = [HYBRID/'green_common_profile/gcp_round_b_small_sample_v1',
         HYBRID/'green_common_profile/gcp_round_b_bg_value_v1',
         HYBRID/'green_v2/shadow_bypass',HERE]
OUT = HERE/'historical_stock_day_preparation_v1'
COLS = ['date','open','high','low','close']


def bytes_used():
    return sum(p.stat().st_size for root in ROOTS for p in root.rglob('*') if p.is_file())


def structural_stock(chunks, calendar):
    """All duplicates audited before any roll; conflicting dates stay present but invalid."""
    rows = np.concatenate(chunks)
    rows = rows[np.argsort(rows[:,0],kind='stable')]
    dates = rows[:,0].astype(np.int64)
    unique,first,counts = np.unique(dates,return_index=True,return_counts=True)
    values = rows[first,1:]
    conflict = np.zeros(len(unique),dtype=bool)
    for j in np.flatnonzero(counts>1):
        group = rows[first[j]:first[j]+counts[j],1:]
        conflict[j] = not np.all(group==group[0])
    valid = (np.isfinite(values).all(axis=1) & (values>0).all(axis=1)
             & (values[:,2]<=np.minimum(values[:,0],values[:,3]))
             & (values[:,1]>=np.maximum(values[:,0],values[:,3])) & ~conflict)
    series = pd.Series(valid,index=unique).reindex(calendar,fill_value=False)
    # Exactly exchange-calendar slots; a missing day does not disappear from the horizon.
    past = series.astype(int).rolling(61,min_periods=61).sum().eq(61).to_numpy()
    future = series.astype(int).rolling(5,min_periods=5).sum().shift(-5).eq(5).to_numpy()
    at = np.searchsorted(calendar,unique)
    return unique,valid,past[at],future[at],int((counts-1).sum()),int(conflict.sum())


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    if bytes_used()+2*1024**2>128*1024**2:
        raise ValueError('REGISTERED_TOTAL_BUDGET')
    paths = sorted(DAILY.glob('part_*.parquet'))
    if not paths:
        raise ValueError('NO_INPUTS')
    inputs = [dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in paths]
    estimated = sum(pq.read_metadata(p).num_rows for p in paths)*5*8*3
    if estimated>2*1024**3:
        raise ValueError('PROJECTED_WORKING_MEMORY_OVER_2GIB')
    groups = defaultdict(list)
    calendar = set()
    physical=0
    excluded=Counter()
    for n,p in enumerate(paths,1):
        frame = pq.read_table(p,columns=['code']+COLS).to_pandas()
        physical += len(frame)
        date_values = frame['date'].to_numpy()
        if not np.isfinite(date_values).all() or not (date_values==date_values.astype(np.int64)).all():
            raise ValueError('INVALID_DATE')
        parsed = pd.to_datetime(frame['date'].astype(str),format='%Y%m%d',errors='raise')
        if (parsed.dt.dayofweek>=5).any():
            raise ValueError('WEEKEND_IN_CALENDAR')
        calendar.update(int(d) for d in date_values)
        for code,group in frame.groupby('code',sort=False):
            if not in_scope(str(code)):
                excluded[str(code)[:3]]+=len(group)
                continue
            groups[str(code)].append(group[COLS].to_numpy(dtype=float))
        print(f'read {n}/{len(paths)} rows={physical}',flush=True)
    cal = np.array(sorted(calendar),dtype=np.int64)
    years=defaultdict(Counter)
    segments=defaultdict(Counter)
    date_sets=defaultdict(set)
    examples=[]
    dup=conflicts=invalid=0
    for code,chunks in groups.items():
        dates,valid,past,future,nd,nc = structural_stock(chunks,cal)
        dup+=nd; conflicts+=nc; invalid+=int((~valid).sum())
        for year in sorted(set(dates//10000)):
            if year<2019 or year>2026:
                continue
            mask=(dates//10000==year)&(dates<=20260930)
            stats=years[str(year)]
            stats['stock_days']+=int(mask.sum())
            stats['valid_P']+=int((mask&valid).sum())
            stats['complete_61d_history']+=int((mask&past).sum())
            stats['complete_future_5d']+=int((mask&future).sum())
            stats['structurally_complete']+=int((mask&past&future).sum())
            if mask.any(): stats['codes']+=1
            date_sets[str(year)].update(int(d) for d in dates[mask])
        for p,ok in zip(dates,past&future):
            if not 20190101<=p<=20260930:
                continue
            i=int(np.searchsorted(cal,p))
            end=int(cal[i+5]) if i+5<len(cal) else None
            name,within=segment(int(p),end)
            segments[name]['stock_days']+=1
            segments[name]['structurally_complete']+=int(ok)
            segments[name]['purged_at_boundary_or_censored']+=int(not within)
            segments[name]['structural_after_purge']+=int(ok and within)
        if nc and len(examples)<20:
            examples.append(dict(code=code,conflict_dates=nc))
    for year,stats in years.items():
        stats['distinct_P']=len(date_sets[year])
    for p,ref in zip(paths,inputs):
        if sha(p)!=ref['sha256']:
            raise ValueError('INPUT_CHANGED_DURING_RUN')
    if paths!=sorted(DAILY.glob('part_*.parquet')):
        raise ValueError('INPUT_FILE_SET_CHANGED')
    bound=[Path(__file__),HERE/'historical_risk_core_v1.py',
           HERE/'test_historical_risk_v1.py',HERE/'contracts/HISTORICAL_STOCK_DAY_V1.md']
    manifest=dict(inputs=inputs,code_and_contract=[dict(path=str(p),sha256=sha(p)) for p in bound],
                  calendar_dates=[int(d) for d in cal],
                  calendar_source='DAILY_UNION_NOT_OFFICIAL_CERTIFICATION',
                  actual_historical_arrival_required=False,fit_authorized=False)
    report=dict(scope='HISTORICAL_STOCK_DAY_NOT_CANDIDATES',physical_rows=physical,
                in_scope_codes=len(groups),calendar_start=int(cal[0]),calendar_end=int(cal[-1]),
                years=dict(years),segments=dict(segments),duplicate_excess=dup,
                conflicting_dates=conflicts,invalid_unique_stock_days=invalid,
                conflict_examples=examples,excluded_prefix_rows=dict(excluded),
                price_scale_verified=False,corporate_action_coverage_verified=False,
                historical_membership_survivorship_verified=False,market_index_source_bound=False,
                labels_generated=0,models_fitted=0,training_ready=False,
                note='Structural counts only. Historical arrival logs and GREEN candidate identity are not gates for this version.')
    gate=sample_gate([])
    gate.update(reason='NO_ADMITTED_LABEL_FEATURE_DATASET',fit_authorized=False,
                required_evidence=['price_scale_and_adjustment','historical_security_coverage',
                                   'market_index_input','admitted_label_feature_matrix'])
    blobs={'manifest.json':encoded(manifest),'population_inventory.json':encoded(report),
           'training_gate.json':encoded(gate)}
    total=sum(map(len,blobs.values()))
    if total>2*1024**2 or bytes_used()+total>128*1024**2:
        raise ValueError('OUTPUT_BUDGET')
    OUT.mkdir(exist_ok=False)
    for name,data in blobs.items():
        with (OUT/name).open('xb') as f: f.write(data)
    print(json.dumps(dict(output=str(OUT),bytes=total,**report),ensure_ascii=False))


if __name__=='__main__':
    run()
