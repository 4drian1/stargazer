"""Read frozen code/date columns only. Mechanical sample and evidence inventory, no fetch/labels/fit."""
from collections import Counter, defaultdict
from hashlib import sha256
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from build_archived_corpus_v3 import frozen_pages
from historical_risk_core_v1 import in_scope
from price_verification_guarded_run import encoded, sha
from research_budget_v2 import bytes_used, TOTAL_CAP
from risk_fit_eval_v1 import board

HERE=Path(__file__).resolve().parent
OUT=HERE/'risk_population_expansion_plan_v1_20261015'
TARGET=400
SALT='R0_EXPANSION_V1|'
CAP=2*1024**2
BOARDS=('GEM','SH_MAIN','STAR','SZ_MAIN')


def require(value,message):
    if not value:
        raise ValueError(message)


def quotas(counts,target):
    require(type(target) is int and 0<target<=sum(counts.values()),'INVALID_TARGET')
    require(set(counts)<=set(BOARDS) and all(type(v) is int and v>=0 for v in counts.values()),'INVALID_STRATA')
    total=sum(counts.values())
    result={b:target*counts.get(b,0)//total for b in BOARDS}
    remaining=target-sum(result.values())
    order=sorted(BOARDS,key=lambda b:(-(target*counts.get(b,0)%total),b))
    for b in order[:remaining]:
        result[b]+=1
    require(all(result[b]<=counts.get(b,0) for b in BOARDS),'QUOTA_EXCEEDS_STRATUM')
    return result


def mechanical_selection(universe,target=TARGET):
    codes=sorted(universe)
    require(len(codes)==len(set(codes)) and all(in_scope(c) for c in codes),'INVALID_UNIVERSE')
    counts=Counter(board(c) for c in codes)
    allocation=quotas(counts,target)
    selected=[]
    for b in BOARDS:
        members=sorted((c for c in codes if board(c)==b),key=lambda c:(sha256((SALT+c).encode()).hexdigest(),c))
        selected.extend(members[:allocation[b]])
    return sorted(selected),dict(counts),allocation


def inventory_dates(paths,calendar):
    stats={}
    physical=0
    excluded=Counter()
    for number,path in enumerate(paths,1):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536,columns=['code','date']):
            frame=batch.to_pandas()
            physical+=len(frame)
            dates=frame['date'].to_numpy()
            require(np.isfinite(dates).all() and np.equal(dates,dates.astype(np.int64)).all()
                    and np.isin(dates,calendar).all(),'DATE_NOT_IN_FROZEN_CALENDAR')
            allowed=frame['code'].map(in_scope)
            excluded.update(frame.loc[~allowed,'code'].astype(str).str[:3].value_counts().to_dict())
            frame=frame.loc[allowed].copy()
            frame['year']=frame['date']//10000
            for (code,year),group in frame.groupby(['code','year'],sort=False):
                item=stats.setdefault(code,dict(code=code,board=board(code),year_rows={},first_visible_date=None,
                    last_visible_date=None,first_visible_date_through_2025=None,last_visible_date_through_2025=None))
                lo,hi=int(group['date'].min()),int(group['date'].max())
                item['first_visible_date']=min(item['first_visible_date'] or lo,lo)
                item['last_visible_date']=max(item['last_visible_date'] or hi,hi)
                key=str(int(year))
                item['year_rows'][key]=item['year_rows'].get(key,0)+len(group)
                if int(year)<=2025:
                    item['first_visible_date_through_2025']=min(item['first_visible_date_through_2025'] or lo,lo)
                    item['last_visible_date_through_2025']=max(item['last_visible_date_through_2025'] or hi,hi)
        print(f'code/date inventory {number}/{len(paths)} rows={physical}',flush=True)
    return stats,physical,dict(excluded)


def study_universe(stats):
    return sorted(c for c,r in stats.items() if sum(v for y,v in r['year_rows'].items() if 2019<=int(y)<=2025)>0)


def native_pair(old,code):
    return [p if p.is_file() else HERE/'historical_archived_corpus_v3_20261015'/p.name
            for p in [old/(code+'_'+suffix+'.json.gz') for suffix in ('daily','actions')]]


def snapshot_file_set(frozen,current):
    require(set(frozen)<=set(current),'FROZEN_DAILY_FILE_MISSING')
    return sorted(set(current)-set(frozen))


def run():
    require(not OUT.exists(),'PLAN_EXISTS_NO_OVERWRITE')
    path=HERE/'historical_stock_day_preparation_v1/manifest.json'
    frozen=json.loads(path.read_bytes())
    refs=frozen['inputs']
    paths=[Path(r['path']) for r in refs]
    require(all(sha(r['path'])==r['sha256'] for r in refs),'FROZEN_DAILY_DRIFT')
    ignored=snapshot_file_set(paths,paths[0].parent.glob('part_*.parquet'))
    calendar=np.array(frozen['calendar_dates'],dtype=np.int64)
    stats,physical,excluded=inventory_dates(paths,calendar)
    universe=study_universe(stats)
    selected,counts,allocation=mechanical_selection(universe)
    pages=frozen_pages()
    old=HERE/'historical_archived_corpus_v4_20261015'
    old_codes=json.loads((old/'plan.json').read_bytes())['codes']
    require(set(old_codes)==set(pages),'OLD_SOURCE_POPULATION_MISMATCH')
    selected_refs=[]
    for code in selected:
        native=native_pair(old,code)
        reused=all(p.is_file() for p in native)
        ext=pages.get(code)
        selected_refs.append(dict(stats[code],selection_hash=sha256((SALT+code).encode()).hexdigest(),
            historical_listing_date_verified=False,old_43_reference_member=code in old_codes,
            archived_external_page_available=ext is not None,native_raw_pair_available=reused,
            external_refs=[dict(path=str(p),sha256=sha(p)) for p in ext] if ext else [],
            native_refs=[dict(path=str(p),sha256=sha(p)) for p in native] if reused else [],
            source_admission='NOT_ASSESSED_NO_NEW_LABELS',training_eligible=False))
    overlap=sorted(set(selected)&set(old_codes))
    missing_ext=[r['code'] for r in selected_refs if not r['archived_external_page_available']]
    missing_native=[r['code'] for r in selected_refs if not r['native_raw_pair_available']]
    max_p_dates=sum(20190101<=int(d)<=20260930 for d in calendar)
    acceptance=json.loads((HERE/'archived_corpus_v4_acceptance_20261015.json').read_bytes())
    matrix_bytes=sum(Path(r['path']).stat().st_size for r in acceptance['matrix_files'])
    old_raw={p for code in old_codes for p in native_pair(old,code) if p.is_file()}
    old_bytes=sum(p.stat().st_size for p in old_raw)
    matrix_upper=TARGET*max_p_dates
    compressed_estimate=int(matrix_upper*matrix_bytes/acceptance['persisted_rows'])
    raw_compressed_estimate=int(TARGET*old_bytes/len(old_codes))
    estimates=dict(method='Observed old corpus bytes per row/code extrapolation, NOT a guarantee.',
        full_calendar_matrix_rows_upper=matrix_upper,old_matrix_bytes=matrix_bytes,
        matrix_compressed_point_estimate=compressed_estimate,
        local_raw_gzip_point_estimate=raw_compressed_estimate,
        combined_point_estimate=compressed_estimate+raw_compressed_estimate,
        combined_twice_point_estimate=2*(compressed_estimate+raw_compressed_estimate))
    budgets=dict(status='PROPOSED_NOT_AUTHORIZED',external_requests=len(missing_ext),
        smoke_requests=2 if len(missing_ext)>=2 else len(missing_ext),smoke_included_not_additional=True,
        external_response_cap=64*1024**2,external_disk_cap=96*1024**2,
        local_requests=2*len(missing_native),local_response_cap=1024**3,local_disk_and_matrix_cap=512*1024**2,
        proposed_total_new_disk_cap=608*1024**2,future_fit_authorized=False,
        cumulative_cap=TOTAL_CAP,registered_actual_bytes=bytes_used(),no_retries=True,
        failure_stops_whole_fixed_plan=True,source_failure_never_replaces_selected_code=True)
    summary=dict(status='MECHANICAL_POPULATION_PLAN_DRAFT',selection_uses_labels=False,
        selection_uses_source_evidence_availability=False,selection_cutoff=20251231,
        source_files=len(paths),source_physical_rows=physical,visible_in_scope_codes=len(stats),
        ignored_new_files=[str(p) for p in ignored],input_scope='ORIGINAL_FROZEN_FILE_LIST_NOT_LIVE_DIRECTORY',
        eligible_visible_codes_2019_2025=len(universe),universe_board_counts=counts,selected_board_counts=allocation,
        selected_codes=len(selected),old_43_overlap=overlap,old_43_forced_inclusion=False,
        new_external_pages_needed=len(missing_ext),new_local_pairs_needed=len(missing_native),
        cached_evidence_sufficiency_not_certified=True,request_and_storage_proposal=budgets,estimates=estimates,
        historical_population_survivorship_verified=False,network_requests_performed=0,new_labels_generated=0,
        models_fitted=0,fit_authorized=False,fetch_authorized=False,
        limitation='Snapshot-visible historical population, not complete delisted universe or new blind research.')
    inputs={Path(r['path']) for r in refs}
    inputs.update((path,Path(__file__),HERE/'test_plan_risk_population_expansion_v1.py',old/'plan.json',
        HERE/'archived_corpus_v4_acceptance_20261015.json',HERE/'build_archived_corpus_v3.py',
        HERE/'historical_risk_core_v1.py',HERE/'risk_fit_eval_v1.py',HERE/'research_budget_v2.py',
        HERE/'contracts/research_storage_4gib_20261015.json'))
    inputs.update(p for pair in pages.values() for p in pair)
    inputs.update(old_raw)
    manifest_refs=[dict(path=str(p),sha256=sha(p)) for p in sorted(inputs)]
    contents={'summary.json':encoded(summary),'selected_codes.json':encoded(selected_refs),
        'visible_universe.json.gz':gzip.compress(encoded(dict(universe_codes=universe,by_code=stats,
            excluded_prefix_rows=excluded,note='Only code/date occurrence counts; no OHLC validity, dedup, scale or training admission certification.')),mtime=0),
        'input_manifest.json':encoded(dict(inputs=manifest_refs,selection_salt=SALT,target=TARGET))}
    terminal=encoded(dict(status='completed',network_requests=0,labels_generated=0,fit_performed=False,
        files=[dict(name=n,bytes=len(v),sha256=sha256(v).hexdigest()) for n,v in contents.items()]))
    total=sum(map(len,contents.values()))+len(terminal)
    require(total<=CAP and bytes_used()+total<=TOTAL_CAP,'PLAN_OUTPUT_BUDGET')
    require(all(sha(r['path'])==r['sha256'] for r in manifest_refs),'INPUT_CHANGED_DURING_PLAN')
    snapshot_file_set(paths,paths[0].parent.glob('part_*.parquet'))
    OUT.mkdir(exist_ok=False)
    for name,data in contents.items():
        with (OUT/name).open('xb') as f:
            f.write(data)
    with (OUT/'stop_record.json').open('xb') as f:
        f.write(terminal)
    print(json.dumps(dict(output=str(OUT),total_bytes=total,**summary),ensure_ascii=False))


if __name__=='__main__':
    run()
