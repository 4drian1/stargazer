"""Local member dependency inventory; file presence is NOT price/source admission."""
from collections import Counter
import json
from pathlib import Path

import pyarrow.parquet as pq

from price_verification_guarded_run import sha
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = Path(__file__).resolve().parent
HYBRID = HERE.parent
CACHE = HYBRID.parent / 'pullback_train/minute_cache_clean'
OUT = HERE / 'synchronized_context_inventory_20261016_v1'
CAP = 2 * 1024**2


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if bytes_used() + CAP > TOTAL_CAP:
        raise ValueError('REGISTERED_BUDGET')
    refs = {}

    def track(path):
        path = Path(path).resolve()
        refs[str(path)] = sha(path)
        return path

    track(Path(__file__))
    track(HERE / 'research_budget_v2.py')
    members = pq.read_table(track(HYBRID / 'opportunity_v2/o5_sector_lag/pit_membership.parquet')).to_pylist()
    original = json.loads(track(HERE / 'intraday_cache_probe_20261016_v1/diagnostics.json').read_bytes())
    clean_probe = json.loads(track(HERE / 'minute_warmup_probe_20261016_v1/diagnostics.json').read_bytes())
    cases, required = [], set()
    for case in original['results']:
        code, p, e = case['code'], case['P'], case['E']
        hits = [r for r in members if r['symbol'] == code and r['effective_from'] <= p < r['effective_to']]
        if len(hits) != 1 or not hits[0]['index_code_l1']:
            raise ValueError('CANDIDATE_MEMBERSHIP_NOT_UNIQUE')
        identity = hits[0]
        active = [r for r in members if r['index_code_l1'] == identity['index_code_l1']
                  and r['effective_from'] <= p < r['effective_to'] and r['symbol'] != code]
        counts = Counter(r['symbol'] for r in active)
        codes = sorted(counts)
        required.update(codes)
        cases.append(dict(code=code, P=p, E=e, membership_record=identity,
            membership_known_at=None, known_at_verdict='UNVERIFIED',
            update_time_after_P=identity['update_time'].replace('-', '')[:8] > str(p),
            sector_identity='SW_L1_NOT_THEME', classification_only_not_feature=True,
            required_member_codes=codes, required_count=len(codes),
            ambiguous_member_codes=sorted(c for c, n in counts.items() if n != 1),
            clean_files_present=sum((CACHE / f'{c}.parquet').is_file() for c in codes),
            target_date_price_coverage='NOT_TESTED', leader_selection_rule=None,
            sector_features=None, leader_features=None, training_eligible=False))
    files = []
    for code in sorted(required):
        path = CACHE / f'{code}.parquet'
        stat = path.stat() if path.is_file() else None
        files.append(dict(code=code, path=str(path), exists=stat is not None,
            bytes=stat.st_size if stat else None, mtime_ns=stat.st_mtime_ns if stat else None,
            content_sha256=None, file_identity_checked=False, target_dates_checked=False))
    # Statistics are observations, NOT a content-hash input binding for these minute files.
    no_single_source = [r for r in clean_probe['cases'] if not r['source_exists']]
    stock_queries = [dict(code=r['code'], date=d, table='分钟k', scope='STOCK',
                         key=f"分钟k:{r['code']}:{d}*") for r in no_single_source for d in (r['P'], r['E'])]
    index_dates = sorted({d for c in cases for d in (c['P'], c['E'])})
    index_queries = [dict(code='sh000001', date=d, table='分钟k', scope='SSE_PRICE_INDEX',
        key=f'分钟k:sh000001:{d}*', identity_required='SH000001_AND_EXPLICIT_SSE_COMPOSITE_NAME') for d in index_dates]
    docs = [Path('D:/a股数据/free-stockdb-main/docs/DATA_SOURCE.md'),
            HYBRID.parent / 'pullback_train/fetch_minute_2025.py',
            HYBRID / 'green_v2/shadow_bypass/source_semantics_evidence.md']
    for path in docs:
        track(path)
    report = dict(fixed_cases=cases, unique_required_members=len(required),
        unique_clean_files_present=sum(f['exists'] for f in files),
        unique_clean_files_missing=sum(not f['exists'] for f in files), file_observations=files,
        scope='EFFECTIVE_DATE_CLASSIFICATION_AND_FILESYSTEM_PRESENCE_ONLY',
        file_observations_are_frozen_price_inputs=False,
        source_time_verdict='UNVERIFIED', source_docs=[str(p) for p in docs],
        leaders_bound=False, sse_15m_bound=False, fit_ready=False,
        proposed_new_snapshot=dict(status='DRAFT_NOT_AUTHORIZED',
            stock_queries=stock_queries, index_queries=index_queries,
            http_request_cap=len(stock_queries) + len(index_queries), retries=0,
            auxiliary_requests=0, response_bytes_cap=4*1024**2, new_disk_bytes_cap=5*1024**2,
            no_http_executed=True, no_industry_collection_authorized=True))
    for path, digest in refs.items():
        if sha(Path(path)) != digest:
            raise ValueError('INPUT_CHANGED_DURING_RUN')
    for f in files:
        p = Path(f['path'])
        stat = p.stat() if p.is_file() else None
        if (stat is not None) != f['exists'] or (stat and (stat.st_size, stat.st_mtime_ns) != (f['bytes'], f['mtime_ns'])):
            raise ValueError('FILE_INVENTORY_CHANGED_DURING_RUN')
    encode = lambda obj: (json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf8')
    import hashlib
    data = encode(report)
    manifest = encode(dict(inputs=[dict(path=p, sha256=h) for p, h in sorted(refs.items())],
        outputs=[dict(path='inventory.json', bytes=len(data), sha256=hashlib.sha256(data).hexdigest())]))
    total = len(data) + len(manifest)
    if total > CAP or bytes_used() + total > TOTAL_CAP:
        raise ValueError('WRITE_BUDGET')
    OUT.mkdir(exist_ok=False)
    for name, body in (('inventory.json', data), ('manifest.json', manifest)):
        with (OUT / name).open('xb') as stream:
            stream.write(body)
    print(json.dumps(dict(bytes=total, members=len(required), present=report['unique_clean_files_present'],
        missing=report['unique_clean_files_missing'], proposed_http=report['proposed_new_snapshot']['http_request_cap'],
        actual_http=0, fit_ready=False)))


if __name__ == '__main__':
    run()
