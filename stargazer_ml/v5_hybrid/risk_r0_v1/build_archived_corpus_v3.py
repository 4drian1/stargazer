"""Mechanical archived-page corpus; bounded local reads, no external fetch or model fit."""
from pathlib import Path
from datetime import datetime,timezone
import json
import gzip
import hashlib
import urllib.parse
import urllib.request
from collections import Counter

from sina_div_client import parse_bonus_page
from independent_fetch_client import _NoRedirect
from build_restricted_history_v3 import reconcile_code,SCOPE
from historical_risk_core_v1 import label_at,stock_features,market_features,segment,sample_gate,in_scope
from stockdb_adjustment_v2 import rebuild_mult_factors,valid_date
from historical_stream_v1 import write_stream
from market_index_source_v1 import bind_from_experiment_config
from research_budget_v2 import bytes_used,TOTAL_CAP
from price_verification_guarded_run import encoded,sha

HERE=Path(__file__).resolve().parent
OUT=HERE/'historical_archived_corpus_v3_20261015'
CAP=64*1024**2
RESPONSE_CAP=128*1024**2
RESERVE=2*1024**2
BATCHES=('batch_b_dividend_probe','batch_c_mainpop_probe','batch_d_mainpop_30')


def decimal_number(v):
    if v is None or str(v).strip() in ('--','-','nan','None',''):return None
    try:
        from decimal import Decimal
        x=Decimal(str(v).replace(',',''))
        return float(x) if x.is_finite() and x>=0 else None
    except Exception:return None


def normalize_external(code,parsed):
    if parsed.get('bonus') is None or parsed.get('allotment') is None:
        raise ValueError('EXTERNAL_TABLES_NOT_IDENTIFIED:'+code)
    records=[]
    for row in parsed['bonus']:
        if row.get('进度')!='实施':continue
        c=decimal_number(row.get('派息(税前)(元)'))
        records.append(dict(code=code,table='bonus',reconcile_status='eligible',progress=row.get('进度'),
                            dates={'除权除息日':row.get('除权除息日')},
                            per_share_cash=c/10 if c is not None else None,
                            bonus_ratio_per10=dict(song=decimal_number(row.get('送股(股)')),
                                                   zhuan=decimal_number(row.get('转增(股)')))))
    for row in parsed['allotment']:
        d=row.get('除权日')
        if not isinstance(d,str) or len(d)!=10:continue
        records.append(dict(code=code,table='allotment',reconcile_status='eligible',
                            dates={'除权日':d},progress=row.get('进度')))
    return records


def frozen_pages():
    pages={}
    for batch in BATCHES:
        for meta_path in sorted((HERE/'independent_fetch'/batch).glob('run_*/*.json')):
            meta=json.loads(meta_path.read_bytes());code=meta.get('code')
            if not in_scope(code):continue
            blob=meta_path.with_suffix('.bin')
            if not blob.is_file() or sha(blob)!=meta.get('sha256'):raise ValueError('EXTERNAL_ARCHIVE_HASH')
            if code in pages:raise ValueError('DUPLICATE_ARCHIVED_CODE')
            pages[code]=(blob,meta_path)
    if len(pages)!=43:raise ValueError('EXPECTED_ARCHIVED_CORPUS_43')
    return dict(sorted(pages.items()))


def run():
    pages=frozen_pages()
    if OUT.exists():raise ValueError('BATCH_ALREADY_CLAIMED')
    if bytes_used()+CAP>TOTAL_CAP:raise ValueError('REGISTERED_BUDGET')
    calendar_path=HERE/'historical_stock_day_preparation_v1/manifest.json'
    calendar=json.loads(calendar_path.read_bytes())['calendar_dates']
    closes=bind_from_experiment_config()
    inputs=[calendar_path,Path(__file__),HERE/'ordinary_action_reconciliation_v3.py',
            HERE/'build_restricted_history_v3.py',HERE/'historical_risk_core_v1.py',
            HERE/'historical_stream_v1.py',HERE/'contracts/RESTRICTED_HISTORICAL_RESEARCH_V3.md',
            HERE/'contracts/historical_input_overlay_20261015.json']
    inputs.extend(p for pair in pages.values() for p in pair)
    refs=[dict(path=str(p),sha256=sha(p)) for p in inputs]
    plan=dict(codes=list(pages),selection='ALL_FROZEN_ARCHIVED_PAGES_NOT_LABEL_BASED',
              inputs=refs,request_cap=86,response_cap=RESPONSE_CAP,disk_cap=CAP,
              scope=SCOPE,fit_authorized=False,
              limitations=['archived source availability/selection bias','vendor event omissions possible',
                           'not full historical market population','not a new blind test'])
    OUT.mkdir(exist_ok=False)
    used=0
    def save(name,data):
        nonlocal used
        if used+len(data)+RESERVE>CAP or bytes_used()+len(data)+RESERVE>TOTAL_CAP:
            raise ValueError('CORPUS_DISK_BUDGET')
        with (OUT/name).open('xb') as f:f.write(data)
        used+=len(data)
    save('plan.json',encoded(plan))
    plan_hash=sha(OUT/'plan.json')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),_NoRedirect())
    ledger=[];response_bytes=0;cases=[];audit={};error=None
    try:
        for code,(blob,meta_path) in pages.items():
            external=normalize_external(code,parse_bonus_page(blob.read_bytes()))
            raw_by_table={}
            for table,tag in (('日k','daily'),('复权','actions')):
                if len(ledger)>=86:raise ValueError('LOCAL_REQUEST_CAP')
                url='http://127.0.0.1:7899/?'+urllib.parse.urlencode(dict(cmd='get',t=f'{table}:{code}:*',json=1))
                entry=dict(code=code,table=table,url=url,fetched_at=datetime.now(timezone.utc).isoformat())
                ledger.append(entry)
                cap=min(8*1024**2,RESPONSE_CAP-response_bytes)
                if cap<=0:raise ValueError('LOCAL_RESPONSE_CAP')
                with opener.open(url,timeout=10) as r:body=r.read(cap)
                response_bytes+=len(body)
                if len(body)>=cap:raise ValueError('LOCAL_RESPONSE_LIMIT_UNCERTIFIED')
                entry.update(bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
                save(f'{code}_{tag}.json.gz',gzip.compress(body,mtime=0))
                raw_by_table[tag]=json.loads(body)
            bars={}
            for key,row in raw_by_table['daily']:
                if not key.startswith(f'日k:{code}:') or row.get('code')!=code:raise ValueError('DAILY_CODE_MISMATCH')
                d=valid_date(row['date']);record={k:row.get(k) for k in ('open','high','low','close')}
                if d in bars and bars[d]!=record:raise ValueError('DUPLICATE_PRICE_CONFLICT')
                bars[d]=record
            events,event_audit,issues=reconcile_code(code,bars,external,raw_by_table['actions'])
            issues=[dict(x,date=None,recorded_date=x['date']) if x['reason']=='SINGLE_SOURCE_EVENT' else x
                    for x in issues]
            factors=rebuild_mult_factors(events,calendar)
            cases.append((code,bars,factors,issues))
            audit[code]=dict(events=event_audit,issues=issues)
            print(f'{code} sources_read={len(ledger)}/86 events={len(event_audit)} issues={len(issues)}',flush=True)
        if any(sha(x['path'])!=x['sha256'] for x in refs):raise ValueError('FROZEN_INPUT_DRIFT')
        rows=[];boards=Counter();isolations=Counter()
        for code,bars,factors,issues in cases:
            for i,p in enumerate(calendar):
                if not 20190101<=p<=20260930:continue
                end=calendar[i+5] if i+5<len(calendar) else None
                start=calendar[max(0,i-60)]
                blocked=end is None or end>20261013 or any(x['date'] is None or start<=x['date']<=end for x in issues)
                label=label_at(bars,calendar,p,factors,scale_verified=not blocked,coverage_verified=not blocked)
                feat,fs=stock_features(bars,calendar,p,factors)
                market,ms=market_features(closes,calendar,p);feat.update(market)
                seg,within=segment(p,label['label_end_date'])
                eligible=bool(not blocked and within and label['status']=='LABEL_COMPLETE'
                              and all(v is not None for v in feat.values()))
                if not eligible:isolations['not_eligible']+=1
                if blocked:isolations['event_or_source_interval_block']+=1
                boards['STAR' if code.startswith('688') else 'GEM' if code.startswith(('300','301')) else
                       'SH_MAIN' if code.startswith('6') else 'SZ_MAIN']+=int(eligible)
                rows.append(dict(code=code,P=p,segment=seg,label_end_date=label['label_end_date'],
                                 mae_5d=label['mae_5d'],y_tail_5d=label['y_tail_5d'],features=feat,
                                 label_status=label['status'],feature_status=fs,market_status=ms,
                                 training_eligible=eligible,source_evidence_sha256=plan_hash,
                                 admission_scope=SCOPE,coverage_verified=False))
        if any(sha(x['path'])!=x['sha256'] for x in refs):raise ValueError('FROZEN_INPUT_DRIFT')
        gate=sample_gate(rows)
        gate.update(scope=SCOPE,fit_authorized=False,full_market_representative=False)
        metadata=encoded(dict(scope=SCOPE,audit=audit,gate=gate,isolation_counts=dict(isolations),
                              board_eligible_counts=dict(boards),coverage_verified=False))
        remaining=CAP-used-len(metadata)-RESERVE
        write_stream(iter(rows),OUT/'matrix',bytes_used,total_cap=TOTAL_CAP,
                     new_cap=remaining,chunk_rows=4096)
        save('evidence_and_gate.json',metadata)
    except (Exception,KeyboardInterrupt) as exc:error=f'{type(exc).__name__}:{exc}'
    terminal=dict(status='failed' if error else 'completed',error=error,local_requests=len(ledger),
                  response_bytes=response_bytes,request_ledger=ledger,fit_performed=False,
                  scope=SCOPE,training_ready=bool(error is None and gate['training_ready']))
    data=encoded(terminal)
    if len(data)>RESERVE:raise ValueError('TERMINAL_RESERVE')
    with (OUT/'stop_record.json').open('xb') as f:f.write(data)
    print(json.dumps(terminal,ensure_ascii=False))


if __name__=='__main__':run()
