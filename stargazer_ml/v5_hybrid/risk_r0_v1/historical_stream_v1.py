"""Bound-source historical matrix streaming; source evidence required, FIT not required."""
from collections import Counter
from pathlib import Path
import json
import math

import pyarrow as pa
import pyarrow.parquet as pq

from historical_risk_core_v1 import (STOCK_FIELDS, MARKET_FIELDS, in_scope,
                                    label_at,stock_features,market_features,segment)
from market_index_source_v1 import load_csi300_closes
from price_verification_guarded_run import encoded,sha
from research_budget_v2 import TOTAL_CAP


def bound_file(ref, base):
    path = Path(ref['path'])
    if not path.is_absolute(): path=base/path
    if not path.is_file() or sha(path)!=ref.get('sha256'):
        raise ValueError('SOURCE_HASH_MISMATCH:'+str(path))
    return path.resolve()


def read_unique(path, fields, expected_code):
    out={}
    for row in pq.read_table(path,columns=['code','date']+list(fields)).to_pylist():
        if row.pop('code')!=expected_code: raise ValueError('SOURCE_STOCK_CODE_MISMATCH')
        d=row.pop('date')
        if type(d) is not int: raise ValueError('INVALID_SOURCE_DATE')
        if d in out and out[d]!=row: raise ValueError('SOURCE_DATE_CONFLICT')
        out[d]=row
    return out


def bound_rows(binding_path, index_path):
    """All files/content hashes checked before yielding any row, then checked again at EOF."""
    binding_path=Path(binding_path).resolve()
    root=binding_path.parent
    binding_hash=sha(binding_path)
    doc=json.loads(binding_path.read_bytes())
    if doc.get('schema')!='HISTORICAL_SOURCE_BINDING_V1' or not doc.get('sources'):
        raise ValueError('SOURCE_BINDING_REQUIRED')
    if doc.get('provenance',{}).get('source')=='local_stockdb_http':
        raise ValueError('LEGACY_STOCKDB_CONTINUITY_BINDING_QUARANTINED_USE_AUDITED_V2')
    calendar_path=bound_file(doc['calendar'],root)
    calendar=json.loads(calendar_path.read_bytes())
    if calendar!=sorted(set(calendar)): raise ValueError('CALENDAR_CONFLICT')
    # Valid date codes, not hypothetical elapsed-day numbers.
    from datetime import datetime
    for d in calendar:
        if type(d) is not int or len(str(d))!=8:
            raise ValueError('CALENDAR_DATE_INVALID')
        dt=datetime.strptime(str(d),'%Y%m%d')
        if dt.weekday()>=5: raise ValueError('CALENDAR_WEEKEND')
    disclosure=bound_file(doc['security_scope_disclosure'],root)
    index_path=Path(index_path).resolve()
    index_binding=index_path.with_name(index_path.name+'.binding.json')
    closes=load_csi300_closes(index_path)
    refs=[(binding_path,binding_hash),(calendar_path,sha(calendar_path)),
          (disclosure,sha(disclosure)),(index_path,sha(index_path)),(index_binding,sha(index_binding))]
    sources=[]
    seen=set()
    for source in doc['sources']:
        code=source['code']
        if not in_scope(code) or code in seen: raise ValueError('CODE_SCOPE_OR_DUPLICATE')
        seen.add(code)
        prices=bound_file(source['prices'],root)
        factors=bound_file(source['factors'],root)
        audit=bound_file(source['evidence'],root)
        evidence=json.loads(audit.read_bytes())
        # A global boolean or a report snapshot cannot admit an individual source interval.
        expected=dict(code=code,scale='RAW',price_sha256=sha(prices),factor_sha256=sha(factors),
                      calendar_sha256=sha(calendar_path),scale_verdict='ACCEPTED',
                      corporate_action_coverage_verdict='ACCEPTED')
        if any(evidence.get(k)!=v for k,v in expected.items()):
            raise ValueError('PRICE_OR_COVERAGE_EVIDENCE_PENDING:'+code)
        if not evidence.get('basis_refs'):
            raise ValueError('EVIDENCE_BASIS_REQUIRED')
        for ref in evidence['basis_refs']:
            p=bound_file(ref,audit.parent)
            refs.append((p,sha(p)))
        for key in ('from','through'):
            datetime.strptime(str(evidence[key]),'%Y%m%d')
        if evidence['from']>evidence['through']: raise ValueError('EVIDENCE_INTERVAL_INVALID')
        refs.extend((p,sha(p)) for p in (prices,factors,audit))
        sources.append((code,prices,factors,evidence,sha(audit)))
    for code,prices,factor_path,evidence,evidence_hash in sources:
        bars=read_unique(prices,('open','high','low','close'),code)
        factor_rows=read_unique(factor_path,('factor',),code)
        factors={d:r['factor'] for d,r in factor_rows.items()}
        # Stored rows outside the chosen calendar must not silently disappear.
        if set(bars)-set(calendar): raise ValueError('BAR_DATE_NOT_IN_CALENDAR')
        for i,p in enumerate(calendar):
            if not 20190101<=p<=20260930 or p not in bars: continue
            end=calendar[i+5] if i+5<len(calendar) else None
            start=calendar[max(0,i-60)]
            coverage=(end is not None and evidence['from']<=start
                      and end<=evidence['through'])
            label=label_at(bars,calendar,p,factors,scale_verified=coverage,
                           coverage_verified=coverage)
            stock,status=stock_features(bars,calendar,p,factors) if coverage else (
                {f:None for f in STOCK_FIELDS},'SOURCE_EVIDENCE_INTERVAL_INCOMPLETE')
            market,market_status=market_features(closes,calendar,p)
            seg,within=segment(p,label['label_end_date'])
            features=dict(stock,**market)
            complete=all(type(v) in (int,float) and math.isfinite(v) for v in features.values())
            eligible=bool(label['status']=='LABEL_COMPLETE' and complete and within)
            yield dict(code=code,P=p,segment=seg,label_end_date=label['label_end_date'],
                       mae_5d=label['mae_5d'],y_tail_5d=label['y_tail_5d'],features=features,
                       label_status=label['status'],feature_status=status,
                       market_status=market_status,training_eligible=eligible,
                       source_evidence_sha256=evidence_hash)
    for p,h in refs:
        if sha(p)!=h: raise ValueError('SOURCE_CHANGED_DURING_STREAM:'+str(p))


def write_stream(rows, output, registered_bytes, *, total_cap=TOTAL_CAP,
                 new_cap=2*1024**2, chunk_rows=256):
    """Immutable Parquet chunks; preflight actual compressed bytes; partial output is terminal."""
    output=Path(output)
    if output.exists(): raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if chunk_rows<1: raise ValueError('INVALID_CHUNK_SIZE')
    reserve=16384
    if new_cap<=reserve or registered_bytes()+reserve>total_cap:
        raise ValueError('BUDGET_NO_TERMINAL_RESERVE')
    counts=Counter();pending=[];used=0;files=[]
    output.mkdir(exist_ok=False)
    failure=None
    schema=pa.schema([
        ('code',pa.string()),('P',pa.int64()),('segment',pa.string()),
        ('label_end_date',pa.int64()),('mae_5d',pa.float64()),('y_tail_5d',pa.int8()),
        ('features',pa.struct([(k,pa.float64()) for k in STOCK_FIELDS+MARKET_FIELDS])),
        ('label_status',pa.string()),('feature_status',pa.string()),('market_status',pa.string()),
        ('training_eligible',pa.bool_()),('source_evidence_sha256',pa.string()),
        ('admission_scope',pa.string()),('coverage_verified',pa.bool_())])

    def flush():
        nonlocal used
        if not pending: return
        sink=pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pylist(pending,schema=schema),sink,compression='zstd')
        data=sink.getvalue().to_pybytes()
        if used+len(data)+reserve>new_cap or registered_bytes()+len(data)+reserve>total_cap:
            raise ValueError('ACTUAL_SERIALIZED_BUDGET_STOP')
        name=f'part_{len(files):05d}.parquet'
        with (output/name).open('xb') as f: f.write(data)
        used+=len(data);files.append(dict(name=name,bytes=len(data),sha256=sha(output/name)))
        pending.clear()

    try:
        for row in rows:
            # Check numeric integrity even for purged or non-admitted records.
            y=row.get('y_tail_5d')
            mae=row.get('mae_5d')
            if y is not None and (type(y) is not int or y not in (0,1)):
                raise ValueError('LABEL_VALUE_INVALID')
            if mae is not None and (type(mae) not in (int,float) or not math.isfinite(mae)):
                raise ValueError('LABEL_VALUE_NONFINITE')
            for value in row.get('features',{}).values():
                if value is not None and (type(value) not in (int,float) or not math.isfinite(value)):
                    raise ValueError('FEATURE_VALUE_NONFINITE')
            pending.append(row)
            counts['rows_seen']+=1
            if row.get('training_eligible') is True:
                counts['admitted']+=1
                y=row.get('y_tail_5d')
                if type(y) is not int or y not in (0,1): raise ValueError('ADMITTED_LABEL_INVALID')
                if any(type(row['features'].get(k)) not in (int,float)
                       or not math.isfinite(row['features'][k]) for k in STOCK_FIELDS+MARKET_FIELDS):
                    raise ValueError('ADMITTED_FEATURE_INVALID')
                counts['positive' if y else 'nonpositive']+=1
            if len(pending)>=chunk_rows: flush()
        flush()
        if not counts['rows_seen']: raise ValueError('EMPTY_INPUT')
    except Exception as exc:
        failure=f'{type(exc).__name__}:{exc}'
    report=dict(status='failed' if failure else 'completed',error=failure,
                counts=dict(counts),parts=files,part_bytes=used,fit_performed=False,
                training_ready=False,note='Full population/class/date gate must run separately; partial batch cannot train.')
    data=encoded(report)
    if len(data)>reserve:
        data=encoded(dict(status='failed',error='TERMINAL_REPORT_RESERVE_EXCEEDED',
                          part_count=len(files),part_bytes=used,training_ready=False))
    with (output/'stop_record.json').open('xb') as f: f.write(data)
    if failure: raise ValueError(failure)
    return report
