"""Ten local-only read requests; canonical diagnostics against preserved price snapshots."""
from pathlib import Path
from datetime import datetime,timezone
import json
import hashlib
import urllib.request
import urllib.parse
import math

import pyarrow.parquet as pq
from independent_fetch_client import _NoRedirect
from price_verification_guarded_run import sha,encoded
from stockdb_adjustment_v2 import continuity_diagnostic,canonical_events
from verify_corporate_action_v1 import FIXED_CASES
from research_budget_v2 import TOTAL_CAP,bytes_used

HERE=Path(__file__).resolve().parent
OUT=HERE/'historical_stock_day_preparation_v1/adjustment_snapshot_v2_20261015'
PRICES=HERE/'historical_stock_day_preparation_v1/real_binding_v1'


def run():
    if OUT.exists():raise ValueError('SNAPSHOT_ALREADY_CLAIMED')
    if bytes_used()+2*1024**2>TOTAL_CAP:raise ValueError('REGISTERED_BUDGET')
    inputs={c:sha(PRICES/f'prices_{c}.parquet') for c in FIXED_CASES}
    OUT.mkdir(exist_ok=False)
    ledger=[];results=[];error=None;received=0
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),_NoRedirect())
    try:
        for code in FIXED_CASES:
            url='http://127.0.0.1:7899/?'+urllib.parse.urlencode(dict(cmd='get',t=f'复权:{code}:*',json=1))
            entry=dict(code=code,url=url,fetched_at=datetime.now(timezone.utc).isoformat())
            ledger.append(entry)
            with opener.open(url,timeout=10) as response:
                raw=response.read(min(128*1024,1024*1024-received))
            received+=len(raw)
            if received>=1024*1024 or len(raw)>=128*1024:raise ValueError('LOCAL_SNAPSHOT_RESPONSE_LIMIT')
            entry.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            with (OUT/f'{code}.json.raw').open('xb') as f:f.write(raw)
            records=[]
            for key,value in json.loads(raw):
                if not key.startswith(f'复权:{code}:'):raise ValueError('ADJUSTMENT_CODE_MISMATCH')
                mult=value.get('mult')
                if type(mult) not in (int,float) or not math.isfinite(mult) or mult<=0:
                    raise ValueError('INVALID_MULT')
                records.append(dict(date=int(key.rsplit(':',1)[1]),mult=mult,cum=value.get('cum')))
            canonical_events(records)  # conflicting same-date events reject rather than overwrite
            records.sort(key=lambda r:r['date'])
            bars=pq.read_table(PRICES/f'prices_{code}.parquet').to_pylist()
            diag=continuity_diagnostic(bars,records)
            after=[x for x in diag['checks'] if x['date']>=20180101]
            outside=[x for x in after if not x['inside_diagnostic_band']]
            stored_cum_checks=[]
            for prev,current in zip(records,records[1:]):
                a,b=prev['cum'],current['cum']
                if type(a) in (int,float) and type(b) in (int,float) and a>0 and b>0:
                    stored_cum_checks.append(dict(date=current['date'],
                        relative_step_error=abs(b/(a*current['mult'])-1)))
            results.append(dict(code=code,n_events=len(records),
                                events_since_2018=sum(r['date']>=20180101 for r in records),
                                checks_since_2018=len(after),outside_diagnostic_band=outside,
                                max_stored_cum_step_error=max((x['relative_step_error'] for x in stored_cum_checks),default=None),
                                diagnostic_status=diag['diagnostic_status'],
                                effective_dates_verified=False,coverage_verified=False,
                                price_snapshot_sha256=inputs[code],training_eligible=False))
        if any(sha(PRICES/f'prices_{c}.parquet')!=h for c,h in inputs.items()):raise ValueError('PRICE_INPUT_DRIFT')
    except (Exception,KeyboardInterrupt) as exc:error=f'{type(exc).__name__}:{exc}'
    report=dict(status='failed' if error else 'completed',error=error,requests=len(ledger),
                response_bytes=received,request_ledger=ledger,results=results,
                note='Local recorded-date engineering diagnostics only; no coverage/effective-date admission or new labels.')
    data=encoded(report)
    if bytes_used()+len(data)>TOTAL_CAP:raise ValueError('REGISTERED_BUDGET')
    with (OUT/'stop_record.json').open('xb') as f:f.write(data)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':run()
