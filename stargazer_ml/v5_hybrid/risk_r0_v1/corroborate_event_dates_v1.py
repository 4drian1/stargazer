"""Cross-check only already archived Sino/stockdb event dates; no network or admission."""
import json
from pathlib import Path
from price_verification_guarded_run import sha,encoded
from research_budget_v2 import bytes_used,TOTAL_CAP

HERE=Path(__file__).parent


def run():
    external=HERE/'independent_fetch/batch_b_dividend_probe/standard_event_table_v1.json'
    folder=HERE/'historical_stock_day_preparation_v1/adjustment_snapshot_v2_20261015'
    inputs=[external]+[folder/f'{c}.json.raw' for c in ('000001','002594')]
    refs=[dict(path=str(p),sha256=sha(p)) for p in inputs]
    rows=json.loads(external.read_bytes());results=[]
    for code in ('000001','002594'):
        native=json.loads((folder/f'{code}.json.raw').read_bytes())
        local={int(k.rsplit(':',1)[1]) for k,v in native
               if 20180101<=int(k.rsplit(':',1)[1])<=20261014 and
               (v['mult']!=1 or any(v.get(f,0) for f in ('div','give','trans')))}
        other=set()
        for r in rows:
            if r['code']!=code or r.get('reconcile_status')!='eligible':continue
            d=r.get('dates',{}).get('除权除息日') or r.get('dates',{}).get('除权日')
            if d and 20180101<=int(d.replace('-',''))<=20261014:other.add(int(d.replace('-','')))
        results.append(dict(code=code,matched=sorted(local&other),local_only=sorted(local-other),
                            external_only=sorted(other-local),coverage_verified=False,
                            conclusion='DATE_CORROBORATION_ONLY_NOT_EVENT_COMPLETENESS'))
    report=dict(inputs=refs,results=results,matched_dates=sum(len(r['matched']) for r in results),
                training_eligible=False,new_labels_generated=0)
    if any(sha(x['path'])!=x['sha256'] for x in refs):raise ValueError('INPUT_DRIFT')
    data=encoded(report);output=HERE/'known_event_date_corroboration_20261015.json'
    if bytes_used()+len(data)>TOTAL_CAP:raise ValueError('REGISTERED_BUDGET')
    with output.open('xb') as f:f.write(data)
    print(json.dumps(dict(output=str(output),matched_dates=report['matched_dates'])))


if __name__=='__main__':run()
