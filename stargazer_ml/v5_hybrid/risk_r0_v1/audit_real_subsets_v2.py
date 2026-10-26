"""Read-only legacy subset audit; writes a new errata report, never repairs old labels in place."""
from pathlib import Path
import json
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

from price_verification_guarded_run import encoded,sha
from prepare_historical_risk_v1 import bytes_used

HERE=Path(__file__).resolve().parent


def audit():
    output=HERE/'real_subset_audit_v2_20261015.json'
    if output.exists(): raise ValueError('AUDIT_EXISTS_NO_OVERWRITE')
    roots=[HERE/'historical_stock_day_real_subset_v1',HERE/'historical_stock_day_real_subset_2code_v1']
    results=[]
    artifacts=[]
    for root in roots:
        report=json.loads((root/'stop_record.json').read_bytes())
        counts=Counter();per_segment={}
        files=sorted(root.glob('*.parquet'))
        tables=[]
        for file in files:
            table=pq.read_table(file)
            tables.append(table)
            for row in table.select(['segment','training_eligible','y_tail_5d']).to_pylist():
                counts['persisted_rows']+=1
                if row['training_eligible']:
                    counts['legacy_admitted']+=1
                    counts['legacy_positive']+=int(row['y_tail_5d']==1)
                seg=per_segment.setdefault(row['segment'],Counter())
                seg['persisted_rows']+=1
        part_bytes=sum(p.stat().st_size for p in files)
        repacked=None
        if tables and report['status']=='completed':
            sink=pa.BufferOutputStream()
            pq.write_table(pa.concat_tables(tables),sink,compression='zstd',row_group_size=4096)
            repacked=sink.getvalue().size
        results.append(dict(path=str(root),terminal_status=report['status'],
                            recorded_rows_seen=report['counts']['rows_seen'],
                            recorded_admitted_seen=report['counts'].get('admitted'),
                            counts=dict(counts),segments={k:dict(v) for k,v in per_segment.items()},
                            part_bytes=part_bytes,legacy_density_bytes_per_row=part_bytes/counts['persisted_rows'],
                            repacked_in_memory_bytes=repacked,
                            reencoded_density_bytes_per_row=repacked/counts['persisted_rows'] if repacked else None,
                            training_use='QUARANTINED_NOT_ADMITTED',
                            reasons=['stored_cum_vs_canonical_mult','recorded_effective_dates_unverified',
                                     'internal_continuity_does_not_certify_coverage','raw_etf_market_features']))
        artifacts.extend(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
                         for p in root.rglob('*') if p.is_file())
    prep=HERE/'historical_stock_day_preparation_v1'
    for root in (prep/'real_binding_v1',prep/'real_binding_2code_v1'):
        artifacts.extend(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
                         for p in root.rglob('*') if p.is_file())
    for p in (prep/'corporate_action_verification_v1.json',prep/'csi300_tracker_510300.csv',
              prep/'csi300_tracker_510300.csv.binding.json'):
        artifacts.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    inventory=json.loads((prep/'population_inventory.json').read_bytes())
    rows=sum(x['structural_after_purge'] for x in inventory['segments'].values())
    estimates=[dict(basis=r['path'],full_matrix_bytes_at_legacy_density=int(rows*r['legacy_density_bytes_per_row']),
                    full_matrix_bytes_at_reencoded_density=int(rows*r['reencoded_density_bytes_per_row'])
                    if r['reencoded_density_bytes_per_row'] else None) for r in results]
    final=dict(status='AUDIT_ONLY_QUARANTINED',legacy_subsets=results,
               full_structural_rows=rows,storage_extrapolations=estimates,
               estimate_note='Measured legacy-data compression only; corrected full-universe density may differ. Not a budget grant.',
               model_fitted=False,new_real_labels_generated=0,training_ready=False,
               preserved_artifacts=artifacts)
    data=encoded(final)
    if len(data)>2*1024**2 or bytes_used()+len(data)>128*1024**2:
        raise ValueError('AUDIT_BUDGET')
    if any(sha(x['path'])!=x['sha256'] for x in artifacts): raise ValueError('LEGACY_INPUT_CHANGED')
    with output.open('xb') as f:f.write(data)
    print(json.dumps(dict(output=str(output),subsets=results,storage_extrapolations=estimates,
                          preserved_files=len(artifacts)),ensure_ascii=False))


if __name__=='__main__':audit()
