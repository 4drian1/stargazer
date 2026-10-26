"""No-network independent saved-sidecar/prediction and compressed-raw readback."""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from postclose_research_scorer_v1 import ResearchScorer,encode,sha,strict_json,HERE
from postclose_local_source_bridge_v1 import latest_day_payload
from research_budget_v2 import TOTAL_CAP,bytes_used

ROOT=HERE/'postclose_incremental_refresh_20261017_v2'


def run():
    target=ROOT/'readback_acceptance.json'
    if target.exists():raise ValueError('ACCEPTANCE_EXISTS_NO_OVERWRITE')
    report=strict_json((ROOT/'report.json').read_bytes())
    if strict_json((ROOT/'stop_record.json').read_bytes())['status']!='COMPLETE':raise ValueError('BATCH_NOT_COMPLETE')
    for p,h in report['final_source_refs'].items():
        if sha(Path(p).read_bytes())!=h:raise ValueError('SOURCE_DRIFT')
    ledger=strict_json((ROOT/'source_ledger.json').read_bytes())
    for record in ledger:
        body=(ROOT/record['file']).read_bytes();raw=gzip.decompress(body)
        if sha(body)!=record['stored_sha256'] or sha(raw)!=record['sha256'] or len(raw)!=record['bytes']:
            raise ValueError('RAW_ARCHIVE_CORRUPT')
    scorer=ResearchScorer();frame=pd.read_parquet(ROOT/'feature_sidecar.parquet')
    saved=strict_json((ROOT/'scores.json').read_bytes());counts={}
    if set(frame.columns)!={'code','day',*scorer.fields}:raise ValueError('NONFEATURE_COLUMN')
    for day in [20261015,20261016]:
        payload,_=latest_day_payload(frame[frame.day.eq(day)],scorer)
        replay=scorer.score(payload)
        expected=next(r for r in saved if r['source_day']==day)
        if replay['rows']!=expected['rows']:raise ValueError('SAVED_PREDICTION_READBACK_FAILED')
        counts[str(day)]=sum(r['status']=='RESEARCH_SCORE' for r in replay['rows'])
    # A second calculation from the archived raw responses verifies all context fields,
    # rather than merely trusting the saved prediction's own input hash.
    from refresh_postclose_incremental_v2 import structured_pairs,valid_daily,contexts
    import msgpack
    daily={}
    for entry in ledger[:3]:
        raw=gzip.decompress((ROOT/entry['file']).read_bytes());day=int(entry['query'].rsplit(':',1)[1])
        daily[day]=valid_daily(structured_pairs(msgpack.unpackb(raw,raw=False,strict_map_key=False),'日k','*'),day)
    catalog=strict_json((HERE/'ths_full_concept_scope_20261017_v1.json').read_bytes())
    membership=sorted([r for r in pd.read_parquet(catalog['source']).to_dict('records') if r['source']=='ths' and r['type']=='concept'],key=lambda r:r['code'])
    # Arrays from Arrow become lists for the original membership helpers.
    for member in membership:member['symbols']=list(member['symbols'])
    context,_=contexts(daily,membership,scorer.manifest['research_codes'])
    a=frame.set_index(['code','day']);b=context.set_index(['code','day'])
    fields=[f for f in context if f.startswith('context_')]
    np.testing.assert_allclose(a[fields],b.loc[a.index,fields],rtol=0,atol=1e-12,equal_nan=True)
    artifacts={p.name:sha(p.read_bytes()) for p in ROOT.iterdir() if p.is_file()}
    result=dict(status='PASS_INCREMENTAL_FEATURE_AND_SCORE_READBACK',raw_archives=len(ledger),
                context_fields_recomputed=fields,score_counts=counts,artifact_sha256=artifacts,
                reader_sha256=sha((HERE/'read_postclose_incremental_v2.py').read_bytes()),
                auditor_sha256=sha(Path(__file__).read_bytes()),
                original_model_unchanged=True,independent_validation_passed=False)
    body=encode(result)
    if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())+len(body)>16*1024**2 or bytes_used()+len(body)>TOTAL_CAP:
        raise ValueError('READBACK_BUDGET_STOP')
    with target.open('xb') as stream:stream.write(body)
    return dict(status=result['status'],raw_archives=len(ledger),score_counts=counts,acceptance_bytes=len(body))


if __name__=='__main__':print(run())
