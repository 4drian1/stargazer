"""Replay a fully captured local snapshot without repeating stockdb requests."""
import json
import io
from pathlib import Path
import pandas as pd

from live_postclose_source_v1 import LocalCapture,CapturedMinutes,digest,now_stamp
from joint_shadow_context_v1 import RiskSnapshot
from joint_shadow_snapshot_v1 import HERE


def captured_producer(previous):
    previous=Path(previous)
    def produce(day,codes,output,progress):
        terminal=json.loads((previous/'terminal.json').read_text(encoding='utf-8'))
        if int(terminal['data_date'])!=int(day):
            raise ValueError('CAPTURE_DATE_MISMATCH')
        capture=LocalCapture(output,budget_root=Path(output).parent)
        manifest_body=capture.read(previous/'source_manifest.json')
        manifest=json.loads(manifest_body)
        scope=manifest['membership_source']
        catalog=pd.read_parquet(io.BytesIO(capture.read(scope['source'],scope['source_sha256'])))
        catalog=catalog[(catalog.source=='ths')&(catalog.type=='concept')]
        if set(catalog.code)!={r['code'] for r in scope['boards']}:
            raise ValueError('COMPARISON_UNIVERSE_DRIFT')
        boards={r.code:list(r.symbols) for r in catalog.sort_values('code').itertuples()}
        cutoff=int(day)*1000000+150000
        indices,receipts={},{}
        for symbol,name in [('000001.SH','sse'),('399006.SZ','cyb')]:
            path=HERE/'index_minutes_v1'/f'{name}_5min.parquet'
            meta=json.loads(capture.read(path.with_suffix('.meta.json')))
            frame=pd.read_parquet(io.BytesIO(capture.read(path,meta['sha256'])))
            frame=frame.rename(columns={'ts':'date','raw_volume':'volume'})
            frame=frame[frame.date<=cutoff]
            if int(frame.date.max())!=cutoff:
                raise ValueError('INDEX_NOT_CURRENT_CUTOFF')
            indices[symbol]=frame
            receipts[symbol]=dict(reference=str(path)+'#sha256='+meta['sha256'],price_basis='RAW',
                clock='END_5M',available_at=now_stamp(),availability_basis='LOCAL_READER_OBSERVED_BYTES')
        entries=manifest['entries']
        for code,entry in entries.items():
            observed=max((r['observed_at'] for r in entry['extensions']),default=now_stamp())
            receipts[code]=dict(reference='captured:'+code+':'+digest(json.dumps(entry,sort_keys=True).encode()),
                price_basis='RAW',clock='END_1M',available_at=observed,
                availability_basis='PRIOR_SAME_DAY_CAPTURE_NOW_HASH_REVERIFIED')
        capture.read_bytes+=manifest['read_bytes']+10*1024**3
        capture.requests=manifest['requests']
        capture.save('source_manifest.json',manifest_body)
        capture.save('replay_lineage.json',json.dumps(dict(prior_run=str(previous),
            prior_manifest_sha256=digest(manifest_body),
            prior_read_upper_bound=manifest['read_bytes']+10*1024**3,
            reason='Invalid-OHLC rows are explicitly masked exactly as the frozen feature kernel; raw archives retained')).encode())
        observed=now_stamp()
        days=sorted(d for d in set((indices['000001.SH'].date//1000000).astype(int)) if d>=20250102)
        stock=CapturedMinutes(entries,capture,cutoff,progress)
        return RiskSnapshot(cutoff,observed,days,stock,indices,boards,receipts,observed,
                            'FORWARD_SNAPSHOT',observed_at=observed),capture,manifest['failures']
    return produce
