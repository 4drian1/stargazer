"""Read-back acceptance of actual joint shards, without source probes or fit."""
import json

import numpy as np
import pyarrow.parquet as pq

from build_joint_candidate_context_v1 import SOURCE,OUT,model_part,AUDIT_ONLY
from build_joint_risk_features_v1 import digest


def run():
    manifest=json.loads((SOURCE/'manifest.json').read_bytes())
    context=json.loads((OUT/'manifest.json').read_bytes())
    if context['source_manifest_sha256']!=digest((SOURCE/'manifest.json').read_bytes()):
        raise ValueError('SOURCE_MANIFEST_DRIFT')
    for directory in [SOURCE,OUT]:
        if json.loads((directory/'stop_record.json').read_bytes())['status']!='COMPLETE':
            raise ValueError('MATERIALIZATION_NOT_COMPLETE')
    shared=pq.read_table(SOURCE/'shared_market_and_leaders.parquet').to_pandas()
    times=shared.decision_ts_assumed_end.to_numpy(dtype=np.int64)
    for column in shared:
        if column.endswith(('anchor_asof_day','reference_close_day')):
            values=shared[column].to_numpy(dtype=float);known=np.isfinite(values)
            if not (values[known]<times[known]//1000000).all():
                raise ValueError('DAILY_CONTEXT_NOT_PRIOR_DAY')
    seen=set();rows=0;earliest=None;latest=None
    for part in manifest['joint_parts']:
        frame=pq.read_table(SOURCE/part['path'],columns=['code','cutoff_index','stock_close',
            'stock_corporate_action_scale_verified']).to_pandas()
        codes=set(frame.code);positions=frame.cutoff_index.to_numpy(dtype=int)
        if seen&codes or frame.duplicated(['code','cutoff_index']).any():
            raise ValueError('JOINT_DUPLICATE_KEY')
        if not (np.isfinite(frame.stock_close)&(frame.stock_close>0)).all():
            raise ValueError('INVALID_STOCK_PRICE')
        if frame.stock_corporate_action_scale_verified.any():
            raise ValueError('UNAUTHORIZED_PRICE_ADMISSION')
        seen.update(codes);rows+=len(frame)
        lo,hi=int(times[positions].min()),int(times[positions].max())
        earliest=lo if earliest is None else min(earliest,lo)
        latest=hi if latest is None else max(latest,hi)
    if rows!=context['rows'] or rows!=manifest['summary']['joint_rows']:
        raise ValueError('JOINT_CONTEXT_ROW_COUNT_MISMATCH')
    for i in sorted({0,len(context['parts'])-1}):
        frame=model_part(i)
        if any(c in frame for c in AUDIT_ONLY) or frame.joint_training_admitted.any():
            raise ValueError('MODEL_READER_ADMISSION_ERROR')
        own=frame.context_current_leader_is_candidate
        if frame.loc[own,'context_current_leader_return_loo_raw'].notna().any():
            raise ValueError('CANDIDATE_SELF_REFERENCE')
    report=dict(status='READBACK_PASS_NOT_TRAINING_READY',rows=rows,stock_codes=len(seen),
        board_rows=manifest['summary']['board_rows'],concepts=manifest['summary']['concepts'],
        actual_stock_start=earliest,actual_stock_end=latest,market_calendar_start=int(times.min()),
        market_calendar_end=int(times.max()),observed_leader_cutoffs=int(shared.leader_observed_board.notna().sum()),
        strict_full_universe_leader_cutoffs=int(shared.leader_full_universe_observed.sum()),
        source_manifest_sha256=digest((SOURCE/'manifest.json').read_bytes()),
        context_manifest_sha256=digest((OUT/'manifest.json').read_bytes()),
        unique_keys=True,finite_positive_stock_close=True,prior_day_context=True,
        model_reader_wide_and_compact_pass=True,candidate_self_reference_excluded=True,
        training_ready=False,http_requests=0,fit_run=False)
    body=json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2).encode()
    from research_budget_v2 import bytes_used,TOTAL_CAP
    if len(body)>1024**2 or bytes_used()+len(body)>TOTAL_CAP:raise ValueError('AUDIT_BUDGET_STOP')
    with (OUT/'readback_acceptance.json').open('xb') as stream:stream.write(body)
    print(body.decode())


if __name__=='__main__':run()
