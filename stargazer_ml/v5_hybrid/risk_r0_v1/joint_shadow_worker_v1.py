"""Desktop child-process entry: no training, HTTP, orders, or notifications."""
import contextlib
import hashlib
import io
import json
import sys
import time

from joint_shadow_compute_v2 import ComputeUnavailable
from joint_shadow_service_v1 import JointShadowService, verify_fingerprints, EXPECTED_FP
from joint_shadow_snapshot_v1 import produce

MAX_REQUEST = 128 * 1024
MAX_RESPONSE = 2 * 1024**2


def selection_scores(candidates):
    """Batch-read daily prefixes once; reuse the existing frozen scoring adapter.

    Scores are parallel research annotations, never replace the rule candidate
    list. Structural qualification/feature failures retain explicit null scores.
    """
    import pandas as pd
    import pyarrow.dataset as ds
    import pyarrow as pa
    import joint_shadow_compute_v2 as compute
    codes = [str(c['code']).zfill(6) for c in candidates]
    P = int(candidates[0]['P'])
    if any(int(c['P']) != P for c in candidates):
        raise ValueError('MIXED_SELECTION_PLAN_DAYS')
    source = ds.dataset(str(compute.DAILY_DIR), format='parquet', exclude_invalid_files=True)
    fields = ['code','date','open','high','low','close','volume','pre_close','is_st','float_mv','amount']
    table = source.to_table(columns=fields, filter=(ds.field('code').isin(codes)) & (ds.field('date') <= P))
    data = table.to_pandas()
    grouped = {str(c): g for c,g in data.groupby('code')}
    empty = pd.DataFrame(columns=fields)
    engine = JointShadowService(enabled=True, daily_loader=lambda code, day: grouped.get(code, empty))
    records = []
    for c in candidates:
        record = engine.post_market_common_candidate(c['code'], c['P'], c['E'], mode='compute')
        g = grouped.get(str(c['code']).zfill(6))
        if g is not None:
            table = pa.Table.from_pandas(g[fields].sort_values('date'), preserve_index=False)
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, table.schema) as stream:
                stream.write_table(table)
            digest = hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()
            record['evidence_refs'].append('daily_prefix_arrow_sha256:' + digest)
        records.append(record)
    return records


def execute(request):
    start = time.monotonic()
    candidates = request.get('candidates')
    if not isinstance(candidates, list) or not 0 < len(candidates) <= 256:
        raise ValueError('CANDIDATE_BATCH_SIZE')
    cutoffs = {int(c['cutoff_ts']) for c in candidates}
    if len(cutoffs) != 1:
        raise ValueError('MIXED_DECISION_CUTOFFS')
    if request.get('mode') not in ('forward_shadow', 'historical_engineering'):
        raise ValueError('EXPLICIT_EXECUTION_MODE_REQUIRED')
    historical = request['mode'] == 'historical_engineering'
    if len({str(c['code']).zfill(6) for c in candidates}) != len(candidates):
        raise ValueError('DUPLICATE_CANDIDATE_CODE')
    mismatch, ok = verify_fingerprints(EXPECTED_FP)
    if not ok:
        raise ComputeUnavailable('RUNTIME_FINGERPRINT_MISMATCH:' + ','.join(mismatch))
    selection_evidence = request.get('selection_evidence')
    if selection_evidence is not None:
        if ({str(r['code']).zfill(6) for r in selection_evidence['rows']} !=
                {str(c['code']).zfill(6) for c in candidates} or
                any(int(c['P']) != int(selection_evidence['P']) or
                    int(c['E']) != int(selection_evidence['E']) for c in candidates)):
            raise ValueError('SELECTION_RISK_KEY_MISMATCH')
    snapshot, audit = produce(next(iter(cutoffs)), historical=historical,
                             engineering_boards=request.get('engineering_boards'))
    engine = JointShadowService(enabled=True)
    universe = {c for m in snapshot.boards.values() for c in m}
    eligible = [c for c in candidates if str(c['code']).zfill(6) in universe]
    scored = engine.same_tick_risk(eligible, mode='prefix', snapshot=snapshot)
    by_code = {r['code']: r for r in scored}
    records = [by_code.get(str(c['code']).zfill(6)) or engine._record(
        str(c['code']).zfill(6), c['P'], c['E'], feature_cutoff=c['cutoff_ts'],
        quality_state='NOT_COMPUTABLE', missing_reasons=['CANDIDATE_MEMBERSHIP_MISSING'],
        evidence_refs=['local_snapshot_producer']) for c in candidates]
    if any(r['quality_state'] in ('RUNTIME_ERROR', 'RUNTIME_FINGERPRINT_MISMATCH') for r in records):
        raise ComputeUnavailable('RISK_PIPELINE_RUNTIME_REFUSAL')
    try:
        selection = selection_scores(candidates)
        selection_state = 'FROZEN_RESEARCH_HEAD_NOT_PROMOTED'
    except Exception as exc:
        # Risk results survive selection-source failure; no unbound score is made up.
        selection = []
        selection_state = 'UNAVAILABLE:' + type(exc).__name__
    scorable = sum(r['risk_amplitude'] is not None for r in records)
    return dict(ok=bool(scorable), status=('NO_SCORABLE_RECORDS' if not scorable else
        'ENGINEERING_REPLAY' if historical else 'SHADOW_OBSERVATION'), scorable_count=scorable,
        schema='JOINT_SHADOW_PACKET_V1', risk_records=records, source_audit=audit,
        selection_evidence=selection_evidence, selection_records=selection,
        selection_state=selection_state, trigger_state='NOT_COMPUTED_NOT_A_BUY_SIGNAL',
        information_state='NOT_CONNECTED',
        decision_authority=False, model_promotion=False, llm_called=False,
        elapsed_seconds=round(time.monotonic()-start, 3),
        discussion_scope='TYPED_EVIDENCE_ONLY_NOT_TRADING_DECISIONS')


def main():
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST + 1)
        if len(raw) > MAX_REQUEST:
            raise ValueError('REQUEST_BUDGET_EXCEEDED')
        # Pure kernels are expected silent; do not let a library diagnostic corrupt
        # the one-JSON worker protocol. The real payload is emitted only below.
        with contextlib.redirect_stdout(io.StringIO()):
            result = execute(json.loads(raw))
    except Exception as exc:
        reason = str(exc) if isinstance(exc, (ComputeUnavailable, ValueError)) else type(exc).__name__
        result = dict(ok=False, status='NOT_COMPUTABLE', reason=reason,
                      risk_records=[], decision_authority=False, llm_called=False)
    body = json.dumps(result, ensure_ascii=False, allow_nan=False).encode('utf-8')
    if len(body) > MAX_RESPONSE:
        body = b'{"ok":false,"status":"OUTPUT_BUDGET_EXCEEDED","risk_records":[]}'
    sys.stdout.buffer.write(body)
    return 0 if result.get('ok') else 2


if __name__ == '__main__':
    raise SystemExit(main())
