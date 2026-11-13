"""Bounded same-cutoff Pyxis worker; shared runtime/quotas, no paid calls or orders."""
import os
for _thread_env in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_thread_env]='1'
import contextlib
import io
import json
import gzip
from pathlib import Path
import sys
import traceback

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'stargazer_desktop'))
from live_postclose_source_v1 import atomic_json,now_stamp,disk_limit
from live_postclose_job_v1 import pack_from_row


def failure_details(error, stage):
    value = dict(error_type=type(error).__name__, stage=stage,
                 reason=type(error).__name__+':'+str(error)[:500],
                 traceback=traceback.format_exc()[-8000:])
    try:
        from tdx_native_kernel_v1 import peak_memory
        value['peak_bytes'] = peak_memory()
    except OSError:
        pass
    return value


def execute(body):
    from joint_shadow_service_v1 import verify_fingerprints
    expected=json.loads(gzip.decompress((HERE/'live_monitor_fingerprints_20261113_v227_flow.json.gz').read_bytes()))
    mismatch,ok=verify_fingerprints(expected)
    if not ok:raise ValueError('RUNTIME_FINGERPRINT_MISMATCH:'+','.join(mismatch))
    cutoff=int(body['cutoff']);day=cutoff//1000000
    if day!=now_stamp()//1000000 or cutoff>now_stamp():raise ValueError('CURRENT_DAY_PREFIX_REQUIRED')
    attempt=int(body.get('attempt') or 0)
    plan=body['plan'];positions=body.get('positions') or []
    if plan.get('rule_version')!='TAROT_A_SECOND_CASH_V2' or int(plan['monitor_date'])>day:
        raise ValueError('PLAN_VERSION_OR_DAY_MISMATCH')
    active_rows=plan['rows'] if int(plan['monitor_date'])==day else []
    codes=sorted({r['code'] for r in active_rows}|{p['code'] for p in positions})
    if not codes or len(codes)>128:raise ValueError('PLAN_AND_POSITION_BATCH_LIMIT')
    root=Path(body['runtime']).resolve()
    from shared_risk_context_v1 import kernel
    import hashlib
    population=hashlib.sha256(json.dumps(codes).encode()).hexdigest()[:12]
    suffix='_r%d' % attempt if attempt else ''
    folder=root/f'run_{day}_{cutoff%1000000}_intraday_native5m_{kernel()[:16]}_{population}{suffix}'
    result_file=folder/'intraday_result.json'
    if result_file.exists():
        cached=json.loads(result_file.read_bytes())
        if cached.get('requested_codes')!=codes:raise ValueError('RISK_TICK_POPULATION_CHANGED')
        return cached
    if folder.exists():raise ValueError('NO_AUTOMATIC_RETRY_OF_TERMINATED_RISK_TICK')
    from runtime_disk_limit_v1 import disk_usage
    used=disk_usage(root)
    if used+2*1024**2>disk_limit(root):raise ValueError('RUNTIME_DISK_BUDGET')
    folder.mkdir(parents=True)
    stage = 'SOURCE_PREPARE'
    def progress(**state):
        nonlocal stage
        stage = state.get('stage', stage)
        atomic_json(folder/'progress.json',dict(cutoff=cutoff,**state))
    try:
        from intraday_public_context_v1 import produce
        snapshot,feature_cache,capture,failures=produce(root,codes,cutoff,folder,progress)
        from tdx_native_kernel_v1 import memory_guard
        memory_guard()
        from joint_shadow_context_v1 import build_features
        import rebuild_features_v1 as frozen
        requests=[dict(code=c,P=snapshot.days[-2],E=day,cutoff_ts=cutoff) for c in codes
            if c in snapshot.stock_minutes and any(c in m for m in snapshot.boards.values())]
        if not requests:raise ValueError('NO_CURRENT_CANDIDATE_CONTEXT')
        progress(stage='BUILD_FEATURES', done=0, total=len(requests))
        frame,x232,x266,_=build_features(snapshot,requests,feature_cache=feature_cache)
        memory_guard()
        capture.save('feature_cache_audit.json',json.dumps(feature_cache.audit).encode())
        progress(stage='PREDICT', done=0, total=len(requests))
        predictions=[np.clip(frozen.load_booster(m).predict(frozen.native_missing_view(x,list(x)),num_threads=1),0,1)
            for m,x in [(frozen.MODEL_232,x232),(frozen.MODEL_266,x266)]]
        required=['stock_day_return_raw','context_sector_mean_return_loo_raw','sse_15m_day_return_prev_close',
            'cyb_15m_day_return_prev_close','new_sse_golden_position']
        bycode={r['code']:r for r in plan['rows']};records={}
        for i,request in enumerate(requests):
            code=request['code'];row=frame.iloc[i]
            if not row[required].notna().all():
                records[code]=dict(status='EVIDENCE_INSUFFICIENT',reason='REQUIRED_MODEL_INPUT_MISSING');continue
            selection=bycode.get(code,dict(code=code,atr_ref=None,distance_atr=None))
            missing={str(n):list(x.columns[x.iloc[i].isna()]) for n,x in [(232,x232),(266,x266)]}
            pack=pack_from_row(row,selection,[float(v[i]) for v in predictions],snapshot,missing,str(folder))
            from jev_adapter_v1 import _risk_evidence_incomplete
            reasons=_risk_evidence_incomplete(pack)
            records[code]=dict(status='AVAILABLE' if not reasons else 'EVIDENCE_INSUFFICIENT',
                reasons=reasons,context=pack.risk_context,asof_ts=cutoff,trading_days=list(map(int,snapshot.days)))
        for code in codes:
            records.setdefault(code,dict(status='EVIDENCE_INSUFFICIENT',reason=failures.get(code,'CURRENT_CONTEXT_UNBOUND')))
        value=dict(cutoff=cutoff,records=records,requested_codes=codes,source_membership='ARCHIVED_OBSERVED_NOW_NOT_HISTORICAL_PIT',
            completed_at=now_stamp(),automatic_trading=False,llm_called=False)
        capture.save('intraday_result.json',json.dumps(value,ensure_ascii=False,allow_nan=False).encode())
        return value
    except Exception as error:
        if isinstance(error, ValueError) and str(error).startswith('LIVE_REQUIRED_PREFIX_MISSING:'):
            missing = str(error).split(':', 1)[1].split(',')
            result = dict(status='SOURCE_PENDING', cutoff=cutoff, requested_codes=codes,
                          reason='LIVE_REQUIRED_PREFIX_MISSING', missing_codes=missing)
            atomic_json(result_file, result)
            return result
        atomic_json(folder/'terminal.json',dict(status='FAILED',automatic_resume=False,**failure_details(error,stage)))
        raise


if __name__=='__main__':
    try:
        with contextlib.redirect_stdout(io.StringIO()):result=execute(json.load(sys.stdin))
    except Exception as error:
        result=dict(status='EVIDENCE_INSUFFICIENT',reason=type(error).__name__+':'+str(error)[:200])
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
