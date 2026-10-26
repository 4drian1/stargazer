"""Bounded same-cutoff Pyxis worker; shared runtime/quotas, no paid calls or orders."""
import contextlib
import io
import json
import gzip
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'stargazer_desktop'))
from live_postclose_source_v1 import produce_live,atomic_json,now_stamp,disk_limit
from live_postclose_job_v1 import pack_from_row


def execute(body):
    from joint_shadow_service_v1 import verify_fingerprints
    expected=json.loads(gzip.decompress((HERE/'live_monitor_fingerprints_20261026_v26_flow.json.gz').read_bytes()))
    mismatch,ok=verify_fingerprints(expected)
    if not ok:raise ValueError('RUNTIME_FINGERPRINT_MISMATCH:'+','.join(mismatch))
    cutoff=int(body['cutoff']);day=cutoff//1000000
    if day!=now_stamp()//1000000 or cutoff>now_stamp():raise ValueError('CURRENT_DAY_PREFIX_REQUIRED')
    plan=body['plan'];positions=body.get('positions') or []
    if plan.get('rule_version')!='TAROT_A_SECOND_CASH_V2' or int(plan['monitor_date'])>day:
        raise ValueError('PLAN_VERSION_OR_DAY_MISMATCH')
    active_rows=plan['rows'] if int(plan['monitor_date'])==day else []
    codes=sorted({r['code'] for r in active_rows}|{p['code'] for p in positions})
    if not codes or len(codes)>128:raise ValueError('PLAN_AND_POSITION_BATCH_LIMIT')
    root=Path(body['runtime']).resolve()
    folder=root/f'run_{day}_{cutoff%1000000}_intraday_v2'
    result_file=folder/'intraday_result.json'
    if result_file.exists():
        cached=json.loads(result_file.read_bytes())
        if cached.get('requested_codes')!=codes:raise ValueError('RISK_TICK_POPULATION_CHANGED')
        return cached
    if folder.exists():raise ValueError('NO_AUTOMATIC_RETRY_OF_TERMINATED_RISK_TICK')
    used=sum(x.stat().st_size for x in root.rglob('*') if x.is_file())
    if used+2*1024**2>disk_limit(root):raise ValueError('RUNTIME_DISK_BUDGET')
    folder.mkdir(parents=True)
    def progress(**state):
        atomic_json(folder/'progress.json',dict(cutoff=cutoff,**state))
    try:
        snapshot,capture,failures=produce_live(day,codes,folder,progress,cutoff=cutoff)
        from joint_shadow_context_v1 import build_features
        import rebuild_features_v1 as frozen
        requests=[dict(code=c,P=snapshot.days[-2],E=day,cutoff_ts=cutoff) for c in codes
            if c in snapshot.stock_minutes and any(c in m for m in snapshot.boards.values())]
        if not requests:raise ValueError('NO_CURRENT_CANDIDATE_CONTEXT')
        frame,x232,x266,_=build_features(snapshot,requests)
        predictions=[np.clip(frozen.load_booster(m).predict(frozen.native_missing_view(x,list(x))),0,1)
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
        atomic_json(folder/'terminal.json',dict(status='FAILED',automatic_resume=False,reason=type(error).__name__))
        raise


if __name__=='__main__':
    try:
        with contextlib.redirect_stdout(io.StringIO()):result=execute(json.load(sys.stdin))
    except Exception as error:
        result=dict(status='EVIDENCE_INSUFFICIENT',reason=str(error)[:200] if isinstance(error,ValueError) else type(error).__name__)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
