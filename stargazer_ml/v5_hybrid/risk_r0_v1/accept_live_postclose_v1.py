"""Independent readback before exposing new-day risk to the plan discussion."""
import argparse
from dataclasses import asdict
import io
import json
import gzip
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from live_postclose_source_v1 import atomic_json, digest, DISK_CAP
from joint_shadow_snapshot_v1 import HERE, now_stamp

sys.path.insert(0,str(HERE.parents[2]/'stargazer_desktop'))


def price_matches(reference, actual):
    try:
        a,b=float(reference),float(actual)
    except (ValueError,TypeError):
        return False
    return bool(np.isfinite([a,b]).all() and min(a,b)>0 and abs(a-b)<=.010000000001)


def plan_calendar(snapshot):
    refs=[r for r in snapshot.get('inputs',[]) if Path(r.get('path','')).name=='calendar.json']
    if len(refs)!=1:
        raise ValueError('PLAN_CALENDAR_REFERENCE_MISSING_OR_AMBIGUOUS')
    reference=refs[0]
    path=Path(reference['path'])
    if path.stat().st_size>1024**2:
        raise ValueError('PLAN_CALENDAR_SIZE')
    raw=path.read_bytes()
    if digest(raw)!=reference['sha256']:
        raise ValueError('PLAN_CALENDAR_HASH_MISMATCH')
    days=sorted({int(d) for d in json.loads(raw)})
    P,E=int(snapshot['data_date']),int(snapshot['monitor_date'])
    if P not in days or days.index(P)+1>=len(days) or days[days.index(P)+1]!=E:
        raise ValueError('PLAN_NEXT_TRADING_DAY_MISMATCH')
    return days,reference


def accept(runtime, refresh_selection=False):
    import rebuild_features_v1 as frozen
    from joint_shadow_service_v1 import JointShadowService,verify_fingerprints
    from oracles_v1 import EvidencePack
    from jev_adapter_v1 import _risk_evidence_incomplete
    state=json.loads((runtime/'state.json').read_text(encoding='utf-8'))
    if state.get('acceptance_status')=='PASS' and not refresh_selection:
        return state
    if state.get('status')!='COMPLETE':
        raise ValueError('RISK_JOB_NOT_COMPLETE')
    run=Path(state['run_dir'])
    raw=(run/'input_snapshot.json').read_bytes()
    if digest(raw)!=state['snapshot_sha256']:
        raise ValueError('INPUT_SNAPSHOT_HASH_MISMATCH')
    snapshot=json.loads(raw)
    # Full v8 manifest is losslessly compressed; v7 remains immutable history.
    expected=json.loads(gzip.decompress((HERE/'live_monitor_fingerprints_20261026_v33_flow.json.gz').read_bytes()))
    mismatch,ok=verify_fingerprints(expected)
    if not ok:
        raise ValueError('RUNTIME_FINGERPRINT_MISMATCH:'+','.join(mismatch))
    frame=pd.read_parquet(run/'source_features.parquet')
    features={232:pd.read_parquet(run/'inputs_232.parquet'),
              266:pd.read_parquet(run/'inputs_266.parquet')}
    _,f232,f266,_=frozen.load_spec()
    assert list(features[232])==list(f232) and list(features[266])==list(f266)
    if len(frame)!=len(features[232]) or frame.code.duplicated().any():
        raise ValueError('FEATURE_KEY_SHAPE')
    prediction={size:frozen.load_booster(folder).predict(frozen.native_missing_view(features[size],list(features[size])))
                for size,folder in [(232,frozen.MODEL_232),(266,frozen.MODEL_266)]}
    source=(run/'packs.jsonl').read_bytes()
    original=json.loads((run/'terminal.json').read_text(encoding='utf-8'))
    if digest(source)!=original['packs_sha256']:
        raise ValueError('RAW_PACK_HASH_MISMATCH')
    packs=[EvidencePack(**json.loads(line)).bind() for line in source.splitlines()]
    source_rows={r['code']:r for r in snapshot['rows']}
    index={str(c):i for i,c in enumerate(frame.code)}
    calendar,calendar_ref=plan_calendar(snapshot)
    engine=JointShadowService(enabled=True,fp_expected=expected,calendar=calendar)
    accepted,records=[],[]
    for pack in packs:
        code=pack.candidates[0]['code']
        i=index[code]
        reasons=[]
        if pack.asof_ts!=str(snapshot['data_date'])+'150000':
            reasons.append('CUTOFF_MISMATCH')
        for size in (232,266):
            p=float(np.clip(prediction[size][i],0,1))
            actual=pack.risk_context[f'amp_{size}']
            if not np.isfinite(p) or abs(p-actual)>1e-12:
                reasons.append(f'REPREDICT_MISMATCH_{size}')
        raw_close=frame.iloc[i].get('stock_close')
        if not price_matches(source_rows[code].get('reference_price'),raw_close):
            reasons.append('P_CLOSE_CROSS_SOURCE_SCALE_MISMATCH')
        rc=pack.risk_context
        rc['stock_close_raw']=float(raw_close) if np.isfinite(raw_close) else None
        rc['reference_close_parts']=source_rows[code].get('reference_price')
        rc['cross_source_close_check']='PASS' if not any('SCALE' in r for r in reasons) else 'FAIL'
        rc['snapshot_sha256']=state['snapshot_sha256']
        from live_postclose_explain_v1 import attach_model_signals
        attach_model_signals(pack,features[232].iloc[i],features[266].iloc[i])
        # Reuse the frozen selection adapter with the new dependency reference,
        # not its older desktop-server fingerprint. Failure does not invent a score.
        live=source_rows[code]
        if live.get('selection_model'):
            selection=dict(selection_score=live['selection_score'],quality_state='USER_AUTHORIZED_ENABLE',
                           missing_reasons=[],golden_plan=dict(F618=live['entry_rule']['f618'],plan_dist_atr=live['distance_atr']))
        else:
            selection=engine.post_market_common_candidate(code,int(snapshot['data_date']),int(snapshot['monitor_date']),mode='compute')
        candidate=pack.candidates[0]
        candidate['p_close_score']=selection.get('selection_score')
        gp=selection.get('golden_plan') or {}
        candidate.update(f618=gp.get('F618'),plan_dist_atr=gp.get('plan_dist_atr'))
        rc['selection_quality_state']=selection.get('quality_state')
        rc['selection_missing_reasons']=selection.get('missing_reasons')
        rc['selection_calendar_reference']=calendar_ref
        reasons+=_risk_evidence_incomplete(pack)
        records.append(dict(code=code,status='AVAILABLE' if not reasons else 'REJECTED',
                            reasons=reasons,reprediction_delta_max=0. if not any('REPREDICT' in r for r in reasons) else None))
        if not reasons:
            accepted.append(asdict(pack))
    seen={r['code'] for r in records}
    records += [r for r in state.get('rows',[]) if r['code'] not in seen]
    body=b''.join((json.dumps(p,ensure_ascii=False,allow_nan=False)+'\n').encode() for p in accepted)
    pack_name='accepted_packs_v2.jsonl' if refresh_selection else 'accepted_packs.jsonl'
    report_name='independent_acceptance_v2.json' if refresh_selection else 'independent_acceptance.json'
    final={**state,'acceptance_status':'PASS','available_count':len(accepted),'rows':records,
        'packs_path':str(run/pack_name),'packs_sha256':digest(body),
        'acceptance_path':str(run/report_name),
        'acceptance_completed_at':now_stamp(),
        'acceptance_scope':'INPUTS_AND_INFERENCE_ENGINEERING_NOT_CAPABILITY_CERTIFICATION',
        'checker_sha256':digest(Path(__file__).read_bytes())}
    report=json.dumps(final,ensure_ascii=False,allow_nan=False,indent=1).encode()
    used=sum(p.stat().st_size for p in runtime.rglob('*') if p.is_file())
    from live_postclose_source_v1 import disk_limit
    if used+len(body)+len(report)+1024**2>disk_limit(runtime):
        raise ValueError('RUNTIME_DISK_BUDGET')
    for path,data in [(run/pack_name,body),(run/report_name,report)]:
        with path.open('xb') as stream:
            stream.write(data)
    atomic_json(runtime/'state.json',final)
    print(json.dumps(dict(status='PASS',available=len(accepted),population=len(source_rows),rows=records),ensure_ascii=False))
    return final


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--refresh-selection',action='store_true')
    args=parser.parse_args()
    accept(args.runtime,refresh_selection=args.refresh_selection)
