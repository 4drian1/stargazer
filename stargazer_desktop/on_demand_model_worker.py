"""One-stock frozen selection inference; stdin/stdout only, no plan writes."""
import json
import gzip
import os
import re
import sys
from pathlib import Path


def infer(request):
    code, p, e = request['code'], int(request['P']), int(request['E'])
    if not re.fullmatch(r'(?:00|30|60|68)\d{4}', code) or not 20200101 <= p < e <= 20991231:
        raise ValueError('INVALID_STOCK_OR_DATE')
    project = Path(os.environ.get('STARGAZER_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
    risk_dir = project / 'stargazer_ml/v5_hybrid/risk_r0_v1'
    sys.path.insert(0, str(risk_dir))
    import joint_shadow_compute_v2 as compute
    from joint_shadow_service_v1 import verify_fingerprints
    expected = json.loads(gzip.decompress((risk_dir / 'live_monitor_fingerprints_20261111_v213_flow.json.gz').read_bytes()))
    mismatch, ok = verify_fingerprints(expected)
    if not ok:
        raise ValueError('RUNTIME_FINGERPRINT_MISMATCH')
    calendar = compute.calendar_dates()
    if e not in calendar:
        if not calendar or calendar[-1] != p or e <= p:
            raise ValueError('DECISION_CALENDAR_NOT_BOUND')
        calendar = [*calendar, e]  # Tomorrow's date is bound by the accepted plan snapshot.
    sys.path.insert(0, str(risk_dir.parent/'green_common_profile'))
    import tarot_live_v2 as live
    import pickle
    import pandas as pd
    g = compute.daily_prefix(compute.read_daily_prefix(code, p), code, p)
    core = live.frozen()
    record = live.latest_record(g, core)
    if record is None:
        row = dict(code=code, P=p, E=e, quality_state='NOT_ADMITTED', selection_score=None,
                   missing_reasons=['CURRENT_A_OR_SECOND_RULE_NOT_PASSED'], selection_model_version=live.VERSION)
    else:
        with live.MODEL.open('rb') as stream:
            model = pickle.load(stream)
        score = float(model.predict(pd.DataFrame([{k:record[k] for k in core.FEATURES}]))[0])
        selected = live.row_for(g, record, core, score, core.sha(live.MODEL))
        row = dict(selected, P=p, E=e, quality_state='OK', selection_model_version=live.VERSION,
                   missing_reasons=[], golden_plan=dict(F618=selected['entry_rule']['f618'],
                   levels=selected['entry_rule']['levels'], plan_dist_atr=selected['distance_atr']))
    return {'code': code, 'P': p, 'E': e, 'selection': row,
            'risk': {'status': 'NOT_COMPUTED',
                     'reason': 'Pyxis需要同截止指数、板块和个股分钟上下文；本次单股日线推理不能替代'},
            'scope': 'FROZEN_SELECTION_HEAD_SINGLE_STOCK_RESEARCH_ONLY',
            'plan_changed': False, 'orders_sent': False}


if __name__ == '__main__':
    try:
        result = infer(json.load(sys.stdin))
    except Exception as exc:
        result = {'status': 'UNAVAILABLE', 'reason': type(exc).__name__ + ':' + str(exc)[:160]}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
