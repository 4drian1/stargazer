"""Optional isolated risk worker. Quotes, notifications and orders never call it.

Import is lightweight: no pandas/LightGBM/LangGraph/LLM imports or worker startup.
Only trusted local config can enable it; a web request cannot change the switch.
"""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import threading

import paths

SHANGHAI = timezone(timedelta(hours=8))
MAX_RESULT = 2 * 1024**2
LOG_CAP = 4 * 1024**2
TIMEOUT = 300
_lock = threading.Lock()
_running = False
_current_key = None
_last = None


def directory():
    return Path(paths.data_file('data/joint_shadow', seed_from_bundle=False))


def _worker_paths():
    app = Path(paths.APP_DIR).resolve()
    project = app.parent.parent if app.name == 'dist' else app.parent
    import os
    python = Path(os.environ.get('STARGAZER_PYTHON') or project / '_dev_venv/Scripts/python.exe')
    return (python,
            project / 'stargazer_ml/v5_hybrid/risk_r0_v1/joint_shadow_worker_v1.py')


def completed_cutoff(now):
    if now.weekday() >= 5:
        return None
    minute = now.hour*60 + now.minute
    grid = [h*60+m for h, m in [(9, 45), (10, 0), (10, 15), (10, 30),
            (10, 45), (11, 0), (11, 15), (11, 30), (13, 15), (13, 30),
            (13, 45), (14, 0), (14, 15), (14, 30), (14, 45), (15, 0)]]
    available = [m for m in grid if m <= minute]
    if not available:
        return None
    m = available[-1]
    return int(now.strftime('%Y%m%d'))*1000000 + (m//60)*10000 + m%60*100


def _read_last():
    file = directory() / 'events.jsonl'
    if not file.exists():
        return None
    if file.stat().st_size > LOG_CAP:
        raise ValueError('SHADOW_LOG_BUDGET_EXCEEDED')
    rows = file.read_text(encoding='utf-8').splitlines()
    return json.loads(rows[-1]) if rows else None


def state(cfg):
    options = cfg.get('joint_shadow') or {}
    with _lock:
        try:
            last = _last or _read_last()
            return dict(enabled=options.get('enabled') is True, running=_running,
                status=('RUNNING' if _running else 'DISABLED' if options.get('enabled') is not True
                        else (last or {}).get('result', {}).get('status', 'NOT_STARTED')),
                last=last, interaction_enabled=False, trading_authority=False,
                result_is_historical=(last or {}).get('request_mode') == 'historical_engineering')
        except (OSError, ValueError) as exc:
            return dict(enabled=False, running=_running, status='LOG_ERROR', error=str(exc),
                        interaction_enabled=False, trading_authority=False)


def _persist(event):
    body = (json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    root = directory()
    file = root / 'events.jsonl'
    if len(body) > MAX_RESULT or (file.stat().st_size if file.exists() else 0) + len(body) > LOG_CAP:
        raise ValueError('SHADOW_LOG_BUDGET_EXCEEDED')
    root.mkdir(parents=True, exist_ok=True)
    with file.open('ab') as stream:
        stream.write(body)


def run(cfg, plan, *, now=None):
    global _running, _current_key, _last
    options = cfg.get('joint_shadow') or {}
    if options.get('enabled') is not True:
        return {'ok': False, 'status': 'DISABLED'}
    now = now or datetime.now(SHANGHAI)
    mode = options.get('mode', 'forward_shadow')
    if mode not in ('forward_shadow', 'historical_engineering'):
        raise ValueError('UNKNOWN_SHADOW_MODE')
    cutoff = options.get('engineering_cutoff') if mode == 'historical_engineering' else completed_cutoff(now)
    if cutoff is None:
        return {'ok': False, 'status': 'NO_COMPLETED_15M_BAR'}
    if not isinstance(plan, dict) or not plan.get('plan_ready'):
        return {'ok': False, 'status': 'PLAN_NOT_READY'}
    P, E = int(plan['data_date']), int(plan['monitor_date'])
    if E != int(cutoff)//1000000:
        return {'ok': False, 'status': 'PLAN_DAY_MISMATCH'}
    codes = [str(r['code']).zfill(6) for r in plan['rows']]
    if not codes or len(codes) > 256 or len(set(codes)) != len(codes):
        raise ValueError('CANDIDATE_BATCH_SIZE_OR_DUPLICATES')
    if any(len(c) != 6 or not c.isdigit() for c in codes):
        raise ValueError('INVALID_CANDIDATE_CODE')
    payload = dict(mode=mode, candidates=[dict(code=c, P=P, E=E, cutoff_ts=int(cutoff)) for c in codes])
    selection = dict(source_schema=plan.get('schema'), P=P, E=E,
        snapshot_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False,
                                                 allow_nan=False).encode('utf-8')).hexdigest(),
        rows=[{k:r.get(k) for k in ('code','setup','reference_price','zone','rank','signal')}
              for r in plan['rows']], scope='ORIGINAL_RULE_OBSERVATION_NOT_PROMOTED_MODEL')
    payload['selection_evidence'] = selection
    if mode == 'historical_engineering' and options.get('engineering_boards'):
        payload['engineering_boards'] = options['engineering_boards']
    raw = json.dumps(payload, sort_keys=True).encode('utf-8')
    if len(raw) > 128*1024:
        raise ValueError('SHADOW_REQUEST_BUDGET_EXCEEDED')
    key = hashlib.sha256(raw).hexdigest()
    python, worker = _worker_paths()
    if not python.is_file() or not worker.is_file():
        return {'ok': False, 'status': 'WORKER_ENVIRONMENT_UNAVAILABLE'}
    with _lock:
        if _running:
            return {'ok': False, 'status': 'ALREADY_RUNNING', 'job_key': _current_key}
        last = _last or _read_last()
        if (last and last['job_key'] == key and last['result'].get('ok')
                and not last.get('persistence_error')):
            return {'ok': True, 'status': 'ALREADY_RECORDED', 'job_key': key}
        _running, _current_key = True, key

    def work():
        global _running, _current_key, _last
        event = dict(job_key=key, request_mode=mode, started_at=now.isoformat(), result=None)
        try:
            proc = subprocess.run([str(python), '-B', '-X', 'utf8', str(worker)],
                input=raw, capture_output=True, timeout=TIMEOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if len(proc.stdout) > MAX_RESULT:
                raise ValueError('SHADOW_OUTPUT_BUDGET_EXCEEDED')
            value = json.loads(proc.stdout, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
            if not isinstance(value, dict) or not isinstance(value.get('ok'), bool):
                raise ValueError('INVALID_WORKER_PROTOCOL')
            if proc.returncode == 0 and not value['ok'] or proc.returncode != 0 and value['ok']:
                raise ValueError('WORKER_EXIT_STATUS_CONFLICT')
            if value['ok']:
                records = value.get('risk_records')
                if (value.get('schema') != 'JOINT_SHADOW_PACKET_V1' or not isinstance(records, list)
                        or len(records) != len(codes) or {r.get('code') for r in records} != set(codes)
                        or any((r.get('P'), r.get('E'), r.get('feature_cutoff'), r.get('risk_horizon'))
                               != (P,E,int(cutoff),120) for r in records)):
                    raise ValueError('WORKER_RESULT_KEY_OR_HORIZON_MISMATCH')
                if value.get('decision_authority') is not False or value.get('llm_called') is not False:
                    raise ValueError('WORKER_AUTHORITY_VIOLATION')
            event['result'] = value
        except subprocess.TimeoutExpired:
            event['result'] = dict(ok=False, status='TIMEOUT', risk_records=[])
        except Exception as exc:
            event['result'] = dict(ok=False, status='WORKER_ERROR', reason=type(exc).__name__, risk_records=[])
        event['finished_at'] = datetime.now(SHANGHAI).isoformat()
        event['valid_for_automatic_decision'] = False
        with _lock:
            try:
                _persist(event)
            except (OSError, ValueError) as exc:
                event['persistence_error'] = type(exc).__name__
            _last = event
            _running, _current_key = False, None

    try:
        threading.Thread(target=work, daemon=True, name='joint-risk-shadow-worker').start()
    except RuntimeError:
        with _lock:
            _running, _current_key = False, None
        return {'ok': False, 'status': 'WORKER_START_ERROR'}
    return {'ok': True, 'status': 'QUEUED', 'job_key': key}
