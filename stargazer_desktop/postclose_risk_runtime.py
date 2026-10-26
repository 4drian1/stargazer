"""Start the local post-close worker and expose only snapshot-bound evidence."""
from dataclasses import fields
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import threading

import paths
import index_minutes_v1  # mirror location + runtime fingerprint for the post-close gate

_thread = None
_cache = {}


def directory():
    return Path(paths.data_file('data/postclose_risk' if paths.FROZEN else
                               'dist/data/postclose_risk', seed_from_bundle=False))


def _load_sync_updater():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'tdx_source_updater',
        index_minutes_v1.MIRROR / 'index_minutes_updater_v1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_expected_sync_fp = None


def _expected_sync_fingerprint():
    global _expected_sync_fp
    if _expected_sync_fp is None:
        try:
            _expected_sync_fp = _load_sync_updater().get_runtime_fingerprint()
        except Exception:
            _expected_sync_fp = None
    return _expected_sync_fp


def _read_release_status():
    p = index_minutes_v1.MIRROR / 'release_status.json'
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _read_mirror_fingerprint():
    p = index_minutes_v1.MIRROR / 'runtime_fingerprint.json'
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8')).get('runtime_fingerprint')
    except (OSError, ValueError):
        return None


def _should_launch_postclose(snapshot, now, index, release_status,
                            mirror_fingerprint, expected_fingerprint):
    """Post-close launch gate.

    Requires the sync contract to prove BOTH indices closed the CURRENT day with a
    complete 48-bar + 15:00 mirror, and that the mirror was built by the expected
    prefix-aware updater version (corresponding mirror version). A prior day's
    release verdict or a stale/legacy mirror must never launch today's worker.
    """
    try:
        day = int(snapshot['data_date'])
    except (KeyError, TypeError, ValueError):
        return False
    if not (day == int(now.strftime('%Y%m%d')) and now.hour >= 15):
        return False
    if index.get('stale'):
        return False
    if not release_status or not mirror_fingerprint or not expected_fingerprint:
        return False
    if mirror_fingerprint != expected_fingerprint:
        return False  # mirror built by a different (legacy) sync contract
    if release_status.get('as_of_date') != day:
        return False  # release verdict belongs to a different observation day
    if not release_status.get('evidence_complete'):
        return False  # not both indices released the current day
    symbols = release_status.get('symbols', {})
    if len(symbols) < 2:
        return False
    mirror_root = index_minutes_v1.MIRROR
    for sym, rec in symbols.items():
        # Business qualification (release) is the authority; the processing verdict
        # (e.g. UNCHANGED) is NOT required to equal POSTCLOSE_OK, because a second
        # sync on the same day + data version keeps the previously verified release.
        if not rec.get('release'):
            return False
        # Bind the mirror DATA, not just the contract version string: the release
        # verdict must match the actual parquet bytes currently on disk, so a mirror
        # rebuilt/tampered by a different contract cannot launch today's worker.
        expected_sha = rec.get('mirror_sha')
        mpath = rec.get('mirror_path')
        actual_path = Path(mpath) if (mpath and Path(mpath).is_absolute()) else (mirror_root / mpath if mpath else None)
        if not expected_sha or actual_path is None or not actual_path.exists():
            return False
        try:
            actual_sha = hashlib.sha256(actual_path.read_bytes()).hexdigest()
        except OSError:
            return False
        if actual_sha != expected_sha:
            return False
    return True


def snapshot_digest(snapshot):
    # Byte binding is recorded by the worker; compare decoded content to handle
    # benign JSON formatting changes without accepting a different candidate list.
    return hashlib.sha256(json.dumps(snapshot,sort_keys=True,ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def state():
    file = directory()/'state.json'
    if not file.exists():
        return {'status':'NOT_STARTED','reason':'等待当日收盘名单和指数数据'}
    try:
        value=json.loads(file.read_text(encoding='utf-8'))
        value['automatic_trading']=False
        return value
    except (OSError,ValueError):
        return {'status':'STATE_ERROR','reason':'风险任务状态文件不可读'}


def find_pack(code,snapshot):
    activity=state()
    try:
        value=accepted_for(snapshot,activity)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return None,'风险证据绑定失败：'+str(exc)
    if value is None:
        value=activity
    else:
        return _pack_from_accepted(code,snapshot,value)
    if str(value.get('data_date')) != str(snapshot.get('data_date')):
        return None,None
    if value.get('status') != 'COMPLETE':
        return None,'风险证据后台任务：'+str(value.get('stage') or value.get('reason') or value['status'])
    if value.get('acceptance_status')!='PASS':
        return None,'风险评分已生成，等待独立读回验收'
    return _pack_from_accepted(code,snapshot,value)


def _pack_from_accepted(code,snapshot,value):
    from oracles_v1 import EvidencePack
    try:
        if not snapshot_bound(value, snapshot):
            raise ValueError('CANDIDATE_SNAPSHOT_CHANGED')
        path=Path(value['packs_path'])
        if path.resolve().parent != Path(value['run_dir']).resolve() or path.stat().st_size>2*1024**2:
            raise ValueError('PACK_PATH_OR_SIZE')
        key=(str(path),path.stat().st_mtime_ns,value['packs_sha256'])
        if key not in _cache:
            body=path.read_bytes()
            if hashlib.sha256(body).hexdigest()!=value['packs_sha256']:
                raise ValueError('PACK_HASH_MISMATCH')
            allowed={f.name for f in fields(EvidencePack)}
            index={}
            for line in body.splitlines():
                row=json.loads(line)
                pack=EvidencePack(**{k:v for k,v in row.items() if k in allowed}).bind()
                if pack.market_date!=str(snapshot['data_date']) or pack.asof_ts!=str(snapshot['data_date'])+'150000':
                    raise ValueError('PACK_CUTOFF_MISMATCH')
                for candidate in pack.candidates:
                    index[candidate['code']]=pack
            _cache.clear()
            _cache[key]=index
        if code in _cache[key]:
            return _cache[key][code],None
        row=next((r for r in value.get('rows',[]) if r.get('code')==code),{})
        return None,' / '.join(row.get('reasons') or ['当前股票风险证据未通过'])
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return None,'风险证据绑定失败：'+str(exc)


def snapshot_bound(value, snapshot):
    raw = (Path(value['run_dir'])/'input_snapshot.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != value['snapshot_sha256']:
        raise ValueError('SNAPSHOT_HASH_MISMATCH')
    parent = json.loads(raw)
    if same_snapshot_data(parent, snapshot):
        return True
    from golden_rules import is_projection_of
    if is_projection_of(snapshot,parent):
        return True
    # Projection copied its parent's generated timestamp. Prove that this alone
    # changed the parent hash, without ignoring prices, provenance or candidate facts.
    if 'created_at' in parent and 'created_at' in snapshot:
        refreshed={**parent,'created_at':snapshot['created_at']}
        if is_projection_of(snapshot,refreshed):
            return True
    return risk_inputs_equivalent(parent, snapshot)


def risk_inputs_equivalent(parent, current):
    """Reuse frozen amplitudes when only the independent execution map changed."""
    if not (parent.get('rule_version') == current.get('rule_version') == 'TAROT_A_SECOND_CASH_V2'
            and parent.get('inputs') and parent.get('rule_inputs') and current.get('rule_inputs')):
        return False
    ignored = {'created_at', 'rows', 'rule_inputs', 'assistant_plan_id', 'boards', 'manual_observations',
               'manual_review_rows','manual_review_population_count','manual_review_scope'}
    if {k:v for k,v in parent.items() if k not in ignored} != {k:v for k,v in current.items() if k not in ignored}:
        return False
    app = Path(paths.APP_DIR).resolve()
    project = app.parent.parent if app.name == 'dist' else app.parent
    # These wrappers/display contracts do not enter the frozen 232/266 matrices.
    mutable = {str((project/'stargazer_desktop'/name).resolve()) for name in (
        'server.py','assistant_workflow.py','golden_monitor.py','index_minutes_v1.py',
        'pentacles_monitor.py','plan_evidence.py','postclose_risk_runtime.py',
        'tarot_monitor_v2.py','trade_facts_v1.py','deepseek_api.py','pentacles_demo.py','plan_board_monitor.py',
        'golden_stock_analysis.py','on_demand_model_worker.py','agent/brain.py',
        'web/app.js','web/pentacles.js','test_tarot_selector_binding_v1.py','Stargazer_demo_safe.spec')}
    base = project/'stargazer_ml/v5_hybrid'
    mutable.update(str((base/'risk_r0_v1'/name).resolve()) for name in (
        'tarot_intraday_risk_v2.py','live_postclose_job_v1.py','accept_live_postclose_v1.py'))
    mutable.update(str((base/'green_common_profile'/name).resolve()) for name in (
        'tarot_live_v2.py','golden_ladder_v3.py','golden_resonance_v4.py','tarot_live_signal_v2.py','gcp_golden_intraday_v1.py','tarot_manual_review_v1.py'))
    def immutable(snapshot):
        items = snapshot['rule_inputs']
        def absolute(value):
            path=Path(value)
            return str((path if path.is_absolute() else project/path).resolve())
        entries = {absolute(r['path']):r['sha256'] for r in items}
        if len(entries) != len(items):
            raise ValueError('DUPLICATE_RULE_INPUT')
        return {k:v for k,v in entries.items() if k not in mutable}
    if immutable(parent) != immutable(current):
        return False
    original = {r['code']:r for r in parent['rows']}
    rows = current.get('rows', [])
    if not rows or len({r['code'] for r in rows}) != len(rows):
        return False
    observation = {'zone','distance_atr','entry_rule','observation_map_status','signal'}
    for row in rows:
        old = original.get(row['code'], {})
        if not old.get('feature_values') or not old.get('selection_model'):
            return False
        compare_fields=observation|({'status'} if old.get('status') in ('GOLDEN_PLAN_READY','WAIT_RESONANCE_ZONE') and row.get('status') in ('GOLDEN_PLAN_READY','WAIT_RESONANCE_ZONE') else set())
        if {k:v for k,v in old.items() if k not in compare_fields} != {k:v for k,v in row.items() if k not in compare_fields}:
            return False
        before, after = old.get('entry_rule', {}), row.get('entry_rule', {})
        if after.get('policy') not in ('GOLDEN_LADDER_STOP_V3','GOLDEN_RESONANCE_CROSSDAY_V4'):
            return False
        if any(before.get(k) != after.get(k) for k in ('version','eligible','atr_ref','stage','close_p')):
            return False
        if before.get('model_reference_node', before.get('f618')) != after.get('model_reference_node'):
            return False
    return True


def same_snapshot_data(a,b):
    left,right=dict(a),dict(b)
    left.pop('created_at',None);right.pop('created_at',None)
    if left.get('rule_version') and left.get('rule_version')==right.get('rule_version'):
        # A projection's parent hash also contains its generated timestamp.
        # Every actual input, rule, admitted/excluded fact remains in the comparison.
        left.pop('projection_parent_hash',None);right.pop('projection_parent_hash',None)
    return snapshot_digest(left)==snapshot_digest(right)


def accepted_for(snapshot, activity=None):
    """Published immutable acceptance is independent of an in-flight job status."""
    activity=state() if activity is None else activity
    if activity.get('status')=='COMPLETE' and activity.get('acceptance_status')=='PASS':
        if snapshot_bound(activity,snapshot):
            return activity
    root=directory().resolve()
    reports=sorted(root.glob('run_*/independent_acceptance*.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True)
    for path in reports:
        if path.stat().st_size>2*1024**2:
            continue
        record=json.loads(path.read_bytes())
        if (record.get('status')!='COMPLETE' or record.get('acceptance_status')!='PASS'
                or str(record.get('data_date'))!=str(snapshot.get('data_date'))):
            continue
        if Path(record['run_dir']).resolve()!=path.resolve().parent:
            raise ValueError('ACCEPTANCE_PATH_MISMATCH')
        if snapshot_bound(record,snapshot):
            return record
    return None


def needs_rebuild(snapshot, activity=None):
    """Repeated daily selection is not authorization to restart a terminated batch."""
    activity=state() if activity is None else activity
    if not snapshot.get('rows') or accepted_for(snapshot,activity) is not None:
        return False
    if activity.get('status')=='RUNNING':
        return False
    if activity.get('run_dir') and snapshot_bound(activity,snapshot):
        return False
    return True


def _worker_paths_real():
    from joint_risk_bridge import _worker_paths
    return _worker_paths()


def _snapshot_path():
    import pentacles_demo
    return pentacles_demo.snapshot_path()


def _decide_and_launch(snapshot, now, snapshot_file=None, *, _state=None,
                       _needs_rebuild=None, _subprocess_run=None, _worker_paths=None):
    """Decide + perform the post-close worker (re)launch.

    Encapsulates the scheduling side effect (a real subprocess call) so the launch
    behaviour can be exercised with stubs. Returns a short status string:
    'gate_closed' | 'no_action' | 'launched_worker' | 'launched_checker'.
    """
    index = index_minutes_v1.get_index_freshness()
    release_status = _read_release_status()
    mirror_fp = _read_mirror_fingerprint()
    expected_fp = _expected_sync_fingerprint()
    if not _should_launch_postclose(snapshot, now, index, release_status,
                                    mirror_fp, expected_fp):
        return 'gate_closed'
    state_fn = _state or state
    needs_fn = _needs_rebuild or needs_rebuild
    run_fn = _subprocess_run or subprocess.run
    paths_fn = _worker_paths or _worker_paths_real
    current = state_fn()
    root = directory()
    root.mkdir(parents=True, exist_ok=True)
    if needs_fn(snapshot, current):
        python, worker = paths_fn()
        worker = worker.with_name('live_postclose_job_v1.py')
        snap = snapshot_file or _snapshot_path()
        with (root / 'worker_output.log').open('ab') as log:
            run_fn([str(python), '-B', '-X', 'utf8', str(worker),
                    '--snapshot', str(snap), '--runtime', str(root)],
                   stdout=log, stderr=log, timeout=10800,
                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if state_fn().get('status') == 'COMPLETE':
                checker = worker.with_name('accept_live_postclose_v1.py')
                run_fn([str(python), '-B', '-X', 'utf8', str(checker),
                        '--runtime', str(root)], stdout=log, stderr=log, timeout=600,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return 'launched_worker'
    elif current.get('status') == 'COMPLETE' and current.get('acceptance_status') != 'PASS':
        python, worker = paths_fn()
        with (root / 'worker_output.log').open('ab') as log:
            run_fn([str(python), '-B', '-X', 'utf8',
                    str(worker.with_name('accept_live_postclose_v1.py')),
                    '--runtime', str(root)], stdout=log, stderr=log, timeout=600,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return 'launched_checker'
    return 'no_action'


def start():
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    def loop():
        import time
        import pentacles_demo
        import index_minutes_v1
        from joint_risk_bridge import _worker_paths
        while True:
            try:
                now=datetime.now()
                file=pentacles_demo.snapshot_path()
                snapshot=json.loads(file.read_text(encoding='utf-8'))
                _decide_and_launch(snapshot, now, snapshot_file=file)
            except (OSError,ValueError,KeyError,subprocess.TimeoutExpired):
                pass
            time.sleep(30)
    _thread=threading.Thread(target=loop,name='postclose-risk-launcher',daemon=True)
    _thread.start()
