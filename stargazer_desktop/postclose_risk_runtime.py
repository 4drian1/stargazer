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
import postclose_index_source

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
        # A changed display/list must not discard already accepted, identical
        # per-stock risk inputs. Other stocks may still be awaiting computation.
        for path in sorted(directory().glob('run_*/independent_acceptance*.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True):
            if path.stat().st_size>2*1024**2:continue
            record=json.loads(path.read_bytes())
            if (record.get('status')=='COMPLETE' and record.get('acceptance_status')=='PASS'
                    and str(record.get('data_date'))==str(snapshot.get('data_date'))
                    and Path(record['run_dir']).resolve()==path.resolve().parent):
                pack,reason=_pack_from_accepted(code,snapshot,record)
                if pack is not None:
                    return pack,None
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


def _accepted_run_for_day(day, root=None):
    """最新 PASS 验收运行及其字节校验过的原始选股快照。返回 (record, parent) 或 (None, None)。"""
    root = Path(root) if root is not None else directory()
    for path in sorted(root.glob('run_*/independent_acceptance*.json'),
                       key=lambda p: p.stat().st_mtime_ns, reverse=True):
        if path.stat().st_size > 2 * 1024**2:
            continue
        try:
            record = json.loads(path.read_bytes())
            if (record.get('status') != 'COMPLETE' or record.get('acceptance_status') != 'PASS'
                    or str(record.get('data_date')) != str(day)
                    or Path(record['run_dir']).resolve() != path.resolve().parent):
                continue
            raw = (Path(record['run_dir']) / 'input_snapshot.json').read_bytes()
            if hashlib.sha256(raw).hexdigest() != record['snapshot_sha256']:
                continue  # 字节绑定失败：该运行不可作绑定基准，继续找更旧的已验收运行
            return record, json.loads(raw)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None, None


def plan_stock_bindings(plan_snapshot, root=None):
    """确认计划→Pyxis 证据的投影绑定：以冻结原始选股快照为绑定基准。

    逐股规则（显式拒绝，不按代码＋日期借包）：
    - 无同日已验收运行/数据日不一致 → 显式缺失，不用其他日期证据
    - 原始快照无此代码 → 人工新增或非当日选股
    - 计划 row 与原始快照 row 任一字段不一致（价格/特征/来源）→ 输入变化
    绑定保留：input_snapshot 字节SHA、packs_sha256、pack market_date/asof 校验。
    返回 {code: {'status': 'AVAILABLE'|'MISSING', 'reason':..., 'pack':..., 'provenance':...}}。
    """
    day = plan_snapshot.get('data_date')
    record, parent = _accepted_run_for_day(day, root)
    if record is None:
        base_reason = '无同日已验收风险运行：不能用其他日期证据，也不能仅凭代码＋日期借包'
        return {row.get('code'): {'status': 'MISSING', 'reason': base_reason, 'pack': None,
                                  'provenance': None}
                for row in plan_snapshot.get('rows') or []}
    provenance = {'binding': 'PLAN_PROJECTION_PER_STOCK', 'data_date': str(day),
                  'original_snapshot_sha256': record.get('snapshot_sha256'),
                  'packs_sha256': record.get('packs_sha256'), 'run_dir': record.get('run_dir')}
    original = {r.get('code'): r for r in parent.get('rows') or []}
    result = {}
    for row in plan_snapshot.get('rows') or []:
        code = row.get('code')
        old = original.get(code)
        if old is None:
            result[code] = {'status': 'MISSING',
                            'reason': '人工新增或非当日选股：原始选股快照无此代码，不借包',
                            'pack': None, 'provenance': provenance}
        elif row != old:
            diff = sorted(k for k in set(row) | set(old) if row.get(k) != old.get(k))
            result[code] = {'status': 'MISSING',
                            'reason': '输入变化：与原始选股快照不一致（' + ','.join(diff[:6]) + '）',
                            'pack': None, 'provenance': provenance}
        else:
            pack, reason = _pack_from_accepted(code, parent, record)
            result[code] = {'status': 'AVAILABLE' if pack is not None else 'MISSING',
                            'reason': None if pack is not None else (reason or '当前股票风险证据未通过'),
                            'pack': pack, 'provenance': provenance}
    return result


def find_pack_for_plan(code, plan_snapshot, root=None):
    """计划投影入口的单股绑定：plan_stock_bindings 的 (pack, reason) 视图。"""
    item = plan_stock_bindings(plan_snapshot, root).get(code)
    if item is None:
        return None, '计划名单无此股票'
    return item.get('pack'), item.get('reason')


def _pack_from_accepted(code,snapshot,value):
    from oracles_v1 import EvidencePack
    try:
        row=next((r for r in snapshot.get('rows',[]) if r.get('code')==code),None)
        same_stock=bool(row) and snapshot_bound(value,{**snapshot,'rows':[row]})
        if not snapshot_bound(value, snapshot) and not same_stock:
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
               'manual_review_rows','manual_review_population_count','manual_review_scope',
               'candidate_population_count','default_candidate_count','excluded_rows','ranking','model_status'}
    if {k:v for k,v in parent.items() if k not in ignored} != {k:v for k,v in current.items() if k not in ignored}:
        return False
    app = Path(paths.APP_DIR).resolve()
    project = app.parent.parent if app.name == 'dist' else app.parent
    # These wrappers/display contracts do not enter the frozen 232/266 matrices.
    mutable = {str((project/'stargazer_desktop'/name).resolve()) for name in (
        'server.py','assistant_workflow.py','assistant_context.py','golden_monitor.py','index_minutes_v1.py',
        'pentacles_monitor.py','plan_evidence.py','postclose_risk_runtime.py',
        'tarot_monitor_v2.py','risk_outlook_v1.py','trade_facts_v1.py','deepseek_api.py','pentacles_demo.py','plan_board_monitor.py',
        'minute_history_v1.py','live_plan_lane_v1.py','tarot_event_live_v1.py','postclose_prepare_v1.py',
        'sector_rule_scan.py','golden_stock_analysis.py','golden_watch_groups.py','on_demand_model_worker.py','agent/brain.py',
        'conversation_plans.py','minute_plan_review.py','plan_rule_profiles.py','chat_archive.py',
        'push_layout_v1.py','push_presenter_v1.py','strong_wave_facts_v1.py','strong_sector_rules_v1.py','strong_low_indicators_v1.py','strong_sector_watch_v1.py',
        'web/app.js','web/pentacles.js','web/index.html','test_tarot_selector_binding_v1.py','Stargazer_demo_safe.spec')}
    base = project/'stargazer_ml/v5_hybrid'
    mutable.update(str((base/'risk_r0_v1'/name).resolve()) for name in (
        'tarot_intraday_risk_v2.py','live_postclose_job_v1.py','accept_live_postclose_v1.py',
        'index_minutes_v1/index_minutes_updater_v1.py'))
    mutable.update(str((base/'green_common_profile'/name).resolve()) for name in (
        'tarot_live_v2.py','golden_ladder_v3.py','golden_resonance_v4.py','tarot_live_signal_v2.py','gcp_golden_intraday_v1.py','tarot_manual_review_v1.py','market_price_evidence_v1.py',
        'public_stock_minutes_v1.py','live_five_preview_v1.py','live_minute_source_v1.py','live_plan_worker_v1.py','live_index_preview_v1.py'))
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
    observation = {'zone','distance_atr','entry_rule','observation_map_status','signal','rank'}
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
    if all(find_pack(row['code'],snapshot)[0] is not None for row in snapshot['rows']):
        return False
    if activity.get('status')=='RUNNING':
        return False
    if activity.get('run_dir') and snapshot_bound(activity,snapshot):
        return False
    return True


def worker_active():
    """Probe the worker's OS lock; a persisted RUNNING flag is not liveness."""
    import errno
    import msvcrt
    file = directory()/'worker.lock'
    if not file.exists():
        return False
    with file.open('r+b') as stream:
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                return True
            raise
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    return False


def queue_status(snapshot):
    """Mutable operational progress, deliberately outside bound model evidence."""
    current = state()
    labels = {'SOURCE_PREFLIGHT':'校验数据源', 'CANDIDATE_MINUTE_PREFLIGHT':'先检查候选当日分钟', 'CAPTURING_MINUTES':'采集分钟数据', 'WAIT_SOURCE':'等待本地数据服务恢复',
              'REBUILDING_FROZEN_FEATURES':'计算风险特征', 'PACKING_AND_VALIDATING':'评分及整理证据',
              'TDX_PREPARE_STOCK':'准备通达信个股分钟缓存','TDX_PREPARE_MARKET':'准备通达信大盘指标缓存',
              'TDX_MEMBER_SESSION_PATHS':'读取本地板块成员当日收益路径（非逐票重算历史指标）',
              'TDX_CANDIDATE_LEADER_FEATURES':'计算候选及实际龙头的历史指标',
              'REUSING_PREPARED_CONTEXT':'复用已验公共上下文，按当前名单评分'}
    if current.get('status') == 'RUNNING':
        day = current.get('data_date')
        target = snapshot.get('data_date')
        prefix = ('本批计算中' if str(day)==str(target) else '本批排队中')
        return {'status':'RUNNING' if str(day)==str(target) else 'QUEUED',
                'message': f'{prefix}；后台数据日{day}：'+labels.get(current.get('stage'),current.get('stage','处理中'))+
                    f' {current.get("done",0)}/{current.get("total","未知")}；名单数据日{target}',
                'updated_at':current.get('updated_at')}
    if str(current.get('data_date')) == str(snapshot.get('data_date')):
        status = current.get('status')
        if status == 'COMPLETE':
            return {'status':'READY' if current.get('acceptance_status')=='PASS' else 'VALIDATING',
                    'message':'风险评分验收完成' if current.get('acceptance_status')=='PASS' else '风险评分已生成，等待独立验收'}
        if status in ('FAILED','INTERRUPTED','WAIT_SOURCE'):
            from risk_source_recovery import retry_candidate
            if retry_candidate(current):
                return {'status':'WAIT_SOURCE','message':'本地数据服务中断或任务已停止；已采数据保留，恢复后自动续算（同日最多3次）'}
        if status in ('FAILED','INTERRUPTED','NO_SCORABLE_PACKS'):
            return {'status':status,'message':'风险任务未完成：'+str(current.get('reason') or status)+'；未自动重试'}
    return {'status':'WAITING','message':wait_reason(snapshot)}


def wait_reason(snapshot):
    """Explain the missing generation step; never substitute another day's score."""
    path = directory()/'launch_status.json'
    if path.exists():
        value = json.loads(path.read_bytes())
        if value.get('snapshot_digest') == snapshot_digest(snapshot) and value.get('reason'):
            return value['reason']
    current = state()
    if current.get('status') == 'RUNNING':
        if str(current.get('data_date')) != str(snapshot.get('data_date')):
            return f'当前名单风险评分排队中；后台正在处理数据日{current.get("data_date")}，尚未生成本名单评分'
        return '当前名单风险评分计算中，尚未完成独立验收；不是指数数据缺失'
    now = datetime.now()
    local = postclose_index_source.inspect(snapshot['data_date'], int(now.strftime('%Y%m%d%H%M%S')))
    if local['status'] != 'READY':
        return '盘后风险尚未生成；本地指数收盘数据未就绪：' + local['reason']
    return '本地指数收盘数据已就绪；当前名单风险评分尚未生成并验收，不代表没有风险'


def _launch_status(snapshot, now, status, reason=''):
    root = directory()
    root.mkdir(parents=True, exist_ok=True)
    previous = json.loads((root/'launch_status.json').read_bytes()) if (root/'launch_status.json').exists() else {}
    identity = snapshot_digest(snapshot)
    value = dict(status=status, reason=reason, snapshot_digest=identity,
                 attempted=status == 'STARTING' or (previous.get('snapshot_digest') == identity and previous.get('attempted', False)),
                 data_date=snapshot['data_date'], observed_at=now.isoformat())
    temp = root/'launch_status.tmp'
    temp.write_text(json.dumps(value, ensure_ascii=False), encoding='utf8')
    temp.replace(root/'launch_status.json')


def _worker_paths_real():
    from joint_risk_bridge import _worker_paths
    return _worker_paths()


def _snapshot_path():
    import pentacles_demo
    file=pentacles_demo.snapshot_path()
    confirmed=file.with_name('assistant_plan.json')
    try:
        value=json.loads(confirmed.read_bytes())
        if not isinstance(value,dict):return file
        if value.get('selected_manual_codes'):
            import trade_plan_v1
            plan=trade_plan_v1.load_plan() or {}
            parent=json.loads(file.read_bytes())
            if (value.get('assistant_plan_id') and value['assistant_plan_id']==plan.get('assistant_plan_id')
                    and all(str(value.get(k))==str(parent.get(k))==str(plan.get(k)) for k in ('data_date','monitor_date'))
                    and {r['code'] for r in value.get('rows',[])}=={r['code'] for r in plan.get('stocks',[]) if r.get('source')!='USER'}
                    and value.get('source_selection_digest')==snapshot_digest(parent)):
                return confirmed
    except (OSError,ValueError,TypeError,KeyError):pass
    return file


def _native_source_choice(snapshot, runtime, worker):
    """Select the native producer after real source/calendar preflight.

    The native module imports heavy risk kernels and the local source adapter;
    executing it in the GUI process would drag the research-storage contract
    chain (and EXE-bundled stale modules) into the frozen desktop. Instead we run
    the readiness + identity check in a SEPARATE python worker -- the same
    external interpreter the real post-close worker uses -- and only read its
    JSON, so the GUI process never imports the native stack.
    """
    import os
    import sys
    import tempfile
    source = worker.parent / 'tdx_postclose_runtime_v1.py'
    if not source.is_file():
        return dict(contract='TDX_NATIVE_POSTCLOSE_V1',ready=False,
                    reason='通达信盘后适配器缺失；不回退stockdb，请修复安装或数据路径')
    python, _ = _worker_paths_real()
    snap_fd, snap_path = tempfile.mkstemp(suffix='.json')
    try:
        with os.fdopen(snap_fd, 'w', encoding='utf-8') as fh:
            json.dump(snapshot, fh, ensure_ascii=False)
        proc = subprocess.run(
            [python, '-B', '-X', 'utf8', str(source), 'check', snap_path, str(runtime)],
            capture_output=True, timeout=180,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    finally:
        try:
            os.unlink(snap_path)
        except OSError:
            pass
    if proc.returncode != 0:
        # Decode the child's stderr explicitly as UTF-8 (a frozen EXE on zh-CN
        # Windows would otherwise default to cp936 and mangle the Chinese reason).
        reason = (proc.stderr or b'').decode('utf-8', 'replace').strip()[-300:]
        return dict(contract='TDX_NATIVE_POSTCLOSE_V1', ready=False,
                    reason=reason or 'native check process failed')
    try:
        result = json.loads(proc.stdout.decode('utf-8', 'replace'))
    except (ValueError, json.JSONDecodeError):
        return dict(contract='TDX_NATIVE_POSTCLOSE_V1', ready=False,
                    reason='native check produced no JSON')
    result.setdefault('contract', 'TDX_NATIVE_POSTCLOSE_V1')
    result.setdefault('reason', 'native check returned no reason')
    return result


def _decide_and_launch(snapshot, now, snapshot_file=None, *, _state=None,
                       _needs_rebuild=None, _subprocess_run=None, _worker_paths=None):
    """Decide + perform the post-close worker (re)launch.

    Encapsulates the scheduling side effect (a real subprocess call) so the launch
    behaviour can be exercised with stubs. Returns a short status string:
    'gate_closed' | 'no_action' | 'launched_worker' | 'launched_checker'.
    """
    # Before selection refresh, yesterday's snapshot may still be on disk at startup.
    # Preserve pre-open catch-up, but do not launch it after its observation day closes.
    if int(snapshot.get('monitor_date',99999999)) <= int(now.strftime('%Y%m%d')) and now.hour >= 15:
        _launch_status(snapshot, now, 'WAIT_LATEST_SELECTION', '观察日已收盘；等待最新盘后名单，不启动过期名单补算')
        return 'gate_closed'
    as_of = int(now.strftime('%Y%m%d%H%M%S'))
    local = (postclose_index_source.inspect(snapshot['data_date'], as_of)
             if postclose_index_source.job_due(snapshot, now) else {})
    use_local = local.get('status') == 'READY'
    if not use_local:
        index = index_minutes_v1.get_index_freshness()
        if not _should_launch_postclose(snapshot, now, index, _read_release_status(),
                                       _read_mirror_fingerprint(), _expected_sync_fingerprint()):
            _launch_status(snapshot, now, 'WAIT_DATA',
                '盘后风险未启动；' + (local.get('reason') or '等待名单数据日的完整收盘数据及放行资格'))
            return 'gate_closed'
    state_fn = _state or state
    needs_fn = _needs_rebuild or needs_rebuild
    run_fn = _subprocess_run or subprocess.run
    paths_fn = _worker_paths or _worker_paths_real
    current = state_fn()
    root = directory()
    root.mkdir(parents=True, exist_ok=True)
    # An accepted immutable result is not a request to prepare the same inputs
    # again. Check the actual per-code pack hashes before any heavy preflight.
    accepted = accepted_for(snapshot, current)
    if accepted is not None and snapshot.get('rows') and all(
            find_pack(row['code'], snapshot)[0] is not None for row in snapshot['rows']):
        _launch_status(snapshot, now, 'READY', '当前名单风险证据已验收；直接复用，不启动重复准备')
        return 'no_action'
    python, worker = paths_fn()
    native_choice=_native_source_choice(snapshot,root,worker)
    if native_choice and not native_choice.get('ready'):
        _launch_status(snapshot,now,'WAIT_DATA',native_choice['reason'])
        return 'wait_source'
    if current.get('status') == 'RUNNING':
        if worker_active():
            _launch_status(snapshot, now, 'WAIT_WORKER',
                '风险任务仍在运行；保留现有任务，完成后重新检查最新名单')
            return 'no_action'
        refreshed = state_fn()
        if refreshed != current:
            return 'no_action'  # Worker may have completed between the state read and lock probe.
        # App/process termination can leave RUNNING on disk forever. Keep its
        # provenance; same-snapshot retry restrictions below remain in force.
        current = {**current, 'status':'INTERRUPTED', 'reason':'任务锁已释放，原进程不再运行；旧证据保留'}
        from pentacles_monitor import atomic_json
        record = root/('interrupted_'+snapshot_digest(current)+'.json')
        if not record.exists():
            atomic_json(record, current)
        atomic_json(root/'state.json', current)
    from risk_source_recovery import retry_candidate, source_available, allowance, reserve
    needed = needs_fn(snapshot, current)
    if native_choice and (current.get('source_contract')!=native_choice['contract'] or
                          current.get('source_identity')!=native_choice['identity']):
        needed=True
    from runtime_disk_limit_v1 import headroom
    disk_recovered=(native_choice and current.get('status')=='FAILED'
        and current.get('reason')=='ValueError:RUNTIME_DISK_BUDGET' and headroom(root)>=64*1024**2)
    recovering = (not needed and (retry_candidate(current) or disk_recovered) and snapshot_bound(current, snapshot)
                  and accepted_for(snapshot, current) is None)
    if needed or recovering:
        if headroom(root)<64*1024**2:
            _launch_status(snapshot,now,'WAIT_STORAGE','运行缓存不足64MiB；未启动整批计算，请先处理缓存空间')
            return 'wait_storage'
        attempt_file = root/'launch_status.json'
        previous = json.loads(attempt_file.read_bytes()) if attempt_file.exists() else {}
        if (not recovering and previous.get('snapshot_digest') == snapshot_digest(snapshot) and previous.get('attempted')
                and previous.get('source_identity')==(native_choice or {}).get('identity')):
            return 'no_action'  # Includes a subprocess that failed before writing worker state.
        stamp = int(now.strftime('%Y%m%d%H%M%S'))
        if recovering:
            allowed, reason = allowance(root, snapshot['data_date'], stamp)
            if not allowed:
                _launch_status(snapshot, now, 'RECOVERY_PAUSED', reason)
                return 'no_action'
        if not native_choice and not source_available():
            _launch_status(snapshot, now, 'WAIT_SOURCE', '本地stockdb未就绪；已采数据保留，服务恢复后自动续算')
            return 'wait_source'
        python, worker = paths_fn()
        worker = worker.with_name('live_postclose_job_v1.py')
        snap = snapshot_file or _snapshot_path()
        extra = []
        if native_choice:
            extra=['--tdx-native']
        elif use_local:
            source, source_sha = postclose_index_source.prepare(snapshot['data_date'], as_of, root)
            extra = ['--index-source', str(source), '--index-source-sha', source_sha]
        if recovering and not reserve(root, snapshot['data_date'], stamp, snapshot_digest(snapshot), current.get('run_dir')):
            return 'no_action'
        _launch_status(snapshot, now, 'STARTING', '当前名单风险任务已启动，等待计算及独立验收')
        if native_choice:
            value=json.loads((root/'launch_status.json').read_bytes())
            value['source_identity']=native_choice['identity']
            from pentacles_monitor import atomic_json
            atomic_json(root/'launch_status.json',value)
        with (root / 'worker_output.log').open('ab') as log:
            result = run_fn([str(python), '-B', '-X', 'utf8', str(worker),
                    '--snapshot', str(snap), '--runtime', str(root)] + extra,
                   stdout=log, stderr=log, timeout=10800,
                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if result.returncode != 0:
                if retry_candidate(state_fn()):
                    _launch_status(snapshot, now, 'WAIT_SOURCE', '本地数据服务中断；已采数据保留，服务恢复后自动续算（同日最多3次）')
                    return 'wait_source'
                _launch_status(snapshot, now, 'ERROR', '盘后风险任务退出失败，详见worker_output.log；未自动重试')
                return 'worker_failed'
            finished = state_fn()
            if finished.get('status') == 'COMPLETE' and str(finished.get('data_date')) == str(snapshot['data_date']):
                checker = worker.with_name('accept_live_postclose_v1.py')
                run_fn([str(python), '-B', '-X', 'utf8', str(checker),
                        '--runtime', str(root)], stdout=log, stderr=log, timeout=600,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return 'launched_worker'
    elif (current.get('status') == 'COMPLETE' and str(current.get('data_date')) == str(snapshot['data_date'])
          and current.get('acceptance_status') != 'PASS' and snapshot_bound(current, snapshot)):
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
                file=_snapshot_path()
                snapshot=json.loads(file.read_text(encoding='utf-8'))
                _decide_and_launch(snapshot, now, snapshot_file=file)
            except (OSError,ValueError,KeyError,subprocess.TimeoutExpired) as exc:
                if 'snapshot' in locals():
                    _launch_status(snapshot, now, 'ERROR', '盘后风险调度失败：'+type(exc).__name__+':'+str(exc))
            time.sleep(30)
    _thread=threading.Thread(target=loop,name='postclose-risk-launcher',daemon=True)
    _thread.start()
