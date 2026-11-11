"""强势池准备的显式入口与预约生命周期（Step2 dev，隔离于 _step2_dev/）。

修复背景（FIX_HANDOFF 2026-11-10）：添加板块并确认后没有自动准备强势池；
schedule 遇到任何 .failed.json 直接 return；invoke 异常只记内存 HEALTH 不改预约，
中断可永久卡住。本模块把"板块名单 → 强势池"变成显式任务。

边界（2026-11-10 复核第二版）：
- owner 存活三态：存活／确认死亡／未知。未知（查询失败、死亡但未超时、超时但
  查不出、RUNNING 无 owner 记录）一律不得抢占；只有"确认死亡且已超时"可回收。
- 跨进程原子性：任务创建、恢复、排队、认领、结果回写、池发布全部在文件锁内
  重新读取并判断；池文件与 READY 提交在同一锁段内完成（先写 .part，锁内替换+记账）。
  被接管/过期 attempt 的结果一律作废，绝不先发布文件再判作废。
- 统一池校验器：提交、复用、查询共用——核对观察日/数据日、实际行情版本
  （source_key）、数学规则版本（契约）、成员 SHA、行哈希、板块名单、盘前时序
  （built_at 早于观察日 09:30，不删时间门）。不绑定 UI/推送文案版本。
- 历史池：过期（观察日早于今日）只标 READY_STALE 保留展示，不作当前就绪。
- state()/recover 纯读：不建目录（jobs_dir(create=False)）、不解析行情上下文、
  不触发任何准备/日线动作；context 解析全部在后台线程。
- 任务文件损坏报 JOB_FILE_CORRUPT 并保留原件，不按"没有任务"重置次数。
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import secrets
import threading
import time
from pathlib import Path

CONTRACT = 'STARGAZER_STRONG_POOL_PREP_V1'
ZONE = timezone(timedelta(hours=8))
MAX_ATTEMPTS = 3
COOLDOWN_SECONDS = 900
RUNNING_TIMEOUT_SECONDS = 1500
# worker 正常返回但非 READY：确定性的数据/规则结果，永久终止，无自动重试
PERMANENT_STATUSES = frozenset({'DATA_UNAVAILABLE'})
_LOCK = threading.RLock()
_RUNNING = set()  # 同进程防重入；跨进程靠文件锁+owner 三态


def _now(now=None):
    return now or datetime.now(ZONE)


def _stamp(now=None):
    return _now(now).strftime('%Y%m%d%H%M%S')


def _today(now=None):
    return int(_now(now).strftime('%Y%m%d'))


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode('utf-8')


def digest(value):
    if isinstance(value, (dict, list)):
        value = _encoded(value)
    elif isinstance(value, str):
        value = value.encode('utf-8')
    return hashlib.sha256(value).hexdigest()


def _pid_alive(pid):
    """True=存活 / False=确认死亡 / None=未知（查询失败或无法确认）。"""
    try:
        from cross_process_lock_v1 import pid_alive
        return bool(pid_alive(int(pid)))
    except Exception:
        return None


def _owner_status(owner, now):
    """三态：alive / dead / unknown。unknown 一律不得抢占。"""
    if not owner or not owner.get('pid'):
        return 'unknown'  # RUNNING 但无 owner 记录：保守按未知处理
    try:
        started = float(owner.get('started_epoch', 0))
    except (TypeError, ValueError):
        started = 0
    age = now.timestamp() - started
    alive = _pid_alive(owner.get('pid'))
    if alive is True:
        return 'alive'
    if alive is False and age > RUNNING_TIMEOUT_SECONDS:
        return 'dead'  # 确认死亡且已超时
    # 确认死亡但未超时（PID 复用风险）、查询失败、超时但查不出：都不得抢占
    return 'unknown'


def _strong_root():
    import strong_sector_watch_v1 as ssw
    return ssw.root()


# ---------------- 旧目录板块成员数（memoized，纯读，不用模糊匹配） ----------------
_LEGACY_CONTEXT_CACHE = {}


def _legacy_member_map(names):
    """一次性解析目录并解析全部请求的板块：{name: info}。
    按目录文件 mtime 记忆化；歧义/缺失的板块返回 None 值，不借用模糊同名。"""
    try:
        from sector_rule_scan import project_root, source_context, resolve_board
        from zoneinfo import ZoneInfo
        from datetime import datetime as _dt
        project = project_root()
        ctx = source_context(project, _dt.now(ZoneInfo('Asia/Shanghai')).isoformat())
        cache_key = (str(ctx['catalog']), Path(ctx['catalog']).stat().st_mtime_ns)
        cached = _LEGACY_CONTEXT_CACHE.get(cache_key)
        if cached is None:
            import pandas as pd
            import re as _re
            catalog = pd.read_parquet(ctx['catalog'])
            aliases = {'pcb': 'PCB概念', 'cpo': '共封装光学(CPO)', '存储': '存储芯片', '煤炭': '煤炭概念'}
            cached = {'catalog': catalog, 'aliases': aliases, 'data_date': str(ctx['P']),
                      'resolve_board': resolve_board}
            _LEGACY_CONTEXT_CACHE.clear()
            _LEGACY_CONTEXT_CACHE[cache_key] = cached
        result = {}
        for name in names:
            board, error = cached['resolve_board'](cached['catalog'],
                                                   cached['aliases'].get(name.casefold(), name))
            if error is None:
                codes = [str(c).zfill(6) for c in board.symbols]
                result[name] = dict(member_count=len(codes), member_source=cached['data_date'],
                                    member_provenance='LEGACY_CATALOG',
                                    identity_index_code=str(board['code']))
            else:
                result[name] = None
        return result
    except Exception:
        return {name: None for name in names}


def _legacy_member_map_async(names):
    """目录解析全程放后台线程（akshare/parquet 首次导入慢，绝不阻塞 GET）：
    有缓存立即返回；无缓存先给 None（页面轮询后续自然填上），不阻塞接口。"""
    keys = [k for k in _LEGACY_CONTEXT_CACHE if k != '_WARMING']
    if keys:
        return _legacy_member_map(names)  # 缓存命中快路径（同 mtime）
    if '_WARMING' not in _LEGACY_CONTEXT_CACHE:
        _LEGACY_CONTEXT_CACHE['_WARMING'] = True

        def warm():
            try:
                _legacy_member_map(list(names))
            finally:
                _LEGACY_CONTEXT_CACHE.pop('_WARMING', None)

        threading.Thread(target=warm, daemon=True, name='legacy-members-warm').start()
    return {name: None for name in names}


def _jobs_dir(create=True):
    folder = _strong_root() / 'prep_jobs'
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    return folder


def _read(path, default):
    try:
        return json.loads(Path(path).read_bytes())
    except (OSError, ValueError):
        return default


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(_encoded(value))
    tmp.replace(path)


def _file_lock(name):
    from cross_process_lock_v1 import FileLock
    _jobs_dir(create=True)  # 锁文件所在目录（写路径，允许创建）
    return FileLock(str(_jobs_dir() / (name + '.lock')), timeout=15)


def _normalize_boards(boards):
    seen, result = set(), []
    for board in boards or []:
        name = board.get('name') if isinstance(board, dict) else board
        if isinstance(name, str) and name.strip() and name.strip() not in seen:
            seen.add(name.strip())
            result.append(name.strip())
    return sorted(result)


def _context_from_market(project, as_of_iso):
    """数据日/观察日/行情版本；日线或日历缺失抛异常（调用方按真实原因分类）。
    只在后台线程调用：GET 纯读路径绝不触达。

    边界约定：sector_rule_scan.source_context 内部用 ``project / '目录'`` 拼接，
    project 必须是 Path。这里统一用 Path(project) 收口，不接受 str 泄漏到下游；
    不能改 sector_rule_scan.py（它在计算缓存身份白名单内，会改变 source_key）。"""
    from sector_rule_scan import source_context
    context = source_context(Path(project), as_of_iso)
    return dict(P=context['P'], E=context['E'], key=context['key'])


def default_runner(request):
    import strong_sector_watch_v1 as ssw
    return ssw.invoke(request)


# ---------------- 统一池校验器（提交/复用/查询共用） ----------------

def pool_matches_job(pool, job):
    """数学缓存绑定校验：观察日/数据日、行情版本、规则版本、成员 SHA、
    行哈希、板块名单、盘前时序。不绑 UI/推送文案版本。"""
    import strong_sector_watch_v1 as ssw
    if not isinstance(pool, dict) or pool.get('contract') != ssw.CONTRACT:
        return False
    if sorted(set(pool.get('boards') or [])) != sorted(set(job.get('boards') or [])):
        return False
    if digest(pool.get('rows')) != pool.get('rows_sha256'):
        return False
    if int(pool.get('observation_date', 0) or 0) != int(job.get('observation_date', -1) or -1):
        return False
    if int(pool.get('data_date', 0) or 0) != int(job.get('data_date', -1) or -1):
        return False
    if str(pool.get('source_key', '')) != str(job.get('source_key', '')):
        return False
    # 盘前时序：built_at 必须早于观察日 09:30（不删时间门，不伪造盘前已知）
    if str(pool.get('built_at', '9' * 14)) >= str(job.get('observation_date')) + '093000':
        return False
    # 成员版本三态一致：任务版本＝当前版本＝池内版本（任务缺绑定=None 时不得宣称有效；
    # 池内版本缺失/null/非 dict 一律拒绝复用和发布）
    job_versions = job.get('member_versions')
    if job_versions != ssw.constituent_versions(job.get('boards') or []):
        return False
    pool_versions = pool.get('constituent_versions')
    if not isinstance(pool_versions, dict):
        return False
    if pool_versions != job_versions:
        return False
    return True


def _validated_pool(job):
    """READY 复用前重新核验：字节 SHA + 统一校验器。"""
    if not job or not job.get('pool_file'):
        return None
    if job.get('corrupt'):
        return None
    path = Path(job['pool_file'])
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if job.get('pool_sha256') and hashlib.sha256(raw).hexdigest() != job['pool_sha256']:
        return None
    pool = _read(path, None)
    if not pool_matches_job(pool, job):
        return None
    return pool


# ---------------- job 存取（创建/恢复/排队全部锁内重读） ----------------

def _job_name(boards, context):
    return '%d_%s' % (context['E'], digest([boards, context['key'], 'TURNOVER_REFERENCE_V1'])[:16])


def _load_or_create_job(boards, context, now):
    """锁内创建：任务文件损坏→corrupt 标记保留原件；并发创建只有一个胜出。"""
    boards = _normalize_boards(boards)
    name = _job_name(boards, context)
    with _file_lock(name):
        file = _jobs_dir() / (name + '.json')
        if file.exists():
            job = _read(file, None)
            if not job or not isinstance(job, dict) or job.get('contract') != CONTRACT:
                return dict(contract=CONTRACT, job_id=name, boards=boards, corrupt=True,
                            state='FAILED', permanent_reason='JOB_FILE_CORRUPT',
                            note='任务文件损坏，原件保留作证据；请人工核对后删除该文件再重试')
            return job
        job = dict(contract=CONTRACT, job_id=name, boards=boards,
                   data_date=context['P'], observation_date=context['E'], source_key=context['key'],
                   rule_version=None, member_versions=_current_member_versions(boards),
                   state='QUEUED', attempts=0,
                   owner=None, attempt_id=None, started_at=None, finished_at=None,
                   last_error=None, stderr_digest=None, pool_file=None, pool_sha256=None,
                   qualified_count=None, per_board={}, permanent_reason=None,
                   created_at=_stamp(now), updated_at=_stamp(now))
        _write(file, job)
        return job


def _current_member_versions(boards):
    try:
        import strong_sector_watch_v1 as ssw
        return ssw.constituent_versions(boards)
    except Exception:
        return None


def get_job(job_id):
    job = _read(_jobs_dir(create=False) / (str(job_id) + '.json'), None)
    return job if isinstance(job, dict) and job.get('contract') == CONTRACT else None


def _claim(job_id, now):
    """锁内认领：复核状态与 owner 三态，写入 RUNNING+attempt_id。"""
    with _file_lock(job_id):
        job = get_job(job_id)
        if job is None or job.get('corrupt'):
            return None
        if job.get('state') == 'RUNNING' and \
                _owner_status(job.get('owner'), now) in ('alive', 'unknown'):
            return None
        attempt = dict(pid=str(_own_pid()), nonce=secrets.token_hex(8),
                       started_epoch=time.time(), attempt_id=secrets.token_hex(8))
        job.update(state='RUNNING', owner=attempt, attempt_id=attempt['attempt_id'],
                   started_at=_stamp(now), last_error=None)
        _write(_jobs_dir() / (job_id + '.json'), job)
        return job, attempt


def _finish(job_id, attempt_id, **updates):
    """结果回写：锁内核对 attempt_id，被接管/过期结果作废（返回 None）。"""
    with _file_lock(job_id):
        job = get_job(job_id)
        if job is None or job.get('attempt_id') != attempt_id:
            return None
        job.update(updates)
        job['updated_at'] = datetime.now(ZONE).strftime('%Y%m%d%H%M%S')
        _write(_jobs_dir() / (job_id + '.json'), job)
        return job


def _set_job(job_id, **updates):
    """测试/维护用的受锁状态写入（不核对 attempt_id）。"""
    with _file_lock(job_id):
        job = get_job(job_id)
        if job is None:
            return None
        job.update(updates)
        job['updated_at'] = datetime.now(ZONE).strftime('%Y%m%d%H%M%S')
        _write(_jobs_dir() / (job_id + '.json'), job)
        return job


# ---------------- 提交 ----------------

def submit(boards, now=None, *, context=None, project=None, runner=None,
           background=True, force=False):
    """提交/恢复一个准备任务。返回任务视图，不抛业务异常。
    background=True 且未给 context 时，行情上下文解析在后台线程完成，立即返回。"""
    now = _now(now)
    boards = _normalize_boards(boards)
    if not boards:
        return {'status': 'FAILED', 'reason': 'BOARDS_EMPTY', 'retryable': False}
    if context is None and background:
        thread = threading.Thread(target=_submit_background,
                                  args=(boards, project, runner or default_runner, now, force),
                                  daemon=True, name='strong-pool-prep')
        thread.start()
        return {'status': 'QUEUED', 'boards': boards, 'context_resolving': True}
    if context is None:
        try:
            project = _project_path(project)
            context = _context_from_market(project, now.astimezone(ZONE).isoformat())
        except Exception as exc:
            return _context_failure(boards, now, exc)
    return _submit_with_context(boards, context, now, runner or default_runner,
                                background=background, force=force)


def _submit_background(boards, project, runner, now, force):
    try:
        project = _project_path(project)
        context = _context_from_market(project, now.astimezone(ZONE).isoformat())
    except Exception as exc:
        _context_failure(boards, now, exc)
        return
    _submit_with_context(boards, context, now, runner, background=True, force=force)


# ---------------- 提交边界：project 统一为 Path；错误按真实原因分类 ----------------

# 只有这些明确的「日线/日历未就绪」信号才可归 WAIT_DAILY。程序/配置异常
# （TypeError、AttributeError、ImportError 等）一律报告为计算/配置错误，
# 绝不冒充「日线没下载」，也不冒充已准备成功。
_DATA_WAIT_SIGNALS = (
    'TRADING_CALENDAR_EXPIRED', 'LOCAL_DAILY_SOURCE_MISSING',
    'OBSERVATION_TIMEZONE_REQUIRED', 'DAILY_DIRECTORY_MISSING',
    'CALENDAR_MISSING', 'DAILY_PART_MISSING',
)
PREP_ERROR_CLASS = 'COMPUTE_OR_CONFIG_ERROR'


def _project_path(project):
    """明确边界：无论调用方传 Path 还是 str，进入行情上下文前统一为 Path。
    sector_rule_scan.source_context 用 project / '目录' 拼接，str 会触发
    TypeError: unsupported operand type(s) for /: 'str' and 'str'。"""
    if project is None:
        from sector_rule_scan import project_root
        project = project_root()
    return Path(project)


def _classify_context_error(exc):
    detail = type(exc).__name__ + ':' + str(exc)[:160]
    message = str(exc)
    if isinstance(exc, ValueError) and any(tag in message for tag in _DATA_WAIT_SIGNALS):
        return 'WAIT_DAILY', detail
    return 'PREP_ERROR', detail


def _context_failure(boards, now, exc):
    kind, detail = _classify_context_error(exc)
    if kind == 'WAIT_DAILY':
        return _wait_daily(boards, now, reason=detail)
    return _prep_error(boards, now, detail)


def _prep_error(boards, now, reason):
    """计算/配置错误：独立落盘如实的当前请求状态，不删改任何旧任务/错误记录，
    不重置尝试次数，不触碰已有 READY 池。仅 GET 状态读取，不启动计算。"""
    folder = _jobs_dir()
    name = 'prep_error_' + digest(boards)[:12] + '.json'
    record = dict(contract=CONTRACT, boards=boards, state='FAILED', status='FAILED',
                  error_class=PREP_ERROR_CLASS, reason=reason,
                  saved_at=_stamp(now), updated_at=_stamp(now),
                  attempts=None, owner=None, pool_file=None, per_board={},
                  job_id=name[:-5], retryable=False,
                  note='计算/配置错误；不是日线未更新，不冒充已准备，需按原因修复')
    _write(folder / name, record)
    return dict(status='FAILED', error_class=PREP_ERROR_CLASS, reason=reason,
                boards=boards, retryable=False)


def _wait_daily(boards, now, reason):
    folder = _jobs_dir()
    name = 'wait_daily_' + digest(boards)[:12] + '.json'
    record = dict(contract=CONTRACT, boards=boards, state='WAIT_DAILY', status='WAIT_DAILY',
                  reason=reason, saved_at=_stamp(now), updated_at=_stamp(now),
                  attempts=0, owner=None, pool_file=None, per_board={}, job_id=name[:-5],
                  note='本地日线/日历未就绪；保留偏好与真实缺失原因，不冒充已准备')
    _write(folder / name, record)
    return dict(status='WAIT_DAILY', reason=reason, boards=boards, retryable=True)


def _submit_with_context(boards, context, now, runner, *, background, force):
    context = dict(P=int(context['P']), E=int(context['E']), key=str(context['key']))
    # 已有池也必须通过当前任务绑定校验后才能复用（frozen 只核对池内自声明）
    try:
        import strong_sector_watch_v1 as ssw
        existing = ssw.frozen(boards, context['E'])
    except Exception:
        existing = None
    if existing is not None:
        binding = dict(boards=boards, data_date=context['P'], observation_date=context['E'],
                       source_key=context['key'],
                       member_versions=_current_member_versions(boards))
        if pool_matches_job(existing, binding):
            per_board = {name: dict(qualified_count=sum(1 for r in existing['rows']
                                                        if name in (r.get('boards') or [])))
                         for name in boards}
            return dict(status='REUSED', boards=boards, pool=existing,
                        pool_sha256=existing.get('rows_sha256'),
                        qualified_count=len(existing['rows']), per_board=per_board,
                        observation_date=existing.get('observation_date'),
                        data_date=existing.get('data_date'))
        existing = None  # 绑定失配：走任务链重建，不复用
    job = _load_or_create_job(boards, context, now)
    if job.get('corrupt'):
        return {'status': 'FAILED', 'job_id': job['job_id'], 'boards': boards,
                'reason': 'JOB_FILE_CORRUPT', 'retryable': False}
    view = _decide(job, now, force=force)
    if view['status'] != 'QUEUED':
        return view
    if background:
        thread = threading.Thread(target=_run_attempt, args=(job['job_id'], context, runner, now),
                                  daemon=True, name='strong-pool-prep')
        thread.start()
        view['started'] = True
        return view
    return _run_attempt(job['job_id'], context, runner, now)


def _decide(job, now, force=False):
    """锁内重读后做状态迁移，避免用旧快照覆盖另一进程刚写入的 RUNNING。"""
    job_id = job['job_id']
    state = job.get('state')
    if state == 'READY':
        pool = _validated_pool(job)
        if pool is not None:
            return dict(status='REUSED', job_id=job_id, boards=job['boards'],
                        pool_file=job.get('pool_file'), pool_sha256=job.get('pool_sha256'),
                        qualified_count=job.get('qualified_count'), per_board=job.get('per_board'),
                        observation_date=job.get('observation_date'),
                        data_date=job.get('data_date'), pool=pool)
    if state == 'RUNNING':
        redecide = None
        with _file_lock(job_id):
            fresh = get_job(job_id)
            if fresh is None:
                return {'status': 'FAILED', 'job_id': job_id, 'reason': 'JOB_MISSING', 'retryable': False}
            if fresh.get('state') != 'RUNNING':
                redecide = fresh  # 状态已变：退出锁后再判，禁止持锁递归
            else:
                status = _owner_status(fresh.get('owner'), now)
                if status in ('alive', 'unknown'):
                    return dict(status='RUNNING', job_id=job_id, boards=fresh['boards'],
                                owner_pid=(fresh.get('owner') or {}).get('pid'), owner_state=status)
                fresh['attempts'] = int(fresh.get('attempts') or 0) + 1
                fresh['last_error'] = 'OWNER_EXITED:' + str((fresh.get('owner') or {}).get('pid'))
                fresh['owner'] = None
                fresh.update(state='FAILED', finished_at=_stamp(now))
                _write(_jobs_dir() / (job_id + '.json'), fresh)
        if redecide is not None:
            return _decide(redecide, now, force=force)  # 锁外递归：FileLock 非重入，持锁会死锁
        return _retry_or_stop(job_id, now)
    if state == 'FAILED':
        if job.get('permanent_reason') and not force and \
                job['permanent_reason'] != 'POOL_READBACK_MISMATCH':
            return dict(status='FAILED', job_id=job_id, boards=job['boards'],
                        reason=job['permanent_reason'], retryable=False,
                        attempts=job.get('attempts'))
        return _retry_or_stop(job_id, now)
    if state == 'READY':
        # 池文件丢失/损坏/绑定失配：不重算不能宣称就绪；保留次数，直接排队重建（非计算失败，无冷却）。
        # 重建前刷新成员版本绑定：旧绑定已漂移，按当前版本重建。
        with _file_lock(job_id):
            fresh = get_job(job_id)
            if fresh is not None and fresh.get('state') == 'READY':
                fresh.update(state='QUEUED', pool_file=None, pool_sha256=None,
                             member_versions=_current_member_versions(fresh.get('boards') or []),
                             last_error='POOL_FILE_INVALID', updated_at=_stamp(now))
                _write(_jobs_dir() / (job_id + '.json'), fresh)
        return dict(status='QUEUED', job_id=job_id, boards=job['boards'],
                    attempts=job.get('attempts'), retryable=True, rebuild=True)
    return _retry_or_stop(job_id, now)


def _retry_or_stop(job_id, now):
    with _file_lock(job_id):
        job = get_job(job_id)
        if job is None:
            return {'status': 'FAILED', 'job_id': job_id, 'reason': 'JOB_MISSING', 'retryable': False}
        # 锁内按最新状态决定：有人正在计算且不可抢占时，绝不写回 QUEUED 覆盖
        if job.get('state') == 'RUNNING' and \
                _owner_status(job.get('owner'), now) in ('alive', 'unknown'):
            return dict(status='RUNNING', job_id=job_id, boards=job.get('boards'),
                        owner_pid=(job.get('owner') or {}).get('pid'), reason='OCCUPIED')
        attempts = int(job.get('attempts') or 0)
        if attempts >= MAX_ATTEMPTS:
            job.update(state='FAILED', permanent_reason=job.get('permanent_reason') or 'MAX_ATTEMPTS',
                       finished_at=_stamp(now), owner=None)
            _write(_jobs_dir() / (job_id + '.json'), job)
            return dict(status='FAILED', job_id=job_id, boards=job['boards'],
                        reason=job['permanent_reason'] or 'MAX_ATTEMPTS', retryable=False,
                        attempts=attempts)
        last = job.get('finished_at') or job.get('updated_at')
        if last and attempts > 0:
            try:
                elapsed = now.timestamp() - datetime.strptime(str(last), '%Y%m%d%H%M%S').replace(tzinfo=ZONE).timestamp()
            except (TypeError, ValueError):
                elapsed = COOLDOWN_SECONDS
            if elapsed < COOLDOWN_SECONDS:
                retry_after = COOLDOWN_SECONDS - elapsed
                return dict(status='RETRY_WAIT', job_id=job_id, boards=job['boards'],
                            retry_after_seconds=int(retry_after), attempts=attempts, retryable=True)
        job.update(state='QUEUED', updated_at=_stamp(now),
                   member_versions=_current_member_versions(job.get('boards') or []))
        _write(_jobs_dir() / (job_id + '.json'), job)
        return dict(status='QUEUED', job_id=job_id, boards=job['boards'],
                    attempts=attempts, retryable=True)


def _own_pid():
    import os
    return os.getpid()


def _run_attempt(job_id, context, runner, started_now):
    with _LOCK:
        if job_id in _RUNNING:
            return dict(status='RUNNING', job_id=job_id, reason='ALREADY_RUNNING_IN_PROCESS')
        _RUNNING.add(job_id)
    try:
        claimed = _claim(job_id, started_now)
        if claimed is None:
            job = get_job(job_id) or {}
            return dict(status='RUNNING', job_id=job_id,
                        owner_pid=(job.get('owner') or {}).get('pid'), reason='OCCUPIED')
        job, attempt = claimed
        attempt_id = attempt['attempt_id']
        request = dict(mode='POOL', boards=job['boards'],
                       as_of=started_now.astimezone(ZONE).isoformat(),
                       built_at=_stamp(started_now))
        try:
            result = runner(request)
            if not isinstance(result, dict):
                result = dict(status='UNAVAILABLE', reason='RUNNER_RESULT_INVALID')
            status = result.get('status')
            if status == 'READY':
                return _commit_ready(job_id, attempt_id, job, result)
            reason = str(result.get('reason') or status or 'UNKNOWN')
            permanent = status in PERMANENT_STATUSES
            attempts = int(job.get('attempts') or 0) + (0 if permanent else 1)
            _finish(job_id, attempt_id, state='FAILED', owner=None, attempts=attempts,
                    permanent_reason=reason if permanent else None,
                    finished_at=datetime.now(ZONE).strftime('%Y%m%d%H%M%S'),
                    last_error=reason, stderr_digest=digest(result)[:16])
            return dict(status='FAILED', job_id=job_id, boards=result.get('boards'),
                        reason=reason, retryable=not permanent and attempts < MAX_ATTEMPTS,
                        attempts=attempts)
        except Exception as exc:
            # 计算及发布/记账路径统一兜底（额度不足、写盘失败、替换失败等）：
            # 按当前 attempt_id 安全结束预约，绝不留 RUNNING 占用
            finished = _finish(job_id, attempt_id, state='FAILED', owner=None,
                               attempts=int(job.get('attempts') or 0) + 1,
                               finished_at=datetime.now(ZONE).strftime('%Y%m%d%H%M%S'),
                               last_error=type(exc).__name__ + ':' + str(exc)[:160],
                               stderr_digest=digest(type(exc).__name__ + str(exc))[:16])
            attempts = (finished or {}).get('attempts', int(job.get('attempts') or 0) + 1)
            return dict(status='FAILED', job_id=job_id, boards=job.get('boards'),
                        reason=type(exc).__name__, retryable=attempts < MAX_ATTEMPTS,
                        attempts=attempts)
    finally:
        with _LOCK:
            _RUNNING.discard(job_id)


def _commit_ready(job_id, attempt_id, job, pool):
    """READY 提交：统一校验器核对绑定（P/E、行情版本、规则版本、成员 SHA、盘前时序），
    先写 .part 到池路径（锁外 IO），再锁内核对 attempt_id、原子替换池文件并记账 READY。
    被接管/过期的结果在发布前作废——绝不出现"文件已发布但任务作废"的中间态。"""
    import strong_sector_watch_v1 as ssw
    if not pool_matches_job(pool, job):
        _finish(job_id, attempt_id, state='FAILED', owner=None,
                permanent_reason='POOL_BINDING_MISMATCH',
                finished_at=datetime.now(ZONE).strftime('%Y%m%d%H%M%S'))
        return dict(status='FAILED', job_id=job_id, reason='POOL_BINDING_MISMATCH', retryable=False)
    if digest(pool.get('rows')) != pool.get('rows_sha256'):
        _finish(job_id, attempt_id, state='FAILED', owner=None,
                permanent_reason='STRONG_POOL_HASH',
                finished_at=datetime.now(ZONE).strftime('%Y%m%d%H%M%S'))
        return dict(status='FAILED', job_id=job_id, reason='STRONG_POOL_HASH', retryable=False)
    identity = digest([sorted(set(pool.get('boards') or [])), pool.get('source_key'),
                       'TURNOVER_REFERENCE_V1', ssw.CONTRACT,
                       {'ths_members': ssw.constituent_versions(pool.get('boards') or [])}])
    target = ssw.root() / 'pools' / (str(pool['observation_date']) + '_' + identity + '.json')
    # 每次尝试独立的临时文件：只允许清理/替换自己的 .part
    part = Path(str(target) + '.part.' + attempt_id)
    ssw.save(part, pool)  # 锁外写临时文件；发布在锁内
    with _file_lock(job_id):
        current = get_job(job_id)
        if current is None or current.get('attempt_id') != attempt_id:
            part.unlink(missing_ok=True)
            return dict(status='CANCELLED_TAKEN_OVER', job_id=job_id)
        raw = part.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        # 内容级回读核验：解析实际字节，用统一校验器核对（不依赖序列化方式，
        # 生产 save 与 ssw.encoded 格式不同不得造成伪失配）
        parsed = _read(part, None)
        if not pool_matches_job(parsed, current):
            part.unlink(missing_ok=True)
            current.update(state='FAILED', owner=None,
                           attempts=int(current.get('attempts') or 0) + 1,
                           permanent_reason=None,
                           finished_at=datetime.now(ZONE).strftime('%Y%m%d%H%M%S'),
                           last_error='POOL_READBACK_CONTENT_MISMATCH')
            _write(_jobs_dir() / (job_id + '.json'), current)
            return dict(status='FAILED', job_id=job_id, reason='POOL_READBACK_CONTENT_MISMATCH',
                        retryable=True)
        part.replace(target)
        per_board = {}
        for name in pool.get('boards') or []:
            rows = [r for r in pool['rows'] if name in (r.get('boards') or [])]
            per_board[name] = dict(qualified_count=len(rows))
        qualified = len(pool['rows'])
        current.update(state='READY', owner=None,
                       finished_at=datetime.now(ZONE).strftime('%Y%m%d%H%M%S'),
                       pool_file=str(target), pool_sha256=sha, qualified_count=qualified,
                       per_board=per_board, rule_version=ssw.CONTRACT,
                       member_versions=ssw.constituent_versions(pool.get('boards') or []),
                       last_error=None)
        current['updated_at'] = datetime.now(ZONE).strftime('%Y%m%d%H%M%S')
        _write(_jobs_dir() / (job_id + '.json'), current)
    try:
        ssw.HEALTH.update(status='FROZEN', count=qualified,
                          observation_date=pool.get('observation_date'))
    except Exception:
        pass
    return dict(status='READY', job_id=job_id, pool_file=str(target), pool_sha256=sha,
                qualified_count=qualified, per_board=per_board,
                observation_date=pool.get('observation_date'), data_date=pool.get('data_date'),
                pool=pool)


# ---------------- 查询（纯读：不建目录、不解析行情上下文、不触发准备/日线动作） ----------------

def _find_job_for_boards(boards):
    """扫描现有任务文件找匹配板块集合的最新任务。读路径不建目录。"""
    folder = _jobs_dir(create=False)
    if not folder.is_dir():
        return None
    best = None
    for file in folder.glob('*_*.json'):
        job = _read(file, None)
        if not job or not isinstance(job, dict) or job.get('contract') != CONTRACT:
            continue
        if job.get('boards') == boards and not job.get('corrupt'):
            if best is None or str(job.get('updated_at', '')) >= str(best.get('updated_at', '')):
                best = job
    return best


def _find_existing_pool(boards):
    """无任务文件时回退扫描已有池文件：统一有效性核验（契约/板块/行哈希）。"""
    import strong_sector_watch_v1 as ssw
    pools_dir = ssw.root() / 'pools'
    if not pools_dir.is_dir():
        return None
    best = None
    for file in pools_dir.glob('*_*.json'):
        if file.name.endswith('.failed.json') or file.name.endswith('.part'):
            continue
        pool = _read(file, None)
        if not pool or not isinstance(pool, dict) or pool.get('contract') != ssw.CONTRACT:
            continue
        if sorted(set(pool.get('boards') or [])) != boards:
            continue
        if digest(pool.get('rows')) != pool.get('rows_sha256'):
            continue
        if best is None or str(pool.get('built_at', '')) >= str(best.get('built_at', '')):
            best = pool
    return best


def _pool_display_status(pool, now):
    """展示口径：有效池若观察日早于今日=READY_STALE（历史参考，不作当前就绪）；
    当日池须满足盘前时序，盘后补算标 READY_STALE。"""
    obs = int(pool.get('observation_date', 0) or 0)
    today = _today(now)
    if obs < today:
        return 'READY_STALE'
    if str(pool.get('built_at', '9' * 14)) >= str(obs) + '093000':
        return 'READY_STALE'
    return 'READY'


def validated_existing_pool(boards, now=None):
    """纯读：独立查找「已有有效结果」，不受最新请求（WAIT_DAILY/FAILED/RUNNING）遮蔽。

    优先级：READY 任务经 _validated_pool（字节 SHA + pool_matches_job 现行严格绑定）
    → 无任务文件时回退池目录扫描，再按同一绑定校验器核对（成员版本/观察日/数据日/
    source_key/行哈希/盘前时序）。任一校验失败都如实返回原因，绝不强行复用。
    不删改任何任务文件、不重置次数、不触发计算/下载/付费/推送。
    """
    now = _now(now)
    boards = _normalize_boards(boards)
    folder = _jobs_dir(create=False)
    candidates = []
    if folder.is_dir():
        for file in folder.glob('*_*.json'):
            job = _read(file, None)
            if not isinstance(job, dict) or job.get('contract') != CONTRACT:
                continue
            if job.get('state') != 'READY' or job.get('corrupt'):
                continue
            if job.get('boards') != boards:
                continue
            candidates.append((str(job.get('finished_at') or job.get('updated_at') or ''),
                               str(job.get('job_id') or ''), job))
    for _, _, job in sorted(candidates, reverse=True):
        pool = _validated_pool(job)
        if pool is None:
            continue
        return dict(available=True, pool=pool, job=job,
                    validation='STRICT_JOB_BINDING_PASS',
                    display_status=_pool_display_status(pool, now))
    pool = _find_existing_pool(boards)
    if pool is not None:
        binding = dict(boards=boards, data_date=pool.get('data_date'),
                       observation_date=pool.get('observation_date'),
                       source_key=pool.get('source_key'),
                       member_versions=pool.get('constituent_versions'))
        if pool_matches_job(pool, binding):
            return dict(available=True, pool=pool, job=None,
                        validation='POOL_SCAN_BINDING_PASS',
                        display_status=_pool_display_status(pool, now))
        return dict(available=False, pool=None, job=None,
                    validation='POOL_SCAN_BINDING_MISMATCH',
                    detail='池目录扫描命中但现行严格绑定校验未通过；不复用、不宣称就绪')
    reason = 'NO_READY_JOB' if candidates else 'NO_VALIDATED_POOL'
    return dict(available=False, pool=None, job=None, validation=reason,
                detail=('找到 READY 任务但文件/绑定校验全部失败；如实报告，不强行复用'
                        if candidates else '未找到已核验强势池'))


def poll_policy():
    """页面等待上限与后台计算超时分别表达：前者只是前端自动刷新期限，
    后者是后台 owner 超时判定，二者不可互相替代。只读常量，不改任何超时语义。"""
    return dict(interval_ms=3000,
                page_wait_limit_seconds=RUNNING_TIMEOUT_SECONDS + COOLDOWN_SECONDS + 300,
                backend_running_timeout_seconds=RUNNING_TIMEOUT_SECONDS,
                max_attempts=MAX_ATTEMPTS, cooldown_seconds=COOLDOWN_SECONDS,
                note='页面自动刷新暂停不等于后台任务超时或失败；后台超时由 owner/PID/nonce 三态独立判定')


def state(boards=None, now=None):
    """逐板块状态：当前请求状态与已有有效结果分别表达。纯读，不启动计算。

    - current_request：最新请求/任务记录（READY/RUNNING/FAILED/WAIT_DAILY 等）如实表达，
      程序/配置错误带 error_class，不冒充日线缺失。
    - existing_result：独立按现行严格绑定校验后的既有结果；后续失败请求不得把它藏掉，
      校验失败则如实报告 validation 与 detail，不显示虚构数量。
    """
    now = _now(now)
    boards = _normalize_boards(boards)
    job = _find_job_for_boards(boards) if boards else None
    found = validated_existing_pool(boards, now) if boards else dict(
        available=False, pool=None, job=None, validation='NO_BOARDS', detail='未指定板块')
    pool = found.get('pool')
    result_state = found.get('display_status')
    job_state = None
    if job is not None:
        job_state = job.get('state')
        if job_state == 'READY' and result_state != 'READY':
            job_state = result_state
    elif result_state is not None:
        job_state = result_state
    counts = {}
    if pool is not None:
        counts = {n: dict(qualified_count=sum(1 for r in pool.get('rows') or []
                                              if n in (r.get('boards') or [])))
                  for n in boards}
    elif job is not None:
        counts = (job.get('per_board') or {})
    request_state = (job or {}).get('state')
    request_reason = ((job or {}).get('permanent_reason') or (job or {}).get('last_error')
                      or (job or {}).get('reason'))
    current_request = dict(
        state=request_state, job_id=(job or {}).get('job_id'),
        attempt_id=(job or {}).get('attempt_id'),
        attempts=(job or {}).get('attempts'),
        started_at=(job or {}).get('started_at'),
        finished_at=(job or {}).get('finished_at'),
        updated_at=(job or {}).get('updated_at'),
        reason=request_reason,
        error_class=(job or {}).get('error_class'),
        retryable=(job or {}).get('retryable'),
        note=(job or {}).get('note'))
    existing_result = dict(
        available=bool(found.get('available')),
        status=result_state if found.get('available') else None,
        validation=found.get('validation'),
        detail=found.get('detail'),
        qualified_count=len(pool.get('rows') or []) if pool is not None else None,
        pool_sha256=(pool or {}).get('rows_sha256') or (found.get('job') or {}).get('pool_sha256'),
        source_key=(pool or {}).get('source_key') or (found.get('job') or {}).get('source_key'),
        data_date=(pool or {}).get('data_date') or (found.get('job') or {}).get('data_date'),
        observation_date=(pool or {}).get('observation_date') or (found.get('job') or {}).get('observation_date'),
        built_at=(pool or {}).get('built_at'),
        job_id=(found.get('job') or {}).get('job_id'),
        pool_file=(found.get('job') or {}).get('pool_file'))
    rows = []
    legacy_map = _legacy_member_map_async([n for n in boards]) if boards else {}
    for name in boards:
        row = dict(board=name)
        try:
            from ths_board_identity_v1 import resolve_for_pool
            identity = resolve_for_pool(name, _strong_root().parent / 'board_identities')
        except Exception:
            identity = None
        if identity is not None:
            provenance = identity.get('provenance') or {}
            row.update(identity_status=identity['status'],
                       member_count=identity.get('member_count'),
                       member_source=(provenance.get('snapshot_date') or provenance.get('day')),
                       member_provenance=provenance.get('kind'),
                       identity_index_code=(identity.get('identity') or {}).get('index_code'))
        else:
            legacy = legacy_map.get(name) or {}
            row.update(identity_status='LEGACY_CATALOG',
                       member_count=legacy.get('member_count'),
                       member_source=legacy.get('member_source'),
                       member_provenance=legacy.get('member_provenance'),
                       identity_index_code=legacy.get('identity_index_code'))
        row.update(current_request_state=request_state,
                   result_origin='EXISTING_VALIDATED_RESULT' if pool is not None else 'CURRENT_REQUEST')
        if result_state == 'READY':
            qualified = counts.get(name, {}).get('qualified_count', 0)
            row.update(pool_status='READY', qualified_count=qualified,
                       note='筛选完成，0只' if qualified == 0 else None)
        elif result_state == 'READY_STALE':
            qualified = counts.get(name, {}).get('qualified_count', 0)
            row.update(pool_status='READY_STALE', qualified_count=qualified,
                       note='过期结果，仅历史参考，不作当前就绪')
        elif request_state in ('QUEUED', 'RUNNING', 'WAIT_SOURCE', 'PREPARING'):
            row.update(pool_status='PREPARING', qualified_count=None)
        elif request_state == 'FAILED':
            row.update(pool_status='FAILED', qualified_count=None,
                       reason=request_reason,
                       retryable=(job or {}).get('retryable',
                                                not bool((job or {}).get('permanent_reason'))))
        elif request_state == 'WAIT_DAILY':
            row.update(pool_status='WAIT_DAILY', qualified_count=None,
                       reason=request_reason)
        else:
            row.update(pool_status='NOT_PREPARED', qualified_count=None)
        rows.append(row)
    return dict(contract=CONTRACT, job_state=job_state, job_id=(job or {}).get('job_id'),
                attempts=(job or {}).get('attempts'), started_at=(job or {}).get('started_at'),
                finished_at=(job or {}).get('finished_at'),
                pool_sha256=(job or {}).get('pool_sha256'),
                existing_pool=bool(pool is not None and job is None),
                data_date=(job or {}).get('data_date') or (pool or {}).get('data_date'),
                observation_date=(job or {}).get('observation_date') or (pool or {}).get('observation_date'),
                rule_version=(job or {}).get('rule_version'), boards=rows,
                current_request=current_request, existing_result=existing_result,
                result_state=result_state if found.get('available') else None,
                qualified_count=existing_result['qualified_count'],
                poll_policy=poll_policy())


def recover_interrupted(now=None):
    """重启恢复：仅"确认死亡"的 RUNNING 可回收（计一次失败，不删旧证据）；
    unknown 一律不动。纯读目录列表+受锁迁移。"""
    now = _now(now)
    handled = []
    folder = _jobs_dir(create=False)
    if not folder.is_dir():
        return handled
    for file in folder.glob('*.json'):
        job = _read(file, None)
        if not job or job.get('state') != 'RUNNING':
            continue
        if _owner_status(job.get('owner'), now) != 'dead':
            continue
        _decide(job, now)
        handled.append(job.get('job_id'))
    return handled
