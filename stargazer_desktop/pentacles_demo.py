"""Isolated common-rule UI and next-day quote reminders; no orders or old state."""
from datetime import datetime
import json
import hashlib
from pathlib import Path
import subprocess
import threading

import paths
from pentacles_monitor import PriceMonitor

_lock = threading.Lock()
_running = False
_error = None
MAX_BYTES = 1024 * 1024
monitor = PriceMonitor(lambda: snapshot_path().parent)


def snapshot_path():
    return Path(paths.data_file('data/pentacles_demo/snapshot.json', seed_from_bundle=False))


def _analysis_status(value=None):
    file = snapshot_path().with_name('analysis_status.json')
    if value is None:
        value = json.loads(file.read_bytes()) if file.exists() else {}
        if not isinstance(value, dict):raise ValueError('ANALYSIS_STATUS_INVALID')
        return value
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False), encoding='utf8')
    temp.replace(file)


def monitoring_plan():
    """A human-confirmed selection takes precedence; never silently replace it."""
    file = snapshot_path().with_name('assistant_plan.json')
    if file.exists():
        try:
            return json.loads(file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
    return state().get('snapshot')


def _default_daily(root):
    """The worker prepares versioned TDX data inside the original runtime budget."""
    return Path(root) / 'stargazer_desktop/dist/data/postclose_risk/tdx_daily_v2'


def state():
    with _lock:
        reminder = monitor.state()
        result = {'running': _running, 'error': _error, 'enabled': reminder['enabled'],
                  'live_ready': False, 'reminder': reminder,
                  'journal': monitor.journal.state(),
                  'monitor_scope': '次日到价提醒；不是实时买卖信号',
                  'snapshot': None}
        status = {}
        try:
            status = _analysis_status()
            result['analysis_status'] = status.get('status')
            result['using_previous_result'] = status.get('status') in ('FAILED', 'RUNNING')
            if not _running and status.get('status') in ('FAILED', 'RUNNING', 'PREPARATION_FAILED'):
                result['error'] = status.get('error') or '上次分析中断，尚未产生新结果，请重新分析'
        except (OSError, ValueError):
            result.update(error='分析任务状态不可读取，不能确认当前结果是否为最新', using_previous_result=True)
        file = snapshot_path()
        if file.exists():
            try:
                if file.stat().st_size > MAX_BYTES:
                    raise ValueError('Demo 快照超过 1MiB')
                payload = json.loads(file.read_text(encoding='utf-8'))
                if payload.get('schema') != 'PAGE_OF_PENTACLES_DEMO_V1' or not isinstance(payload.get('rows'), list):
                    raise ValueError('Demo 快照版本或结构不兼容')
                result['snapshot'] = payload
                result['stale'] = not payload.get('plan_ready') or int(payload.get('monitor_date') or 0) < int(datetime.now().strftime('%Y%m%d'))
                if status.get('snapshot_sha256') and status['snapshot_sha256'] != hashlib.sha256(file.read_bytes()).hexdigest():
                    result.update(error='当前快照与最近成功任务不匹配，请重新分析', using_previous_result=True)
            except Exception as exc:
                result['error'] = str(exc)
        return result


def set_enabled(value):
    monitor.enable(value)
    return state()


def run(cfg):
    global _running, _error
    options = cfg.get('pentacles_demo') or {}
    app_dir = Path(paths.APP_DIR)
    root = Path(options['project_root']) if options.get('project_root') else next(
        (parent for parent in app_dir.parents if (parent / 'stargazer_ml').is_dir()), app_dir.parent)
    script = root / 'stargazer_ml/v5_hybrid/green_common_profile/tarot_live_v2.py'
    python = Path(options.get('python') or root / '_dev_venv/Scripts/python.exe')
    daily = Path(options.get('daily_dir') or _default_daily(root))
    if not script.is_file() or not python.is_file():
        raise ValueError('重算脚本或 Python 不存在；监控机可先读取迁移的 Demo 快照，或配置 pentacles_demo 路径')
    with _lock:
        if _running:
            return {'ok': False, 'error': '分析正在运行，请勿重复启动'}
        _analysis_status(dict(status='RUNNING', started_at=datetime.now().isoformat()))
        _running, _error = True, None

    def work():
        global _running, _error
        published = False
        try:
            proc = subprocess.run([str(python), '-B', '-X', 'utf8', str(script),
                                   '--daily-dir', str(daily),'--golden-flow-v3'], capture_output=True, timeout=900,
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if proc.returncode:
                raise RuntimeError(proc.stderr.decode('utf-8', errors='replace')[-700:])
            payload = json.loads(proc.stdout.decode('utf-8'))
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
            if len(encoded) > MAX_BYTES:
                raise ValueError('Demo 快照超过 1MiB，停止写入')
            target = snapshot_path()
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix('.tmp')
            temp.write_bytes(encoded)
            temp.replace(target)
            published = True
            _analysis_status(dict(status='SUCCEEDED', finished_at=datetime.now().isoformat(),
                                  data_date=payload.get('data_date'), monitor_date=payload.get('monitor_date'),
                                  snapshot_sha256=hashlib.sha256(encoded).hexdigest()))
            monitor.journal.register(payload)
            from postclose_prepare_v1 import after_selection
            after_selection(payload)
        except Exception as exc:
            with _lock:
                _error = ('选股已完成，后续准备失败：' if published else '') + str(exc)
                _analysis_status(dict(status='PREPARATION_FAILED' if published else 'FAILED',
                                      finished_at=datetime.now().isoformat(), error=_error))
        finally:
            with _lock:
                _running = False
    threading.Thread(target=work, daemon=True, name='pentacles-demo-analysis').start()
    return {'ok': True}
