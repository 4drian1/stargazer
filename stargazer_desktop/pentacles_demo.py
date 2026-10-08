"""Isolated common-rule UI and next-day quote reminders; no orders or old state."""
from datetime import datetime
import json
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


def _default_daily(root):
    """日线分片目录：env STARGAZER_DAILY_PARTS → 仓库根 data_daily_parts（监控机布局）
    → 本机 D:/a股数据 缺省（与 build_pool/refresh_pool 同模式）。"""
    import os
    v = os.environ.get('STARGAZER_DAILY_PARTS')
    if v:
        return Path(v)
    for cand in (Path(root) / 'data_daily_parts', Path('D:/a股数据/data_daily_parts')):
        if cand.exists():
            return cand
    return Path('D:/a股数据/data_daily_parts')


def state():
    with _lock:
        reminder = monitor.state()
        result = {'running': _running, 'error': _error, 'enabled': reminder['enabled'],
                  'live_ready': False, 'reminder': reminder,
                  'journal': monitor.journal.state(),
                  'monitor_scope': '次日到价提醒；不是实时买卖信号',
                  'snapshot': None}
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
    script = root / 'stargazer_ml/v5_hybrid/green_common_profile/gcp_demo_snapshot.py'
    python = Path(options.get('python') or root / '_dev_venv/Scripts/python.exe')
    daily = Path(options.get('daily_dir') or _default_daily(root))
    if not script.is_file() or not python.is_file():
        raise ValueError('重算脚本或 Python 不存在；监控机可先读取迁移的 Demo 快照，或配置 pentacles_demo 路径')
    with _lock:
        if _running:
            return {'ok': False, 'error': '分析正在运行，请勿重复启动'}
        _running, _error = True, None

    def work():
        global _running, _error
        try:
            refresh = root / 'stargazer_ml/v5_hybrid/green_v2/refresh_pool.py'
            if refresh.is_file():
                import os
                env = dict(os.environ, PYTHONIOENCODING='utf-8',
                           STARGAZER_DAILY_PARTS=str(daily))
                rp = subprocess.run([str(python), '-B', '-X', 'utf8', str(refresh)],
                                    capture_output=True, timeout=900, env=env,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if rp.returncode:
                    raise RuntimeError('日线增量(refresh_pool)失败：'
                                       + rp.stderr.decode('utf-8', errors='replace')[-400:])
            proc = subprocess.run([str(python), '-B', '-X', 'utf8', str(script),
                                   '--daily-dir', str(daily)], capture_output=True, timeout=900,
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
            monitor.journal.register(payload)
        except Exception as exc:
            with _lock:
                _error = str(exc)
        finally:
            with _lock:
                _running = False
    threading.Thread(target=work, daemon=True, name='pentacles-demo-analysis').start()
    return {'ok': True}
