"""LangGraph runs out-of-process; quote/risk threads never import graph packages."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import paths
import assistant_workflow as workflow


def gate(kind, draft, *, approved=None):
    with workflow.LOCK:
        workflow.reclaim_records()
        total = sum(p.stat().st_size for p in workflow.root().rglob('*') if p.is_file())
        db = workflow.root() / 'graph_checkpoint.sqlite'
        from persistent_records_v1 import is_record, reserve_space
        persistent = is_record(workflow.root())
        if persistent:reserve_space(workflow.root(),1024*1024)
        if not persistent and total + max(0, 1024 * 1024 - (db.stat().st_size if db.exists() else 0)) > workflow.LIMIT:
            raise ValueError('助手图恢复预算不足；报价和持仓风控继续运行')
        # Dedicated interpreter, never sys.executable (that would recursively launch the exe).
        app = Path(paths.APP_DIR).resolve()
        project = app.parent.parent if app.name == 'dist' else app.parent
        python = Path(os.environ.get('STARGAZER_PYTHON') or project / '_dev_venv/Scripts/python.exe')
        worker = Path(paths.resource('assistant_graph_worker.py'))
        if not python.is_file() or not worker.is_file():
            raise ValueError('LangGraph独立运行环境未就绪，草案保留；确认暂停，报价和持仓风控不受影响')
        local_zone = timezone(timedelta(hours=8))
        expires = (workflow.plan_deadline(draft['snapshot']) if kind == 'plan' else
            datetime.fromisoformat(draft['created_at']).replace(tzinfo=local_zone) + timedelta(days=1))
        if approved is True and datetime.now(local_zone) >= expires:
            raise ValueError('草案已过期，请重新生成；旧记录保留')
        body = {'root': str(workflow.root()), 'kind': kind, 'draft_id': draft['id'], 'persistent_records':persistent,
                'action': 'prepare' if approved is None else 'confirm', 'approved': approved,
                'expires_at': expires.isoformat(), 'now': datetime.now(local_zone).isoformat()}
        try:
            completed = subprocess.run([str(python), '-B', '-X', 'utf8', str(worker)],
                input=json.dumps(body), capture_output=True, text=True, encoding='utf-8', timeout=25,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            result = json.loads(completed.stdout)
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            raise ValueError('助手编排暂不可用，草案保留；不影响报价与持仓风控') from None
        if completed.returncode or not result.get('ok'):
            raise ValueError('助手编排验收失败，未确认；不影响报价与持仓风控')
        if approved is True and result['decision'] != 'HUMAN_CONFIRMED_OBSERVATION_ONLY':
            raise ValueError('草案过期或未确认，请重新生成')
        return result
