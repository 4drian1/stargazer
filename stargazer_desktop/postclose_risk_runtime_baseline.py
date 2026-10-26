"""Start the local post-close worker and expose only snapshot-bound evidence."""
from dataclasses import fields
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import threading

import paths

_thread = None
_cache = {}


def directory():
    return Path(paths.data_file('data/postclose_risk' if paths.FROZEN else
                               'dist/data/postclose_risk', seed_from_bundle=False))


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
        return is_projection_of(snapshot,refreshed)
    return False


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
                day=int(snapshot['data_date'])
                index=index_minutes_v1.get_index_freshness()
                if (day==int(now.strftime('%Y%m%d')) and now.hour>=15
                        and index.get('last_date')==day and not index.get('stale')):
                    current=state()
                    if needs_rebuild(snapshot,current):
                        python,worker=_worker_paths()
                        worker=worker.with_name('live_postclose_job_v1.py')
                        root=directory()
                        root.mkdir(parents=True,exist_ok=True)
                        # This worker owns a cross-process lock and its terminal.
                        # Never create a second daily job simply because UI restarted.
                        with (root/'worker_output.log').open('ab') as log:
                            subprocess.run([str(python),'-B','-X','utf8',str(worker),
                                '--snapshot',str(file),'--runtime',str(root)],
                                stdout=log,stderr=log,timeout=10800,
                                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                            if state().get('status')=='COMPLETE':
                                checker=worker.with_name('accept_live_postclose_v1.py')
                                subprocess.run([str(python),'-B','-X','utf8',str(checker),
                                    '--runtime',str(root)],stdout=log,stderr=log,timeout=600,
                                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    elif current.get('status')=='COMPLETE' and current.get('acceptance_status')!='PASS':
                        python,worker=_worker_paths()
                        root=directory()
                        with (root/'worker_output.log').open('ab') as log:
                            subprocess.run([str(python),'-B','-X','utf8',
                                str(worker.with_name('accept_live_postclose_v1.py')),
                                '--runtime',str(root)],stdout=log,stderr=log,timeout=600,
                                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            except (OSError,ValueError,KeyError,subprocess.TimeoutExpired):
                pass
            time.sleep(30)
    _thread=threading.Thread(target=loop,name='postclose-risk-launcher',daemon=True)
    _thread.start()
