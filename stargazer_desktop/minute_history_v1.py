"""Shared fixed-day minute preparation for plans and sector observations."""
from pathlib import Path
import hashlib,json,threading,time
import os,subprocess,sys
from contextlib import contextmanager
LOCK=threading.RLock()
_thread=None
_last={}
_pending={}
CONTRACT='FIXED_DAY_MINUTE_CONTEXT_V1'

@contextmanager
def cache_lock(folder):
    import msvcrt
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'read.lock').open('a+b') as stream:
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        try:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise ValueError('HISTORY_PREPARING')
        try:yield
        finally:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)

def root():
    import postclose_risk_runtime
    return postclose_risk_runtime.directory()/'minute_context'

def _write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf8');temporary.replace(path)

def day_minutes(code,day,budget):
    code=str(code).zfill(6);day=int(day)
    if len(code)!=6 or not code.isdigit() or len(str(day))!=8:raise ValueError('HISTORY_KEY')
    from tdx_native_history_v1 import configured,day_minutes as native_day
    if not configured():raise ValueError('TDX_HISTORY_SOURCE_NOT_CONFIGURED')
    return native_day(code,day,budget,root().parent)


def load_days(code,days,budget):
    rows=[];receipts=[]
    for day in days:
        values,ref=day_minutes(code,int(day),budget);rows.extend(values);receipts.append(ref)
    return rows,receipts

class DiskBudget:
    """One directory scan per worker; account the worker's new writes explicitly."""
    def __init__(self):
        import postclose_risk_runtime
        from runtime_disk_limit_v1 import disk_limit,disk_usage
        base=postclose_risk_runtime.directory()
        self.used=disk_usage(base)
        self.limit=disk_limit(base)
    def __call__(self,size):
        if self.used+size+65536>self.limit:raise ValueError('RUNTIME_DISK_BUDGET')
    def written(self,size):self.used+=size

def schedule(plan):
    """Run independently of confirmation; preparation never enables reminders."""
    global _thread
    P=plan.get('data_date');E=plan.get('monitor_date',plan.get('observation_date'))
    rows=plan.get('rows',plan.get('stocks',[]))
    if not P or not E or not rows:return
    import hashlib
    cohort=sorted((r['code'],list((r.get('entry_rule') or r).get('history_days') or [int(P)]),r.get('turnover_reference')) for r in rows)
    source_version=None
    if any(item[2] is not None for item in cohort):
        state_file=root().parent/'state.json'
        state=json.loads(state_file.read_bytes()) if state_file.exists() else {}
        if state.get('status')=='COMPLETE' and state.get('acceptance_status')=='PASS':source_version=[state.get('run_dir'),state.get('packs_sha256')]
    key=hashlib.sha256(json.dumps([P,E,cohort,source_version],sort_keys=True).encode()).hexdigest()
    target=root()/('prepared_'+str(E)+'_'+key+'.json')
    with LOCK:
        if target.exists() and json.loads(target.read_bytes()).get('status')=='READY':return target
        if time.monotonic()-_last.get(key,-1000)<60:return target
        _pending[key]=(target,P,E,cohort)
        if _thread and _thread.is_alive():return target
        def work():
            while True:
                with LOCK:
                    if not _pending:return
                    k,(target,P,E,cohort)=_pending.popitem();_last[k]=time.monotonic()
                from joint_risk_bridge import _worker_paths
                python,worker=_worker_paths();project=worker.parents[3]
                request=dict(project=str(project),target=str(target),P=P,E=E,cohort=cohort)
                try:
                    proc=subprocess.run([str(python),'-B','-X','utf8',str(project/'stargazer_desktop/minute_history_v1.py')],
                        input=json.dumps(request),capture_output=True,text=True,timeout=900,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
                        env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
                    if proc.returncode:raise ValueError('MINUTE_PREPARE_WORKER_FAILED')
                except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
                    _write(target,dict(status='WAIT_SOURCE',data_date=P,monitor_date=E,reason=str(exc)[:160]))
        _thread=threading.Thread(target=work,daemon=True,name='fixed-minute-preparation');_thread.start()
    return target

def prepare(request):
    project=Path(request['project'])
    sys.path[:0]=[str(project/'stargazer_desktop'),str(project/'stargazer_ml/v5_hybrid/green_common_profile'),
        str(project/'stargazer_ml/v5_hybrid/risk_r0_v1')]
    import paths
    paths.APP_DIR=str(project/'stargazer_desktop/dist');paths.FROZEN=True
    target=Path(request['target'])
    if root().resolve() not in target.resolve().parents:raise ValueError('PREPARE_TARGET')
    budget=DiskBudget()
    result=dict(status='PREPARING',data_date=request['P'],monitor_date=request['E'],rows={},contract=CONTRACT)
    _write(target,result)
    for item in request['cohort']:
        code,days=item[:2]
        turnover_spec=item[2] if len(item)>2 else None
        try:
            values,refs=load_days(code,days,budget)
            from golden_ladder_v3 import completed_bars
            from strong_low_indicators_v1 import pair,aggregate15
            previous,older,_,_=completed_bars(values,code,days[-1],int(days[-1])*1000000+150000)
            bars=older+previous
            # Use a copied last-day reference as the query day only to materialize
            # historical indicator values; never infer next-day prices/signals.
            baselines={}
            for period,series in [(5,bars),(15,aggregate15(bars))]:
                compared=pair(series,int(days[-1]),int(days[-1]),period)
                baselines[str(period)]=compared.get('previous')
            result['rows'][code]=dict(status='READY',days=days,minute_rows=len(values),receipts=refs,previous_low_indicators=baselines)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            result['rows'][code]=dict(status='WAIT_SOURCE',reason=str(exc)[:160])
        if turnover_spec is not None:
            from relative_turnover_v1 import prepare as prepare_turnover
            result['rows'][code]['turnover_reference']=prepare_turnover(root().parent,code,turnover_spec,budget)
    result['status']='READY' if all(x['status']=='READY' for x in result['rows'].values()) else 'PARTIAL'
    _write(target,result)
    return result['status']

if __name__=='__main__':print(prepare(json.load(sys.stdin)))
