"""Explicit frozen-tool acceptance mode; never starts the server or monitors."""
import json,os,socket,subprocess,sys,time
from pathlib import Path

def run(output):
    output=Path(output)
    if output.exists():raise ValueError('SELF_CHECK_OUTPUT_EXISTS')
    project=next((p for p in Path(sys.executable).resolve().parents if (p/'stargazer_ml').is_dir()),None)
    if project is None:raise ValueError('SELF_CHECK_PROJECT_NOT_FOUND')
    import paths
    paths.APP_DIR=str(project/'stargazer_desktop/dist')
    paths.FROZEN=True
    temporary=project/'stargazer_desktop/dist/data/postclose_risk/tmp'
    temporary.mkdir(exist_ok=True,parents=True)
    os.environ['TEMP']=os.environ['TMP']=str(temporary)
    import tempfile
    tempfile.tempdir=str(temporary)
    def denied(*args,**kwargs):raise AssertionError('SELF_CHECK_NETWORK_FORBIDDEN')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    os.environ['OMP_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='1'
    guard=project/'stargazer_ml/v5_hybrid/risk_r0_v1/tdx_postclose_publish_20261031_v1/frozen_tools_v79/guard_worker.py'
    if not guard.is_file():raise ValueError('SELF_CHECK_GUARD_MISSING')
    original_run=subprocess.run
    def guarded_run(command,*args,**kwargs):
        command=list(command)
        if Path(str(command[0])).name.lower().startswith('python'):
            script=next((i for i,v in enumerate(command) if str(v).endswith('.py')),None)
            if script is not None:command.insert(script,str(guard))
        return original_run(command,*args,**kwargs)
    subprocess.run=guarded_run
    import minute_plan_review as review
    import runtime_disk_limit_v1 as storage
    import postclose_risk_runtime as runtime
    import conversation_plans
    from agent.brain import Brain
    brain=Brain.__new__(Brain);brain.cfg={};brain.messages=[]
    brain.archive=output.parent/'frozen_acceptance_session.jsonl'
    result=dict(frozen=bool(getattr(sys,'frozen',False)),paid_calls=0,monitor_started=False,
        actual_network=0,source='LOCAL_TDX_MINUTES_AND_CORPORATE_EVENT_DATES')
    expected_current_day=int(json.loads((project/'stargazer_desktop/dist/data/pentacles_demo/snapshot.json').read_bytes())['data_date'])
    result['reference_days']=dict(historical_replay=20261030,current_tools=expected_current_day)
    started=time.perf_counter()
    try:
        if result['frozen']:
            bundle=Path(sys._MEIPASS).resolve()
            for m in (review,storage,conversation_plans):
                if not Path(m.__file__).resolve().is_relative_to(bundle):raise ValueError('NOT_BUNDLED:'+m.__name__)
            if (bundle/'contracts/research_storage_4gib_20261015.json').exists():raise ValueError('RESEARCH_CONTRACT_IN_BUNDLE')
        value=brain._dispatch('review_plan_minutes',dict(message='复盘交易计划和人工名单',day=20261030))
        statuses={r['code']:dict(status=r['status'],reason=r.get('reason')) for r in value.get('rows',[])}
        result['review']=dict(status=value.get('status'),rows=statuses,evidence_file=value.get('evidence_file'),
            http_attempts=value.get('http_attempts'),reply=value.get('reply'),assessment_file=value.get('assessment_file'))
        if len(statuses)!=10 or any(r['status']!='REPLAYED' for r in statuses.values()):raise ValueError('REVIEW_ROWS_NOT_ALL_REPLAYED')
        if value.get('http_attempts')!=0:raise ValueError('REVIEW_HTTP_USED')
        view=conversation_plans.tool_view(value)
        if view.get('kind')!='MINUTE_REVIEW_VIEW_V1' or len(view['rows'])!=10:raise ValueError('REVIEW_NOT_READABLE')
        confirmed=conversation_plans.tool_view(brain._dispatch('confirmed_plan',{}))
        if confirmed.get('kind')!='CONFIRMED_PLAN_VIEW_V1':raise ValueError('CONFIRMED_PLAN_NOT_READABLE')
        result['confirmed_plan']=dict(kind=confirmed['kind'],stocks=len(confirmed['stocks']))
        golden=brain._dispatch('stock_golden_levels',dict(code='002313'))
        result['golden']={k:golden.get(k) for k in ('status','reason','data_date','requested_data_date','total_bands')}
        if golden.get('status')=='UNAVAILABLE' or golden.get('data_date')!=expected_current_day:raise ValueError('CURRENT_GOLDEN_UNAVAILABLE')
        board=brain._dispatch('scan_sector_rules',dict(board_name='存储芯片',top=5))
        result['current_board']={k:board.get(k) for k in ('status','reason','formal_count','manual_review_count','returned_count','data_date')}
        if board.get('status')!='READY' or board.get('data_date')!=expected_current_day:raise ValueError('CURRENT_BOARD_TOOL_FAILED')
        result['current_daily_ready']=board.get('status')=='READY'
        result['daily_source_integration_accepted']=True
        result['price_basis']='RAW_MODEL_AND_EXECUTION_SEPARATE_FROM_P_FIXED_QFQ'
        # Use the real newly computed P30 snapshot as explicit test context;
        # never replace the user's persisted snapshot or confirmed plan.
        import assistant_workflow
        old_snapshot=assistant_workflow.snapshot
        current_file=project/'stargazer_desktop/dist/data/postclose_risk/tdx_daily_acceptance/v86/SELECTION_P30.json'
        current=json.loads(current_file.read_bytes())
        if current.get('data_date')!=20261030:raise ValueError('CURRENT_SELECTION_CONTEXT_MISSING')
        assistant_workflow.snapshot=lambda:current
        try:inferred=brain._dispatch('model_stock_inference',dict(code='002436',model='tarot'))
        finally:assistant_workflow.snapshot=old_snapshot
        result['single_stock']={k:inferred.get(k) for k in ('status','inference_status','inference_reason')}
        result['single_stock']['on_demand']=inferred.get('on_demand')
        if not isinstance(inferred.get('on_demand'),dict):raise ValueError('SINGLE_STOCK_WORKER_NOT_CALLED')
        if (inferred['on_demand'].get('selection') or {}).get('quality_state') not in ('OK','NOT_ADMITTED'):
            raise ValueError('SINGLE_STOCK_INFERENCE_FAILED')
        snapshot=json.loads((project/'stargazer_desktop/dist/data/pentacles_demo/snapshot.json').read_bytes())
        _,worker=runtime._worker_paths_real()
        choice=runtime._native_source_choice(snapshot,runtime.directory(),worker)
        result['native_preflight']=choice
        if not choice or choice.get('ready') is not True:raise ValueError('NATIVE_PREFLIGHT_NOT_READY')
        research=[n for n in ('research_budget_v2','joint_shadow_snapshot_v1','tdx_index_acceptance_v1') if n in sys.modules]
        result['research_imports']=research
        if research:raise ValueError('RESEARCH_STACK_IMPORTED_IN_GUI')
        from self_check_flow_v1 import exercise
        result['full_loop']=exercise(project,value['evidence_file'])
        result['passed']=True
    except Exception as exc:
        result.update(passed=False,error=type(exc).__name__+':'+str(exc))
        raise
    finally:
        result['seconds']=round(time.perf_counter()-started,3)
        with output.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
