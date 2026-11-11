# -*- coding: utf-8 -*-
"""新 UI 接入候选 EXE 专用验收入口（不调用 server.start，不启动监控/采集/worker）。

用法：
  Stargazer.exe --acceptance <sandbox_dir> <output_json>           # 隔离初始化 + 真实保存 + 进程内核验
  Stargazer.exe --acceptance-restart <sandbox_dir> <output_json>   # 仅读回，模拟新进程重启后的持久化核验
两者均由 app.py 在 import server 之前完成隔离（沙箱/边界拦截）后调用。

隔离边界：
- 可写数据（assistant 草案/确认、trade_plan、assistant_plan、isolated 快照）全部重定向到 <sandbox>，
  只读资源（calendar.json / pentacles_demo/snapshot.json / journal）保持安装目录原路径，不写生产。
- 网络(market.quotes)/付费(deepseek_api.chat)/分钟准备(minute_history.schedule) 在边界拦截。
"""
import os
import sys
import json
import time
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime

ACCEPTANCE_ENV = 'STARGAZER_ACCEPTANCE_MODE'

_REDIRECT_PREFIXES = ('data/assistant', 'data/isolated')
_REDIRECT_EQ = ('data/trade_plan.json', 'dist/data/trade_plan.json',
                'data/pentacles_demo/assistant_plan.json')


def _patch_data_file(sandbox):
    """只把可写数据重定向到沙箱；只读资源（calendar/snapshot/journal）保持原路径。"""
    import paths
    orig = paths.data_file

    def patched(name, seed_from_bundle=True):
        if name.startswith(_REDIRECT_PREFIXES) or name in _REDIRECT_EQ:
            return str(sandbox / name)
        return orig(name, seed_from_bundle=seed_from_bundle)

    paths.data_file = patched


def _install_intercepts():
    """防御性边界拦截：行情（网络）、DeepSeek（付费）、分钟准备（后台）。验收只调安全只读路由。"""
    try:
        import market
        market.quotes = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError('acceptance: 网络边界拦截（market.quotes）'))
    except Exception:
        pass
    try:
        import deepseek_api
        deepseek_api.chat = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError('acceptance: 付费边界拦截（deepseek_api.chat）'))
    except Exception:
        pass
    try:
        import minute_history_v1
        minute_history_v1.schedule = lambda *a, **k: None
    except Exception:
        pass


def _clear_sandbox(sandbox):
    for p in (sandbox / 'data' / 'assistant').glob('*.json'):
        try:
            p.unlink()
        except OSError:
            pass
    for rel in _REDIRECT_EQ:
        f = sandbox / rel
        if f.exists():
            try:
                f.unlink()
            except OSError:
                pass


def _seed_trades_db(path, code, buy_date, sell_date):
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    # 与生产 trades 表同列：/api/trade/add 的真实 INSERT 才能在隔离沙箱执行
    conn.execute('CREATE TABLE trades (id INTEGER PRIMARY KEY, ts TEXT, code TEXT, '
                 'trade_date TEXT, action TEXT, name TEXT, price REAL, shares INTEGER, '
                 'amount REAL, reason TEXT, source TEXT, raw TEXT, request_id TEXT, plan_ref TEXT)')
    conn.execute('INSERT INTO trades (id, code, trade_date, action, price, shares) '
                 'VALUES (1,?,?,?,?,?)', (code, buy_date, 'buy', 10.0, 100))
    conn.execute('INSERT INTO trades (id, code, trade_date, action, price, shares) '
                 'VALUES (2,?,?,?,?,?)', (code, sell_date, 'sell', 10.5, 100))
    conn.commit()
    conn.close()


def _iso(day):
    return datetime.strptime(str(day), '%Y%m%d').date().isoformat()


def _safe(v):
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return repr(v)


def _make_check(result):
    def check(name, cond, detail=None):
        result['checks'][name] = {'pass': bool(cond), 'detail': _safe(detail)}
        if not cond:
            result['ok'] = False
            result['errors'].append(name)
    return check


def _run_init(sandbox, out):
    import paths
    import server
    import assistant_workflow as workflow
    import assistant_runtime

    workflow.root = lambda: sandbox / 'data' / 'assistant'
    try:
        from i34_bridge import register_i34_routes
        register_i34_routes(server.app, server.get_cfg)
    except Exception as exc:
        print('[i34] 未注册（可忽略）:', repr(exc))
    _install_intercepts()
    client = server.app.test_client()

    result = {'ok': True, 'phase': 'init', 'checks': {}, 'errors': []}
    check = _make_check(result)

    # ---- 1) 新 UI 及 API 加载 ----
    check('index_200', client.get('/').status_code == 200)
    r = client.get('/api/calendar/next_trading_day?from=20261109')
    j = r.get_json()
    check('calendar_from_equals_200_and_next', r.status_code == 200 and j.get('ok') is True
          and j.get('next_trading_day') == 20261110, j)
    check('calendar_default_today', client.get('/api/calendar/next_trading_day').status_code == 200)
    check('calendar_invalid_rejected_400', client.get(
        '/api/calendar/next_trading_day?from=2026-11-09').status_code == 400)
    rp = client.get('/api/plan/versions')
    j = rp.get_json() or {}
    check('plan_versions_200', rp.status_code == 200 and isinstance(j, dict)
          and 'versions' in j and 'current' in j,
          {'status': rp.status_code, 'versions_count': len(j.get('versions') or []),
           'current_status': (j.get('current') or {}).get('status')})
    rg = client.get('/api/market/global_overview')
    j = rg.get_json() or {}
    check('global_overview_200_no_fake', rg.status_code == 200 and bool(j.get('ok'))
          and j.get('decision_use') is False,
          {'status': rg.status_code, 'ok': j.get('ok'), 'decision_use': j.get('decision_use'),
           'items': len(j.get('items') or [])})
    check('trade_plan_200', client.get('/api/trade_plan').status_code == 200)
    check('assistant_state_200', client.get('/api/assistant/state').status_code == 200)

    # ---- 2) 确认状态闭环：真实 golden 草案 → 真实 confirm_plan 保存 → 读回 ----
    identifier = None
    current = None
    try:
        from plan_rule_profiles import validate as _pp_validate, policy as _pp_policy
        profile = _pp_validate(None)
        policy = _pp_policy(profile)
        real_snap = Path(paths.data_file('data/pentacles_demo/snapshot.json', seed_from_bundle=False))
        ref_hash = __import__('hashlib').sha256(real_snap.read_bytes()).hexdigest()
        raw_snap = json.loads(real_snap.read_text(encoding='utf-8'))
        row = {'code': '600000', 'name': '验收样例', 'analysis_id': 'acceptance',
               'zone': {'zone_lower': 10.0, 'zone_upper': 10.5},
               'entry_rule': {'execution_profile': None, 'combination_policy': policy,
                              'observation_bands': [], 'version': raw_snap.get('rule_version')}}
        snap = {'schema': 'PAGE_OF_PENTACLES_DEMO_V1',
                'data_date': raw_snap.get('data_date'), 'monitor_date': raw_snap.get('monitor_date'),
                'rows': [row], 'rule_inputs': [{'path': str(real_snap), 'sha256': ref_hash}],
                'rule_version': raw_snap.get('rule_version'), 'source': 'USER_GOLDEN_ANALYSIS'}
        current = snap
        draft = dict(origin='GOLDEN_ANALYSIS', snapshot=snap, snapshot_hash=workflow.digest(snap),
                     selected_codes=['600000'], boards=[], manual_observations=[],
                     model_evidence={}, model_commentary={}, reply='验收草案（仅沙箱）',
                     dialogue=[], created_at=datetime.now().isoformat(), analysis_ids=['acceptance'])
        stored = workflow.store('plan', draft)
        identifier = stored['id']
        _orig_gate = assistant_runtime.gate
        _orig_expired = workflow.plan_expired
        _orig_deadline = workflow.plan_deadline
        assistant_runtime.gate = lambda kind, d, approved=None: {
            'ok': True, 'confirmed': True, 'decision': 'HUMAN_CONFIRMED_OBSERVATION_ONLY'}
        workflow.plan_expired = lambda v, now=None: False
        workflow.plan_deadline = lambda v: datetime.now(workflow.PLAN_ZONE).replace(year=2099)
        try:
            resp = client.post('/api/assistant/plan/confirm',
                               json={'id': identifier, 'confirmed': True},
                               content_type='application/json')
        finally:
            assistant_runtime.gate = _orig_gate
            workflow.plan_expired = _orig_expired
            workflow.plan_deadline = _orig_deadline
        check('confirm_http_200', resp.status_code == 200, resp.get_json())
        st = client.get('/api/assistant/state').get_json()
        check('confirmed_true_after_confirm',
              (st.get('plan') or {}).get('confirmed') is True,
              {'plan_confirmed': (st.get('plan') or {}).get('confirmed'),
               'confirmed_status': (st.get('plan') or {}).get('confirmed_status')})
        # 过期拒绝：plan_expired 对过去观察日应为真
        expired = workflow.plan_expired({'monitor_date': 20200101},
                                       now=datetime(2026, 11, 9, tzinfo=workflow.PLAN_ZONE))
        check('plan_expired_true_for_old', expired is True, expired)
    except Exception as exc:
        check('confirmation_flow', False, repr(exc))

    # ---- 3) 复盘保存成功（save-loop）→ 读回；错误观察日拒绝 ----
    rev_id = None
    if current is not None:
        try:
            review_value = dict(day=current.get('monitor_date'), facts={}, facts_hash='acceptance',
                                origin='ACCEPTANCE')
            rev_stored = workflow.store('review', review_value)
            rev_id = rev_stored['id']
            reread = workflow.load_draft('review', rev_id)
            check('review_save_then_readback', reread is not None
                  and reread.get('facts_hash') == 'acceptance', {'facts_hash': reread.get('facts_hash')})
        except Exception as exc:
            check('review_save_flow', False, repr(exc))
        # review_start 是异步：错误观察日应在后台线程内被拒（JOB['error'] 置位），不调用 collect/DeepSeek
        try:
            workflow.JOB.update(running=False, error=None, draft=None)
            workflow.review_start(server.get_cfg(), 20200101)
            deadline = time.time() + 20
            while workflow.JOB.get('running') and time.time() < deadline:
                time.sleep(0.1)
            err = workflow.JOB.get('error')
            check('review_wrong_day_rejected', bool(err), err)
        except Exception as exc:
            check('review_wrong_day_rejected', False, repr(exc))

    # ---- 4) 成交归属当前计划 + 历史成交不得冒充本计划已执行 ----
    try:
        code0 = '600000'
        plan_day = (current or {}).get('monitor_date') or 20261109
        trades_db = sandbox / 'data' / 'trades.db'
        from datetime import timedelta
        _tgt = datetime.strptime(str(plan_day), '%Y%m%d')
        _seed_trades_db(trades_db, code0, (_tgt - timedelta(days=5)).date().isoformat(),
                        _tgt.date().isoformat())
        scope_id = 'acceptance-plan-scope-001'
        _c = sqlite3.connect(str(trades_db))
        _c.execute("INSERT INTO trades (code, trade_date, action, price, shares, plan_ref) "
                   "VALUES (?,?,?,?,?,?)", (code0, _iso(plan_day), 'buy', 10.0, 100, scope_id))
        _c.execute("INSERT INTO trades (code, trade_date, action, price, shares, plan_ref) "
                   "VALUES (?,?,?,?,?,?)", ('600498', '2020-01-02', 'buy', 8.0, 100, None))
        _c.commit()
        _c.close()
        real_cfg = dict(server.get_cfg() or {})
        real_cfg['agent'] = dict(real_cfg.get('agent') or {})
        real_cfg['agent']['trades_db'] = str(trades_db)
        server.get_cfg = lambda: real_cfg  # 仅本进程生效
        plan = dict(monitor_date=plan_day, data_date=plan_day,
                    assistant_plan_id=scope_id,
                    stocks=[{'code': code0}, {'code': '600498'}])
        enriched = server._enrich_execution_status(plan)
        det0 = (enriched['stocks'][0].get('execution_detail') or {})
        check('execution_registered_with_real_trade',
              det0.get('manual_trade_registered') == 'REGISTERED'
              and det0.get('registered_trade_count') == 1, det0)
        det1 = (enriched['stocks'][1].get('execution_detail') or {})
        check('historical_trade_not_current_plan',
              det1.get('manual_trade_registered') == 'HISTORICAL_ONLY'
              and det1.get('registered_trade_count') == 0
              and det1.get('historical_trade_count') == 1
              and det1.get('latest_historical_trade_date') == '2020-01-02', det1)
        scope_bad = dict(plan)
        scope_bad.pop('assistant_plan_id', None)
        det_bad = (server._enrich_execution_status(scope_bad)['stocks'][0]
                   .get('execution_detail') or {})
        check('plan_scope_unavailable_not_registered',
              det_bad.get('manual_trade_registered') == 'PLAN_SCOPE_UNAVAILABLE', det_bad)
    except Exception as exc:
        check('real_trade_enrichment', False, repr(exc))

    # ---- 4b) 成交 request_id 幂等 + 两笔真实成交分别登记 ----
    try:
        base = dict(action='buy', code='600497', name='UI2幂等验收', price=10.0, shares=100,
                    trade_date=_iso(plan_day), source='acceptance')
        b1 = dict(base, request_id='ui2-idem-001')
        r1 = client.post('/api/trade/add', json=b1).get_json()
        r2 = client.post('/api/trade/add', json=b1).get_json()
        b3 = dict(base, request_id='ui2-idem-002')
        r3 = client.post('/api/trade/add', json=b3).get_json()
        _c = sqlite3.connect(str(trades_db))
        n1 = _c.execute("SELECT COUNT(*) FROM trades WHERE request_id=?", ('ui2-idem-001',)).fetchone()[0]
        n2 = _c.execute("SELECT COUNT(*) FROM trades WHERE request_id=?", ('ui2-idem-002',)).fetchone()[0]
        nall = _c.execute("SELECT COUNT(*) FROM trades WHERE code=?", ('600497',)).fetchone()[0]
        _c.close()
        check('trade_request_idempotent',
              r1.get('ok') is True and r2.get('duplicate') is True
              and r1.get('id') == r2.get('id') and n1 == 1,
              {'r1': r1, 'r2': r2, 'rows_for_request': n1})
        check('two_distinct_trades_not_merged',
              r3.get('ok') is True and r3.get('id') != r1.get('id')
              and n2 == 1 and nall == 2,
              {'r3': r3, 'rows_for_code': nall})
    except Exception as exc:
        check('trade_idempotency_flow', False, repr(exc))

    # ---- 4c) 同一草案重复确认幂等：不重复准备/付费/推送，生效计划不变 ----
    try:
        import hashlib as _hl
        import paths as _p
        formal_rel = 'data/trade_plan.json' if _p.FROZEN else 'dist/data/trade_plan.json'
        formal_path = Path(_p.data_file(formal_rel, seed_from_bundle=False))
        before = _hl.sha256(formal_path.read_bytes()).hexdigest() if formal_path.is_file() else None
        n_before = len(list((sandbox / 'data' / 'assistant').glob('confirmed_*.json')))
        _g, _e, _d = assistant_runtime.gate, workflow.plan_expired, workflow.plan_deadline
        assistant_runtime.gate = lambda kind, d, approved=None: {
            'ok': True, 'confirmed': True, 'decision': 'HUMAN_CONFIRMED_OBSERVATION_ONLY'}
        workflow.plan_expired = lambda v, now=None: False
        workflow.plan_deadline = lambda v: datetime.now(workflow.PLAN_ZONE).replace(year=2099)
        try:
            resp2 = client.post('/api/assistant/plan/confirm',
                                json={'id': identifier, 'confirmed': True},
                                content_type='application/json')
        finally:
            assistant_runtime.gate, workflow.plan_expired, workflow.plan_deadline = _g, _e, _d
        after = _hl.sha256(formal_path.read_bytes()).hexdigest() if formal_path.is_file() else None
        n_after = len(list((sandbox / 'data' / 'assistant').glob('confirmed_*.json')))
        _resp2 = resp2.get_json() or {}
        check('duplicate_confirm_idempotent',
              before is not None and before == after and n_before == n_after and n_before == 1
              and _resp2.get('duplicate') is True,
              {'http': resp2.status_code, 'plan_sha_unchanged': before == after,
               'confirmed_markers': [n_before, n_after],
               'second_response': _resp2})
    except Exception as exc:
        check('duplicate_confirm_idempotent', False, repr(exc))

    # ---- 4d) 历史确认记录不得让新草案继承已确认状态（不落盘、不改变当前草案） ----
    try:
        _conf, _status = workflow._ui2_plan_confirmation({'id': 'f' * 64})
        check('historical_confirmation_not_inherited',
              _conf is False and isinstance(_status, str),
              {'confirmed': _conf, 'status': _status})
    except Exception as exc:
        check('historical_confirmation_not_inherited', False, repr(exc))

    # ---- 4e) 损坏确认记录 / 缺字段 / 正式计划缺失一律不可核验 ----
    try:
        _marker = sandbox / 'data' / 'assistant' / ('confirmed_%s.json' % identifier)
        _saved = _marker.read_bytes() if _marker.is_file() else None
        broken = [
            ('corrupt_json', b'{ this is not json'),
            ('null_record', b'null'),
            ('missing_fields', b'{}'),
            ('identity_mismatch', json.dumps(
                {'draft_id': 'other-draft', 'plan': {}, 'confirmed_at': '2026-11-09T09:00:00'}
            ).encode('utf-8')),
        ]
        _states = {}
        for _name, _raw in broken:
            _marker.write_bytes(_raw)
            _conf, _status = workflow._ui2_plan_confirmation({'id': identifier})
            _states[_name] = {'confirmed': _conf, 'status': _status}
        if _saved is not None:
            _marker.write_bytes(_saved)
        elif _marker.is_file():
            _marker.unlink()
        check('corrupt_confirmation_unverifiable',
              all(v['confirmed'] is False
                  and str(v['status']).startswith('CONFIRMATION_UNVERIFIABLE_')
                  for v in _states.values()), _states)
        _conf_ok, _status_ok = workflow._ui2_plan_confirmation({'id': identifier})
        check('valid_confirmation_still_confirmed',
              _conf_ok is True and _status_ok == 'CONFIRMED_CURRENT',
              {'confirmed': _conf_ok, 'status': _status_ok})
    except Exception as exc:
        check('confirmation_fail_closed_flow', False, repr(exc))

    # ---- 4f) 真实 /api/trade/add 写入服务端权威计划归属 ----
    try:
        _c = sqlite3.connect(str(sandbox / 'data' / 'trades.db'))
        _refs = _c.execute("SELECT DISTINCT plan_ref FROM trades "
                            "WHERE request_id IN ('ui2-idem-001','ui2-idem-002')").fetchall()
        _c.close()
        check('trade_plan_ref_recorded', len(_refs) == 1 and bool(_refs[0][0])
              and str(_refs[0][0]) == str(identifier), _refs)
    except Exception as exc:
        check('trade_plan_ref_recorded', False, repr(exc))

    # ---- 触发新进程重启读回核验（独立子进程） ----
    restart_out = out.with_suffix('.restart.json')
    try:
        if getattr(sys, 'frozen', False):
            args = [sys.executable, '--acceptance-restart', str(sandbox), str(restart_out)]
        else:
            args = [sys.executable, str(Path(__file__).with_name('app.py')),
                    '--acceptance-restart', str(sandbox), str(restart_out)]
        child = subprocess.run(args, capture_output=True, text=True, timeout=300)
        if restart_out.exists():
            restart_res = json.loads(restart_out.read_text(encoding='utf-8'))
            check('restart_persistence', restart_res.get('ok') is True,
                  {'checks': restart_res.get('checks'), 'errors': restart_res.get('errors')})
        else:
            check('restart_persistence', False, (child.stdout + child.stderr)[-400:])
    except Exception as exc:
        check('restart_persistence', False, repr(exc))

    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def _run_restart(sandbox, out):
    """模拟新进程重启：重新隔离沙箱，读回持久化确认/复盘，验证跨进程保留。"""
    import server
    import assistant_workflow as workflow

    workflow.root = lambda: sandbox / 'data' / 'assistant'
    _install_intercepts()
    client = server.app.test_client()

    result = {'ok': True, 'phase': 'restart', 'checks': {}, 'errors': []}
    check = _make_check(result)

    st = client.get('/api/assistant/state').get_json()
    check('confirmed_preserved_after_restart',
          (st.get('plan') or {}).get('confirmed') is True
          and str((st.get('plan') or {}).get('confirmed_status')) == 'CONFIRMED_CURRENT',
          {'plan_confirmed': (st.get('plan') or {}).get('confirmed')})
    rev_files = sorted((sandbox / 'data' / 'assistant').glob('review_*.json'))
    check('review_preserved_after_restart', len(rev_files) >= 1, [p.name for p in rev_files])
    try:
        rev_vals = [workflow.load_draft('review', f.stem.split('_', 1)[1])
                    for f in rev_files]
        check('review_content_readback_after_restart',
              any((v or {}).get('facts_hash') == 'acceptance' for v in rev_vals),
              [str((v or {}).get('facts_hash')) for v in rev_vals])
    except Exception as exc:
        check('review_content_readback_after_restart', False, repr(exc))
    try:
        _c = sqlite3.connect(str(sandbox / 'data' / 'trades.db'))
        rows = _c.execute(
            "SELECT id, request_id FROM trades WHERE request_id IN ('ui2-idem-001','ui2-idem-002') ORDER BY id"
        ).fetchall()
        _c.close()
        check('trades_preserved_after_restart', len(rows) == 2, rows)
    except Exception as exc:
        check('trades_preserved_after_restart', False, repr(exc))

    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def run(mode, argv):
    sandbox = Path(argv[0]).resolve()
    out = Path(argv[1]) if len(argv) > 1 else (sandbox / 'acceptance_result.json')
    os.environ[ACCEPTANCE_ENV] = '1'
    os.environ['STARGAZER_GLOBAL_OVERVIEW_SNAPSHOT'] = str(
        sandbox / 'data' / 'isolated' / 'global_overview_snapshot.json')
    if mode != '--acceptance-restart':
        sandbox.mkdir(parents=True, exist_ok=True)
        _clear_sandbox(sandbox)
    _patch_data_file(sandbox)
    if mode == '--acceptance-restart':
        return _run_restart(sandbox, out)
    return _run_init(sandbox, out)


if __name__ == '__main__':
    raise SystemExit(0 if run(sys.argv[1], sys.argv[2:]).get('ok') else 1)
