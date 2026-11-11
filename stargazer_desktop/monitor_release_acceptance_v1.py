"""Explicit offline release acceptance; redirects before any business import."""
import hashlib
import json
import os
from pathlib import Path
import socket
from datetime import datetime, timedelta, timezone
from unittest.mock import patch


def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    sandbox = output.parent / 'acceptance_state'
    sandbox.mkdir(parents=True, exist_ok=True)
    import paths
    import sys
    if not getattr(sys, 'frozen', False) and (Path(__file__).resolve().parent / 'web').is_dir():
        paths.BUNDLE_DIR = str(Path(__file__).resolve().parent)
    paths.APP_DIR = str(sandbox)
    paths.FROZEN = True

    def data_file(name, **kwargs):
        target = sandbox / name
        target.parent.mkdir(parents=True, exist_ok=True)
        return str(target)

    paths.data_file = data_file
    cfg = {'agent': {'trades_db': str(sandbox / 'trades.sqlite')},
           'push': {'enabled': False}, 'plan': {'enabled': False}}
    data_file('config.yaml')
    (sandbox / 'config.yaml').write_text('push:\n  enabled: false\n', encoding='utf8')
    results = {}

    def check(name, condition):
        results[name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    def no_network(*args, **kwargs):
        raise AssertionError('ACCEPTANCE_EXTERNAL_NETWORK_FORBIDDEN')

    with patch.object(socket.socket, 'connect', no_network), \
         patch.object(socket.socket, 'connect_ex', no_network):
        import server
        import assistant_workflow as workflow
        import trade_plan_v1
        import postclose_risk_runtime as risk
        import pentacles_demo
        import plan_board_change_v1 as boards
        import ths_board_identity_v1 as ths
        server.get_cfg = lambda: cfg
        risk.directory = lambda: sandbox / 'risk'
        workflow.learning_path = lambda: sandbox / 'PENTACLES_LEARNING.md'
        server.register_extension_routes()
        server.register_extension_routes()  # no duplicate Flask registration
        client = server.app.test_client()
        for page in ('/strong-pullback', '/strong-pullback/fills', '/local-storage', '/i34'):
            check('panel_' + page, client.get(page).status_code == 200)
        for endpoint in ('/api/strong_pullback/boards', '/api/strong_pullback/fills', '/api/i34/panel'):
            check('api_' + endpoint, client.get(endpoint).status_code == 200)
        now = datetime.now(timezone(timedelta(hours=8)))
        summary = sandbox / 'risk/strong_sector/summary.json'
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps({'pool_day': int(now.strftime('%Y%m%d')),
                                      'boards': [{'name': 'pcb', 'valid_count': 0, 'pool_count': 0}]}), encoding='utf8')
        approved = client.post('/api/strong_pullback/boards', json={'board': 'pcb', 'approved': True})
        check('board_environment_confirm_real_save', approved.status_code == 200 and
              approved.get_json()['approvals']['boards']['pcb'] is True)
        readback = client.get('/api/strong_pullback/boards').get_json()
        check('board_environment_readback', readback['approvals']['boards']['pcb'] is True)
        check('board_environment_reject_unknown', client.post('/api/strong_pullback/boards',
              json={'board': '不存在板块', 'approved': True}).status_code == 400)
        check('local_storage_api', client.get('/api/local-storage').status_code == 200)
        future = (now + timedelta(days=1)).strftime('%Y%m%d')
        file = trade_plan_v1.TRADE_PLAN_FILE
        base = dict(version=1, created_ts=now.strftime('%Y%m%d%H%M%S'),
                    data_date=now.strftime('%Y%m%d'), monitor_date=future,
                    assistant_plan_id='OFFLINE_FIXTURE', boards=[{'name': 'pcb', 'note': 'fixture'}],
                    stocks=[dict(code='300652', name='隔离样例', buy_zone=[25.01, 25.45],
                                 observe_only=False, valid_until=future + '150000',
                                 invalidations=['仅隔离验收'], evidence_refs=['fixture'])])
        if not file.is_file():
            trade_plan_v1.save_plan(base, file)
            assistant = pentacles_demo.snapshot_path().with_name('assistant_plan.json')
            assistant.parent.mkdir(parents=True, exist_ok=True)
            assistant.write_text(json.dumps({'rows': [], 'boards': base['boards']}), encoding='utf8')
        restarted = (sandbox / 'completed.json').is_file()
        stocks_before = trade_plan_v1.load_plan()['stocks']
        assistant_file = pentacles_demo.snapshot_path().with_name('assistant_plan.json')
        stock_identity_before = boards.stock_alert_identity(json.loads(assistant_file.read_bytes()))
        home = client.get('/')
        check('legacy_home', home.status_code == 200 and b'app.js' in home.data)
        for resource in ('app.js', 'pentacles.js', 'style.css', 'marked.min.js', 'icon.png'):
            check('resource_' + resource, client.get('/' + resource).status_code == 200)
        app_js = client.get('/app.js').data
        check('board_confirmation_ui', b'/api/assistant/boards/confirm' in app_js)

        message = '我想把同花顺概念板块 影视院线（881274） 短剧游戏（886060） 光伏概念（885531）也加入计划监控板块'
        response = client.post('/api/chat', json={'message': message, 'mode': 'chat'})
        value = response.get_json()
        check('chat_board_only_no_llm', response.status_code == 200 and value.get('sector_only'))
        check('no_0930_error', '09:30' not in value.get('reply', ''))
        draft = value['draft']
        result = client.post('/api/assistant/boards/confirm', json={'id': draft['id'], 'confirmed': True}).get_json()
        check('confirm_board', result.get('ok'))
        formal = trade_plan_v1.load_plan()
        check('stocks_unchanged', formal['stocks'] == stocks_before)
        current_assistant = json.loads(assistant_file.read_bytes())
        check('board_change_preserves_stock_touch_identity',
              boards.stock_alert_identity(current_assistant) == stock_identity_before)
        changed_stock = dict(current_assistant, monitor_date='20990101')
        check('changed_stock_plan_cannot_reuse_touch_identity',
              boards.stock_alert_identity(changed_stock) == boards.digest(changed_stock))
        check('boards_persisted', {b['name'] for b in formal['boards']} >= {'pcb', '影视院线', '短剧游戏', '光伏概念'})
        repeat = client.post('/api/assistant/boards/confirm', json={'id': draft['id'], 'confirmed': True}).get_json()
        check('same_draft_idempotent', repeat.get('ok') and repeat.get('duplicate'))
        state = client.get('/api/assistant/boards/state').get_json()
        check('confirmed_readback', state['draft']['confirmed'])
        check('stock_draft_namespace_separate', workflow.latest('plan') is None)
        check('pool_identity_consumed', ths.resolve_for_pool('短剧游戏', risk.directory() / 'board_identities')['member_count'] == 87)
        industry = ths.resolve_for_pool('影视院线', risk.directory() / 'board_identities')
        check('industry_real_members_prepared', industry['member_count'] == 20)
        check('industry_provenance_present', industry['provenance']['source_url'].endswith('/881274/'))
        check('industry_consumer_reads_members', industry['status'] == 'MONITOR_ENABLED')
        check('negative_not_edit', not boards.is_sector_only('不要加入光伏概念板块'))
        check('mixed_not_edit', not boards.is_sector_only('把光伏概念加入板块，同时把雷迪克移出计划'))

        # A real board-only commit must preserve already-sent stock observations.
        identity_root = sandbox / ('identity_restart' if restarted else 'identity_first')
        identity_store = identity_root / 'board_identities'
        identity_plan = identity_root / 'formal.json'
        identity_assistant = identity_root / 'assistant.json'
        fixture_now = now.replace(hour=10, minute=0, second=0, microsecond=0, tzinfo=None)
        fixture = dict(assistant_plan_id='IDENTITY_FIXTURE', monitor_date=int(now.strftime('%Y%m%d')),
                       rows=[dict(code='300652', reference_price=26,
                                  entry_rule={'levels': [25.0]})], boards=[{'name': 'pcb'}])
        trade_plan_v1.save_plan(dict(base, boards=fixture['boards']), identity_plan)
        identity_assistant.write_text(json.dumps(fixture), encoding='utf8')
        quotes = {'300652': dict(price=25, low=25, prev=26, time=fixture_now.strftime('%Y%m%d%H%M%S'))}
        from pentacles_monitor import observation_reviews
        with patch.object(risk, 'directory', return_value=identity_root):
            notice = observation_reviews(fixture, quotes, fixture_now, {})
            check('stock_observation_before_board_change', len(notice) == 1)
            alert_ids = {'tarot-v2:' + v: True for v in notice[0]['covered_event_ids']}
            change = boards.preview('把光伏概念加入监控板块', fixture['boards'], identity_root / 'drafts',
                                    identity_store, formal=trade_plan_v1.load_plan(identity_plan),
                                    assistant_plan_path=identity_assistant)
            boards.confirm(change['draft']['id'], True, root=identity_root / 'drafts', store_dir=identity_store,
                           formal_plan_path=identity_plan, assistant_plan_path=identity_assistant, now=now)
            amended = json.loads(identity_assistant.read_bytes())
            check('board_only_no_repeat_stock_observation',
                  observation_reviews(amended, quotes, fixture_now, alert_ids) == [])

        # An expired plan stores future preferences and never extends stock validity.
        expired = dict(formal, monitor_date='20200102', stocks=[dict(formal['stocks'][0], valid_until='20200102150000')])
        trade_plan_v1.save_plan(expired, file)
        old_bytes = file.read_bytes()
        other = client.post('/api/chat', json={'message': '把光伏概念移出监控板块'}).get_json()['draft']
        saved = client.post('/api/assistant/boards/confirm', json={'id': other['id'], 'confirmed': True}).get_json()
        check('expired_preferences_only', saved.get('ok') and not saved['effective'] and file.read_bytes() == old_bytes)
        pref = boards.pending_preferences(risk.directory() / 'board_identities')
        check('next_plan_preferences_available', {b['name'] for b in pref['boards']} == {'pcb', '影视院线', '短剧游戏'})
        empty_snapshot = dict(schema='PAGE_OF_PENTACLES_DEMO_V1', data_date=int(base['data_date']),
                              monitor_date=int(future), plan_ready=True, rows=[], manual_review_rows=[])
        with patch.object(workflow, 'snapshot', return_value=empty_snapshot), \
             patch.object(workflow, 'latest', return_value={'boards': [{'name': 'pcb'}], 'created_at': '2000-01-01'}), \
             patch.object(workflow, 'store', side_effect=lambda kind, value: {'id': 'OFFLINE_ONLY', **value}):
            zero = workflow.plan_chat(cfg, '请根据当前名单制定次日观察计划')
        check('zero_candidates_inherit_board_preferences',
              {b['name'] for b in zero['draft']['boards']} == {'pcb', '影视院线', '短剧游戏'})
        trade_plan_v1.save_plan(formal, file)
        (sandbox / 'completed.json').write_text(json.dumps({'formal': formal}), encoding='utf8')
        check('independent_restart_readback', not restarted or len(formal['boards']) == 4)

        from instance_guard_v1 import acquire_instance_lock, release_instance_lock
        check('instance_lock', acquire_instance_lock())
        release_instance_lock()
        from notifier import stable_digest_key
        check('stable_dedup', isinstance(stable_digest_key('fixture'), str))

        # Capture import evidence, distinguishing bundled application code from shared dependencies.
        import sys
        modules = {n: str(getattr(sys.modules.get(n), '__file__', '')) for n in (
            'server', 'notifier', 'strong_sector_watch_v1', 'plan_board_change_v1',
            'ths_board_identity_v1', 'instance_guard_v1', 'instance_state_store_v1')}
        record = dict(ok=all(results.values()), frozen=getattr(sys, 'frozen', False),
                      checks=results, restarted=restarted, modules=modules,
                      network_calls=0, paid_calls=0, real_push=0, monitor_started=False,
                      formal_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf8')
        print('MONITOR_ACCEPTANCE_PASS', len(results), flush=True)


if __name__ == '__main__':
    import sys
    run(sys.argv[1])
