# -*- coding: utf-8 -*-
"""无副作用 HTTP 验收入口：只跑 Flask，不调用 server.start()，不启动任何监控线程。

用法：Stargazer.exe --acceptance-serve <sandbox> <port> <ready_json>
隔离与边界拦截复用 acceptance_ui_integration_v1；缺 I34 资产会直接抛错，不返回假空列表。
"""
import json
import os
import sys
from pathlib import Path


def run(argv):
    sandbox = Path(argv[0]).resolve()
    port = int(argv[1])
    out = Path(argv[2])
    os.environ['STARGAZER_ACCEPTANCE_MODE'] = '1'
    os.environ['STARGAZER_GLOBAL_OVERVIEW_SNAPSHOT'] = str(
        sandbox / 'data' / 'isolated' / 'global_overview_snapshot.json')
    import acceptance_ui_integration_v1 as acc
    acc._patch_data_file(sandbox)
    acc._install_intercepts()
    import server
    import assistant_workflow as workflow
    workflow.root = lambda: sandbox / 'data' / 'assistant'
    _web_root = os.environ.get('STARGAZER_UI2_WEB_ROOT')
    if _web_root:
        # 验收专用：指向按 sync_map 构造的候选 web 布局，不触碰生产 web/
        server.WEB = Path(_web_root).resolve()
    server.register_extension_routes()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'ok': True, 'mode': 'ACCEPTANCE_SERVE', 'port': port,
        'sandbox': str(sandbox), 'monitor_started': False, 'paid_calls': 0,
        'frozen': bool(getattr(sys, 'frozen', False)),
        'web': str(server.WEB),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    server.app.run(host='127.0.0.1', port=port, threaded=True, use_reloader=False)
    return {'ok': True}
