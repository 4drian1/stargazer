"""Stargazer 桌面版入口。
启动 Flask 后端 + 定时巡检线程，然后开桌面窗口（pywebview）。
没有 pywebview（或未装桌面环境）时退回系统浏览器。
"""
import sys
import time
import urllib.request

if len(sys.argv) == 3 and sys.argv[1] == '--acceptance-monitor':
    from monitor_release_acceptance_v1 import run
    run(sys.argv[2])
    raise SystemExit(0)

if getattr(sys, 'frozen', False):
    import os
    from pathlib import Path
    temporary = Path(sys.executable).parent/'data/postclose_risk/tmp'
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ['TEMP'] = os.environ['TMP'] = str(temporary)
    os.environ['MPLCONFIGDIR'] = str(temporary/'matplotlib')
    os.environ['PYTHONPYCACHEPREFIX'] = str(temporary/'bytecode')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

if len(sys.argv) >= 4 and sys.argv[1] in ('--acceptance', '--acceptance-restart'):
    from acceptance_ui_integration_v1 import run as _ui2_acceptance_run
    raise SystemExit(0 if _ui2_acceptance_run(sys.argv[1], sys.argv[2:]).get('ok') else 1)

if len(sys.argv) >= 5 and sys.argv[1] == '--acceptance-serve':
    from acceptance_ui2_serve_v1 import run as _ui2_serve_run
    raise SystemExit(0 if _ui2_serve_run(sys.argv[2:]).get('ok') else 1)

import server


def wait_ready(port, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=2)
            return True
        except Exception:
            time.sleep(0.3)
    return False


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--self-check':
        import json
        from pathlib import Path
        import paths
        import shadow_ui
        import trade_plan_v1
        import shadow_journal
        import oracles_v1
        import decision_layer
        import index_minutes_v1
        import assistant_workflow
        import plan_evidence
        import postclose_risk_runtime
        client = server.app.test_client()
        result = dict(
            frozen=paths.FROZEN, app_dir=paths.APP_DIR,
            page_ok=client.get('/').status_code == 200,
            catalog=client.get('/api/shadow_evidence').get_json(),
            missing=client.get('/api/shadow_evidence?code=000000&date=20260908').get_json(),
            index_status=client.get('/api/index_freshness').get_json(),
            index_source_updater_available=callable(index_minutes_v1._load_updater().parse_lc5),
            packs_path=str(shadow_ui.PACKS_FILE),
            plan_path=str(trade_plan_v1.TRADE_PLAN_FILE),
            journal_path=str(shadow_journal.JOURNAL),
            paid_calls=0, monitor_started=False)
        current_snapshot = assistant_workflow.snapshot()
        result['current_plan_evidence'] = plan_evidence.status(current_snapshot)
        result['postclose_rebuild_required'] = postclose_risk_runtime.needs_rebuild(current_snapshot)
        with Path(sys.argv[2]).open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
        return
    port = server.start()
    url = f"http://127.0.0.1:{port}/"
    if not wait_ready(port):
        print("后端启动超时，请查看日志")
        sys.exit(1)
    print(f"Stargazer 后端已就绪: {url}")
    try:
        import webview
        webview.create_window("Stargazer", url, width=1180, height=780,
                              min_size=(900, 600),
                              text_select=True)  # 正文可拖选+Ctrl+C；按钮/拖拽区不受影响
        webview.start()
    except ImportError:
        import webbrowser
        print("未安装 pywebview，改用系统浏览器打开（pip install pywebview 可获得独立窗口）")
        webbrowser.open(url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
